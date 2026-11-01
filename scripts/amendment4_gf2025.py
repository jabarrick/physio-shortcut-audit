"""Amendment 4: replication of the PhysioNet eye-movement analysis in data with a lateral cue, a true horizontal
EOG channel and an eye tracker (Guttmann-Flury, Sheng & Zhu 2025, Sci Data 12:587; motor-imagery part, Zenodo
record 18970793, CC0).  Registered analysis; see Paper3_prereg_amendment4 (OSF).  Nothing in configs/ or p3audit/
is changed; the training settings of configs/default.yaml are read only.

Steps (run from the repository root):
  1. python scripts/amendment4_gf2025.py --prepare --zips <folder with S01.zip ... S31.zip>
       CPU.  Reads every session directly from the archives (identified by docs/amendment4_zip_sha256.txt; checked
       here), extracts BDF / sync / Tobii / annotation files to a temporary folder outside the repository, and
       writes one cache file per subject to results/gf2025_cache/ plus results/gf2025_cache/sessions.json.
  2. python scripts/amendment4_gf2025.py [--models ...] [--folds ...] [--seeds ...]
       GPU.  Subject-wise 5-fold cross-validation, six arms, seeds 0-2.  Output results/gf2025/.  Resumes.
  3. python scripts/amendment4_gf2025.py --analyse
       CPU.  Registered tests A4-R1 ... A4-R4 and the descriptive outputs.  Writes results/analysis/amendment4.json.
  Dry run (before registration; output excluded from the repository and from every analysis):
       python scripts/amendment4_gf2025.py --models eegnet --folds 0 --seeds 0 --max-epochs 1 --out results/gf2025_dryrun

Signals (section 4 of the amendment)
  Cue onset    first camera frame showing the rectangle (sync file, column Cues), about 2.2 s after the trigger
               and the MOABB "imagery" annotation, which mark the fixation cross (see cue_onsets).
  Model input  60 EEG channels (CB1, CB2 dropped), 200 Hz, 0.5-75 Hz band-pass, 50 Hz notch, average reference;
               epochs -1 ... 5 s around the cue (0-4 s model window with 1 s padding); no ICA.
  HOR          F8 - F7 (right minus left; positive = rightward) of the model input: the horizontal ocular regressor
               of the primary control and the source of d_HOR; a copy filtered 0.1-20 Hz feeds the step detector.
               The dataset has no horizontal EOG: the channel labelled HEO is the bipolar vertical EOG of the left
               eye (Guttmann-Flury et al. 2025, Methods; confirmed by its coupling with vertical gaze).
  VEO          that channel with the model-input filters, not re-referenced (descriptive control only).
  Gaze         Tobii validity code 0 or 1 = valid eye; horizontal gaze point in mm averaged over valid eyes,
               converted to degrees with the eye-screen distance, relative to the session median during fixation
               (2 s before each cue); gaps <= 75 ms interpolated; mapped to the EEG time base with the sync file.

Arms (section 5): main, hor_ols (primary control), veo_ols, plc_shuf, frontal_loc, central_loc.
"""
from __future__ import annotations

import argparse, glob, hashlib, json, re, sys, tempfile, time, warnings, zipfile
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", message="Mean of empty slice")          # gaze-invalid trials are NaN by design
warnings.filterwarnings("ignore", message="All-NaN slice encountered")   # sessions without valid fixation gaze

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CACHE = Path("results/gf2025_cache")
OUT = Path("results/gf2025")
SFREQ, PAD_S, WIN_S = 200.0, 1.0, 4.0
NOTCH = 50.0
EEG62 = ["Fp1", "Fpz", "Fp2", "AF3", "AF4", "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3",
         "FC1", "FCz", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8", "TP7", "CP5",
         "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8", "PO7",
         "PO5", "PO3", "POz", "PO4", "PO6", "PO8", "O1", "Oz", "O2", "CB1", "CB2"]
EEG = [c for c in EEG62 if c not in ("CB1", "CB2")]                       # 60 model channels
FRONTAL = ["Fp1", "Fp2", "AF3", "AF4", "F7", "F8"]
CENTRAL = ["FC3", "FC4", "C3", "C4", "CP3", "CP4"]
ARMS = ["main", "hor_ols", "veo_ols", "plc_shuf", "frontal_loc", "central_loc"]
MODELS = ["csoanet", "eegnet", "shallow", "cbramod"]                      # csoanet = MS-CNN in the manuscript
FOLD_SEED, SHUF_SEED, BOOT_SEED = 20261031, 20261103, 20261102
N_BOOT, ALPHA = 2000, 0.0125                                              # four tests: two-sided 98.75 % intervals
GAZE_FLOOR_DEG, HOR_FLOOR_UV = 0.5, 15.0
GAP_MS, NEAREST_MS = 75.0, 10.0
TRIAL_VALID_MIN, SESSION_VALID_MIN = 0.75, 0.50
K_MIN, R1_MIN_SUBJECTS = 10, 10
# windows in seconds relative to the cue
W_BASE, W_DEFL, W_DET, W_VALID = (-0.2, 0.0), (0.3, 1.2), (0.0, 1.0), (-0.2, 1.2)
STEP_S = 0.1                                                              # +-100 ms step statistic

TOBII_COLS = {"t": "RecordingTimestamp", "xl": "GazePointLeftX (ADCSmm)", "xr": "GazePointRightX (ADCSmm)",
              "vl": "ValidityLeft", "vr": "ValidityRight", "dl": "DistanceLeft", "dr": "DistanceRight"}


# --------------------------------------------------------------------------------------------- small helpers
def idx(t0, t1, pad=int(PAD_S * SFREQ)):
    """Sample slice of an epoch (which starts 1 s before the cue) for [t0, t1) s."""
    return slice(pad + int(round(t0 * SFREQ)), pad + int(round(t1 * SFREQ)))


def sha256(p: Path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def read_table(path: Path, usecols):
    """Read a CSV exported by Tobii Studio / the dataset authors: delimiter and decimal mark are detected."""
    import pandas as pd
    with open(path, encoding="utf-8-sig", errors="replace", newline="") as f:
        head = f.readline()
    sep = max([",", ";", "\t"], key=head.count)
    df = pd.read_csv(path, sep=sep, usecols=lambda c: c in usecols, dtype=str, encoding="utf-8-sig", low_memory=False)
    out = {}
    for c in usecols:
        if c not in df.columns:
            out[c] = None
            continue
        s = df[c].str.strip().replace({"": None})
        out[c] = pd.to_numeric(s.str.replace(",", ".", regex=False), errors="coerce").to_numpy(float)
    return out


def step_stat(x, n):
    """s(t) = mean(x[t:t+n]) - mean(x[t-n:t]) along the last axis (NaN where undefined)."""
    c = np.nancumsum(np.nan_to_num(x), axis=-1); c = np.concatenate([np.zeros(x.shape[:-1] + (1,)), c], -1)
    s = np.full(x.shape, np.nan)
    T = x.shape[-1]
    t = np.arange(n, T - n + 1)
    s[..., t] = (c[..., t + n] - c[..., t]) / n - (c[..., t] - c[..., t - n]) / n
    return s


def mad(v):
    v = v[np.isfinite(v)]
    return float(np.median(np.abs(v - np.median(v)))) if v.size else np.nan


def classify(sig, cue_sign, thr, valid=None):
    """Registered step classification in 0-1 s: 'congruent' / 'incongruent' / 'none' (or 'invalid')."""
    n = int(round(STEP_S * SFREQ)); s = step_stat(sig, n)[:, idx(*W_DET)]
    out = []
    for k in range(len(sig)):
        if valid is not None and not valid[k]:
            out.append("invalid"); continue
        sk = s[k]; a = np.abs(sk)
        if not np.isfinite(a).any() or np.nanmax(a) <= thr[k]:
            out.append("none"); continue
        j = int(np.nanargmax(a))
        out.append("congruent" if np.sign(sk[j]) == cue_sign[k] else "incongruent")
    return np.array(out)


def deflection(sig, cue_sign=None):
    d = np.nanmean(sig[:, idx(*W_DEFL)], 1) - np.nanmean(sig[:, idx(*W_BASE)], 1)
    return d if cue_sign is None else d * cue_sign


# --------------------------------------------------------------------------------------------- 1. prepare
def cue_onsets(raw, ann_path: Path, sync_path: Path):
    """Cue (rectangle) onsets in s on the EEG time base and labels (0 = left, 1 = right).

    The synchronisation file records, at every high-speed-camera frame, the screen shown (column Cues: Welcome,
    Fixation, Left, Right, Random, Goodbye).  The trigger channel and the annotation file mark the onset of the
    fixation cross, about 2.2 s before the rectangle; the registered cue onset is therefore the first frame showing
    Left or Right after each fixation onset.  Checks (reported): every fixation onset has a trigger within 10 ms
    whose code matches the side (1 = left; right is coded 2 or 131071), and the annotation codes match."""
    import mne
    import pandas as pd
    S = pd.read_csv(sync_path, usecols=["Time", "Cues"], dtype={"Cues": str})
    t = S["Time"].to_numpy(float); lab = S["Cues"].where(S["Cues"].str.strip() != "").ffill().fillna("").to_numpy()
    ch = np.where(lab[1:] != lab[:-1])[0] + 1
    fix, on, side = [], [], []
    for k, i in enumerate(ch):
        if lab[i] == "Fixation":
            nxt = next((j for j in ch[k + 1:k + 3] if lab[j] in ("Left", "Right")), None)
            if nxt is not None and t[nxt] - t[i] < 3.0:
                fix.append(t[i]); on.append(t[nxt]); side.append(1 if lab[nxt] == "Right" else 0)
    fix, on, side = np.array(fix), np.array(on), np.array(side)
    chk = {"n_cues": int(len(on)), "fixation_to_rectangle_s": [float(np.min(on - fix)), float(np.median(on - fix)), float(np.max(on - fix))] if len(on) else None}
    try:
        ev = mne.find_events(raw, stim_channel="Trig", shortest_event=1, verbose="ERROR")
        tt = (ev[:, 0] - raw.first_samp) / raw.info["sfreq"]; code = np.where(ev[:, 2] == 1, 0, 1)
        d = np.array([np.min(np.abs(tt - x)) for x in fix]); jj = np.array([np.argmin(np.abs(tt - x)) for x in fix])
        chk.update(n_trigger=int(len(tt)), trigger_fixation_max_abs_ms=float(d.max() * 1000),
                   trigger_codes_match=bool(np.all(code[jj] == side)), trigger_codes=sorted(int(c) for c in set(ev[:, 2])))
    except Exception as e:                                                # noqa: BLE001  (reported, not fatal)
        chk["trigger_error"] = repr(e)
    try:
        ann = json.loads(ann_path.read_text(encoding="utf-8"))
        im = [a for a in ann if a.get("phase") == "imagery"]
        ao = np.array([float(a["onset"]) for a in im]); ac = np.array([int(a["event_code"]) - 1 for a in im])
        jj = np.array([np.argmin(np.abs(ao - x)) for x in fix])
        chk.update(annotation_imagery_minus_fixation_max_abs_ms=float(np.max(np.abs(ao[jj] - fix)) * 1000),
                   annotation_codes_match=bool(np.all(ac[jj] == side)))
    except Exception as e:                                                # noqa: BLE001
        chk["annotation_error"] = repr(e)
    chk["fixation_onsets"] = [float(x) for x in fix]
    return on, side, chk


def tobii_to_grid(tobii: Path, sync: Path, n_grid: int, cue_on, fix_on):
    """Horizontal gaze (deg) and validity on the 200 Hz EEG grid of the session.  Returns (gaze, valid, info) or
    (None, None, info) if the session cannot be aligned.

    Alignment: the synchronisation file gives Tobii RecordingTimestamps of fixation onsets, but they can be
    assigned to the wrong trials, also in blocks within a session (Tobii missed markers while the rows were filled
    in order).  Each Tobii time stamp is therefore matched to a fixation onset by time: (1) an initial map with
    slope 1000 ms/s and the intercept of the largest cluster of offsets R - 1000 E (time stamps paired in order
    at the lag whose inter-event intervals match best; cluster = offsets within 30 ms of each other); (2) every
    time stamp is assigned to the fixation onset whose predicted time is nearest, and kept if the residual is
    <= 20 ms (about one 60 Hz frame of marker jitter; the gaze measures use windows of 100 ms and more); (3) the
    least-squares line through the kept pairs replaces the map and (2) is repeated once.  At least 10 pairs are
    required.  Gaze is used only between the first and last matched fixation onsets, extended by 10 s."""
    info = {}
    S = read_table(sync, ["Time", "RecordingTimestamp"])
    if S["RecordingTimestamp"] is None:
        info["unavailable"] = "sync file has no RecordingTimestamp column"; return None, None, info
    R = S["RecordingTimestamp"][np.isfinite(S["RecordingTimestamp"])]
    E = np.asarray(fix_on, float)
    if len(R) < 10 or len(E) < 10:
        info["unavailable"] = f"only {len(R)} Tobii event time stamps"; return None, None, info
    best = None
    for L in range(-len(R) + 10, len(E) - 9):
        jr = np.arange(len(R)); je = jr + L; ok = (je >= 0) & (je < len(E))
        if ok.sum() < 10:
            continue
        m = float(np.median(np.abs(np.diff(R[jr[ok]]) / 1000.0 - np.diff(E[je[ok]]))))
        if best is None or m < best[0]:
            best = (m, L, jr[ok], je[ok])
    o = R[best[2]] - 1000.0 * E[best[3]]
    cnt = np.array([(np.abs(o - x) <= 30).sum() for x in o]); core = o[np.abs(o - o[np.argmax(cnt)]) <= 30]
    a, b = float(np.median(core)), 1000.0

    def assign(a, b):
        k = np.array([int(np.argmin(np.abs(r - (a + b * E)))) for r in R]); res = R - (a + b * E[k]); keep = np.abs(res) <= 20.0
        # one time stamp per fixation onset: keep the closest
        for kk in np.unique(k[keep]):
            ii = np.where(keep & (k == kk))[0]
            if len(ii) > 1:
                keep[ii[np.argsort(np.abs(res[ii]))[1:]]] = False
        return k, keep

    k, keep = assign(a, b)
    if keep.sum() < 10:
        info["unavailable"] = f"only {int(keep.sum())} Tobii time stamps match a fixation onset"; return None, None, info
    b, a = np.polyfit(E[k[keep]], R[keep], 1); k, keep = assign(a, b)
    if keep.sum() < 10:
        info["unavailable"] = f"only {int(keep.sum())} Tobii time stamps match a fixation onset"; return None, None, info
    b, a = np.polyfit(E[k[keep]], R[keep], 1); res = R[keep] - (a + b * E[k[keep]])
    span = (E[k[keep]].min() - 10.0, E[k[keep]].max() + 10.0)
    info.update(sync_initial_lag_trials=int(best[1]), sync_pairs=int(keep.sum()), sync_unmatched=int((~keep).sum()),
                sync_slope=float(b), sync_intercept=float(a), sync_max_residual_ms=float(np.max(np.abs(res))), sync_span_s=[float(x) for x in span])
    if np.max(np.abs(res)) > 20.0:
        info["unavailable"] = "sync residual > 20 ms"; return None, None, info
    T = read_table(tobii, list(TOBII_COLS.values()))
    if any(T[c] is None for c in TOBII_COLS.values()):
        info["unavailable"] = "Tobii columns missing"; return None, None, info
    t = T[TOBII_COLS["t"]]
    keep = np.isfinite(t) & (np.isfinite(T[TOBII_COLS["vl"]]) | np.isfinite(T[TOBII_COLS["vr"]]))   # gaze-sample rows
    t = t[keep]; order = np.argsort(t, kind="stable"); t = t[order]
    first = np.r_[True, np.diff(t) > 0]                                   # drop duplicate time stamps (event rows)
    col = {k: T[v][keep][order][first] for k, v in TOBII_COLS.items()}; t = t[first]
    vl = (col["vl"] <= 1) & np.isfinite(col["xl"]) & (col["dl"] > 0)
    vr = (col["vr"] <= 1) & np.isfinite(col["xr"]) & (col["dr"] > 0)
    nv = vl.astype(float) + vr
    x = np.where(nv > 0, (np.where(vl, col["xl"], 0) + np.where(vr, col["xr"], 0)) / np.maximum(nv, 1), np.nan)
    D = np.where(nv > 0, (np.where(vl, col["dl"], 0) + np.where(vr, col["dr"], 0)) / np.maximum(nv, 1), np.nan)
    v = nv > 0
    # interpolate gaps of at most 75 ms between valid samples
    vi = np.where(v)[0]
    if vi.size < 2:
        info["unavailable"] = "no valid gaze"; return None, None, info
    gaps = np.where(np.diff(vi) > 1)[0]
    for g in gaps:
        i0, i1 = vi[g], vi[g + 1]
        if t[i1] - t[i0] <= GAP_MS:
            w = (t[i0 + 1:i1] - t[i0]) / (t[i1] - t[i0])
            x[i0 + 1:i1] = x[i0] + w * (x[i1] - x[i0]); D[i0 + 1:i1] = D[i0] + w * (D[i1] - D[i0]); v[i0 + 1:i1] = True
    # map the EEG grid to Tobii time and take the nearest Tobii sample within 10 ms
    tg = a + b * (np.arange(n_grid) / SFREQ)
    j = np.clip(np.searchsorted(t, tg), 1, len(t) - 1); j = np.where(np.abs(t[j - 1] - tg) < np.abs(t[j] - tg), j - 1, j)
    near = (np.abs(t[j] - tg) <= NEAREST_MS) & (np.arange(n_grid) / SFREQ >= span[0]) & (np.arange(n_grid) / SFREQ <= span[1])
    valid = near & v[j]
    xg = np.where(valid, x[j], np.nan); Dg = np.where(valid, D[j], np.nan)
    # reference: median gaze during fixation (2 s before each cue)
    fix = np.zeros(n_grid, bool)
    for c in cue_on:
        fix[max(0, int(round((c - 2.0) * SFREQ))):max(0, int(round(c * SFREQ)))] = True
    xref = np.nanmedian(xg[fix & valid]) if (fix & valid).any() else np.nanmedian(xg)
    gaze = np.degrees(np.arctan((xg - xref) / Dg))
    info.update(tobii_rows=int(len(t)), tobii_median_dt_ms=float(np.median(np.diff(t))), valid_fraction=float(valid.mean()))
    return gaze.astype(np.float32), valid, info


def prepare(a):
    import mne
    zips = Path(a.zips)
    sha_ref = {}
    ref = Path("docs/amendment4_zip_sha256.txt")
    if ref.exists():
        for line in ref.read_text().splitlines():
            if line.strip():
                n, h = line.split("\t"); sha_ref[n.strip()] = h.strip()
    CACHE.mkdir(parents=True, exist_ok=True)
    log = json.loads((CACHE / "sessions.json").read_text()) if (CACHE / "sessions.json").exists() else {}
    for z in sorted(zips.glob("S*.zip")):
        subj = int(re.findall(r"S(\d+)", z.name)[0])
        if a.subjects and subj not in a.subjects:
            continue
        q = CACHE / f"sub-{subj:02d}.npz"
        if q.exists():
            continue
        if sha_ref:
            h = sha256(z)
            if sha_ref.get(z.name) != h:
                sys.exit(f"{z.name}: SHA-256 {h} does not match docs/amendment4_zip_sha256.txt")
        parts = {k: [] for k in ("X", "HOR", "HORD", "VEO", "GAZE", "GVALID", "y", "session")}
        with zipfile.ZipFile(z) as zf, tempfile.TemporaryDirectory() as td:
            sess_dirs = sorted({m.split("/")[0] for m in zf.namelist() if m.lower().endswith(".bdf")})
            for si, sd in enumerate(sess_dirs):
                names = [m for m in zf.namelist() if m.startswith(sd + "/")]
                pick = lambda suf: next((m for m in names if m.endswith(suf)), None)
                bdf = next(m for m in names if m.lower().endswith(".bdf"))
                files = {k: pick(s) for k, s in (("sync", "_sync.csv"), ("tobii", "_tobii.csv"), ("ann", "_annotations.json"))}
                for m in [bdf] + [v for v in files.values() if v]:
                    zf.extract(m, td)
                key = f"S{subj:02d}/{sd}"; L = log[key] = {"bdf": bdf}
                raw = mne.io.read_raw_bdf(Path(td) / bdf, preload=True, stim_channel="Trig", verbose="ERROR")
                if files["sync"] is None:
                    L["excluded"] = "no synchronisation file"; continue
                on, lab, chk = cue_onsets(raw, Path(td) / files["ann"] if files["ann"] else Path(td) / "missing.json", Path(td) / files["sync"])
                L["cue_check"] = {k: v for k, v in chk.items() if k != "fixation_onsets"}
                if len(on) < 30:
                    L["excluded"] = f"{len(on)} imagery events"; continue
                raw.pick(EEG62 + ["HEO"]); raw.set_channel_types({"HEO": "eog"}, verbose="ERROR")
                det = raw.copy().pick(["HEO", "F7", "F8"]).filter(0.1, 20.0, picks="all", verbose="ERROR").resample(SFREQ, verbose="ERROR")
                raw.resample(SFREQ, verbose="ERROR")
                raw.filter(0.5, 75.0, picks="all", verbose="ERROR"); raw.notch_filter(NOTCH, picks="all", verbose="ERROR")
                Dm = raw.get_data() * 1e6; Dd = det.get_data() * 1e6
                ie = [EEG62.index(c) for c in EEG]
                eeg = Dm[ie] - Dm[ie].mean(0, keepdims=True); veo = Dm[len(EEG62)]
                hor = eeg[EEG.index("F8")] - eeg[EEG.index("F7")]
                veo_d = Dd[0]; hor_d = Dd[2] - Dd[1]
                n_grid = Dm.shape[1]
                gaze, gvalid, ginfo = (None, None, {"unavailable": "no Tobii file"})
                if files["tobii"] and files["sync"]:
                    gaze, gvalid, ginfo = tobii_to_grid(Path(td) / files["tobii"], Path(td) / files["sync"], n_grid, on, chk["fixation_onsets"])
                # session gaze availability: valid fraction from -0.2 to 1.2 s around the cues
                if gaze is not None:
                    m = np.zeros(n_grid, bool)
                    for c in on:
                        m[int(round((c + W_VALID[0]) * SFREQ)):int(round((c + W_VALID[1]) * SFREQ))] = True
                    ginfo["cue_window_valid_fraction"] = float(gvalid[m].mean())
                    if gvalid[m].mean() < SESSION_VALID_MIN:
                        ginfo["unavailable"] = "valid gaze < 50 % around the cues"; gaze = None
                L["gaze"] = ginfo
                # descriptive, label-free: coupling of sample-to-sample changes with those of horizontal gaze
                if gaze is not None:
                    dg = np.diff(gaze.astype(float)); okk = np.isfinite(dg)
                    L["coupling_with_gaze_x"] = {"F8-F7": float(np.corrcoef(np.diff(hor_d)[okk], dg[okk])[0, 1]),
                                                 "VEO": float(np.corrcoef(np.diff(veo_d)[okk], dg[okk])[0, 1])}
                n0, n1 = int(PAD_S * SFREQ), int((WIN_S + PAD_S) * SFREQ)
                kept = 0
                for c, l in zip(on, lab):
                    i = int(round(c * SFREQ))
                    if i - n0 < 0 or i + n1 > n_grid:
                        continue
                    sl = slice(i - n0, i + n1)
                    parts["X"].append(eeg[:, sl]); parts["HOR"].append(hor[sl]); parts["HORD"].append(hor_d[sl]); parts["VEO"].append(veo[sl])
                    parts["GAZE"].append(gaze[sl] if gaze is not None else np.full(n0 + n1, np.nan, np.float32))
                    parts["GVALID"].append(gvalid[sl] if gaze is not None else np.zeros(n0 + n1, bool))
                    parts["y"].append(int(l)); parts["session"].append(si); kept += 1
                L["trials"] = kept; L["gaze_available"] = gaze is not None
                print(f"{key}: {kept} trials, gaze {'yes' if gaze is not None else 'NO (' + ginfo.get('unavailable', '') + ')'}, "
                      "", flush=True)
        if not parts["y"]:
            continue
        tmp = q.with_suffix(".tmp.npz")
        np.savez(tmp, X=np.stack(parts["X"]).astype(np.float32), HOR=np.stack(parts["HOR"]).astype(np.float32),
                 HORD=np.stack(parts["HORD"]).astype(np.float32), VEO=np.stack(parts["VEO"]).astype(np.float32), GAZE=np.stack(parts["GAZE"]).astype(np.float32),
                 GVALID=np.stack(parts["GVALID"]), y=np.array(parts["y"]), session=np.array(parts["session"]),
                 pad=np.array(int(PAD_S * SFREQ)), core_len=np.array(int(WIN_S * SFREQ)))
        tmp.replace(q)
        (CACHE / "sessions.json").write_text(json.dumps(log, indent=1))


# --------------------------------------------------------------------------------------------- 2. train
def subjects_in_cache():
    return sorted(int(re.findall(r"sub-(\d+)", p.name)[0]) for p in CACHE.glob("sub-*.npz") if ".tmp" not in p.name)


def folds():
    """Registered folds: default_rng(20261031) permutation of subject numbers split into 5 folds, then one
    early-stopping subject per fold drawn from that fold's training subjects with the same generator."""
    subs = np.array(subjects_in_cache()); rng = np.random.default_rng(FOLD_SEED)
    perm = rng.permutation(subs); F = [sorted(int(s) for s in f) for f in np.array_split(perm, 5)]
    out = []
    for k, te in enumerate(F):
        tr = [int(s) for s in subs if s not in te]
        es = int(rng.choice(tr)); out.append({"fold": k, "test": te, "es": es, "train": [s for s in tr if s != es]})
    return out


def load(subjects):
    P = [np.load(CACHE / f"sub-{s:02d}.npz") for s in subjects]
    d = {k: np.concatenate([p[k] for p in P]) for k in ("X", "HOR", "VEO", "y", "session")}
    d["subj"] = np.concatenate([np.full(len(p["y"]), s) for s, p in zip(subjects, P)])
    d["trial"] = np.concatenate([np.arange(len(p["y"])) for p in P])
    d["pad"], d["core_len"] = int(P[0]["pad"]), int(P[0]["core_len"])
    return d


def fit_ols(X, R):   # X: n x c x t ; R: n x k x t  ->  B: c x k
    Xc = X - X.mean(-1, keepdims=True); Rc = R - R.mean(-1, keepdims=True)
    G = np.einsum("nkt,njt->kj", Rc, Rc); C = np.einsum("nct,nkt->ck", Xc, Rc)
    return C @ np.linalg.pinv(G)


def apply_ols(X, R, B):
    Rc = R - R.mean(-1, keepdims=True)
    return (X - np.einsum("ck,nkt->nct", B, Rc)).astype(np.float32)


def local(X, chans):
    keep = np.zeros(len(EEG), bool); keep[[EEG.index(c) for c in chans]] = True
    out = np.zeros_like(X); sub = X[:, keep, :]; out[:, keep, :] = sub - sub.mean(1, keepdims=True)
    return out


def partner(subj, seed=SHUF_SEED):
    """For every trial, another trial of the same subject (cyclic shift of a random order; as Amendment 2)."""
    rng = np.random.default_rng(seed); p = np.arange(len(subj))
    for s in np.unique(subj):
        ii = np.where(subj == s)[0]
        if len(ii) < 2:
            continue
        o = ii[rng.permutation(len(ii))]; p[o] = o[np.r_[1:len(ii), 0]]
    return p


def transform(arm, TR, D):
    if arm == "main":
        return D["X"], {}
    if arm == "frontal_loc":
        return local(D["X"], FRONTAL), {"channels": FRONTAL, "reference": "local mean"}
    if arm == "central_loc":
        return local(D["X"], CENTRAL), {"channels": CENTRAL, "reference": "local mean"}
    if arm == "veo_ols":
        b = fit_ols(TR["X"], TR["VEO"][:, None, :])
        return apply_ols(D["X"], D["VEO"][:, None, :], b), {"regressor": "VEO (channel labelled HEO)", "b_norm": float(np.linalg.norm(b))}
    b = fit_ols(TR["X"], TR["HOR"][:, None, :])
    if arm == "hor_ols":
        return apply_ols(D["X"], D["HOR"][:, None, :], b), {"regressor": "F8-F7", "b_norm": float(np.linalg.norm(b))}
    if arm == "plc_shuf":
        p = partner(D["subj"])
        return apply_ols(D["X"], D["HOR"][p][:, None, :], b), {"regressor": "F8-F7 of a paired trial of the same subject", "shuf_seed": SHUF_SEED}
    raise ValueError(arm)


def train(a):
    import pandas as pd
    from p3audit.config import load_config
    from p3audit.metrics.intervention import crop
    from p3audit.models.registry import build_model
    from p3audit.training.trainer import predict_logits, resolve_device, train_model
    from p3audit.utils.common import balanced_accuracy
    cfg = load_config(a.config); device = resolve_device(cfg["training"]["device"])
    out = Path(a.out) if a.out else OUT; out.mkdir(parents=True, exist_ok=True)
    F = folds(); (out / "folds.json").write_text(json.dumps(F, indent=1))
    for f in F:
        if a.folds is not None and f["fold"] not in a.folds:
            continue
        TR, ES, TE = load(f["train"]), load([f["es"]]), load(f["test"]); pad, L = TE["pad"], TE["core_len"]
        for arm in a.arms:
            Xtr, info = transform(arm, TR, TR); Xes, _ = transform(arm, TR, ES); Xte, _ = transform(arm, TR, TE)
            for mdl in a.models:
                for sd in a.seeds:
                    p = out / f"{arm}__{mdl}__fold{f['fold']}__seed{sd}.json"
                    if p.exists():
                        continue
                    m = build_model(mdl, cfg, len(EEG), L, EEG); t0 = time.time()
                    tinfo = train_model(m, crop(Xtr, pad, L), TR["y"], crop(Xes, pad, L), ES["y"], cfg, sd, device, a.max_epochs, verbose=False)
                    m.eval(); pred = predict_logits(m, crop(Xte, pad, L), device).argmax(1)
                    pd.DataFrame({"subject": TE["subj"], "trial": TE["trial"], "session": TE["session"], "label": TE["y"],
                                  "correct": (pred == TE["y"]).astype(int), "model": mdl, "seed": sd, "arm": arm,
                                  "fold": f["fold"]}).to_csv(p.with_name(p.stem + "_trials.csv"), index=False)
                    rec = {"arm": arm, "model": mdl, "seed": sd, "fold": f["fold"], "test_subjects": f["test"], "es_subject": f["es"],
                           "ba": float(balanced_accuracy(TE["y"], pred)), "max_epochs_override": a.max_epochs,
                           "training": {k: tinfo[k] for k in ("best_val_ba", "epochs") if k in tinfo}, "seconds": time.time() - t0, **info}
                    p.write_text(json.dumps(rec, indent=1))
                    print(f"{arm:11s} {mdl:8s} fold {f['fold']} seed {sd}: BA {rec['ba']:.3f} ({rec['seconds']:.0f} s)", flush=True)


# --------------------------------------------------------------------------------------------- 3. analyse
def trial_table():
    """Per-trial signals of the registered analysis (no model output)."""
    import pandas as pd
    rows = []
    for s in subjects_in_cache():
        z = np.load(CACHE / f"sub-{s:02d}.npz")
        y = z["y"]; cue = np.where(y == 1, 1.0, -1.0)                       # +1 right cue, -1 left cue
        G, V, H, HD, ses = z["GAZE"].astype(float), z["GVALID"], z["HOR"].astype(float), z["HORD"].astype(float), z["session"]
        vfrac = V[:, idx(*W_VALID)].mean(1); gvalid_trial = vfrac >= TRIAL_VALID_MIN
        # gaze classification: threshold per session max(5 MAD, 0.5 deg) of s over -0.2 ... 1.2 s
        n = int(round(STEP_S * SFREQ)); sg = step_stat(np.where(V, G, np.nan), n)
        thr_g = np.full(len(y), np.nan)
        for q in np.unique(ses):
            mm = ses == q; thr_g[mm] = max(5 * mad(sg[mm][:, idx(*W_VALID)].ravel()), GAZE_FLOOR_DEG)
        cls_g = classify(np.where(V, G, np.nan), cue, thr_g, valid=gvalid_trial)
        # F8 - F7 classification (registered detector on the 0.1-20 Hz copy): threshold per subject max(5 MAD, 15 uV)
        sh = step_stat(HD, n); thr_h = np.full(len(y), max(5 * mad(sh[:, idx(*W_VALID)].ravel()), HOR_FLOOR_UV))
        cls_h = classify(HD, cue, thr_h)
        d_hor = deflection(H, cue); d_gaze = np.where(gvalid_trial, deflection(np.where(V, G, np.nan), cue), np.nan)
        # unsigned (rightward-positive) deflections for the label AUC
        u_hor = deflection(H); u_gaze = np.where(gvalid_trial, deflection(np.where(V, G, np.nan)), np.nan)
        # split halves of the deflection window, each against its own half of the baseline
        h1 = lambda S: (np.nanmean(S[:, idx(0.3, 0.75)], 1) - np.nanmean(S[:, idx(-0.2, -0.1)], 1)) * cue
        h2 = lambda S: (np.nanmean(S[:, idx(0.75, 1.2)], 1) - np.nanmean(S[:, idx(-0.1, 0.0)], 1)) * cue
        Gm = np.where(V, G, np.nan)
        for k in range(len(y)):
            rows.append({"subject": s, "trial": k, "session": int(ses[k]), "label": int(y[k]), "gaze_valid": bool(gvalid_trial[k]),
                         "type_gaze": cls_g[k], "type_hor": cls_h[k], "d_hor": d_hor[k], "d_gaze": d_gaze[k],
                         "u_hor": u_hor[k], "u_gaze": u_gaze[k], "h1_hor": h1(H)[k], "h2_hor": h2(H)[k],
                         "h1_gaze": h1(Gm)[k] if gvalid_trial[k] else np.nan, "h2_gaze": h2(Gm)[k] if gvalid_trial[k] else np.nan})
    T = pd.DataFrame(rows)
    for c in ("d_hor", "d_gaze"):
        T[c + "_z"] = T.groupby("subject")[c].transform(lambda x: (x - x.mean()) / (x.std() + 1e-12))
    nos = T.gaze_valid & (T.type_gaze == "none")
    for c in ("d_hor", "d_gaze"):
        T["tert_" + c] = np.nan
        T.loc[nos, "tert_" + c] = pd.qcut(T.loc[nos, c + "_z"], 3, labels=False).astype(float)
    return T


def analyse(a):
    import pandas as pd
    res_dir = Path(a.out) if a.out else OUT
    df = pd.concat([pd.read_csv(f) for f in glob.glob(str(res_dir / "*_trials.csv"))])
    key = ["model", "subject", "trial"]
    arms = [x for x in ARMS if x in set(df.arm)]
    W = None
    for arm in arms:
        t = df[df.arm == arm].groupby(key, as_index=False).correct.mean().rename(columns={"correct": arm})
        W = t if W is None else W.merge(t, on=key, how="outer")
    T = trial_table()
    W = W.merge(T, on=["subject", "trial"], how="left")

    def G(x, a1, a2, tcol):
        t0, t2 = x[x[tcol] == 0], x[x[tcol] == 2]
        return (t2[a1] - t2[a2]).mean() - (t0[a1] - t0[a2]).mean()

    eligible = []
    for s, x in W[W.model == W.model.iloc[0]].groupby("subject"):
        if ((x.type_gaze == "congruent") & x.gaze_valid).sum() >= K_MIN and ((x.type_gaze == "none") & x.gaze_valid).sum() >= K_MIN:
            eligible.append(s)

    def stats(x):
        o = {}
        xn = x[x.gaze_valid & (x.type_gaze == "none")]
        e = x[x.subject.isin(eligible) & x.gaze_valid]
        c, n = e[e.type_gaze == "congruent"], e[e.type_gaze == "none"]
        if len(eligible) >= R1_MIN_SUBJECTS:
            o["R1_bias_corrected_contrast"] = (c.main.mean() - n.main.mean()) - (c.hor_ols.mean() - n.hor_ols.mean())
        o["R2_gap_no_saccade"] = (xn.main - xn.hor_ols).mean()
        o["R3_G_hor"] = G(xn, "main", "hor_ols", "tert_d_hor")
        o["R4_G_gaze"] = G(xn, "main", "hor_ols", "tert_d_gaze")
        for arm in arms:
            o[f"acc_{arm}"] = x[arm].mean()
        for arm in ("veo_ols", "plc_shuf"):
            if arm in arms:
                o[f"gap_no_saccade_{arm}"] = (xn.main - xn[arm]).mean()
                o[f"G_hor_{arm}"] = G(xn, "main", arm, "tert_d_hor"); o[f"G_gaze_{arm}"] = G(xn, "main", arm, "tert_d_gaze")
        for arm in ("frontal_loc", "central_loc"):
            if arm in arms:
                for tc in ("tert_d_hor", "tert_d_gaze"):
                    o[f"own_gradient_{arm}_{tc[5:]}"] = xn[xn[tc] == 2][arm].mean() - xn[xn[tc] == 0][arm].mean()
        return o

    rng = np.random.default_rng(BOOT_SEED); subs = np.array(sorted(W.subject.unique())); g = {s: x for s, x in W.groupby("subject")}
    est = stats(W); B = pd.DataFrame([stats(pd.concat([g[s] for s in rng.choice(subs, subs.size)])) for _ in range(N_BOOT)])
    ci = lambda k: [float(B[k].quantile(ALPHA / 2)), float(B[k].quantile(1 - ALPHA / 2))] if k in B else None
    tests = {}
    for k, name in (("R1_bias_corrected_contrast", "A4-R1"), ("R2_gap_no_saccade", "A4-R2"), ("R3_G_hor", "A4-R3"), ("R4_G_gaze", "A4-R4")):
        if k not in est:
            tests[name] = {"decision": "not identifiable", "eligible_subjects": len(eligible)}; continue
        c = ci(k); tests[name] = {"statistic": k, "est": float(est[k]), "ci_98.75": c,
                                  "decision": "supported" if c[0] > 0 else "not supported"}
    tests["A4-R1"]["eligible_subjects"] = len(eligible)
    one = W[W.model == W.model.iloc[0]]
    # descriptive outputs
    from collections import Counter
    gv = one[one.gaze_valid]
    labs = ["congruent", "incongruent", "none"]
    cm = pd.crosstab(gv.type_gaze, gv.type_hor).reindex(index=labs, columns=labs, fill_value=0).to_numpy(float)
    po = np.trace(cm) / cm.sum(); pe = (cm.sum(0) * cm.sum(1)).sum() / cm.sum() ** 2
    auc = lambda x, z: float((x[:, None] > z[None, :]).mean() + 0.5 * (x[:, None] == z[None, :]).mean())
    aucs = {}
    for c in ("u_hor", "u_gaze"):
        A = [auc(x[c][(x.label == 1) & x[c].notna()].to_numpy(), x[c][(x.label == 0) & x[c].notna()].to_numpy())
             for _, x in one.groupby("subject") if x[c].notna().sum() >= 10 and x.label[x[c].notna()].nunique() == 2]
        aucs[c] = {"mean": float(np.mean(A)), "n_subjects": len(A), "per_subject": [round(v, 3) for v in A]}
    rel = {}
    for c in ("hor", "gaze"):
        x = one[[f"h1_{c}", f"h2_{c}", "subject"]].dropna()
        z = x.groupby("subject").transform(lambda v: (v - v.mean()) / (v.std() + 1e-12))
        r = float(np.corrcoef(z[f"h1_{c}"], z[f"h2_{c}"])[0, 1]); rel[c] = {"r_halves": r, "spearman_brown": 2 * r / (1 + r)}
    sess = json.loads((CACHE / "sessions.json").read_text())
    out = {"tests": tests, "eligible_subjects_R1": [int(s) for s in eligible],
           "estimates": {k: {"est": float(v), "ci_98.75": ci(k)} for k, v in est.items()},
           "per_model": {m: {arm: float(x[arm].mean()) for arm in arms} | {"gap_no_saccade": float((x[x.gaze_valid & (x.type_gaze == 'none')].main - x[x.gaze_valid & (x.type_gaze == 'none')].hor_ols).mean())}
                         for m, x in W.groupby("model")},
           "trial_counts": {"all": int(len(one)), "gaze_valid": int(one.gaze_valid.sum()),
                            "gaze_type": dict(Counter(one.type_gaze)), "hor_type": dict(Counter(one.type_hor)),
                            "by_label": dict(Counter(one.label.astype(str)))},
           "agreement_gaze_hor": {"crosstab_rows_gaze_cols_hor": cm.astype(int).tolist(), "labels": labs, "kappa": float((po - pe) / (1 - pe))},
           "corr_d_gaze_d_hor": float(one[["d_gaze", "d_hor"]].dropna().corr().iloc[0, 1]),
           "label_auc_0.3-1.2s": aucs, "split_half_reliability": rel,
           "sessions": {"n": len(sess), "gaze_unavailable": [k for k, v in sess.items() if not v.get("gaze_available", False)],
                        "excluded": {k: v["excluded"] for k, v in sess.items() if "excluded" in v},
                        "coupling_with_gaze_x": {k: v["coupling_with_gaze_x"] for k, v in sess.items() if "coupling_with_gaze_x" in v}},
           "settings": {"n_boot": N_BOOT, "seed": BOOT_SEED, "interval": "two-sided 98.75 %", "models": sorted(W.model.unique()), "arms": arms}}
    Path("results/analysis").mkdir(parents=True, exist_ok=True)
    Path("results/analysis/amendment4.json").write_text(json.dumps(out, indent=1))
    for k, v in tests.items():
        print(k, v)


# --------------------------------------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true"); ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--zips", default=None); ap.add_argument("--subjects", nargs="+", type=int, default=None)
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS); ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--folds", nargs="+", type=int, default=None); ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--config", default="configs/default.yaml"); ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.prepare:
        if not a.zips:
            sys.exit("--prepare needs --zips <folder>")
        prepare(a)
    elif a.analyse:
        analyse(a)
    else:
        train(a)
