"""Post hoc (exploratory, after the second critical review, 30 Sep 2026): is posterior alpha LATERALISATION coupled
to the label in real PhysioNet imagery (runs 4, 8, 12)?  The H5 alpha proxy (median split of total posterior alpha
power) was nearly independent of the label (phi = 0.01); a cue on the left or right of the screen should instead
lateralise alpha.  Uses the cached model-stream epochs (results/real_cache; 200 Hz, 0.5-75 Hz, average reference).
Alpha index per trial: ALI = (P_left - P_right) / (P_left + P_right), 8-13 Hz power from the FFT of a Hann-windowed
segment; left = P7, P5, P3, PO7, PO3, O1; right = P8, P6, P4, PO8, PO4, O2.  Label y = 0 (T1, left target), 1 (right).
Per subject: point-biserial r(ALI, y) and AUC; summary: mean over subjects with a subject bootstrap (2000, seed 7).
Whole window 0.5-4.0 s after the cue, and 1 s windows every 0.5 s (0-1, 0.5-1.5, ..., 3-4 s); all trials and trials
without a detected saccade.  For comparison, the same statistics for total posterior alpha log power.
By region (added the same day, before interpretation was written): occipital (O1, PO7, PO3 vs O2, PO8, PO4),
parietal (P7, P5, P3 vs P8, P6, P4) and central (C3, C5, CP3 vs C4, C6, CP4, where imagery itself lateralises the mu
rhythm) in 0-1, 1-2 and 2-4 s after the cue, all trials.
Writes results/analysis/posthoc_revision5_alpha_lateral.json.
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p3audit.constants import PHYSIONET_CHANNELS as CH
FS = 200.0
L = [CH.index(c) for c in ("P7", "P5", "P3", "PO7", "PO3", "O1")]
R = [CH.index(c) for c in ("P8", "P6", "P4", "PO8", "PO4", "O2")]

def band_power(seg):  # seg: trials x ch x t
    n = seg.shape[-1]; w = np.hanning(n)
    F = np.fft.rfft((seg - seg.mean(-1, keepdims=True)) * w, axis=-1); f = np.fft.rfftfreq(n, 1 / FS)
    k = (f >= 8) & (f <= 13); return (np.abs(F[..., k]) ** 2).mean(-1)

def auc(x, y):
    a, b = x[y == 1], x[y == 0]
    if len(a) < 3 or len(b) < 3: return np.nan
    return float((a[:, None] > b[None, :]).mean() + 0.5 * (a[:, None] == b[None, :]).mean())

def pbr(x, y):
    return float(np.corrcoef(x, y)[0, 1]) if len(np.unique(y)) == 2 and x.std() > 0 else np.nan

windows = {"0.5-4.0": (0.5, 4.0)}
for s in np.arange(0, 3.01, 0.5): windows[f"{s:g}-{s+1:g}"] = (s, s + 1)
per = {k: {"ali_r": [], "ali_auc": [], "tot_r": [], "ali_r_none": [], "ali_auc_none": []} for k in windows}
subs = []
for p in sorted(Path("results/real_cache").glob("sub-*.npz")):
    z = np.load(p); X, y, typ, pad = z["X"], z["y"].astype(int), z["type"], int(z["pad"])
    subs.append(p.stem)
    for k, (a, b) in windows.items():
        seg = X[:, :, pad + int(a * FS): pad + int(b * FS)]
        P = band_power(seg); pl, pr = P[:, L].mean(1), P[:, R].mean(1)
        ali = (pl - pr) / (pl + pr); tot = np.log(P[:, L + R].mean(1))
        none = typ == "none"
        per[k]["ali_r"].append(pbr(ali, y)); per[k]["ali_auc"].append(auc(ali, y)); per[k]["tot_r"].append(pbr(tot, y))
        per[k]["ali_r_none"].append(pbr(ali[none], y[none])); per[k]["ali_auc_none"].append(auc(ali[none], y[none]))
rng = np.random.default_rng(7); out = {"n_subjects": len(subs), "left": [CH[i] for i in L], "right": [CH[i] for i in R], "windows": {}}
for k, d in per.items():
    out["windows"][k] = {}
    for m, v in d.items():
        v = np.array(v, float); v = v[np.isfinite(v)]
        bs = [rng.choice(v, len(v)).mean() for _ in range(2000)]
        out["windows"][k][m] = {"mean": float(v.mean()), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                                "n": int(len(v)), "frac_subjects_r_negative" if m.endswith("_r") or m.endswith("r_none") else "frac_auc_below_0.5": float((v < (0.5 if "auc" in m else 0)).mean())}
REG = {"occipital": (("O1", "PO7", "PO3"), ("O2", "PO8", "PO4")), "parietal": (("P7", "P5", "P3"), ("P8", "P6", "P4")),
       "central": (("C3", "C5", "CP3"), ("C4", "C6", "CP4"))}
RW = {"0-1": (0, 1), "1-2": (1, 2), "2-4": (2, 4)}
reg = {g: {w: [] for w in RW} for g in REG}
for p in sorted(Path("results/real_cache").glob("sub-*.npz")):
    z = np.load(p); X, y, pad = z["X"], z["y"].astype(int), int(z["pad"])
    for w, (a, b) in RW.items():
        P = band_power(X[:, :, pad + int(a * FS): pad + int(b * FS)])
        for g, (l, r) in REG.items():
            pl = P[:, [CH.index(c) for c in l]].mean(1); pr = P[:, [CH.index(c) for c in r]].mean(1)
            reg[g][w].append(pbr((pl - pr) / (pl + pr), y))
out["by_region_ali_r"] = {}
for g in REG:
    out["by_region_ali_r"][g] = {}
    for w, v in reg[g].items():
        v = np.array(v, float); v = v[np.isfinite(v)]; bs = [rng.choice(v, len(v)).mean() for _ in range(2000)]
        out["by_region_ali_r"][g][w] = {"mean": float(v.mean()), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]}
Path("results/analysis").mkdir(parents=True, exist_ok=True)
json.dump(out, open("results/analysis/posthoc_revision5_alpha_lateral.json", "w"), indent=1)
print(json.dumps(out["by_region_ali_r"]))
for k, d in out["windows"].items():
    print(k, {m: (round(v["mean"], 3), [round(c, 3) for c in v["ci95"]]) for m, v in d.items()})
