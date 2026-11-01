"""Exploratory analyses after review round 6 (NOT registered; computed after all registered results were known).

(a) Amendment 4 data: accuracy of main and F8 - F7-regressed decoders by gaze-defined trial type, and a cross-subject
    single-feature baseline (sign of the unsigned 0.3-1.2 s deflection of gaze or of F8 - F7, threshold 0, no training)
    to compare with the per-subject AUCs.
(b) Amendment 4 data: horizontal microsaccades in gaze-valid trials without a measured saccade (Engbert & Kliegl 2003
    velocity criterion on the 200 Hz gaze series: 5-point velocity, threshold 6 x median-based SD per session, >= 3
    samples, amplitude < 1 deg).  Cue-signed microsaccade index per trial (toward minus away, 0-1.5 s and 1.5-4 s) and
    the main-minus-control gap by index group.  The Tobii TX300 is a remote tracker; microsaccade detection is noisy.
(c) BCI Competition IV 2a: label-free check of the assumed HEOG polarity (EOG3 - EOG1, positive = rightward): correlation
    of the sample-to-sample changes of HEOG with those of FC4 - FC3 (cached, average-referenced EEG) per subject.
Writes results/analysis/posthoc_revision14_review6.json.  Subject bootstrap 2000, seed 20261106.
"""
import glob, json, importlib.util
from pathlib import Path
import numpy as np, pandas as pd

rng = np.random.default_rng(20261106)
spec = importlib.util.spec_from_file_location("a4", "scripts/amendment4_gf2025.py"); a4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(a4)
out = {}

def boot(x, fn, n=2000):
    subs = np.array(sorted(x.subject.unique())); g = {s: v for s, v in x.groupby("subject")}
    est = fn(x); B = [fn(pd.concat([g[s] for s in rng.choice(subs, subs.size)])) for _ in range(n)]
    return {"est": float(est), "ci_95": [float(np.nanpercentile(B, 2.5)), float(np.nanpercentile(B, 97.5))]}

# ---------------- (a)
df = pd.concat([pd.read_csv(f) for f in glob.glob("results/gf2025/*_trials.csv")])
W = None
for arm in ("main", "hor_ols"):
    t = df[df.arm == arm].groupby(["model", "subject", "trial"], as_index=False).correct.mean().rename(columns={"correct": arm})
    W = t if W is None else W.merge(t, on=["model", "subject", "trial"])
T = a4.trial_table(); W = W.merge(T, on=["subject", "trial"])
out["a_accuracy_by_gaze_type"] = {ty: {"n_trials": int((x.model == x.model.iloc[0]).sum()), "main": boot(x, lambda v: v.main.mean()), "hor_ols": boot(x, lambda v: v.hor_ols.mean())}
                                  for ty, x in W.groupby("type_gaze")}
one = T.copy()
for c in ("u_gaze", "u_hor"):
    x = one[one[c].notna()].copy(); x["corr"] = ((x[c] > 0).astype(int) == x.label).astype(float)
    out[f"a_sign_rule_cross_subject_{c}"] = boot(x, lambda v: v["corr"].mean())

# ---------------- (b)
def ek_microsaccades(g, valid, sd_med, fs=200.0, lam=6.0, min_n=3):
    v = np.full(g.shape, np.nan); v[2:-2] = (g[4:] + g[3:-1] - g[1:-3] - g[:-4]) * fs / 6.0
    v[~valid] = np.nan
    over = np.abs(v) > lam * sd_med
    ev, i = [], 0
    while i < len(over):
        if over[i]:
            j = i
            while j + 1 < len(over) and over[j + 1]:
                j += 1
            if j - i + 1 >= min_n:
                amp = g[min(j + 1, len(g) - 1)] - g[max(i - 1, 0)]
                if np.isfinite(amp) and abs(amp) < 1.0:
                    ev.append((i, np.sign(amp)))
            i = j + 1
        else:
            i += 1
    return ev
rows = []
for s in a4.subjects_in_cache():
    z = np.load(a4.CACHE / f"sub-{s:02d}.npz"); G, V, y, ses = z["GAZE"].astype(float), z["GVALID"], z["y"], z["session"]
    pad = int(z["pad"])
    for q in np.unique(ses):
        m = ses == q
        vv = (G[m][:, 4:] + G[m][:, 3:-1] - G[m][:, 1:-3] - G[m][:, :-4]) * 200.0 / 6.0
        vv = vv[V[m][:, 2:-2] & np.isfinite(vv)]
        sd_med = np.sqrt(np.median(vv ** 2) - np.median(vv) ** 2) if vv.size > 100 else np.nan
        for k in np.where(m)[0]:
            cue = 1.0 if y[k] == 1 else -1.0
            ev = ek_microsaccades(G[k], V[k], sd_med) if np.isfinite(sd_med) else []
            early = [d for i, d in ev if pad <= i < pad + 300]; late = [d for i, d in ev if pad + 300 <= i < pad + 800]
            rows.append({"subject": s, "trial": k, "ms_early_idx": float(sum(d * cue for d in early)), "ms_late_idx": float(sum(d * cue for d in late)),
                         "ms_n": len(early) + len(late)})
M = pd.DataFrame(rows); X = W.merge(M, on=["subject", "trial"])
N = X[X.gaze_valid & (X.type_gaze == "none")]
n1 = N[N.model == N.model.iloc[0]]
out["b_microsaccades"] = {"n_trials": int(len(n1)), "trials_with_any": int((n1.ms_n > 0).sum()),
                          "mean_per_trial_0_4s": float(n1.ms_n.mean()),
                          "toward_minus_away_early_mean": boot(n1, lambda v: v.ms_early_idx.mean()),
                          "toward_minus_away_late_mean": boot(n1, lambda v: v.ms_late_idx.mean())}
for w in ("ms_early_idx", "ms_late_idx"):
    grp = {"toward": N[N[w] > 0], "none_or_balanced": N[N[w] == 0], "away": N[N[w] < 0]}
    out[f"b_gap_by_{w}"] = {k: {"n_trials": int((v.model == N.model.iloc[0]).sum()), "gap": boot(v, lambda u: (u.main - u.hor_ols).mean()) if len(v) else None,
                                "main": float(v.main.mean()) if len(v) else None} for k, v in grp.items()}

# ---------------- (c)
EEG = ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz"]
pol = {}
for f in sorted(glob.glob("results/bnci2a_cache/sub-*.npz")):
    z = np.load(f); X2, E = z["X"].astype(float), z["EOG"].astype(float)
    h = E[:, 2, :] - E[:, 0, :]; fc = X2[:, EEG.index("FC4"), :] - X2[:, EEG.index("FC3"), :]; c6 = X2[:, EEG.index("C6"), :] - X2[:, EEG.index("C5"), :]
    dh, dfc, dc6 = np.diff(h, axis=1).ravel(), np.diff(fc, axis=1).ravel(), np.diff(c6, axis=1).ravel()
    pol[Path(f).stem] = {"r_dHEOG_dFC4-FC3": float(np.corrcoef(dh, dfc)[0, 1]), "r_dHEOG_dC6-C5": float(np.corrcoef(dh, dc6)[0, 1])}
out["c_bnci2a_heog_polarity"] = pol
Path("results/analysis/posthoc_revision14_review6.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1)[:6000])
