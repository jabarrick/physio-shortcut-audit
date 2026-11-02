"""Sensitivity reference for the microsaccade detector of scripts/posthoc_revision14_review6.py (b): detection rate of
the same detector (Engbert-Kliegl, 200 Hz gaze grid, threshold 6 x median-based SD per session, >= 3 samples,
amplitude < 1 deg) in all gaze-valid trials by trial type, per second, over 0-4 s and 1.5-4 s after the cue.
Output: results/analysis/posthoc_a4_microsaccade_rate.json"""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, "scripts")
import amendment4_gf2025 as a
def ek(g, valid, sd, fs=200.0, lam=6.0, min_n=3):
    v = np.full(g.shape, np.nan); v[2:-2] = (g[4:] + g[3:-1] - g[1:-3] - g[:-4]) * fs / 6.0; v[~valid] = np.nan
    over = np.abs(v) > lam * sd; ev, i = [], 0
    while i < len(over):
        if over[i]:
            j = i
            while j + 1 < len(over) and over[j + 1]: j += 1
            if j - i + 1 >= min_n:
                amp = g[min(j + 1, len(g) - 1)] - g[max(i - 1, 0)]
                if np.isfinite(amp) and abs(amp) < 1.0: ev.append(i)
            i = j + 1
        else: i += 1
    return ev
T = a.trial_table(); rows = []
for s in a.subjects_in_cache():
    z = np.load(a.CACHE / f"sub-{s:02d}.npz"); G, V, ses = z["GAZE"].astype(float), z["GVALID"], z["session"]; pad = int(z["pad"])
    for q in np.unique(ses):
        m = ses == q; vv = (G[m][:, 4:] + G[m][:, 3:-1] - G[m][:, 1:-3] - G[m][:, :-4]) * 200.0 / 6.0
        vv = vv[V[m][:, 2:-2] & np.isfinite(vv)]; sd = np.sqrt(np.median(vv ** 2) - np.median(vv) ** 2) if vv.size > 100 else np.nan
        for k in np.where(m)[0]:
            ev = ek(G[k], V[k], sd) if np.isfinite(sd) else []
            rows.append({"subject": s, "trial": k, "n_0_4": sum(pad <= i < pad + 800 for i in ev), "n_15_4": sum(pad + 300 <= i < pad + 800 for i in ev)})
M = pd.DataFrame(rows).merge(T[["subject", "trial", "gaze_valid", "type_gaze"]], on=["subject", "trial"]); M = M[M.gaze_valid]
out = {ty: {"n_trials": int(len(x)), "per_s_0_4": float(x.n_0_4.mean() / 4), "per_s_1.5_4": float(x.n_15_4.mean() / 2.5),
            "share_trials_with_any_0_4": float((x.n_0_4 > 0).mean())} for ty, x in M.groupby("type_gaze")}
out["all_gaze_valid"] = {"n_trials": int(len(M)), "per_s_0_4": float(M.n_0_4.mean() / 4), "per_s_1.5_4": float(M.n_15_4.mean() / 2.5)}
json.dump(out, open("results/analysis/posthoc_a4_microsaccade_rate.json", "w"), indent=1); print(json.dumps(out, indent=1))
