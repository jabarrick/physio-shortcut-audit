"""Exploratory descriptives for the Amendment 4 data (NOT registered; computed after the registered analysis).

Uses the registered trial table of scripts/amendment4_gf2025.py (same classification, deflections and terciles) and
the records in results/gf2025.  Outputs: mean gaze shift (deg) and F8 - F7 deflection (uV) by gaze-defined trial type;
per-model no-saccade gap, G_HOR and G_gaze with 95% subject-bootstrap intervals (2000 resamples, seed 20261104);
accuracy of main and hor_ols by tercile of each deflection (pooled models).
Writes results/analysis/posthoc_revision13_a4_descriptives.json.
"""
import glob, json, sys, importlib.util
from pathlib import Path
import numpy as np, pandas as pd

spec = importlib.util.spec_from_file_location("a4", "scripts/amendment4_gf2025.py"); a4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(a4)
df = pd.concat([pd.read_csv(f) for f in glob.glob("results/gf2025/*_trials.csv")])
W = None
for arm in ("main", "hor_ols"):
    t = df[df.arm == arm].groupby(["model", "subject", "trial"], as_index=False).correct.mean().rename(columns={"correct": arm})
    W = t if W is None else W.merge(t, on=["model", "subject", "trial"])
T = a4.trial_table(); W = W.merge(T, on=["subject", "trial"])
one = W[W.model == W.model.iloc[0]]
out = {"by_gaze_type": {}}
for ty, x in one.groupby("type_gaze"):
    out["by_gaze_type"][ty] = {"n": int(len(x)), "d_gaze_deg_mean": float(x.d_gaze.mean()), "d_hor_uV_mean": float(x.d_hor.mean())}
rng = np.random.default_rng(20261104)
def stats(x):
    xn = x[x.gaze_valid & (x.type_gaze == "none")]; o = {"gap": (xn.main - xn.hor_ols).mean()}
    for tc in ("tert_d_hor", "tert_d_gaze"):
        t0, t2 = xn[xn[tc] == 0], xn[xn[tc] == 2]
        o["G_" + tc[7:]] = (t2.main - t2.hor_ols).mean() - (t0.main - t0.hor_ols).mean()
    return o
out["per_model"] = {}
for m, x in W.groupby("model"):
    subs = np.array(sorted(x.subject.unique())); g = {s: v for s, v in x.groupby("subject")}
    est = stats(x); B = pd.DataFrame([stats(pd.concat([g[s] for s in rng.choice(subs, subs.size)])) for _ in range(2000)])
    out["per_model"][m] = {k: {"est": float(v), "ci_95": [float(B[k].quantile(.025)), float(B[k].quantile(.975))]} for k, v in est.items()}
xn = W[W.gaze_valid & (W.type_gaze == "none")]
out["tercile_accuracy"] = {tc: {int(t): {"main": float(v.main.mean()), "hor_ols": float(v.hor_ols.mean())} for t, v in xn.groupby(tc)} for tc in ("tert_d_hor", "tert_d_gaze")}
Path("results/analysis/posthoc_revision13_a4_descriptives.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
