"""Descriptive outputs for Amendment 5 (computed after the registered analysis): per-model estimates and accuracy by
tercile of the deflection for every arm, in the evaluation subsets of scripts/amendment5_exclusion_gaze.py.
Output: results/analysis/posthoc_a5_descriptives.json"""
import sys, glob, json, numpy as np, pandas as pd
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import amendment5_exclusion_gaze as m
from p3audit.constants import PHYSIONET_CHANNELS as CH
rd = lambda p: pd.concat([pd.read_csv(f) for f in glob.glob(p)])
out = {}
# PhysioNet
key = ["model", "subject", "trial", "type", "label"]
main = rd("results/real/*_trials.csv"); main = main[main.arm == "main"]; r3 = rd("results/real_r3/*_trials.csv"); a5 = rd("results/real_a5/*_trials.csv")
W = m.wide({"main": main, "heog_ols": r3[r3.arm == "heog_ols"], "main_excl": a5[a5.arm == "main_excl"], "ols_excl": a5[a5.arm == "ols_excl"]}, key)
i7, i8 = CH.index("AF7"), CH.index("AF8"); defl = {}
for s in sorted(W.subject.unique()):
    z = np.load(f"results/real_cache/sub-{s:03d}.npz", allow_pickle=True); h = z["X"][:, i7, :] - z["X"][:, i8, :]; p = int(z["pad"])
    v = (h[:, p + 60:p + 240].mean(1) - h[:, p - 40:p].mean(1)) * np.where(z["y"] == 1, 1.0, -1.0); defl.update({(s, k): v[k] for k in range(len(v))})
W["d"] = [defl[(s, k)] for s, k in zip(W.subject, W.trial)]; W["d"] *= 1.0 if W.loc[W.type == "congruent", "d"].mean() > 0 else -1.0
W["dz"] = W.groupby(["model", "subject"]).d.transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
n = W.type == "none"; W.loc[n, "tert"] = W[n].groupby("model").dz.transform(lambda x: pd.qcut(x, 3, labels=False))
N = W[n]; arms = ["main", "heog_ols", "main_excl", "ols_excl"]
out["physionet"] = {"tercile_accuracy": {a: [round(float(N[N.tert == k][a].mean()), 3) for k in range(3)] for a in arms},
    "per_model": {mm: {"G_excl": float(m.G(x, "main_excl", "ols_excl")), "G_full": float(m.G(x, "main", "heog_ols")), "acc_main_excl": float(W[W.model == mm].main_excl.mean()),
                       "acc_ols_excl": float(W[W.model == mm].ols_excl.mean())} for mm, x in N.groupby("model")},
    "accuracy_by_type": {a: {t: round(float(W[W.type == t][a].mean()), 3) for t in ("congruent", "incongruent", "none")} for a in arms}}
# eye-tracking dataset
_, Tc = m.gf_classes()
g4 = rd("results/gf2025/*_trials.csv"); g5 = rd("results/gf2025_a5/*_trials.csv")
arms = ["main", "hor_ols", "gaze_ols", "main_excl", "ols_excl"]
W = m.wide({a: (g4 if a in ("main", "hor_ols") else g5)[lambda d, a=a: d.arm == a] for a in arms}, ["model", "subject", "trial"])
W = W.merge(Tc[["subject", "trial", "gaze_valid", "type_gaze", "d_hor", "d_gaze"]], on=["subject", "trial"])
for c, zc in (("d_hor", "zh"), ("d_gaze", "zg")):
    W[zc] = W.groupby(["model", "subject"])[c].transform(lambda x: (x - x.mean()) / (x.std() + 1e-12))
nn = W.gaze_valid & (W.type_gaze == "none"); N = W[nn].copy()
N["tert"] = N.groupby("model").zh.transform(lambda x: pd.qcut(x, 3, labels=False)); N["tg"] = N.groupby("model").zg.transform(lambda x: pd.qcut(x, 3, labels=False))
out["eye_tracking"] = {"n_trials": int(nn.sum() // 4),
    "tercile_accuracy_hor": {a: [round(float(N[N.tert == k][a].mean()), 3) for k in range(3)] for a in arms},
    "tercile_accuracy_gaze": {a: [round(float(N[N.tg == k][a].mean()), 3) for k in range(3)] for a in arms},
    "per_model": {mm: {"G_excl": float(m.G(x, "main_excl", "ols_excl")), "gap_gaze_ols": float((x.main - x.gaze_ols).mean()), "G_gaze_ols": float(m.G(x, "main", "gaze_ols")),
                       "acc_main_excl": float(W[W.model == mm].main_excl.mean()), "acc_gaze_ols": float(W[W.model == mm].gaze_ols.mean())} for mm, x in N.groupby("model")},
    "accuracy_by_gaze_type": {a: {t: round(float(W[W.gaze_valid & (W.type_gaze == t)][a].mean()), 3) for t in ("congruent", "incongruent", "none")} for a in arms}}
json.dump(out, open("results/analysis/posthoc_a5_descriptives.json", "w"), indent=1); print(json.dumps(out, indent=1))
