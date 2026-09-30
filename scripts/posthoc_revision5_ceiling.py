"""Post hoc (exploratory, 30 Sep 2026): does the cross-fitted within-cell correlation of each metric with ground truth
exceed the ceiling sqrt(rel_truth * rel_metric) implied by the split-half reliabilities of the within-cell residuals?
Cluster bootstrap (cell x model groups, 4000, seed 20261006) of the ratio r_xfit / ceiling and of r_xfit - ceiling,
for the original p = 0.9 units (as posthoc_revision2_diff.py) and the Amendment 1 units.
Writes results/analysis/posthoc_revision5_ceiling.json.
"""
import json, sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore"); sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from p3audit.experiments import analysis as A
from posthoc_revision5_bamatched import corr, build

def stat(X, groups, metrics):
    idx = np.concatenate(groups); lab = np.concatenate([np.full(len(m), j) for j, m in enumerate(groups)])
    def res(v):
        v = v[idx]; ok = np.isfinite(v); s = np.bincount(lab[ok], v[ok], minlength=len(groups)); n = np.bincount(lab[ok], minlength=len(groups))
        return v - (s / np.maximum(n, 1))[lab]
    R = {k: res(v) for k, v in X.items()}; rt = corr(R["t0"], R["t1"]); o = {"rel_truth": rt}
    for n in metrics:
        rm = corr(R[n + "0"], R[n + "1"]); x = np.nanmean([corr(R["t0"], R[n + "1"]), corr(R["t1"], R[n + "0"])])
        c = np.sqrt(max(rt, 0) * max(rm, 0)); o.update({n + "_xfit": x, n + "_rel": rm, n + "_ceiling": c, n + "_ratio": x / c if c > 0 else np.nan, n + "_excess": x - c})
    return o

def run(units, conf, metrics):
    X, groups = build(units, conf, metrics); est = stat(X, groups, metrics)
    rng = np.random.default_rng(20261006); B = [stat(X, [groups[k] for k in rng.integers(0, len(groups), len(groups))], metrics) for _ in range(4000)]
    out = {"estimates": {k: float(v) for k, v in est.items()}, "ci95": {}}
    for k in est:
        v = np.array([b[k] for b in B], float); v = v[np.isfinite(v)]
        out["ci95"][k] = [float(np.quantile(v, .025)), float(np.quantile(v, .975))]
    return out

res = {}
orig = [json.load(open(p, encoding="utf-8")) for p in sorted(Path("results/units").glob("*.json"))]
for conf in ("alpha", "saccade"):
    us = [r for r in orig if r["cell"]["confound"] == conf and r["cell"]["p"] == 0.9 and r["cell"]["module"] in ("factorial", "p_series")]
    res[f"original_{conf}"] = run(us, conf, A.CONFIRMATORY[conf])
a1 = [json.load(open(p, encoding="utf-8")) for p in sorted(Path("results/units_a1").glob("*.json"))]
res["amendment1_saccade"] = run([r for r in a1 if r["cell"]["confound"] == "saccade" and r["cell"]["p"] == 0.9], "saccade", ["heog_reg", "ig_share", "erasure_keep"])
json.dump(res, open("results/analysis/posthoc_revision5_ceiling.json", "w"), indent=1)
for k, v in res.items():
    print(k, "rel_truth", round(v["estimates"]["rel_truth"], 2), v["ci95"]["rel_truth"])
    for m in [x[:-6] for x in v["estimates"] if x.endswith("_ratio")]:
        e = v["estimates"]; c = v["ci95"]
        print("   ", m, "xfit", round(e[m + "_xfit"], 2), "ceil", round(e[m + "_ceiling"], 2), "excess", round(e[m + "_excess"], 2), [round(x, 2) for x in c[m + "_excess"]])
