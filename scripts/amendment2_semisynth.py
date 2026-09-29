"""Amendment 2 (OSF): semi-synthetic re-analysis of the Amendment 1 units (results/units_a1); no new units.
(1) A2-S1: the within-cell comparison of A1-H1 repeated with a probe baseline that is not at ceiling: the probe
    score measured at test amplitudes 0.125 and 0.25 a* (trained and tested at that amplitude, as recorded in each
    unit), instead of at the cell amplitude.  Statistic, cross-fitting and bootstrap exactly as in
    scripts/amendment1_analyse.py (4000 cluster resamples over cell x model groups, seed 20261003).
    Decision per metric (HEOG regression, IG share, residual erasure), Bonferroni over the three metrics:
    supported against a baseline if the lower bound of the 98.33% interval of (metric - baseline) is > 0.
    Registered result: a metric is reported as beating the probe only if it does so against BOTH baselines.
(2) Descriptive: split-half reliability of the residual (within-cell) ground truth and of each metric in the
    Amendment 1 units, and the within-cell partial correlation of D at every recorded test amplitude.
Writes results/analysis/amendment2_semisynth.json.
"""
import json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")
from p3audit.experiments import analysis as A
from amendment1_analyse import corr

METRICS = ["heog_reg", "ig_share", "erasure_keep"]
BASELINES = ["D_0.125", "D_0.25"]
N_BOOT, ALPHA, SEED = 4000, 0.05 / 3, 20261003

units = [json.load(open(p, encoding="utf-8")) for p in sorted(Path("results/units_a1").glob("*.json"))]
us = [r for r in units if r["cell"]["confound"] == "saccade" and r["cell"]["p"] == 0.9]
subs = A._test_subjects(us); fns = A.DEPENDENCE_METRICS["saccade"]
rows = []
for r in us:
    row = {"model": r["model"], "cid": r["cid"], "seed": r["seed"]}
    for k, v in r["probe"].items():
        row[f"D_{k}"] = v["D"]
    row["D_cell"] = r["probe"][f"{float(r['cell']['amp']):g}"]["D"]
    for h in ("half0", "half1"):
        row[f"truth_{h}"] = A.truth(r, subs, h)
        for n in METRICS:
            row[f"{n}_{h}"] = fns[n](r, subs, h)
    rows.append(row)
d = pd.DataFrame(rows)
Dcols = sorted(c for c in d.columns if c.startswith("D_"))


def resid(dd, c):
    return (dd[c] - dd.groupby(["model", "cid"])[c].transform("mean")).values


def xfit(dd, n):
    t0, t1 = resid(dd, "truth_half0"), resid(dd, "truth_half1")
    if n.startswith("D_"):
        x = resid(dd, n); return float(np.nanmean([corr(t0, x), corr(t1, x)]))
    return float(np.nanmean([corr(t0, resid(dd, f"{n}_half1")), corr(t1, resid(dd, f"{n}_half0"))]))


out = {"n_units": len(d), "D_columns": Dcols}
out["reliability"] = {"truth_resid_half_r": corr(resid(d, "truth_half0"), resid(d, "truth_half1")),
                      **{f"{n}_resid_half_r": corr(resid(d, f"{n}_half0"), resid(d, f"{n}_half1")) for n in METRICS}}
out["D_mean"] = {c: float(d[c].mean()) for c in Dcols}
est = {n: xfit(d, n) for n in METRICS + Dcols}
out["partial_r"] = est
grp = d.groupby(["model", "cid"]).indices; keys = list(grp); rng = np.random.default_rng(SEED); B = []
for _ in range(N_BOOT):
    parts = []
    for j, k in enumerate(rng.integers(0, len(keys), len(keys))):
        sub = d.iloc[grp[keys[k]]].copy(); sub["cid"] = sub["cid"] + f"#{j}"; parts.append(sub)
    dd = pd.concat(parts, ignore_index=True)
    v = {n: xfit(dd, n) for n in METRICS + BASELINES}
    B.append({f"{m}-{b}": v[m] - v[b] for m in METRICS for b in BASELINES})
B = pd.DataFrame(B)
out["tests"] = {}
for m in METRICS:
    out["tests"][m] = {}
    for b in BASELINES:
        lo, hi = B[f"{m}-{b}"].quantile([ALPHA / 2, 1 - ALPHA / 2])
        out["tests"][m][b] = {"diff": est[m] - est[b], "ci_98.33": [float(lo), float(hi)], "pass": bool(lo > 0)}
    out["tests"][m]["supported_against_both"] = all(out["tests"][m][b]["pass"] for b in BASELINES)
Path("results/analysis").mkdir(parents=True, exist_ok=True)
json.dump(out, open("results/analysis/amendment2_semisynth.json", "w"), indent=1)
print(json.dumps({"reliability": out["reliability"], "D_mean": out["D_mean"]}, indent=1))
