"""Post hoc analyses added in response to the internal referee report (PILOT_LOG 17.28).
Exploratory; not preregistered. Uses the frozen analysis functions and results/units.
Run from the repository root: python scripts/posthoc_revision.py"""
import sys, json, warnings, numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
from p3audit.experiments import analysis as A

units = [json.load(open(p, encoding="utf-8")) for p in sorted(Path("results/units").glob("*.json"))]
out = {}
for conf, metrics in A.DEPENDENCE_METRICS.items():
    us = [r for r in units if r["cell"]["confound"] == conf and r["cell"]["p"] == 0.9 and r["cell"]["module"] in ("factorial", "p_series")]
    subs = A._test_subjects(us)
    rows = []
    for r in us:
        a = float(r["cell"]["amp"])
        row = {"model": r["model"], "s": r["cell"]["s"], "amp": a, "truth": A.truth(r, subs),
               "truth_rev": A.truth(r, subs, kind="reversed"), "D": r["probe"][f"{a:g}"]["D"]}
        row.update({n: fn(r, subs) for n, fn in metrics.items()})
        rows.append(row)
    d = pd.DataFrame(rows)
    zw = lambda c: d.groupby("model")[c].transform(lambda v: (v - v.mean()) / v.std(ddof=1))
    d["tz"] = zw("truth"); res = {"design_R2": smf.ols("tz ~ C(amp)*C(s)*C(model)", d).fit().rsquared,
                                  "amp_r": float(np.corrcoef(d.tz, d.groupby("model").amp.transform(lambda v: v - v.mean()))[0, 1])}
    rt = smf.ols("tz ~ C(amp)*C(s)*C(model)", d).fit().resid
    for c in ["D"] + list(metrics):
        d["mz"] = zw(c); rm = smf.ols("mz ~ C(amp)*C(s)*C(model)", d).fit().resid
        res[c] = {"r": float(np.corrcoef(d.tz, d.mz)[0, 1]), "partial_r": float(stats.pearsonr(rt, rm)[0])}
        if c != "D":
            res[c]["r_rev"] = float(np.corrcoef(zw("truth_rev"), d.mz)[0, 1])
    # subject-level cross-fitted validity
    for n, fn in metrics.items():
        zs = []
        for r in us:
            t0, t1 = (np.array([A.truth(r, [s], h) for s in subs]) for h in ("half0", "half1"))
            m0, m1 = (np.array([fn(r, [s], h) for s in subs]) for h in ("half0", "half1"))
            v = [np.corrcoef(x[np.isfinite(x) & np.isfinite(y)], y[np.isfinite(x) & np.isfinite(y)])[0, 1] for x, y in ((t0, m1), (t1, m0))]
            v = [x for x in v if np.isfinite(x)]
            if v: zs.append(np.mean(np.arctanh(np.clip(v, -.999, .999))))
        res[n]["subject_r"] = float(np.tanh(np.mean(zs)))
    # subject-only bootstrap with Bonferroni (6 confirmatory tests)
    ma = A.MetricArrays(us, subs, ["truth"] + [m for m in metrics if m in A.SPECS]); S = len(subs)
    for n in A.CONFIRMATORY[conf]:
        rng = np.random.default_rng(1)
        sv = [A.xfit_corr(ma, n, np.bincount(rng.integers(0, S, S), minlength=S).astype(float), np.arange(len(ma.units))) for _ in range(2000)]
        res[n]["subject_only_ci_bonf"] = [float(np.quantile(sv, .05 / 12)), float(np.quantile(sv, 1 - .05 / 12))]
    out[conf] = res
json.dump(out, open("results/analysis/posthoc_revision.json", "w"), indent=1)
print(json.dumps(out, indent=1))
