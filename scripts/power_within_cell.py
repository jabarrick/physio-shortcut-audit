"""Power simulation for a confirmatory within-cell test (planning for the OSF amendment).

Question: with k training seeds per cell x model group, how often would the cross-fitted
within-cell partial correlation of a reliance metric with ground truth (DeltaBA_neu) exceed
that of the probe score D, with a 95% interval above zero?

Method (moment-matched Gaussian model, estimated from the 358 confirmatory units):
  after removing cell x model means, each unit i has a latent true reliance deviation t_i and
      truth half h:  g_ih = t_i + e_ih                       (window-sampling noise, halves independent)
      metric half h: m_ih = b*t_i + v_i + u_ih               (v: seed-level metric noise, u: window noise)
      probe:         D_i  = c*t_i + w_i
  Var(t)=cov(g0,g1); b*Var(t)=cov(g_h,m_h') (cross halves); Var(u) from the half difference, etc.
  Because every quantity is residualised within groups, variances are rescaled by n/(n-1) per group.
The simulated groups have the same count as observed; each gets k seeds. For each k the
statistic (metric partial r minus D partial r, both cross-fitted as in posthoc_revision2.py) is
simulated 2000 times; power = share of replicates whose value exceeds 1.96 x its simulated SD.
Exploratory planning tool, outside the frozen package.  Run from the repo root.
"""
import sys, json, warnings
import numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
from p3audit.experiments import analysis as A

units = [json.load(open(p, encoding="utf-8")) for p in sorted(Path("results/units").glob("*.json"))]
rng = np.random.default_rng(20260927)
KS = [2, 3, 4, 5, 6, 8]
NSIM = 2000


def build(conf, metrics):
    us = [r for r in units if r["cell"]["confound"] == conf and r["cell"]["p"] == 0.9
          and r["cell"]["module"] in ("factorial", "p_series")]
    subs = A._test_subjects(us)
    rows = []
    for r in us:
        a = float(r["cell"]["amp"])
        row = {"model": r["model"], "cid": r["cid"], "D": r["probe"][f"{a:g}"]["D"]}
        for h in ("half0", "half1"):
            row[f"g_{h}"] = A.truth(r, subs, h)
            for n, fn in metrics.items():
                row[f"{n}_{h}"] = fn(r, subs, h)
        rows.append(row)
    d = pd.DataFrame(rows)
    n = d.groupby(["model", "cid"])["D"].transform("size")
    d = d[n >= 2].copy()
    scale = np.sqrt(d.groupby(["model", "cid"])["D"].transform("size") / (d.groupby(["model", "cid"])["D"].transform("size") - 1))
    R = {c: (d[c] - d.groupby(["model", "cid"])[c].transform("mean")) * scale for c in d.columns if c not in ("model", "cid")}
    return pd.DataFrame(R), d.groupby(["model", "cid"]).ngroups


def cov(a, b):
    k = np.isfinite(a) & np.isfinite(b)
    return float(np.cov(a[k], b[k])[0, 1])


def params(R, n):
    g0, g1, m0, m1, D = R["g_half0"], R["g_half1"], R[f"{n}_half0"], R[f"{n}_half1"], R["D"]
    vt = max(cov(g0, g1), 1e-12)
    ve = max((np.nanvar(g0 - g1) / 2), 1e-12)
    b = np.mean([cov(g0, m1), cov(g1, m0)]) / vt
    vu = max(np.nanvar(m0 - m1) / 2, 0)
    vv = max(cov(m0, m1) - b * b * vt, 0)
    c = np.mean([cov(g0, D), cov(g1, D)]) / vt
    vw = max(np.nanvar(D) - c * c * vt, 1e-12)
    return dict(vt=vt, ve=ve, b=b, vu=vu, vv=vv, c=c, vw=vw)


def xfit(g0, g1, x0, x1, grp):
    def res(x):
        s = pd.Series(x); return (s - s.groupby(grp).transform("mean")).values
    G0, G1, X0, X1 = map(res, (g0, g1, x0, x1))
    return 0.5 * (np.corrcoef(G0, X1)[0, 1] + np.corrcoef(G1, X0)[0, 1])


def simulate(p, ngroups, k):
    grp = np.repeat(np.arange(ngroups), k); N = len(grp)
    out = np.empty((NSIM, 3))
    for s in range(NSIM):
        t = rng.normal(0, np.sqrt(p["vt"]), N)
        g0 = t + rng.normal(0, np.sqrt(p["ve"]), N); g1 = t + rng.normal(0, np.sqrt(p["ve"]), N)
        v = rng.normal(0, np.sqrt(p["vv"]), N)
        m0 = p["b"] * t + v + rng.normal(0, np.sqrt(p["vu"]), N); m1 = p["b"] * t + v + rng.normal(0, np.sqrt(p["vu"]), N)
        D = p["c"] * t + rng.normal(0, np.sqrt(p["vw"]), N)
        rm = xfit(g0, g1, m0, m1, grp); rd = xfit(g0, g1, D, D, grp)
        out[s] = (rm, rd, rm - rd)
    return out


report = {}
for conf, metrics in A.DEPENDENCE_METRICS.items():
    R, ng = build(conf, metrics)
    rel_g = cov(R["g_half0"], R["g_half1"]) / np.sqrt(np.nanvar(R["g_half0"]) * np.nanvar(R["g_half1"]))
    report[conf] = {"groups": ng, "truth_resid_half_r": rel_g, "metrics": {}}
    for n in metrics:
        p = params(R, n)
        rows = {}
        for k in KS:
            sim = simulate(p, ng, k)
            diff = sim[:, 2]; sd = diff.std()
            rows[k] = {"units": ng * k, "metric_r": float(sim[:, 0].mean()), "D_r": float(sim[:, 1].mean()),
                       "diff_mean": float(diff.mean()), "diff_ci_halfwidth": float(1.96 * sd),
                       "power_diff_gt0": float(np.mean(diff > 1.96 * sd)),
                       "metric_r_ci_halfwidth": float(1.96 * sim[:, 0].std())}
        report[conf]["metrics"][n] = {"params": {k2: float(v) for k2, v in p.items()}, "by_k": rows}
        print(conf, n, {k: (round(v["diff_mean"], 2), round(v["diff_ci_halfwidth"], 2), round(v["power_diff_gt0"], 2)) for k, v in rows.items()})

Path("results/analysis").mkdir(parents=True, exist_ok=True)
json.dump(report, open("results/analysis/power_within_cell.json", "w"), indent=1)
