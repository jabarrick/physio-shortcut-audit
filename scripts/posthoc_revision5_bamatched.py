"""Post hoc (exploratory, after the second critical review, 30 Sep 2026): the model's own matched test accuracy
BA(matched) as a second within-cell baseline, alongside the probe score D.

For the original units (results/units; p = 0.9, factorial + p-series cells, as posthoc_revision2_diff.py) and the
Amendment 1 units (results/units_a1; saccade, p = 0.9), per confound and metric:
  * cross-fitted within-cell partial r with ground truth (cell x model means removed; metric/BA(matched) of one half
    against ground truth of the other half, both directions averaged) -- as in the registered A1-H1 statistic;
  * difference metric - BA(matched), cluster bootstrap over cell x model groups (95% and 98.33% intervals);
  * partial r of each metric with ground truth after controlling for BA(matched), cross-fitted: ground truth of half h
    is residualised on BA(matched) of the other half, the metric of the other half on BA(matched) of half h;
  * split-half reliability of the within-cell residuals of ground truth, BA(matched) and each metric.
Writes results/analysis/posthoc_revision5_bamatched.json.
"""
import json, sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore"); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p3audit.experiments import analysis as A

N_BOOT, SEED = 4000, 20261005

def corr(a, b):
    k = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[k], b[k])[0, 1]) if k.sum() > 3 else np.nan

def partial_out(x, c):
    k = np.isfinite(x) & np.isfinite(c); out = np.full_like(x, np.nan)
    C = np.c_[np.ones(k.sum()), c[k]]; beta = np.linalg.lstsq(C, x[k], rcond=None)[0]; out[k] = x[k] - C @ beta
    return out

def build(units, conf, metrics, D_amp="cell"):
    subs = A._test_subjects(units)
    keys = sorted({(r["model"], r["cid"]) for r in units}); gi = {k: i for i, k in enumerate(keys)}
    g = np.array([gi[(r["model"], r["cid"])] for r in units])
    X = {}
    for h in ("0", "1"):
        X["t" + h] = [A.truth(r, subs, "half" + h) for r in units]
        X["bam" + h] = [A._ba(r["truth_counts"]["matched"]["half" + h], subs) for r in units]
        for n in metrics:
            X[n + h] = [A.DEPENDENCE_METRICS[conf][n](r, subs, "half" + h) for r in units]
    X["D"] = [r["probe"][f"{float(r['cell']['amp']):g}"]["D"] for r in units]
    for a in ("0.125", "0.25"):
        X["D" + a] = [r["probe"].get(a, {}).get("D", np.nan) for r in units]
    return {k: np.array(v, float) for k, v in X.items()}, [np.where(g == i)[0] for i in range(len(keys))]

def stat(X, groups, metrics):
    idx = np.concatenate(groups); lab = np.concatenate([np.full(len(m), j) for j, m in enumerate(groups)])
    def res(v):
        v = v[idx]; ok = np.isfinite(v)
        s = np.bincount(lab[ok], v[ok], minlength=len(groups)); n = np.bincount(lab[ok], minlength=len(groups))
        mu = s / np.maximum(n, 1); return v - mu[lab]
    R = {k: res(v) for k, v in X.items()}
    xf = lambda a: np.nanmean([corr(R["t0"], R[a + "1"]), corr(R["t1"], R[a + "0"])])
    o = {"BA_matched": xf("bam"), "D": np.nanmean([corr(R["t0"], R["D"]), corr(R["t1"], R["D"])])}
    for a in ("0.125", "0.25"):
        if np.isfinite(R["D" + a]).sum() > 3: o["D" + a] = np.nanmean([corr(R["t0"], R["D" + a]), corr(R["t1"], R["D" + a])])
    for n in metrics:
        o[n] = xf(n)
        o[n + "|BAm"] = np.nanmean([corr(partial_out(R["t0"], R["bam1"]), partial_out(R[n + "1"], R["bam0"])),
                                    corr(partial_out(R["t1"], R["bam0"]), partial_out(R[n + "0"], R["bam1"]))])
    o["D|BAm"] = np.nanmean([corr(partial_out(R["t0"], R["bam1"]), partial_out(R["D"], R["bam0"])),
                             corr(partial_out(R["t1"], R["bam0"]), partial_out(R["D"], R["bam1"]))])
    rel = {"truth": corr(R["t0"], R["t1"]), "BA_matched": corr(R["bam0"], R["bam1"])}
    rel.update({n: corr(R[n + "0"], R[n + "1"]) for n in metrics})
    return o, rel

def run(units, conf, metrics):
    X, groups = build(units, conf, metrics)
    est, rel = stat(X, groups, metrics)
    rng = np.random.default_rng(SEED); B = []
    for _ in range(N_BOOT):
        B.append(stat(X, [groups[k] for k in rng.integers(0, len(groups), len(groups))], metrics)[0])
    def ci(vals, a):
        v = np.array(vals); v = v[np.isfinite(v)]; return [float(np.quantile(v, a / 2)), float(np.quantile(v, 1 - a / 2))]
    out = {"n_units": len(units), "n_groups": len(groups), "estimates": {k: float(v) for k, v in est.items()},
           "reliability_resid_half_r": {k: float(v) for k, v in rel.items()}, "differences": {}, "controlled_ci95": {}}
    for n in metrics:
        dif = [b[n] - b["BA_matched"] for b in B]
        out["differences"][f"{n}-BA_matched"] = {"est": float(est[n] - est["BA_matched"]), "ci95": ci(dif, .05), "ci98.33": ci(dif, .05 / 3)}
        out["controlled_ci95"][n + "|BAm"] = ci([b[n + "|BAm"] for b in B], .05)
    out["differences"]["BA_matched-D"] = {"est": float(est["BA_matched"] - est["D"]), "ci95": ci([b["BA_matched"] - b["D"] for b in B], .05)}
    out["controlled_ci95"]["D|BAm"] = ci([b["D|BAm"] for b in B], .05)
    out["BA_matched_ci95"] = ci([b["BA_matched"] for b in B], .05)
    return out

if __name__ == "__main__":
    res = {}
    orig = [json.load(open(p, encoding="utf-8")) for p in sorted(Path("results/units").glob("*.json"))]
    for conf in ("alpha", "saccade"):
        us = [r for r in orig if r["cell"]["confound"] == conf and r["cell"]["p"] == 0.9 and r["cell"]["module"] in ("factorial", "p_series")]
        res[f"original_{conf}"] = run(us, conf, A.CONFIRMATORY[conf])
    a1 = [json.load(open(p, encoding="utf-8")) for p in sorted(Path("results/units_a1").glob("*.json"))]
    a1 = [r for r in a1 if r["cell"]["confound"] == "saccade" and r["cell"]["p"] == 0.9]
    res["amendment1_saccade"] = run(a1, "saccade", ["heog_reg", "ig_share", "erasure_keep"])
    Path("results/analysis").mkdir(parents=True, exist_ok=True)
    json.dump(res, open("results/analysis/posthoc_revision5_bamatched.json", "w"), indent=1)
    for k, v in res.items():
        print(k, v["n_units"], {a: round(b, 3) for a, b in v["estimates"].items()})
