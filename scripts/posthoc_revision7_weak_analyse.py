"""Post hoc (exploratory): H3-style test with weak confounds (results/units_weak; 0.125 and 0.25 a*, p = 0.9,
three task strengths, four core models, seeds 20 and 21).  Per confound:
  * mean ground truth (Delta BA_neu) and probe score D per amplitude;
  * cross-fitted correlation with ground truth across units (values standardised within model, pooled; truth of one
    half of the test windows against the metric of the other half, both directions averaged) for the confirmatory
    metrics, for the model's matched accuracy BA(matched), and (not cross-fitted) for the injected amplitude and D;
  * 95% intervals from a bootstrap over test subjects with seeds fixed (2000 resamples, seed 20261009), which ignores
    seed variance.
Writes results/analysis/posthoc_revision7_weak.json.
"""
import json, sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore"); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p3audit.experiments import analysis as A

N_BOOT, SEED = 2000, 20261009
units = [json.load(open(p, encoding="utf-8")) for p in sorted(Path("results/units_weak").glob("*.json"))]


def corr(a, b):
    k = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[k], b[k])[0, 1]) if k.sum() > 3 and a[k].std() > 0 and b[k].std() > 0 else np.nan


def zwithin(v, mod):
    out = np.full(len(v), np.nan)
    for m in np.unique(mod):
        k = (mod == m) & np.isfinite(v)
        if k.sum() > 1 and v[k].std() > 0: out[k] = (v[k] - v[k].mean()) / v[k].std()
    return out


def stats(us, conf, subs):
    mod = np.array([r["model"] for r in us]); o = {}
    t = {h: np.array([A.truth(r, subs, h) for r in us]) for h in ("half0", "half1")}
    xf = lambda f: float(np.nanmean([corr(zwithin(t["half0"], mod), zwithin(np.array([f(r, "half1") for r in us]), mod)),
                                     corr(zwithin(t["half1"], mod), zwithin(np.array([f(r, "half0") for r in us]), mod))]))
    for n in A.CONFIRMATORY[conf]:
        o[n] = xf(lambda r, h, n=n: A.DEPENDENCE_METRICS[conf][n](r, subs, h))
    o["BA_matched"] = xf(lambda r, h: A._ba(r["truth_counts"]["matched"][h], subs))
    tall = zwithin(np.array([A.truth(r, subs, "all") for r in us]), mod)
    o["amplitude"] = corr(tall, zwithin(np.array([float(r["cell"]["amp"]) for r in us]), mod))
    o["D"] = corr(tall, zwithin(np.array([r["probe"][f"{float(r['cell']['amp']):g}"]["D"] for r in us]), mod))
    return o


out = {"n_units": len(units)}
for conf in ("alpha", "saccade"):
    us = [r for r in units if r["cell"]["confound"] == conf]
    if not us:
        continue
    subs = A._test_subjects(us); rng = np.random.default_rng(SEED)
    est = stats(us, conf, subs)
    B = [stats(us, conf, list(rng.choice(subs, len(subs)))) for _ in range(N_BOOT)]
    res = {"n_units": len(us), "n_subjects": len(subs), "by_amp": {}}
    for a in sorted({float(r["cell"]["amp"]) for r in us}):
        ua = [r for r in us if float(r["cell"]["amp"]) == a]
        res["by_amp"][f"{a:g}"] = {"truth_mean": float(np.mean([A.truth(r, subs, "all") for r in ua])),
                                   "D_mean": float(np.mean([r["probe"][f"{a:g}"]["D"] for r in ua])),
                                   "BA_matched_mean": float(np.mean([A._ba(r["truth_counts"]["matched"]["all"], subs) for r in ua]))}
    res["r_with_truth"] = {}
    for k, v in est.items():
        b = np.array([x[k] for x in B], float); b = b[np.isfinite(b)]
        res["r_with_truth"][k] = {"est": v, "ci95": [float(np.quantile(b, .025)), float(np.quantile(b, .975))] if len(b) else None}
    out[conf] = res
    print(conf, len(us), res["by_amp"]); [print("   ", k, round(v["est"], 3), [round(x, 2) for x in v["ci95"]] if v["ci95"] else None) for k, v in res["r_with_truth"].items()]
Path("results/analysis").mkdir(parents=True, exist_ok=True)
json.dump(out, open("results/analysis/posthoc_revision7_weak.json", "w"), indent=1)
