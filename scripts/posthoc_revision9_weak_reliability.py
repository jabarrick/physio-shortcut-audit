"""Post hoc (exploratory; third review, point 6): reliability and bootstrap checks for the weak-confound units
(results/units_weak; see posthoc_revision7_weak_analyse.py, whose statistic is reused unchanged).
Per confound:
  * split-half reliability across units (values standardised within model; half0 vs half1 of the test windows,
    Spearman-Brown corrected) of ground truth, of each confirmatory metric and of BA(matched), and the resulting
    ceiling sqrt(rel_truth * rel_metric) for the cross-fitted correlation;
  * the same reliabilities separately at each confound amplitude;
  * intervals for the correlations with ground truth from (a) the subject bootstrap with seeds fixed (as in
    revision 7) and (b) the registered two-stage scheme: both seeds redrawn with replacement, then the 20 test
    subjects (2000 resamples each, seed 20261018); for each, the mean bootstrap bias and a bias-corrected
    percentile (BC) interval.
Writes results/analysis/posthoc_revision9_weak_reliability.json.
"""
import json, sys, warnings
from pathlib import Path
import numpy as np
from scipy.stats import norm
warnings.filterwarnings("ignore"); ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from p3audit.experiments import analysis as A

import os
N_BOOT, SEED = int(os.environ.get("P9_NBOOT", 2000)), 20261018
units = [json.load(open(p, encoding="utf-8")) for p in sorted((ROOT / "results/units_weak").glob("*.json"))]


def corr(a, b):
    k = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[k], b[k])[0, 1]) if k.sum() > 3 and a[k].std() > 0 and b[k].std() > 0 else np.nan


def zwithin(v, mod):
    out = np.full(len(v), np.nan)
    for m in np.unique(mod):
        k = (mod == m) & np.isfinite(v)
        if k.sum() > 1 and v[k].std() > 0: out[k] = (v[k] - v[k].mean()) / v[k].std()
    return out


def funcs(conf):
    f = {"truth": lambda r, s, h: A.truth(r, s, h)}
    for n in A.CONFIRMATORY[conf]:
        f[n] = lambda r, s, h, n=n: A.DEPENDENCE_METRICS[conf][n](r, s, h)
    f["BA_matched"] = lambda r, s, h: A._ba(r["truth_counts"]["matched"][h], s)
    return f


def stats(us, conf, subs):  # identical statistic to posthoc_revision7_weak_analyse.py
    mod = np.array([r["model"] for r in us]); o = {}; F = funcs(conf)
    t = {h: np.array([F["truth"](r, subs, h) for r in us]) for h in ("half0", "half1")}
    for n in [k for k in F if k != "truth"]:
        m = {h: np.array([F[n](r, subs, h) for r in us]) for h in ("half0", "half1")}
        o[n] = float(np.nanmean([corr(zwithin(t["half0"], mod), zwithin(m["half1"], mod)),
                                 corr(zwithin(t["half1"], mod), zwithin(m["half0"], mod))]))
    tall = zwithin(np.array([A.truth(r, subs, "all") for r in us]), mod)
    o["amplitude"] = corr(tall, zwithin(np.array([float(r["cell"]["amp"]) for r in us]), mod))
    o["D"] = corr(tall, zwithin(np.array([r["probe"][f"{float(r['cell']['amp']):g}"]["D"] for r in us]), mod))
    return o


def rel(us, conf, subs):
    mod = np.array([r["model"] for r in us]); out = {}
    for n, f in funcs(conf).items():
        r = corr(zwithin(np.array([f(u, subs, "half0") for u in us]), mod), zwithin(np.array([f(u, subs, "half1") for u in us]), mod))
        out[n] = {"r_half": r, "spearman_brown": float(2 * r / (1 + r)) if np.isfinite(r) and r > -1 else np.nan}
    return out


def summarise(est, B):
    res = {}
    for k, v in est.items():
        b = np.array([x[k] for x in B], float); b = b[np.isfinite(b)]
        p0 = np.clip(np.mean(b < v), 1 / len(b), 1 - 1 / len(b)); z0 = norm.ppf(p0)
        lo, hi = norm.cdf(2 * z0 + norm.ppf(.025)), norm.cdf(2 * z0 + norm.ppf(.975))
        res[k] = {"est": v, "n_valid": int(len(b)), "boot_mean": float(b.mean()), "bias": float(b.mean() - v),
                  "share_below_est": float(np.mean(b < v)),
                  "ci95_percentile": [float(np.quantile(b, .025)), float(np.quantile(b, .975))],
                  "ci95_bc": [float(np.quantile(b, lo)), float(np.quantile(b, hi))]}
    return res


def _boot(job):
    conf, ds, sb = job
    us = [r for r in units if r["cell"]["confound"] == conf]
    if ds is not None:
        us = [r for s in ds for r in us if r["seed"] == s]
    return stats(us, conf, sb)


if __name__ != "__main__":
    raise SystemExit
from multiprocessing import Pool
out = {"n_units": len(units), "n_boot": N_BOOT, "seed": SEED}
CONFS = os.environ.get("P9_CONF", "alpha,saccade").split(",")  # each confound can be run separately
for conf in CONFS:
    us = [r for r in units if r["cell"]["confound"] == conf]
    subs = A._test_subjects(us); seeds = sorted({r["seed"] for r in us})
    est = stats(us, conf, subs)
    rng = np.random.default_rng(SEED)
    jobs = [(conf, None, list(rng.choice(subs, len(subs)))) for _ in range(N_BOOT)]
    rng = np.random.default_rng(SEED + 1)
    for _ in range(N_BOOT):
        ds = [int(x) for x in rng.choice(seeds, len(seeds))]
        jobs.append((conf, ds, list(rng.choice(subs, len(subs)))))
    with Pool() as pool:
        res_all = pool.map(_boot, jobs, chunksize=50)
    B_subj, B_two = res_all[:N_BOOT], res_all[N_BOOT:]
    R = rel(us, conf, subs)
    ceil = {n: float(np.sqrt(R["truth"]["spearman_brown"] * R[n]["spearman_brown"]))
            if R["truth"]["spearman_brown"] > 0 and R[n]["spearman_brown"] > 0 else np.nan for n in R if n != "truth"}
    by_amp = {}
    for a in sorted({float(r["cell"]["amp"]) for r in us}):
        by_amp[f"{a:g}"] = rel([r for r in us if float(r["cell"]["amp"]) == a], conf, subs)
    out[conf] = {"n_units": len(us), "seeds": seeds, "reliability": R, "ceiling": ceil, "reliability_by_amp": by_amp,
                 "subject_only": summarise(est, B_subj), "seeds_then_subjects": summarise(est, B_two)}
    print("==", conf)
    print(" reliability:", {k: round(v["spearman_brown"], 2) for k, v in R.items()})
    for a, d in by_amp.items(): print("  amp", a, {k: round(v["spearman_brown"], 2) for k, v in d.items()})
    print(" ceiling:", {k: round(v, 2) for k, v in ceil.items()})
    for lab, S in (("subject_only", out[conf]["subject_only"]), ("seeds_then_subjects", out[conf]["seeds_then_subjects"])):
        print(" ", lab)
        for k, v in S.items():
            print(f"    {k:13s} {v['est']:.2f} pct[{v['ci95_percentile'][0]:.2f},{v['ci95_percentile'][1]:.2f}] "
                  f"bc[{v['ci95_bc'][0]:.2f},{v['ci95_bc'][1]:.2f}] bias {v['bias']:+.3f} below {v['share_below_est']:.2f}")
OUT = ROOT / "results/analysis"
for conf in CONFS:
    json.dump({k: out[k] for k in ("n_units", "n_boot", "seed", conf)}, open(OUT / f"posthoc_revision9_weak_reliability_{conf}.json", "w"), indent=1)
parts = {c: OUT / f"posthoc_revision9_weak_reliability_{c}.json" for c in ("alpha", "saccade")}
if all(p.exists() for p in parts.values()):
    merged = {}
    for c, p in parts.items():
        merged.update(json.load(open(p)))
    json.dump(merged, open(OUT / "posthoc_revision9_weak_reliability.json", "w"), indent=1)
