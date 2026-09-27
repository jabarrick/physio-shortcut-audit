"""Amendment 1 (OSF): confirmatory within-cell test.  Outside the frozen package.
Data: results/units_a1 only (new seeds 10-15; the original units are not used).
For each confirmatory saccade metric (heog_reg, ig_share, erasure_keep) and the probe score D:
  residuals = values minus their cell x model mean; cross-fitted partial correlation with ground truth
  (truth on one half of the test windows, metric on the other, averaged over both assignments), as in
  scripts/posthoc_revision2.py.  D has no halves and is correlated with each truth half.
Statistic: difference metric - D.  Interval: cluster bootstrap over cell x model groups (4000), paired.
Decision (per metric): the lower bound of the Bonferroni-adjusted (3 tests) two-sided 98.33% interval > 0.
Writes results/analysis/amendment1_within_cell.json.
Usage (repo root): python scripts/amendment1_analyse.py [--units results/units_a1] [--confound saccade]
"""
import argparse, json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); warnings.filterwarnings("ignore")
from p3audit.experiments import analysis as A

METRICS = {"saccade": ["heog_reg", "ig_share", "erasure_keep"], "alpha": ["erasure_keep", "sri_post", "ig_share"]}
N_BOOT, ALPHA = 4000, 0.05 / 3


def table(units, conf):
    us = [r for r in units if r["cell"]["confound"] == conf and r["cell"]["p"] == 0.9]
    subs = A._test_subjects(us); fns = A.DEPENDENCE_METRICS[conf]
    rows = []
    for r in us:
        a = float(r["cell"]["amp"])
        row = {"model": r["model"], "cid": r["cid"], "seed": r["seed"], "D": r["probe"][f"{a:g}"]["D"]}
        for h in ("half0", "half1"):
            row[f"truth_{h}"] = A.truth(r, subs, h)
            for n in METRICS[conf]:
                row[f"{n}_{h}"] = fns[n](r, subs, h)
        rows.append(row)
    return pd.DataFrame(rows)


def corr(a, b):
    k = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[k], b[k])[0, 1]) if k.sum() > 3 else np.nan


def xfit(d, n):
    g = d.groupby(["model", "cid"])
    R = lambda c: (d[c] - g[c].transform("mean")).values
    t0, t1 = R("truth_half0"), R("truth_half1")
    if n == "D":
        x = R("D"); return float(np.nanmean([corr(t0, x), corr(t1, x)]))
    return float(np.nanmean([corr(t0, R(f"{n}_half1")), corr(t1, R(f"{n}_half0"))]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--units", default="results/units_a1"); ap.add_argument("--confound", default="saccade")
    a = ap.parse_args()
    units = [json.load(open(p, encoding="utf-8")) for p in sorted(Path(a.units).glob("*.json"))]
    d = table(units, a.confound)
    grp = d.groupby(["model", "cid"]).indices; keys = list(grp); rng = np.random.default_rng(20260928)
    est = {n: xfit(d, n) for n in ["D"] + METRICS[a.confound]}
    B = []
    for _ in range(N_BOOT):
        parts = []
        for j, k in enumerate(rng.integers(0, len(keys), len(keys))):
            sub = d.iloc[grp[keys[k]]].copy(); sub["cid"] = sub["cid"] + f"#{j}"; parts.append(sub)
        dd = pd.concat(parts, ignore_index=True)
        v = {n: xfit(dd, n) for n in est}; B.append({n: v[n] - v["D"] for n in METRICS[a.confound]})
    B = pd.DataFrame(B)
    out = {"n_units": len(d), "n_groups": len(keys), "seeds": sorted(d.seed.unique().tolist()), "partial_r": est, "tests": {}}
    for n in METRICS[a.confound]:
        lo, hi = B[n].quantile([ALPHA / 2, 1 - ALPHA / 2])
        out["tests"][n] = {"diff": est[n] - est["D"], "ci_98.33": [float(lo), float(hi)], "pass": bool(lo > 0)}
    Path("results/analysis").mkdir(parents=True, exist_ok=True)
    json.dump(out, open(f"results/analysis/amendment1_within_cell_{a.confound}.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
