"""Round-2 post hoc analyses (exploratory). Run from repo root: python scripts/posthoc_revision2.py
1. Cross-fitted partial correlations (truth and metric on different halves of the test windows) after
   removing cell x model means; cluster bootstrap over cell x model groups.
2. Reliability of the residual (within-cell) ground truth and metrics -> attenuation ceiling.
3. Shared-noise check: ΔBA_neu, ΔBA_keep and ΔBA_reg all contain BA(matched); correlation of residuals
   computed on the same half versus across halves.
"""
import sys, json, warnings, numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
from p3audit.experiments import analysis as A

units = [json.load(open(p, encoding="utf-8")) for p in sorted(Path("results/units").glob("*.json"))]


def ba_matched(r, subs, part):
    return A._ba(r["truth_counts"]["matched"][part], subs)


def resid(d, col):
    return d[col] - d.groupby(["model", "cid"])[col].transform("mean")


def corr(a, b):
    k = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[k], b[k])[0, 1]) if k.sum() > 3 else np.nan


out = {}
for conf, metrics in A.DEPENDENCE_METRICS.items():
    us = [r for r in units if r["cell"]["confound"] == conf and r["cell"]["p"] == 0.9
          and r["cell"]["module"] in ("factorial", "p_series")]
    subs = A._test_subjects(us)
    rows = []
    for r in us:
        a = float(r["cell"]["amp"])
        row = {"model": r["model"], "cid": r["cid"], "seed": r["seed"], "D": r["probe"][f"{a:g}"]["D"]}
        for h in ("all", "half0", "half1"):
            row[f"truth_{h}"] = A.truth(r, subs, h)
            row[f"bam_{h}"] = ba_matched(r, subs, h)
            for n, fn in metrics.items():
                row[f"{n}_{h}"] = fn(r, subs, h)
        rows.append(row)
    d = pd.DataFrame(rows)
    g = d.groupby(["model", "cid"]).size()
    res = {"n_units": len(d), "n_groups": int(len(g)), "n_groups_ge2": int((g >= 2).sum()),
           "residual_df": int(len(d) - len(g))}
    R = {c: resid(d, c).values for c in d.columns if c not in ("model", "cid", "seed")}
    # reliability of residual truth (half vs half) and Spearman-Brown to full length
    rt = corr(R["truth_half0"], R["truth_half1"])
    res["truth_resid_half_r"] = rt
    res["truth_resid_rel_full"] = 2 * rt / (1 + rt) if np.isfinite(rt) else np.nan
    res["bam_vs_truth_same_half"] = float(np.mean([corr(R["bam_half0"], R["truth_half0"]), corr(R["bam_half1"], R["truth_half1"])]))
    res["bam_vs_truth_cross_half"] = float(np.mean([corr(R["bam_half0"], R["truth_half1"]), corr(R["bam_half1"], R["truth_half0"])]))

    groups = d.groupby(["model", "cid"]).indices
    keys = list(groups)
    rng = np.random.default_rng(7)

    def stats_for(idx):
        dd = d.iloc[idx].copy()
        dd["cid"] = dd["cid"] + "_" + pd.Series(range(len(dd)), index=dd.index).astype(str).str[:0]  # keep
        return dd

    def xfit_partial(dd, n):
        Rr = {c: resid(dd, c).values for c in [f"truth_half0", "truth_half1", f"{n}_half0", f"{n}_half1"]}
        return float(np.nanmean([corr(Rr["truth_half0"], Rr[f"{n}_half1"]), corr(Rr["truth_half1"], Rr[f"{n}_half0"])]))

    def xfit_partial_D(dd):
        Rr = {c: resid(dd, c).values for c in ["truth_half0", "truth_half1", "D"]}
        return float(np.nanmean([corr(Rr["truth_half0"], Rr["D"]), corr(Rr["truth_half1"], Rr["D"])]))

    def boot(fn):
        vals = []
        for _ in range(2000):
            pick = rng.integers(0, len(keys), len(keys))
            parts = []
            for j, k in enumerate(pick):
                sub = d.iloc[groups[keys[k]]].copy(); sub["cid"] = sub["cid"] + f"#{j}"; parts.append(sub)
            dd = pd.concat(parts, ignore_index=True)
            v = fn(dd)
            if np.isfinite(v):
                vals.append(v)
        return [float(np.quantile(vals, .025)), float(np.quantile(vals, .975))]

    for n in ["D"] + list(metrics):
        e = {}
        if n == "D":
            e["partial_same"] = corr(R["truth_all"], R["D"])
            e["partial_xfit"] = xfit_partial_D(d)
            e["ci_xfit"] = boot(xfit_partial_D)
        else:
            e["partial_same"] = corr(R["truth_all"], R[f"{n}_all"])
            e["partial_xfit"] = xfit_partial(d, n)
            e["ci_xfit"] = boot(lambda dd, n=n: xfit_partial(dd, n))
            rm = corr(R[f"{n}_half0"], R[f"{n}_half1"])
            e["metric_resid_half_r"] = rm
            if np.isfinite(rm) and rm > 0 and rt > 0:
                e["ceiling_xfit"] = float(np.sqrt(rt * rm))
                e["disattenuated"] = float(e["partial_xfit"] / np.sqrt(rt * rm))
            # the same-half partial (as reported in revision 1) vs cross-fitted
            e["partial_same_half_mean"] = float(np.mean([corr(R["truth_half0"], R[f"{n}_half0"]), corr(R["truth_half1"], R[f"{n}_half1"])]))
            # H3 cross-fitted (non-partial) re-computed independently for verification
            e["h3_xfit_check"] = A.cross_fitted_corr(us, metrics[n], subs)
        res[n] = e
    out[conf] = res

# HEOG regression calibration: matched vs mismatched templates; OLS and contaminated regressors
cal = {}
for r in units:
    if r["cell"]["confound"] != "saccade" or r["cell"]["p"] != 0.9 or "heog" not in r:
        continue
    kind = "mismatch" if "sacmis" in r["cid"] else ("matched" if r["cell"]["s"] == "s_star" and float(r["cell"]["amp"]) == 1.0 and r["cell"]["module"] in ("factorial", "p_series") else None)
    if kind is None:
        continue
    t = r["truth"]["delta_ba_neu"]
    if t <= 0.05:
        continue
    for c in ("0", "0.1", "0.25", "ols"):
        if c in r["heog"]:
            cal.setdefault(kind, {}).setdefault(c, []).append(r["heog"][c]["delta_ba_reg"] / t)
out["heog_ratio"] = {k: {c: {"median": float(np.median(v)), "n": len(v)} for c, v in dv.items()} for k, dv in cal.items()}
json.dump(out, open("results/analysis/posthoc_revision2.json", "w"), indent=1)
print(json.dumps(out, indent=1))
