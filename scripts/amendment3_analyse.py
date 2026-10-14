"""Amendment 3 (OSF): registered analysis of the real-saccade units.  Outside the frozen package.
Data: results/units_a3_<ARM> (ARM = RH, primary; RV, secondary), 216 units each (saccade arm, p = 0.9 factorial cells,
four core models, seeds 30-35).  Ground truth: Delta BA_neu as registered; halves: the registered window-parity halves,
in which the coupling is now exact (z assigned within subject x label x half).
Within-cell statistic as in Amendment 1: residuals from the cell x model means; cross-fitted partial correlation with
ground truth (ground truth of one half against the quantity of the other half, both directions averaged); D (no halves)
is correlated with each ground-truth half.  Quantities: probe score D at the cell amplitude, D(0.125 a*), D(0.25 a*),
BA(matched) and the metrics HEOG regression, IG share and residual erasure (frozen definitions,
p3audit.experiments.analysis.DEPENDENCE_METRICS['saccade']).  Partial r given BA(matched): ground truth of half h
residualised on BA(matched) of the other half, the metric of the other half on BA(matched) of half h (as in the
exploratory analysis of the manuscript).  Cluster bootstrap over cell x model groups, N_BOOT resamples, seed SEED.
Decisions (per metric m; Bonferroni over the three metrics: two-sided 98.33% intervals):
  A3-H1  m - D > 0                      lower bound > 0 -> supported, otherwise inconclusive
  A3-H2  m - BA(matched) > 0            lower bound > 0 -> supported, otherwise inconclusive
  A3-H3  partial r(m | BA(matched)) > 0 lower bound > 0 -> supported, otherwise inconclusive
  Reliability: if the split-half r of the residual ground truth (point estimate, no Spearman-Brown) is below 0.5,
  every decision of the arm is 'undecidable'.
Descriptive: split-half r of each quantity; D(0.125 a*), D(0.25 a*) differences; the advantage of HEOG regression
over BA(matched) in this arm minus that in the Amendment 1 units (results/units_a1; independent bootstraps paired by
draw index); calibration transfer: per model, OLS of Delta BA_neu on the metric over the Amendment 1 units (all test
windows), applied to this arm's units, mean bias with a cluster-bootstrap interval; mean ground truth per cell.
Writes results/analysis/amendment3_<ARM>.json.
Usage (repo root): python scripts/amendment3_analyse.py --arm RH [--units DIR] [--n-boot 4000]
"""
import argparse, json, sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore"); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
METRICS = ["heog_reg", "ig_share", "erasure_keep"]; PROBES = ["D", "D0.125", "D0.25"]
N_BOOT, SEED, ALPHA = 4000, 20261014, 0.05 / 3


def corr(a, b):
    k = np.isfinite(a) & np.isfinite(b)
    if k.sum() <= 3:
        return np.nan
    a = a[k] - a[k].mean(); b = b[k] - b[k].mean(); return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def partial_out(x, c):
    k = np.isfinite(x) & np.isfinite(c); out = np.full_like(x, np.nan)
    cc = c[k] - c[k].mean(); xc = x[k] - x[k].mean(); out[k] = xc - cc * (cc @ xc / (cc @ cc)); return out


def build(units):
    from p3audit.experiments import analysis as A
    subs = A._test_subjects(units); fns = A.DEPENDENCE_METRICS["saccade"]
    keys = sorted({(r["model"], r["cid"]) for r in units}); gi = {k: i for i, k in enumerate(keys)}
    g = np.array([gi[(r["model"], r["cid"])] for r in units]); X = {}
    for h in ("0", "1"):
        X["t" + h] = [A.truth(r, subs, "half" + h) for r in units]
        X["bam" + h] = [A._ba(r["truth_counts"]["matched"]["half" + h], subs) for r in units]
        for m in METRICS:
            X[m + h] = [fns[m](r, subs, "half" + h) for r in units]
    X["D"] = [r["probe"][f"{float(r['cell']['amp']):g}"]["D"] for r in units]
    for a in ("0.125", "0.25"):
        X["D" + a] = [r["probe"].get(a, {}).get("D", np.nan) for r in units]
    X["t_all"] = [A.truth(r, subs, "all") for r in units]
    for m in METRICS:
        X[m + "_all"] = [fns[m](r, subs, "all") for r in units]
    return {k: np.array(v, float) for k, v in X.items()}, [np.where(g == i)[0] for i in range(len(keys))], [r["model"] for r in units]


def stat(X, groups):
    idx = np.concatenate(groups); lab = np.concatenate([np.full(len(m), j) for j, m in enumerate(groups)])
    def res(v):
        v = v[idx]; ok = np.isfinite(v); s = np.bincount(lab[ok], v[ok], minlength=len(groups)); n = np.bincount(lab[ok], minlength=len(groups))
        return v - (s / np.maximum(n, 1))[lab]
    R = {k: res(v) for k, v in X.items() if not k.endswith("_all") and k != "t_all"}; t0, t1 = R["t0"], R["t1"]
    xf = lambda a: np.nanmean([corr(t0, R[a + "1"]), corr(t1, R[a + "0"])])
    o = {"rel_truth": corr(t0, t1), "BA_matched": xf("bam")}
    for d in PROBES:
        o[d] = np.nanmean([corr(t0, R[d]), corr(t1, R[d])])
    for m in METRICS:
        o[m] = xf(m); o[f"{m}-BA_matched"] = o[m] - o["BA_matched"]
        for d in PROBES:
            o[f"{m}-{d}"] = o[m] - o[d]
        o[f"{m}|BA_matched"] = np.nanmean([corr(partial_out(t0, R["bam1"]), partial_out(R[m + "1"], R["bam0"])),
                                           corr(partial_out(t1, R["bam0"]), partial_out(R[m + "0"], R["bam1"]))])
        o[f"rel_{m}"] = corr(R[m + "0"], R[m + "1"])
    o["rel_BA_matched"] = corr(R["bam0"], R["bam1"]); return o


def load(d):
    return [json.load(open(p, encoding="utf-8")) for p in sorted(Path(d).glob("*.json"))]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arm", choices=["RH", "RV"], required=True); ap.add_argument("--units", default=None)
    ap.add_argument("--a1", default="results/units_a1"); ap.add_argument("--n-boot", type=int, default=N_BOOT); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    units = [r for r in load(a.units or f"results/units_a3_{a.arm}") if r["cell"]["confound"] == "saccade" and r["cell"]["p"] == 0.9]
    a1 = [r for r in load(a.a1) if r["cell"]["confound"] == "saccade" and r["cell"]["p"] == 0.9]
    X, G, models = build(units); X1, G1, models1 = build(a1)
    est = stat(X, G); est1 = stat(X1, G1)
    rng = np.random.default_rng(SEED); rng1 = np.random.default_rng(SEED + 1); B, B1 = [], []
    for _ in range(a.n_boot):
        B.append(stat(X, [G[k] for k in rng.integers(0, len(G), len(G))])); B1.append(stat(X1, [G1[k] for k in rng1.integers(0, len(G1), len(G1))]))
    def ci(vals, alpha):
        v = np.array(vals, float); v = v[np.isfinite(v)]; return [float(np.quantile(v, alpha / 2)), float(np.quantile(v, 1 - alpha / 2))]
    out = {"arm": a.arm, "role": "primary" if a.arm == "RH" else "secondary", "n_units": len(units), "n_groups": len(G), "n_boot": a.n_boot, "seed": SEED,
           "estimates": {k: float(v) for k, v in est.items()}, "ci95": {k: ci([b[k] for b in B], 0.05) for k in est}, "decisions": {}}
    gate = est["rel_truth"] >= 0.5; out["reliability_gate_passed"] = bool(gate)
    for m in METRICS:
        dec = {}
        for test, key in (("A3-H1", f"{m}-D"), ("A3-H2", f"{m}-BA_matched"), ("A3-H3", f"{m}|BA_matched")):
            lo, hi = ci([b[key] for b in B], ALPHA)
            dec[test] = {"estimate": float(est[key]), "ci98.33": [lo, hi], "decision": ("supported" if lo > 0 else "inconclusive") if gate else "undecidable"}
        out["decisions"][m] = dec
    key = "heog_reg-BA_matched"; diff = [b[key] - b1[key] for b, b1 in zip(B, B1)]
    out["descriptive"] = {"heog_advantage_over_BA_matched_this_arm_minus_A1": {"estimate": float(est[key] - est1[key]), "ci95": ci(diff, 0.05),
                                                                               "A1_estimate": float(est1[key])}}
    tr = {}
    for m in METRICS:
        bias = np.full(len(units), np.nan)
        for mod in sorted(set(models)):
            i1 = np.array([k for k, x in enumerate(models1) if x == mod]); i3 = np.array([k for k, x in enumerate(models) if x == mod])
            ok = np.isfinite(X1[m + "_all"][i1]) & np.isfinite(X1["t_all"][i1])
            if ok.sum() < 3 or i3.size == 0:
                continue
            c1, c0 = np.polyfit(X1[m + "_all"][i1][ok], X1["t_all"][i1][ok], 1); bias[i3] = c0 + c1 * X[m + "_all"][i3] - X["t_all"][i3]
        bs = [np.nanmean(np.concatenate([bias[G[k]] for k in np.random.default_rng(SEED + 2 + j).integers(0, len(G), len(G))])) for j in range(min(a.n_boot, 2000))]
        tr[m] = {"mean_bias": float(np.nanmean(bias)), "ci95": ci(bs, 0.05)}
    out["descriptive"]["calibration_transfer_from_A1"] = tr
    cells = sorted({r["cid"] for r in units})
    out["descriptive"]["mean_truth_by_cell"] = {c: float(np.mean([X["t_all"][k] for k, r in enumerate(units) if r["cid"] == c])) for c in cells}
    p = Path(a.out or f"results/analysis/amendment3_{a.arm}.json"); p.parent.mkdir(parents=True, exist_ok=True); json.dump(out, open(p, "w"), indent=1)
    print("written", p)


if __name__ == "__main__":
    main()
