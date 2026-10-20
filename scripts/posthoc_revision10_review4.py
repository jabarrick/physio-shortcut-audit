"""Post hoc (exploratory; fourth review): within-cell checks on the saccade units of Amendment 1 (results/units_a1,
registered halves) and Amendment 3 (results/units_a3_RH, results/units_a3_RV, exactly coupled halves).
Uses build() and stat() of scripts/amendment3_analyse.py unchanged.
  * disattenuated within-cell partial r: estimate / sqrt(split-half r of ground truth x split-half r of the quantity)
    (point estimates; split-half r without Spearman-Brown, as in the registered reliability rule);
  * the same within-cell statistics separately for each core model (9 cell groups per model; cluster bootstrap over
    cells, 2000 resamples, seed 20261020);
  * variance decomposition of the residual ground truth: signal = covariance of the two halves, noise = half the
    variance of the half difference, total = variance of the full-sample residual (all on cell x model residuals).
  * IG share with the zero baseline (stored in every unit as ig.zero) against the band-removed baseline of the
    registered metric: H3-style cross-fitted correlation with ground truth across the p = 0.9 factorial and p-series
    units of the preregistered runs (within-model standardised, pooled; subject bootstrap with seeds fixed, 2000
    resamples), and the within-cell statistics of the saccade batches.
Writes results/analysis/posthoc_revision10_review4.json.
"""
import json, sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore"); ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from amendment3_analyse import build, stat, load, METRICS
N_BOOT, SEED = 2000, 20261020
KEYS = ["D", "BA_matched"] + METRICS + [f"{m}|BA_matched" for m in METRICS] + [f"{m}-BA_matched" for m in METRICS] + [f"{m}-D" for m in METRICS]


def resid(v, groups):
    out = np.full(len(v), np.nan)
    for g in groups:
        x = v[g]; ok = np.isfinite(x); out[g[ok]] = x[ok] - x[ok].mean()
    return out


def ci(vals):
    v = np.array(vals, float); v = v[np.isfinite(v)]
    return [float(np.quantile(v, .025)), float(np.quantile(v, .975))] if len(v) > 10 else None


out = {"n_boot": N_BOOT, "seed": SEED}
for name, d in (("A1", "results/units_a1"), ("A3_RH", "results/units_a3_RH"), ("A3_RV", "results/units_a3_RV")):
    units = [r for r in load(ROOT / d) if r["cell"]["confound"] == "saccade" and r["cell"]["p"] == 0.9]
    X, G, models = build(units); est = stat(X, G)
    dis = {}
    for q, rq in [("BA_matched", "rel_BA_matched")] + [(m, f"rel_{m}") for m in METRICS]:
        den = est["rel_truth"] * est[rq]
        dis[q] = float(est[q] / np.sqrt(den)) if den > 0 else None
    t0, t1 = resid(X["t0"], G), resid(X["t1"], G); ta = resid(X["t_all"], G)
    vd = {"signal_cov_halves": float(np.nanmean(t0 * t1)), "noise_half_diff": float(np.nanvar(t0 - t1) / 2),
          "total_resid_var_full": float(np.nanvar(ta)), "sd_resid_full": float(np.nanstd(ta))}
    per = {}
    rng = np.random.default_rng(SEED)
    for mod in sorted(set(models)):
        Gm = [g for g in G if models[g[0]] == mod]
        e = stat(X, Gm); B = [stat(X, [Gm[k] for k in rng.integers(0, len(Gm), len(Gm))]) for _ in range(N_BOOT)]
        per[mod] = {"n_groups": len(Gm), "rel_truth": float(e["rel_truth"]),
                    **{k: {"est": float(e[k]), "ci95": ci([b[k] for b in B])} for k in KEYS}}
    out[name] = {"n_units": len(units), "rel_truth": float(est["rel_truth"]),
                 "pooled": {k: float(est[k]) for k in KEYS}, "reliability": {k: float(v) for k, v in est.items() if k.startswith("rel_")},
                 "disattenuated": dis, "variance": vd, "per_model": per}
    print("==", name, "rel_truth %.2f" % est["rel_truth"], {k: (round(v, 2) if v is not None else None) for k, v in dis.items()})
    print("   variance", {k: round(v, 5) for k, v in vd.items()})
    for mod, p in per.items():
        print("   %-9s" % mod, " ".join(f"{k}={p[k]['est']:+.2f}" for k in ["D", "BA_matched", "heog_reg", "ig_share", "erasure_keep", "ig_share|BA_matched", "heog_reg-BA_matched"]))
# ---- IG baseline comparison
import amendment3_analyse as a3
from p3audit.experiments import analysis as A
A.SPECS["ig_zero_share"] = ("ratio", ("ig", "zero", "sums"), (0, 1))
for conf in ("alpha", "saccade"):
    A.DEPENDENCE_METRICS[conf]["ig_zero_share"] = A.m_ig((0, 1), baseline="zero")
pre = [json.load(open(p, encoding="utf-8")) for p in sorted((ROOT / "results/units").glob("*.json"))]
igc = {}
for conf in ("alpha", "saccade"):
    us = [r for r in pre if r["cell"]["confound"] == conf and r["cell"]["p"] == 0.9 and r["cell"]["module"] in ("factorial", "p_series")
          and r.get("ig", {}).get("zero") is not None]
    subs = A._test_subjects(us); ma = A.MetricArrays(us, subs, ["truth", "ig_share", "ig_zero_share"]); S = len(subs)
    res = {"n_units": len(us)}
    for n in ("ig_share", "ig_zero_share"):
        e = A.xfit_corr(ma, n, np.ones(S), np.arange(len(us)))
        rng = np.random.default_rng(SEED)
        b = [A.xfit_corr(ma, n, np.bincount(rng.integers(0, S, S), minlength=S).astype(float), np.arange(len(us))) for _ in range(N_BOOT)]
        res[n] = {"est": float(e), "ci95_subject_only": ci(b)}
    igc["H3_" + conf] = res
    print("IG", conf, res)
a3.METRICS = ["ig_share", "ig_zero_share"]
for name, d in (("A1", "results/units_a1"), ("A3_RH", "results/units_a3_RH"), ("A3_RV", "results/units_a3_RV")):
    units = [r for r in a3.load(ROOT / d) if r["cell"]["confound"] == "saccade" and r["cell"]["p"] == 0.9]
    X, G, _ = a3.build(units); e = a3.stat(X, G); rng = np.random.default_rng(SEED)
    B = [a3.stat(X, [G[k] for k in rng.integers(0, len(G), len(G))]) for _ in range(N_BOOT)]
    keys = ["ig_share", "ig_zero_share", "ig_share|BA_matched", "ig_zero_share|BA_matched", "ig_share-D", "ig_zero_share-D"]
    igc["within_" + name] = {k: {"est": float(e[k]), "ci95": ci([b[k] for b in B])} for k in keys}
    print("IG within", name, {k: round(e[k], 2) for k in keys})
out["ig_baseline"] = igc
json.dump(out, open(ROOT / "results/analysis/posthoc_revision10_review4.json", "w"), indent=1)
