"""3.8 / §2 statistical procedures.

H1   D ~ amp × model (sum-to-zero), random intercept subject, seed variance
     component; unit of decision = "metric × confound"; pass iff the main
     (model-averaged) amp slope CI lower bound > 0.  Per-model slopes from the
     interaction are descriptive.  Detection threshold is descriptive
     (uncorrected CI, first amplitude with CI lower > 0).
H2   per model: TOST on D(s_high) − D(s_low) with ±δ at the training amplitude
     whose D lies in [0.3, 0.8]; otherwise "undecidable".
H3   within-model standardised truth vs metric, model fixed effect × metric,
     overlap |corr(a_mu, a_α)| as continuous moderator (reported, not in the
     pass rule); hierarchical bootstrap (seeds, then subjects) of the pooled
     correlation; reliability gates.
H3r  calibration transfer: fit truth ~ metric in matched units, predict
     mismatch / p = 1.0 units, bias = mean(pred − truth).
H4   trial-level binomial GLMM: correct ~ type × model + (1|subject), bias
     control models as a factor; contrast congruent − none, control-corrected.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats as sps


# ------------------------------------------------------------------ bootstrap
def hierarchical_bootstrap(stat_fn, seeds, subjects, n_boot: int = 2000, seed: int = 0, ci: float = 0.95) -> dict:
    """stat_fn(seed_sample, subject_sample) -> scalar.  Seeds resampled first,
    then subjects (shared across the resampled seeds)."""
    rng = np.random.default_rng(seed)
    seeds, subjects = list(seeds), list(subjects)
    est = stat_fn(seeds, subjects)
    boots = []
    for _ in range(n_boot):
        ss = list(rng.choice(seeds, size=len(seeds), replace=True))
        sj = list(rng.choice(subjects, size=len(subjects), replace=True))
        v = stat_fn(ss, sj)
        if np.isfinite(v):
            boots.append(v)
    boots = np.asarray(boots)
    a = (1 - ci) / 2
    lo, hi = (np.quantile(boots, [a, 1 - a]) if boots.size else (np.nan, np.nan))
    return {"estimate": float(est), "ci_low": float(lo), "ci_high": float(hi), "n_boot": int(boots.size)}


# ------------------------------------------------------------------ H1
def h1_mixed_model(df: pd.DataFrame) -> dict:
    """df columns: D, amp, model, subject, seed (one row per subject × unit × amplitude)."""
    import statsmodels.formula.api as smf
    d = df.copy()
    d["model"] = d["model"].astype("category")
    models = list(d["model"].cat.categories)
    formula = "D ~ amp * C(model, Sum)"
    vc = {"seed": "0 + C(seed)"} if d["seed"].nunique() > 1 else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = smf.mixedlm(formula, d, groups=d["subject"], vc_formula=vc).fit(reml=True, method="lbfgs")
    params, cov = fit.params, fit.cov_params()
    z = sps.norm.ppf(0.975)
    main = float(params["amp"]); se = float(np.sqrt(cov.loc["amp", "amp"]))
    per_model = {}
    inter = [f"amp:C(model, Sum)[S.{m}]" for m in models[:-1]]
    for i, m in enumerate(models):
        L = pd.Series(0.0, index=params.index); L["amp"] = 1.0
        if i < len(models) - 1:
            L[inter[i]] = 1.0
        else:
            for t in inter:
                L[t] = -1.0
        est = float(L @ params); s = float(np.sqrt(L @ cov @ L))
        per_model[m] = {"slope": est, "ci_low": est - z * s, "ci_high": est + z * s}
    return {"slope": main, "ci_low": main - z * se, "ci_high": main + z * se,
            "pass": bool(main - z * se > 0), "per_model": per_model, "converged": bool(fit.converged)}


def detection_threshold(df: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    """Descriptive: first test amplitude whose subject-bootstrap CI of mean D excludes 0 (uncorrected)."""
    rng = np.random.default_rng(seed)
    out = {}
    for amp, g in sorted(df.groupby("amp"), key=lambda t: t[0]):
        per_subj = g.groupby("subject")["D"].mean().values
        bs = [rng.choice(per_subj, per_subj.size).mean() for _ in range(n_boot)]
        out[float(amp)] = {"mean": float(per_subj.mean()), "ci_low": float(np.quantile(bs, 0.025)),
                           "ci_high": float(np.quantile(bs, 0.975))}
    thr = next((a for a, v in out.items() if v["ci_low"] > 0 and a > 0), None)
    return {"threshold_amp": thr, "by_amp": out}


# ------------------------------------------------------------------ H2
def tost_paired(diff: np.ndarray, delta: float, alpha: float = 0.05) -> dict:
    diff = np.asarray(diff, dtype=float)
    n = diff.size
    m, se = diff.mean(), diff.std(ddof=1) / np.sqrt(n)
    t_lo = (m + delta) / se; t_hi = (m - delta) / se
    p_lo = 1 - sps.t.cdf(t_lo, n - 1); p_hi = sps.t.cdf(t_hi, n - 1)
    p = max(p_lo, p_hi)
    tcrit = sps.t.ppf(1 - alpha, n - 1)
    return {"mean_diff": float(m), "ci90_low": float(m - tcrit * se), "ci90_high": float(m + tcrit * se),
            "p_tost": float(p), "equivalent": bool(p < alpha), "n": int(n)}


def h2_select_amp(D_by_amp: dict[float, float], d_range=(0.3, 0.8)) -> float | None:
    """Pick the training amplitude whose D (matched test conditions) lies in d_range;
    closest to the range centre if several.  None -> undecidable for this model."""
    ok = {a: d for a, d in D_by_amp.items() if d_range[0] <= d <= d_range[1]}
    if not ok:
        return None
    c = np.mean(d_range)
    return min(ok, key=lambda a: abs(ok[a] - c))


# ------------------------------------------------------------------ H3
def zscore_within(df: pd.DataFrame, cols, by="model") -> pd.DataFrame:
    d = df.copy()
    for c in cols:
        d[c + "_z"] = d.groupby(by)[c].transform(lambda v: (v - v.mean()) / (v.std(ddof=1) + 1e-12))
    return d


def pooled_within_model_corr(df: pd.DataFrame, truth="truth", metric="metric", by="model") -> float:
    d = zscore_within(df, [truth, metric], by)
    return float(np.corrcoef(d[truth + "_z"], d[metric + "_z"])[0, 1]) if len(d) > 2 else float("nan")


def h3_model(df: pd.DataFrame, truth="truth", metric="metric") -> dict:
    """OLS: truth_z ~ metric_z × C(model, Sum) + metric_z:overlap (overlap = moderator)."""
    import statsmodels.formula.api as smf
    d = zscore_within(df, [truth, metric])
    f = f"{truth}_z ~ {metric}_z * C(model, Sum)"
    if "overlap" in d and d["overlap"].nunique() > 1:
        f += f" + {metric}_z:overlap"
    fit = smf.ols(f, d).fit()
    ci = fit.conf_int()
    res = {"slope": float(fit.params[f"{metric}_z"]), "ci_low": float(ci.loc[f"{metric}_z", 0]),
           "ci_high": float(ci.loc[f"{metric}_z", 1])}
    key = f"{metric}_z:overlap"
    if key in fit.params:
        res.update({"overlap_slope": float(fit.params[key]), "overlap_ci_low": float(ci.loc[key, 0]),
                    "overlap_ci_high": float(ci.loc[key, 1]), "overlap_p": float(fit.pvalues[key])})
    return res


def h3_decision(corr_ci_low: float, threshold: float, truth_rel: float, metric_rel: float,
                min_rel: float = 0.5) -> str:
    if not np.isfinite(truth_rel) or truth_rel < min_rel:
        return "undecidable (truth reliability)"
    if not np.isfinite(metric_rel) or metric_rel < min_rel:
        return "fail (metric reliability)"
    return "pass" if corr_ci_low > threshold else "fail"


# ------------------------------------------------------------------ H3r
def calibration_transfer(matched: pd.DataFrame, target: pd.DataFrame, truth="truth", metric="metric") -> dict:
    """Fit truth ~ metric per model on matched units; predict target units."""
    biases = []
    for m, g in matched.groupby("model"):
        t = target[target.model == m]
        if len(g) < 3 or t.empty:
            continue
        coef = np.polyfit(g[metric], g[truth], 1)
        biases.append(np.polyval(coef, t[metric]) - t[truth])
    b = np.concatenate(biases) if biases else np.array([])
    return {"bias": float(b.mean()) if b.size else float("nan"),
            "abs_bias": float(np.abs(b).mean()) if b.size else float("nan"), "n": int(b.size)}


# ------------------------------------------------------------------ H4 / H5
def trial_glmm(df: pd.DataFrame, formula: str | None = None) -> dict:
    """Binomial mixed model with random subject intercept (variational Bayes),
    plus a GEE (exchangeable within subject) fit as a frequentist cross-check and an
    overdispersion ratio on subject-aggregated counts (3.8: check, then (1|subject:model)
    or beta-binomial if needed)."""
    import patsy
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    if formula is None:
        rhs = "C(type, Treatment('none'))" + (" * C(model)" if df["model"].nunique() > 1 else "")
        formula = "correct ~ " + rhs
    rhs = formula.split("~", 1)[1]
    out = {"formula": formula}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        glmm = BinomialBayesMixedGLM.from_formula(formula, {"subject": "0 + C(subject)"}, df).fit_vb()
        out["glmm"] = pd.DataFrame({"mean": glmm.fe_mean, "sd": glmm.fe_sd}, index=glmm.model.exog_names)
        gee = smf.gee(formula, "subject", df, family=sm.families.Binomial(),
                      cov_struct=sm.cov_struct.Exchangeable()).fit()
        ci = gee.conf_int()
        out["gee"] = pd.DataFrame({"coef": gee.params, "ci_low": ci[0], "ci_high": ci[1], "p": gee.pvalues})
        agg = df.groupby(["subject", "type", "model"])["correct"].agg(["sum", "count"]).reset_index()
        X = patsy.dmatrix(rhs, agg, return_type="dataframe")
        endog = np.column_stack([agg["sum"], agg["count"] - agg["sum"]])
        binom = sm.GLM(endog, X, family=sm.families.Binomial()).fit()
        out["dispersion"] = float(binom.pearson_chi2 / binom.df_resid) if binom.df_resid > 0 else float("nan")
    return out


def h4_bias_corrected(df: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    """(acc_congruent − acc_none)_main − (same)_control, subject bootstrap.
    df columns: subject, type, correct, arm ∈ {main, control}."""
    rng = np.random.default_rng(seed)
    subs = df.subject.unique()
    g = df[df.type.isin(["congruent", "none"])].groupby(["subject", "arm", "type"])["correct"].agg(["sum", "count"])

    def stat(sample):
        w = pd.Series(sample).value_counts()          # resampled subjects with multiplicity
        tot = sum(g.xs(s, level=0) * c for s, c in w.items() if s in g.index.get_level_values(0))
        acc = tot["sum"] / tot["count"]
        return (acc[("main", "congruent")] - acc[("main", "none")]) - (acc[("control", "congruent")] - acc[("control", "none")])

    est = stat(subs)
    bs = [stat(rng.choice(subs, subs.size)) for _ in range(n_boot)]
    return {"estimate": float(est), "ci_low": float(np.quantile(bs, 0.025)), "ci_high": float(np.quantile(bs, 0.975)),
            "pass": bool(np.quantile(bs, 0.025) > 0 or np.quantile(bs, 0.975) < 0)}
