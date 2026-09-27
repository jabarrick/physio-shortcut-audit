"""Hypothesis tests H1–H5 from stored unit records (outline §2, §4).

Every unit-level quantity is recomputed from per-subject sufficient statistics,
so the hierarchical bootstrap (seeds, then subjects) re-derives BA-type values
exactly.  Outputs: results/analysis/<H>.json (+ CSV tables).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from ..metrics.probe import D_from_ba
from ..metrics.truth import spearman_brown
from ..stats.inference import (calibration_transfer, detection_threshold, h1_mixed_model, h2_select_amp, h3_decision,
                               h3_model, h4_bias_corrected, hierarchical_bootstrap, pooled_within_model_corr,
                               tost_paired, trial_glmm)
from ..stats.suffstats import ba_from_counts, ratio_from_sums
from ..utils.common import get_logger, write_json
from .context import Context

log = get_logger("p3audit.analysis")
REFERENCE_PREFIXES = ("random_", "linear_raw")


# ------------------------------------------------------------------ loading
def load_units(ctx: Context, sub: str = "units") -> list[dict]:
    out = []
    for p in sorted((ctx.results / sub).glob("*.json")):
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        if r.get("standin"):
            log.warning(f"{p.name}: produced with a stand-in foundation model — excluded")
            continue
        out.append(r)
    return out


def _is_reference(model: str) -> bool:
    return model.startswith(REFERENCE_PREFIXES)


# ------------------------------------------------------------------ metric registry
def _ba(c, subs):
    return ba_from_counts(c, subs)


def truth(rec, subs=None, part="all", kind="neutral"):
    tc = rec["truth_counts"]
    if kind not in tc:
        return np.nan
    return _ba(tc["matched"][part], subs) - _ba(tc[kind][part], subs)


def m_erasure(rec, subs=None, part="all"):
    e = rec.get("erasure")
    return np.nan if e is None else _ba(e["counts_head"][part], subs) - _ba(e["counts_erased"][part], subs)


def m_sri(rec, subs=None, part="all"):
    s = rec.get("sri")
    return np.nan if s is None else _ba(s["counts"]["post_ctrl"][part], subs) - _ba(s["counts"]["post_alpha"][part], subs)


def m_central_alpha_drop(rec, subs=None, part="all"):
    s = rec.get("sri")
    return np.nan if s is None else _ba(rec["truth_counts"]["matched"][part], subs) - _ba(s["counts"]["central_alpha"][part], subs)


def m_heog(c="0"):
    def f(rec, subs=None, part="all"):
        h = rec.get("heog", {}).get(c)
        return np.nan if h is None else _ba(rec["truth_counts"]["matched"][part], subs) - _ba(h["counts"][part], subs)
    return f


def m_ig(field_idx=(0, 1), baseline="band_removed"):
    def f(rec, subs=None, part="all"):
        g = rec.get("ig", {}).get(baseline)
        return np.nan if g is None else ratio_from_sums(g["sums"][part], field_idx[0], field_idx[1], subs)
    return f


DEPENDENCE_METRICS = {
    "alpha": {"erasure_keep": m_erasure, "sri_post": m_sri, "ig_share": m_ig((0, 1)), "ig_norm_abs": m_ig((0, 2)),
              "heog_reg": m_heog("0")},
    "saccade": {"erasure_keep": m_erasure, "heog_reg": m_heog("0"), "ig_share": m_ig((0, 1)),
                "ig_norm_abs": m_ig((0, 2))},
}
# metrics that enter the confirmatory H3 decision per confound (others descriptive)
CONFIRMATORY = {"alpha": ["erasure_keep", "sri_post", "ig_share"], "saccade": ["erasure_keep", "heog_reg", "ig_share"]}


# ------------------------------------------------------------------ helpers
def _test_subjects(units):
    s = set()
    for r in units:
        s |= {int(k) for k in r["truth_counts"]["matched"]["all"]}
    return sorted(s)


def degenerate_subjects(units, cfg) -> dict:
    """Per-subject truth degeneration (3.4.3): mean ΔBA_neu over p = 0.9 units below threshold."""
    thr = cfg["alpha_component"]["degenerate_truth"]["min_delta_ba_neu"]
    out = {}
    for conf in ("alpha", "saccade"):
        us = [r for r in units if r["cell"]["confound"] == conf and r["cell"]["p"] == 0.9 and not _is_reference(r["model"])]
        if not us:
            continue
        per = {s: np.nanmean([truth(r, [s]) for r in us]) for s in _test_subjects(us)}
        bad = [s for s, v in per.items() if not np.isfinite(v) or v < thr]
        out[conf] = {"per_subject": per, "degenerate": bad, "fraction": len(bad) / max(1, len(per)),
                     "generator_unusable": len(bad) / max(1, len(per)) > cfg["alpha_component"]["max_degenerate_fraction"]}
    return out


def _overlap(rec, s):
    return float(rec["overlap"].get(str(s), rec["overlap"].get(s, np.nan)))


# ------------------------------------------------------------------ H1
def analyse_h1(ctx: Context, units, p_level: float = 0.9) -> dict:
    res = {}
    for conf in ("alpha", "saccade"):
        rows = []
        for r in units:
            if r["cell"]["confound"] != conf or r["cell"]["p"] != p_level or r["cell"]["module"] != "p_series":
                continue
            if _is_reference(r["model"]) or "probe" not in r:
                continue
            for amp, pr in r["probe"].items():
                if float(amp) not in ctx.cfg["design"]["h1_test_amps"]:
                    continue
                for s, c in pr["counts"].items():
                    rows.append({"D": D_from_ba(ba_from_counts(pr["counts"], [s])), "amp": float(amp), "model": r["model"],
                                 "subject": int(s), "seed": r["seed"]})
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df.to_csv(ctx.results / "analysis" / f"H1_{conf}_rows.csv", index=False)
        res[conf] = {"lmm": h1_mixed_model(df), "detection_threshold": detection_threshold(df),
                     "unit": f"linear_probe × {conf}"}
    return res


# ------------------------------------------------------------------ H2
def analyse_h2(ctx: Context, units) -> dict:
    cfg = ctx.cfg
    st = cfg["stats"]
    res = {}
    for conf in ("alpha", "saccade"):
        fac = [r for r in units if r["cell"]["confound"] == conf and r["cell"]["p"] == cfg["design"]["factorial_p"]
               and r["cell"]["module"] in ("factorial", "p_series") and not _is_reference(r["model"])]
        for model in sorted({r["model"] for r in fac}):
            um = [r for r in fac if r["model"] == model]
            D_by_amp = {}
            for r in um:
                if r["cell"]["s"] == "s_star":
                    a = float(r["cell"]["amp"])
                    D_by_amp.setdefault(a, []).append(r["probe"][f"{a:g}"]["D"])
            D_by_amp = {a: float(np.mean(v)) for a, v in D_by_amp.items()}
            amp = h2_select_amp(D_by_amp, tuple(st["h2_d_range"]))
            key = f"{conf}/{model}"
            if amp is None:
                res[key] = {"decision": "undecidable (no training amplitude with D in range)", "D_by_amp": D_by_amp}
                continue

            def per_subject(level, fn):
                rr = [r for r in um if r["cell"]["s"] == level and float(r["cell"]["amp"]) == amp]
                subs = _test_subjects(rr)
                return {s: np.nanmean([fn(r, s) for r in rr]) for s in subs}

            dfun = lambda r, s: D_from_ba(ba_from_counts(r["probe"][f"{amp:g}"]["counts"], [s]))
            hi, lo = per_subject("s_high", dfun), per_subject("s_low", dfun)
            common = sorted(set(hi) & set(lo))
            diff = np.array([hi[s] - lo[s] for s in common])
            tost = tost_paired(diff, st["h2_delta"])
            th, tl = per_subject("s_high", lambda r, s: truth(r, [s])), per_subject("s_low", lambda r, s: truth(r, [s]))
            mdiff = np.array([th[s] - tl[s] for s in common])
            from scipy.stats import ttest_1samp
            res[key] = {"amp": amp, "D_by_amp": D_by_amp, "tost": tost,
                        "manipulation_check": {"mean_delta_ba_neu_diff": float(mdiff.mean()),
                                               "p": float(ttest_1samp(mdiff, 0).pvalue)},
                        "decision": "encoding ≠ dependence supported" if tost["equivalent"] else "not supported"}
    return res


# ------------------------------------------------------------------ vectorised bootstrap machinery
SPECS = {  # metric -> how to rebuild it from stored per-subject statistics
    "truth": ("ba_diff", ("truth_counts", "matched"), ("truth_counts", "neutral")),
    "erasure_keep": ("ba_diff", ("erasure", "counts_head"), ("erasure", "counts_erased")),
    "sri_post": ("ba_diff", ("sri", "counts", "post_ctrl"), ("sri", "counts", "post_alpha")),
    "heog_reg": ("ba_diff", ("truth_counts", "matched"), ("heog", "0", "counts")),
    "ig_share": ("ratio", ("ig", "band_removed", "sums"), (0, 1)),
    "ig_norm_abs": ("ratio", ("ig", "band_removed", "sums"), (0, 2)),
}


def _dig(rec, path):
    node = rec
    for k in path:
        if not isinstance(node, dict) or k not in node:
            return None
        node = node[k]
    return node


class MetricArrays:
    """Per-unit × per-subject statistics as dense arrays so that one bootstrap draw
    (subject multiplicity weights w, unit multiset) costs a few einsums."""

    def __init__(self, units, subjects, names):
        self.units, self.subjects = units, list(subjects)
        self.models = np.array([r["model"] for r in units])
        self.seeds = np.array([r["seed"] for r in units])
        self.arr = {}
        for name in names:
            kind, a, b = SPECS[name]
            for part in ("all", "half0", "half1"):
                if kind == "ba_diff":
                    self.arr[(name, part)] = (kind, self._stack(a, part, 4), self._stack(b, part, 4))
                else:
                    self.arr[(name, part)] = (kind, self._stack(a, part, 4), b)

    def _stack(self, path, part, width):
        out = np.full((len(self.units), len(self.subjects), width), np.nan)
        for i, r in enumerate(self.units):
            node = _dig(r, path)
            if node is None:
                continue
            node = node.get(part, node) if isinstance(node, dict) and part in node else node
            for j, s in enumerate(self.subjects):
                v = node.get(str(s), node.get(s)) if isinstance(node, dict) else None
                if v is not None:
                    out[i, j, :len(v[:width])] = v[:width]
        return out

    @staticmethod
    def _ba(C, w):
        c = np.einsum("usk,s->uk", np.nan_to_num(C), w)
        valid = ~np.isnan(C).all(axis=(1, 2))
        with np.errstate(invalid="ignore", divide="ignore"):
            ba = 0.5 * (c[:, 0] / (c[:, 0] + c[:, 1]) + c[:, 3] / (c[:, 2] + c[:, 3]))
        ba[~valid] = np.nan
        return ba

    def value(self, name, part, w):
        kind, A, B = self.arr[(name, part)]
        if kind == "ba_diff":
            return self._ba(A, w) - self._ba(B, w)
        c = np.einsum("usk,s->uk", np.nan_to_num(A), w)
        with np.errstate(invalid="ignore", divide="ignore"):
            v = c[:, B[0]] / c[:, B[1]]
        v[np.isnan(A).all(axis=(1, 2))] = np.nan
        return v


def _zwithin(v, models):
    out = np.full_like(v, np.nan)
    for m in np.unique(models):
        k = (models == m) & np.isfinite(v)
        if k.sum() > 1:
            out[k] = (v[k] - v[k].mean()) / (v[k].std(ddof=1) + 1e-12)
    return out


def _corr(a, b):
    k = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[k], b[k])[0, 1]) if k.sum() > 3 else np.nan


def xfit_corr(ma: MetricArrays, metric, w, idx) -> float:
    """Cross-fitted pooled within-model correlation (truth on one half, metric on the other)."""
    mod = ma.models[idx]
    t0, t1 = ma.value("truth", "half0", w)[idx], ma.value("truth", "half1", w)[idx]
    m0, m1 = ma.value(metric, "half0", w)[idx], ma.value(metric, "half1", w)[idx]
    return float(np.nanmean([_corr(_zwithin(t0, mod), _zwithin(m1, mod)), _corr(_zwithin(t1, mod), _zwithin(m0, mod))]))


def half_rel(ma: MetricArrays, metric, w) -> float:
    return spearman_brown(_corr(ma.value(metric, "half0", w), ma.value(metric, "half1", w)))


def bootstrap_xfit(ma: MetricArrays, metric, n_boot, seed=0, ci=0.95) -> dict:
    rng = np.random.default_rng(seed)
    S = len(ma.subjects)
    seeds = np.unique(ma.seeds)
    by_seed = {s: np.where(ma.seeds == s)[0] for s in seeds}
    w1 = np.ones(S)
    est = xfit_corr(ma, metric, w1, np.arange(len(ma.units)))
    boots = []
    for _ in range(n_boot):
        ss = rng.choice(seeds, seeds.size)
        idx = np.concatenate([by_seed[s] for s in ss])
        w = np.bincount(rng.integers(0, S, S), minlength=S).astype(float)
        v = xfit_corr(ma, metric, w, idx)
        if np.isfinite(v):
            boots.append(v)
    a = (1 - ci) / 2
    lo, hi = np.quantile(boots, [a, 1 - a]) if boots else (np.nan, np.nan)
    return {"estimate": est, "ci_low": float(lo), "ci_high": float(hi), "n_boot": len(boots)}


# ------------------------------------------------------------------ H3
def _unit_table(units, metric_fn, subs=None, part="all"):
    rows = []
    for r in units:
        rows.append({"uid": r["uid"], "model": r["model"], "seed": r["seed"], "cell": r["cid"],
                     "truth": truth(r, subs, part), "metric": metric_fn(r, subs, part)})
    return pd.DataFrame(rows).dropna()


def cross_fitted_corr(units, metric_fn, subs=None) -> float:
    rows = []
    for r in units:
        rows.append({"model": r["model"], "t0": truth(r, subs, "half0"), "t1": truth(r, subs, "half1"),
                     "m0": metric_fn(r, subs, "half0"), "m1": metric_fn(r, subs, "half1")})
    d = pd.DataFrame(rows).dropna()
    if len(d) < 4:
        return np.nan
    a = pooled_within_model_corr(d.rename(columns={"t0": "truth", "m1": "metric"}))
    b = pooled_within_model_corr(d.rename(columns={"t1": "truth", "m0": "metric"}))
    return float(np.nanmean([a, b]))


def half_reliability(units, fn, subs=None) -> float:
    h0 = np.array([fn(r, subs, "half0") for r in units]); h1 = np.array([fn(r, subs, "half1") for r in units])
    ok = np.isfinite(h0) & np.isfinite(h1)
    if ok.sum() < 4:
        return np.nan
    return spearman_brown(float(np.corrcoef(h0[ok], h1[ok])[0, 1]))


def analyse_h3(ctx: Context, units) -> dict:
    cfg = ctx.cfg
    st = cfg["stats"]
    degen = degenerate_subjects(units, cfg)
    res = {"degenerate": {k: {kk: vv for kk, vv in v.items() if kk != "per_subject"} for k, v in degen.items()}}
    for conf, metrics in DEPENDENCE_METRICS.items():
        us = [r for r in units if r["cell"]["confound"] == conf and r["cell"]["p"] == cfg["design"]["factorial_p"]
              and r["cell"]["module"] in ("factorial", "p_series") and not _is_reference(r["model"])]
        if not us:
            continue
        subs = [s for s in _test_subjects(us) if s not in set(degen.get(conf, {}).get("degenerate", []))]
        names = ["truth"] + [m for m in metrics if m in SPECS]
        ma = MetricArrays(us, subs, names)
        w1 = np.ones(len(subs))
        rel_truth = half_rel(ma, "truth", w1)
        for name, fn in metrics.items():
            if name not in SPECS:
                continue
            boot = bootstrap_xfit(ma, name, st_n_boot(cfg))
            rel_m = half_rel(ma, name, w1)
            # subject-level moderator model (overlap varies across subjects, not across units)
            rows = []
            for r in us:
                for s in subs:
                    rows.append({"model": r["model"], "truth": truth(r, [s]), "metric": fn(r, [s]),
                                 "overlap": _overlap(r, s)})
            mod = h3_model(pd.DataFrame(rows).dropna()) if rows else {}
            dec = h3_decision(boot["ci_low"], st["h3_min_corr_lower"], rel_truth, rel_m, st["min_reliability"])
            res[f"{conf}/{name}"] = {"corr_cross_fitted": boot, "reliability_truth": rel_truth,
                                     "reliability_metric": rel_m, "overlap_moderation": mod,
                                     "decision": dec, "confirmatory": name in CONFIRMATORY[conf],
                                     "n_units": len(us), "n_subjects": len(subs)}
            _unit_table(us, fn, subs).to_csv(ctx.results / "analysis" / f"H3_{conf}_{name}_units.csv", index=False)
    return res


def st_n_boot(cfg):
    return int(cfg["stats"]["n_boot"])


# ------------------------------------------------------------------ H3r
def analyse_h3r(ctx: Context, units) -> dict:
    cfg = ctx.cfg
    res = {}
    for conf, metrics in DEPENDENCE_METRICS.items():
        matched = [r for r in units if r["cell"]["confound"] == conf and r["cell"]["p"] == cfg["design"]["factorial_p"]
                   and r["cell"]["module"] in ("factorial", "p_series") and not _is_reference(r["model"])]
        targets = {"mismatch": [r for r in units if r["cell"]["confound"] == conf and r["cell"]["module"] == "mismatch"],
                   "p1": [r for r in units if r["cell"]["confound"] == conf and r["cell"]["module"] == "corner"]}
        for name, fn in metrics.items():
            for tname, tu in targets.items():
                if not tu or (tname == "p1" and name == "erasure_keep"):   # erasure unidentifiable at p = 1
                    continue
                m, t = _unit_table(matched, fn), _unit_table(tu, fn)
                if m.empty or t.empty:
                    continue
                r = calibration_transfer(m, t)
                r["pass"] = bool(np.isfinite(r["abs_bias"]) and r["abs_bias"] <= cfg["stats"]["h3r_max_bias"])
                res[f"{conf}/{name}/{tname}"] = r
    return res


# ------------------------------------------------------------------ descriptive (4.1 / 4.4)
def analyse_descriptive(ctx: Context, units) -> dict:
    out = {"group_template": {}, "within_subject": {}}
    for r in units:
        if r["cell"]["module"] == "group_template":
            twin = next((u for u in units if u["model"] == r["model"] and u["seed"] == r["seed"]
                         and u["cell"]["confound"] == "alpha" and u["cell"]["p"] == 0.9
                         and u["cell"]["s"] == "s_star" and u["cell"]["amp"] == 1.0
                         and u["cell"]["module"] == "p_series"), None)
            if twin:
                out["group_template"][r["uid"]] = {
                    "truth_group": truth(r), "truth_indiv": truth(twin),
                    "D_group": r["probe"]["1"]["D"], "D_indiv": twin["probe"]["1"]["D"],
                    "keep_group": m_erasure(r), "keep_indiv": m_erasure(twin)}
        if "probe" in r and "1" in r["probe"] and "within_subject" in r["probe"]["1"]:
            out["within_subject"][r["uid"]] = {"D_main": r["probe"]["1"]["D"],
                                               "D_within": r["probe"]["1"]["within_subject"]["D"],
                                               "keep_main": m_erasure(r),
                                               "keep_within": r.get("erasure", {}).get("within_subject", {}).get("delta_ba_keep")}
        if "erasure" in r:
            out.setdefault("angle_vs_bias", []).append({"uid": r["uid"], "angle": r["erasure"]["angle_deg"],
                                                       "bias": m_erasure(r) - truth(r)})
    return out


# ------------------------------------------------------------------ H4 / H5
def analyse_h4(ctx: Context) -> dict:
    files = sorted((ctx.results / "real").glob("*_trials.csv"))
    if not files:
        return {}
    df = pd.concat([pd.read_csv(f) for f in files])
    k = ctx.cfg["h4"]["k_min"]
    main = df[df.arm == "main"]
    cnt = main.drop_duplicates(["subject", "run", "trial"]).groupby(["subject", "type"]).size().unstack(fill_value=0)
    elig = cnt[(cnt.get("congruent", 0) >= k) & (cnt.get("none", 0) >= k)].index.tolist()
    res = {"n_eligible": len(elig), "n_subjects": int(cnt.shape[0])}
    if len(elig) < 5:
        res["decision"] = "not identifiable (too few eligible subjects)"
        return res
    d = df[df.subject.isin(elig) & df.type.isin(["congruent", "none"])]
    res["glmm"] = {k2: v.to_dict() if hasattr(v, "to_dict") else v for k2, v in trial_glmm(d[d.arm == "main"]).items()}
    res["bias_corrected"] = {m: h4_bias_corrected(g) for m, g in d.groupby("model")}
    return res


def analyse_h5(ctx: Context, h3: dict | None = None) -> dict:
    recs = [json.load(open(p, encoding="utf-8")) for p in sorted((ctx.results / "real").glob("*__seed*.json"))]
    if not recs:
        return {}
    passed = {k for k, v in (h3 or {}).items() if isinstance(v, dict) and v.get("decision") == "pass"}
    rows = []
    for r in recs:
        base = ba_from_counts(r["counts"])
        rows.append({"model": r["model"], "fold": r["fold"], "seed": r["seed"], "ba": base,
                     "sri_post": ba_from_counts(r["sri"]["post_ctrl"]) - ba_from_counts(r["sri"]["post_alpha"]),
                     "heog_reg": base - ba_from_counts(r["heog"]),
                     "ig_alpha_share": r["ig"]["alpha"]["share"], "ig_saccade_share": r["ig"]["saccade"]["share"],
                     "alpha_erasure_keep": r["alpha_erasure"]["delta_ba_keep"], "alpha_probe_D": r["alpha_probe_D"]})
    df = pd.DataFrame(rows)
    summ = df.groupby("model").agg(["mean", "std"]).drop(columns=["fold", "seed"], level=0)
    status = {"sri_post": "alpha/sri_post", "heog_reg": "saccade/heog_reg", "ig_alpha_share": "alpha/ig_share",
              "ig_saccade_share": "saccade/ig_share", "alpha_erasure_keep": "alpha/erasure_keep"}
    return {"summary": json.loads(summ.to_json()),
            "interpretable": {m: (k in passed) for m, k in status.items()},
            "note": "metrics not passing H3/H3r are exploratory only (3.7)"}


# ------------------------------------------------------------------ entry
def run_all(ctx: Context) -> dict:
    (ctx.results / "analysis").mkdir(parents=True, exist_ok=True)
    units = load_units(ctx)
    out = {"n_units": len(units)}
    out["H1"] = analyse_h1(ctx, units); write_json(ctx.results / "analysis" / "H1.json", out["H1"])
    out["H2"] = analyse_h2(ctx, units); write_json(ctx.results / "analysis" / "H2.json", out["H2"])
    out["H3"] = analyse_h3(ctx, units); write_json(ctx.results / "analysis" / "H3.json", out["H3"])
    out["H3r"] = analyse_h3r(ctx, units); write_json(ctx.results / "analysis" / "H3r.json", out["H3r"])
    out["descriptive"] = analyse_descriptive(ctx, units); write_json(ctx.results / "analysis" / "descriptive.json", out["descriptive"])
    out["H4"] = analyse_h4(ctx); write_json(ctx.results / "analysis" / "H4.json", out["H4"])
    out["H5"] = analyse_h5(ctx, out["H3"]); write_json(ctx.results / "analysis" / "H5.json", out["H5"])
    return out
