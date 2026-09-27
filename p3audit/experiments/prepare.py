"""Preparation stages:  split -> components -> saccade pool -> caches.

    p3audit prepare --stage split        3.1 metadata exclusion + 3.2 split
    p3audit prepare --stage components   per-subject GED alpha / mu (all subjects, no labels)
    p3audit prepare --stage pool         P6 split-half ICA templates (pilot subjects only)
    p3audit prepare --stage cache        per-subject streamed bases for the generator
"""
from __future__ import annotations

import json

import numpy as np

from ..generator.cache import NoUsableWindows, build_subject_cache
from ..generator.components import (estimate_alpha, estimate_mu, group_component,
                                    required_components, select_mu)
from ..generator.overlap import overlap_summary
from ..generator.saccade_ica import pool_feasible, template_for_subject
from ..saccades.detect import saccade_mask
from ..utils.common import get_logger, write_json
from .context import Context

log = get_logger("p3audit.prepare")


def stage_components(ctx: Context, subjects=None, overwrite: bool = False) -> dict:
    cfg = ctx.cfg
    subjects = subjects or ctx.all_subjects
    summary = {"n": 0, "alpha_fail": [], "mu_fail": []}
    for s in subjects:
        p = ctx.comp_path(s)
        if p.exists() and not overwrite:
            continue
        ctx.guard.record(s, [1, 2], "gen_alpha_component")
        alpha = estimate_alpha(ctx.source.load(s, 1), ctx.source.load(s, 2), cfg)
        ctx.guard.record(s, [3, 7, 11], "gen_mu_component")
        mu_l, mu_r = estimate_mu([ctx.source.load(s, r) for r in (3, 7, 11)], cfg)
        comps = {"alpha": alpha, "mu_L": mu_l, "mu_R": mu_r}
        # 3.4.2 revision 2026-09-19: one hemisphere per subject.  mu_L / mu_R are still
        # estimated and stored so the per-side record stays auditable and the bilateral
        # design remains reproducible under hemisphere: both.
        if cfg["task_component"].get("hemisphere", "both") == "single":
            comps["mu"] = select_mu(mu_l, mu_r, cfg)
        ov = overlap_summary(comps, ctx.ch_names, cfg)
        write_json(p, {"subject": s, "components": {k: v.to_dict() for k, v in comps.items()}, "overlap": ov,
                       "role": ctx.require_split().role(s), "pilot": s in ctx.require_split().pilot})
        summary["n"] += 1
        if not alpha.passed:
            summary["alpha_fail"].append(s)
        task_ok = comps["mu"].passed if "mu" in comps else (mu_l.passed and mu_r.passed)
        if not task_ok:
            summary["mu_fail"].append(s)
        side = comps["mu"].checks["side"] if "mu" in comps else None
        log.info(f"sub {s}: alpha {alpha.passed} muL {mu_l.passed} muR {mu_r.passed} "
                 f"side {side} overlap {ov['overlap']:.2f}")
    return summary


def stage_pool(ctx: Context, seed: int = 0) -> dict:
    cfg = ctx.cfg
    templates = []
    method = cfg["saccade"].get("template_method", "ica")
    if method == "eog_regression":
        from ..generator.saccade_regression import template_for_subject as fit_template
        purpose = "pilot_saccade_regression"
    else:
        fit_template, purpose = template_for_subject, "pilot_saccade_ica"
    for s in ctx.require_split().pilot:
        ctx.guard.record(s, cfg["saccade"]["lateral_runs"], purpose)
        runs = [ctx.source.load(s, r) for r in cfg["saccade"]["lateral_runs"]]
        t = fit_template(s, runs, cfg, seed=seed)
        templates.append(t)
        g = [h.n_gated for h in t.halves]
        log.info(f"pilot {s}: saccade template ok={t.ok} split-half r={t.split_half_r:.2f} "
                 f"gated={g} enr={[round(h.enrichment, 2) for h in t.halves]} "
                 f"bip={[round(h.bipolarity, 2) for h in t.halves]}")
    feasible, n_fail = pool_feasible(templates, cfg)
    ok = [t for t in templates if t.ok]
    rng = np.random.default_rng(cfg["seeds"]["split"])
    n_mis = min(cfg["saccade"]["mismatch_pool_n"], max(0, len(ok) - 2))
    order = rng.permutation(len(ok))
    mis_ids = sorted(ok[i].subject for i in order[:n_mis])
    np.savez(ctx.results / "saccade_pool.npz", **{f"s_{t.subject}": t.template for t in ok if t.subject not in mis_ids})
    np.savez(ctx.results / "saccade_pool_mismatch.npz", **{f"s_{t.subject}": t.template for t in ok if t.subject in mis_ids})
    report = {"feasible": feasible, "n_failed": n_fail, "n_pilot": len(templates),
              "main_pool": sorted(t.subject for t in ok if t.subject not in mis_ids), "mismatch_pool": mis_ids,
              "per_subject": [{"subject": t.subject, "ok": t.ok, "split_half_r": t.split_half_r,
                               "halves": [{"ok": h.ok, "r": h.r, "threshold": h.threshold,
                                           "component": h.component, "n_gated": h.n_gated,
                                           "enrichment": h.enrichment, "bipolarity": h.bipolarity,
                                           "reason": h.reason}
                                          for h in t.halves]} for t in templates]}
    write_json(ctx.results / "saccade_pool.json", report)
    if not feasible:
        log.warning(f"P6 rule: {n_fail} pilot subjects failed (> {cfg['saccade']['max_failed_subjects']}) "
                    "-> saccade injection infeasible; confound 2 must be replaced (6.4 (2)).")
    return report


def _load_npz(path):
    d = np.load(path)
    return {int(k.split("_")[1]): d[k] for k in d.files}


def stage_cache(ctx: Context, subjects=None, overwrite: bool = False,
                extras=("alpha_shift", "sacc_mismatch", "group")) -> dict:
    cfg = ctx.cfg
    pool = _load_npz(ctx.results / "saccade_pool.npz")
    mis_path = ctx.results / "saccade_pool_mismatch.npz"
    mis_pool = _load_npz(mis_path) if mis_path.exists() else {}
    group = None
    if "group" in extras:
        need = required_components(cfg)
        pil = [ctx.load_components(s) for s in ctx.require_split().pilot]
        pil = [c for c in pil if c and all(k in c and c[k].passed for k in need)]
        if len(pil) >= 2:
            group = {k: group_component([c[k] for c in pil], k) for k in need}
            write_json(ctx.results / "group_components.json", {k: v.to_dict() for k, v in group.items()})
        else:
            extras = tuple(e for e in extras if e != "group")
    subjects = subjects or ctx.all_subjects
    built, skipped = [], {}
    for s in subjects:
        if ctx.has_cache(s) and not overwrite:
            built.append(s); continue
        comps = ctx.load_components(s)
        if comps is None:
            skipped[s] = "no components"; continue
        # only the components the generator actually uses gate inclusion; under
        # hemisphere: single the unselected side may legitimately have failed
        failed = [k for k in required_components(cfg) if k not in comps or not comps[k].passed]
        if failed:
            skipped[s] = f"component inclusion failed: {failed}"; continue   # 3.4.2: background not used
        masks = None
        if cfg["background"]["trim_mode"] in ("drop_saccade_windows", "trim_and_drop"):
            masks = {r: saccade_mask(ctx.source.load(s, r), cfg) for r in (4, 8, 12, 6, 10, 14)}
        try:
            build_subject_cache(ctx.source, s, cfg, comps, pool, ctx.cache_root, extras=extras, group_comps=group,
                                mismatch_pool=mis_pool, guard=ctx.guard, saccade_masks=masks,
                                dtype=np.dtype(cfg.get("cache_dtype", "float32")))
        except NoUsableWindows as e:
            skipped[s] = f"no usable background windows: {e.counts}"
            log.warning(f"subject {s} skipped: {e}")
            continue
        built.append(s)
        log.info(f"cache built for subject {s}")
    rep = {"built": built, "skipped": {str(k): v for k, v in skipped.items()}}
    write_json(ctx.results / "cache_report.json", rep)
    return rep
