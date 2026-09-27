"""One semi-synthetic unit = (model, cell, seed): train, then compute ground truth
and every audit metric; store per-subject sufficient statistics so all
unit-level quantities can be re-derived under the hierarchical bootstrap.

Output: results/units/<uid>.json
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch

from ..data.windows import half_assignment
from ..generator.design import Unit, test_couplings
from ..generator.semisynth import ComposedSet, compose
from ..metrics.attribution import ig_batch_for, ig_filterbank, ig_targets, relevance_summary
from ..metrics.erasure import (delta_ba_keep, delta_ba_retrain, erasure_task_angle, fit_leace, head_predict,
                               residual_concept, within_subject_keep)
from ..metrics.intervention import apply_heog_regression, bandstop_channels, crop, fit_heog_regression
from ..metrics.probe import D_from_ba, fit_probe, within_subject_D
from ..metrics.truth import unit_truth
from ..models.registry import build_model
from ..stats.suffstats import ba_from_counts, confusion_by_subject, sums_by_subject
from ..training.trainer import embeddings, predict_logits, resolve_device, set_normalizer, train_model
from ..utils.common import ch_index, get_logger, rng as keyed_rng, write_json
from .context import Context

log = get_logger("p3audit.unit")


def _subsample(cs: ComposedSet, n_max: int, key) -> np.ndarray:
    idx = np.arange(len(cs))
    if len(idx) <= n_max:
        return idx
    return np.sort(keyed_rng(*key).choice(idx, n_max, replace=False))


def _stratified_sample(subj: np.ndarray, n_total: int, key) -> np.ndarray:
    subs = np.unique(subj)
    per = max(1, n_total // subs.size)
    out = []
    for s in subs:
        idx = np.where(subj == s)[0]
        out.append(keyed_rng(*key, int(s)).permutation(idx)[:per])
    return np.sort(np.concatenate(out))


def counts3(y, pred, subj, half) -> dict:
    """Per-subject confusion counts for all windows and for each half (reliability, 3.8)."""
    return {"all": confusion_by_subject(y, pred, subj),
            "half0": confusion_by_subject(y[half == 0], pred[half == 0], subj[half == 0]),
            "half1": confusion_by_subject(y[half == 1], pred[half == 1], subj[half == 1])}


def build_for_unit(ctx: Context, unit: Unit, fallback_standin: bool):
    cfg = ctx.cfg
    name = unit.model
    arch = name.replace("random_", "")
    any_c = ctx.cache(ctx.usable(ctx.split.train)[0])
    n_times = any_c.meta["core_len"]
    m = build_model(arch, cfg, len(ctx.ch_names), n_times, ctx.ch_names, fallback_standin=fallback_standin)
    return m, name.startswith("random_")


def run_unit(ctx: Context, unit: Unit, device=None, fallback_standin: bool = False, max_epochs: int | None = None,
             overwrite: bool = False, verbose: bool = False, metrics=("truth", "probe", "erasure", "sri", "heog", "ig"),
             roles: dict | None = None, out_dir: str = "units") -> dict:
    """roles: optional {'train','val','test'} subject lists (P8 uses validation subjects as the
    evaluation set so that no test subject is touched before preregistration)."""
    cfg = ctx.cfg
    out_path = ctx.results / out_dir / f"{unit.uid}.json"
    if out_path.exists() and not overwrite:
        with open(out_path, encoding="utf-8") as f:
            return json.load(f)
    t0 = time.time()
    device = device or resolve_device(cfg["training"]["device"])
    cell, seed = unit.cell, unit.seed
    sp = ctx.require_split()
    roles = roles or {"train": sp.train, "val": sp.val, "test": sp.test}
    tr_s, va_s, te_s = ctx.usable(roles["train"]), ctx.usable(roles["val"]), ctx.usable(roles["test"])
    caches = ctx.caches(tr_s + va_s + te_s)
    ch = ctx.ch_names
    # PILOT_LOG 13.8: training.device is 'auto' and resolve_device falls back to the CPU without
    # an error when the GPU disappears (it did on 2026-09-20).  Such a unit is numerically fine but
    # far slower, and until now nothing in the record said where it ran.  Provenance only.
    rec: dict = {"uid": unit.uid, "model": unit.model, "cell": cell.as_dict(), "cid": cell.cid, "seed": seed,
                 "n_subjects": {"train": len(tr_s), "val": len(va_s), "test": len(te_s)},
                 "config_hash": cfg.hash(), "device": str(device),
                 "torch_version": torch.__version__, "cuda_available": bool(torch.cuda.is_available())}
    # H3 moderator (十·2).  Under hemisphere: single the generator injects the SELECTED
    # hemisphere, so the moderator that matches what was actually injected is that side's
    # |corr(a_mu, a_alpha)|; the max-over-hemispheres value stays in the component record as a
    # robustness check.  Under hemisphere: both there is no selected side and max is used.
    def _moderator(ov):
        v = ov.get("overlap_selected")
        return ov["overlap"] if v is None else v
    rec["overlap"] = {s: _moderator(json.load(open(ctx.comp_path(s), encoding="utf-8"))["overlap"])
                      for s in te_s}

    # ------------------------------------------------------------ train
    tr = compose(caches, tr_s, cell, cfg, cell.p, key=(cell.cid, "train", seed))
    va = compose(caches, va_s, cell, cfg, cell.p, key=(cell.cid, "val", seed))
    X_tr, X_va = tr.materialize(), va.materialize()
    model, is_random = build_for_unit(ctx, unit, fallback_standin)
    rec["standin"] = bool(getattr(model, "is_standin", False))
    if is_random:
        model.to(device); set_normalizer(model, X_tr, cfg, seed=seed)
        rec["training"] = {"random_init": True}
    else:
        rec["training"] = train_model(model, X_tr, tr.y, X_va, va.y, cfg, seed, device, max_epochs, verbose)
    model.eval()
    predict = lambda X: predict_logits(model, X, device).argmax(1)

    # ------------------------------------------------------------ test sets & truth
    tests = {k: compose(caches, te_s, cell, cfg, q, key=(cell.cid, "test", k, seed), role="test")
             for k, q in test_couplings(cell.p).items()}
    base = tests["matched"]
    half = half_assignment(base.wids, cfg["seeds"]["half_split"])
    rec["test_half"] = half.tolist()
    preds, counts = {}, {}
    for k, cs in tests.items():
        preds[k] = predict(cs.materialize())
        counts[k] = counts3(cs.y, preds[k], cs.subj, half)
    rec["truth"] = unit_truth(base.y, preds, base.subj, half)
    rec["truth_counts"] = counts
    rec["z_test"] = {k: cs.z.tolist() for k, cs in tests.items()}

    # ------------------------------------------------------------ encoding (3.6.1, H1/H2)
    if "probe" in metrics:
        pc = cfg["probe"]
        amps = sorted(set(cfg["design"]["h1_test_amps"]) | {float(cell.amp)})
        rec["probe"] = {}
        for amp in amps:
            ptr = compose(caches, tr_s, cell, cfg, 0.5, key=(cell.cid, "probe_train", amp, seed), amp=amp)
            pte = compose(caches, te_s, cell, cfg, 0.5, key=(cell.cid, "probe_test", amp, seed), amp=amp, role="test")
            idx = _subsample(ptr, pc["max_train_windows"], (cell.cid, "probe_sub", seed))
            E_tr = embeddings(model, ptr.materialize(indices=idx), device)
            E_te = embeddings(model, pte.materialize(), device)
            z_tr, g_tr = ptr.z[idx], ptr.subj[idx]
            pr = fit_probe(E_tr, z_tr, g_tr, pc["Cs"], pc["inner_folds"])
            zhat = pr.predict(E_te)
            cnt = confusion_by_subject(pte.z, zhat, pte.subj)
            r = {"D": D_from_ba(ba_from_counts(cnt)), "C": pr.chosen_C_, "counts": cnt}
            if abs(amp - 1.0) < 1e-9 or abs(amp - cell.amp) < 1e-9:
                ws = within_subject_D(E_te, pte.z, pte.subj, pr.chosen_C_, pc["within_subject_folds"])
                r["within_subject"] = ws
                r["pca"] = {}
                for dim in pc["pca_dims"]:
                    if dim < E_tr.shape[1] and dim < 0.5 * E_tr.shape[0]:
                        pp = fit_probe(E_tr, z_tr, g_tr, pc["Cs"], pc["inner_folds"], pca_dim=dim)
                        r["pca"][dim] = probe_D_safe(pp, E_te, pte.z)
            rec["probe"][f"{amp:g}"] = r

    if cell.p is None:
        rec["seconds"] = time.time() - t0
        write_json(out_path, rec)
        return rec

    # ------------------------------------------------------------ dependence: erasure (3.6.2)
    W = model.head.weight.detach().cpu().numpy(); b = model.head.bias.detach().cpu().numpy()
    if "erasure" in metrics and cell.p in cfg["erasure"]["p_levels"]:
        idx = _subsample(tr, cfg["probe"]["max_train_windows"], (cell.cid, "erase_sub", seed))
        H_tr = embeddings(model, X_tr[idx], device)
        H_te = embeddings(model, base.materialize(), device)
        r_tr, means = residual_concept(tr.z[idx], tr.y[idx])
        er = fit_leace(H_tr, r_tr, cfg["erasure"]["reg"])
        keep = delta_ba_keep(er, W, b, H_te, base.y)
        pe = head_predict(W, b, er(H_te))
        keep["counts_erased"] = counts3(base.y, pe, base.subj, half)
        keep["counts_head"] = counts3(base.y, head_predict(W, b, H_te), base.subj, half)
        keep["within_subject"] = within_subject_keep(H_te, base.z, base.y, base.subj, W, b, cfg["erasure"]["reg"])
        keep["ba_retrain"] = delta_ba_retrain(er, H_tr, tr.y[idx], H_te, base.y)
        keep["angle_deg"] = erasure_task_angle(er, H_tr, W)
        keep["cond_means"] = {str(k): v for k, v in means.items()}
        keep["unidentifiable"] = bool(cell.p >= 1.0)
        rec["erasure"] = keep

    # ------------------------------------------------------------ dependence: interventions
    pad, L, sf = base.pad, base.core_len, cfg["streams"]["model"]["sfreq"]
    subj_list = np.unique(base.subj)
    if "sri" in metrics or "heog" in metrics:
        ic = cfg["intervention"]
        post = ch_index(ch, cfg["alpha_component"]["posterior"]); cent = ch_index(ch, cfg["alpha_component"]["central"])
        conds = {"post_alpha": (post, ic["alpha_band"]), "post_ctrl": (post, ic["control_band"]),
                 "central_alpha": (cent, ic["alpha_band"]), "central_ctrl": (cent, ic["control_band"])}
        regs = {}
        if "heog" in metrics:
            idx = _subsample(tr, 600, (cell.cid, "heog_sub", seed))
            Xp_tr = tr.materialize(padded=True, indices=idx)
            meth = cfg["intervention"].get("heog_method", "saccade_locked")
            regs = {c: fit_heog_regression(Xp_tr, ch, c, meth, sf, cfg) for c in cfg["design"]["regressor_contamination"]}
            if meth != "ols":                                  # OLS kept as a sensitivity analysis
                regs["ols"] = fit_heog_regression(Xp_tr, ch, 0.0, "ols")
            del Xp_tr
        sri_pred = {k: np.empty(len(base), dtype=int) for k in conds}
        reg_pred = {c: np.empty(len(base), dtype=int) for c in regs}
        for s in subj_list:
            ii = np.where(base.subj == s)[0]
            Xp = base.materialize(padded=True, indices=ii)
            if "sri" in metrics:
                for k, (ci, band) in conds.items():
                    sri_pred[k][ii] = predict(crop(bandstop_channels(Xp, sf, ci, band, ic["transition_hz"]), pad, L))
            for c, reg in regs.items():
                reg_pred[c][ii] = predict(crop(apply_heog_regression(Xp, ch, reg), pad, L))
        if "sri" in metrics:
            rec["sri"] = {"counts": {k: counts3(base.y, v, base.subj, half) for k, v in sri_pred.items()}}
            cb = {k: ba_from_counts(v["all"]) for k, v in rec["sri"]["counts"].items()}
            rec["sri"].update({"ba_" + k: v for k, v in cb.items()})
            rec["sri"]["sri_post"] = cb["post_ctrl"] - cb["post_alpha"]
            rec["sri"]["sri_central"] = cb["central_ctrl"] - cb["central_alpha"]
            rec["sri"]["drop_post_alpha"] = rec["truth"]["ba_matched"] - cb["post_alpha"]
            rec["sri"]["drop_central_alpha"] = rec["truth"]["ba_matched"] - cb["central_alpha"]
        if regs:
            rec["heog"] = {}
            for c, v in reg_pred.items():
                cnt = counts3(base.y, v, base.subj, half)
                rec["heog"][c if isinstance(c, str) else f"{c:g}"] = {"counts": cnt, "method": regs[c]["method"],
                                         "delta_ba_reg": rec["truth"]["ba_matched"] - ba_from_counts(cnt["all"])}

    # ------------------------------------------------------------ dependence: IG (3.6.2)
    if "ig" in metrics and cell.p in cfg["ig"]["p_levels"]:
        igc = cfg["ig"]
        ii = _stratified_sample(base.subj, igc["windows_per_unit"], (cell.cid, "ig", seed))
        Xi = base.materialize(indices=ii)
        tgt_ch, tgt_band = ig_targets(cfg, cell.confound)
        rec["ig"] = {}
        for bl in (["band_removed", "zero"] if igc["zero_baseline"] else ["band_removed"]):
            names, R, fx = ig_filterbank(model, Xi, sf, igc["bands"], igc["steps"], bl, device,
                                         batch=ig_batch_for(cfg, unit.model))
            summ = relevance_summary(R, fx, names, ch, tgt_ch, tgt_band)
            ci = ch_index(ch, tgt_ch); k = names.index(tgt_band)
            per = np.stack([np.abs(R[:, ci, k]).sum(1), np.abs(R).sum((1, 2)), np.abs(fx)], 1)
            hh = half[ii]
            summ["sums"] = {"all": sums_by_subject(per, base.subj[ii]),          # [target, total, |f|, n]
                            "half0": sums_by_subject(per[hh == 0], base.subj[ii][hh == 0]),
                            "half1": sums_by_subject(per[hh == 1], base.subj[ii][hh == 1])}
            summ["mean_relevance"] = np.abs(R).mean(0).tolist()   # (C, K) for topography figures
            summ["bands"] = names
            rec["ig"][bl] = summ

    rec["seconds"] = time.time() - t0
    write_json(out_path, rec)
    log.info(f"{unit.uid}: ΔBA_neu={rec['truth'].get('delta_ba_neu', float('nan')):.3f} "
             f"({rec['seconds']:.0f}s)")
    return rec


def probe_D_safe(pipe, E, z):
    from ..metrics.probe import probe_D
    try:
        return probe_D(pipe, E, z)
    except ValueError:
        return float("nan")
