"""3.7 natural counterfactual (H4) and real audit (H5).

PhysioNet-MI left/right-fist imagery (runs 4/8/12): subject-grouped 5-fold CV
(pilot subjects fixed in training folds), core models × 3 seeds (LaBraM excluded:
EEGMMIDB is in its pre-training corpus).  Per fold:
  * main models on model-stream input  -> trial-level correctness (H4) + metrics (H5)
  * bias-control models (same architecture) trained and tested on HEOG-regressed
    input ("去扫视输入") -> selection-bias estimate for H4
Metrics by identifiability (3.6.2, 九·10):
  horizontal saccade  : interventions (HEOG regression) + attribution only (coupling ≈ 1)
  posterior alpha     : erasure (z proxy = posterior alpha power, median split on the
                        training fold, spectral stream) + SRI + attribution
Only metrics that passed H3/H3r may be interpreted; the analysis step enforces this.
Four-class run confound: not audited (run type and class pair are collinear) — design note only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.splits import real_audit_folds
from ..data.streams import apply_stream, out_index, stream_specs
from ..metrics.attribution import ig_filterbank, relevance_summary
from ..metrics.erasure import delta_ba_keep, fit_leace, residual_concept
from ..metrics.intervention import (apply_heog_regression, bandstop_channels, crop, fit_heog_regression)
from ..metrics.probe import D_from_ba, fit_probe
from ..metrics.spectral import posterior_alpha
from ..models.registry import build_model
from ..saccades.detect import classify_trials
from ..stats.suffstats import confusion_by_subject, sums_by_subject
from ..training.trainer import embeddings, predict_logits, resolve_device, train_model
from ..utils.common import balanced_accuracy, ch_index, get_logger, write_json
from .context import Context

log = get_logger("p3audit.real")


# ------------------------------------------------------------------ data
def real_epochs(ctx: Context, subject: int) -> dict:
    """Padded model-stream epochs around the cue, labels, trial types and alpha z proxy."""
    path = ctx.results / "real_cache" / f"sub-{subject:03d}.npz"
    if path.exists():
        d = np.load(path, allow_pickle=True)
        return {k: d[k] for k in d.files}
    cfg = ctx.cfg
    ctx.guard.record(subject, [4, 8, 12], "real_audit")
    specs = stream_specs(cfg)
    a, b = cfg["real_audit"]["window"]
    Xs, Ss, ys, types, runs = [], [], [], [], []
    for r in (4, 8, 12):
        run = ctx.source.load(subject, r)
        sf = run.sfreq
        Xm, sfm = apply_stream(run.data, sf, specs["model"])
        Xsp, _ = apply_stream(run.data, sf, specs["spectral"])
        trials = classify_trials(run, cfg)
        pad = int(round(cfg["streams"]["pad_s"] * sfm))
        L = int(round((b - a) * sfm))
        for (o, _, s), tr in zip(run.event_samples(("T1", "T2")), trials):
            grid = 4
            o_al = int(np.round((o + a * sf) / grid) * grid)
            i0 = out_index(o_al, sf, specs["model"])
            if i0 - pad < 0 or i0 + L + pad > Xm.shape[1]:
                continue
            Xs.append(Xm[:, i0 - pad:i0 + L + pad].astype(np.float32))
            Ss.append(Xsp[:, i0:i0 + L].astype(np.float32))
            ys.append(0 if s == "T1" else 1); types.append(tr["type"]); runs.append(r)
    X = np.stack(Xs); S = np.stack(Ss)
    alpha = posterior_alpha(S, cfg["streams"]["spectral"]["sfreq"], ctx.ch_names, cfg)
    out = {"X": X, "y": np.array(ys), "type": np.array(types), "run": np.array(runs), "alpha_logpow": alpha,
           "pad": np.array(pad), "core_len": np.array(L)}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    return out


def _stack(ctx, subjects):
    parts = [real_epochs(ctx, s) for s in subjects]
    subj = np.concatenate([np.full(len(p["y"]), s) for s, p in zip(subjects, parts)])
    trial = np.concatenate([np.arange(len(p["y"])) for p in parts])
    cat = {k: np.concatenate([p[k] for p in parts]) for k in ("X", "y", "type", "run", "alpha_logpow")}
    cat.update({"subj": subj, "trial": trial, "pad": int(parts[0]["pad"]), "core_len": int(parts[0]["core_len"])})
    return cat


# ------------------------------------------------------------------ one fold × model × seed
def run_real_fold(ctx: Context, fold: dict, model_name: str, seed: int, device=None, fallback_standin=False,
                  max_epochs=None, with_control: bool = True) -> dict:
    cfg = ctx.cfg
    device = device or resolve_device(cfg["training"]["device"])
    ch = ctx.ch_names
    sp = ctx.split
    nonpilot = [s for s in fold["train"] if s not in sp.pilot]
    es_s = nonpilot[: max(2, len(nonpilot) // 10)]
    tr_s = [s for s in fold["train"] if s not in es_s]
    TR, ES, TE = _stack(ctx, tr_s), _stack(ctx, es_s), _stack(ctx, fold["test"])
    pad, L = TE["pad"], TE["core_len"]
    sf = cfg["streams"]["model"]["sfreq"]

    def fit(Xtr_p, Xes_p):
        m = build_model(model_name, cfg, len(ch), L, ch, fallback_standin=fallback_standin)
        info = train_model(m, crop(Xtr_p, pad, L), TR["y"], crop(Xes_p, pad, L), ES["y"], cfg, seed, device,
                           max_epochs, verbose=False)
        m.eval()
        return m, info

    model, info = fit(TR["X"], ES["X"])
    predict = lambda X: predict_logits(model, X, device).argmax(1)
    pred = predict(crop(TE["X"], pad, L))
    trials = pd.DataFrame({"subject": TE["subj"], "trial": TE["trial"], "run": TE["run"], "type": TE["type"],
                           "label": TE["y"], "correct": (pred == TE["y"]).astype(int), "model": model_name,
                           "seed": seed, "fold": fold["fold"], "arm": "main"})
    rec = {"model": model_name, "seed": seed, "fold": fold["fold"], "training": {k: info[k] for k in ("best_val_ba", "epochs", "seconds")},
           "ba": balanced_accuracy(TE["y"], pred), "counts": confusion_by_subject(TE["y"], pred, TE["subj"])}

    # --- interventions: SRI (posterior alpha) and HEOG regression
    ic = cfg["intervention"]
    post = ch_index(ch, cfg["alpha_component"]["posterior"]); cent = ch_index(ch, cfg["alpha_component"]["central"])
    rec["sri"] = {}
    for tag, (ci, band) in {"post_alpha": (post, ic["alpha_band"]), "post_ctrl": (post, ic["control_band"]),
                            "central_alpha": (cent, ic["alpha_band"]), "central_ctrl": (cent, ic["control_band"])}.items():
        p = predict(crop(bandstop_channels(TE["X"], sf, ci, band, ic["transition_hz"]), pad, L))
        rec["sri"][tag] = confusion_by_subject(TE["y"], p, TE["subj"])
    meth = cfg["intervention"].get("heog_method", "saccade_locked")
    reg = fit_heog_regression(TR["X"][:: max(1, len(TR["X"]) // 600)], ch, 0.0, meth, sf, cfg)
    X_te_clean = apply_heog_regression(TE["X"], ch, reg)
    rec["heog"] = confusion_by_subject(TE["y"], predict(crop(X_te_clean, pad, L)), TE["subj"])

    # --- attribution (both targets)
    idx = np.random.default_rng(seed).permutation(len(TE["y"]))[: cfg["ig"]["windows_per_unit"]]
    names, R, fx = ig_filterbank(model, crop(TE["X"][idx], pad, L), sf, cfg["ig"]["bands"], cfg["ig"]["steps"],
                                 "band_removed", device)
    rec["ig"] = {}
    for tgt, (chs, band) in {"alpha": (cfg["alpha_component"]["posterior"], "alpha"),
                             "saccade": (cfg["ig"]["frontal_lateral"], "delta")}.items():
        s = relevance_summary(R, fx, names, ch, chs, band)
        ci = ch_index(ch, chs); k = names.index(band)
        s["sums"] = sums_by_subject(np.stack([np.abs(R[:, ci, k]).sum(1), np.abs(R).sum((1, 2)), np.abs(fx)], 1),
                                    TE["subj"][idx])
        rec["ig"][tgt] = s

    # --- alpha erasure + encoding with z proxy (posterior alpha, training-fold median split)
    thr = float(np.median(TR["alpha_logpow"]))
    z_tr, z_te = (TR["alpha_logpow"] > thr).astype(int), (TE["alpha_logpow"] > thr).astype(int)
    H_tr = embeddings(model, crop(TR["X"], pad, L), device); H_te = embeddings(model, crop(TE["X"], pad, L), device)
    W = model.head.weight.detach().cpu().numpy(); b = model.head.bias.detach().cpu().numpy()
    r_tr, _ = residual_concept(z_tr, TR["y"])
    rec["alpha_erasure"] = delta_ba_keep(fit_leace(H_tr, r_tr, cfg["erasure"]["reg"]), W, b, H_te, TE["y"])
    pr = fit_probe(H_tr, z_tr, TR["subj"], cfg["probe"]["Cs"], cfg["probe"]["inner_folds"])
    rec["alpha_probe_D"] = D_from_ba(balanced_accuracy(z_te, pr.predict(H_te)))
    rec["alpha_probe_note"] = "z proxy estimated from the same signal as the embedding: D biased upward (十一·2)"

    # --- bias-control model on de-saccaded input
    if with_control:
        reg_tr = reg
        ctrl, _ = fit(apply_heog_regression(TR["X"], ch, reg_tr), apply_heog_regression(ES["X"], ch, reg_tr))
        pc = predict_logits(ctrl, crop(apply_heog_regression(TE["X"], ch, reg_tr), pad, L), device).argmax(1)
        t2 = trials.copy(); t2["correct"] = (pc == TE["y"]).astype(int); t2["arm"] = "control"
        trials = pd.concat([trials, t2])
    return {"record": rec, "trials": trials}


def run_real_audit(ctx: Context, models=None, seeds=(0, 1, 2), **kw) -> dict:
    cfg = ctx.cfg
    sp = ctx.split
    included = sorted(set(sp.train) | set(sp.val) | set(sp.test))
    folds = real_audit_folds(included, sp.pilot, cfg["split"]["real_audit_folds"], cfg["seeds"]["split"])
    models = models or [m for m in cfg["design"]["core_models"] if m != "labram"]
    out_dir = ctx.results / "real"
    out_dir.mkdir(parents=True, exist_ok=True)
    recs, trials = [], []
    for f in folds:
        for m in models:
            for sd in seeds:
                p = out_dir / f"{m}__fold{f['fold']}__seed{sd}.json"
                if p.exists():
                    continue
                r = run_real_fold(ctx, f, m, sd, **kw)
                write_json(p, r["record"])
                r["trials"].to_csv(out_dir / f"{m}__fold{f['fold']}__seed{sd}_trials.csv", index=False)
                recs.append(r["record"]); trials.append(r["trials"])
                log.info(f"real audit {m} fold {f['fold']} seed {sd}: BA {r['record']['ba']:.3f}")
    return {"n_records": len(recs)}
