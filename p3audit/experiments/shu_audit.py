"""Supplementary real audit on SHU-MI (3.1, 3.5): 25 subjects × 5 sessions,
32 channels, 250 Hz, left/right grasp imagery, public data already 0.5–40 Hz.

The public release is epoched (verify in P11), so the model-stream filters
are applied per epoch with 1-s reflection padding.  Metrics: posterior-alpha
SRI, alpha erasure (z proxy), IG (posterior alpha; frontal-lateral delta with
F7/F8 as a HEOG proxy).  HEOG regression is not run (no AF7/AF8).  LaBraM
here is 5 folds × 1 seed: qualitative reference only, never in comparisons.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly

from ..data.external import load_shu
from ..data.streams import fir
from ..metrics.attribution import ig_filterbank, relevance_summary
from ..metrics.erasure import delta_ba_keep, fit_leace, residual_concept
from ..metrics.intervention import bandstop_channels, crop
from ..metrics.spectral import log_band_power
from ..models.registry import build_model
from ..stats.suffstats import confusion_by_subject
from ..training.trainer import embeddings, predict_logits, resolve_device, train_model
from ..utils.common import balanced_accuracy, ch_index, get_logger, write_json
from .context import Context

log = get_logger("p3audit.shu")
SHU_POSTERIOR = ["P3", "Pz", "P4", "PO3", "PO4", "O1", "Oz", "O2"]
SHU_FRONTAL = ["Fp1", "Fp2", "F7", "F8"]


def shu_subject(ctx: Context, subject: int, n_sessions: int = 5):
    cfg = ctx.cfg
    Xs, ys = [], []
    for ses in range(1, n_sessions + 1):
        e = load_shu(ctx.cfg.resolve_path("shu_root"), subject, ses)
        X = resample_poly(e.X, 4, 5, axis=-1)                       # 250 -> 200 Hz
        pad = int(cfg["streams"]["pad_s"] * 200)
        Xp = np.pad(X, ((0, 0), (0, 0), (pad, pad)), mode="reflect")
        Xp = fir(Xp, 200.0, cfg["streams"]["model"]["l_freq"], None)
        Xp = Xp - Xp.mean(axis=1, keepdims=True)
        Xs.append(Xp.astype(np.float32)); ys.append(e.y)
        ch = e.ch_names
    return np.concatenate(Xs), np.concatenate(ys), pad, X.shape[-1], ch


def run_shu_audit(ctx: Context, models=None, seeds=(0, 1, 2), n_subjects: int = 25, folds: int = 5,
                  device=None, fallback_standin=False, max_epochs=None) -> list[dict]:
    cfg = ctx.cfg
    device = device or resolve_device(cfg["training"]["device"])
    data = {s: shu_subject(ctx, s) for s in range(1, n_subjects + 1)}
    ch = next(iter(data.values()))[4]
    pad, L = next(iter(data.values()))[2], next(iter(data.values()))[3]
    subs = np.random.default_rng(cfg["seeds"]["split"]).permutation(list(data))
    chunks = np.array_split(subs, folds)
    models = models or cfg["design"]["core_models"] + ["labram"]
    post, sf = ch_index(ch, SHU_POSTERIOR), 200.0
    out = []
    for k, test in enumerate(chunks):
        train = [s for s in subs if s not in set(test)]
        es, tr = train[:2], train[2:]
        cat = lambda ss: (np.concatenate([data[s][0] for s in ss]), np.concatenate([data[s][1] for s in ss]),
                          np.concatenate([np.full(len(data[s][1]), s) for s in ss]))
        Xtr, ytr, gtr = cat(tr); Xes, yes, _ = cat(es); Xte, yte, gte = cat(test)
        for m_name in models:
            for sd in (seeds if m_name != "labram" else seeds[:1]):
                m = build_model(m_name, cfg, len(ch), L, ch, fallback_standin=fallback_standin)
                train_model(m, crop(Xtr, pad, L), ytr, crop(Xes, pad, L), yes, cfg, sd, device, max_epochs, verbose=False)
                pred = lambda X: predict_logits(m, X, device).argmax(1)
                p0 = pred(crop(Xte, pad, L))
                rec = {"fold": k, "model": m_name, "seed": sd, "qualitative_only": m_name == "labram",
                       "ba": balanced_accuracy(yte, p0), "counts": confusion_by_subject(yte, p0, gte)}
                for tag, band in (("post_alpha", cfg["intervention"]["alpha_band"]), ("post_ctrl", cfg["intervention"]["control_band"])):
                    rec[f"ba_{tag}"] = balanced_accuracy(yte, pred(crop(bandstop_channels(Xte, sf, post, band, 1.0), pad, L)))
                rec["sri_post"] = rec["ba_post_ctrl"] - rec["ba_post_alpha"]
                a_tr = log_band_power(crop(Xtr, pad, L), sf, cfg["alpha_component"]["band"], post)
                a_te = log_band_power(crop(Xte, pad, L), sf, cfg["alpha_component"]["band"], post)
                thr = np.median(a_tr)
                H_tr, H_te = embeddings(m, crop(Xtr, pad, L), device), embeddings(m, crop(Xte, pad, L), device)
                r_tr, _ = residual_concept((a_tr > thr).astype(int), ytr)
                W, b = m.head.weight.detach().cpu().numpy(), m.head.bias.detach().cpu().numpy()
                rec["alpha_erasure"] = delta_ba_keep(fit_leace(H_tr, r_tr), W, b, H_te, yte)
                idx = np.random.default_rng(sd).permutation(len(yte))[: cfg["ig"]["windows_per_unit"]]
                names, R, fx = ig_filterbank(m, crop(Xte[idx], pad, L), sf, cfg["ig"]["bands"], cfg["ig"]["steps"],
                                             "band_removed", device)
                rec["ig_alpha"] = relevance_summary(R, fx, names, ch, SHU_POSTERIOR, "alpha")
                rec["ig_frontal_delta"] = relevance_summary(R, fx, names, ch, SHU_FRONTAL, "delta")
                out.append(rec)
                write_json(ctx.results / "shu" / f"{m_name}__fold{k}__seed{sd}.json", rec)
    return out
