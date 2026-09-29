"""Amendment 2 (OSF): additional real-data arms (needs a GPU).  Outside the frozen package.

Same folds, seeds (0-2), early-stopping subjects and training settings as the registered real audit
and as scripts/posthoc_revision3_controls.py.  Trial-level correctness is written in the format of
results/real/*_trials.csv so that scripts/amendment2_real.py can compare the arms.

  frontal_loc  decoder that sees only Fp1, Fp2, AF7, AF8, F7, F8, re-referenced to the mean of these six
               channels (removes the common-average component that carries every other channel);
               all other channels set to 0
  plc_shuf     placebo control: the AF7 - AF8 signal of ANOTHER trial of the same subject is regressed out
               of all channels with the OLS coefficients of heog_ols (same spatial pattern and spectrum,
               no trial-specific information)
  plc_CP3CP4   placebo control: CP3 - CP4 regressed out of all channels by OLS (lateralised sensorimotor
               pair with low ocular loading in the exploratory scalp profile)
  main_late    main-style model whose input is set to 0 from 0 to 1.5 s after the cue
  heog_late    heog_ols control (AF7 - AF8 regressed out by OLS) with the same 0-1.5 s mask

Regression coefficients are fitted on training-fold trials only.  Runs resume: finished
(arm, model, fold, seed) files are skipped.  Output: results/real_a2/ (or --out).
Usage (repository root):
  python scripts/amendment2_controls.py --models csoanet eegnet shallow cbramod
  dry run:  python scripts/amendment2_controls.py --models eegnet --seeds 0 --folds 0 --max-epochs 1 --out results/real_a2_dryrun
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from posthoc_revision3_controls import FRONTAL, fit_pair_regression, apply_pair_regression  # noqa: E402

ARMS = ["frontal_loc", "plc_shuf", "plc_CP3CP4", "main_late", "heog_late"]
LATE_MASK_S = 1.5
SHUF_SEED = 20261001


def frontal_local(Xp, ch):
    keep = np.zeros(len(ch), bool); keep[[ch.index(c) for c in FRONTAL]] = True
    out = np.zeros_like(Xp)
    sub = Xp[:, keep, :]
    out[:, keep, :] = sub - sub.mean(1, keepdims=True)
    return out


def shuffled_partner(subj, seed=SHUF_SEED):
    """For every trial, the index of another trial of the same subject (random derangement)."""
    rng = np.random.default_rng(seed); partner = np.arange(len(subj))
    for s in np.unique(subj):
        ii = np.where(subj == s)[0]
        if len(ii) < 2:
            continue
        order = ii[rng.permutation(len(ii))]                  # random order, then each trial takes the next one:
        partner[order] = order[np.r_[1:len(ii), 0]]           # a cyclic shift, so no trial is its own partner
    return partner


def apply_shuffled_regression(Xp, ch, reg, subj):
    ia, ib = ch.index("AF7"), ch.index("AF8")
    r = Xp[:, ia, :] - Xp[:, ib, :]
    r = r - r.mean(-1, keepdims=True)
    r = r[shuffled_partner(subj)]
    return (Xp - reg["b"][None, :, None] * r[:, None, :]).astype(np.float32)


def late_mask(Xp, pad, sfreq):
    out = Xp.copy(); out[..., pad:pad + int(round(LATE_MASK_S * sfreq))] = 0.0
    return out


def transform(arm, TR, ES, TE, ch, sfreq):
    D = (TR, ES, TE)
    if arm == "frontal_loc":
        return (*[frontal_local(d["X"], ch) for d in D], {"channels": FRONTAL, "reference": "local mean of the six"})
    if arm == "main_late":
        return (*[late_mask(d["X"], d["pad"], sfreq) for d in D], {"mask_s": [0, LATE_MASK_S]})
    if arm == "heog_late":
        reg = fit_pair_regression(TR["X"], ch, ("AF7", "AF8"))
        return (*[late_mask(apply_pair_regression(d["X"], ch, reg), d["pad"], sfreq) for d in D],
                {"pair": ["AF7", "AF8"], "mask_s": [0, LATE_MASK_S], "b_norm": float(np.linalg.norm(reg["b"]))})
    if arm == "plc_CP3CP4":
        reg = fit_pair_regression(TR["X"], ch, ("CP3", "CP4"))
        return (*[apply_pair_regression(d["X"], ch, reg) for d in D], {"pair": ["CP3", "CP4"], "b_norm": float(np.linalg.norm(reg["b"]))})
    if arm == "plc_shuf":
        reg = fit_pair_regression(TR["X"], ch, ("AF7", "AF8"))
        return (*[apply_shuffled_regression(d["X"], ch, reg, d["subj"]) for d in D],
                {"pair": ["AF7", "AF8"], "shuffled_within_subject": True, "shuf_seed": SHUF_SEED})
    raise ValueError(arm)


def run_arm(cfg, ch, TR, ES, TE, arm, model_name, seed, fold_id, device, max_epochs=None):
    from p3audit.metrics.intervention import crop
    from p3audit.models.registry import build_model
    from p3audit.training.trainer import predict_logits, train_model
    from p3audit.utils.common import balanced_accuracy
    pad, L = TE["pad"], TE["core_len"]
    Xtr, Xes, Xte, info = transform(arm, TR, ES, TE, ch, cfg["streams"]["model"]["sfreq"])
    m = build_model(model_name, cfg, len(ch), L, ch)
    t0 = time.time()
    tinfo = train_model(m, crop(Xtr, pad, L), TR["y"], crop(Xes, pad, L), ES["y"], cfg, seed, device, max_epochs, verbose=False)
    m.eval()
    pred = predict_logits(m, crop(Xte, pad, L), device).argmax(1)
    trials = pd.DataFrame({"subject": TE["subj"], "trial": TE["trial"], "run": TE["run"], "type": TE["type"],
                           "label": TE["y"], "correct": (pred == TE["y"]).astype(int), "model": model_name,
                           "seed": seed, "fold": fold_id, "arm": arm})
    rec = {"arm": arm, "model": model_name, "seed": seed, "fold": fold_id, "ba": float(balanced_accuracy(TE["y"], pred)),
           "training": {k: tinfo[k] for k in ("best_val_ba", "epochs") if k in tinfo}, "seconds": time.time() - t0, **info}
    return rec, trials


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    ap.add_argument("--models", nargs="+", default=["csoanet", "eegnet", "shallow", "cbramod"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--folds", nargs="+", type=int, default=None)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from p3audit.config import load_config
    from p3audit.data.splits import real_audit_folds
    from p3audit.experiments.context import Context
    from p3audit.experiments.real_audit import _stack
    from p3audit.training.trainer import resolve_device
    cfg = load_config(a.config)
    ctx = Context.create(cfg)
    sp, ch = ctx.split, list(ctx.ch_names)
    included = sorted(set(sp.train) | set(sp.val) | set(sp.test))
    folds = real_audit_folds(included, sp.pilot, cfg["split"]["real_audit_folds"], cfg["seeds"]["split"])
    device = resolve_device(cfg["training"]["device"])
    out = Path(a.out) if a.out else ctx.results / "real_a2"; out.mkdir(parents=True, exist_ok=True)
    for f in folds:
        if a.folds is not None and f["fold"] not in a.folds:
            continue
        nonpilot = [s for s in f["train"] if s not in sp.pilot]          # identical to run_real_fold
        es_s = nonpilot[: max(2, len(nonpilot) // 10)]
        tr_s = [s for s in f["train"] if s not in es_s]
        TR, ES, TE = _stack(ctx, tr_s), _stack(ctx, es_s), _stack(ctx, f["test"])
        for arm in a.arms:
            for mdl in a.models:
                for sd in a.seeds:
                    p = out / f"{arm}__{mdl}__fold{f['fold']}__seed{sd}.json"
                    if p.exists():
                        continue
                    rec, trials = run_arm(cfg, ch, TR, ES, TE, arm, mdl, sd, f["fold"], device, a.max_epochs)
                    trials.to_csv(p.with_name(p.stem + "_trials.csv"), index=False)
                    p.write_text(json.dumps(rec, indent=1))
                    print(f"{arm:11s} {mdl:8s} fold {f['fold']} seed {sd}: BA {rec['ba']:.3f} ({rec['seconds'] / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
