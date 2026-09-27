"""Round-3 real-data control models (exploratory; needs a GPU).  Outside the frozen package.

For the same folds, seeds and early-stopping subjects as the real audit, trains extra models and
writes trial-level correctness in the format of results/real/*_trials.csv, so the tercile analysis
(scripts/posthoc_revision3_terciles.py) can compare every arm with the main models:

  frontal    main-style model that sees only Fp1, Fp2, AF7, AF8, F7, F8 (other channels set to 0)
  heog_ols   control model: AF7 - AF8 regressed out of all channels by ordinary least squares
  plc_P7P8   placebo control: P7 - P8 regressed out the same way (no ocular source expected)
  plc_O1O2   placebo control: O1 - O2 regressed out the same way

heog_ols uses the same OLS estimator as the placebo arms, so the placebo comparison is like for like
(the registered control models used saccade-locked coefficients).  Regression coefficients are fitted
on the training-fold trials only.

Usage (from the repository root, same environment as the real audit):
  python scripts/posthoc_revision3_controls.py --arms frontal heog_ols plc_P7P8 plc_O1O2 --models csoanet eegnet
  python scripts/posthoc_revision3_controls.py --models csoanet eegnet shallow cbramod        # everything
Runs resume: finished (arm, model, fold, seed) files are skipped.  Output: results/real_r3/.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FRONTAL = ["Fp1", "Fp2", "AF7", "AF8", "F7", "F8"]
PAIRS = {"heog_ols": ("AF7", "AF8"), "plc_P7P8": ("P7", "P8"), "plc_O1O2": ("O1", "O2")}
ARMS = ["frontal", *PAIRS]


def fit_pair_regression(Xp, ch, pair, max_trials=600):
    """OLS propagation coefficients of the pair difference signal onto every channel."""
    ia, ib = ch.index(pair[0]), ch.index(pair[1])
    Xs = Xp[:: max(1, len(Xp) // max_trials)]
    r = Xs[:, ia, :] - Xs[:, ib, :]
    r = r - r.mean(-1, keepdims=True)
    Xc = Xs - Xs.mean(-1, keepdims=True)
    b = np.einsum("nct,nt->c", Xc, r) / (np.einsum("nt,nt->", r, r) + 1e-12)
    return {"pair": pair, "b": b.astype(np.float32)}


def apply_pair_regression(Xp, ch, reg):
    ia, ib = ch.index(reg["pair"][0]), ch.index(reg["pair"][1])
    r = Xp[:, ia, :] - Xp[:, ib, :]
    r = r - r.mean(-1, keepdims=True)
    return (Xp - reg["b"][None, :, None] * r[:, None, :]).astype(np.float32)


def frontal_only(Xp, ch):
    keep = np.zeros(len(ch), bool); keep[[ch.index(c) for c in FRONTAL]] = True
    out = Xp.copy(); out[:, ~keep, :] = 0.0
    return out


def transform(arm, TR, ES, TE, ch):
    if arm == "frontal":
        return frontal_only(TR["X"], ch), frontal_only(ES["X"], ch), frontal_only(TE["X"], ch), {"channels": FRONTAL}
    reg = fit_pair_regression(TR["X"], ch, PAIRS[arm])
    info = {"pair": list(PAIRS[arm]), "b_norm": float(np.linalg.norm(reg["b"]))}
    return (apply_pair_regression(TR["X"], ch, reg), apply_pair_regression(ES["X"], ch, reg),
            apply_pair_regression(TE["X"], ch, reg), info)


def run_arm(cfg, ch, TR, ES, TE, arm, model_name, seed, fold_id, device, max_epochs=None):
    from p3audit.metrics.intervention import crop
    from p3audit.models.registry import build_model
    from p3audit.training.trainer import predict_logits, train_model
    from p3audit.utils.common import balanced_accuracy
    pad, L = TE["pad"], TE["core_len"]
    Xtr, Xes, Xte, info = transform(arm, TR, ES, TE, ch)
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
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--max-epochs", type=int, default=None)
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
    out = ctx.results / "real_r3"; out.mkdir(parents=True, exist_ok=True)
    for f in folds:
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
                    print(f"{arm:9s} {mdl:8s} fold {f['fold']} seed {sd}: BA {rec['ba']:.3f} ({rec['seconds'] / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
