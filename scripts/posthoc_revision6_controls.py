"""Post hoc (exploratory, NOT registered; after the second critical review, 30 Sep 2026): real-data arms that
test whether the decoders' cue-related advantage is ocular or a stimulus-position (neural) signal.  Needs a GPU.
Outside the frozen package; same folds, seeds (0-2), early-stopping subjects and training settings as
scripts/amendment2_controls.py, whose helpers it reuses.  Output: results/real_r6/ (or --out), same record format.

  post_loc          six posterior channels O1, O2, PO7, PO8, PO3, PO4, re-referenced to their own mean
  post_loc_late     post_loc with the input set to 0 from 0 to 1.5 s after the cue
  cent_loc          six central channels FC3, FC4, C3, C4, CP3, CP4, re-referenced to their own mean
                    (control decoder with the same number of channels as the frontal one)
  frontal_loc_late  frontal_loc of Amendment 2 with the input set to 0 from 0 to 1.5 s after the cue
  frontal_loc_heog  AF7 - AF8 regressed out of all channels by OLS (heog_ols), then frontal_loc
  plc_orth          null control: the trial's own AF7 - AF8 signal is subtracted along a random spatial pattern
                    orthogonal to the heog_ols coefficient vector b and of the same norm (seed 20261007);
                    removes nothing ocular and adds variance of matched size outside the ocular pattern
  plc_rev           control: the trial's own AF7 - AF8 signal, reversed in time, is subtracted with the
                    heog_ols coefficients (same spatial pattern and spectrum, no time alignment)

Regression coefficients are fitted on training-fold trials only.  Runs resume (finished files are skipped).
Suggested order (fast models first):
  python scripts/posthoc_revision6_controls.py --models csoanet eegnet
  python scripts/posthoc_revision6_controls.py --models shallow cbramod
  dry run: python scripts/posthoc_revision6_controls.py --models eegnet --seeds 0 --folds 0 --max-epochs 1 --out results/real_r6_dryrun
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from posthoc_revision3_controls import FRONTAL, fit_pair_regression, apply_pair_regression  # noqa: E402
import amendment2_controls as A2  # noqa: E402

ARMS = ["post_loc", "post_loc_late", "cent_loc", "frontal_loc_late", "frontal_loc_heog", "plc_orth", "plc_rev"]
POSTERIOR = ["O1", "O2", "PO7", "PO8", "PO3", "PO4"]
CENTRAL = ["FC3", "FC4", "C3", "C4", "CP3", "CP4"]
ORTH_SEED = 20261007


def local_ref(Xp, ch, names):
    keep = np.zeros(len(ch), bool); keep[[ch.index(c) for c in names]] = True
    out = np.zeros_like(Xp); sub = Xp[:, keep, :]
    out[:, keep, :] = sub - sub.mean(1, keepdims=True)
    return out


def own_signal(Xp, ch):
    r = Xp[:, ch.index("AF7"), :] - Xp[:, ch.index("AF8"), :]
    return r - r.mean(-1, keepdims=True)


def orth_pattern(b, seed=ORTH_SEED):
    q = np.random.default_rng(seed).standard_normal(b.shape)
    bn = b / (np.linalg.norm(b) + 1e-12); q = q - (q @ bn) * bn
    return q / (np.linalg.norm(q) + 1e-12) * np.linalg.norm(b)


def transform(arm, TR, ES, TE, ch, sfreq):
    D = (TR, ES, TE); late = lambda X, d: A2.late_mask(X, d["pad"], sfreq)
    if arm == "post_loc":
        return (*[local_ref(d["X"], ch, POSTERIOR) for d in D], {"channels": POSTERIOR, "reference": "local"})
    if arm == "post_loc_late":
        return (*[late(local_ref(d["X"], ch, POSTERIOR), d) for d in D], {"channels": POSTERIOR, "reference": "local", "mask_s": [0, A2.LATE_MASK_S]})
    if arm == "cent_loc":
        return (*[local_ref(d["X"], ch, CENTRAL) for d in D], {"channels": CENTRAL, "reference": "local"})
    if arm == "frontal_loc_late":
        return (*[late(local_ref(d["X"], ch, FRONTAL), d) for d in D], {"channels": FRONTAL, "reference": "local", "mask_s": [0, A2.LATE_MASK_S]})
    reg = fit_pair_regression(TR["X"], ch, ("AF7", "AF8"))
    if arm == "frontal_loc_heog":
        return (*[local_ref(apply_pair_regression(d["X"], ch, reg), ch, FRONTAL) for d in D],
                {"channels": FRONTAL, "reference": "local", "after": "heog_ols", "b_norm": float(np.linalg.norm(reg["b"]))})
    if arm == "plc_orth":
        q = orth_pattern(np.asarray(reg["b"], float))
        f = lambda X: (X - q[None, :, None] * own_signal(X, ch)[:, None, :]).astype(np.float32)
        return (*[f(d["X"]) for d in D], {"pattern": "random, orthogonal to b, same norm", "seed": ORTH_SEED,
                                           "cos_q_b": float(q @ reg["b"] / (np.linalg.norm(q) * np.linalg.norm(reg["b"]) + 1e-12))})
    if arm == "plc_rev":
        f = lambda X: (X - np.asarray(reg["b"])[None, :, None] * own_signal(X, ch)[:, None, ::-1]).astype(np.float32)
        return (*[f(d["X"]) for d in D], {"pair": ["AF7", "AF8"], "time_reversed": True})
    raise ValueError(arm)


A2.transform = transform          # run_arm of amendment2_controls calls transform() from its own module


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
    cfg = load_config(a.config); ctx = Context.create(cfg)
    sp, ch = ctx.split, list(ctx.ch_names)
    included = sorted(set(sp.train) | set(sp.val) | set(sp.test))
    folds = real_audit_folds(included, sp.pilot, cfg["split"]["real_audit_folds"], cfg["seeds"]["split"])
    device = resolve_device(cfg["training"]["device"])
    out = Path(a.out) if a.out else ctx.results / "real_r6"; out.mkdir(parents=True, exist_ok=True)
    for f in folds:
        if a.folds is not None and f["fold"] not in a.folds:
            continue
        nonpilot = [s for s in f["train"] if s not in sp.pilot]
        es_s = nonpilot[: max(2, len(nonpilot) // 10)]
        tr_s = [s for s in f["train"] if s not in es_s]
        TR, ES, TE = _stack(ctx, tr_s), _stack(ctx, es_s), _stack(ctx, f["test"])
        for arm in a.arms:
            for mdl in a.models:
                for sd in a.seeds:
                    p = out / f"{arm}__{mdl}__fold{f['fold']}__seed{sd}.json"
                    if p.exists():
                        continue
                    rec, trials = A2.run_arm(cfg, ch, TR, ES, TE, arm, mdl, sd, f["fold"], device, a.max_epochs)
                    trials.to_csv(p.with_name(p.stem + "_trials.csv"), index=False)
                    p.write_text(json.dumps(rec, indent=1))
                    print(f"{arm:16s} {mdl:8s} fold {f['fold']} seed {sd}: BA {rec['ba']:.3f} ({rec['seconds'] / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
