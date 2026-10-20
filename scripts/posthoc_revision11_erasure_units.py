"""Post hoc (exploratory; fourth review, point 12): an identified erasure variant.  The registered residual erasure
fits LEACE on training embeddings to remove the part of z not explained by y; at p = 0.9 (corr(z, y) = 0.8) that part
carries little variance, so the metric underestimates reliance by construction.  Variant: fit LEACE for z itself on
embeddings of a separately composed training set with p = 0.5 (z independent of y, cell amplitude), where removing
z removes no label information, and apply it to the test embeddings.
Recorded per unit (frozen training, composition and metric code; p3audit unchanged):
  truth counts (matched / neutral / reversed, window-parity halves) as in a unit record;
  registered residual erasure (counts_head, counts_erased) exactly as in run_unit;
  variant: counts of the head on z-erased embeddings for the matched and the neutral test set (counts_erased05_matched,
  counts_erased05_neutral).
Design: p = 0.9 factorial (3 task strengths x 3 amplitudes), both confounds, four core models, seed 40: 72 units.
Needs a GPU (training as in the registered units; probe, interventions and IG are skipped, so about 3 min per unit).
Output: results/units_erasure05/<uid>.json.  Resumable.
Usage (repo root): python scripts/posthoc_revision11_erasure_units.py [--confound alpha|saccade|both] [--limit N] [--max-epochs E] [--out DIR]
Dry run:           python scripts/posthoc_revision11_erasure_units.py --confound saccade --limit 1 --max-epochs 1 --out units_erasure05_dryrun
"""
import argparse, json, sys, time, traceback
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
SEEDS = [40]
MODELS = ["csoanet", "eegnet", "shallow", "cbramod"]


def run(ctx, cfg, unit, out_dir, max_epochs, device):
    from p3audit.data.windows import half_assignment
    from p3audit.generator.design import test_couplings
    from p3audit.generator.semisynth import compose
    from p3audit.metrics.truth import unit_truth
    from p3audit.metrics.erasure import fit_leace, head_predict, residual_concept
    from p3audit.experiments.unit import build_for_unit, counts3, _subsample
    from p3audit.training.trainer import embeddings, predict_logits, train_model
    out = ctx.results / out_dir / f"{unit.uid}.json"
    if out.exists():
        return
    t0 = time.time(); cell, seed = unit.cell, unit.seed; sp = ctx.require_split()
    tr_s, va_s, te_s = ctx.usable(sp.train), ctx.usable(sp.val), ctx.usable(sp.test); caches = ctx.caches(tr_s + va_s + te_s)
    tr = compose(caches, tr_s, cell, cfg, cell.p, key=(cell.cid, "train", seed))
    va = compose(caches, va_s, cell, cfg, cell.p, key=(cell.cid, "val", seed))
    X_tr, X_va = tr.materialize(), va.materialize()
    model, _ = build_for_unit(ctx, unit, False)
    rec = {"uid": unit.uid, "model": unit.model, "cell": cell.as_dict(), "cid": cell.cid, "seed": seed, "config_hash": cfg.hash(), "device": str(device)}
    rec["training"] = train_model(model, X_tr, tr.y, X_va, va.y, cfg, seed, device, max_epochs, False); model.eval()
    tests = {k: compose(caches, te_s, cell, cfg, q, key=(cell.cid, "test", k, seed), role="test") for k, q in test_couplings(cell.p).items()}
    base = tests["matched"]; half = half_assignment(base.wids, cfg["seeds"]["half_split"])
    preds, counts, H = {}, {}, {}
    for k, cs in tests.items():
        Xk = cs.materialize(); preds[k] = predict_logits(model, Xk, device).argmax(1); counts[k] = counts3(cs.y, preds[k], cs.subj, half)
        if k in ("matched", "neutral"):
            H[k] = embeddings(model, Xk, device)
    rec["truth"] = unit_truth(base.y, preds, base.subj, half); rec["truth_counts"] = counts
    W = model.head.weight.detach().cpu().numpy(); b = model.head.bias.detach().cpu().numpy()
    # registered residual erasure (as in run_unit)
    idx = _subsample(tr, cfg["probe"]["max_train_windows"], (cell.cid, "erase_sub", seed))
    H_tr = embeddings(model, X_tr[idx], device); del X_tr, X_va
    r_tr, _ = residual_concept(tr.z[idx], tr.y[idx]); er = fit_leace(H_tr, r_tr, cfg["erasure"]["reg"])
    rec["erasure"] = {"counts_head": counts3(base.y, head_predict(W, b, H["matched"]), base.subj, half),
                      "counts_erased": counts3(base.y, head_predict(W, b, er(H["matched"])), base.subj, half)}
    # variant: LEACE for z on a p = 0.5 training composition at the cell amplitude
    p5 = compose(caches, tr_s, cell, cfg, 0.5, key=(cell.cid, "erase05_train", seed), amp=float(cell.amp))
    j = _subsample(p5, cfg["probe"]["max_train_windows"], (cell.cid, "erase05_sub", seed))
    H5 = embeddings(model, p5.materialize(indices=j), device)
    er5 = fit_leace(H5, p5.z[j].astype(float), cfg["erasure"]["reg"])
    neu = tests["neutral"]
    rec["erasure05"] = {"counts_erased05_matched": counts3(base.y, head_predict(W, b, er5(H["matched"])), base.subj, half),
                        "counts_head_neutral": counts3(neu.y, head_predict(W, b, H["neutral"]), neu.subj, half),
                        "counts_erased05_neutral": counts3(neu.y, head_predict(W, b, er5(H["neutral"])), neu.subj, half),
                        "n_fit_windows": int(len(j))}
    rec["seconds"] = time.time() - t0
    out.parent.mkdir(parents=True, exist_ok=True); json.dump(rec, open(out, "w", encoding="utf-8"))
    print("done", unit.uid, "dBA_neu %.3f" % rec["truth"].get("delta_ba_neu", float("nan")), "%.0fs" % rec["seconds"], flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--confound", default="both", choices=["alpha", "saccade", "both"])
    ap.add_argument("--limit", type=int, default=None); ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--out", default="units_erasure05"); ap.add_argument("--allow-cpu", action="store_true", help="execution check only"); ap.add_argument("--config", default="configs/default.yaml")
    a = ap.parse_args()
    import torch
    from p3audit.config import load_config
    from p3audit.experiments.context import Context
    from p3audit.generator.design import Cell, Unit
    from p3audit.training.trainer import resolve_device
    cfg = load_config(a.config); print("config hash", cfg.hash(), flush=True); ctx = Context.create(cfg); d = cfg["design"]
    device = resolve_device(cfg["training"]["device"])
    if device.type != "cuda" and not a.allow_cpu:
        print("CUDA unavailable; stopping (these units must run on the GPU)", flush=True); sys.exit(2)
    confs = ["saccade", "alpha"] if a.confound == "both" else [a.confound]
    units = [Unit(m, Cell(c, d["factorial_p"], s, float(am), "factorial"), sd)
             for c in confs for s in d["factorial_s"] for am in d["factorial_amp"] for m in MODELS for sd in SEEDS]
    if a.limit:
        units = units[:a.limit]
    print(len(units), "units", flush=True)
    for u in units:
        try:
            run(ctx, cfg, u, a.out, a.max_epochs, device)
        except Exception as e:
            (ctx.results / (a.out + "_failed")).mkdir(parents=True, exist_ok=True)
            (ctx.results / (a.out + "_failed") / f"{u.uid}.txt").write_text(traceback.format_exc(), encoding="utf-8")
            print("FAILED", u.uid, e, flush=True)


if __name__ == "__main__":
    main()
