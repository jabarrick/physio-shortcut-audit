"""Amendment 3 (in preparation; not yet registered): semi-synthetic units with real recorded saccades.
Outside the frozen package.  Uses the frozen unit code (p3audit.experiments.unit.run_unit) and the frozen configuration
unchanged; scripts/amendment3_common.py replaces, in memory only, the composition step (z exact within
subject x label x half in the ground-truth test sets; sham epochs in z = 0 windows) and serves the new bases.
Units per arm: saccade, p = 0.9 factorial cells (3 task strengths x 3 amplitudes), the four core models, seeds 30-35:
216 units.  ARM: RH (primary) or RV (secondary).  Cell.sacc_basis = sacc_<ARM>.
Outputs: results/units_a3_<ARM>/<uid>.json (written by the frozen run_unit) and <uid>.preds.npz: the per-window
predictions behind every confusion count of the record (ground-truth test sets, erasure, SRI, HEOG regression),
with labels, subjects, halves and z, so that any later analysis needs no retraining.
Requires: cache_a3/ (scripts/amendment3_build_bases.py --arm ARM) and results/a3/sham_gain_<ARM>.json
(scripts/amendment3_check_balance.py --arm ARM --calibrate).
Usage (repo root):
  dry run   python scripts/amendment3_run_units.py --arm RH --models eegnet --seeds 30 --one --max-epochs 1 --out results/units_a3_RH_dryrun
  full run  python scripts/amendment3_run_units.py --arm RH [--shard i/n]
"""
import argparse, json, sys, traceback
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
SEEDS = list(range(30, 36)); MODELS = ["csoanet", "eegnet", "shallow", "cbramod"]


def unit_list(cfg, arm, models, seeds, one=False):
    from p3audit.generator.design import Cell, Unit
    d = cfg["design"]
    cells = [Cell("saccade", d["factorial_p"], s, float(a), "factorial", sacc_basis=f"sacc_{arm}") for s in d["factorial_s"] for a in d["factorial_amp"]]
    if one:
        cells = [c for c in cells if c.s == "s_star" and c.amp == 1.0]
    return [Unit(m, c, sd) for c in cells for m in models for sd in seeds]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["RH", "RV"], required=True); ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS); ap.add_argument("--one", action="store_true", help="only the s_star, 1 a* cell")
    ap.add_argument("--max-epochs", type=int, default=None); ap.add_argument("--shard", default=None); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from p3audit.config import load_config
    from p3audit.experiments.context import Context
    import p3audit.experiments.unit as U
    import amendment3_common as A
    gain_file = Path("results/a3") / f"sham_gain_{a.arm}.json"
    if not gain_file.exists():
        raise SystemExit(f"missing {gain_file}: run scripts/amendment3_check_balance.py --arm {a.arm} --calibrate first")
    cfg = load_config("configs/default.yaml"); print("config hash", cfg.hash(), "| sham gain", A.sham_gain(a.arm))
    ctx = Context.create(cfg); A.install(ctx, a.arm)
    out_dir = Path(a.out) if a.out else Path("results") / f"units_a3_{a.arm}"; rel = out_dir.relative_to("results") if out_dir.parts[0] == "results" else out_dir
    captured = []
    orig_counts3 = U.counts3

    def counts3_capture(y, pred, subj, half):
        c = orig_counts3(y, pred, subj, half); captured.append((c, np.asarray(pred, dtype=np.int8))); return c
    U.counts3 = counts3_capture
    units = unit_list(cfg, a.arm, a.models, a.seeds, a.one)
    if a.shard:
        i, n = map(int, a.shard.split("/")); units = units[i::n]
    print(len(units), "units ->", out_dir)
    for u in units:
        js = out_dir / f"{u.uid}.json"
        if js.exists():
            continue
        captured.clear()
        try:
            rec = U.run_unit(ctx, u, out_dir=str(rel), max_epochs=a.max_epochs)
        except Exception as e:
            (Path(str(out_dir) + "_failed")).mkdir(parents=True, exist_ok=True)
            (Path(str(out_dir) + "_failed") / f"{u.uid}.txt").write_text(traceback.format_exc(), encoding="utf-8"); print("FAILED", u.uid, e); continue
        named = {}
        targets = {f"truth_{k}": v for k, v in rec.get("truth_counts", {}).items()}
        if "erasure" in rec:
            targets.update({"erasure_erased": rec["erasure"]["counts_erased"], "erasure_head": rec["erasure"]["counts_head"]})
        targets.update({f"sri_{k}": v for k, v in rec.get("sri", {}).get("counts", {}).items()})
        targets.update({f"heog_{k}": v["counts"] for k, v in rec.get("heog", {}).items()})
        for name, obj in targets.items():
            hit = [p for c, p in captured if c is obj]
            if len(hit) != 1:
                raise RuntimeError(f"{u.uid}: could not match predictions for {name}")
            named[name] = hit[0]
        z = rec["z_test"]
        np.savez_compressed(out_dir / f"{u.uid}.preds.npz", **{f"pred_{k}": v for k, v in named.items()},
                            half=np.asarray(rec["test_half"], dtype=np.int8), **{f"z_{k}": np.asarray(v, dtype=np.int8) for k, v in z.items()},
                            arm=a.arm, sham_gain=A.sham_gain(a.arm))
        print("done", u.uid, f"{rec['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
