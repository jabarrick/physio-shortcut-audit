"""Post hoc (exploratory, NOT registered; requested by the second critical review): semi-synthetic units with WEAK
confounds.  The registered design used confound amplitudes 0.5, 1 and 1.5 a*; here the p = 0.9 factorial is repeated
at 0.125 and 0.25 a* (3 task strengths x 2 amplitudes), both confounds, four core models, new seeds 20 and 21:
96 units.  Uses the frozen unit code (p3audit.experiments.unit.run_unit) and configuration unchanged.
Output: results/units_weak/<uid>.json.  Resumable.  Needs a GPU (about 8-10 hours on the study laptop).
Usage (repo root):  python scripts/posthoc_revision7_weak_units.py [--confound saccade|alpha|both] [--shard i/n]
Dry run:            python scripts/posthoc_revision7_weak_units.py --confound saccade --limit 1 --max-epochs 1 --out units_weak_dryrun
"""
import argparse, sys, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
SEEDS = [20, 21]
AMPS = [0.125, 0.25]
MODELS = ["csoanet", "eegnet", "shallow", "cbramod"]


def unit_list(cfg, confounds):
    from p3audit.generator.design import Cell, Unit
    d = cfg["design"]
    return [Unit(m, Cell(c, d["factorial_p"], s, float(a), "factorial"), sd)
            for c in confounds for s in d["factorial_s"] for a in AMPS for m in MODELS for sd in SEEDS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confound", default="both", choices=["saccade", "alpha", "both"])
    ap.add_argument("--shard", default=None, help="i/n")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--out", default="units_weak")
    ap.add_argument("--config", default="configs/default.yaml")
    a = ap.parse_args()
    from p3audit.config import load_config
    from p3audit.experiments.context import Context
    from p3audit.experiments.unit import run_unit
    cfg = load_config(a.config); print("config hash", cfg.hash())
    ctx = Context.create(cfg)
    units = unit_list(cfg, ["saccade", "alpha"] if a.confound == "both" else [a.confound])
    if a.shard:
        i, n = map(int, a.shard.split("/")); units = units[i::n]
    if a.limit:
        units = units[:a.limit]
    print(len(units), "units")
    for u in units:
        try:
            run_unit(ctx, u, out_dir=a.out, max_epochs=a.max_epochs)
            print("done", u.uid, flush=True)
        except Exception as e:
            (ctx.results / (a.out + "_failed")).mkdir(parents=True, exist_ok=True)
            (ctx.results / (a.out + "_failed") / f"{u.uid}.txt").write_text(traceback.format_exc(), encoding="utf-8")
            print("FAILED", u.uid, e, flush=True)


if __name__ == "__main__":
    main()
