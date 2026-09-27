"""Amendment 1 (OSF): additional semi-synthetic units for the confirmatory within-cell test.
Outside the frozen package; uses the frozen unit code (p3audit.experiments.unit.run_unit) and the
frozen configuration unchanged.  Units: saccade arm, p = 0.9 factorial cells (3 task strengths x
3 confound amplitudes), the four core models, NEW seeds 10-15 only.  216 units.
Output: results/units_a1/<uid>.json (never results/units).  Resumable.
Usage (repo root):  python watchdog.py --watch results/units_a1 -- ... is not applicable to scripts;
run directly:        python scripts/amendment1_run_units.py [--confound saccade] [--shard i/n]
"""
import argparse, sys, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
NEW_SEEDS = list(range(10, 16))
MODELS = ["csoanet", "eegnet", "shallow", "cbramod"]


def unit_list(cfg, confound):
    from p3audit.generator.design import Cell, Unit
    d = cfg["design"]
    return [Unit(m, Cell(confound, d["factorial_p"], s, float(a), "factorial"), sd)
            for s in d["factorial_s"] for a in d["factorial_amp"] for m in MODELS for sd in NEW_SEEDS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confound", default="saccade", choices=["saccade", "alpha"])
    ap.add_argument("--shard", default=None, help="i/n")
    ap.add_argument("--config", default="configs/default.yaml")
    a = ap.parse_args()
    from p3audit.config import load_config
    from p3audit.experiments.context import Context
    from p3audit.experiments.unit import run_unit
    cfg = load_config(a.config)
    print("config hash", cfg.hash())
    ctx = Context.create(cfg)
    units = unit_list(cfg, a.confound)
    if a.shard:
        i, n = map(int, a.shard.split("/")); units = units[i::n]
    print(len(units), "units")
    for u in units:
        try:
            run_unit(ctx, u, out_dir="units_a1")
        except Exception as e:  # keep going; failures listed
            (ctx.results / "units_a1_failed").mkdir(parents=True, exist_ok=True)
            (ctx.results / "units_a1_failed" / f"{u.uid}.txt").write_text(traceback.format_exc(), encoding="utf-8")
            print("FAILED", u.uid, e)


if __name__ == "__main__":
    main()
