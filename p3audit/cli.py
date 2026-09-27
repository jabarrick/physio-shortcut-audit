"""Command-line entry point:  p3audit <command> [options]

  tbd                          list provisional values and their pilot sources
  prepare --stage S            split | components | pool | cache | all
  pilot NAME                   P2 P6 P1a sensitivity P7 P3 P4 P5 P8 P8b P8c P8L P9 P10a P11 P12
  grid                         enumerate units (model × cell × seed) and budget per module
  run-units [filters]          train + audit semi-synthetic units (sharding supported)
  real-audit / shu-audit       H4/H5 on PhysioNet-MI, supplementary SHU-MI
  analyse                      H1–H5 from stored results
  freeze                       preregistration snapshot (config/code hashes, split, access log)

Global:  --config FILE  --set key=value (repeatable)  --synthetic N  --confirmatory
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from .config import assert_confirmatory_ready, load_config, tbd_status
from .utils.common import get_logger, write_json

log = get_logger("p3audit")


def _parse_set(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        out[k] = yaml.safe_load(v)
    return out


def _ctx(args):
    from .experiments.context import Context
    cfg = load_config(args.config, _parse_set(args.set))
    if args.confirmatory:
        assert_confirmatory_ready(cfg)
    return Context.create(cfg, synthetic=args.synthetic is not None, n_synthetic=args.synthetic or 70)


def cmd_tbd(args):
    cfg = load_config(args.config, _parse_set(args.set))
    rows = tbd_status(cfg)
    w = max(len(r["key"]) for r in rows)
    for r in rows:
        mark = "✓" if r["confirmed"] else "·"
        print(f"{mark} {r['key']:<{w}}  {r['source']:<10} {r['value']}")
    print(f"\n{sum(not r['confirmed'] for r in rows)} unconfirmed of {len(rows)}")


def cmd_prepare(args):
    from .experiments import prepare
    ctx = _ctx(args)
    subs = [int(s) for s in args.subjects.split(",")] if args.subjects else None
    stages = ["split", "components", "pool", "cache"] if args.stage == "all" else [args.stage]
    for st in stages:
        if st == "split":
            ctx.run_exclusion_and_split()
        elif st == "components":
            print(prepare.stage_components(ctx, subs, args.overwrite))
        elif st == "pool":
            r = prepare.stage_pool(ctx)
            print({k: r[k] for k in ("feasible", "n_failed", "main_pool", "mismatch_pool")})
        elif st == "cache":
            print(prepare.stage_cache(ctx, subs, args.overwrite)["skipped"])


def cmd_pilot(args):
    from .experiments import calibration, pilots
    ctx = _ctx(args)
    kw = {"fallback_standin": args.fallback_standin} if args.fallback_standin else {}
    name = args.name.upper() if args.name.lower() != "sensitivity" else "sensitivity"
    fn = {"P2": pilots.p2, "P6": pilots.p6, "P1A": pilots.p1a, "sensitivity": pilots.sensitivity, "P7": pilots.p7,
          "P9": pilots.p9, "P11": pilots.p11, "P12": pilots.p12}
    if name in fn:
        res = fn[name](ctx, **({} if name not in ("P9",) else kw))
    elif name == "P3":
        res = pilots.p3(ctx, fallback_standin=True)
    elif name == "P4":
        res = calibration.calibrate_task_strength(ctx, args.confound, max_epochs=args.max_epochs, **kw)
    elif name == "P5":
        res = calibration.calibrate_amplitude(ctx, args.confound, max_epochs=args.max_epochs, **kw)
    elif name == "P8":
        res = pilots.p8(ctx, max_epochs=args.max_epochs, **kw)
    elif name == "P8B":
        res = pilots.p8b(ctx, max_epochs=args.max_epochs, **kw)
    elif name == "P8L":
        res = pilots.p8_labram(ctx, max_epochs=args.max_epochs, **kw)
    elif name == "P8C":
        res = pilots.p8c(ctx, max_epochs=args.max_epochs, **kw)
    elif name == "P10A":
        res = pilots.p10a(ctx, args.npz, args.px_per_deg)
    else:
        raise SystemExit(f"unknown pilot {args.name}")
    print(json.dumps(res, indent=1, default=str)[:4000])


def _select_units(cfg, args):
    from .generator.design import enumerate_units
    units = enumerate_units(cfg)
    if args.module:
        units = [u for u in units if u.cell.module in args.module.split(",")]
    if args.model:
        units = [u for u in units if u.model in args.model.split(",")]
    if args.confound:
        units = [u for u in units if u.cell.confound == args.confound]
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        units = units[i::n]
    return units


def cmd_grid(args):
    from .generator.design import enumerate_units, unit_counts
    cfg = load_config(args.config, _parse_set(args.set))
    units = enumerate_units(cfg)
    print(json.dumps(unit_counts(units), indent=1))
    print(f"total trainings: {len(units)}")
    if args.out:
        Path(args.out).write_text("\n".join(u.uid for u in units), encoding="utf-8")


def cmd_run_units(args):
    from .experiments.unit import run_unit
    ctx = _ctx(args)
    units = _select_units(ctx.cfg, args)
    log.info(f"{len(units)} units selected")
    failed = []
    for u in units:
        try:
            run_unit(ctx, u, fallback_standin=args.fallback_standin, max_epochs=args.max_epochs, overwrite=args.overwrite)
        except Exception as e:  # noqa: BLE001 — keep going; failures are listed at the end
            import traceback
            log.error(f"{u.uid} failed: {e}")
            (ctx.results / "units_failed").mkdir(parents=True, exist_ok=True)
            (ctx.results / "units_failed" / f"{u.uid}.txt").write_text(traceback.format_exc(), encoding="utf-8")
            failed.append(u.uid)
    log.info(f"done; {len(failed)} failed: {failed}")


def cmd_real(args):
    from .experiments.real_audit import run_real_audit
    ctx = _ctx(args)
    print(run_real_audit(ctx, max_epochs=args.max_epochs, fallback_standin=args.fallback_standin))


def cmd_shu(args):
    from .experiments.shu_audit import run_shu_audit
    ctx = _ctx(args)
    print(len(run_shu_audit(ctx, max_epochs=args.max_epochs, fallback_standin=args.fallback_standin)))


def cmd_analyse(args):
    from .experiments.analysis import run_all
    ctx = _ctx(args)
    out = run_all(ctx)
    print(json.dumps({k: (v if k == "n_units" else "written") for k, v in out.items()}, indent=1))


def cmd_freeze(args):
    """Preregistration snapshot (6.3 item 1): what pilots have seen, with hashes."""
    ctx = _ctx(args)
    pkg = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for p in sorted(pkg.rglob("*.py")):
        h.update(p.read_bytes())
    log_path = ctx.cfg.resolve_path("access_log")
    access = []
    if log_path.exists():
        access = [json.loads(line) for line in open(log_path, encoding="utf-8")]
    seen = sorted({(a["subject"], a["purpose"]) for a in access})
    snap = {"config_hash": ctx.cfg.hash(), "code_sha256": h.hexdigest(), "split": ctx.split.as_dict() if ctx.split else None,
            "signal_access": [{"subject": s, "purpose": p} for s, p in seen],
            "pilot_reports": sorted(p.name for p in (ctx.results / "pilots").glob("*.json")),
            "tbd": tbd_status(ctx.cfg), "config": dict(ctx.cfg)}
    write_json(ctx.results / "prereg_snapshot.json", snap)
    print(f"snapshot written: config {snap['config_hash']} code {snap['code_sha256'][:16]}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="p3audit")
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", help="override: key.path=value (YAML)")
    ap.add_argument("--synthetic", type=int, default=None, help="use N synthetic subjects (dry run)")
    ap.add_argument("--confirmatory", action="store_true", help="refuse to run with unconfirmed TBD values")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("tbd").set_defaults(fn=cmd_tbd)
    p = sub.add_parser("prepare"); p.add_argument("--stage", default="all",
                                                  choices=["split", "components", "pool", "cache", "all"])
    p.add_argument("--subjects"); p.add_argument("--overwrite", action="store_true"); p.set_defaults(fn=cmd_prepare)
    p = sub.add_parser("pilot"); p.add_argument("name"); p.add_argument("--confound", default="alpha")
    p.add_argument("--max-epochs", type=int); p.add_argument("--fallback-standin", action="store_true")
    p.add_argument("--npz"); p.add_argument("--px-per-deg", type=float); p.set_defaults(fn=cmd_pilot)
    p = sub.add_parser("grid"); p.add_argument("--out"); p.set_defaults(fn=cmd_grid)
    p = sub.add_parser("run-units")
    for a in ("--module", "--model", "--confound", "--shard"):
        p.add_argument(a)
    p.add_argument("--max-epochs", type=int); p.add_argument("--fallback-standin", action="store_true")
    p.add_argument("--overwrite", action="store_true"); p.set_defaults(fn=cmd_run_units)
    for name, fn in (("real-audit", cmd_real), ("shu-audit", cmd_shu)):
        p = sub.add_parser(name); p.add_argument("--max-epochs", type=int)
        p.add_argument("--fallback-standin", action="store_true"); p.set_defaults(fn=fn)
    sub.add_parser("analyse").set_defaults(fn=cmd_analyse)
    sub.add_parser("freeze").set_defaults(fn=cmd_freeze)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
