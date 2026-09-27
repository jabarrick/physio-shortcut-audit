"""3.4.7 calibration (P4 task strength first, then P5 confound amplitude).

(1) confound amplitude fixed at the reference (alpha k = 1; saccade = median real amplitude);
(2) choose s* so that every core model reaches BA ≥ 0.65 at p = 0.5 AND the median
    ΔBA_neu of CSOANet and CBraMod at the p = 0.9 centre cell lies in [0.12, 0.28];
(3) only if no s in a reasonable range works, raise the confound amplitude and
    report a*/reference in the abstract-level conclusions.
s_low / s_high = s giving p = 0.5 accuracy −/+ 10 pp relative to s*.
Uses training and validation subjects only: models are trained on non-pilot-held-out
training subjects, early-stopped on a slice of training subjects and *evaluated on the
10 validation subjects*.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..generator.design import Cell, test_couplings
from ..generator.semisynth import compose
from ..models.registry import build_model
from ..training.trainer import evaluate_ba, resolve_device, train_model
from ..utils.common import balanced_accuracy, get_logger, write_json
from .context import Context

log = get_logger("p3audit.calib")


def _roles(ctx):
    sp = ctx.split
    nonpilot = [s for s in sp.train if s not in sp.pilot]
    es = nonpilot[: max(2, len(nonpilot) // 6)]
    return ctx.usable([s for s in sp.train if s not in es]), ctx.usable(es), ctx.usable(sp.val)


def train_eval(ctx: Context, model_name: str, cell: Cell, seed: int = 0, device=None, fallback_standin=False,
               max_epochs=None) -> dict:
    cfg = ctx.cfg
    tr_s, es_s, ev_s = _roles(ctx)
    caches = ctx.caches(tr_s + es_s + ev_s)
    tr = compose(caches, tr_s, cell, cfg, cell.p, key=(cell.cid, "cal_train", seed))
    es = compose(caches, es_s, cell, cfg, cell.p, key=(cell.cid, "cal_es", seed))
    n_t = tr.core_len
    m = build_model(model_name, cfg, len(ctx.ch_names), n_t, ctx.ch_names, fallback_standin=fallback_standin)
    device = device or resolve_device(cfg["training"]["device"])
    train_model(m, tr.materialize(), tr.y, es.materialize(), es.y, cfg, seed, device, max_epochs, verbose=False)
    out = {}
    for k, q in test_couplings(cell.p).items():
        ev = compose(caches, ev_s, cell, cfg, q, key=(cell.cid, "cal_eval", k, seed), role="test")
        out[f"ba_{k}"] = evaluate_ba(m, ev.materialize(), ev.y, device)
    if "ba_neutral" in out:
        out["delta_ba_neu"] = out["ba_matched"] - out["ba_neutral"]
    return out


def _crossing(curve, s0: float, target: float, direction: int):
    """First s (walking from s0 in `direction`) at which the p = 0.5 curve crosses `target`,
    linearly interpolated between adjacent grid points."""
    xs = list(curve.index.values)
    i = xs.index(s0)
    path = xs[i::1] if direction > 0 else xs[i::-1]
    for a, b in zip(path[:-1], path[1:]):
        ya, yb = float(curve.loc[a]), float(curve.loc[b])
        if (ya - target) * (yb - target) <= 0 and ya != yb:
            s = a + (target - ya) * (b - a) / (yb - ya)
            return float(s), {"reached": True, "target_ba": target}
    end = path[-1]
    return float(end), {"reached": False, "target_ba": target, "ba_at_end": float(curve.loc[end]),
                        "achieved_diff_pp": 100 * (float(curve.loc[end]) - float(curve.loc[s0]))}


def calibrate_task_strength(ctx: Context, confound: str = "alpha", seed: int = 0, **kw) -> dict:
    cfg = ctx.cfg
    cal = cfg["calibration"]
    core = cfg["design"]["core_models"]
    # resumable: every finished training is appended to P4_<confound>_partial.jsonl and
    # skipped on a re-run (same s / model / p / seed); delete that file to start from scratch
    import json, time
    part = ctx.results / "pilots" / f"P4_{confound}_partial.jsonl"
    rows = []
    if part.exists():
        rows = [json.loads(l) for l in part.read_text(encoding="utf-8").splitlines() if l.strip()]
        rows = [r for r in rows if r.get("seed", 0) == seed]
        log.info(f"resuming: {len(rows)} trainings already done")
    done = {(float(r["s"]), r["model"], float(r["p"])) for r in rows}
    jobs = []
    for s in cal["s_grid"]:
        jobs += [(float(s), m, 0.5) for m in core] + [(float(s), m, 0.9) for m in cal["calib_models"]]
    ext = [float(s) for s in cal.get("s_grid_extension", []) or []]
    if any(s >= 2.0 for s in ext):
        raise ValueError("s_grid_extension must stay below 2 (gain 1 - s/2 > 0)")
    for s in ext:   # p = 0.5 only; used for s_low / s_high, never for s* (PILOT_LOG 11.16)
        jobs += [(s, m, 0.5) for m in core]
    for i, (s, m, p) in enumerate(jobs, 1):
        if (s, m, p) in done:
            continue
        t0 = time.time()
        log.info(f"[{i}/{len(jobs)}] s={s} model={m} p={p} ...")
        r = train_eval(ctx, m, Cell(confound, p, s, 1.0), seed, **kw)
        row = {"s": s, "model": m, "p": p, "seed": seed, **r}
        rows.append(row)
        with open(part, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        log.info(f"[{i}/{len(jobs)}] done in {time.time() - t0:.0f} s: " +
                 ", ".join(f"{k}={v:.3f}" for k, v in r.items() if isinstance(v, float)))
    df = pd.DataFrame(rows)
    ok = []
    lo, hi = cal["target_delta_ba_neu"]
    for s, g in df[df.s.isin([float(x) for x in cal["s_grid"]])].groupby("s"):
        ba_ok = (g[g.p == 0.5].ba_matched >= cal["min_ba_p05"]).all()
        dneu = g[g.p == 0.9].delta_ba_neu.median()
        ok.append({"s": s, "all_ba_ok": bool(ba_ok), "median_delta_ba_neu": float(dneu),
                   "ok": bool(ba_ok and lo <= dneu <= hi)})
    okdf = pd.DataFrame(ok)
    cand = okdf[okdf.ok]
    s_star = float(cand.s.iloc[len(cand) // 2]) if len(cand) else None
    res = {"rows": rows, "grid": ok, "s_star": s_star, "needs_amplitude_increase": s_star is None}
    # 2026-09-20 ~12:36 Lisbon (decided by Yu Gao BEFORE the cbramod rows at s = 1.077 / 1.333 of the
    # second P4 run existed; PILOT_LOG 11.9): when no s meets both conditions, P5 runs at the SMALLEST
    # s at which every core model reaches min_ba_p05 at p = 0.5.  Step (3) of 3.4.7 relaxes only the
    # ΔBA_neu condition; the BA condition is kept, and the smallest such s keeps the task hardest,
    # leaving the confound the most room to be used.
    ba_ok = sorted(g["s"] for g in ok if g["all_ba_ok"])
    res["s_for_p5"] = (s_star if s_star is not None else (float(ba_ok[0]) if ba_ok else None))
    res["s_for_p5_rule"] = "s_star if found, else smallest s with all core models BA >= min_ba_p05 at p = 0.5"
    anchor = s_star if s_star is not None else res["s_for_p5"]
    if anchor is not None:
        # p = 0.5 mean-BA curve over core models, grid + extension.  The curve is not monotone, so
        # np.interp (which needs increasing x) is wrong here; walk outward from s* instead and take the
        # first crossing (PILOT_LOG 11.16).  If the target is not reached, the end point is returned
        # with 'reached': False and the achieved difference is reported.
        curve = df[df.p == 0.5].groupby("s").ba_matched.mean().sort_index()
        target = float(curve.loc[anchor])
        lo_s, lo_meta = _crossing(curve, anchor, target - 0.10, direction=-1)
        hi_s, hi_meta = _crossing(curve, anchor, target + 0.10, direction=+1)
        res.update({"s_anchor": float(anchor), "ba_at_anchor": target, "s_low": lo_s, "s_high": hi_s,
                    "s_low_meta": lo_meta, "s_high_meta": hi_meta,
                    "p05_curve": {float(k): float(v) for k, v in curve.items()}})
    if s_star is not None:
        res["config_snippet"] = {f"task_component.by_confound.{confound}.s_star": s_star,
                                 f"task_component.by_confound.{confound}.s_low": res["s_low"],
                                 f"task_component.by_confound.{confound}.s_high": res["s_high"]}
    write_json(ctx.results / "pilots" / f"P4_{confound}.json", res)
    return res


def calibrate_amplitude(ctx: Context, confound: str = "alpha", s: float | None = None, seed: int = 0, **kw) -> dict:
    """P5: only when P4 fails — raise the confound amplitude until the ΔBA_neu window is met."""
    cfg = ctx.cfg
    cal = cfg["calibration"]
    if s is None:
        # never fall back to the task_component.s_star placeholder (invalid under symmetric
        # modulation, PILOT_LOG 11.6); take the s chosen by P4 under the 11.9 rule
        import json
        p4 = ctx.results / "pilots" / f"P4_{confound}.json"
        if not p4.exists():
            raise RuntimeError(f"{p4} missing: run P4 first")
        s = json.loads(p4.read_text(encoding="utf-8")).get("s_for_p5")
        if s is None:
            raise RuntimeError("P4 found no s with all core models at BA >= min_ba_p05; P5 cannot run")
        log.info(f"P5 at s = {s} (from {p4.name}, rule PILOT_LOG 11.9)")
    lo, hi = cal["target_delta_ba_neu"]
    rows = []
    chosen = None
    # resumable like P4 (two GPU hangs so far): every training is appended to P5_<confound>_partial.jsonl
    import json, time
    ref_tag = cfg["alpha_component"].get("reference", "sigma") if confound == "alpha" else "uv"
    grid = [float(x) for x in (cal.get("amp_grid_by_confound") or {}).get(confound, cal["amp_grid"])]
    part = ctx.results / "pilots" / f"P5_{confound}_partial.jsonl"
    done = {}
    if part.exists():
        for l in part.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                if (r.get("seed", 0) == seed and abs(float(r["s"]) - float(s)) < 1e-9
                        and r.get("reference", "sigma") == ref_tag):
                    done[(float(r["amp"]), r["model"])] = r["delta_ba_neu"]
        log.info(f"resuming: {len(done)} trainings already done")
    status = "infeasible"

    def _median_at(a):
        d = []
        for m in cal["calib_models"]:
            if (float(a), m) not in done:
                t0 = time.time()
                log.info(f"amp={a} s={s} model={m} p=0.9 ...")
                r = train_eval(ctx, m, Cell(confound, 0.9, float(s), float(a)), seed, **kw)
                done[(float(a), m)] = r["delta_ba_neu"]
                with open(part, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"s": float(s), "amp": float(a), "model": m, "seed": seed,
                                        "reference": ref_tag, **r}) + "\n")
                log.info(f"amp={a} model={m} done in {time.time() - t0:.0f} s: delta_ba_neu={r['delta_ba_neu']:.3f}")
            d.append(done[(float(a), m)])
        med = float(np.median(d))
        rows.append({"amp": a, "median_delta_ba_neu": med, "per_model": dict(zip(cal["calib_models"], d))})
        return med

    prev = None
    for a in grid:
        med = _median_at(a)
        if lo <= med <= hi:
            chosen, status = a, "ok"
            break
        if med > hi:
            # overshoot (rule PILOT_LOG 11.13): one extra run at the midpoint of the two grid points;
            # accept it if inside the window, else take whichever point is nearer the window.
            status = "overshoot"
            if prev is not None:
                mid = round((prev[0] + a) / 2.0, 4)
                mm = _median_at(mid)
                if lo <= mm <= hi:
                    chosen, status = mid, "ok_midpoint"
                else:
                    dist = lambda v: (lo - v) if v < lo else (v - hi)
                    cands = [(prev[0], prev[1]), (a, med), (mid, mm)]
                    chosen = min(cands, key=lambda c: dist(c[1]))[0]
                    status = "overshoot_nearest"
            break
        prev = (a, med)
    # PILOT_LOG 11.11: if the whole grid stays below the window the confound is NOT FEASIBLE at the
    # tested amplitudes; the grid is never extended.  For alpha under the ec_eo reference (11.13) this
    # moves the alpha arm out of the confirmatory H1-H3 tests (descriptive 'weak shortcut' arm at the
    # top grid amplitude); there is no further attempt.
    key = "alpha_component.a_star" if confound == "alpha" else "saccade.amplitude_uv"
    res = {"s": float(s), "reference": ref_tag, "grid": grid, "rows": rows, "a_star_multiplier": chosen, "status": status,
           "infeasible": status == "infeasible",
           "note": "report a*/reference in abstract-level conclusions if > 1"}
    if chosen is not None:
        ref = cfg.get_path(key)
        res["config_snippet"] = {key: ref * chosen}
    write_json(ctx.results / "pilots" / f"P5_{confound}.json", res)
    return res
