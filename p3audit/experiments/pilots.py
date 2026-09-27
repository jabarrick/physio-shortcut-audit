"""Pilot experiments (outline §6).  Each function writes results/pilots/<name>.json
and returns the dict.  First batch, in order: P2 -> P6 -> P1a -> sensitivity -> P7;
P3 in parallel.  Second batch: P4/P5 (calibration.py), P8, P9, P10a, P12; third: P11.
Signal statistics here touch pilot subjects only (SignalAccessGuard enforces it),
except P8, which evaluates on validation subjects (never on test subjects).
"""
from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import pandas as pd

from ..constants import IMAGERY_RUNS
from ..data.windows import t0_windows
from ..generator.cache import sigma_log_power
from ..generator.components import task_bases
from ..generator.design import Cell, Unit, enumerate_units, unit_counts
from ..generator.overlap import measure_injection, topo_dispersion
from ..models.registry import build_model
from ..saccades.detect import classify_trials, return_saccades, trim_ms_from_returns
from ..saccades.sensitivity import sensitivity_curve
from ..utils.common import get_logger, write_json
from .context import Context

log = get_logger("p3audit.pilots")


def _save(ctx, name, obj):
    write_json(ctx.results / "pilots" / f"{name}.json", obj)
    return obj


# ------------------------------------------------------------------ P2
def p2(ctx: Context) -> dict:
    """Metadata exclusion summary; T0 stats, return-saccade 95th pct (X) and window counts (pilot only)."""
    cfg = ctx.cfg
    ex = pd.read_csv(ctx.results / "exclusion.csv")
    rows, t0_len, counts = [], [], []
    for s in ctx.require_split().pilot:
        ctx.guard.record(s, IMAGERY_RUNS, "pilot_t0_stats")
        ctx.guard.record(s, IMAGERY_RUNS, "pilot_return_saccades")
        for r in IMAGERY_RUNS:
            run = ctx.source.load(s, r)
            t0_len += [d / run.sfreq for _, d, _ in run.event_samples("T0")]
            rows += return_saccades(run, cfg)
    X = trim_ms_from_returns(rows, 95)
    quant_ms = {q: trim_ms_from_returns(rows, q) for q in (50, 75, 90, 95)}
    t0_p5 = float(np.percentile(t0_len, 5)) if t0_len else float("nan")
    # masks are independent of window_s / trim_mode, so build each one once (3.4.4 detector is slow)
    from ..saccades.detect import saccade_mask
    masks = {(s, r): saccade_mask(ctx.source.load(s, r), cfg)
             for s in ctx.require_split().pilot for r in IMAGERY_RUNS}
    for win in (3.0, 4.0):
        # trim_and_drop candidates: sub-95th quantiles of the return saccade that still leave the
        # window inside the shortest T0 segment (see the P2 note in data/windows.py)
        headroom_ms = max(0.0, (t0_p5 - win) * 1000)
        cands = sorted({float(round(v / 50) * 50) for q, v in quant_ms.items()
                        if q < 95 and np.isfinite(v) and v <= headroom_ms})
        modes = [("trim_start", X), ("drop_saccade_windows", 0.0)] + [("trim_and_drop", t) for t in cands]
        for mode, trim in modes:
            c2 = ctx.cfg.copy()
            c2["background"]["window_s"] = win
            c2["background"]["trim_mode"] = mode
            c2["background"]["trim_ms"] = trim if np.isfinite(trim) else 0
            for s in ctx.require_split().pilot:
                n = 0
                for r in IMAGERY_RUNS:
                    run = ctx.source.load(s, r)
                    n += len(t0_windows(run, c2, None if mode == "trim_start" else masks[(s, r)])[0])
                counts.append({"window_s": win, "mode": mode, "trim_ms": float(c2["background"]["trim_ms"]),
                               "subject": s, "n_windows": n})
    cdf = pd.DataFrame(counts)
    summary = (cdf.groupby(["window_s", "mode", "trim_ms"])["n_windows"]
               .describe()[["mean", "min", "max"]].reset_index())
    rs = pd.DataFrame(rows)
    return _save(ctx, "P2", {
        "n_subjects": int(len(ex)), "n_excluded": int(ex.excluded.sum()),
        "excluded": ex[ex.excluded][["subject", "reasons"]].to_dict("records"),
        "t0_duration_s": {"median": float(np.median(t0_len)), "p5": float(np.percentile(t0_len, 5)),
                          "p95": float(np.percentile(t0_len, 95))},
        "return_saccades": {"detection_rate": float(rs.detected.mean()) if len(rs) else float("nan"),
                            "latency_plus_duration_p95_ms": X,
                            "latency_plus_duration_quantiles_ms": {str(q): v for q, v in quant_ms.items()}},
        "recommended_trim_ms": X,
        "trim_headroom_ms": {"3.0": max(0.0, (t0_p5 - 3.0) * 1000), "4.0": max(0.0, (t0_p5 - 4.0) * 1000)},
        "window_counts": summary.to_dict("records"),
        "decision_note": "set background.window_s / trim_mode / trim_ms (TBD P2) from these numbers; "
                         "trim_start is infeasible here (trim + window_s > T0 duration), so the choice "
                         "is drop_saccade_windows vs trim_and_drop at a sub-95th-percentile trim",
    })


# ------------------------------------------------------------------ P6
def p6(ctx: Context) -> dict:
    from .prepare import stage_pool
    cfg = ctx.cfg
    pool = stage_pool(ctx)
    amps = []
    for s in ctx.require_split().pilot:
        ctx.guard.record(s, cfg["saccade"]["lateral_runs"], "pilot_saccade_amplitude")
        for r in cfg["saccade"]["lateral_runs"]:
            amps += [row["amplitude"] for row in classify_trials(ctx.source.load(s, r), cfg)
                     if row["type"] != "none"]
    a = np.abs(np.asarray(amps, dtype=float))
    pool["cue_saccade_amplitude_uv"] = {"n": int(a.size), "median": float(np.median(a)) if a.size else np.nan,
                                        "q25": float(np.percentile(a, 25)) if a.size else np.nan,
                                        "q75": float(np.percentile(a, 75)) if a.size else np.nan,
                                        "note": "truncated by the detector threshold; see P10a (EEGEyeNet)"}
    pool["split_half_r"] = [t["split_half_r"] for t in pool["per_subject"] if t["ok"]]
    return _save(ctx, "P6", pool)


# ------------------------------------------------------------------ P1a
def p1a(ctx: Context) -> dict:
    """Component inclusion, realised injection effects, overlap and topographic dispersion."""
    cfg = ctx.cfg
    ac = cfg["alpha_component"]
    pilot_rows = []
    for s in ctx.require_split().pilot:
        comps = ctx.load_components(s)
        if comps is None:
            continue
        row = {"subject": s, **{f"{k}_passed": v.passed for k, v in comps.items()}}
        if comps["alpha"].passed:
            ctx.guard.record(s, IMAGERY_RUNS, "pilot_injection_check")
            runs = [ctx.source.load(s, r) for r in IMAGERY_RUNS]
            series, run_of, t_of, wins_by_run = [], [], [], []
            from ..data.streams import fir
            drop_mode = cfg["background"]["trim_mode"] in ("drop_saccade_windows", "trim_and_drop")
            for run in runs:
                # measure on the windows the generator will actually use (3.4.1), saccade drop included
                m = None
                if drop_mode:
                    from ..saccades.detect import saccade_mask
                    m = saccade_mask(run, cfg)
                wins, _ = t0_windows(run, cfg, m)
                sa = comps["alpha"].w @ fir(run.data, run.sfreq, *comps["alpha"].band, trans=ac["fir_transition_hz"])
                series += [sa[w.start:w.start + w.length] for w in wins]
                run_of += [run.run] * len(wins); t_of += [w.start / run.sfreq for w in wins]
                wins_by_run.append((run, wins))
            sigma = sigma_log_power(series, run_of, t_of)
            # PILOT_LOG 12.4 (3): read the amplitude through the same function the generator uses,
            # so P1a follows alpha_component.reference (sigma | ec_eo) instead of assuming sigma.
            from ..generator.semisynth import alpha_log_change
            target_log_change = float(alpha_log_change(cfg, 1.0, sigma))
            g = float(np.exp(target_log_change / 2))
            meas = [measure_injection(run, wins, comps["alpha"], g, cfg, ac["central"]) for run, wins in wins_by_run if wins]
            row.update({"sigma_alpha": sigma, "g": g,
                        "alpha_reference": ac.get("reference", "sigma"),
                        "target_log_change": target_log_change,
                        "component_log_change": float(np.mean([m["component_log_change"] for m in meas])),
                        "central_log_change": float(np.mean([m["central_log_change"] for m in meas]))})
            tb = task_bases(cfg)
            tkey = tb[0] if len(tb) == 1 else "mu_L"
            if tkey in comps and comps[tkey].passed:
                # PILOT_LOG 12.4 (3): s* is calibrated per confound (11.16); the global
                # task_component.s_star is a dead placeholder.  Report both arms.
                side = comps[tkey].checks.get("side")
                anchor = (["C4"] if side == "R" else ["C3"]) if len(tb) == 1 else ["C3"]
                row["task_side"] = side
                row["task_anchor"] = anchor[0]
                for conf in ("alpha", "saccade"):
                    s_star = cfg.task_strength("s_star", conf)
                    # symmetric modulation: the y=1 arm is g = 1 − s/2; probe that arm
                    g_task = (1 - s_star / 2.0) if len(tb) == 1 else (1 - s_star)
                    mm = [measure_injection(run, wins, comps[tkey], g_task, cfg, anchor)
                          for run, wins in wins_by_run if wins]
                    row[f"task_s_star_{conf}"] = s_star
                    row[f"task_g_{conf}"] = g_task
                    row[f"task_central_log_change_{conf}"] = float(np.mean([m["central_log_change"] for m in mm]))
        pilot_rows.append(row)
    pdf = pd.DataFrame(pilot_rows)
    tb = task_bases(cfg)
    if not len(pdf):
        mu_fail = 0
    elif len(tb) == 1:
        mu_fail = int((~pdf["mu_passed"].astype(bool)).sum())
    else:
        mu_fail = int((~(pdf.mu_L_passed & pdf.mu_R_passed)).sum())
    # component-derived quantities for all subjects (allowed; 十·3)
    ov, pats = [], {k: {} for k in ("alpha", *tb)}
    for s in ctx.all_subjects:
        import json
        p = ctx.comp_path(s)
        if not p.exists():
            continue
        d = json.load(open(p, encoding="utf-8"))
        ov.append({"subject": s, "pilot": s in ctx.require_split().pilot, **d["overlap"]})
        comps = ctx.load_components(s)
        for k in pats:
            if k in comps and comps[k].passed:
                pats[k][s] = comps[k].a
    odf = pd.DataFrame(ov)
    frac_fail = mu_fail / max(1, len(pdf))
    return _save(ctx, "P1a", {
        "pilot": pilot_rows,
        "mu_fail_pilot": mu_fail, "mu_fail_fraction": frac_fail,
        "mu_redesign_flag": bool(frac_fail > cfg["task_component"]["max_fail_fraction"]),
        "overlap_all_subjects": {c: odf[c].describe().to_dict() for c in ("projection_ratio", "overlap")} if len(odf) else {},
        "overlap_values": odf.to_dict("records"),
        "projection_ratio_below_min": bool(len(odf) and odf.projection_ratio.median() < ac["min_projection_ratio"]),
        "topo_dispersion": {k: topo_dispersion(v) for k, v in pats.items()},
    })


# ------------------------------------------------------------------ sensitivity curve
def sensitivity(ctx: Context, amplitudes=(5, 10, 15, 20, 30, 40, 60, 80)) -> dict:
    pool = ctx.load_pool()
    rows = sensitivity_curve(ctx.source, ctx.require_split().pilot, [4, 8, 12], pool, amplitudes, ctx.cfg, guard=ctx.guard)
    df = pd.DataFrame(rows)
    curve = df.groupby("amplitude_uv").apply(lambda g: g.hits.sum() / g.n.sum()).to_dict() if len(df) else {}
    return _save(ctx, "sensitivity", {"curve": {float(k): float(v) for k, v in curve.items()},
                                      "miss_rate_lower_bound": {float(k): 1 - float(v) for k, v in curve.items()},
                                      "rows": rows})


# ------------------------------------------------------------------ P7
def p7(ctx: Context) -> dict:
    cfg = ctx.cfg
    rows = []
    for s in ctx.require_split().pilot:
        ctx.guard.record(s, [4, 8, 12], "pilot_trial_counts")
        for r in (4, 8, 12):
            rows += classify_trials(ctx.source.load(s, r), cfg)
    df = pd.DataFrame(rows)
    ct = df.groupby(["subject", "type"]).size().unstack(fill_value=0)
    for c in ("congruent", "incongruent", "none"):
        if c not in ct:
            ct[c] = 0
    k = cfg["h4"]["k_min"]
    ok = (ct.congruent >= k) & (ct.none >= k)
    return _save(ctx, "P7", {"counts": ct.reset_index().to_dict("records"), "k_min": k,
                             "fraction_eligible": float(ok.mean()), "provisional": True})


# ------------------------------------------------------------------ P3
def p3(ctx: Context, models=None, n_windows: int = 512, device=None, fallback_standin: bool = True) -> dict:
    """Single-epoch training time, inference throughput and peak memory per model,
    then GPU-hour estimates for every design module (table 7)."""
    import torch
    from ..training.trainer import resolve_device
    cfg = ctx.cfg
    device = device or resolve_device(cfg["training"]["device"])
    models = models or cfg["design"]["core_models"] + cfg["design"]["supplementary_models"]
    n_t = int(round(cfg["background"]["window_s"] * cfg["streams"]["model"]["sfreq"]))
    n_ch = len(ctx.ch_names)
    X = torch.randn(n_windows, n_ch, n_t) * 20
    y = torch.randint(0, 2, (n_windows,))
    out = {}
    for name in models:
        m = build_model(name, cfg, n_ch, n_t, ctx.ch_names, fallback_standin=fallback_standin).to(device)
        opt = torch.optim.AdamW(m.parameters(), 1e-3)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t = time.time()
        m.train()
        for i in range(0, n_windows, cfg["training"]["batch_size"]):
            xb, yb = X[i:i + 64].to(device), y[i:i + 64].to(device)
            opt.zero_grad(); torch.nn.functional.cross_entropy(m(xb), yb).backward(); opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        per_window_train = (time.time() - t) / n_windows
        m.eval(); t = time.time()
        with torch.no_grad():
            for i in range(0, n_windows, 256):
                m(X[i:i + 256].to(device))
        if device.type == "cuda":
            torch.cuda.synchronize()   # CUDA is async: without this the inference timer stops early
        per_window_inf = (time.time() - t) / n_windows
        out[name] = {"params": int(sum(p.numel() for p in m.parameters())),
                     "train_s_per_window_epoch": per_window_train, "infer_s_per_window": per_window_inf,
                     "peak_mem_mb": (torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else None,
                     "standin": bool(getattr(m, "is_standin", False))}
    units = enumerate_units(cfg)
    counts = unit_counts(units)
    # expected windows: ~ (#train subjects) × windows/subject; epochs ~ max_epochs/2 (early stopping)
    wps, wps_source = 80.0, "fallback 80 - NOT measured; run P2 first and re-run P3 before quoting a budget"
    p2_path = ctx.results / "pilots" / "P2.json"
    if p2_path.exists():
        import json
        rows = json.load(open(p2_path, encoding="utf-8")).get("window_counts", [])
        bg = cfg["background"]

        def _hit(r, with_trim):
            ok = (abs(float(r.get("window_s", -1)) - float(bg["window_s"])) < 1e-9
                  and r.get("mode") == bg["trim_mode"])
            return ok and (abs(float(r.get("trim_ms", -1)) - float(bg["trim_ms"])) < 1e-6 if with_trim else True)

        for with_trim in (True, False):
            hit = [r for r in rows if _hit(r, with_trim)]
            if hit:
                wps = float(hit[0]["mean"])
                wps_source = (f"P2 measured: window_s={bg['window_s']} mode={bg['trim_mode']}"
                              + (f" trim_ms={bg['trim_ms']}" if with_trim else " (trim_ms NOT matched)"))
                break
    n_train = len(ctx.require_split().train) if ctx.split else 50
    est = {}
    for mod, c in counts.items():
        h = 0.0
        for name, k in c.items():
            if name == "total" or name not in out:
                continue
            h += k * out[name]["train_s_per_window_epoch"] * n_train * wps * cfg["training"]["max_epochs"] / 2 / 3600
        est[mod] = {"units": c["total"], "train_hours_est": h}
    return _save(ctx, "P3", {"device": str(device), "models": out, "module_estimates": est,
                             "budget_assumptions": {
                                 "windows_per_subject": wps, "windows_per_subject_source": wps_source,
                                 "n_train_subjects": n_train,
                                 "epochs_assumed": cfg["training"]["max_epochs"] / 2,
                                 "excludes": "data loading, per-epoch validation, memmap reads "
                                             "- expect 1.5-2x wall clock"},
                             "note": "inference-side cost (IG, probes, LEACE, bootstrap) measured in P9"})


# ------------------------------------------------------------------ P8
def p8(ctx: Context, models=("csoanet", "cbramod"), cells=None, seeds=(0, 1), tag: str = "P8", **kw) -> dict:
    """Variance / reliability of ΔBA_neu and metrics: 2 models × 3 cells × 2 seeds (12 trainings).
    Evaluated on the 10 validation subjects; early stopping on a held-out slice of non-pilot
    training subjects — no test subject is touched before preregistration."""
    from .unit import run_unit
    sp = ctx.split
    cfg = ctx.cfg
    cells = cells or [Cell("alpha", 0.9, "s_star", 1.0, "factorial"), Cell("alpha", 0.9, "s_low", 1.5, "factorial"),
                      Cell("saccade", 0.9, "s_star", 1.0, "factorial")]
    nonpilot = [s for s in sp.train if s not in sp.pilot]
    es = nonpilot[: max(2, len(nonpilot) // 6)]
    roles = {"train": [s for s in sp.train if s not in es], "val": es, "test": sp.val}
    recs = []
    for m in models:
        for c in cells:
            for sd in seeds:
                recs.append(run_unit(ctx, Unit(m, c, sd), roles=roles, out_dir="pilots/p8_units", **kw))
    summ = [{"model": r["model"], "cell": r["cid"], "seed": r["seed"], **{k: v for k, v in r["truth"].items()
                                                                          if not k.endswith(("half0", "half1"))}}
            for r in recs]
    df = pd.DataFrame(summ)
    var = df.groupby(["model", "cell"])["delta_ba_neu"].agg(["mean", "std"]).reset_index() if "delta_ba_neu" in df else None
    return _save(ctx, tag, {"units": summ, "seed_variance": var.to_dict("records") if var is not None else None,
                            "roles": roles})


def p8b_verdict(dba: list, window, s_star: float) -> dict:
    """calibration.p8b rule (PILOT_LOG 15.4 D1, written before any P8b run): unweighted mean of
    dBA_neu over all 8 runs of the saccade calibration cell against target_delta_ba_neu."""
    lo, hi = float(window[0]), float(window[1])
    m = float(np.mean(dba))
    if len(dba) != 8:
        return {"n_runs": len(dba), "mean": m, "verdict": "incomplete"}
    v = "keep_s_star" if lo <= m <= hi else ("try_s_1.6" if m > hi else "try_s_1.077")
    return {"n_runs": 8, "mean": m, "window": [lo, hi], "s_star": s_star, "verdict": v}


def p8b(ctx: Context, **kw) -> dict:
    """PILOT_LOG 15.4 D1: seeds 2-3 on the saccade calibration cell, then the pre-written verdict.
    P8.json is not touched; units go to the same p8_units folder (distinct seed in the file name)."""
    import json
    cfg = ctx.cfg
    rule = cfg["calibration"]["p8b"]
    conf = rule["confound"]
    cell = Cell(conf, 0.9, "s_star", 1.0, "factorial")
    res = p8(ctx, cells=[cell], seeds=tuple(rule["seeds"]), tag="P8b_units", **kw)
    old = json.load(open(ctx.results / "pilots" / "P8.json", encoding="utf-8"))["units"]
    cid = res["units"][0]["cell"] if res["units"] else None
    runs = [u for u in old if u["cell"] == cid] + res["units"]
    verdict = p8b_verdict([u["delta_ba_neu"] for u in runs], cfg["calibration"]["target_delta_ba_neu"],
                          cfg.task_strength("s_star", conf))
    return _save(ctx, "P8b", {"units": res["units"], "all_runs": runs, **verdict, "rule": rule})


def p8_labram(ctx: Context, **kw) -> dict:
    """PILOT_LOG 15.26: functioning check for the newly wired LaBraM, written BEFORE running.
    One unit (alpha calibration cell p 0.9, s*, a*, seed 0), same roles as P8.  Not a calibration:
    LaBraM shares the core models' s* and a*.  Reading: ba_matched <= 0.55 or constant prediction
    -> the fine-tuning setup is broken (fix before registration); otherwise wired correctly."""
    cell = Cell("alpha", 0.9, "s_star", 1.0, "factorial")
    return p8(ctx, models=("labram",), cells=[cell], seeds=(0,), tag="P8_labram", **kw)


def p8c(ctx: Context, **kw) -> dict:
    """calibration.p8b second step (PILOT_LOG 15.4 D1, written before P8b ran): P8b gave
    'try_s_1.6' or 'try_s_1.077' -> the same 8-run check (2 models x seeds 0-3) at that s.
    In window -> adopt it as the saccade s*; otherwise keep 1.333, stop, report outside the window."""
    import json
    cfg = ctx.cfg
    rule = cfg["calibration"]["p8b"]
    b = json.load(open(ctx.results / "pilots" / "P8b.json", encoding="utf-8"))
    if b.get("verdict") not in ("try_s_1.6", "try_s_1.077"):
        raise SystemExit(f"P8c not called for: P8b verdict is {b.get('verdict')}")
    s_new = 1.6 if b["verdict"] == "try_s_1.6" else 1.077
    cell = Cell(rule["confound"], 0.9, s_new, 1.0, "factorial")
    res = p8(ctx, cells=[cell], seeds=(0, 1, 2, 3), tag="P8c_units", **kw)
    v = p8b_verdict([u["delta_ba_neu"] for u in res["units"]], cfg["calibration"]["target_delta_ba_neu"], s_new)
    final = ({"keep_s_star": f"adopt_s_{s_new:g}"}.get(v["verdict"], "keep_s_1.333_report_outside_window")
             if v["verdict"] != "incomplete" else "incomplete")
    return _save(ctx, "P8c", {"units": res["units"], "s_tested": s_new, "mean": v["mean"], "n_runs": v["n_runs"],
                              "window": cfg["calibration"]["target_delta_ba_neu"], "final": final, "rule": rule})


# ------------------------------------------------------------------ P9
def p9(ctx: Context, models=None, n_windows: int = 32, device=None, fallback_standin=True) -> dict:
    """IG cost (3.6.2).  Rewritten 2026-09-23 (PILOT_LOG 13.8); the first version
      (a) timed CBraMod only and charged its time to all 286 IG units, conv nets included;
      (b) timed batch 4 while run_unit used the ig_filterbank default of 16;
      (c) counted the zero baseline as a second full pass - it is ONE forward+backward per
          batch against K (bands incl. 'rest') for band_removed, so the factor is (K+1)/K, not 2;
      (d) timed CUDA start-up together with the work.
    Here every model is timed separately after a warm-up, at batch 4 and at the batch run_unit
    will use (ig_batch_for).  Before a GPU run at a batch above 4 the peak memory is predicted
    from the batch-4 measurement; if it exceeds 90 % of dedicated VRAM that run is SKIPPED, not
    attempted: on Windows (WDDM) CUDA then spills into shared system memory instead of failing,
    which is slow and on this machine a plausible cause of the 2026-09-20 'GPU is lost'.
    The result is saved after every model, so a crash keeps what was measured."""
    import torch
    from collections import Counter
    from ..metrics.attribution import band_masks, decompose, ig_batch_for, ig_filterbank
    from ..training.trainer import resolve_device
    cfg = ctx.cfg
    igc = cfg["ig"]
    device = device or resolve_device(cfg["training"]["device"])
    cuda = device.type == "cuda"
    n_t = int(round(cfg["background"]["window_s"] * cfg["streams"]["model"]["sfreq"]))
    sf = cfg["streams"]["model"]["sfreq"]
    models = list(models or cfg["design"]["core_models"])
    X = np.random.default_rng(0).standard_normal((n_windows, len(ctx.ch_names), n_t)).astype(np.float32) * 20
    names, masks = band_masks(n_t, sf, igc["bands"])
    K = len(names)
    xt = torch.as_tensor(X)
    recon = float((decompose(xt, masks).sum(1) - xt).abs().max() / xt.abs().max())
    pass_factor = (K + 1) / K if igc["zero_baseline"] else 1.0
    vram_mb = torch.cuda.get_device_properties(device).total_memory / 2**20 if cuda else None
    units = Counter(u.model for u in enumerate_units(cfg) if u.cell.p in igc["p_levels"])

    def timed(m, batch):
        ig_filterbank(m, X[:batch], sf, igc["bands"], igc["steps"], "band_removed", device, batch=batch)
        if cuda:
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t = time.time()
        ig_filterbank(m, X, sf, igc["bands"], igc["steps"], "band_removed", device, batch=batch)
        if cuda:
            torch.cuda.synchronize()
        return (time.time() - t) / n_windows, (torch.cuda.max_memory_allocated() / 2**20 if cuda else None)

    out = {"device": str(device), "cuda_available": bool(torch.cuda.is_available()),
           "gpu_name": torch.cuda.get_device_name(device) if cuda else None, "dedicated_vram_mb": vram_mb,
           "bands": names, "passes_band_removed": K, "zero_baseline_factor": pass_factor,
           "steps": igc["steps"], "windows_per_unit": igc["windows_per_unit"], "n_windows_timed": n_windows,
           "filterbank_max_rel_reconstruction_error": recon, "models": {}}
    if not cuda:
        log.warning("P9 is running on the CPU - the timings are not the GPU cost")
    for name in models:
        m = build_model(name, cfg, len(ctx.ch_names), n_t, ctx.ch_names, fallback_standin=fallback_standin).to(device)
        want = ig_batch_for(cfg, name)
        r = {"standin": bool(getattr(m, "is_standin", False)), "units_with_ig": units.get(name, 0),
             "configured_batch": want}
        # PILOT_LOG 13.10: never probe ABOVE the configured batch.  v2 always probed batch 4 first,
        # which put shallow (configured 2) at 90 % of dedicated VRAM just before the second GPU loss.
        probe = min(4, want)
        s4, mem4 = timed(m, probe)
        r[f"at_batch_{probe}"] = {"seconds_per_window": s4, "peak_mem_mb": mem4}
        use_s, use_b = s4, probe
        if want > probe:
            pred = mem4 * want / probe if mem4 is not None else None
            r["predicted_peak_mem_mb_at_configured"] = pred
            if cuda and pred is not None and pred > 0.9 * vram_mb:
                r["configured_batch_measured"] = False
                r["note"] = (f"batch {want}: predicted {pred:.0f} MB > 90% of {vram_mb:.0f} MB dedicated VRAM, "
                             f"not run (WDDM would spill into shared memory).  Hours use batch 4; "
                             f"set ig.batch_by_model.{name} to a batch that fits.")
            else:
                sb, memb = timed(m, want)
                r["at_configured_batch"] = {"seconds_per_window": sb, "peak_mem_mb": memb}
                r["configured_batch_measured"] = True
                use_s, use_b = sb, want
        r["hours_basis_batch"] = use_b
        r["hours"] = units.get(name, 0) * igc["windows_per_unit"] * use_s * pass_factor / 3600
        out["models"][name] = r
        log.info(f"P9 {name}: {use_s:.4f} s/window at batch {use_b} -> {r['hours']:.2f} h")
        del m
        if cuda:
            torch.cuda.empty_cache()
        out["hours_total_timed_models"] = sum(v["hours"] for v in out["models"].values())
        _save(ctx, "P9", out)
    out["units_with_ig_not_timed"] = {k: v for k, v in units.items() if k not in out["models"]}
    out["note"] = ("hours = units_with_ig x windows_per_unit x s/window x (K+1)/K, per model, at "
                   "hours_basis_batch.  IG only: probes, LEACE, bootstrap not included.  Models in "
                   "units_with_ig_not_timed (LaBraM, a stand-in) are not in the total.")
    return _save(ctx, "P9", out)


# ------------------------------------------------------------------ P12
def p12_inputs(units: list, extra: list, exclude: list) -> pd.DataFrame:
    """PILOT_LOG 15.4 D2: P8 units + extra seeds (P8b), minus the excluded cells (alpha 1.5 a*).
    Duplicated (model, cell, seed) rows are an error, not silently averaged."""
    rows = list(units) + [u for e in extra for u in e]
    df = pd.DataFrame(rows)
    if df.duplicated(["model", "cell", "seed"]).any():
        raise ValueError("duplicate (model, cell, seed) in P12 inputs")
    return df[~df["cell"].isin(list(exclude))].reset_index(drop=True)


def p12(ctx: Context, n_sim: int = 200) -> dict:
    from ..stats.power import simulate_h3
    import json
    cfg = ctx.cfg
    p8r = json.load(open(ctx.results / "pilots" / "P8.json", encoding="utf-8"))
    df = p12_inputs(p8r["units"], [json.load(open(ctx.results / "pilots" / f"{n}.json", encoding="utf-8"))["units"]
                                   for n in cfg["stats"].get("p12_extra_results", [])],
                    cfg["stats"].get("p12_exclude_cells", []))
    sd_seed = float(df.groupby(["model", "cell"])["delta_ba_neu"].std().mean())
    cell_means = df.groupby("cell")["delta_ba_neu"].mean().values
    grid = []
    for beta, sd_metric in ((1.0, 0.5 * sd_seed), (1.0, sd_seed), (0.5, sd_seed)):
        r = simulate_h3(len(cfg["design"]["core_models"]), np.resize(cell_means, 9), 2, sd_seed, sd_seed,
                        beta, sd_metric, cfg["stats"]["h3_min_corr_lower"], n_sim=n_sim)
        grid.append({"beta": beta, "sd_metric": sd_metric, **r})
    return _save(ctx, "P12", {"sd_seed": sd_seed, "cell_means": cell_means.tolist(), "power_grid": grid,
                              "cells_used": sorted(df["cell"].unique().tolist()),
                              "excluded": list(cfg["stats"].get("p12_exclude_cells", [])),
                              "note": "set stats.sesoi_* and stats.h3_min_corr_lower from this grid"})


# ------------------------------------------------------------------ P10a / P11
def p10a(ctx: Context, npz_path: str, px_per_deg: float | None = None) -> dict:
    from ..data.external import eegeyenet_saccade_amplitudes
    a = eegeyenet_saccade_amplitudes(npz_path, px_per_deg)
    return _save(ctx, "P10a", {"n": int(a.size), "median": float(np.median(a)),
                               "quantiles": {q: float(np.percentile(a, q)) for q in (5, 25, 50, 75, 95)},
                               "unit": "deg" if px_per_deg else "label units",
                               "note": "paradigm/montage/reference differ from EEGMMIDB: approximate reference only"})


def p11(ctx: Context, n_subjects: int = 25, n_sessions: int = 5) -> dict:
    from ..data.external import load_shu
    rows = []
    for s in range(1, n_subjects + 1):
        for ses in range(1, n_sessions + 1):
            try:
                e = load_shu(ctx.cfg.resolve_path("shu_root"), s, ses)
                rows.append({"subject": s, "session": ses, "shape": list(e.X.shape), "labels": np.bincount(e.y).tolist()})
            except Exception as ex:  # noqa: BLE001
                rows.append({"subject": s, "session": ses, "error": str(ex)})
    return _save(ctx, "P11", {"files": rows, "continuous_available": False,
                              "note": "public release is epoched (verify); channel list and filter band to confirm"})
