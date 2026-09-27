"""Extended detection-sensitivity check (PILOT_LOG 15.14).  Written 2026-09-23, BEFORE running.
Pilot subjects only (access guard).  Descriptive; changes no parameter.

Why: `pilot sensitivity` (5-80 uV) found hit rate 0-16 %, even at 80 uV.  Either the detector's
per-subject threshold max(5*MAD, 15 uV) sits far above 80 uV (then the 149.34 uV reference, measured
on DETECTED saccades, is truncated much more strongly than assumed), or the sensitivity injection /
matching is broken.  Same code path, larger amplitudes, plus the thresholds themselves.

Pre-declared readings:
  R1  pooled hit rate at 150 uV >= 0.5 and at 300 uV >= 0.8 -> detector works; the low curve is the
      threshold.  Report thresholds; the 149.34 uV reference is 'strongly truncated' in 4.1.
  R2  pooled hit rate at 300 uV < 0.5 -> sensitivity injection or matching is broken; the
      5-80 uV curve is void until fixed.
  Anything in between: report as is, no reading.
"""
import json
import numpy as np
from p3audit.config import load_config
from p3audit.experiments.context import Context
from p3audit.saccades.detect import heog, velocity_threshold
from p3audit.saccades.sensitivity import sensitivity_curve

cfg = load_config()
ctx = Context.create(cfg)
pilot = ctx.require_split().pilot
runs = [4, 8, 12]
thr = {}
for s in pilot:
    ctx.guard.record(s, runs, "pilot_sensitivity_curve")
    thr[s] = [velocity_threshold(heog(ctx.source.load(s, r), cfg), ctx.source.load(s, r).sfreq, cfg) for r in runs]
amps = (80, 100, 150, 200, 300)
rows = sensitivity_curve(ctx.source, pilot, runs, ctx.load_pool(), amps, cfg, guard=ctx.guard)
curve = {a: sum(r["hits"] for r in rows if r["amplitude_uv"] == a) / max(1, sum(r["n"] for r in rows if r["amplitude_uv"] == a))
         for a in amps}
allthr = np.concatenate([np.asarray(v, float) for v in thr.values()])
out = {"curve": curve, "threshold_uv_by_subject": thr,
       "threshold_uv_summary": {"min": float(allthr.min()), "median": float(np.median(allthr)), "max": float(allthr.max())},
       "rows": rows}
json.dump(out, open("results/pilots/sensitivity_ext_diag.json", "w"), indent=1, default=float)
print(json.dumps({"curve": curve, "threshold_uv_summary": out["threshold_uv_summary"],
                  "threshold_median_by_subject": {s: float(np.median(v)) for s, v in thr.items()}}, indent=1))
