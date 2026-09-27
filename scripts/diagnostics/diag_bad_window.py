"""Why do some subjects lose 100% of background windows to bad_window (max_abs_uv)?

READING DECLARED BEFORE RUNNING (2026-09-19):
  - If one or two channels exceed max_abs_uv in (nearly) every window while the rest stay
    well below, the cause is a persistently bad channel, not a noisy recording: the whole-window
    rule is then discarding 62+ good channels because of one.
  - If exceedance is spread over many channels, the recording itself is bad and exclusion
    is the right outcome.
  This script only describes; it does not change any rule.
  SCOPE: pilot subjects only (all 15).  Deciding bad_window is an exclusion check, and the
  outline (3.1) restricts signal statistics for exclusion checks to the pilot subjects, so the
  access guard is used with the pilot-only purpose 'pilot_t0_stats'.  Of the 7 subjects that
  lost every window, only 53 is a pilot subject; the other six are not inspected here.
Run:  python diag_bad_window.py
Out:  results/pilots/bad_window_diag.json
"""
from __future__ import annotations

import numpy as np

from p3audit.config import load_config
from p3audit.constants import IMAGERY_RUNS
from p3audit.data.streams import apply_stream, out_index, stream_specs
from p3audit.data.windows import amp_channels, t0_windows
from p3audit.experiments.context import Context
from p3audit.saccades.detect import saccade_mask
from p3audit.utils.common import write_json

LOST_ALL = {13, 32, 53, 60, 69, 77, 109}   # from prepare --stage cache, 2026-09-19


def subject_stats(ctx, s, cfg):
    spec = stream_specs(cfg)["model"]
    thr = cfg["background"]["bad_window"]["max_abs_uv"]
    use_mask = cfg["background"]["trim_mode"] in ("drop_saccade_windows", "trim_and_drop")
    peaks, names = [], None
    ctx.guard.record(s, IMAGERY_RUNS, "pilot_t0_stats")   # pilot-only purpose: raises on a non-pilot subject
    for run_id in IMAGERY_RUNS:
        run = ctx.source.load(s, run_id)
        names = list(run.ch_names)
        mask = saccade_mask(run, cfg) if use_mask else None
        wins, _ = t0_windows(run, cfg, mask)
        if not wins:
            continue
        X, _ = apply_stream(run.data, run.sfreq, spec)
        for w in wins:
            core = X[:, out_index(w.start, run.sfreq, spec): out_index(w.start + w.length, run.sfreq, spec)]
            peaks.append(np.abs(core).max(axis=1))
    P = np.array(peaks)                       # windows x channels, µV
    ocular_kept = int((P[:, amp_channels(names, cfg)].max(axis=1) <= thr).sum())   # 2026-09-19 rule
    frac = (P > thr).mean(axis=0)
    order = np.argsort(-frac)
    top = [{"ch": names[i], "frac_windows_over": round(float(frac[i]), 3),
            "median_peak_uv": round(float(np.median(P[:, i])), 1)} for i in order[:5]]
    # windows that would pass if the k worst channels were ignored (descriptive only)
    survive = {}
    for k in (0, 1, 2, 3):
        keep = np.ones(P.shape[1], bool); keep[order[:k]] = False
        survive[k] = int((P[:, keep].max(axis=1) <= thr).sum())
    return {"n_windows": int(P.shape[0]), "n_channels_over_in_half_of_windows": int((frac > 0.5).sum()),
            "windows_passing_rule_ocular_excluded": ocular_kept,
            "median_peak_all_channels_uv": round(float(np.median(P)), 1),
            "worst_channels": top, "windows_passing_if_k_worst_channels_ignored": survive}


def main():
    cfg = load_config(None, {})
    ctx = Context.create(cfg)
    out = {}
    for s in sorted(ctx.require_split().pilot):
        r = subject_stats(ctx, s, cfg)
        out[s] = {"group": "lost_all" if s in LOST_ALL else "pilot", **r}
        w = r["worst_channels"][0]
        print(f"sub {s:3d} [{out[s]['group']:8s}] n={r['n_windows']:3d}  ch>50%: {r['n_channels_over_in_half_of_windows']:2d}  "
              f"worst {w['ch']} {w['frac_windows_over']:.2f} (median {w['median_peak_uv']} uV)  "
              f"pass if ignore 0/1/2/3 worst: {list(r['windows_passing_if_k_worst_channels_ignored'].values())}  "
              f"NEW RULE: {r['windows_passing_rule_ocular_excluded']}")
    write_json(ctx.results / "pilots" / "bad_window_diag.json", out)


if __name__ == "__main__":
    main()
