"""Diagnostic: on which segments should the mu component's 8-13 Hz peak be tested?

`diag_mu_search.py` refuted the search-budget hypothesis (budget 6 -> 40 recovered one side
out of thirty).  The binding criterion is `band_peak`: the component spectrum must rise at
least 1 dB above a 2-30 Hz power-law fit somewhere in 8-13 Hz.

estimate_mu currently evaluates that spectrum on the WHOLE execution run — task periods and
T0 rest concatenated.  Two things about that are worth measuring before touching anything:

  1. Mu is suppressed during movement by construction (that suppression is the ERD the GED
     was built to find).  Concatenating task with rest dilutes exactly the peak being tested.
     3.4.2 says "the component spectrum has a peak in 8-13 Hz" without naming the segments;
     testing it where the rhythm is expected to be present is the faithful reading.
  2. The aperiodic fit spans 2-30 Hz, which includes the 20-30 Hz range where movement EMG
     sits during execution runs.  EMG flattens the fitted slope and lifts the fitted line
     under 8-13 Hz, so a real mu bump can fall below the 1 dB bar for a reason that has
     nothing to do with mu.  The separate `emg_not_rising` check only catches the extremes
     (7 of 210 sides), not sub-threshold contamination of the fit.

This script changes NO criterion and writes NO components.  For every GED candidate whose
topographic peak lies in the right hemisphere's neighbourhood, it recomputes the band-peak
residual three ways — on the concatenation (status quo), on T0 rest only, on task only —
and reports how many sides clear 1 dB under each, plus the 20-40 Hz slope on each segment
set so the EMG story can be checked rather than assumed.

Decide the specification from the physiology, not from the pass counts: if rest-only and
concatenated give the same answer, the criterion is not the problem and 3.4.2's inclusion
rule genuinely needs rewriting (its own branch).  If they differ sharply, the concatenation
was measuring mu where mu is suppressed, and that is a defect in the implementation of the
criterion rather than a change to it.

Usage:
    python diag_mu_bandpeak.py              # pilot subjects (fast)
    python diag_mu_bandpeak.py all          # every subject in the split

Writes results/pilots/mu_bandpeak_diag.json .
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from p3audit.config import load_config
from p3audit.data.streams import fir
from p3audit.experiments.context import Context
from p3audit.generator.components import (_task_rest_segments, band_peak, component_psd, ged,
                                          pattern, rising_high_band, seg_cov)
from p3audit.utils.common import ch_index

SCOPE = sys.argv[1] if len(sys.argv) > 1 else "pilot"
MAX_RANK = 16          # how deep to look for a topographically plausible candidate
SEGMENTS = ("all", "rest", "task")


def main() -> int:
    cfg = load_config(None, {})
    tc = cfg["task_component"]
    band = tuple(tc["band"])
    ctx = Context.create(cfg)
    split = ctx.require_split()
    subjects = ctx.all_subjects if SCOPE == "all" else split.pilot

    rows, t0 = [], time.time()
    for i, s in enumerate(subjects, 1):
        runs = [ctx.source.load(s, r) for r in (3, 7, 11)]
        sf = runs[0].sfreq
        ch = runs[0].ch_names
        Cr, Ct = [], []
        parts = {k: [] for k in SEGMENTS}
        for r in runs:
            xb = fir(r.data, sf, *band, trans=tc["fir_transition_hz"])
            rest, task = _task_rest_segments(r)
            Cr.append(seg_cov(xb, rest))
            Ct.append(seg_cov(xb, task))
            b = fir(r.data, sf, 1.0, 45.0)
            parts["all"].append(b)
            parts["rest"].append(np.concatenate([b[:, a:c] for a, c in rest], axis=1))
            parts["task"].append(np.concatenate([b[:, a:c] for a, c in task], axis=1))
        C_rest, C_task = np.mean(Cr, 0), np.mean(Ct, 0)
        C_all = (C_rest + C_task) / 2
        evals, W = ged(C_rest, C_task, tc["reg"])
        B = {k: np.concatenate(v, axis=1) for k, v in parts.items()}

        for side, neigh, anchor in (("mu_L", tc["left_neighborhood"], "C3"),
                                    ("mu_R", tc["right_neighborhood"], "C4")):
            nidx = set(ch_index(ch, neigh).tolist())
            ai = ch_index(ch, [anchor])[0]
            best = None
            for k in range(min(MAX_RANK, W.shape[1])):
                w = W[:, k]
                a = pattern(C_all, w)
                if a[ai] < 0:
                    w, a = -w, -a
                if np.argmax(np.abs(a)) not in nidx:
                    continue
                d = {"subject": s, "side": side, "rank": k, "eigval": float(evals[k]),
                     "role": split.role(s), "pilot": s in split.pilot}
                for name, x in B.items():
                    f, p = component_psd(x, w, sf)
                    ok, db = band_peak(f, p, band)
                    d[f"db_{name}"] = float(db)
                    d[f"ok_{name}"] = bool(ok)
                    d[f"emg_slope_{name}"] = float(rising_high_band(f, p, tuple(tc["emg_band"]))[1])
                # keep the candidate that looks best on rest, the segment where mu should exist
                if best is None or d["db_rest"] > best["db_rest"]:
                    best = d
            if best is not None:
                rows.append(best)
        if i % 10 == 0 or i == len(subjects):
            el = time.time() - t0
            print(f"[{i:3d}/{len(subjects)}] {el/60:.1f} min, ETA {(len(subjects)-i)*el/i/60:.1f} min",
                  flush=True)

    summary = {}
    for name in SEGMENTS:
        v = [r[f"db_{name}"] for r in rows]
        summary[name] = {
            "median_db": float(np.median(v)) if v else None,
            "p25_db": float(np.percentile(v, 25)) if v else None,
            "p75_db": float(np.percentile(v, 75)) if v else None,
            "sides_clearing_1db": int(sum(r[f"ok_{name}"] for r in rows)),
            "median_emg_slope": float(np.median([r[f"emg_slope_{name}"] for r in rows])) if rows else None,
        }
    by_subj = {}
    for r in rows:
        by_subj.setdefault(r["subject"], {})[r["side"]] = r
    both = {name: sum(1 for d in by_subj.values()
                      if len(d) == 2 and all(x[f"ok_{name}"] for x in d.values()))
            for name in SEGMENTS}

    out = {"scope": SCOPE, "n_subjects": len(subjects), "n_sides_with_a_candidate": len(rows),
           "max_rank_searched": MAX_RANK, "per_segment": summary,
           "subjects_with_both_sides_clearing_1db": both, "per_side": rows,
           "note": "diagnostic only; no criterion changed, results/components/ untouched"}
    p = ctx.results / "pilots" / "mu_bandpeak_diag.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")

    print(f"\n{len(rows)} sides with a topographically plausible candidate "
          f"(searched to rank {MAX_RANK}), {len(subjects)} subjects\n")
    print(f"{'PSD on':>8} {'median dB':>10} {'IQR':>16} {'clears 1 dB':>13} {'both sides':>11} {'EMG slope':>10}")
    for name in SEGMENTS:
        s = summary[name]
        print(f"{name:>8} {s['median_db']:>10.2f} {s['p25_db']:>7.2f}..{s['p75_db']:<7.2f} "
              f"{s['sides_clearing_1db']:>7}/{len(rows):<5} {both[name]:>7}/{len(subjects):<3} "
              f"{s['median_emg_slope']:>10.2f}")
    print(f"\nwritten: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
