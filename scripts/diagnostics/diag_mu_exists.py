"""Diagnostic: do these subjects have a detectable mu rhythm at all?

Two hypotheses about the 30/105 mu inclusion rate have now failed:
  - `n_candidates` as a binding search budget (6 -> 40 recovered one side out of thirty);
  - the band-peak criterion being evaluated on task+rest concatenated rather than rest, and
    the 2-30 Hz aperiodic fit being flattened by movement EMG (rest-only moves the median
    from 1.56 to 2.50 dB and recovers two sides; EMG slopes are solidly negative, median
    -1.8, so the spectra are not myogenic).

Both were guesses about the estimator.  This script asks the prior question, with no
estimator involved at all: taking the sensorimotor channels directly, is there an 8-13 Hz
peak in the rest segments of the execution runs?

For each subject it measures the band-peak residual (the same `band_peak` used by the
inclusion rule, same 1 dB bar, same 2-30 Hz fit) on:

  C3, C4                       raw channels
  C3lap, C4lap                 small surface Laplacian, centre minus the mean of its
                               four nearest neighbours - removes the volume-conducted
                               posterior alpha that swamps raw central channels
  Oz, POz, Ozlap               POSITIVE CONTROL.  The alpha component passes in 90 of 105
                               subjects, so the machinery must find peaks here.  If the
                               occipital sites clear 1 dB at a high rate and the central
                               ones do not, `band_peak` works and mu is genuinely weak;
                               if the occipital sites also fail, the measurement is broken
                               and nothing about mu can be concluded from it.

Reading the result:
  - central sites peaky in most subjects  -> the components exist and the GED estimator is
    the problem.  A rest-vs-task GED maximises a band-limited POWER RATIO; the criterion
    demands a spectral PEAK, and a broadband-ERD component can do the former without the
    latter.  The fix is then to align the estimator with the criterion (an SSD-style
    centre-vs-flank objective, or SSD first and rest>task selection second), which is a
    principled change rather than a loosened bar.
  - central sites flat in most subjects -> mu is genuinely not detectable in this dataset at
    this bar, and 3.4.2's own branch applies: redefine the inclusion criteria or the band,
    with this measurement as the justification.

Writes results/pilots/mu_exists_diag.json .  Changes nothing.

Usage:
    python diag_mu_exists.py            # pilot subjects
    python diag_mu_exists.py all        # every subject in the split
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from p3audit.config import load_config
from p3audit.data.streams import fir
from p3audit.experiments.context import Context
from p3audit.generator.components import _task_rest_segments, band_peak
from p3audit.utils.common import ch_index

SCOPE = sys.argv[1] if len(sys.argv) > 1 else "pilot"

# centre -> four nearest neighbours for the small Laplacian
LAPLACIAN = {
    "C3": ["FC3", "CP3", "C1", "C5"],
    "C4": ["FC4", "CP4", "C2", "C6"],
    "Oz": ["O1", "O2", "POz", "Iz"],
}
SITES = ["C3", "C4", "Oz", "POz"]
CENTRAL = ["C3", "C4", "C3lap", "C4lap"]
OCCIPITAL = ["Oz", "POz", "Ozlap"]


def psd_peak(x, sfreq, band):
    from p3audit.generator.components import component_psd
    f, p = component_psd(x[None, :], np.array([1.0]), sfreq)
    return band_peak(f, p, band)


def main() -> int:
    cfg = load_config(None, {})
    band = tuple(cfg["task_component"]["band"])
    ctx = Context.create(cfg)
    split = ctx.require_split()
    subjects = ctx.all_subjects if SCOPE == "all" else split.pilot

    rows, t0 = [], time.time()
    for i, s in enumerate(subjects, 1):
        runs = [ctx.source.load(s, r) for r in (3, 7, 11)]
        sf = runs[0].sfreq
        ch = runs[0].ch_names
        rest_parts = []
        for r in runs:
            b = fir(r.data, sf, 1.0, 45.0)
            rest, _ = _task_rest_segments(r)
            rest_parts.append(np.concatenate([b[:, a:c] for a, c in rest], axis=1))
        X = np.concatenate(rest_parts, axis=1)

        row = {"subject": s, "role": split.role(s), "pilot": s in split.pilot,
               "rest_seconds": float(X.shape[1] / sf)}
        for site in SITES:
            idx = ch_index(ch, [site])[0]
            ok, db = psd_peak(X[idx], sf, band)
            row[site] = {"db": float(db), "ok": bool(ok)}
        for centre, neigh in LAPLACIAN.items():
            ci = ch_index(ch, [centre])[0]
            ni = ch_index(ch, neigh)
            ok, db = psd_peak(X[ci] - X[ni].mean(0), sf, band)
            row[f"{centre}lap"] = {"db": float(db), "ok": bool(ok)}
        rows.append(row)
        if i % 10 == 0 or i == len(subjects):
            el = time.time() - t0
            print(f"[{i:3d}/{len(subjects)}] {el/60:.1f} min, ETA {(len(subjects)-i)*el/i/60:.1f} min",
                  flush=True)

    keys = CENTRAL + OCCIPITAL
    summary = {}
    for k in keys:
        v = [r[k]["db"] for r in rows]
        summary[k] = {"median_db": float(np.median(v)), "p25_db": float(np.percentile(v, 25)),
                      "p75_db": float(np.percentile(v, 75)),
                      "n_clearing_1db": int(sum(r[k]["ok"] for r in rows)), "n": len(rows)}
    both_lap = sum(1 for r in rows if r["C3lap"]["ok"] and r["C4lap"]["ok"])

    out = {"scope": SCOPE, "n_subjects": len(subjects), "band": list(band),
           "per_site": summary, "subjects_with_both_central_laplacians_peaky": both_lap,
           "per_subject": rows,
           "note": "estimator-free; no criterion changed, nothing written to results/components/"}
    p = ctx.results / "pilots" / "mu_exists_diag.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")

    print(f"\n{len(rows)} subjects, rest segments of runs 3/7/11, "
          f"{np.median([r['rest_seconds'] for r in rows]):.0f} s of rest per subject\n")
    print(f"{'site':>8} {'median dB':>10} {'IQR':>16} {'clears 1 dB':>14}")
    for k in keys:
        s = summary[k]
        tag = "  <- control" if k in OCCIPITAL else ""
        print(f"{k:>8} {s['median_db']:>10.2f} {s['p25_db']:>7.2f}..{s['p75_db']:<7.2f} "
              f"{s['n_clearing_1db']:>6}/{s['n']:<6}{tag}")
    print(f"\nboth central Laplacians peaky: {both_lap}/{len(rows)}")
    print(f"written: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
