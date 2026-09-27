"""3.1 metadata-only exclusion rules (executable form).

A subject is excluded when any rule fires:
  R1  sampling rate != required_sfreq (catches 88 and any other 128-Hz file)
  R2  any imagery run (4, 8, 12, 6, 10, 14) missing
  R3  an imagery run's trial count differs from the modal count by > max_trial_count_dev
  R4  an imagery run's duration differs from the modal duration by > max_duration_dev_s
Only `RunSource.header` is called — no signal is read.
"""
from __future__ import annotations

from collections import Counter

import pandas as pd

from ..constants import IMAGERY_RUNS
from .sources import RunSource


def _mode(values):
    return Counter(values).most_common(1)[0][0]


def exclusion_table(source: RunSource, cfg, subjects=None) -> pd.DataFrame:
    subjects = subjects or source.subjects()
    ex = cfg["exclusion"]
    need_runs = sorted(set(IMAGERY_RUNS) | {1, 2, 3, 7, 11})
    rows = []
    for s in subjects:
        for r in need_runs:
            h = source.header(s, r)
            rows.append({"subject": s, "run": r, "present": h is not None,
                         "sfreq": h.sfreq if h else None,
                         "n_trials": h.n_trials() if h else None,
                         "duration": round(h.duration, 3) if h else None})
    df = pd.DataFrame(rows)
    imag = df[df.run.isin(IMAGERY_RUNS) & df.present]
    mode_trials = {r: _mode(g.n_trials.tolist()) for r, g in imag.groupby("run")}
    mode_dur = {r: _mode(g.duration.round(0).tolist()) for r, g in imag.groupby("run")}

    out = []
    for s, g in df.groupby("subject"):
        reasons = []
        present = g[g.present]
        if (present.sfreq != ex["required_sfreq"]).any():
            reasons.append(f"R1 sfreq {sorted(set(present.sfreq))}")
        gi = g[g.run.isin(IMAGERY_RUNS)]
        missing = gi[~gi.present].run.tolist()
        if missing:
            reasons.append(f"R2 missing imagery runs {missing}")
        for _, row in gi[gi.present].iterrows():
            if abs(row.n_trials - mode_trials[row.run]) > ex["max_trial_count_dev"]:
                reasons.append(f"R3 run {row.run} trials {row.n_trials} vs mode {mode_trials[row.run]}")
            if abs(row.duration - mode_dur[row.run]) > ex["max_duration_dev_s"]:
                reasons.append(f"R4 run {row.run} duration {row.duration:.1f}s vs mode {mode_dur[row.run]}")
        # runs needed by the generator but not by the exclusion rule are recorded, not excluded
        gen_missing = g[g.run.isin([1, 2, 3, 7, 11]) & ~g.present].run.tolist()
        out.append({"subject": s, "excluded": bool(reasons), "reasons": "; ".join(reasons),
                    "generator_runs_missing": gen_missing})
    return pd.DataFrame(out)


def included_subjects(table: pd.DataFrame) -> list[int]:
    return sorted(table.loc[~table.excluded, "subject"].astype(int).tolist())
