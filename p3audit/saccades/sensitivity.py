"""3.7 detection-sensitivity curve (lower bound on the miss rate).

Saccades of decreasing amplitude are injected into background windows that
contain no detected real saccade (real return saccades removed), using
leave-one-subject-out templates with per-trial topography jitter.  Because
injection and detection share AF7/AF8, the result is only a *lower bound* on
the true miss rate.
"""
from __future__ import annotations

import numpy as np

from ..data.windows import t0_windows
from ..generator.inject import cosine_envelope, saccade_delta
from ..utils.common import rng as keyed_rng
from .detect import detect_all, heog, saccade_mask, velocity_threshold


def sensitivity_curve(source, subjects, runs, pool: dict[int, np.ndarray], amplitudes, cfg,
                      threshold_by_subject: dict | None = None, guard=None) -> list[dict]:
    rows = []
    ref = cfg["physionet"]["original_reference"]
    for s in subjects:
        if guard:
            guard.record(s, runs, "pilot_sensitivity_curve")
        loo = sorted(k for k in pool if k != s)
        if not loo:
            continue
        for r_id in runs:
            run = source.load(s, r_id)
            sf = run.sfreq
            thr = (threshold_by_subject or {}).get(s) or velocity_threshold(heog(run, cfg), sf, cfg)
            mask = saccade_mask(run, cfg, thr)
            wins, _ = t0_windows(run, cfg, mask)
            if not wins:
                continue
            for amp in amplitudes:
                x = run.data.copy()
                onsets = []
                for w in wins:
                    r = keyed_rng("sens", s, r_id, w.t0_index, amp)
                    env = cosine_envelope(w.length, int(0.25 * sf))
                    onset = int(r.uniform(0.3, w.length / sf - 0.8) * sf)
                    tid = int(r.choice(loo))
                    d = saccade_delta(pool[tid], run.ch_names, sf, w.length + 2 * w.pad, w.pad, env, onset, cfg, r,
                                      rightward=bool(r.random() < 0.5), reference=ref)
                    a, b = w.region
                    x[:, a:b] += amp * d
                    onsets.append(w.start + onset)
                from ..data.sources import RunData
                inj = RunData(run.subject, run.run, sf, x, run.ch_names, run.events)
                sacs = detect_all(heog(inj, cfg), sf, cfg, thr)
                tol = int(0.15 * sf)
                hits = sum(any(abs(sc.onset - o) <= tol for sc in sacs) for o in onsets)
                rows.append({"subject": s, "run": r_id, "amplitude_uv": amp, "n": len(onsets), "hits": hits,
                             "hit_rate": hits / len(onsets)})
    return rows
