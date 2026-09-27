"""Horizontal saccade detection on the eye-probe stream (0.1 Hz HP, native rate,
original reference; AF7 − AF8), trial classification for the natural
counterfactual (3.7) and return-saccade statistics for the T0 trim (3.4.1/P2)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data.streams import fir
from ..utils.common import ch_index


@dataclass
class Saccade:
    onset: int          # sample
    offset: int
    amplitude: float    # µV on AF7−AF8 (signed)
    direction: int      # +1 rightward, −1 leftward


def heog(run, cfg) -> np.ndarray:
    i7, i8 = ch_index(run.ch_names, cfg["streams"]["eye"]["heog"])
    x = fir(run.data[[i7, i8]], run.sfreq, cfg["streams"]["eye"]["l_freq"], None)
    return x[0] - x[1]


def step_statistic(h: np.ndarray, sfreq: float, cfg) -> np.ndarray:
    """d(t) = mean(h[t:t+w]) − mean(h[t−w:t]) on the low-passed HEOG: a matched step
    detector that is far more robust to EEG background than a raw velocity threshold."""
    sd = cfg["saccade_detection"]
    hl = fir(h, sfreq, None, sd["lowpass"])
    w = max(2, int(round(sd["step_window_ms"] / 1000 * sfreq)))
    c = np.concatenate([[0.0], np.cumsum(hl)])
    n = hl.size
    d = np.zeros(n)
    t = np.arange(w, n - w)
    d[t] = (c[t + w] - c[t]) / w - (c[t] - c[t - w]) / w
    return d


def velocity_threshold(h: np.ndarray, sfreq: float, cfg) -> float:
    """Detection threshold on |d(t)|: max(k · MAD(d), min_amplitude_uv)."""
    sd = cfg["saccade_detection"]
    d = step_statistic(h, sfreq, cfg)
    mad = np.median(np.abs(d - np.median(d))) * 1.4826
    return float(max(sd["step_mad_k"] * mad, sd["min_amplitude_uv"]))


def detect_all(h: np.ndarray, sfreq: float, cfg, threshold: float | None = None) -> list[Saccade]:
    """Peaks of |d(t)| above threshold (non-maximum suppression over ±200 ms); onset/offset
    where the low-passed velocity falls below 20 % of its peak around the step."""
    sd = cfg["saccade_detection"]
    d = step_statistic(h, sfreq, cfg)
    thr = threshold if threshold is not None else velocity_threshold(h, sfreq, cfg)
    hl = fir(h, sfreq, None, sd["lowpass"])
    v = np.abs(np.gradient(hl))
    w = max(2, int(round(sd["step_window_ms"] / 1000 * sfreq)))
    nms = int(0.2 * sfreq)
    sign_right = cfg["saccade"]["rightward_sign"]
    cand = np.where(np.abs(d) > thr)[0]
    out: list[Saccade] = []
    taken = np.zeros(d.size, dtype=bool)
    for i in cand[np.argsort(-np.abs(d[cand]))]:
        if taken[i]:
            continue
        taken[max(0, i - nms): i + nms] = True
        lo, hi = max(0, i - w), min(d.size, i + w)
        pk = lo + int(np.argmax(v[lo:hi]))
        on = pk
        while on > lo and v[on] > 0.2 * v[pk]:
            on -= 1
        off = pk
        while off < hi - 1 and v[off] > 0.2 * v[pk]:
            off += 1
        amp = float(d[i])
        out.append(Saccade(on, off, amp, int(np.sign(amp) * sign_right)))
    return sorted(out, key=lambda s: s.onset)


def saccade_mask(run, cfg, threshold=None) -> np.ndarray:
    h = heog(run, cfg)
    m = np.zeros(h.size, dtype=bool)
    for s in detect_all(h, run.sfreq, cfg, threshold):
        m[s.onset:s.offset + 1] = True
    return m


def first_in_window(sacs: list[Saccade], onset: int, sfreq: float, window) -> Saccade | None:
    a, b = onset + int(window[0] * sfreq), onset + int(window[1] * sfreq)
    for s in sacs:
        if a <= s.onset < b:
            return s
    return None


def classify_trials(run, cfg, threshold=None) -> list[dict]:
    """L/R imagery trials -> congruent / incongruent / none (3.7).  T1 = left target."""
    h = heog(run, cfg)
    sacs = detect_all(h, run.sfreq, cfg, threshold)
    rows = []
    for k, (o, d, s) in enumerate(run.event_samples(("T1", "T2"))):
        side = -1 if s == "T1" else 1
        sac = first_in_window(sacs, o, run.sfreq, cfg["saccade_detection"]["window"])
        if sac is None:
            kind = "none"
        else:
            kind = "congruent" if sac.direction == side else "incongruent"
        rows.append({"subject": run.subject, "run": run.run, "trial": k, "onset": o, "target_side": side,
                     "label": 0 if s == "T1" else 1, "type": kind,
                     "amplitude": sac.amplitude if sac else np.nan,
                     "latency_s": (sac.onset - o) / run.sfreq if sac else np.nan})
    return rows


def return_saccades(run, cfg, threshold=None) -> list[dict]:
    """Saccades after target offset (T0 onsets that follow a task period): latency + duration."""
    h = heog(run, cfg)
    sacs = detect_all(h, run.sfreq, cfg, threshold)
    rows = []
    t0s = run.event_samples("T0")
    for k, (o, d, _) in enumerate(t0s):
        if k == 0:
            continue
        sac = first_in_window(sacs, o, run.sfreq, cfg["saccade_detection"]["return_saccade_window"])
        rows.append({"subject": run.subject, "run": run.run, "t0": k, "detected": sac is not None,
                     "latency_s": (sac.onset - o) / run.sfreq if sac else np.nan,
                     "duration_s": (sac.offset - sac.onset) / run.sfreq if sac else np.nan,
                     "amplitude": sac.amplitude if sac else np.nan})
    return rows


def trim_ms_from_returns(rows: list[dict], q: float = 95.0) -> float:
    lat = np.array([r["latency_s"] + r["duration_s"] for r in rows if r["detected"]])
    return float(np.percentile(lat, q) * 1000) if lat.size else float("nan")
