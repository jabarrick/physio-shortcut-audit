"""3.4.1 background windows from T0 rest segments of the imagery runs,
pseudo-labels, and the 3.8 half-split by background-window ID."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..utils.common import rng


@dataclass
class WindowSpec:
    subject: int
    run: int
    t0_index: int            # ordinal of the T0 segment within the run
    start: int               # core start sample (native rate, on alignment grid)
    length: int              # core length (native samples)
    pad: int                 # padding each side (native samples)

    @property
    def wid(self) -> str:
        return f"{self.subject:03d}-{self.run:02d}-{self.t0_index:02d}"

    @property
    def region(self) -> tuple[int, int]:
        return self.start - self.pad, self.start + self.length + self.pad


def t0_windows(run, cfg, saccade_mask: np.ndarray | None = None) -> tuple[list[WindowSpec], dict]:
    """Cut background windows from T0 segments.

    trim_mode = 'trim_start'           : drop the first X ms of each T0 (return-saccade latency+duration
                                          95th pct from P2), take one window if the remainder fits.
    trim_mode = 'drop_saccade_windows' : no trimming; windows overlapping a detected saccade
                                          (`saccade_mask`, native-rate boolean) are dropped.
    trim_mode = 'trim_and_drop'        : both — trim the first X ms *and* drop whatever still overlaps
                                          a detected saccade.  P2 (2026-09-18) forced this third mode:
                                          T0 segments run ~4.1 s, so the 95th-percentile trim of
                                          ~1.39 s leaves no room even for a 3-s window and
                                          'trim_start' yields zero windows on this dataset; but
                                          'drop_saccade_windows' alone keeps windows that start at
                                          target offset, exactly where the return saccade falls, and
                                          the detection rate is only ~0.32 with a miss rate the
                                          sensitivity curve bounds only from below.  A sub-95th trim
                                          removes the bulk of the return saccade, the mask removes
                                          the tail.  X must satisfy X + window_s <= min T0 duration.
    Windows whose padded region leaves the run are dropped (edge filtering).
    """
    bg = cfg["background"]
    sf = run.sfreq
    L = int(round(bg["window_s"] * sf))
    pad = int(round(cfg["streams"]["pad_s"] * sf))
    grid = int(bg["align_samples"])
    trim = int(round(bg["trim_ms"] / 1000 * sf)) if bg["trim_mode"] in ("trim_start", "trim_and_drop") else 0
    n = run.data.shape[1]
    # the mask is honoured only by the modes that drop; ignore whatever the caller passed otherwise
    if bg["trim_mode"] not in ("drop_saccade_windows", "trim_and_drop"):
        saccade_mask = None
    out, stats = [], {"t0_segments": 0, "too_short": 0, "edge": 0, "saccade": 0}
    for k, (o, d, s) in enumerate(run.event_samples("T0")):
        stats["t0_segments"] += 1
        start = o + trim
        start = int(np.ceil(start / grid) * grid)
        if start + L > o + d:
            stats["too_short"] += 1
            continue
        w = WindowSpec(run.subject, run.run, k, start, L, pad)
        a, b = w.region
        # strict: a padded region that merely *touches* the recording boundary is not enough.  The
        # filters in data/streams.py handle the first and last samples of a run by their own edge
        # convention, so stream(x + Δ) == stream(x) + stream(Δ) — the exact superposition that
        # 3.4.2/3.4.3 rely on, and that stream_window_deltas implements — fails there by ~1e-2
        # relative instead of ~1e-7.  Costs at most the first and last window of a run.
        if a <= 0 or b >= n:
            stats["edge"] += 1
            continue
        if saccade_mask is not None and saccade_mask[start:start + L].any():
            stats["saccade"] += 1
            continue
        out.append(w)
    return out, stats


def amp_channels(ch_names, cfg) -> np.ndarray:
    """Boolean mask of channels the max_abs rule applies to.  max_abs_exclude: ocular drops
    saccade.ocular_sites (blinks/eye movements are physiological); none = v8.1 (all channels)."""
    bw = cfg["background"]["bad_window"]
    keep = np.ones(len(ch_names), bool)
    if bw.get("max_abs_exclude", "none") == "ocular":
        oc = set(cfg["saccade"]["ocular_sites"])
        keep = np.array([c not in oc for c in ch_names])
    return keep


def robust_z(x: np.ndarray) -> np.ndarray:
    """(x - median) / (1.4826 * MAD); 0 where MAD is 0."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    med = np.median(x)
    mad = 1.4826 * np.median(np.abs(x - med))
    return np.zeros_like(x) if mad == 0 else (x - med) / mad


def bad_window_reason(core: np.ndarray, cfg, amp_mask: np.ndarray | None = None, amp: bool = True) -> str:
    """Prereg item 6 on the model-stream core (µV): '' if usable, else 'max_abs' or 'flat'.
    Flat-signal check always covers every channel; max_abs only the channels in amp_mask."""
    bw = cfg["background"]["bad_window"]
    sub = core if amp_mask is None else core[amp_mask]
    if amp and np.abs(sub).max() > bw["max_abs_uv"]:
        return "max_abs"
    if core.std(axis=-1).min() < bw["min_std_uv"]:
        return "flat"
    return ""


def bad_window(core: np.ndarray, cfg, amp_mask: np.ndarray | None = None) -> bool:
    return bool(bad_window_reason(core, cfg, amp_mask))


def pseudo_labels(subject: int, n: int, seed: int) -> np.ndarray:
    """Balanced random labels independent of the background (3.4.1)."""
    y = np.arange(n) % 2
    return rng(seed, "pseudo", subject).permutation(y).astype(np.int64)


def half_assignment(window_ids: list[str], seed: int) -> np.ndarray:
    """3.8: split by background-window ID parity after a seeded permutation.
    Permutation and alternation are done within each subject, so every test
    subject contributes to both halves (the unit keeps the same subjects and
    labels; only background windows are halved).  Returns 0/1 per window;
    the same window always lands in the same half."""
    by_subj: dict[str, list[str]] = {}
    for w in window_ids:
        by_subj.setdefault(w.split("-")[0], []).append(w)
    rank = {}
    for subj, ws in by_subj.items():
        order = sorted(ws, key=lambda w: rng(seed, "half", w).random())
        rank.update({w: i for i, w in enumerate(order)})
    return np.array([rank[w] % 2 for w in window_ids], dtype=np.int64)
