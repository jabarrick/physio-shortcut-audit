"""Per-subject basis cache for the semi-synthetic generator.

For each background window w of subject i the cache stores, in the *model
stream* (200 Hz, 0.5–75 Hz, notch, average reference, padded by pad_s):

  X0        streamed background
  alpha     stream(unit alpha Δ)            coefficient  exp(kσ_α/2) − 1
  mu        stream(unit task Δ, e·a·s)      coefficient  ∓s/2      (y=1 -> −s/2, y=0 -> +s/2)
            hemisphere: both restores the v8.1 pair mu_L/mu_R with coefficient −s on the
            hemisphere given by y.
  sacc_R/L  stream(unit 1-µV saccade Δ)     coefficient  amplitude (µV AF7−AF8)
  + optional extras (alpha band shift ±2 Hz, mismatch saccade, group templates)

Any condition is then  X = X0 + Σ coeff_b · B_b  — exact by linearity (streams.py).
Bases are computed in the raw 160-Hz domain on continuous runs and streamed with
zero padding ≥ the stream's impulse-response support; windows are streamed in groups
whose gaps exceed that support, so no Δ leaks into another window.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..constants import IMAGERY_RUNS
from ..data.streams import apply_stream, fir, out_index, stream_specs, stream_support_s
from ..data.windows import WindowSpec, amp_channels, bad_window_reason, pseudo_labels, robust_z, t0_windows
from ..utils.access import SignalAccessGuard, NULL_GUARD
from ..utils.common import rng as keyed_rng
from .components import Component, task_bases
from .inject import component_delta, cosine_envelope, perturb_template, saccade_delta

def core_bases(cfg) -> list[str]:
    """Always-present bases; the task bases depend on `task_component.hemisphere`."""
    return ["X0", "alpha", *task_bases(cfg), "sacc_R", "sacc_L"]


@dataclass
class SubjectCache:
    subject: int
    root: Path
    meta: dict

    def load(self, name: str, mmap: bool = True) -> np.ndarray:
        return np.load(self.root / f"{name}.npy", mmap_mode="r" if mmap else None)

    @property
    def n(self) -> int:
        return len(self.meta["wids"])

    @property
    def bases(self) -> list[str]:
        return self.meta["bases"]


def open_cache(cache_root: str | Path, subject: int) -> SubjectCache:
    root = Path(cache_root).expanduser() / f"sub-{subject:03d}"
    with open(root / "meta.json", encoding="utf-8") as f:
        return SubjectCache(subject, root, json.load(f))


# ------------------------------------------------------------------ helpers
def _groups(windows: list[WindowSpec], min_gap: int) -> list[list[int]]:
    order = sorted(range(len(windows)), key=lambda i: windows[i].start)
    groups: list[list[int]] = []
    last_end: list[int] = []
    for i in order:
        a, b = windows[i].region
        for gi, e in enumerate(last_end):
            if a - e >= min_gap:
                groups[gi].append(i); last_end[gi] = b
                break
        else:
            groups.append([i]); last_end.append(b)
    return groups


def stream_window_deltas(deltas: list[np.ndarray], windows: list[WindowSpec], n_run: int, sf: float,
                         spec, grid: int = 4) -> list[np.ndarray]:
    """Stream each window's region-Δ exactly as if embedded alone in the run."""
    support = stream_support_s(spec, sf)
    sup_n = int(np.ceil(support * sf / grid) * grid)
    edge = sup_n
    out: list[np.ndarray | None] = [None] * len(windows)
    for g in _groups(windows, sup_n):
        buf = np.zeros((deltas[g[0]].shape[0], n_run + 2 * edge))
        for i in g:
            a, b = windows[i].region
            buf[:, edge + a: edge + b] += deltas[i]
        y, _ = apply_stream(buf, sf, spec)
        for i in g:
            a, b = windows[i].region
            oa, ob = out_index(edge + a, sf, spec), out_index(edge + b, sf, spec)
            out[i] = y[:, oa:ob]
    return out  # type: ignore[return-value]


def sigma_log_power(series_by_window: list[np.ndarray], run_of: list[int], t_of: list[float]) -> float:
    """σ_α: within-subject SD of component log power across T0 windows after linear
    detrending within each run (3.4.3)."""
    lp = np.array([np.log(np.mean(s ** 2)) for s in series_by_window])
    run_of, t_of = np.asarray(run_of), np.asarray(t_of)
    resid = np.empty_like(lp)
    for r in np.unique(run_of):
        m = run_of == r
        if m.sum() >= 3:
            coef = np.polyfit(t_of[m], lp[m], 1)
            resid[m] = lp[m] - np.polyval(coef, t_of[m])
        else:
            resid[m] = lp[m] - lp[m].mean()
    return float(resid.std(ddof=1))


# ------------------------------------------------------------------ builder
class NoUsableWindows(ValueError):
    """Fewer than background.bad_window.min_windows windows survive (T0 mask + prereg-6 rule).  stage_cache records the subject as skipped with the
    counts below instead of crashing."""

    def __init__(self, subject: int, counts: dict):
        self.counts = counts
        super().__init__(f"subject {subject}: no usable background windows {counts}")


def build_subject_cache(source, subject: int, cfg, comps: dict[str, Component],
                        sacc_pool: dict[int, np.ndarray], cache_root: str | Path,
                        extras: tuple[str, ...] = (), group_comps: dict[str, Component] | None = None,
                        mismatch_pool: dict[int, np.ndarray] | None = None,
                        guard: SignalAccessGuard = NULL_GUARD, dtype=np.float32,
                        saccade_masks: dict[int, np.ndarray] | None = None) -> SubjectCache:
    """comps: the subject's components; must contain generator.components.required_components(cfg)
    ('alpha' plus 'mu' under hemisphere: single, or 'mu_L'/'mu_R' under hemisphere: both).
    sacc_pool: {pilot_subject: template}; the subject's own template is excluded (leave-one-out).
    extras: subset of {'alpha_shift', 'sacc_mismatch', 'group'}."""
    spec = stream_specs(cfg)["model"]
    ac, sc = cfg["alpha_component"], cfg["saccade"]
    ref = cfg["physionet"]["original_reference"]
    pool_ids = sorted(k for k in sacc_pool if k != subject)
    if not pool_ids:
        raise ValueError("empty saccade template pool")
    mis_ids = sorted(k for k in (mismatch_pool or {}) if k != subject)
    guard.record(subject, IMAGERY_RUNS, "background_windows")

    tb = task_bases(cfg)
    names = core_bases(cfg)
    if "alpha_shift" in extras:
        names += ["alpha_m2", "alpha_p2"]
    if "sacc_mismatch" in extras:
        names += ["sacc_R_mis"]
    if "group" in extras:
        names += ["alpha_group", *[f"{b}_group" for b in tb]]
    acc: dict[str, list[np.ndarray]] = {k: [] for k in names}
    meta_rows, alpha_series, run_of, t_of = [], [], [], []
    stats_all = {}
    bw = cfg["background"]["bad_window"]
    counts = {"t0_windows": 0, "rej_max_abs": 0, "rej_flat": 0, "kept": 0}

    # pre-pass: all runs' windows first, so a per-subject amplitude rule (amp_rule: robust_z)
    # can be computed over the subject's whole set of background windows
    prepared = {}
    for run_id in IMAGERY_RUNS:
        run = source.load(subject, run_id)
        mask = saccade_masks.get(run_id) if saccade_masks else None
        wins, st = t0_windows(run, cfg, mask)
        stats_all[run_id] = st
        counts["t0_windows"] += len(wins)
        if not wins:
            continue
        X0_run, _ = apply_stream(run.data, run.sfreq, spec)
        prepared[run_id] = (run, wins, X0_run)
    robust_reject = set()
    if bw.get("amp_rule", "absolute") == "robust_z":
        keys, stat = [], []
        for run_id, (run, wins, X0_run) in prepared.items():
            for w in wins:
                core = X0_run[:, out_index(w.start, run.sfreq, spec): out_index(w.start + w.length, run.sfreq, spec)]
                keys.append((run_id, w.start)); stat.append(float(np.log10(np.ptp(core, axis=-1).max())))
        z = robust_z(np.asarray(stat))
        robust_reject = {k for k, zz in zip(keys, z) if zz > bw["robust_z_max"]}
        counts["robust_threshold_uv_ptp"] = float(10 ** (np.median(stat) + bw["robust_z_max"] * 1.4826
                                                       * np.median(np.abs(np.asarray(stat) - np.median(stat))))) if stat else None

    for run_id, (run, wins, X0_run) in prepared.items():
        sf = run.sfreq
        amp_mask = amp_channels(run.ch_names, cfg)
        n_taper = int(round(ac["taper_s"] * sf))

        def comp_series(c: Component, band):
            return c.w @ fir(run.data, sf, *band, trans=ac["fir_transition_hz"])

        series = {k: comp_series(comps[k], comps[k].band) for k in ("alpha", *tb)}
        comp_of = {k: comps[k] for k in ("alpha", *tb)}
        if "alpha_shift" in extras:
            lo, hi = comps["alpha"].band
            for tag, sh in (("alpha_m2", cfg["design"]["mismatch_alpha_shift_hz"][0]),
                            ("alpha_p2", cfg["design"]["mismatch_alpha_shift_hz"][1])):
                series[tag] = comp_series(comps["alpha"], (lo + sh, hi + sh)); comp_of[tag] = comps["alpha"]
        if "group" in extras and group_comps:
            for key in ("alpha", *tb):
                tag = f"{key}_group"
                series[tag] = comp_series(group_comps[key], group_comps[key].band); comp_of[tag] = group_comps[key]

        deltas: dict[str, list[np.ndarray]] = {k: [] for k in names if k != "X0"}
        keep = []
        for w in wins:
            a, b = w.region
            core = X0_run[:, out_index(w.start, sf, spec): out_index(w.start + w.length, sf, spec)]
            if bw.get("amp_rule", "absolute") == "robust_z":
                why = "max_abs" if (run_id, w.start) in robust_reject else bad_window_reason(core, cfg, amp_mask, amp=False)
            else:
                why = bad_window_reason(core, cfg, amp_mask)
            if why:
                counts["rej_max_abs" if why == "max_abs" else "rej_flat"] += 1
                continue
            counts["kept"] += 1
            keep.append(w)
            env = cosine_envelope(w.length, n_taper)
            n_reg = b - a
            for tag in series:
                deltas[tag].append(component_delta(comp_of[tag].a, series[tag][w.start:w.start + w.length],
                                                   slice(None), n_reg, w.pad, env))
            r = keyed_rng(cfg["seeds"]["pseudo_label"], "sacc", w.wid)
            lo_s, hi_s = sc["onset_window_s"]
            onset = int(round(r.uniform(lo_s, min(hi_s, w.length / sf - 0.3)) * sf))
            tid = int(r.choice(pool_ids))
            meta_rows.append({"wid": w.wid, "run": run_id, "start": w.start, "sacc_onset": onset,
                              "sacc_template": tid})
            for tag, right in (("sacc_R", True), ("sacc_L", False)):
                deltas[tag].append(saccade_delta(sacc_pool[tid], run.ch_names, sf, n_reg, w.pad, env, onset,
                                                 cfg, keyed_rng(cfg["seeds"]["pseudo_label"], tag, w.wid),
                                                 rightward=right, reference=ref))
            if "sacc_mismatch" in extras:
                src = mismatch_pool if mis_ids else sacc_pool
                mid = int(r.choice(mis_ids or pool_ids))
                mm = sc["mismatch"]
                tmpl = perturb_template(src[mid], run.ch_names,
                                        r.uniform(-mm["dipole_rotation_deg"], mm["dipole_rotation_deg"]),
                                        float(np.exp(r.uniform(*np.log(mm["frontotemporal_ratio"])))))
                deltas["sacc_R_mis"].append(saccade_delta(tmpl, run.ch_names, sf, n_reg, w.pad, env, onset, cfg,
                                                          keyed_rng(cfg["seeds"]["pseudo_label"], "mis", w.wid),
                                                          rightward=True, reference=ref))
            alpha_series.append(series["alpha"][w.start:w.start + w.length])
            run_of.append(run_id); t_of.append(w.start / sf)
        if not keep:
            continue
        for w in keep:
            oa, ob = out_index(w.region[0], sf, spec), out_index(w.region[1], sf, spec)
            acc["X0"].append(X0_run[:, oa:ob])
        for tag, dl in deltas.items():
            acc[tag].extend(stream_window_deltas(dl, keep, run.data.shape[1], sf, spec))

    if len(meta_rows) < bw.get("min_windows", 1):
        raise NoUsableWindows(subject, counts)
    root = Path(cache_root).expanduser() / f"sub-{subject:03d}"
    root.mkdir(parents=True, exist_ok=True)
    for tag, arrs in acc.items():
        np.save(root / f"{tag}.npy", np.stack(arrs).astype(dtype))
    n = len(meta_rows)
    meta = {
        "subject": subject,
        "wids": [m["wid"] for m in meta_rows],
        "rows": meta_rows,
        "y": pseudo_labels(subject, n, cfg["seeds"]["pseudo_label"]).tolist(),
        "sigma_alpha": sigma_log_power(alpha_series, run_of, t_of) if n >= 3 else float("nan"),
        "sfreq": spec.sfreq,
        "pad": int(round(cfg["streams"]["pad_s"] * spec.sfreq)),
        "core_len": int(round(cfg["background"]["window_s"] * spec.sfreq)),
        "bases": names,
        "window_stats": stats_all,
        "window_counts": counts,
        "components": {k: v.to_dict() for k, v in comps.items()},
        "config_hash": cfg.hash(),
    }
    with open(root / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)
    return SubjectCache(subject, root, meta)
