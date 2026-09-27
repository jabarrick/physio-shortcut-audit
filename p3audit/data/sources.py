"""Run sources: PhysioNet EEGMMIDB (via MNE) and a synthetic stand-in for tests.

All arrays returned by this package are in **microvolts**, channels in
`PHYSIONET_CHANNELS` order, at the file's native sampling rate (160 Hz), in
the file's original reference — i.e. the "raw domain" in which every
injection happens (3.4.2: before any filtering or re-referencing).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..constants import PHYSIONET_CHANNELS, RUN_KIND


@dataclass
class RunHeader:
    """Metadata only — reading it never touches the signal (3.1)."""
    subject: int
    run: int
    sfreq: float
    n_times: int
    events: list[tuple[float, float, str]]  # (onset_s, duration_s, description)

    @property
    def duration(self) -> float:
        return self.n_times / self.sfreq

    def n_trials(self) -> int:
        return sum(1 for _, _, d in self.events if d in ("T1", "T2"))


@dataclass
class RunData:
    subject: int
    run: int
    sfreq: float
    data: np.ndarray              # (n_channels, n_times) µV, original reference
    ch_names: list[str]
    events: list[tuple[float, float, str]]
    kind: str = field(default="")

    def __post_init__(self):
        self.kind = RUN_KIND.get(self.run, "unknown")

    def event_samples(self, desc: str | tuple[str, ...]):
        desc = (desc,) if isinstance(desc, str) else desc
        return [(int(round(o * self.sfreq)), int(round(d * self.sfreq)), s)
                for o, d, s in self.events if s in desc]


class RunSource:
    def header(self, subject: int, run: int) -> RunHeader | None:
        raise NotImplementedError

    def load(self, subject: int, run: int) -> RunData:
        raise NotImplementedError

    def subjects(self) -> list[int]:
        raise NotImplementedError


# ============================================================ PhysioNet (MNE)
class PhysioNetSource(RunSource):
    """EEGMMIDB via mne.datasets.eegbci.load_data.

    physionet.org may be unreachable from cloud machines (outline §7 note):
    run locally or pre-download into `data_root`.
    """

    def __init__(self, data_root: str | Path, n_subjects: int = 109, download: bool = True):
        self.root = Path(data_root).expanduser()
        self.n_subjects = n_subjects
        self.download = download

    def subjects(self) -> list[int]:
        return list(range(1, self.n_subjects + 1))

    def _path(self, subject: int, run: int) -> Path | None:
        from mne.datasets import eegbci
        try:
            paths = eegbci.load_data(subject, [run], path=str(self.root), update_path=False,
                                     verbose="ERROR", download=self.download)
        except TypeError:  # older MNE without `download`
            paths = eegbci.load_data(subject, [run], path=str(self.root), update_path=False, verbose="ERROR")
        except Exception:
            return None
        return Path(paths[0]) if paths else None

    @lru_cache(maxsize=4096)
    def header(self, subject: int, run: int) -> RunHeader | None:
        import mne
        p = self._path(subject, run)
        if p is None or not p.exists():
            return None
        raw = mne.io.read_raw_edf(p, preload=False, verbose="ERROR")  # header + annotations only
        ev = [(float(a["onset"]), float(a["duration"]), str(a["description"])) for a in raw.annotations]
        return RunHeader(subject, run, float(raw.info["sfreq"]), int(raw.n_times), ev)

    def load(self, subject: int, run: int) -> RunData:
        import mne
        from mne.datasets import eegbci
        p = self._path(subject, run)
        raw = mne.io.read_raw_edf(p, preload=True, verbose="ERROR")
        eegbci.standardize(raw)
        raw.pick(PHYSIONET_CHANNELS)
        ev = [(float(a["onset"]), float(a["duration"]), str(a["description"])) for a in raw.annotations]
        return RunData(subject, run, float(raw.info["sfreq"]), raw.get_data() * 1e6,
                       list(raw.ch_names), ev)


# ============================================================ synthetic source
def channel_positions(ch_names=PHYSIONET_CHANNELS) -> np.ndarray:
    """Unit-sphere 3-D positions from the standard_1005 montage."""
    import mne
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mont = mne.channels.make_standard_montage("standard_1005")
    pos = mont.get_positions()["ch_pos"]
    lookup = {k.lower(): v for k, v in pos.items()}
    xyz = np.array([lookup[c.lower()] for c in ch_names])
    return xyz / np.linalg.norm(xyz, axis=1, keepdims=True)


def _gauss_topo(xyz, centre_ch, width=0.35, ch_names=PHYSIONET_CHANNELS):
    c = xyz[ch_names.index(centre_ch)]
    d = np.linalg.norm(xyz - c, axis=1)
    return np.exp(-(d / width) ** 2)


def _pink(rng, n_ch, n_t):
    f = np.fft.rfftfreq(n_t)
    spec = (rng.standard_normal((n_ch, f.size)) + 1j * rng.standard_normal((n_ch, f.size)))
    spec /= np.maximum(f, 1.0 / n_t) ** 0.5
    x = np.fft.irfft(spec, n=n_t, axis=1)
    return x / x.std(axis=1, keepdims=True)


def _narrowband(rng, n_t, sfreq, f0, bw=1.5):
    f = np.fft.rfftfreq(n_t, 1 / sfreq)
    spec = (rng.standard_normal(f.size) + 1j * rng.standard_normal(f.size)) * np.exp(-0.5 * ((f - f0) / bw) ** 2)
    x = np.fft.irfft(spec, n=n_t)
    return x / (x.std() + 1e-12)


class SyntheticSource(RunSource):
    """Physiologically-shaped fake EEGMMIDB for tests and dry runs.

    Contains: pink background with spatial mixing; a posterior alpha source
    (×3 in eyes-closed run 2); left/right mu sources with contralateral ERD
    during T1/T2 (T1 = left fist -> right-hemisphere ERD); cue-locked
    horizontal saccade steps toward the target in lateral runs and return
    saccades at T0 onset.  `bad_subjects` get a 128 Hz header (mimics subject 88).
    """

    def __init__(self, n_subjects: int = 20, n_trials: int = 15, seed: int = 0,
                 bad_subjects=(88,), saccade_prob: float = 0.7, sfreq: float = 160.0):
        self.n_subjects = n_subjects
        self.n_trials = n_trials
        self.seed = seed
        self.bad = set(bad_subjects)
        self.saccade_prob = saccade_prob
        self.sfreq = sfreq
        self._xyz = channel_positions()

    def subjects(self):
        return list(range(1, self.n_subjects + 1))

    def _events(self, run):
        if run in (1, 2):
            return [(0.0, 61.0, "T0")], 61.0
        ev, t = [], 0.0
        for i in range(self.n_trials):
            ev.append((t, 4.2, "T0")); t += 4.2
            ev.append((t, 4.1, "T1" if i % 2 == 0 else "T2")); t += 4.1
        ev.append((t, 4.2, "T0")); t += 4.2
        return ev, t

    def header(self, subject, run):
        ev, dur = self._events(run)
        sf = 128.0 if subject in self.bad else self.sfreq
        return RunHeader(subject, run, sf, int(round(dur * sf)), ev)

    def load(self, subject, run):
        rng = np.random.default_rng([self.seed, subject, run])
        srng = np.random.default_rng([self.seed, subject])          # subject-constant topographies
        sf = self.header(subject, run).sfreq
        ev, dur = self._events(run)
        n_t = int(round(dur * sf))
        n_ch = len(PHYSIONET_CHANNELS)
        xyz = self._xyz
        mix = srng.standard_normal((n_ch, 16)) * 0.6
        x = mix @ _pink(rng, 16, n_t) * 8.0 + _pink(rng, n_ch, n_t) * 4.0

        # alpha source (posterior), subject-specific peak shift
        a_topo = _gauss_topo(xyz, "Oz", 0.45) + 0.3 * srng.standard_normal(n_ch) * 0.1
        alpha = _narrowband(rng, n_t, sf, 10.0 + srng.uniform(-0.8, 0.8)) * (1 + 0.3 * _pink(rng, 1, n_t)[0].clip(-1, 1))
        alpha_gain = 3.0 if run == 2 else 1.0
        x += np.outer(a_topo, alpha) * 6.0 * alpha_gain

        # mu sources
        t = np.arange(n_t) / sf
        env_l = np.ones(n_t); env_r = np.ones(n_t)
        if run not in (1, 2):
            for o, d, s in ev:
                sl = slice(int(o * sf), int((o + d) * sf))
                if s == "T1":           # left fist (or both fists) -> right hemisphere ERD
                    env_r[sl] = 0.4
                    if RUN_KIND[run].endswith("ff"):
                        env_l[sl] = 0.4
                elif s == "T2":
                    env_l[sl] = 0.4
                    if RUN_KIND[run].endswith("ff"):
                        env_r[sl] = 0.4
        for centre, env in (("C3", env_l), ("C4", env_r)):
            topo = _gauss_topo(xyz, centre, 0.3)
            x += np.outer(topo, _narrowband(rng, n_t, sf, 11.5) * env) * 5.0

        # saccades: rightward -> AF7-AF8 negative (cornea-positive dipole)
        eye_topo = np.zeros(n_ch)
        for ch, w in (("AF7", 1.0), ("F7", 0.8), ("FT7", 0.4), ("Fp1", 0.5),
                      ("AF8", -1.0), ("F8", -0.8), ("FT8", -0.4), ("Fp2", -0.5)):
            eye_topo[PHYSIONET_CHANNELS.index(ch)] = w
        gaze = np.zeros(n_t)
        if RUN_KIND.get(run, "").endswith("lr"):
            cur = 0.0
            for o, d, s in ev:
                i0 = int(o * sf)
                if s in ("T1", "T2") and rng.random() < self.saccade_prob:
                    lat = int((0.2 + 0.1 * rng.random()) * sf)
                    cur = -1.0 if s == "T1" else 1.0      # T1 left target
                    gaze[i0 + lat:] = cur
                elif s == "T0" and cur != 0.0:
                    lat = int((0.15 + 0.1 * rng.random()) * sf)
                    gaze[i0 + lat:] = 0.0; cur = 0.0
        if gaze.any():
            k = int(0.04 * sf)
            gaze = np.convolve(gaze, np.ones(k) / k, mode="same")
            x += np.outer(eye_topo, -gaze) * 40.0   # rightward (gaze=+1) -> AF7-AF8 = -80 µV
        return RunData(subject, run, sf, x.astype(np.float64), list(PHYSIONET_CHANNELS), ev)
