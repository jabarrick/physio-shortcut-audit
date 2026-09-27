"""SHU-MI (supplementary real audit) and EEGEyeNet (P10a amplitude reference).

Both loaders are written against the public descriptions and must be checked
in P11 / P10a ([待核实]): file naming, array keys, channel lists, units and
whether continuous recordings exist.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class EpochSet:
    subject: int
    session: int
    X: np.ndarray          # (n_trials, n_ch, n_times) µV
    y: np.ndarray          # 0 = left, 1 = right
    sfreq: float
    ch_names: list[str]


SHU_CHANNELS = [  # 32-channel 10-20 layout as documented; verify in P11 [待核实]
    "Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8", "FC1", "FC2", "FC5", "FC6", "Cz", "C3", "C4", "T3", "T4",
    "A1", "A2", "CP1", "CP2", "CP5", "CP6", "Pz", "P3", "P4", "T5", "T6", "PO3", "PO4", "Oz", "O1", "O2",
]


def load_shu(root: str | Path, subject: int, session: int, sfreq: float = 250.0,
             ch_names: list[str] | None = None) -> EpochSet:
    from scipy.io import loadmat
    root = Path(root).expanduser()
    # figshare 19228725 names files "Sub-001_ses-01_..." (capital S): match case-insensitively (PILOT_LOG 15.19)
    pat = re.compile(rf"sub-0*{subject}_ses-0*{session}.*\.mat$", re.IGNORECASE)
    files = [p for p in root.rglob("*.mat") if pat.search(p.name)]
    if not files:
        raise FileNotFoundError(f"SHU-MI subject {subject} session {session} not found under {root}")
    m = loadmat(files[0])
    xkey = next(k for k in ("data", "X", "x", "eeg") if k in m)
    ykey = next(k for k in ("labels", "label", "y", "Y") if k in m)
    X = np.asarray(m[xkey], dtype=np.float64)
    y = np.asarray(m[ykey]).ravel().astype(int)
    y = y - y.min()                      # {1,2} -> {0,1}
    if X.shape[1] > X.shape[2]:          # (trials, times, ch) -> (trials, ch, times)
        X = X.transpose(0, 2, 1)
    return EpochSet(subject, session, X, y, sfreq, ch_names or SHU_CHANNELS[: X.shape[1]])


def eegeyenet_saccade_amplitudes(npz_path: str | Path, px_per_deg: float | None = None) -> np.ndarray:
    """P10a: saccade amplitudes from the EEGEyeNet eye-tracker targets.

    Expects the benchmark npz with an 'EEG' array and 'labels' whose columns
    include start/end gaze coordinates (layout differs by task — verify).
    Returns amplitudes in the unit of the labels (px or deg).  Conversion to
    AF7−AF8 µV requires the per-degree calibration measured on pilot data.
    """
    # PILOT_LOG 15.19: Direction and Position files both have 3 label columns ([id, amplitude, angle]
    # vs [id, x, y]), so the old shape test could not tell them apart and rejected the Direction file.
    # Decide by file name, then sanity-check the angle column (radians, within [-pi, pi]).
    name = Path(npz_path).name
    if not name.startswith("Direction_task"):
        raise ValueError(f"P10a needs the Direction_task file (labels [id, amplitude, angle]); got {name}")
    d = np.load(npz_path, allow_pickle=True)
    lab = d["labels"]
    if lab.ndim != 2 or lab.shape[1] < 3:
        raise ValueError(f"unrecognised label layout {lab.shape}")
    ang = lab[:, 2].astype(float)
    if np.nanmax(np.abs(ang)) > np.pi + 1e-6:
        raise ValueError("column 2 is not an angle in radians - check the label layout before using column 1")
    amp = lab[:, 1].astype(float)
    if px_per_deg:
        amp = amp / px_per_deg
    return amp
