"""Fixed facts about the datasets (not tunable)."""

# PhysioNet EEGMMIDB channel order after mne.datasets.eegbci.standardize()
PHYSIONET_CHANNELS = [
    "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
    "Fp1", "Fpz", "Fp2",
    "AF7", "AF3", "AFz", "AF4", "AF8",
    "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    "FT7", "FT8", "T7", "T8", "T9", "T10", "TP7", "TP8",
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8",
    "O1", "Oz", "O2", "Iz",
]
assert len(PHYSIONET_CHANNELS) == 64

# run -> semantics.  T0 rest; in L/R runs T1 = left fist, T2 = right fist;
# in fists/feet runs T1 = both fists, T2 = both feet.
RUN_KIND = {
    1: "baseline_open", 2: "baseline_closed",
    3: "exec_lr", 7: "exec_lr", 11: "exec_lr",
    4: "imag_lr", 8: "imag_lr", 12: "imag_lr",
    5: "exec_ff", 9: "exec_ff", 13: "exec_ff",
    6: "imag_ff", 10: "imag_ff", 14: "imag_ff",
}
IMAGERY_RUNS = (4, 8, 12, 6, 10, 14)
LATERAL_RUNS = (3, 7, 11, 4, 8, 12)

# Pseudo-label convention (3.4.2): y = 0 -> left-hemisphere mu component,
# y = 1 -> right-hemisphere component.
Y_LEFT_HEMI, Y_RIGHT_HEMI = 0, 1
