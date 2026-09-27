"""Round-3 post hoc real-data checks that need no training (exploratory).
Run from repo root: python scripts/posthoc_revision3_deflection.py

(a) Time course of the cue-signed AF7 - AF8 deflection by trial type and, for trials without a detected
    saccade, by tercile of the 0.3-1.2 s deflection (same definition as posthoc_revision2_real.py).
(b) The same signed difference for other left-right channel pairs (Fp1-Fp2 ... O1-O2): an ocular source
    falls off steeply from the front of the head, a lateralised neural source does not.
(c) Coupling of the real-data alpha proxy z (posterior alpha log power, median split) with the label y.
Reads results/real_cache only. Writes results/analysis/posthoc_revision3_deflection.{json,npz}.
Note: the model input is band-passed from 0.5 Hz, so a sustained ocular step decays over about a second.
"""
import glob, json, sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from p3audit.constants import PHYSIONET_CHANNELS as CH

SF = 200
PAIRS = [("Fp1", "Fp2"), ("AF7", "AF8"), ("F7", "F8"), ("F3", "F4"), ("FC3", "FC4"), ("C3", "C4"),
         ("T7", "T8"), ("CP3", "CP4"), ("P7", "P8"), ("O1", "O2")]
PAIRS = [(a, b) for a, b in PAIRS if a in CH and b in CH]
rng = np.random.default_rng(11)

rows, waves, alpha = [], {}, []
for f in sorted(glob.glob("results/real_cache/sub-*.npz")):
    s = int(f.split("sub-")[1][:3])
    z = np.load(f, allow_pickle=True)
    X, y, ty, pad = z["X"], z["y"], z["type"], int(z["pad"])
    sg = np.where(y == 1, 1.0, -1.0)[:, None]
    base = slice(pad - 40, pad)
    for a, b in PAIRS:
        h = X[:, CH.index(a), :] - X[:, CH.index(b), :]
        h = (h - h[:, base].mean(1, keepdims=True)) * sg
        waves[(s, a + "-" + b)] = h[:, pad - 200: pad + 600].astype(np.float32)   # -1 s .. +3 s
    for k in range(len(y)):
        rows.append({"subject": s, "trial": k, "type": str(ty[k]), "y": int(y[k]), "alpha": float(z["alpha_logpow"][k])})
d = pd.DataFrame(rows)
win = slice(200 + 60, 200 + 240)   # 0.3-1.2 s in the stored -1..+3 s segment
d["d"] = np.concatenate([waves[(s, "AF7-AF8")][:, win].mean(1) for s in sorted(d.subject.unique())])
flip = 1.0 if d.loc[d.type == "congruent", "d"].mean() > 0 else -1.0
d["d"] *= flip
d["dz"] = d.groupby("subject").d.transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
none = d.type == "none"
d.loc[none, "tert"] = pd.qcut(d.loc[none, "dz"], 3, labels=False)
subs = sorted(d.subject.unique())


def group_mean(pair, mask):
    """Subject-averaged mean waveform (average within subject first)."""
    per = []
    for s in subs:
        m = mask[d.subject == s].values
        if m.sum():
            per.append(flip * waves[(s, pair)][m].mean(0))
    return np.array(per)


groups = {"congruent": d.type == "congruent", "incongruent": d.type == "incongruent", "none": none}
for k in (0, 1, 2):
    groups[f"none_t{k}"] = none & (d.tert == k)
t = (np.arange(800) - 200) / SF
out = {"time_s": t.tolist(), "pairs": [a + "-" + b for a, b in PAIRS], "n_trials": {g: int(m.sum()) for g, m in groups.items()}}
curves = {}
for pair in out["pairs"]:
    for g, m in groups.items():
        per = group_mean(pair, m)
        curves[f"{pair}|{g}"] = per.mean(0)
        amp = per[:, win].mean(1)                     # per-subject 0.3-1.2 s amplitude
        late = per[:, 200 + 240: 200 + 400].mean(1)   # 1.2-2.0 s
        B = [amp[rng.integers(0, len(amp), len(amp))].mean() for _ in range(2000)]
        out.setdefault("amp_0.3_1.2", {}).setdefault(pair, {})[g] = {"mean_uV": float(amp.mean()), "ci": [float(np.quantile(B, .025)), float(np.quantile(B, .975))], "n_subjects": int(len(amp))}
        out.setdefault("amp_1.2_2.0", {}).setdefault(pair, {})[g] = float(late.mean())
# anterior-posterior gradient of the top-minus-bottom tercile difference and congruent-minus-none difference
for lab, (g1, g0) in {"t2_minus_t0": ("none_t2", "none_t0"), "congruent_minus_none": ("congruent", "none")}.items():
    out.setdefault("contrast", {})[lab] = {p: out["amp_0.3_1.2"][p][g1]["mean_uV"] - out["amp_0.3_1.2"][p][g0]["mean_uV"] for p in out["pairs"]}
# peak latency of the tercile-2 curve on AF7-AF8
c2 = curves["AF7-AF8|none_t2"]; out["t2_peak_latency_s"] = float(t[200 + np.argmax(c2[200:600])])
c = curves["AF7-AF8|congruent"]; out["congruent_peak_latency_s"] = float(t[200 + np.argmax(c[200:600])])

# (c) alpha proxy vs label
r_sub, phi_sub = [], []
for s, g in d.groupby("subject"):
    if g.y.nunique() == 2 and len(g) > 5:
        r_sub.append(np.corrcoef(g.alpha, g.y)[0, 1])
        zz = (g.alpha > g.alpha.median()).astype(int)
        phi_sub.append(np.corrcoef(zz, g.y)[0, 1] if zz.nunique() == 2 else np.nan)
r_sub, phi_sub = np.array(r_sub), np.array(phi_sub)
zg = (d.alpha > d.alpha.median()).astype(int)
Br = [np.nanmean(r_sub[rng.integers(0, len(r_sub), len(r_sub))]) for _ in range(2000)]
out["alpha_z_y"] = {"pooled_phi_global_median": float(np.corrcoef(zg, d.y)[0, 1]),
                    "mean_within_subject_r": float(np.nanmean(r_sub)), "ci_r": [float(np.quantile(Br, .025)), float(np.quantile(Br, .975))],
                    "mean_within_subject_phi": float(np.nanmean(phi_sub)), "mean_abs_within_subject_r": float(np.nanmean(np.abs(r_sub))),
                    "n_subjects": int(len(r_sub)), "n_trials": int(len(d))}
json.dump(out, open("results/analysis/posthoc_revision3_deflection.json", "w"), indent=1)
np.savez_compressed("results/analysis/posthoc_revision3_deflection_curves.npz", t=t, **{k.replace("|", "__"): v for k, v in curves.items()})
print(json.dumps({k: v for k, v in out.items() if k not in ("time_s", "amp_1.2_2.0")}, indent=1)[:6000])
print("late (1.2-2.0 s):", {p: {g: round(v, 1) for g, v in gg.items()} for p, gg in out["amp_1.2_2.0"].items() if p in ("AF7-AF8", "C3-C4", "P7-P8")})
print("peaks:", out["t2_peak_latency_s"], out["congruent_peak_latency_s"])
