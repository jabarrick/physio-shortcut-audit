"""Round-4 post hoc checks on the undetected deflections (exploratory; no training).
Run from repo root: python scripts/posthoc_revision4_undetected.py   (numpy/pandas only)

(a) Selection control: terciles of a deflection whose sign is randomised (instead of signed by the cue)
    show how extreme tercile means become from selection on a noisy measure alone.
(b) Approximate re-detection on the cached model stream (0.5-75 Hz, 200 Hz): the registered detector's
    step statistic d(t) = mean(next 100 ms) - mean(previous 100 ms) of AF7 - AF8, threshold 5 x MAD of d
    over all of a subject's epochs (min 15 uV).  For trials classed 'none' by the registered detector
    (eye stream, 0.1 Hz high-pass, per-run threshold), how often does a supra-threshold step occur
    0-1 s or 1-2 s after the cue, and at what latency?  This approximates, not reproduces, the detector.
Writes results/analysis/posthoc_revision4_undetected.json
"""
import glob, json, sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
CH = None
try:
    from p3audit.constants import PHYSIONET_CHANNELS as CH
except Exception:
    pass
i7, i8 = CH.index("AF7"), CH.index("AF8")
SF, W = 200, 20            # 100 ms half-window at 200 Hz
rng = np.random.default_rng(13)


def step_stat(h):
    c = np.concatenate([np.zeros((h.shape[0], 1)), np.cumsum(h, 1)], 1)
    n = h.shape[1]; d = np.zeros_like(h); t = np.arange(W, n - W)
    d[:, t] = (c[:, t + W] - c[:, t]) / W - (c[:, t] - c[:, t - W]) / W
    return d


rows = []
for f in sorted(glob.glob("results/real_cache/sub-*.npz")):
    s = int(f.split("sub-")[1][:3]); z = np.load(f, allow_pickle=True)
    X, y, ty, pad = z["X"], z["y"], z["type"], int(z["pad"])
    h = X[:, i7, :] - X[:, i8, :]
    sg = np.where(y == 1, 1.0, -1.0)
    dfl = (h[:, pad + 60:pad + 240].mean(1) - h[:, pad - 40:pad].mean(1)) * sg
    d = step_stat(h)
    core = d[:, pad:pad + 800]
    mad = np.median(np.abs(core - np.median(core))) * 1.4826
    thr = max(5 * mad, 15.0)
    for k in range(len(y)):
        w1 = d[k, pad:pad + 200]; w2 = d[k, pad + 200:pad + 400]
        i1, i2 = int(np.argmax(np.abs(w1))), int(np.argmax(np.abs(w2)))
        rows.append({"subject": s, "type": str(ty[k]), "d": float(dfl[k]), "sign": float(sg[k]), "thr": thr,
                     "max1": float(abs(w1[i1])), "lat1": i1 / SF, "dir1": float(np.sign(w1[i1]) * sg[k]),
                     "max2": float(abs(w2[i2])), "lat2": 1 + i2 / SF, "dir2": float(np.sign(w2[i2]) * sg[k])})
D = pd.DataFrame(rows)
flip = 1.0 if D.loc[D.type == "congruent", "d"].mean() > 0 else -1.0
D["d"] *= flip; D["dir1"] *= flip; D["dir2"] *= flip


def terc(df, col):
    z = df.groupby("subject")[col].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
    return pd.qcut(z, 3, labels=False)


N = D[D.type == "none"].copy()
N["tert"] = terc(N, "d")
out = {"n_none": int(len(N))}
# (a) selection control: random signs, 200 repetitions
top, bot = [], []
for _ in range(200):
    N["dr"] = N["d"] * rng.choice([-1.0, 1.0], len(N))
    t = terc(N, "dr")
    top.append(N.loc[t == 2, "dr"].mean()); bot.append(N.loc[t == 0, "dr"].mean())
out["tercile_means_cue_signed"] = {int(k): float(v) for k, v in N.groupby("tert").d.mean().items()}
out["tercile_means_random_sign"] = {"lowest": float(np.mean(bot)), "highest": float(np.mean(top)),
                                    "highest_2.5_97.5": [float(np.quantile(top, .025)), float(np.quantile(top, .975))]}
out["sd_within_subject_none"] = float(N.groupby("subject").d.std().median())
# (b) approximate re-detection
for lab, g in [("congruent", D[D.type == "congruent"]), ("incongruent", D[D.type == "incongruent"])] + \
              [(f"none_t{k}", N[N.tert == k]) for k in (0, 1, 2)]:
    s1 = g.max1 > g.thr; s2 = g.max2 > g.thr
    out.setdefault("redetect", {})[lab] = {
        "n": int(len(g)),
        "share_step_0_1s": float(s1.mean()), "share_step_1_2s_only": float((~s1 & s2).mean()),
        "share_step_0_1s_towards_cue": float((s1 & (g.dir1 > 0)).mean()),
        "median_latency_0_1s": float(g.loc[s1, "lat1"].median()) if s1.any() else None,
        "median_step_over_thr_0_1s": float((g.max1 / g.thr).median()),
        "share_from_high_threshold_subjects": float((g.thr > D.groupby("subject").thr.first().median()).mean())}
out["threshold_uv_median_range"] = [float(D.groupby("subject").thr.first().median()), float(D.groupby("subject").thr.first().min()), float(D.groupby("subject").thr.first().max())]
json.dump(out, open("results/analysis/posthoc_revision4_undetected.json", "w"), indent=1)
print(json.dumps(out, indent=1))
