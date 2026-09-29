"""Amendment 2 (OSF): registered real-data tests.  Run from the repository root after
scripts/amendment2_controls.py.  Deflection and terciles exactly as in scripts/posthoc_revision3_terciles.py
(cue-signed AF7 - AF8 at 0.3-1.2 s minus -0.2-0 s, standardised within model x subject; terciles over trials
without a detected saccade).  Trial correctness is averaged over the three seeds; models are pooled (each
model's trials enter once); subject cluster bootstrap, 2000 resamples, seed 20261002.
Three registered tests, Bonferroni-adjusted: two-sided 98.33% intervals.

G(main, arm) = [main - arm gap in the highest tercile] - [the same in the lowest tercile].
A2-R1  placebo, shuffled regressor : G(main, heog_ols) - G(main, plc_shuf) > 0
A2-R2  frontal, local reference    : accuracy(frontal_loc) > 0.5 AND acc_t2 - acc_t0 (frontal_loc) > 0
A2-R3  time window                 : G(main, heog_ols) - G(main_late, heog_late) > 0
Secondary (reported, no decision): G(main, heog_ols) - G(main, plc_CP3CP4); accuracy of main_late and heog_late;
frontal_loc versus the registered frontal arm of Amendment 1.
Writes results/analysis/amendment2_real.json.
"""
import glob, json, sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from p3audit.constants import PHYSIONET_CHANNELS as CH

N_BOOT, SEED, ALPHA = 2000, 20261002, 0.05 / 3
i7, i8 = CH.index("AF7"), CH.index("AF8")
key = ["model", "subject", "trial", "type", "label"]


def load(pattern, arm=None):
    df = pd.concat([pd.read_csv(f) for f in glob.glob(pattern)])
    return df if arm is None else df[df.arm == arm]


main = load("results/real/*_trials.csv", "main")
r3 = load("results/real_r3/*_trials.csv")
a2 = load("results/real_a2/*_trials.csv")

defl = {}
for s in sorted(main.subject.unique()):
    z = np.load(f"results/real_cache/sub-{s:03d}.npz", allow_pickle=True)
    h = z["X"][:, i7, :] - z["X"][:, i8, :]; p = int(z["pad"])
    v = (h[:, p + 60:p + 240].mean(1) - h[:, p - 40:p].mean(1)) * np.where(z["y"] == 1, 1.0, -1.0)
    for k in range(len(v)):
        defl[(s, k)] = v[k]


def prep(df, name):
    t = df.groupby(key, as_index=False).correct.mean().rename(columns={"correct": name})
    return t


wide = prep(main, "main")
arms = {"heog_ols": r3[r3.arm == "heog_ols"], "frontal": r3[r3.arm == "frontal"]}
for a in ("frontal_loc", "plc_shuf", "plc_CP3CP4", "main_late", "heog_late"):
    arms[a] = a2[a2.arm == a]
for a, df in arms.items():
    assert len(df), f"missing arm {a}"
    wide = wide.merge(prep(df, a)[key + [a]], on=key)
wide["d"] = [defl[(s, k)] for s, k in zip(wide.subject, wide.trial)]
flip = 1.0 if wide.loc[wide.type == "congruent", "d"].mean() > 0 else -1.0
wide["d"] *= flip
wide["dz"] = wide.groupby(["model", "subject"]).d.transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
n = wide.type == "none"
wide.loc[n, "tert"] = wide[n].groupby("model").dz.transform(lambda x: pd.qcut(x, 3, labels=False))


def stats(w):
    nn = w[w.type == "none"]; t0, t2 = nn[nn.tert == 0], nn[nn.tert == 2]
    G = lambda a, b: (t2[a] - t2[b]).mean() - (t0[a] - t0[b]).mean()
    o = {"G_heog_ols": G("main", "heog_ols"), "G_plc_shuf": G("main", "plc_shuf"),
         "G_plc_CP3CP4": G("main", "plc_CP3CP4"), "G_late": G("main_late", "heog_late")}
    o["A2_R1"] = o["G_heog_ols"] - o["G_plc_shuf"]
    o["A2_R3"] = o["G_heog_ols"] - o["G_late"]
    o["sec_CP3CP4"] = o["G_heog_ols"] - o["G_plc_CP3CP4"]
    o["A2_R2_acc"] = w.frontal_loc.mean()
    o["A2_R2_gradient"] = t2.frontal_loc.mean() - t0.frontal_loc.mean()
    o["frontal_car_acc"] = w.frontal.mean(); o["frontal_car_gradient"] = t2.frontal.mean() - t0.frontal.mean()
    o["acc_main"] = w.main.mean(); o["acc_main_late"] = w.main_late.mean(); o["acc_heog_late"] = w.heog_late.mean()
    o["gap_none_late"] = (nn.main_late - nn.heog_late).mean(); o["gap_none_full_ols"] = (nn.main - nn.heog_ols).mean()
    return o


rng = np.random.default_rng(SEED)
subs = np.array(sorted(wide.subject.unique())); g = {s: x for s, x in wide.groupby("subject")}
est = stats(wide); B = []
for _ in range(N_BOOT):
    B.append(stats(pd.concat([g[s] for s in rng.choice(subs, subs.size)])))
B = pd.DataFrame(B)
res = {k: {"est": float(v), "ci_95": [float(B[k].quantile(.025)), float(B[k].quantile(.975))],
           "ci_98.33": [float(B[k].quantile(ALPHA / 2)), float(B[k].quantile(1 - ALPHA / 2))]} for k, v in est.items()}
res["decisions"] = {"A2-R1": res["A2_R1"]["ci_98.33"][0] > 0,
                    "A2-R2": res["A2_R2_acc"]["ci_98.33"][0] > 0.5 and res["A2_R2_gradient"]["ci_98.33"][0] > 0,
                    "A2-R3": res["A2_R3"]["ci_98.33"][0] > 0}
res["n_trials"] = int(len(wide)); res["n_subjects"] = int(subs.size)
json.dump(res, open("results/analysis/amendment2_real.json", "w"), indent=1, default=bool)
print(json.dumps(res["decisions"]), {k: round(v["est"], 3) for k, v in res.items() if isinstance(v, dict) and "est" in v})
