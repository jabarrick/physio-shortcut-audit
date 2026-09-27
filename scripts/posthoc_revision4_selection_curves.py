import glob, json, sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
from p3audit.constants import PHYSIONET_CHANNELS as CH
i7, i8 = CH.index("AF7"), CH.index("AF8"); rng = np.random.default_rng(21)
H, rows = [], []
for f in sorted(glob.glob("results/real_cache/sub-*.npz")):
    s = int(f.split("sub-")[1][:3]); z = np.load(f, allow_pickle=True); pad = int(z["pad"])
    h = z["X"][:, i7, pad-200:pad+600] - z["X"][:, i8, pad-200:pad+600]
    h = h - h[:, 160:200].mean(1, keepdims=True)
    sg = np.where(z["y"] == 1, 1.0, -1.0)
    H.append(h * sg[:, None])
    for k in range(len(sg)): rows.append((s, str(z["type"][k])))
H = np.concatenate(H); D = pd.DataFrame(rows, columns=["subject", "type"])
flip = 1.0 if H[(D.type == "congruent").values, 260:440].mean() > 0 else -1.0; H *= flip
N = (D.type == "none").values
def terc(v, subj):
    z = pd.Series(v).groupby(subj).transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
    return pd.qcut(z, 3, labels=False).values
d = H[:, 260:440].mean(1)
Hn, dn, sn = H[N], d[N], D.subject.values[N]
tt = terc(dn, sn)
cue_top = Hn[tt == 2].mean(0); cue_bot = Hn[tt == 0].mean(0)
rt, rb = [], []
for _ in range(100):
    r = rng.choice([-1.0, 1.0], len(dn)); t2 = terc(dn * r, sn)
    rt.append((Hn * r[:, None])[t2 == 2].mean(0)); rb.append((Hn * r[:, None])[t2 == 0].mean(0))
rt, rb = np.mean(rt, 0), np.mean(rb, 0)
def win(c, a, b): return float(c[200 + int(a * 200):200 + int(b * 200)].mean())
out = {}
for lab, c in {"cue_top": cue_top, "rand_top": rt, "cue_top_minus_rand_top": cue_top - rt, "cue_bot": cue_bot, "rand_bot": rb,
               "cue_bot_minus_rand_bot": cue_bot - rb, "all_none_mean": Hn.mean(0), "congruent": H[(D.type == "congruent").values].mean(0)}.items():
    e, l = win(c, .3, 1.2), win(c, 1.2, 2.0)
    out[lab] = {"early": round(e, 1), "late": round(l, 1), "late_over_early": round(l / e, 2) if abs(e) > 1e-6 else None,
                "pre_cue_-0.8_-0.3": round(win(c, -.8, -.3), 1), "rise_0.2_0.3": round(win(c, .2, .3), 1)}
print(json.dumps(out, indent=1))
np.savez_compressed("results/analysis/posthoc_revision4_selection_curves.npz", t=(np.arange(800) - 200) / 200,
                    cue_top=cue_top, rand_top=rt, cue_bot=cue_bot, rand_bot=rb, all_none=Hn.mean(0), congruent=H[(D.type == "congruent").values].mean(0))
