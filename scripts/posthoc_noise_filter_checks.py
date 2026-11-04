"""Post hoc checks without training of the 'learned ocular filter applied to noise' account (exploratory).

In trials without a detected (PhysioNet, AF7 - AF8 detector) or measured (eye-tracking dataset, registered gaze
classification) saccade:
 (a) label information in the HEOG deflection itself: per-subject AUC of the unsigned 0.3-1.2 s deflection for the
     cue side, the standardised mean cue-signed deflection (Cohen's d within subject), and a within-subject
     cross-validated shrinkage LDA on the 0.3-1.2 s deflections of all channels;
 (b) persistence: correlation (within subject) of the cue-signed 0.3-1.2 s deflection with the mean cue-signed
     deflection at 1.5-4.0 s (same baseline -0.2-0 s), and the same correlation with sign-randomised cues;
 (c) eye-tracking dataset: positive control with step-vector-like coefficients (fitted, per fold, on the 0.3-1.2 s
     deflections of training subjects' gaze-classified saccade trials) instead of the registered whole-epoch OLS.
Output: results/analysis/posthoc_noise_filter_checks.json
"""
import sys, glob, json, numpy as np, pandas as pd
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import amendment4_gf2025 as a
from p3audit.constants import PHYSIONET_CHANNELS as CH
rng = np.random.default_rng(20261103)
def auc(x, y):
    p, n = x[y == 1], x[y == 0]; return float((p[:, None] > n[None, :]).mean() + 0.5 * (p[:, None] == n[None, :]).mean())
def lda_auc(Y, y, seed):
    k = np.arange(len(y)) % 5; np.random.default_rng(seed).shuffle(k); sc = np.zeros(len(y))
    for j in range(5):
        tr, te = k != j, k == j
        if len(np.unique(y[tr])) < 2: return np.nan
        m1, m0 = Y[tr][y[tr] == 1].mean(0), Y[tr][y[tr] == 0].mean(0); Sw = np.cov((Y[tr] - np.where(y[tr][:, None] == 1, m1, m0)).T)
        Sw = 0.5 * Sw + 0.5 * np.trace(Sw) / len(Sw) * np.eye(len(Sw)); sc[te] = Y[te] @ np.linalg.solve(Sw, m1 - m0)
    return auc(sc, y)
def summarise(df):
    """df: subject, y, u (unsigned deflection, oriented so that y = 1 is positive for congruent saccades), late_u, F (channel features)"""
    o = {}; A, D, L, R, Rr = [], [], [], [], []
    for s, x in df.groupby("subject"):
        y = x.y.to_numpy(); u = x.u.to_numpy(); sg = np.where(y == 1, 1, -1)
        if len(x) >= 10 and len(np.unique(y)) == 2:
            A.append(auc(u, y)); d = u * sg; D.append(d.mean() / (d.std() + 1e-12))
            ds, ls = d, x.late_u.to_numpy() * sg
            if ds.std() > 0 and ls.std() > 0: R.append(np.corrcoef(ds, ls)[0, 1])
            rs = rng.choice([-1, 1], len(y)); Rr.append(np.corrcoef(u * rs, x.late_u.to_numpy() * rs)[0, 1])
        if len(x) >= 15 and min((y == 1).sum(), (y == 0).sum()) >= 5:
            L.append(lda_auc(np.stack(x.F.to_numpy()), y, int(s)))
    m = lambda v: [float(np.nanmean(v)), int(np.isfinite(v).sum())]
    o["auc_deflection_label"] = m(A); o["cohen_d_cue_signed"] = m(D); o["lda_all_channels_auc"] = m(L)
    o["corr_early_late_cue_signed"] = m(R); o["corr_early_late_random_signs"] = m(Rr); o["n_trials"] = int(len(df))
    return o
out = {}
# ---------------- PhysioNet
rows = []
for f in sorted(glob.glob("results/real_cache/sub-*.npz")):
    s = int(f.split("sub-")[1][:3]); z = np.load(f, allow_pickle=True); X = z["X"].astype(float); y = z["y"]; ty = z["type"]; pad = int(z["pad"])
    sl = lambda t0, t1: slice(pad + int(t0 * 200), pad + int(t1 * 200))
    base = X[:, :, sl(-0.2, 0)].mean(-1); Fe = X[:, :, sl(0.3, 1.2)].mean(-1) - base; Fl = X[:, :, sl(1.5, 4.0)].mean(-1) - base
    h = lambda F: F[:, CH.index("AF7")] - F[:, CH.index("AF8")]
    for k in range(len(y)):
        rows.append({"subject": s, "y": int(y[k]), "type": str(ty[k]), "u": h(Fe)[k], "late_u": h(Fl)[k], "F": Fe[k]})
P = pd.DataFrame(rows)
orient = 1.0 if (P[P.type == "congruent"].u * np.where(P[P.type == "congruent"].y == 1, 1, -1)).mean() > 0 else -1.0
P["u"] *= orient; P["late_u"] *= orient
out["physionet"] = {ty: summarise(P[P.type == ty]) for ty in ("none", "congruent")}
# ---------------- eye-tracking dataset
T = a.trial_table(); rows = []; S = {}
for s in a.subjects_in_cache():
    z = np.load(a.CACHE / f"sub-{s:02d}.npz"); X = z["X"].astype(float); H = z["HOR"].astype(float)
    base = X[:, :, a.idx(-0.2, 0)].mean(-1); Fe = X[:, :, a.idx(0.3, 1.2)].mean(-1) - base; Fl = X[:, :, a.idx(1.5, 4.0)].mean(-1) - base
    hb = H[:, a.idx(-0.2, 0)].mean(-1); he = H[:, a.idx(0.3, 1.2)].mean(-1) - hb; hl = H[:, a.idx(1.5, 4.0)].mean(-1) - hb
    S[s] = (Fe, he)
    for k in range(len(he)):
        rows.append({"subject": s, "trial": k, "u": he[k], "late_u": hl[k], "F": Fe[k]})
G = pd.DataFrame(rows).merge(T[["subject", "trial", "label", "gaze_valid", "type_gaze", "u_gaze"]], on=["subject", "trial"]).rename(columns={"label": "y"})
out["eye_tracking"] = {ty: summarise(G[G.gaze_valid & (G.type_gaze == ty)]) for ty in ("none", "congruent")}
# (c) step-vector-like coefficients
V = G[G.gaze_valid & G.u_gaze.notna()].copy(); V["cue"] = np.where(V.y == 1, 1, -1)
sac = G[G.gaze_valid & G.type_gaze.isin(["congruent", "incongruent"])]
res = []
for f in a.folds():
    tr = sac[sac.subject.isin(f["train"])]; Fx = np.stack(tr.F.to_numpy()); hx = tr.u.to_numpy()
    b = (Fx * hx[:, None]).sum(0) / (hx ** 2).sum()
    te = V[V.subject.isin(f["test"])]
    for (_, r) in te.iterrows():
        res.append({"subject": r.subject, "trial": r.trial, "after_sv": r.F - b * r.u})
V = V.merge(pd.DataFrame(res), on=["subject", "trial"])
def ridge_r2(df, col, lam=1.0):
    num = den = 0.0
    for s, x in df.groupby("subject"):
        if len(x) < 20: continue
        Y = np.stack(x[col].to_numpy()).copy(); u = x.u_gaze.to_numpy(float).copy(); cs = x.cue.to_numpy()
        for c in (-1, 1):
            m = cs == c; Y[m] -= Y[m].mean(0); u[m] -= u[m].mean()
        Y = Y / (Y.std(0) + 1e-9); k = np.arange(len(u)) % 5; np.random.default_rng(int(s)).shuffle(k); pred = np.zeros_like(u)
        for j in range(5):
            tr_, te_ = k != j, k == j; A_ = Y[tr_]; w = np.linalg.solve(A_.T @ A_ + lam * len(A_) * np.eye(A_.shape[1]), A_.T @ u[tr_]); pred[te_] = Y[te_] @ w
        num += ((u - pred) ** 2).sum(); den += (u ** 2).sum()
    return float(1 - num / den)
cg = V[V.type_gaze == "congruent"]
L0 = [lda_auc(np.stack(x.F.to_numpy()), x.y.to_numpy(), int(s)) for s, x in cg.groupby("subject") if len(x) >= 15 and x.y.value_counts().min() >= 5]
L1 = [lda_auc(np.stack(x.after_sv.to_numpy()), x.y.to_numpy(), int(s)) for s, x in cg.groupby("subject") if len(x) >= 15 and x.y.value_counts().min() >= 5]
out["eye_tracking_step_vector_positive_control"] = {"ridge_r2_before": ridge_r2(V, "F"), "ridge_r2_after_step_vector": ridge_r2(V, "after_sv"),
    "lda_auc_congruent_before": float(np.nanmean(L0)), "lda_auc_congruent_after_step_vector": float(np.nanmean(L1)), "n_subjects_lda": int(np.isfinite(L1).sum()),
    "note": "coefficients per fold from training subjects' gaze-classified saccade trials, b_c = sum(dX_c dH) / sum(dH^2) on 0.3-1.2 s deflections"}
json.dump(out, open("results/analysis/posthoc_noise_filter_checks.json", "w"), indent=1); print(json.dumps(out, indent=1))
