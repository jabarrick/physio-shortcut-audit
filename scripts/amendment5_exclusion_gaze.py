"""Amendment 5 (OSF): saccade exclusion during training, and a control with measured gaze regressed out.
Outside the frozen package (p3audit/ and configs/ are unchanged). Needs a GPU for training.

Part A - exclusion in training (both datasets). Decoders and controls are trained and early-stopped only on trials
without a saccade, as in a pipeline that rejects saccade trials; they are tested on all test trials.
  PhysioNet (folds, seeds, early-stopping subjects and settings of the registered real audit; output results/real_a5):
    main_excl  main decoder, training and early-stopping trials restricted to type == "none" (AF7 - AF8 detector)
    ols_excl   AF7 - AF8 regressed out of all channels by OLS (as heog_ols), coefficients fitted on the
               restricted training trials, training restricted likewise
  Eye-tracking dataset (folds, seeds and settings of Amendment 4; output results/gf2025_a5):
    main_excl, ols_excl (F8 - F7, as hor_ols)  with training and early-stopping trials restricted to trials without a
               (early stopping on the kept trials of the registered early-stopping subject plus three training
               subjects drawn with default_rng(20261105 + fold), which are then not used for training)
               saccade on the F8 - F7 detector AND without a gaze-classified saccade (gaze classification with the
               corrected step statistic of scripts/posthoc_a4_corrected_gaze_step.py; trials without valid gaze are
               kept if the F8 - F7 detector found no saccade)
Part B - measured gaze regressed out (eye-tracking dataset, all training trials):
    gaze_ols   horizontal gaze position (deg) and its first difference regressed out of all 60 channels by OLS,
               coefficients fitted on training trials with valid gaze; within a trial invalid samples are linearly
               interpolated; trials without valid gaze (< 75 % valid samples at -0.2-1.2 s, or a session without gaze)
               get zero regressors and are not corrected
Analysis: python scripts/amendment5_exclusion_gaze.py analyse  ->  results/analysis/amendment5.json
Usage (repository root):
  python scripts/amendment5_exclusion_gaze.py train-physionet [--models ...] [--seeds ...] [--folds ...] [--max-epochs N --out DIR]
  python scripts/amendment5_exclusion_gaze.py train-gf        [--arms main_excl ols_excl gaze_ols] [...]
  dry run: add --models eegnet --seeds 0 --folds 0 --max-epochs 1 --out results/<name>_dryrun
"""
from __future__ import annotations
import argparse, glob, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

MODELS = ["csoanet", "eegnet", "shallow", "cbramod"]
N_BOOT, BOOT_SEED, ALPHA = 2000, 20261104, 0.0125
EXCL_ES_SEED = 20261105                                   # + fold: extra early-stopping subjects, eye-tracking exclusion arms        # four tests: two-sided 98.75 % intervals


def subset(D, keep):
    out = {k: (v[keep] if isinstance(v, np.ndarray) and len(v) == len(keep) else v) for k, v in D.items()}
    return out


# ------------------------------------------------------------------ PhysioNet
def train_physionet(a):
    from p3audit.config import load_config
    from p3audit.data.splits import real_audit_folds
    from p3audit.experiments.context import Context
    from p3audit.experiments.real_audit import _stack
    from p3audit.metrics.intervention import crop
    from p3audit.models.registry import build_model
    from p3audit.training.trainer import predict_logits, resolve_device, train_model
    from p3audit.utils.common import balanced_accuracy
    from posthoc_revision3_controls import fit_pair_regression, apply_pair_regression
    cfg = load_config(a.config); ctx = Context.create(cfg); sp, ch = ctx.split, list(ctx.ch_names)
    included = sorted(set(sp.train) | set(sp.val) | set(sp.test))
    folds = real_audit_folds(included, sp.pilot, cfg["split"]["real_audit_folds"], cfg["seeds"]["split"])
    device = resolve_device(cfg["training"]["device"])
    out = Path(a.out) if a.out else ctx.results / "real_a5"; out.mkdir(parents=True, exist_ok=True)
    for f in folds:
        if a.folds is not None and f["fold"] not in a.folds:
            continue
        nonpilot = [s for s in f["train"] if s not in sp.pilot]          # identical to run_real_fold
        es_s = nonpilot[: max(2, len(nonpilot) // 10)]; tr_s = [s for s in f["train"] if s not in es_s]
        TR, ES, TE = _stack(ctx, tr_s), _stack(ctx, es_s), _stack(ctx, f["test"])
        TRx, ESx = subset(TR, TR["type"] == "none"), subset(ES, ES["type"] == "none")
        pad, L = TE["pad"], TE["core_len"]
        for arm in ("main_excl", "ols_excl"):
            if arm == "main_excl":
                Xtr, Xes, Xte, info = TRx["X"], ESx["X"], TE["X"], {}
            else:
                reg = fit_pair_regression(TRx["X"], ch, ("AF7", "AF8"))
                Xtr, Xes, Xte = (apply_pair_regression(d["X"], ch, reg) for d in (TRx, ESx, TE))
                info = {"pair": ["AF7", "AF8"], "b_norm": float(np.linalg.norm(reg["b"]))}
            info |= {"n_train_trials": int(len(TRx["y"])), "n_es_trials": int(len(ESx["y"])), "training_trials": "type == none"}
            for mdl in a.models:
                for sd in a.seeds:
                    p = out / f"{arm}__{mdl}__fold{f['fold']}__seed{sd}.json"
                    if p.exists():
                        continue
                    m = build_model(mdl, cfg, len(ch), L, ch); t0 = time.time()
                    tinfo = train_model(m, crop(Xtr, pad, L), TRx["y"], crop(Xes, pad, L), ESx["y"], cfg, sd, device, a.max_epochs, verbose=False)
                    m.eval(); pred = predict_logits(m, crop(Xte, pad, L), device).argmax(1)
                    pd.DataFrame({"subject": TE["subj"], "trial": TE["trial"], "run": TE["run"], "type": TE["type"], "label": TE["y"],
                                  "correct": (pred == TE["y"]).astype(int), "model": mdl, "seed": sd, "fold": f["fold"], "arm": arm}
                                 ).to_csv(p.with_name(p.stem + "_trials.csv"), index=False)
                    rec = {"arm": arm, "model": mdl, "seed": sd, "fold": f["fold"], "ba": float(balanced_accuracy(TE["y"], pred)),
                           "max_epochs_override": a.max_epochs, "training": {k: tinfo[k] for k in ("best_val_ba", "epochs") if k in tinfo},
                           "seconds": time.time() - t0, **info}
                    p.write_text(json.dumps(rec, indent=1))
                    print(f"PN {arm:9s} {mdl:8s} fold {f['fold']} seed {sd}: BA {rec['ba']:.3f} ({rec['seconds'] / 60:.1f} min)", flush=True)


# ------------------------------------------------------------------ eye-tracking dataset
def step_stat_valid(x, n):
    """Step statistic over valid samples only (the correction of scripts/posthoc_a4_corrected_gaze_step.py)."""
    v = np.isfinite(x); x0 = np.where(v, x, 0.0)
    cs = np.concatenate([np.zeros(x.shape[:-1] + (1,)), np.cumsum(x0, -1)], -1)
    cn = np.concatenate([np.zeros(x.shape[:-1] + (1,)), np.cumsum(v, -1)], -1)
    s = np.full(x.shape, np.nan); T = x.shape[-1]; t = np.arange(n, T - n + 1)
    with np.errstate(all="ignore"):
        nf = cn[..., t + n] - cn[..., t]; nb = cn[..., t] - cn[..., t - n]
        f = (cs[..., t + n] - cs[..., t]) / nf; b = (cs[..., t] - cs[..., t - n]) / nb
        s[..., t] = np.where((nf > 0) & (nb > 0), f - b, np.nan)
    return s


def gf_classes():
    """Per-trial exclusion flag for the eye-tracking dataset (True = trial kept for training)."""
    import amendment4_gf2025 as a4
    orig = a4.step_stat; a4.step_stat = step_stat_valid
    try:
        T = a4.trial_table()
    finally:
        a4.step_stat = orig
    keep = (T.type_hor == "none") & ~T.type_gaze.isin(["congruent", "incongruent"])
    return {(int(s), int(k)): bool(v) for s, k, v in zip(T.subject, T.trial, keep)}, T


def gaze_regressors(D, a4):
    """n x 2 x t: interpolated horizontal gaze and its first difference; zeros for trials without valid gaze."""
    G, V = D["GAZE"].astype(float), D["GVALID"].astype(bool)
    n, T = G.shape; R = np.zeros((n, 2, T)); t = np.arange(T)
    vfrac = V[:, a4.idx(*a4.W_VALID)].mean(1)
    ok = (vfrac >= a4.TRIAL_VALID_MIN) & D["gaze_session"]
    for k in np.where(ok)[0]:
        v = V[k] & np.isfinite(G[k])
        g = np.interp(t, t[v], G[k][v]); R[k, 0] = g; R[k, 1] = np.r_[0.0, np.diff(g)]
    return R, ok


def gf_load(a4, subjects):
    D = a4.load(subjects)
    P = [np.load(a4.CACHE / f"sub-{s:02d}.npz") for s in subjects]
    D["GAZE"] = np.concatenate([p["GAZE"] for p in P]); D["GVALID"] = np.concatenate([p["GVALID"] for p in P])
    sess = json.loads((a4.CACHE / "sessions.json").read_text())
    def avail(s, q):
        return bool(sess[f"S{s:02d}/Sess{q + 1:02d}"].get("gaze_available", False))
    D["gaze_session"] = np.array([avail(int(s), int(q)) for s, q in zip(D["subj"], D["session"])])
    return D


def train_gf(a):
    import amendment4_gf2025 as a4
    from p3audit.config import load_config
    from p3audit.metrics.intervention import crop
    from p3audit.models.registry import build_model
    from p3audit.training.trainer import predict_logits, resolve_device, train_model
    from p3audit.utils.common import balanced_accuracy
    keepmap, _ = gf_classes()
    cfg = load_config(a.config); device = resolve_device(cfg["training"]["device"])
    out = Path(a.out) if a.out else ROOT / "results" / "gf2025_a5"; out.mkdir(parents=True, exist_ok=True)
    for f in a4.folds():
        if a.folds is not None and f["fold"] not in a.folds:
            continue
        TR, ES, TE = gf_load(a4, f["train"]), gf_load(a4, [f["es"]]), gf_load(a4, f["test"]); pad, L = TE["pad"], TE["core_len"]
        # exclusion arms: the registered early-stopping subject keeps too few trials, so early stopping uses the kept
        # trials of that subject plus three training subjects drawn with a fixed generator (removed from training)
        extra = sorted(int(x) for x in np.random.default_rng(EXCL_ES_SEED + f["fold"]).choice(f["train"], 3, replace=False))
        tr_x = [s for s in f["train"] if s not in extra]
        TRx0, ESx0 = gf_load(a4, tr_x), gf_load(a4, [f["es"]] + extra)
        kt = np.array([keepmap[(int(s), int(k))] for s, k in zip(TRx0["subj"], TRx0["trial"])])
        ke = np.array([keepmap[(int(s), int(k))] for s, k in zip(ESx0["subj"], ESx0["trial"])])
        for arm in a.arms:
            if arm == "main_excl":
                TRa, ESa = subset(TRx0, kt), subset(ESx0, ke); Xtr, Xes, Xte = TRa["X"], ESa["X"], TE["X"]
                info = {"training_trials": "no saccade on F8-F7 and none on gaze (corrected step statistic)", "es_subjects": [f["es"]] + extra}
            elif arm == "ols_excl":
                TRa, ESa = subset(TRx0, kt), subset(ESx0, ke); b = a4.fit_ols(TRa["X"], TRa["HOR"][:, None, :])
                Xtr, Xes, Xte = (a4.apply_ols(d["X"], d["HOR"][:, None, :], b) for d in (TRa, ESa, TE))
                info = {"regressor": "F8-F7", "b_norm": float(np.linalg.norm(b)), "training_trials": "as main_excl", "es_subjects": [f["es"]] + extra}
            elif arm == "gaze_ols":
                TRa, ESa = TR, ES
                (Rtr, oktr), (Res, _), (Rte, _) = gaze_regressors(TR, a4), gaze_regressors(ES, a4), gaze_regressors(TE, a4)
                b = a4.fit_ols(TR["X"][oktr], Rtr[oktr])
                Xtr, Xes, Xte = a4.apply_ols(TR["X"], Rtr, b), a4.apply_ols(ES["X"], Res, b), a4.apply_ols(TE["X"], Rte, b)
                info = {"regressors": ["gaze_x_deg", "diff_gaze_x"], "b_norm": float(np.linalg.norm(b)), "n_fit_trials": int(oktr.sum())}
            else:
                raise ValueError(arm)
            info |= {"n_train_trials": int(len(TRa["y"])), "n_es_trials": int(len(ESa["y"]))}
            for mdl in a.models:
                for sd in a.seeds:
                    p = out / f"{arm}__{mdl}__fold{f['fold']}__seed{sd}.json"
                    if p.exists():
                        continue
                    m = build_model(mdl, cfg, len(a4.EEG), L, a4.EEG); t0 = time.time()
                    tinfo = train_model(m, crop(Xtr, pad, L), TRa["y"], crop(Xes, pad, L), ESa["y"], cfg, sd, device, a.max_epochs, verbose=False)
                    m.eval(); pred = predict_logits(m, crop(Xte, pad, L), device).argmax(1)
                    pd.DataFrame({"subject": TE["subj"], "trial": TE["trial"], "session": TE["session"], "label": TE["y"],
                                  "correct": (pred == TE["y"]).astype(int), "model": mdl, "seed": sd, "arm": arm, "fold": f["fold"]}
                                 ).to_csv(p.with_name(p.stem + "_trials.csv"), index=False)
                    rec = {"arm": arm, "model": mdl, "seed": sd, "fold": f["fold"], "test_subjects": f["test"], "es_subject": f["es"],
                           "ba": float(balanced_accuracy(TE["y"], pred)), "max_epochs_override": a.max_epochs,
                           "training": {k: tinfo[k] for k in ("best_val_ba", "epochs") if k in tinfo}, "seconds": time.time() - t0, **info}
                    p.write_text(json.dumps(rec, indent=1))
                    print(f"GF {arm:9s} {mdl:8s} fold {f['fold']} seed {sd}: BA {rec['ba']:.3f} ({rec['seconds']:.0f} s)", flush=True)


# ------------------------------------------------------------------ analysis
def boot(W, stats, alpha):
    rng = np.random.default_rng(BOOT_SEED); subs = np.array(sorted(W.subject.unique())); g = {s: x for s, x in W.groupby("subject")}
    est = stats(W); B = pd.DataFrame([stats(pd.concat([g[s] for s in rng.choice(subs, subs.size)])) for _ in range(N_BOOT)])
    return {k: {"est": float(v), "ci_98.75": [float(B[k].quantile(alpha / 2)), float(B[k].quantile(1 - alpha / 2))],
                "ci_95": [float(B[k].quantile(.025)), float(B[k].quantile(.975))]} for k, v in est.items()}


def wide(frames, key):
    W = None
    for name, df in frames.items():
        t = df.groupby(key, as_index=False).correct.mean().rename(columns={"correct": name})
        W = t if W is None else W.merge(t, on=key)
    return W


def G(x, a1, a2, tcol="tert"):
    t0, t2 = x[x[tcol] == 0], x[x[tcol] == 2]
    return (t2[a1] - t2[a2]).mean() - (t0[a1] - t0[a2]).mean()


def analyse(a):
    from p3audit.constants import PHYSIONET_CHANNELS as CH
    rd = lambda pat: pd.concat([pd.read_csv(f) for f in glob.glob(pat)])
    out = {}
    # PhysioNet: terciles exactly as scripts/amendment2_real.py
    key = ["model", "subject", "trial", "type", "label"]
    main = rd("results/real/*_trials.csv"); main = main[main.arm == "main"]; r3 = rd("results/real_r3/*_trials.csv"); a5 = rd("results/real_a5/*_trials.csv")
    W = wide({"main": main, "heog_ols": r3[r3.arm == "heog_ols"], "main_excl": a5[a5.arm == "main_excl"], "ols_excl": a5[a5.arm == "ols_excl"]}, key)
    i7, i8 = CH.index("AF7"), CH.index("AF8"); defl = {}
    for s in sorted(W.subject.unique()):
        z = np.load(f"results/real_cache/sub-{s:03d}.npz", allow_pickle=True); h = z["X"][:, i7, :] - z["X"][:, i8, :]; p = int(z["pad"])
        v = (h[:, p + 60:p + 240].mean(1) - h[:, p - 40:p].mean(1)) * np.where(z["y"] == 1, 1.0, -1.0)
        defl.update({(s, k): v[k] for k in range(len(v))})
    W["d"] = [defl[(s, k)] for s, k in zip(W.subject, W.trial)]
    W["d"] *= 1.0 if W.loc[W.type == "congruent", "d"].mean() > 0 else -1.0
    W["dz"] = W.groupby(["model", "subject"]).d.transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
    n = W.type == "none"; W.loc[n, "tert"] = W[n].groupby("model").dz.transform(lambda x: pd.qcut(x, 3, labels=False))
    def pn(w):
        nn = w[w.type == "none"]
        return {"A5_R1_G_excl": G(nn, "main_excl", "ols_excl"), "G_full_ols": G(nn, "main", "heog_ols"),
                "G_full_minus_excl": G(nn, "main", "heog_ols") - G(nn, "main_excl", "ols_excl"),
                "gap_none_excl": (nn.main_excl - nn.ols_excl).mean(), "gap_none_full": (nn.main - nn.heog_ols).mean(),
                "acc_main_excl": w.main_excl.mean(), "acc_ols_excl": w.ols_excl.mean(), "acc_main": w.main.mean(),
                "acc_main_excl_none": nn.main_excl.mean(), "acc_main_none": nn.main.mean()}
    out["physionet"] = boot(W, pn, ALPHA)
    # eye-tracking dataset: registered terciles (amendment4 trial table), corrected classification for the subset
    import amendment4_gf2025 as a4
    Treg = a4.trial_table()
    _, Tc = gf_classes()
    g4 = rd("results/gf2025/*_trials.csv"); g5 = rd("results/gf2025_a5/*_trials.csv"); k2 = ["model", "subject", "trial"]
    W = wide({"main": g4[g4.arm == "main"], "hor_ols": g4[g4.arm == "hor_ols"], "main_excl": g5[g5.arm == "main_excl"],
              "ols_excl": g5[g5.arm == "ols_excl"], "gaze_ols": g5[g5.arm == "gaze_ols"]}, k2)
    W = W.merge(Tc[["subject", "trial", "gaze_valid", "type_gaze", "d_hor"]], on=["subject", "trial"])
    W["zh"] = W.groupby(["model", "subject"]).d_hor.transform(lambda x: (x - x.mean()) / (x.std() + 1e-12))
    nn = W.gaze_valid & (W.type_gaze == "none"); W.loc[nn, "tert"] = W[nn].groupby("model").zh.transform(lambda x: pd.qcut(x, 3, labels=False))
    def gf(w):
        x = w[w.gaze_valid & (w.type_gaze == "none")]
        return {"A5_R2_G_excl": G(x, "main_excl", "ols_excl"), "A5_R3_gap_gaze_ols": (x.main - x.gaze_ols).mean(),
                "A5_R4_G_gaze_ols": G(x, "main", "gaze_ols"), "G_full_hor": G(x, "main", "hor_ols"),
                "G_full_minus_excl": G(x, "main", "hor_ols") - G(x, "main_excl", "ols_excl"),
                "gap_none_excl": (x.main_excl - x.ols_excl).mean(), "acc_main_excl": w.main_excl.mean(), "acc_ols_excl": w.ols_excl.mean(),
                "acc_gaze_ols": w.gaze_ols.mean(), "acc_main": w.main.mean(), "acc_hor_ols": w.hor_ols.mean()}
    out["eye_tracking"] = boot(W, gf, ALPHA)
    dec = lambda r: "supported" if r["ci_98.75"][0] > 0 else "not supported"
    out["tests"] = {"A5-R1": dec(out["physionet"]["A5_R1_G_excl"]), "A5-R2": dec(out["eye_tracking"]["A5_R2_G_excl"]),
                    "A5-R3": dec(out["eye_tracking"]["A5_R3_gap_gaze_ols"]), "A5-R4": dec(out["eye_tracking"]["A5_R4_G_gaze_ols"])}
    out["settings"] = {"n_boot": N_BOOT, "seed": BOOT_SEED, "interval": "two-sided 98.75 %"}
    Path("results/analysis").mkdir(parents=True, exist_ok=True)
    Path("results/analysis/amendment5.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out["tests"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["train-physionet", "train-gf", "analyse"])
    ap.add_argument("--arms", nargs="+", default=["main_excl", "ols_excl", "gaze_ols"])
    ap.add_argument("--models", nargs="+", default=MODELS); ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--folds", nargs="+", type=int, default=None); ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--max-epochs", type=int, default=None); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    {"train-physionet": train_physionet, "train-gf": train_gf, "analyse": analyse}[a.cmd](a)
