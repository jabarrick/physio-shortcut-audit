"""Post hoc (exploratory, NOT registered): the eye-movement analysis in an independent dataset with TRUE EOG channels,
BCI Competition IV 2a (BNCI2014-001; 9 subjects, 22 EEG + 3 monopolar EOG channels, 250 Hz, two sessions).
Left- versus right-hand imagery (144 trials per session and class pair).  Unlike PhysioNet, the cue is an arrow at
the centre of the screen, not a target on one side.

Step 1 (--prepare, CPU; needs `pip install moabb`): per subject, EEG and EOG are resampled to 200 Hz, band-passed
  0.5-75 Hz, notch-filtered at 50 Hz; the EEG is average-referenced; epochs run from -1 to 5 s around the cue
  (cue = trial start + 2 s), i.e. the 0-4 s model window with 1 s of padding, as in the PhysioNet real audit.
  HEOG = EOG3 - EOG1 (right minus left electrode; positive = towards the right), VEOG = EOG2 - (EOG1 + EOG3)/2.
  Cache: results/bnci2a_cache/sub-0X.npz.
Step 2 (GPU, small): leave-one-subject-out; one training subject is held out for early stopping; models csoanet,
  eegnet, shallow (the convolutional core models); seeds 0-2; training settings of configs/default.yaml.  Arms:
    main       average-referenced EEG
    heog_true  true HEOG regressed out of every EEG channel by OLS (coefficients from training trials)
    eog_all    the three EOG channels regressed out jointly by OLS
    plc_orth   null control: the HEOG signal subtracted along a random spatial pattern orthogonal to the HEOG
               coefficient vector and of the same norm (seed 20261012)
  Output: results/bnci2a/.
Step 3 (--analyse, CPU): accuracy per arm; cue-signed HEOG deflection (0.3-1.2 s minus -0.2-0 s; positive =
  towards the cued side), its coupling with the label, and the tercile gradient G(main, control) of the
  main-minus-control accuracy gap over terciles of the deflection standardised within subject (all trials; no
  saccade detector is applied).  Subject bootstrap, 2000 resamples (9 subjects: intervals are wide).
  Writes results/analysis/posthoc_revision7_bnci2a.json.
Usage (repo root):
  pip install moabb
  python scripts/posthoc_revision7_bnci2a.py --prepare
  python scripts/posthoc_revision7_bnci2a.py                      # training
  python scripts/posthoc_revision7_bnci2a.py --analyse
  dry run: python scripts/posthoc_revision7_bnci2a.py --models eegnet --seeds 0 --subjects 1 --max-epochs 1 --out results/bnci2a_dryrun
"""
from __future__ import annotations

import argparse, glob, json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
CACHE = Path("results/bnci2a_cache")
SFREQ, PAD_S, WIN_S, CUE_S = 200.0, 1.0, 4.0, 2.0
EEG = ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz"]
ARMS = ["main", "heog_true", "eog_all", "plc_orth"]
ORTH_SEED = 20261012


def prepare():
    import mne
    from moabb.datasets import BNCI2014_001
    CACHE.mkdir(parents=True, exist_ok=True)
    ds = BNCI2014_001()
    for s in range(1, 10):
        q = CACHE / f"sub-{s:02d}.npz"
        if q.exists():
            continue
        Xs, Es, ys, ses = [], [], [], []
        for si, (sname, runs) in enumerate(sorted(ds.get_data(subjects=[s])[s].items())):
            for rname, raw in sorted(runs.items()):
                raw = raw.copy().load_data()
                stim = [c for c, t in zip(raw.ch_names, raw.get_channel_types()) if t == "stim"][0]
                ev = mne.find_events(raw, stim_channel=stim, verbose="ERROR")
                ev = ev[np.isin(ev[:, 2], [1, 2])]                       # 1 = left hand, 2 = right hand
                if len(ev) == 0:
                    continue
                t_cue = raw.times[ev[:, 0] - raw.first_samp] + CUE_S
                raw.pick(EEG + ["EOG1", "EOG2", "EOG3"]); raw.reorder_channels(EEG + ["EOG1", "EOG2", "EOG3"])
                raw.resample(SFREQ, verbose="ERROR")
                raw.filter(0.5, 75.0, picks="all", verbose="ERROR"); raw.notch_filter(50.0, picks="all", verbose="ERROR")
                D = raw.get_data() * 1e6                                   # microvolts
                eeg = D[:22] - D[:22].mean(0, keepdims=True); eog = D[22:25]
                n0, n1 = int(round(PAD_S * SFREQ)), int(round((WIN_S + PAD_S) * SFREQ))
                for t, lab in zip(t_cue, ev[:, 2]):
                    i = int(round(t * SFREQ))
                    if i - n0 < 0 or i + n1 > D.shape[1]:
                        continue
                    Xs.append(eeg[:, i - n0:i + n1]); Es.append(eog[:, i - n0:i + n1]); ys.append(int(lab) - 1); ses.append(si)
        tmp = q.with_suffix(".tmp.npz")
        np.savez(tmp, X=np.stack(Xs).astype(np.float32), EOG=np.stack(Es).astype(np.float32), y=np.array(ys), session=np.array(ses),
                 pad=np.array(int(round(PAD_S * SFREQ))), core_len=np.array(int(round(WIN_S * SFREQ))))
        tmp.replace(q); print(q.name, len(ys), "trials, left/right", int((np.array(ys) == 0).sum()), int((np.array(ys) == 1).sum()), flush=True)


def load(subjects):
    P = [np.load(CACHE / f"sub-{s:02d}.npz") for s in subjects]
    d = {k: np.concatenate([p[k] for p in P]) for k in ("X", "EOG", "y", "session")}
    d["subj"] = np.concatenate([np.full(len(p["y"]), s) for s, p in zip(subjects, P)])
    d["trial"] = np.concatenate([np.arange(len(p["y"])) for p in P])
    d["pad"], d["core_len"] = int(P[0]["pad"]), int(P[0]["core_len"])
    return d


def heog(E):       # E: trials x 3 x t
    return E[:, 2, :] - E[:, 0, :]


def fit_ols(X, R):   # X: n x c x t ; R: n x k x t  ->  B: c x k
    Xc = X - X.mean(-1, keepdims=True); Rc = R - R.mean(-1, keepdims=True)
    G = np.einsum("nkt,njt->kj", Rc, Rc); C = np.einsum("nct,nkt->ck", Xc, Rc)
    return C @ np.linalg.pinv(G)


def apply_ols(X, R, B):
    Rc = R - R.mean(-1, keepdims=True)
    return (X - np.einsum("ck,nkt->nct", B, Rc)).astype(np.float32)


def transform(arm, TR, D):
    if arm == "main":
        return D["X"], {}
    hT = heog(TR["EOG"])[:, None, :]; b = fit_ols(TR["X"], hT)
    if arm == "heog_true":
        return apply_ols(D["X"], heog(D["EOG"])[:, None, :], b), {"b_norm": float(np.linalg.norm(b))}
    if arm == "eog_all":
        B = fit_ols(TR["X"], TR["EOG"]); return apply_ols(D["X"], D["EOG"], B), {"B_norm": float(np.linalg.norm(B))}
    if arm == "plc_orth":
        v = b[:, 0]; q = np.random.default_rng(ORTH_SEED).standard_normal(v.shape)
        vn = v / (np.linalg.norm(v) + 1e-12); q = q - (q @ vn) * vn; q = q / (np.linalg.norm(q) + 1e-12) * np.linalg.norm(v)
        return apply_ols(D["X"], heog(D["EOG"])[:, None, :], q[:, None]), {"orth_seed": ORTH_SEED, "cos_q_b": float(q @ v / (np.linalg.norm(q) * np.linalg.norm(v) + 1e-12))}
    raise ValueError(arm)


def train(a):
    import pandas as pd
    from p3audit.config import load_config
    from p3audit.metrics.intervention import crop
    from p3audit.models.registry import build_model
    from p3audit.training.trainer import predict_logits, resolve_device, train_model
    from p3audit.utils.common import balanced_accuracy
    cfg = load_config(a.config); device = resolve_device(cfg["training"]["device"])
    out = Path(a.out) if a.out else Path("results/bnci2a"); out.mkdir(parents=True, exist_ok=True)
    allsub = list(range(1, 10))
    for te in (a.subjects or allsub):
        es = te % 9 + 1; tr = [s for s in allsub if s not in (te, es)]
        TR, ES, TE = load(tr), load([es]), load([te]); pad, L = TE["pad"], TE["core_len"]
        for arm in a.arms:
            Xtr, info = transform(arm, TR, TR); Xes, _ = transform(arm, TR, ES); Xte, _ = transform(arm, TR, TE)
            for mdl in a.models:
                for sd in a.seeds:
                    p = out / f"{arm}__{mdl}__sub{te}__seed{sd}.json"
                    if p.exists():
                        continue
                    m = build_model(mdl, cfg, len(EEG), L, EEG); t0 = time.time()
                    tinfo = train_model(m, crop(Xtr, pad, L), TR["y"], crop(Xes, pad, L), ES["y"], cfg, sd, device, a.max_epochs, verbose=False)
                    m.eval(); pred = predict_logits(m, crop(Xte, pad, L), device).argmax(1)
                    pd.DataFrame({"subject": TE["subj"], "trial": TE["trial"], "session": TE["session"], "label": TE["y"],
                                  "correct": (pred == TE["y"]).astype(int), "model": mdl, "seed": sd, "arm": arm}).to_csv(p.with_name(p.stem + "_trials.csv"), index=False)
                    rec = {"arm": arm, "model": mdl, "seed": sd, "test_subject": te, "es_subject": es, "ba": float(balanced_accuracy(TE["y"], pred)),
                           "training": {k: tinfo[k] for k in ("best_val_ba", "epochs") if k in tinfo}, "seconds": time.time() - t0, **info}
                    p.write_text(json.dumps(rec, indent=1))
                    print(f"{arm:10s} {mdl:8s} sub {te} seed {sd}: BA {rec['ba']:.3f} ({rec['seconds']:.0f} s)", flush=True)


def analyse():
    import pandas as pd
    df = pd.concat([pd.read_csv(f) for f in glob.glob("results/bnci2a/*_trials.csv")])
    key = ["model", "subject", "trial", "label"]
    arms = [x for x in ARMS if x in set(df.arm)]
    w = None
    for arm in arms:
        t = df[df.arm == arm].assign(correct=lambda x: x.correct.astype(float)).groupby(key, as_index=False).correct.mean().rename(columns={"correct": arm})
        w = t if w is None else w.merge(t, on=key)
    defl = {}
    for s in sorted(w.subject.unique()):
        z = np.load(CACHE / f"sub-{s:02d}.npz"); h = heog(z["EOG"]); p = int(z["pad"])
        v = (h[:, p + 60:p + 240].mean(1) - h[:, p - 40:p].mean(1)) * np.where(z["y"] == 1, 1.0, -1.0)
        for k in range(len(v)): defl[(s, k)] = v[k]
    w["d"] = [defl[(s, k)] for s, k in zip(w.subject, w.trial)]
    w["dz"] = w.groupby(["model", "subject"]).d.transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
    w["tert"] = w.groupby("model").dz.transform(lambda x: pd.qcut(x, 3, labels=False))

    def stats(x):
        t0, t2 = x[x.tert == 0], x[x.tert == 2]; o = {f"acc_{a}": x[a].mean() for a in arms}
        o["deflection_cue_signed_uV"] = x[x.model == x.model.iloc[0]].d.mean()
        for a in arms[1:]:
            o[f"gap_main-{a}"] = (x.main - x[a]).mean()
            o[f"G_{a}"] = (t2.main - t2[a]).mean() - (t0.main - t0[a]).mean()
        if "heog_true" in arms and "plc_orth" in arms: o["G_heog_true-G_plc_orth"] = o["G_heog_true"] - o["G_plc_orth"]
        o["acc_main_gradient"] = t2.main.mean() - t0.main.mean()
        return o

    rng = np.random.default_rng(20261013); subs = np.array(sorted(w.subject.unique())); g = {s: x for s, x in w.groupby("subject")}
    est = stats(w); B = pd.DataFrame([stats(pd.concat([g[s] for s in rng.choice(subs, subs.size)])) for _ in range(2000)])
    one = w[w.model == w.model.iloc[0]]
    per_subj = one.groupby("subject").d.mean()
    out = {"models": sorted(w.model.unique()), "n_subjects": int(subs.size), "n_trials": int(len(one)), "arms": arms,
           "deflection_by_subject_uV": {int(k): float(v) for k, v in per_subj.items()},
           "estimates": {k: {"est": float(v), "ci_95": [float(B[k].quantile(.025)), float(B[k].quantile(.975))]} for k, v in est.items()},
           "per_model_acc": {a: w.groupby("model")[a].mean().round(4).to_dict() for a in arms}}
    # label information in the true HEOG itself: per-subject AUC of the baseline-corrected HEOG (positive = right)
    # for right- against left-hand trials, in four windows (added after the first look at the results; exploratory)
    auc = lambda x, z: float((x[:, None] > z[None, :]).mean() + 0.5 * (x[:, None] == z[None, :]).mean())
    out["heog_label_auc"] = {}
    for wn, (t0_, t1_) in {"0-0.5": (0, .5), "0.3-1.2": (.3, 1.2), "0.5-1.5": (.5, 1.5), "1.5-4": (1.5, 4.0)}.items():
        A = []
        for s_ in subs:
            z = np.load(CACHE / f"sub-{s_:02d}.npz"); h = heog(z["EOG"]); p = int(z["pad"])
            v = h[:, p + int(t0_ * SFREQ):p + int(t1_ * SFREQ)].mean(1) - h[:, p - 40:p].mean(1)
            A.append(auc(v[z["y"] == 1], v[z["y"] == 0]))
        A = np.array(A); bs = [rng.choice(A, len(A)).mean() for _ in range(5000)]
        out["heog_label_auc"][wn] = {"mean": float(A.mean()), "ci_95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))], "per_subject": [round(float(x), 3) for x in A]}
        print(f"HEOG label AUC {wn:8s} {A.mean():.3f} [{np.quantile(bs, .025):.3f}, {np.quantile(bs, .975):.3f}]")
    Path("results/analysis").mkdir(parents=True, exist_ok=True)
    json.dump(out, open("results/analysis/posthoc_revision7_bnci2a.json", "w"), indent=1)
    for k, v in out["estimates"].items(): print(f"{k:28s} {v['est']:.3f} [{v['ci_95'][0]:.3f}, {v['ci_95'][1]:.3f}]")
    print(out["per_model_acc"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true"); ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS); ap.add_argument("--models", nargs="+", default=["csoanet", "eegnet", "shallow"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2]); ap.add_argument("--subjects", nargs="+", type=int, default=None)
    ap.add_argument("--config", default="configs/default.yaml"); ap.add_argument("--max-epochs", type=int, default=None); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    prepare() if a.prepare else analyse() if a.analyse else train(a)
