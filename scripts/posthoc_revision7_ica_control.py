"""Post hoc (exploratory, NOT registered; requested by the second critical review): an ICA-based ocular control.
Step 1 (--prepare, CPU): for every subject of the real audit, fit an ICA (extended infomax, 30 components) on that
  subject's cached model-stream epochs (results/real_cache; a 1 Hz high-passed copy is used for fitting, as ICLabel
  expects), label the components with ICLabel (mne-icalabel) and remove those labelled 'eye blink' (ICLabel's eye
  class) with probability >= 0.5.  The cleaned epochs are written to results/real_cache_ica/sub-XXX.npz together with
  the number and labels of removed components.  Labels (y) are not used.
Step 2 (GPU): control models trained and tested on the cleaned input (arm 'ica_eye'), same folds, seeds (0-2),
  early-stopping subjects and training settings as scripts/amendment2_controls.py.  Output: results/real_r7/.
Step 3 (--analyse, CPU): accuracy of the ICA control, main - control gap in trials without a detected saccade,
  tercile gradient G(main, ica_eye) against G(main, heog_ols), and the cue-signed AF7 - AF8 deflection that remains
  after cleaning.  Writes results/analysis/posthoc_revision7_ica.json.
Usage (repo root):
  pip install mne-icalabel            (needs torch or onnxruntime)
  python scripts/posthoc_revision7_ica_control.py --prepare
  python scripts/posthoc_revision7_ica_control.py --models csoanet eegnet shallow cbramod
  python scripts/posthoc_revision7_ica_control.py --analyse
  dry run: python scripts/posthoc_revision7_ica_control.py --models eegnet --seeds 0 --folds 0 --max-epochs 1 --out results/real_r7_dryrun
"""
from __future__ import annotations

import argparse, glob, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
ICA_DIR = Path("results/real_cache_ica")
N_COMP, ICA_SEED, P_EYE, SFREQ = 30, 20261010, 0.5, 200.0


def clean_subject(path, ch):
    import mne
    from mne.preprocessing import ICA
    from mne_icalabel import label_components
    z = np.load(path, allow_pickle=True); X = z["X"].astype(np.float64) * 1e-6      # cache is in microvolts
    info = mne.create_info(list(ch), SFREQ, "eeg")
    ep = mne.EpochsArray(X, info, verbose="ERROR"); ep.set_montage("standard_1005", match_case=False, on_missing="warn", verbose="ERROR")
    ep.set_eeg_reference("average", projection=False, verbose="ERROR")
    fit = ep.copy().filter(1.0, None, verbose="ERROR")
    ica = ICA(n_components=N_COMP, method="infomax", fit_params=dict(extended=True), random_state=ICA_SEED, max_iter="auto", verbose="ERROR")
    ica.fit(fit, verbose="ERROR")
    lab = label_components(fit, ica, method="iclabel")
    labels, probs = list(lab["labels"]), np.asarray(lab["y_pred_proba"], float)
    excl = [i for i, (l, p) in enumerate(zip(labels, probs)) if l == "eye blink" and p >= P_EYE]
    out = ica.apply(ep.copy(), exclude=excl, verbose="ERROR").get_data(copy=False) * 1e6
    return out.astype(np.float32), {"n_removed": len(excl), "removed": excl, "labels": labels, "probs": [round(float(p), 3) for p in probs]}


def prepare():
    from p3audit.constants import PHYSIONET_CHANNELS as CH
    ICA_DIR.mkdir(parents=True, exist_ok=True)
    for p in sorted(Path("results/real_cache").glob("sub-*.npz")):
        q = ICA_DIR / p.name
        if q.exists():
            continue
        X, meta = clean_subject(p, CH)
        tmp = q.with_suffix(".tmp.npz"); np.savez(tmp, X=X, meta=json.dumps(meta)); tmp.replace(q)   # atomic
        print(p.stem, "removed", meta["n_removed"], "eye components", flush=True)


def cleaned(D):
    """Cleaned epochs in the row order of a stacked dict from real_audit._stack (keys subj, trial)."""
    cache, X = {}, np.empty_like(D["X"])
    for i, (s, k) in enumerate(zip(D["subj"], D["trial"])):
        s = int(s)
        if s not in cache:
            cache[s] = np.load(ICA_DIR / f"sub-{s:03d}.npz", allow_pickle=True)["X"]
        X[i] = cache[s][int(k)]
    return X


def transform(arm, TR, ES, TE, ch, sfreq):
    assert arm == "ica_eye"
    return cleaned(TR), cleaned(ES), cleaned(TE), {"cleaning": "ICA (extended infomax, 30 components), ICLabel eye class p >= 0.5 removed, per subject"}


def train(a):
    import amendment2_controls as A2
    A2.transform = transform
    from p3audit.config import load_config
    from p3audit.data.splits import real_audit_folds
    from p3audit.experiments.context import Context
    from p3audit.experiments.real_audit import _stack
    from p3audit.training.trainer import resolve_device
    cfg = load_config(a.config); ctx = Context.create(cfg)
    sp, ch = ctx.split, list(ctx.ch_names)
    included = sorted(set(sp.train) | set(sp.val) | set(sp.test))
    folds = real_audit_folds(included, sp.pilot, cfg["split"]["real_audit_folds"], cfg["seeds"]["split"])
    device = resolve_device(cfg["training"]["device"])
    out = Path(a.out) if a.out else ctx.results / "real_r7"; out.mkdir(parents=True, exist_ok=True)
    for f in folds:
        if a.folds is not None and f["fold"] not in a.folds:
            continue
        nonpilot = [s for s in f["train"] if s not in sp.pilot]
        es_s = nonpilot[: max(2, len(nonpilot) // 10)]; tr_s = [s for s in f["train"] if s not in es_s]
        TR, ES, TE = _stack(ctx, tr_s), _stack(ctx, es_s), _stack(ctx, f["test"])
        for mdl in a.models:
            for sd in a.seeds:
                p = out / f"ica_eye__{mdl}__fold{f['fold']}__seed{sd}.json"
                if p.exists():
                    continue
                rec, trials = A2.run_arm(cfg, ch, TR, ES, TE, "ica_eye", mdl, sd, f["fold"], device, a.max_epochs)
                trials.to_csv(p.with_name(p.stem + "_trials.csv"), index=False); p.write_text(json.dumps(rec, indent=1))
                print(f"ica_eye {mdl:8s} fold {f['fold']} seed {sd}: BA {rec['ba']:.3f} ({rec['seconds'] / 60:.1f} min)", flush=True)


def analyse():
    import pandas as pd
    from p3audit.constants import PHYSIONET_CHANNELS as CH
    i7, i8 = CH.index("AF7"), CH.index("AF8"); key = ["model", "subject", "trial", "type", "label"]
    load = lambda pat: pd.concat([pd.read_csv(f) for f in glob.glob(pat)])
    main = load("results/real/*_trials.csv"); main = main[main.arm == "main"]
    ols = load("results/real_r3/*_trials.csv"); ols = ols[ols.arm == "heog_ols"]
    ica = load("results/real_r7/*_trials.csv"); models = sorted(ica.model.unique())
    prep = lambda df, n: df[df.model.isin(models)].assign(correct=lambda x: x.correct.astype(float)).groupby(key, as_index=False).correct.mean().rename(columns={"correct": n})
    w = prep(main, "main").merge(prep(ols, "heog_ols"), on=key).merge(prep(ica, "ica_eye"), on=key)
    defl, defl_ica, nrem = {}, {}, []
    for s in sorted(w.subject.unique()):
        z = np.load(f"results/real_cache/sub-{s:03d}.npz", allow_pickle=True); c = np.load(ICA_DIR / f"sub-{s:03d}.npz", allow_pickle=True)
        nrem.append(json.loads(str(c["meta"]))["n_removed"]); p = int(z["pad"]); sg = np.where(z["y"] == 1, 1.0, -1.0)
        for store, X in ((defl, z["X"]), (defl_ica, c["X"])):
            h = X[:, i7, :] - X[:, i8, :]; v = (h[:, p + 60:p + 240].mean(1) - h[:, p - 40:p].mean(1)) * sg
            for k in range(len(v)): store[(s, k)] = v[k]
    w["d"] = [defl[(s, k)] for s, k in zip(w.subject, w.trial)]; w["d_ica"] = [defl_ica[(s, k)] for s, k in zip(w.subject, w.trial)]
    if w.loc[w.type == "congruent", "d"].mean() < 0:
        w["d"] *= -1; w["d_ica"] *= -1
    w["dz"] = w.groupby(["model", "subject"]).d.transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
    n = w.type == "none"; w.loc[n, "tert"] = w[n].groupby("model").dz.transform(lambda x: pd.qcut(x, 3, labels=False))

    def stats(x):
        nn = x[x.type == "none"]; t0, t2 = nn[nn.tert == 0], nn[nn.tert == 2]
        G = lambda a: (t2.main - t2[a]).mean() - (t0.main - t0[a]).mean()
        return {"acc_main": x.main.mean(), "acc_ica_eye": x.ica_eye.mean(), "acc_heog_ols": x.heog_ols.mean(),
                "gap_none_ica": (nn.main - nn.ica_eye).mean(), "gap_none_ols": (nn.main - nn.heog_ols).mean(),
                "gap_congruent_ica": (x[x.type == "congruent"].main - x[x.type == "congruent"].ica_eye).mean(),
                "G_ica_eye": G("ica_eye"), "G_heog_ols": G("heog_ols"), "G_heog_ols-G_ica_eye": G("heog_ols") - G("ica_eye"),
                "deflection_none_uV": nn.d.mean(), "deflection_none_after_ica_uV": nn.d_ica.mean(),
                "deflection_congruent_uV": x[x.type == "congruent"].d.mean(), "deflection_congruent_after_ica_uV": x[x.type == "congruent"].d_ica.mean()}

    rng = np.random.default_rng(20261011); subs = np.array(sorted(w.subject.unique())); g = {s: x for s, x in w.groupby("subject")}
    est = stats(w); B = pd.DataFrame([stats(pd.concat([g[s] for s in rng.choice(subs, subs.size)])) for _ in range(2000)])
    out = {"models": models, "n_subjects": int(subs.size), "eye_components_removed": {"mean": float(np.mean(nrem)), "min": int(min(nrem)), "max": int(max(nrem)), "subjects_with_none": int(sum(x == 0 for x in nrem))},
           "estimates": {k: {"est": float(v), "ci_95": [float(B[k].quantile(.025)), float(B[k].quantile(.975))]} for k, v in est.items()}}
    json.dump(out, open("results/analysis/posthoc_revision7_ica.json", "w"), indent=1)
    print(out["eye_components_removed"])
    for k, v in out["estimates"].items(): print(f"{k:36s} {v['est']:.3f} [{v['ci_95'][0]:.3f}, {v['ci_95'][1]:.3f}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true"); ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--models", nargs="+", default=["csoanet", "eegnet", "shallow", "cbramod"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2]); ap.add_argument("--folds", nargs="+", type=int, default=None)
    ap.add_argument("--config", default="configs/default.yaml"); ap.add_argument("--max-epochs", type=int, default=None); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    prepare() if a.prepare else analyse() if a.analyse else train(a)
