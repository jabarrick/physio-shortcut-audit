import warnings; warnings.filterwarnings('ignore')
"""Independent recomputation of key Paper B numbers from per-trial record files.

Written from scratch; does NOT import p3audit, the amendment*/posthoc_* scripts, or
read results/analysis/*.json.  Inputs read:
  results/real/*_trials.csv            PhysioNet main + step-vector control ('control')
  results/real_r3/heog_ols__*_trials.csv  PhysioNet OLS control (AF7-AF8 regressed out)
  results/real_r7/ica_eye__*_trials.csv   PhysioNet ICA control
  results/real_cache/sub-*.npz         PhysioNet model-input epochs (X: trials x 64 x 1200, 200 Hz,
                                       t=0 at sample 200) -> AF7-AF8 deflection
  results/gf2025/*_trials.csv          eye-tracking replication, 6 arms
  results/gf2025_cache/sub-*.npz       HOR (F8-F7 of model input), GAZE (deg), GVALID, y, session
  results/gf2025_cache/sessions.json   per-session gaze_available flag
Output: results/analysis/recompute_independent_paperB.json

Definitions (as given by the paper / Amendments 1 and 4):
  * correctness of each trial is averaged over seeds 0-2 per (arm, model) first.
  * balanced accuracy (BA) per record = mean of class recalls, record = (arm, model, fold) after
    seed averaging; pooled = mean over models and folds.
  * subset accuracy = mean seed-averaged correctness over trials; pooled over models = mean of
    model-level trial values (all models see the same trials, so this is the mean over model-trials).
  * gap = main - control (paired, same trials).
  * deflection = cue-signed (positive towards cued side) mean over 0.3-1.2 s minus mean over
    -0.2-0 s, z-scored within subject (over the subject's trials where it is defined).
      PhysioNet: AF7-AF8 (channels 24, 28 of the EEGMMIDB order) of the model-input stream;
                 rightward gaze -> AF7-AF8 negative, so sign = +1 for left cue (y=0), -1 for right (y=1).
      GF2025: d_HOR on HOR (F8-F7, positive rightward); d_gaze on GAZE (deg, positive rightward);
              sign = +1 for right cue (y=1), -1 for left (y=0). (Both checked empirically.)
  * terciles: pooled over subjects, within the subset (PhysioNet: type=='none' from the record CSV;
    GF: gaze-valid trials with no gaze-detected saccade).  Tercile assignment is fixed on the full
    sample (not recomputed within bootstrap resamples).
    G = gap(top tercile) - gap(bottom tercile).
  * GF gaze validity: trial valid if >= 75% of samples in -0.2..1.2 s are GVALID, and its session
    has gaze_available == True.
  * GF gaze saccade classification (0-1 s): s(t) = mean gaze over next 100 ms - mean over previous
    100 ms; threshold per session = max(5 * MAD(s), 0.5 deg), MAD = median |s - median(s)| (unscaled)
    over all samples t in -0.2..1.2 s with valid gaze, all trials of the session.  Supra-threshold
    -> congruent if the largest |s| in 0-1 s points to the cued side, else incongruent; otherwise none.
  * H4 contrast: per subject with >= 10 congruent and >= 10 no-saccade trials:
      [acc_main(cong) - acc_main(none)] - [acc_ctrl(cong) - acc_ctrl(none)];
    point estimate = mean over eligible subjects (primary); a trial-pooled version is also given.
  * intervals: subject cluster bootstrap, 2000 resamples, percentile; 98.75% for A4-R1..R4,
    95% otherwise.  Seed 12345 (any seed).
"""
import glob, json, os, re
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
R = os.path.join(ROOT, 'results')
FS, PAD = 200, 200
NB = 2000
RNG = np.random.default_rng(12345)
MODELS = ['csoanet', 'eegnet', 'shallow', 'cbramod']


def idx(t):  # time (s) -> sample index
    return PAD + int(round(t * FS))


# ------------------------------------------------------------------ loading
def load_trials(pattern):
    fs = sorted(glob.glob(os.path.join(R, pattern)))
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    return df, len(fs)


def seed_avg(df, keys):
    g = df.groupby(['arm', 'model', 'fold'] + keys, as_index=False)
    out = g.agg(correct=('correct', 'mean'), nseed=('seed', 'nunique'), label=('label', 'first'))
    return out


def ba_records(sa):
    rec = sa.groupby(['arm', 'model', 'fold', 'label']).correct.mean().unstack('label').mean(1)
    return rec  # index arm, model, fold


# ------------------------------------------------------------------ bootstrap helpers
def boot_weights(nsub):
    W = np.zeros((NB, nsub))
    for b in range(NB):
        W[b] = np.bincount(RNG.integers(0, nsub, nsub), minlength=nsub)
    return W


def pct(x, level):
    a = (1 - level) / 2
    return [float(np.nanpercentile(x, 100 * a)), float(np.nanpercentile(x, 100 * (1 - a)))]


def ratio_stat(S, N, W):
    """S, N: per-subject sums and counts (nsub,) -> point and bootstrap of sum(S)/sum(N)."""
    pt = S.sum() / N.sum()
    bs = (W @ S) / (W @ N)
    return pt, bs


def subset_gap(tbl, mask, W, subj_index, level):
    """tbl: DataFrame with subject, gap (model-averaged per trial).  Mean gap over masked trials."""
    t = tbl[mask]
    S = np.bincount(subj_index[t.subject].values, weights=t.gap.values, minlength=W.shape[1])
    N = np.bincount(subj_index[t.subject].values, minlength=W.shape[1]).astype(float)
    pt, bs = ratio_stat(S, N, W)
    return {'est': float(pt), 'ci': pct(bs, level), 'n_trials': int(mask.sum())}


def tercile_G(tbl, mask, dcol, W, subj_index, level):
    t = tbl[mask].copy()
    t['terc'] = pd.qcut(t[dcol], 3, labels=False)
    ns = W.shape[1]
    out = {}
    stats = []
    for k in range(3):
        u = t[t.terc == k]
        S = np.bincount(subj_index[u.subject].values, weights=u.gap.values, minlength=ns)
        N = np.bincount(subj_index[u.subject].values, minlength=ns).astype(float)
        pt, bs = ratio_stat(S, N, W)
        stats.append((pt, bs))
        out[f'gap_tercile{k+1}'] = float(pt)
        out[f'n_tercile{k+1}'] = int(len(u))
    G = stats[2][0] - stats[0][0]
    Gb = stats[2][1] - stats[0][1]
    out.update({'G': float(G), 'ci': pct(Gb, level), 'n_trials': int(len(t))})
    return out


def h4_contrast(tbl, W, subj_list, level, ccol='c_ctrl', mcol='c_main', kmin=10):
    """tbl: per-trial rows with subject, type, c_main, c_ctrl (model-averaged or one model)."""
    per = []
    for s in subj_list:
        u = tbl[tbl.subject == s]
        c, n = u[u.type == 'congruent'], u[u.type == 'none']
        if len(c) >= kmin and len(n) >= kmin:
            per.append(((c[mcol].mean() - n[mcol].mean()) - (c[ccol].mean() - n[ccol].mean()),
                        s, len(c), len(n)))
    if not per:
        return {'n_eligible': 0}
    v = np.array([p[0] for p in per])
    ne = len(v)
    bs = np.array([v[RNG.integers(0, ne, ne)].mean() for _ in range(NB)])
    # trial-pooled version in the same eligible subjects
    el = [p[1] for p in per]
    u = tbl[tbl.subject.isin(el)]
    def acc(col, ty):
        return u[u.type == ty][col].mean()
    pooled = (acc(mcol, 'congruent') - acc(mcol, 'none')) - (acc(ccol, 'congruent') - acc(ccol, 'none'))
    return {'est_subject_mean': float(v.mean()), 'ci': pct(bs, level), 'n_eligible': ne,
            'est_trial_pooled': float(pooled)}


# ================================================================== PhysioNet
def physionet():
    out = {}
    real, nf = load_trials('real/*_trials.csv')
    ols, nf2 = load_trials('real_r3/heog_ols__*_trials.csv')
    ica, nf3 = load_trials('real_r7/ica_eye__*_trials.csv')
    out['n_record_files'] = {'real(main+control)': nf, 'real_r3/heog_ols': nf2, 'real_r7/ica_eye': nf3}
    allt = pd.concat([real, ols, ica], ignore_index=True)
    keys = ['subject', 'trial']
    sa = allt.groupby(['arm', 'model', 'fold'] + keys, as_index=False).agg(
        correct=('correct', 'mean'), nseed=('seed', 'nunique'), label=('label', 'first'),
        type=('type', 'first'))
    assert (sa.nseed == 3).all()
    # ---- pooled BA of main models
    rec = ba_records(sa)
    ba = {}
    for arm in ['main', 'control', 'heog_ols', 'ica_eye']:
        r = rec.loc[arm]
        ba[arm] = {'pooled_BA': float(r.mean()),
                   'per_model_BA': {m: float(r.loc[m].mean()) for m in MODELS}}
    out['balanced_accuracy_seedavg_per_model_fold'] = ba
    # per-seed-record BA (alternative definition, for reference)
    recs = allt.groupby(['arm', 'model', 'fold', 'seed', 'label']).correct.mean().unstack('label').mean(1)
    out['main_pooled_BA_per_seed_record'] = float(recs.loc['main'].mean())

    # ---- deflection AF7-AF8 from model-input cache
    rows = []
    for f in sorted(glob.glob(os.path.join(R, 'real_cache', 'sub-*.npz'))):
        z = np.load(f)
        s = int(re.findall(r'sub-(\d+)', f)[0])
        X, y = z['X'], z['y']
        h = X[:, 24, :] - X[:, 28, :]
        d = h[:, idx(0.3):idx(1.2)].mean(1) - h[:, idx(-0.2):idx(0)].mean(1)
        d = d * np.where(y == 0, 1.0, -1.0)  # positive towards cued side
        dz = (d - d.mean()) / d.std(ddof=1)
        for i in range(len(y)):
            rows.append((s, i, int(y[i]), str(z['type'][i]), float(d[i]), float(dz[i])))
    dfl = pd.DataFrame(rows, columns=['subject', 'trial', 'y_cache', 'type_cache', 'd_raw', 'd'])

    wide = sa.pivot_table(index=keys + ['model'], columns='arm', values='correct').reset_index()
    meta = sa[sa.arm == 'main'][keys + ['label', 'type']].drop_duplicates(keys)
    wide = wide.merge(meta, on=keys).merge(dfl, on=keys, how='left')
    chk = wide.drop_duplicates(keys)
    out['cache_check'] = {'trials_matched': int(chk.d.notna().sum()), 'trials_total': int(len(chk)),
                          'label_agree': float((chk.y_cache == chk.label).mean()),
                          'type_agree': float((chk.type_cache == chk.type).mean())}
    subs = np.sort(wide.subject.unique())
    sidx = pd.Series(np.arange(len(subs)), index=subs)
    W = boot_weights(len(subs))
    out['n_subjects'] = int(len(subs))
    out['n_trials'] = int(len(chk))
    out['type_counts'] = chk.type.value_counts().to_dict()

    ctrls = {'step_vector(control, results/real)': 'control',
             'OLS(heog_ols, results/real_r3)': 'heog_ols',
             'ICA(ica_eye, results/real_r7)': 'ica_eye'}
    gaps = {}
    for lab, c in ctrls.items():
        gaps[lab] = {}
        for m in MODELS + ['pooled']:
            t = wide if m == 'pooled' else wide[wide.model == m]
            t = t.assign(gap=t['main'] - t[c])
            if m == 'pooled':
                t = t.groupby(keys, as_index=False).agg(gap=('gap', 'mean'), type=('type', 'first'),
                                                        d=('d', 'first'))
            gaps[lab][m] = subset_gap(t, (t.type == 'none').values, W, sidx, 0.95)
    out['nosaccade_gap_main_minus_control'] = gaps

    # ---- G(main vs OLS), pooled over models, 95%
    t = wide.assign(gap=wide['main'] - wide['heog_ols']).groupby(keys, as_index=False).agg(
        gap=('gap', 'mean'), type=('type', 'first'), d=('d', 'first'))
    out['G_main_vs_heog_ols_pooled'] = tercile_G(t, (t.type == 'none').values, 'd', W, sidx, 0.95)
    Gm = {}
    for m in MODELS:
        u = wide[wide.model == m].assign(gap=lambda x: x['main'] - x['heog_ols'])
        Gm[m] = tercile_G(u, (u.type == 'none').values, 'd', W, sidx, 0.95)
    out['G_main_vs_heog_ols_per_model'] = Gm
    # extra: G vs step-vector and ICA controls, pooled
    for lab, c in [('control', 'control'), ('ica_eye', 'ica_eye')]:
        u = wide.assign(gap=wide['main'] - wide[c]).groupby(keys, as_index=False).agg(
            gap=('gap', 'mean'), type=('type', 'first'), d=('d', 'first'))
        out[f'G_main_vs_{lab}_pooled_extra'] = tercile_G(u, (u.type == 'none').values, 'd', W, sidx, 0.95)

    # ---- H4 per model, 95%
    h4 = {}
    for lab, c in ctrls.items():
        h4[lab] = {}
        for m in MODELS:
            u = wide[wide.model == m].rename(columns={'main': 'c_main', c: 'c_ctrl'})
            h4[lab][m] = h4_contrast(u, W, subs, 0.95)
    out['H4_contrast_per_model'] = h4
    return out


# ================================================================== GF2025
def gaze_classify(G, V, y, ses_ok, sessions):
    n = G.shape[0]
    w = int(0.1 * FS)
    Gf = np.where(V, G, np.nan)  # use only valid samples
    ts = np.arange(idx(-0.2), idx(1.2) + 1)
    S = np.full((n, len(ts)), np.nan)
    for j, t in enumerate(ts):
        a = Gf[:, t:t + w]; b = Gf[:, t - w:t]
        with np.errstate(all='ignore'):
            S[:, j] = np.nanmean(a, 1) - np.nanmean(b, 1)
        S[~V[:, t], j] = np.nan
    vf = V[:, idx(-0.2):idx(1.2)].mean(1)
    valid = (vf >= 0.75) & ses_ok
    typ = np.array(['invalid'] * n, dtype=object)
    thr_s = {}
    for se in np.unique(sessions):
        m = sessions == se
        v = S[m][np.isfinite(S[m])]
        if len(v) == 0:
            continue
        thr = max(5 * np.median(np.abs(v - np.median(v))), 0.5)
        thr_s[int(se)] = float(thr)
        j0, j1 = idx(0) - idx(-0.2), idx(1.0) - idx(-0.2)
        for i in np.where(m & valid)[0]:
            s = S[i, j0:j1 + 1]
            if not np.isfinite(s).any() or np.nanmax(np.abs(s)) <= thr:
                typ[i] = 'none'
            else:
                k = np.nanargmax(np.abs(s))
                cued = 1 if y[i] == 1 else -1
                typ[i] = 'congruent' if np.sign(s[k]) == cued else 'incongruent'
    return typ, valid, thr_s


def gf2025():
    out = {}
    df, nf = load_trials('gf2025/*_trials.csv')
    out['n_record_files'] = nf
    out['records_per_arm'] = (df[['arm', 'model', 'fold', 'seed']].drop_duplicates().groupby('arm').size().to_dict())  # noqa
    keys = ['subject', 'trial']
    sa = df.groupby(['arm', 'model', 'fold'] + keys, as_index=False).agg(
        correct=('correct', 'mean'), nseed=('seed', 'nunique'), label=('label', 'first'))
    assert (sa.nseed == 3).all()
    rec = ba_records(sa)
    arms = sorted(sa.arm.unique())
    out['accuracy_per_arm_all_trials'] = {
        a: {'pooled_BA': float(rec.loc[a].mean()),
            'pooled_mean_correct': float(sa[sa.arm == a].correct.mean()),
            'per_model_BA': {m: float(rec.loc[a].loc[m].mean()) for m in MODELS}} for a in arms}

    sess = json.load(open(os.path.join(R, 'gf2025_cache', 'sessions.json')))
    rows = []
    for f in sorted(glob.glob(os.path.join(R, 'gf2025_cache', 'sub-*.npz'))):
        s = int(re.findall(r'sub-(\d+)', f)[0])
        z = np.load(f)
        y, ses = z['y'], z['session']
        keys_s = sorted(k for k in sess if k.startswith(f'S{s:02d}/'))
        avail = np.array([sess[keys_s[k]]['gaze_available'] for k in ses])
        typ, valid, _ = gaze_classify(z['GAZE'], z['GVALID'], y, avail, ses)
        sign = np.where(y == 1, 1.0, -1.0)
        H = z['HOR']
        dh = (H[:, idx(0.3):idx(1.2)].mean(1) - H[:, idx(-0.2):idx(0)].mean(1)) * sign
        Gm = np.where(z['GVALID'], z['GAZE'], np.nan)
        with np.errstate(all='ignore'):
            dg = (np.nanmean(Gm[:, idx(0.3):idx(1.2)], 1) - np.nanmean(Gm[:, idx(-0.2):idx(0)], 1)) * sign
        dg[~valid] = np.nan
        dhz = (dh - dh.mean()) / dh.std(ddof=1)
        if np.isfinite(dg).sum() >= 2:
            dgz = (dg - np.nanmean(dg)) / np.nanstd(dg, ddof=1)
        else:
            dgz = np.full_like(dg, np.nan)
        for i in range(len(y)):
            rows.append((s, i, int(y[i]), typ[i], bool(valid[i]), dh[i], dhz[i], dg[i], dgz[i]))
    tr = pd.DataFrame(rows, columns=['subject', 'trial', 'y', 'gtype', 'gvalid', 'dH_raw', 'dH',
                                     'dG_raw', 'dG'])
    out['n_trials'] = int(len(tr)); out['n_gaze_valid'] = int(tr.gvalid.sum())
    out['gaze_valid_counts_by_type'] = tr[tr.gvalid].gtype.value_counts().to_dict()
    out['corr_dG_dH_in_gaze_valid'] = float(tr[tr.gvalid][['dG', 'dH']].corr().iloc[0, 1])

    wide = sa.pivot_table(index=keys + ['model'], columns='arm', values='correct').reset_index()
    wide = wide.merge(tr, on=keys, how='left')
    lab = sa[sa.arm == 'main'][keys + ['label']].drop_duplicates(keys)
    chk = wide.drop_duplicates(keys).merge(lab, on=keys)
    out['label_check_agree'] = float((chk.y == chk.label).mean())

    subs = np.sort(wide.subject.unique())
    sidx = pd.Series(np.arange(len(subs)), index=subs)
    W = boot_weights(len(subs))
    L = 0.9875

    def pooled(ctrl):
        return wide.assign(gap=wide['main'] - wide[ctrl]).groupby(keys, as_index=False).agg(
            gap=('gap', 'mean'), c_main=('main', 'mean'), c_ctrl=(ctrl, 'mean'),
            gtype=('gtype', 'first'), gvalid=('gvalid', 'first'), dH=('dH', 'first'), dG=('dG', 'first'))

    p = pooled('hor_ols')
    nos = (p.gvalid & (p.gtype == 'none')).values
    # A4-R1
    u = p[p.gvalid].rename(columns={'gtype': 'type'})
    out['A4_R1_bias_corrected_contrast'] = h4_contrast(u, W, subs, L)
    # A4-R2
    out['A4_R2_nosaccade_gap_main_minus_hor_ols'] = subset_gap(p, nos, W, sidx, L)
    # A4-R3 / R4
    out['A4_R3_G_dHOR'] = tercile_G(p, nos, 'dH', W, sidx, L)
    out['A4_R4_G_dgaze'] = tercile_G(p, nos, 'dG', W, sidx, L)
    # per model descriptives for R2-R4
    pm = {}
    for m in MODELS:
        q = wide[wide.model == m].assign(gap=lambda x: x['main'] - x['hor_ols'])
        nm = (q.gvalid & (q.gtype == 'none')).values
        pm[m] = {'R2_gap': subset_gap(q, nm, W, sidx, L)['est'],
                 'R3_G': tercile_G(q, nm, 'dH', W, sidx, L)['G'],
                 'R4_G': tercile_G(q, nm, 'dG', W, sidx, L)['G']}
    out['per_model_hor_ols'] = pm
    # accuracy by gaze type, main and hor_ols (models pooled; mean seed-averaged correctness)
    abt = {}
    for arm in ['main', 'hor_ols']:
        abt[arm] = {}
        for ty in ['congruent', 'incongruent', 'none']:
            m = wide.gvalid & (wide.gtype == ty)
            sub = wide[m]
            ba_ty = sub.groupby('y')[arm].mean().mean()
            abt[arm][ty] = {'mean_correct': float(sub[arm].mean()), 'BA_pooled_trials': float(ba_ty),
                            'n_trials': int(m.sum() / 4)}
    out['accuracy_by_gaze_type'] = abt
    # descriptive: other controls no-saccade gap and G (98.75% for uniformity)
    desc = {}
    for c in ['veo_ols', 'plc_shuf', 'frontal_loc', 'central_loc']:
        q = pooled(c)
        desc[c] = {'nosaccade_gap': subset_gap(q, nos, W, sidx, L),
                   'G_dHOR': tercile_G(q, nos, 'dH', W, sidx, L)['G'],
                   'G_dgaze': tercile_G(q, nos, 'dG', W, sidx, L)['G']}
    out['descriptive_other_arms_main_minus_arm'] = desc
    return out


if __name__ == '__main__':
    import sys
    # usage: python recompute_independent_paperB.py [gf|pn|all]; parts are merged into the output json
    part = sys.argv[1] if len(sys.argv) > 1 else 'all'
    op = os.path.join(R, 'analysis', 'recompute_independent_paperB.json')
    res = {}
    if os.path.exists(op):
        try:
            res = json.load(open(op))
        except Exception:
            res = {}
    res.update({'script': 'scripts/recompute_independent_paperB.py', 'bootstrap': {'n': NB, 'seed': 12345}})
    if part in ('gf', 'all'):
        res['gf2025'] = gf2025()
    if part in ('pn', 'all'):
        res['physionet'] = physionet()
    conv = lambda o: o.item() if hasattr(o, 'item') else str(o)
    with open(op, 'w') as fh:
        json.dump(res, fh, indent=1, default=conv)
    print(json.dumps(res.get('gf2025' if part == 'gf' else 'physionet', res), indent=1, default=conv))
