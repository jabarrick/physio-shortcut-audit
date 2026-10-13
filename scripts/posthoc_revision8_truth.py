"""Post hoc (exploratory, not registered; 3 Oct 2026): within-cell results under alternative ground-truth contrasts and
with the coupling imbalance of the window-parity halves removed.

Background.  assign_z (p3audit/generator/design.py) fixes the fraction of z = 1 windows within every (subject, label)
stratum over ALL test windows; the registered halves are formed afterwards by window parity, so the realised coupling of
the two halves deviates from the nominal value in opposite directions.  Unit records hold per-subject confusion counts
per half but no per-window predictions, so the halves cannot be re-drawn.  They can be corrected: the labels and
subjects of the test windows are rebuilt from the cache metadata (and checked against the stored class counts, half
counts and the exact stratum sums of z), which gives the realised congruent fraction f = mean(z == y) of every
(subject, label, half, test set) stratum.

Ground-truth variants (per unit, for all windows and for each half):
  neu     BA(matched) - BA(neutral)                registered primary (as posthoc_revision5_bamatched.py)
  rev     BA(matched) - BA(reversed)               registered secondary
  nr      BA(neutral) - BA(reversed)               shares no test set with BA(matched)
  neu_fe  (p - 0.5) x slope of recall on f, matched + neutral sets, subject x label strata as fixed effects
  all_fe  the same slope over all three test sets
With exact couplings neu_fe equals the stratum-weighted form of neu; the slope uses the realised f of each half instead.

For the units of posthoc_revision5_bamatched.py (original p = 0.9 factorial + p-series cells; Amendment 1 saccade units)
and each variant: split-half r of the within-cell residuals, cross-fitted within-cell partial r of BA(matched), the
probe score D (at the cell amplitude and at 0.125 and 0.25 a*) and every metric, metric - D (the statistics of A1-H1
and A2-S1), metric - BA(matched), and the partial r after controlling for BA(matched); cluster bootstrap over
cell x model groups (4000 resamples, seed and resamples as in posthoc_revision5_bamatched.py, so the `neu` rows
reproduce that file).  Writes results/analysis/posthoc_revision8_truth.json and, when the cache is present,
results/analysis/posthoc_revision8_test_labels.json (labels of the test windows, so the analysis can be rerun without
the cache).
"""
import json, sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore"); sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from p3audit.experiments import analysis as A
from posthoc_revision5_bamatched import N_BOOT, SEED


def corr(a, b):
    """Pearson r over the finite pairs; same value as posthoc_revision5_bamatched.corr, without np.corrcoef (speed)."""
    k = np.isfinite(a) & np.isfinite(b)
    if k.sum() <= 3:
        return np.nan
    a = a[k] - a[k].mean(); b = b[k] - b[k].mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def partial_out(x, c):
    """Residual of x on [1, c] over the finite pairs; same value as posthoc_revision5_bamatched.partial_out."""
    k = np.isfinite(x) & np.isfinite(c); out = np.full_like(x, np.nan)
    cc = c[k] - c[k].mean(); xc = x[k] - x[k].mean(); out[k] = xc - cc * (cc @ xc / (cc @ cc))
    return out

LABELS = Path("results/analysis/posthoc_revision8_test_labels.json")
SETS = {"matched": None, "neutral": 0.5, "reversed": None}          # nominal q; matched = p, reversed = 1 - p
TRUTHS = ("neu", "rev", "nr", "neu_fe", "all_fe")
PARTS = {"all": None, "half0": 0, "half1": 1}
PROBE_AMPS = ("0.125", "0.25"); PROBES = ("D",) + tuple("D" + a for a in PROBE_AMPS)


def test_labels(subjects):
    """Labels of the cached windows of each test subject (cache/sub-XXX/meta.json), stored for reruns without the cache."""
    have = json.load(open(LABELS, encoding="utf-8")) if LABELS.exists() else {}
    need = [s for s in subjects if str(s) not in have]
    for s in need:
        have[str(s)] = [int(v) for v in json.load(open(Path("cache") / f"sub-{s:03d}" / "meta.json", encoding="utf-8"))["y"]]
    if need:
        LABELS.parent.mkdir(parents=True, exist_ok=True)
        json.dump({k: have[k] for k in sorted(have, key=int)}, open(LABELS, "w", encoding="utf-8"))
    return {int(k): np.asarray(v, int) for k, v in have.items()}


def strata(rec, labels):
    """Per (set, part, subject, label): windows n, correct, congruent fraction f.  Verifies the rebuilt labels."""
    subs = [int(k) for k in rec["truth_counts"]["matched"]["all"]]
    y = np.concatenate([labels[s] for s in subs]); sj = np.concatenate([np.full(labels[s].size, s) for s in subs])
    half = np.asarray(rec["test_half"]); p = float(rec["cell"]["p"])
    assert y.size == half.size, rec["uid"]
    out = {}
    for k in SETS:
        z = np.asarray(rec["z_test"][k]); q = {"matched": p, "neutral": 0.5, "reversed": 1 - p}[k]
        for part, h in PARTS.items():
            rows = []
            for s in subs:
                c = rec["truth_counts"][k][part].get(str(s), [0, 0, 0, 0])
                for yy in (0, 1):
                    m = (sj == s) & (y == yy) & ((half == h) if h is not None else True); n = int(m.sum())
                    n_rec, ok = (c[0] + c[1], c[0]) if yy == 0 else (c[2] + c[3], c[3])
                    assert n == n_rec, (rec["uid"], k, part, s, yy, n, n_rec)          # class and half counts match
                    if part == "all":                                                    # exact stratum sums of z
                        assert int(z[m].sum()) == int(round((q if yy == 1 else 1 - q) * n)), (rec["uid"], k, s, yy)
                    rows.append((s, yy, n, ok, float((z[m] == yy).mean()) if n else np.nan))
            out[k, part] = np.array(rows, float)
    return out


def fe_slope(st, part, sets):
    """Slope of recall on the realised congruent fraction, strata as fixed effects, per label, then averaged; and the
    recall at f = 0.5 implied by the fit (corrected neutral accuracy)."""
    b, a = [], []
    for yy in (0, 1):
        T = [st[k, part][st[k, part][:, 1] == yy] for k in sets]
        n = T[0][:, 2]; keep = n > 0; n = n[keep]
        r = np.stack([t[keep, 3] / n for t in T]); f = np.stack([t[keep, 4] for t in T])
        rc, fc = r - r.mean(0), f - f.mean(0)
        slope = float((n * rc * fc).sum() / (n * fc * fc).sum()); b.append(slope)
        a.append(float((n * (r.mean(0) - slope * (f.mean(0) - 0.5))).sum() / n.sum()))
    return float(np.mean(b)), float(np.mean(a))


def unit_values(rec, labels, subs):
    st = strata(rec, labels); p = float(rec["cell"]["p"]); v = {}
    for part in PARTS:
        M, N, R = (A._ba(rec["truth_counts"][k][part], subs) for k in SETS)
        b2, _ = fe_slope(st, part, ("matched", "neutral")); b3, s3 = fe_slope(st, part, ("matched", "neutral", "reversed"))
        v[part] = {"M": M, "N": N, "R": R, "neu": M - N, "rev": M - R, "nr": N - R, "neu_fe": (p - 0.5) * b2, "all_fe": (p - 0.5) * b3,
                   "slope": b3, "skill": s3, "f_dev": {k: float(np.average(st[k, part][:, 4], weights=st[k, part][:, 2]) -
                                                                np.average(st[k, "all"][:, 4], weights=st[k, "all"][:, 2])) for k in SETS}}
    return v


def build(units, conf, metrics, labels):
    subs = A._test_subjects(units)
    keys = sorted({(r["model"], r["cid"]) for r in units}); gi = {k: i for i, k in enumerate(keys)}
    g = np.array([gi[(r["model"], r["cid"])] for r in units]); U = [unit_values(r, labels, subs) for r in units]
    X = {}
    for h in ("0", "1"):
        for t in TRUTHS:
            X[t + h] = [u["half" + h][t] for u in U]
        X["bam" + h] = [u["half" + h]["M"] for u in U]
        for n in metrics:
            X[n + h] = [A.DEPENDENCE_METRICS[conf][n](r, subs, "half" + h) for r in units]
    X["D"] = [r["probe"][f"{float(r['cell']['amp']):g}"]["D"] for r in units]
    for a in PROBE_AMPS:                                             # probe measured below its ceiling (A2-S1)
        X["D" + a] = [r["probe"].get(a, {}).get("D", np.nan) for r in units]
    return {k: np.array(v, float) for k, v in X.items()}, [np.where(g == i)[0] for i in range(len(keys))], U


def stat(X, groups, metrics):
    idx = np.concatenate(groups); lab = np.concatenate([np.full(len(m), j) for j, m in enumerate(groups)])
    def res(v):
        v = v[idx]; ok = np.isfinite(v)
        s = np.bincount(lab[ok], v[ok], minlength=len(groups)); n = np.bincount(lab[ok], minlength=len(groups))
        return v - (s / np.maximum(n, 1))[lab]
    R = {k: res(v) for k, v in X.items()}; o = {}
    for t in TRUTHS:
        t0, t1 = R[t + "0"], R[t + "1"]
        xf = lambda a: np.nanmean([corr(t0, R[a + "1"]), corr(t1, R[a + "0"])])
        o[t, "rel_truth"] = corr(t0, t1); o[t, "BA_matched"] = xf("bam")
        for d in PROBES:
            o[t, d] = np.nanmean([corr(t0, R[d]), corr(t1, R[d])]); o[t, "BA_matched-" + d] = o[t, "BA_matched"] - o[t, d]
        for n in metrics:
            o[t, n] = xf(n); o[t, n + "-BA_matched"] = o[t, n] - o[t, "BA_matched"]
            for d in PROBES:                                         # the registered A1-H1 and A2-S1 statistics
                o[t, n + "-" + d] = o[t, n] - o[t, d]
            o[t, n + "|BAm"] = np.nanmean([corr(partial_out(t0, R["bam1"]), partial_out(R[n + "1"], R["bam0"])),
                                           corr(partial_out(t1, R["bam0"]), partial_out(R[n + "0"], R["bam1"]))])
    for n in ["bam"] + list(metrics):
        o["rel", "BA_matched" if n == "bam" else n] = corr(R[n + "0"], R[n + "1"])
    return o


def run(units, conf, metrics, labels):
    X, groups, _ = build(units, conf, metrics, labels); est = stat(X, groups, metrics)
    rng = np.random.default_rng(SEED); B = [stat(X, [groups[k] for k in rng.integers(0, len(groups), len(groups))], metrics) for _ in range(N_BOOT)]
    def ci(key, a=.05):
        v = np.array([b[key] for b in B], float); v = v[np.isfinite(v)]; return [float(np.quantile(v, a / 2)), float(np.quantile(v, 1 - a / 2))]
    out = {"n_units": len(units), "n_groups": len(groups), "reliability_resid_half_r": {k[1]: {"est": float(est[k]), "ci95": ci(k)} for k in est if k[0] == "rel"}}
    for t in TRUTHS:
        out[t] = {k[1]: {"est": float(est[k]), "ci95": ci(k), **({"ci98.33": ci(k, .05 / 3)} if "-" in k[1] else {})} for k in est if k[0] == t}
    return out


def unit_sets():
    orig = [json.load(open(p, encoding="utf-8")) for p in sorted(Path("results/units").glob("*.json"))]
    a1 = [json.load(open(p, encoding="utf-8")) for p in sorted(Path("results/units_a1").glob("*.json"))]
    S = {f"original_{c}": ([r for r in orig if r["cell"]["confound"] == c and r["cell"]["p"] == 0.9 and r["cell"]["module"] in ("factorial", "p_series")],
                           c, A.CONFIRMATORY[c]) for c in ("alpha", "saccade")}
    S["amendment1_saccade"] = ([r for r in a1 if r["cell"]["confound"] == "saccade" and r["cell"]["p"] == 0.9], "saccade", ["heog_reg", "ig_share", "erasure_keep"])
    return S


if __name__ == "__main__":
    S = unit_sets(); labels = test_labels(sorted({int(k) for us, _, _ in S.values() for r in us for k in r["truth_counts"]["matched"]["all"]}))
    res = {name: run(us, conf, mets, labels) for name, (us, conf, mets) in S.items()}
    Path("results/analysis").mkdir(parents=True, exist_ok=True)
    json.dump(res, open("results/analysis/posthoc_revision8_truth.json", "w"), indent=1)
    f = lambda d: f"{d['est']:.2f} [{d['ci95'][0]:.2f}, {d['ci95'][1]:.2f}]"
    for name, v in res.items():
        print(f"\n{name}: {v['n_units']} units, {v['n_groups']} groups")
        for t in TRUTHS:
            print(f"  {t:7s} split-half r {f(v[t]['rel_truth'])} | BA(matched) {f(v[t]['BA_matched'])} | D {f(v[t]['D'])}")
            for n in [k for k in v[t] if k.endswith("-BA_matched")]:
                m = n[:-11]; print(f"          {m:13s} {f(v[t][m])}   minus BA(matched) {f(v[t][n])}   | BA(matched) {f(v[t][m + '|BAm'])}")
