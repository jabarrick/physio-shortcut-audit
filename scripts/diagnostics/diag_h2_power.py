"""H2 TOST power (PILOT_LOG 15.17).  Written 2026-09-23, BEFORE running.  numpy only; reads stored
P8/P8b/P8c unit files (validation subjects, already evaluated) - no new data access.

H2 per model: TOST on per-subject D(s_high) - D(s_low), each subject's D averaged over the
factorial seeds (design.seeds_other = 2), n = test subjects (20), margin +/- stats.h2_delta (0.05),
alpha 0.05 (analysis.analyse_h2 / inference.tost_paired).

Noise model per subject: diff_s = mu + i_s + (e_hi - e_lo), i_s = subject x cell interaction
(var s2_int), e = seed noise of the per-subject D (var s2_seed / n_seeds per cell).
Estimated per model from the only pair of cells that differ in s at equal training amplitude:
saccade s = 1.333 (P8+P8b, seeds 0-3) and s = 1.6 (P8c, seeds 0-3), probe D at test amp 1.
  s2_seed : pooled within-subject variance across the 4 seeds, both cells
  var_diff4 : across-subject variance of [mean_4seeds(1.6) - mean_4seeds(1.333)]
  s2_int  : max(0, var_diff4 - 2 * s2_seed / 4)
Power by simulation (20000 draws) at mu in {0, 0.01, 0.02, 0.03}, n_seeds in {2, 3, 4}.

Pre-declared readings (Claude, authorised by Yu Gao):
  P1  power(mu = 0, 2 seeds) >= 0.8 for every model  -> H2 adequately powered; nothing changes.
  P2  otherwise: the smallest n_seeds in {3, 4} reaching 0.8 for all models is adopted for the
      H2 cells ONLY (factorial s_low / s_high at the selected amplitude) and the cost reported;
      if 4 is not enough, H2 is preregistered as underpowered, as is.  delta is NOT changed (15.14).
Caveats (report): n = 7 validation subjects; the s-gap here (1.333 vs 1.6) is smaller than
s_low vs s_high, so s2_int is likely UNDER-estimated; D near ceiling (~0.9) for saccade.
"""
import json, glob
import numpy as np

T_CRIT = 1.7291328115213678   # t_{0.95, 19}
N_TEST = 20
DELTA = 0.05

def per_subject_D(f):
    d = json.load(open(f, encoding="utf-8"))
    out = {}
    for s, c in d["probe"]["1"]["counts"].items():
        c = np.asarray(c, float)
        r0 = c[0] / (c[0] + c[1]) if c[0] + c[1] else np.nan
        r1 = c[3] / (c[2] + c[3]) if c[2] + c[3] else np.nan
        out[s] = (np.nanmean([r0, r1]) - 0.5) / 0.5
    return out

res = {}
rng = np.random.default_rng(0)
for model in ("csoanet", "cbramod"):
    cells = {}
    for tag, pat in (("s1.333", f"results/pilots/p8_units/{model}__saccade_p0.9_s_star_a1__seed*.json"),
                     ("s1.6", f"results/pilots/p8_units/{model}__saccade_p0.9_1.6_a1__seed*.json")):
        files = sorted(glob.glob(pat))
        assert len(files) == 4, (model, tag, files)
        Ds = [per_subject_D(f) for f in files]
        subs = sorted(set.intersection(*[set(x) for x in Ds]))
        cells[tag] = np.array([[x[s] for s in subs] for x in Ds])   # seeds x subjects
    A, B = cells["s1.333"], cells["s1.6"]
    s2_seed = float(np.mean([A.var(axis=0, ddof=1).mean(), B.var(axis=0, ddof=1).mean()]))
    diff4 = B.mean(0) - A.mean(0)
    var_diff4 = float(diff4.var(ddof=1))
    s2_int = max(0.0, var_diff4 - 2 * s2_seed / 4)
    pw = {}
    for k in (2, 3, 4):
        sd = np.sqrt(s2_int + 2 * s2_seed / k)
        for mu in (0.0, 0.01, 0.02, 0.03):
            x = rng.normal(mu, sd, (20000, N_TEST))
            m, se = x.mean(1), x.std(1, ddof=1) / np.sqrt(N_TEST)
            ok = ((m + DELTA) / se > T_CRIT) & ((m - DELTA) / se < -T_CRIT)
            pw[f"seeds{k}_mu{mu:g}"] = float(ok.mean())
    res[model] = {"n_subjects": len(subs), "s2_seed": s2_seed, "var_diff4": var_diff4, "s2_int": s2_int,
                  "sd_diff_2seeds": float(np.sqrt(s2_int + s2_seed)), "mean_diff_1.6_minus_1.333": float(diff4.mean()),
                  "power": pw}
ok2 = all(r["power"]["seeds2_mu0"] >= 0.8 for r in res.values())
need = next((k for k in (3, 4) if all(r["power"][f"seeds{k}_mu0"] >= 0.8 for r in res.values())), None)
res["reading"] = "P1" if ok2 else ("P2: adopt %d seeds for H2 cells" % need if need else "P2: underpowered even at 4 seeds")
json.dump(res, open("results/pilots/h2_power_diag.json", "w"), indent=1)
print(json.dumps(res, indent=1))
