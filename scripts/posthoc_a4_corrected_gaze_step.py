"""Amendment 4 registered analysis with one correction (post hoc).

The registered step statistic (scripts/amendment4_gf2025.py, step_stat) uses nancumsum on nan_to_num(x), so invalid
gaze samples inside the +-100 ms windows enter the window means as 0 deg (the fixation position). This script reruns
the registered analysis unchanged except that the step statistic ignores invalid samples (window means over valid
samples only). The registered script and its output (results/analysis/amendment4.json) are not modified.
Output: results/analysis/posthoc_a4_corrected_gaze_step.json
"""
import sys, types, pathlib, numpy as np
sys.path.insert(0, "scripts")
import amendment4_gf2025 as a

def step_stat_valid(x, n):
    v = np.isfinite(x); x0 = np.where(v, x, 0.0)
    cs = np.concatenate([np.zeros(x.shape[:-1] + (1,)), np.cumsum(x0, -1)], -1)
    cn = np.concatenate([np.zeros(x.shape[:-1] + (1,)), np.cumsum(v, -1)], -1)
    s = np.full(x.shape, np.nan); T = x.shape[-1]; t = np.arange(n, T - n + 1)
    with np.errstate(all="ignore"):
        nf = cn[..., t + n] - cn[..., t]; nb = cn[..., t] - cn[..., t - n]
        f = (cs[..., t + n] - cs[..., t]) / nf; b = (cs[..., t] - cs[..., t - n]) / nb
        s[..., t] = np.where((nf > 0) & (nb > 0), f - b, np.nan)
    return s

OUTP = "results/analysis/posthoc_a4_corrected_gaze_step.json"
_P = pathlib.Path
def P(x, *r):
    return _P(OUTP) if str(x).replace("\\", "/").endswith("results/analysis/amendment4.json") else _P(x, *r)
a.Path = P
# F8 - F7 (HORD) contains no NaN, so the corrected statistic is identical there; gaze is the only signal affected.
a.step_stat = step_stat_valid
a.analyse(types.SimpleNamespace(out=None))
print("written", OUTP)
