"""Sanity check on synthetic data: does each metric move with the ground truth?

Trains EEGNet on the saccade confound at p = 0.5 and p = 1.0 (weak task, s = 0.3) and
prints ΔBA_neu, ΔBA_rev, HEOG-regression ΔBA_reg (saccade-locked and OLS), IG share
and residual-erasure ΔBA_keep.  Expected pattern: truth ≈ 0 → ≈ p − 0.5; erasure
tracks truth for p < 1 and collapses to 0 at p = 1 (non-identifiable); IG share rises.
Requires the dry-run caches:  p3audit --config configs/dryrun.yaml --synthetic 14 prepare
"""
import warnings; warnings.filterwarnings("ignore")
from p3audit.config import load_config
from p3audit.experiments.context import Context
from p3audit.experiments.unit import run_unit
from p3audit.generator.design import Cell, Unit
cfg = load_config("configs/dryrun.yaml", {"training.patience": 4, "ig.windows_per_unit": 24, "ig.steps": 8, "ig.p_levels": [0.5, 0.75, 0.9, 1.0], "ig.zero_baseline": False})
ctx = Context.create(cfg, synthetic=True, n_synthetic=14)
for p in (0.5, 1.0):
    r = run_unit(ctx, Unit("eegnet", Cell("saccade", p, 0.3, 1.0, "sanity"), 0), max_epochs=12, out_dir="sanity",
                 metrics=("truth", "heog", "ig", "erasure"))
    print(p, "BA_m %.3f dNeu %.3f dRev %.3f | dReg %.3f (ols %.3f) | IG share %.3f | keep %.3f" % (
        r["truth"]["ba_matched"], r["truth"]["delta_ba_neu"], r["truth"]["delta_ba_rev"],
        r["heog"]["0"]["delta_ba_reg"], r["heog"]["ols"]["delta_ba_reg"], r["ig"]["band_removed"]["share"], r["erasure"]["delta_ba_keep"]), flush=True)
