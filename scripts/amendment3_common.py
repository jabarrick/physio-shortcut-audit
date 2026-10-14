"""Amendment 3 (in preparation; not yet registered): shared pieces of the real-saccade units.  No torch import.

* A3Cache: a frozen SubjectCache whose load() also serves the new bases from cache_a3/ (scripts/amendment3_build_bases.py).
* compose_a3: wraps the frozen p3audit.generator.semisynth.compose.  After the frozen call it
    - for the three ground-truth test sets (key[1] == 'test'): re-assigns z with the frozen assign_z within
      subject x label x half strata (half = the frozen half_assignment of the window ids), so that each half has the
      intended coupling exactly; the saccade coefficient is recomputed from the new z;
    - for every injected set: adds the sham basis with coefficient (1 - z) x amplitude, so that z = 0 windows receive a
      real background segment of the same size as the one that comes with an event in z = 1 windows.  The sham
      coefficient carries a gain (results/a3/sham_gain_<ARM>.json, scripts/amendment3_check_balance.py --calibrate)
      fixed on training subjects so that the injected power before the event onset is equal in z = 1 and z = 0.
  Nothing else of the frozen composition changes.
"""
from pathlib import Path
import numpy as np

A3_ROOT = Path("cache_a3")


def make_cache_class():
    from p3audit.generator.cache import SubjectCache

    class A3Cache(SubjectCache):
        def load(self, name, mmap=True):
            p = A3_ROOT / self.root.name / f"{name}.npy"
            if name.startswith(("sacc_RH", "sacc_RV", "sham_RH", "sham_RV")):
                return np.load(p, mmap_mode="r" if mmap else None)
            return super().load(name, mmap)
    return A3Cache


def sham_gain(arm):
    import json
    p = Path("results/a3") / f"sham_gain_{arm}.json"
    return float(json.load(open(p))["gain"]) if p.exists() else 1.0


def install(ctx, arm, patch_unit=True, gain=None):
    """Patch the frozen modules in memory (files on disk unchanged) and pre-load A3 caches into ctx."""
    from p3audit.data.windows import half_assignment
    from p3audit.generator import semisynth
    from p3audit.generator.cache import open_cache
    from p3audit.generator.design import assign_z
    A3Cache = make_cache_class(); cfg = ctx.cfg
    for d in sorted(Path(ctx.cache_root).glob("sub-*")):
        if (d / "meta.json").exists():
            s = int(d.name[4:]); c = open_cache(ctx.cache_root, s); ctx._caches[s] = A3Cache(c.subject, c.root, c.meta)
    orig = semisynth.compose; sham = f"sham_{arm}"; g = sham_gain(arm) if gain is None else gain; amp_uv0 = cfg["saccade"]["amplitude_uv"]; seed_half = cfg["seeds"]["half_split"]

    def compose_a3(caches, subjects, cell, cfg_, q, key, amp=None, y_override=None, role="train"):
        cs = orig(caches, subjects, cell, cfg_, q, key, amp=amp, y_override=y_override, role=role)
        if q is None:
            return cs
        assert cell.confound == "saccade" and cfg_["saccade"]["mode"] == "presence"
        basis = cell.sacc_basis_test if (role == "test" and cell.sacc_basis_test) else cell.sacc_basis
        a = cell.amp if amp is None else amp; amp_uv = a * amp_uv0
        if role == "test" and len(key) > 1 and key[1] == "test":
            half = half_assignment(cs.wids, seed_half)
            cs.z = assign_z(cs.y, cs.subj * 2 + half, q, tuple(key) + ("a3_half_strata",))
            cs.coef[basis] = cs.z * amp_uv
        cs.coef[sham] = (1 - cs.z) * amp_uv * g
        cs.info["a3"] = {"sham": sham, "sham_gain": g, "half_stratified": role == "test" and len(key) > 1 and key[1] == "test"}
        return cs

    if patch_unit:                                   # the unit module imports torch
        import p3audit.experiments.unit as U
        U.compose = compose_a3
    return compose_a3
