"""3.4.5 experimental design: cells, units, coupling assignment.

A *cell* fixes the generator (confound, p, s, confound amplitude, bases);
a *unit* is (model, cell, seed) — the aggregation unit of H1–H3 (3.4.6/3.8).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np

from ..utils.common import rng as keyed_rng


@dataclass(frozen=True)
class Cell:
    confound: str                 # 'alpha' | 'saccade'
    p: float | None               # None = no injection
    s: str | float = "s_star"     # task strength name or value
    amp: float = 1.0              # confound amplitude in units of a*
    module: str = "p_series"      # p_series | factorial | corner | mismatch | group_template
    alpha_basis: str = "alpha"    # alpha | alpha_m2 | alpha_p2 | alpha_group
    task_suffix: str = ""         # '' | '_group'
    sacc_basis: str = "sacc_R"    # training/validation saccade basis
    sacc_basis_test: str = ""     # test-time basis; '' = same as training ('sacc_R_mis' for the mismatch cell)

    @property
    def cid(self) -> str:
        p = "none" if self.p is None else f"{self.p:g}"
        s = self.s if isinstance(self.s, str) else f"{self.s:g}"
        tag = f"{self.confound}_p{p}_{s}_a{self.amp:g}"
        extra = [x for x in (self.alpha_basis if self.alpha_basis != "alpha" else "",
                             "group" if self.task_suffix else "",
                             "sacmis" if self.sacc_basis_test not in ("", self.sacc_basis) else "") if x]
        return tag + ("_" + "_".join(extra) if extra else "")

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class Unit:
    model: str
    cell: Cell
    seed: int

    @property
    def uid(self) -> str:
        return f"{self.model}__{self.cell.cid}__seed{self.seed}"


def enumerate_units(cfg) -> list[Unit]:
    d = cfg["design"]
    core, supp = list(d["core_models"]), list(d["supplementary_models"])
    units: list[Unit] = []
    seen = set()

    def add(model, cell, n_seeds):
        for sd in range(n_seeds):
            u = Unit(model, cell, sd)
            if u.uid not in seen:
                seen.add(u.uid); units.append(u)

    sup = d.get("supplementary", {})
    seed_over = {float(k): v for k, v in (d.get("p_series_seed_overrides") or {}).items()}
    sup_fact = {(str(a), float(b)) for a, b in sup.get("factorial_cells", [])}
    for conf in ("alpha", "saccade"):
        sup_on = conf in sup.get("confounds", [])
        for p in d["p_series"]:
            c = Cell(conf, p, "s_star", 1.0, "p_series")
            n_seeds = seed_over.get(p, d["seeds_p_series"]) if p is not None else d["seeds_p_series"]
            for m in core:
                add(m, c, n_seeds)
            for m in supp if sup_on else []:
                add(m, c, min(n_seeds, sup.get("p_series_seeds", n_seeds)))
            for m in d.get("reference_models", []) or []:
                add(m, c, 1)
        for s in d["factorial_s"]:
            for a in d["factorial_amp"]:
                c = Cell(conf, d["factorial_p"], s, a, "factorial")
                for m in core:
                    add(m, c, d["seeds_other"])
                if sup_on and (s, float(a)) in sup_fact:
                    for m in supp:
                        add(m, c, sup.get("factorial_seeds", d["seeds_other"]))
        for s in d["corner_s"]:
            for a in d["corner_amp"]:
                c = Cell(conf, d["corner_p"], s, a, "corner")
                for m in core:
                    add(m, c, d["seeds_other"])
        if conf == "alpha":
            for basis in ("alpha_m2", "alpha_p2"):
                c = Cell(conf, d["factorial_p"], "s_star", 1.0, "mismatch", alpha_basis=basis)
                for m in core:
                    add(m, c, d["seeds_other"])
            c = Cell(conf, d["factorial_p"], "s_star", 1.0, "group_template", alpha_basis="alpha_group",
                     task_suffix="_group")
            for m in core:
                add(m, c, d["seeds_other"])
        else:
            c = Cell(conf, d["factorial_p"], "s_star", 1.0, "mismatch", sacc_basis_test="sacc_R_mis")
            for m in core:
                add(m, c, d["seeds_other"])
    return units


def unit_counts(units: list[Unit]) -> dict:
    out: dict = {}
    for u in units:
        k = u.cell.module
        out.setdefault(k, {"total": 0})
        out[k]["total"] += 1
        out[k][u.model] = out[k].get(u.model, 0) + 1
    return out


# ------------------------------------------------------------------ coupling
def assign_z(y: np.ndarray, subjects: np.ndarray, q: float, key: tuple) -> np.ndarray:
    """Exact stratified coupling: within every (subject, y) stratum a fraction q of y=1
    windows and 1−q of y=0 windows get z=1  =>  P(z=1|y=1)=q, P(z=1|y=0)=1−q."""
    z = np.zeros_like(y)
    for s in np.unique(subjects):
        for yy in (0, 1):
            idx = np.where((subjects == s) & (y == yy))[0]
            if idx.size == 0:
                continue
            frac = q if yy == 1 else 1 - q
            k = int(round(frac * idx.size))
            chosen = keyed_rng(*key, int(s), yy).permutation(idx)[:k]
            z[chosen] = 1
    return z


def test_couplings(p: float | None) -> dict[str, float | None]:
    if p is None:
        return {"matched": None}
    return {"matched": p, "neutral": 0.5, "reversed": 1 - p}


def with_amp(cell: Cell, amp: float) -> Cell:
    return replace(cell, amp=amp)
