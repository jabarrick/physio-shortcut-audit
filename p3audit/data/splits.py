"""3.2 subject partition.

pilot (15, seeded)  -> train only, never test/val
test (34), val (10) -> drawn from the non-pilot remainder
train               -> everything else (+ pilot)
Calibration and final training use the *same* split.  Real audit: subject-
grouped 5-fold CV with pilot subjects fixed in the training folds.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class Split:
    pilot: list[int]
    train: list[int]
    val: list[int]
    test: list[int]
    seed_pilot: int
    seed_split: int

    def as_dict(self):
        return asdict(self)

    def role(self, subject: int) -> str:
        if subject in self.test:
            return "test"
        if subject in self.val:
            return "val"
        return "train"


def make_split(included: list[int], cfg) -> Split:
    sp, sd = cfg["split"], cfg["seeds"]
    included = sorted(included)
    n = len(included)
    if n < sp["n_pilot"] + sp["n_test"] + sp["n_val"] + 1:
        raise ValueError(f"only {n} subjects after exclusion")
    r1 = np.random.default_rng(sd["pilot_draw"])
    pilot = sorted(r1.choice(included, size=sp["n_pilot"], replace=False).tolist())
    rest = [s for s in included if s not in pilot]
    r2 = np.random.default_rng(sd["split"])
    perm = r2.permutation(rest).tolist()
    test = sorted(perm[: sp["n_test"]])
    val = sorted(perm[sp["n_test"]: sp["n_test"] + sp["n_val"]])
    train = sorted(set(included) - set(test) - set(val))
    assert set(pilot) <= set(train)
    return Split(pilot, train, val, test, sd["pilot_draw"], sd["split"])


def real_audit_folds(included: list[int], pilot: list[int], n_folds: int, seed: int) -> list[dict]:
    """Subject-grouped folds; pilot subjects are always in training."""
    rest = [s for s in sorted(included) if s not in set(pilot)]
    perm = np.random.default_rng(seed).permutation(rest)
    chunks = np.array_split(perm, n_folds)
    folds = []
    for k, test in enumerate(chunks):
        test = sorted(int(s) for s in test)
        train = sorted(set(included) - set(test))
        folds.append({"fold": k, "train": train, "test": test})
    return folds
