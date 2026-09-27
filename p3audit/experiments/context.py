"""Project context: config, data source, split, access guard and artefact paths.

Artefacts written under paths.results:
  exclusion.csv, split.json                       (3.1, 3.2)
  components/sub-XXX.json                         (3.4.2, 3.4.3 per subject)
  saccade_pool.npz, saccade_pool.json             (3.4.4, P6)
  pilots/*.json                                   (P1a–P12 reports)
  units/<uid>.json                                (one per model × cell × seed)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..data.exclusion import exclusion_table, included_subjects
from ..data.sources import PhysioNetSource, RunSource, SyntheticSource
from ..data.splits import Split, make_split
from ..generator.cache import SubjectCache, open_cache
from ..generator.components import Component, required_components
from ..utils.access import SignalAccessGuard
from ..utils.common import get_logger, write_json

log = get_logger("p3audit")


@dataclass
class Context:
    cfg: Config
    source: RunSource
    results: Path
    cache_root: Path
    split: Split | None = None
    guard: SignalAccessGuard | None = None
    _caches: dict = field(default_factory=dict)

    # -------------------------------------------------------------- construction
    @staticmethod
    def create(cfg: Config, synthetic: bool = False, n_synthetic: int = 70) -> "Context":
        results = cfg.resolve_path("results")
        cache = cfg.resolve_path("cache")
        results.mkdir(parents=True, exist_ok=True); cache.mkdir(parents=True, exist_ok=True)
        if synthetic:
            src: RunSource = SyntheticSource(n_subjects=n_synthetic)
        else:
            src = PhysioNetSource(cfg.resolve_path("data_root"), cfg["physionet"]["n_subjects"])
        ctx = Context(cfg, src, results, cache)
        sp = results / "split.json"
        if sp.exists():
            with open(sp, encoding="utf-8") as f:
                ctx.split = Split(**json.load(f))
        ctx.guard = SignalAccessGuard(cfg.resolve_path("access_log"),
                                      set(ctx.split.pilot) if ctx.split else set())
        return ctx

    # -------------------------------------------------------------- 3.1 / 3.2
    def run_exclusion_and_split(self) -> tuple[pd.DataFrame, Split]:
        tab = exclusion_table(self.source, self.cfg)
        tab.to_csv(self.results / "exclusion.csv", index=False)
        inc = included_subjects(tab)
        self.split = make_split(inc, self.cfg)
        write_json(self.results / "split.json", self.split.as_dict())
        self.guard.pilot = set(self.split.pilot)
        log.info(f"included {len(inc)} / {len(tab)}; pilot {len(self.split.pilot)}, "
                 f"train {len(self.split.train)}, val {len(self.split.val)}, test {len(self.split.test)}")
        return tab, self.split

    def require_split(self) -> Split:
        if self.split is None:
            raise RuntimeError(
                f"no split.json under {self.results}. Run `p3audit prepare --stage split` first with the same "
                "--config / --synthetic options (a dry run with --config configs/dryrun.yaml --synthetic 14 "
                "writes to ./dryrun/results, the real run to ./results).")
        return self.split

    @property
    def all_subjects(self) -> list[int]:
        s = self.require_split()
        return sorted(set(s.train) | set(s.val) | set(s.test))

    # -------------------------------------------------------------- components
    def comp_path(self, subject: int) -> Path:
        return self.results / "components" / f"sub-{subject:03d}.json"

    def load_components(self, subject: int) -> dict[str, Component] | None:
        p = self.comp_path(subject)
        if not p.exists():
            return None
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        return {k: Component.from_dict(v) for k, v in d["components"].items()}

    def subjects_with_valid_components(self) -> list[int]:
        """Subjects the generator can use: alpha plus the task component.

        Under hemisphere: single the task component is the selected side ('mu'); the per-side
        mu_L / mu_R records are kept for the audit trail and are not required to pass."""
        need = required_components(self.cfg)
        out = []
        for s in self.all_subjects:
            c = self.load_components(s)
            if c and all(k in c and c[k].passed for k in need):
                out.append(s)
        return out

    # -------------------------------------------------------------- saccade pool
    def load_pool(self) -> dict[int, np.ndarray]:
        d = np.load(self.results / "saccade_pool.npz")
        return {int(k.split("_")[1]): d[k] for k in d.files}

    # -------------------------------------------------------------- caches
    def cache(self, subject: int) -> SubjectCache:
        if subject not in self._caches:
            self._caches[subject] = open_cache(self.cache_root, subject)
        return self._caches[subject]

    def caches(self, subjects) -> dict[int, SubjectCache]:
        return {s: self.cache(s) for s in subjects}

    def has_cache(self, subject: int) -> bool:
        return (self.cache_root / f"sub-{subject:03d}" / "meta.json").exists()

    def usable(self, subjects) -> list[int]:
        """Subjects that have a cache (i.e. passed component inclusion and were built)."""
        return [s for s in subjects if self.has_cache(s)]

    @property
    def ch_names(self) -> list[str]:
        from ..constants import PHYSIONET_CHANNELS
        return list(PHYSIONET_CHANNELS)
