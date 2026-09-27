"""Configuration loading, dotted access, TBD registry and config hashing."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


class Config(dict):
    """dict with dotted get/set:  cfg.get_path('alpha_component.band')."""

    def get_path(self, dotted: str, default: Any = KeyError) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                if default is KeyError:
                    raise KeyError(dotted)
                return default
        return node

    def set_path(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node = self
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def resolve_path(self, key: str) -> Path:
        return Path(self["paths"][key]).expanduser()

    def copy(self) -> "Config":  # type: ignore[override]
        return Config(copy.deepcopy(dict(self)))

    def hash(self) -> str:
        blob = json.dumps(self, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    # ---- design scalars: resolve 's_low' / 's_star' / 's_high' names
    def task_strength(self, name_or_value, confound: str | None = None) -> float:
        """'s_star' / 's_low' / 's_high' -> value.  Per-confound values under
        task_component.by_confound.<confound> take precedence (PILOT_LOG 11.16)."""
        if isinstance(name_or_value, str):
            tc = self["task_component"]
            per = (tc.get("by_confound") or {}).get(confound or "", {}) or {}
            if name_or_value in per:
                v = per[name_or_value]
                if v is None:
                    raise ValueError(f"task_component.by_confound.{confound}.{name_or_value} not calibrated yet")
                return float(v)
            return float(tc[name_or_value])
        return float(name_or_value)


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> Config:
    with open(DEFAULT_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if path is not None and Path(path).resolve() != DEFAULT_CONFIG:
        with open(path, encoding="utf-8") as f:
            _deep_update(cfg, yaml.safe_load(f) or {})
    if overrides:
        c = Config(cfg)
        for k, v in overrides.items():
            c.set_path(k, v)
        return c
    return Config(cfg)


# ----------------------------------------------------------------- TBD registry
def tbd_status(cfg: Config) -> list[dict]:
    confirmed = set(cfg.get("confirmed", []) or [])
    rows = []
    for item in cfg.get("tbd_registry", []):
        rows.append({
            "key": item["key"],
            "source": item["source"],
            "value": cfg.get_path(item["key"], None),
            "confirmed": item["key"] in confirmed,
        })
    return rows


def assert_confirmatory_ready(cfg: Config) -> None:
    """Confirmatory (preregistered) runs require every TBD to be confirmed."""
    open_items = [r for r in tbd_status(cfg) if not r["confirmed"]]
    if open_items:
        keys = ", ".join(f"{r['key']} ({r['source']})" for r in open_items)
        raise RuntimeError(f"Confirmatory run blocked; unconfirmed TBD values: {keys}")
