"""Recompute the code and configuration hashes recorded in the preregistration.

The OSF registration (DOI 10.17605/OSF.IO/8Q7N6) fixed the analysis code by
code_sha256 = SHA-256 over the bytes of every p3audit/**/*.py file, in sorted
path order (identical to `p3audit freeze`, p3audit/cli.py).  The only change to
p3audit/ after registration is deviation D-2026-09-26 (deviations/), which
touched real_audit.py and shu_audit.py.  This script checks both states:

  frozen   : current tree with the two files replaced by deviations/*.frozen
  current  : the tree as checked out

and the configuration hash of configs/default.yaml (p3audit.config.Config.hash).

Run from the repository root:  python scripts/verify_code_hash.py
Needs only the standard library (plus PyYAML for the config hash).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "p3audit"

EXPECTED = {
    "frozen": "4e1ffab94988e93b1742059b57f1684da828dec6b0aea952c06aacf869226e21",
    "current": "0d594c152acb40f6c5f5919f1281f6b5a2a79b89ff9228bff686696d06eb785a",
    "config": "d3d4d36229fe8662",
}
FROZEN_SUBSTITUTES = {
    "experiments/real_audit.py": ROOT / "deviations" / "real_audit.py.frozen",
    "experiments/shu_audit.py": ROOT / "deviations" / "shu_audit.py.frozen",
}


def code_hash(substitutes: dict[str, Path] | None = None) -> str:
    substitutes = substitutes or {}
    h = hashlib.sha256()
    # Sort on the POSIX relative path so the order is the same on every OS
    # (it matches the order `p3audit freeze` produced when the hash was registered).
    files = sorted(PKG.rglob("*.py"), key=lambda p: p.relative_to(PKG).as_posix())
    for p in files:
        rel = p.relative_to(PKG).as_posix()
        h.update(Path(substitutes.get(rel, p)).read_bytes())
    return h.hexdigest()


def config_hash() -> str | None:
    try:
        import yaml
    except ImportError:
        return None
    with open(ROOT / "configs" / "default.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]


def main() -> int:
    got = {"frozen": code_hash(FROZEN_SUBSTITUTES), "current": code_hash(), "config": config_hash()}
    ok = True
    for k, exp in EXPECTED.items():
        if got[k] is None:
            print(f"{k:8s} skipped (PyYAML not installed)")
            continue
        match = got[k] == exp
        ok &= match
        print(f"{k:8s} {'OK      ' if match else 'MISMATCH'} {got[k]}")
    if not ok:
        print("\nA mismatch usually means line endings were converted on checkout "
              "(see .gitattributes) or a file under p3audit/ or configs/ was edited.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
