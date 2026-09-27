"""Registered sensitivity analysis (PILOT_LOG 17.25): H3 and H3r without the supra-physiological
1.5 a* alpha cells.  Uses the frozen analysis functions unchanged; only the unit list is filtered.
Run from the repository root:  python scripts/h3_without_stress.py"""
import json, sys, types, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p3audit.config import load_config
from p3audit.experiments import analysis as A
warnings.filterwarnings("ignore")
cfg = load_config("configs/default.yaml")
out = Path("results/analysis_sensitivity"); (out / "analysis").mkdir(parents=True, exist_ok=True)
ctx = types.SimpleNamespace(cfg=cfg, results=out)
units = [json.load(open(p, encoding="utf-8")) for p in sorted(Path("results/units").glob("*.json"))]
keep = [u for u in units if not (u["cell"]["confound"] == "alpha" and float(u["cell"]["amp"]) == 1.5)]
h3 = A.analyse_h3(ctx, keep); h3r = A.analyse_h3r(ctx, keep)
json.dump({k: v for k, v in h3.items()}, open("results/analysis/h3_without_1p5_full.json", "w"), indent=1, default=float)
json.dump(h3r, open("results/analysis/h3r_without_1p5.json", "w"), indent=1)
for k, v in h3.items():
    if k.startswith("alpha"):
        c = v["corr_cross_fitted"]; print(k, round(c["estimate"], 3), round(c["ci_low"], 3), v["decision"])
