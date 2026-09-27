"""Flat per-unit summary for manuscript tables/figures (PILOT_LOG 17.25). Read-only on results/units."""
import glob, json, csv, os
rows = []
for f in sorted(glob.glob("results/units/*.json")):
    d = json.load(open(f, encoding="utf-8")); c = d["cell"]; t = d["truth"]
    r = {"uid": d["uid"], "model": d["model"], "seed": d["seed"], "cid": d["cid"], "confound": c["confound"],
         "p": c["p"], "s": c["s"], "amp": c["amp"], "module": c["module"], "alpha_basis": c.get("alpha_basis"),
         "sacc_basis": c.get("sacc_basis"), "sacc_basis_test": c.get("sacc_basis_test"),
         "best_val_ba": d["training"].get("best_val_ba"), "epochs": d["training"].get("epochs"), "seconds": d["seconds"]}
    r.update({k: t[k] for k in t})
    for a, pr in d["probe"].items():
        r[f"D_{a}"] = pr["D"]
        if "within_subject" in pr: r[f"Dws_{a}"] = pr["within_subject"]["D"]
        for dim, v in (pr.get("pca") or {}).items(): r[f"Dpca{dim}_{a}"] = v
    e = d.get("erasure")
    if e:
        r.update({"er_ba": e["ba"], "er_ba_erased": e["ba_erased"], "er_keep": e["delta_ba_keep"],
                  "er_keep_ws": e["within_subject"].get("delta_ba_keep"), "er_ba_retrain": e["ba_retrain"],
                  "er_angle": e["angle_deg"]})
    s = d.get("sri")
    if s: r.update({k: s[k] for k in s if k != "counts"})
    for k, v in (d.get("heog") or {}).items(): r[f"heog_{k}"] = v["delta_ba_reg"]
    for bl, g in (d.get("ig") or {}).items():
        r[f"ig_{bl}_share"] = g["share"]; r[f"ig_{bl}_norm_abs"] = g["norm_abs"]; r[f"ig_{bl}_signed"] = g["signed_target"]
    rows.append(r)
keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in rows[0], k))
os.makedirs("results/analysis", exist_ok=True)
with open("results/analysis/units_summary.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
print(len(rows), len(keys))
