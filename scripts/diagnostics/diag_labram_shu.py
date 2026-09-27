"""EXPLORATORY diagnostic (PILOT_LOG 17.23): why LaBraM predicted a constant class on SHU-MI in
all 5 folds of the confirmatory shu-audit.  Re-trains fold 0, seed 0 exactly as run_shu_audit does
(same subjects, split, early-stopping subjects, recipe) but keeps the per-epoch history.
Writes ONLY results/diag/labram_shu_diag.json; never touches results/shu (the confirmatory record
stands as recorded).  Imports frozen/deviated code unchanged.

Reading rule, written BEFORE running:
  R1 optimisation failure  : train loss never falls below 0.68 (~ln 2)
  R2 generalisation / early-stopping failure : train loss falls below 0.60 but val BA never > 0.55
  R3 early-stopping-set artefact : val BA > 0.55 at some epoch but the restored model is constant on test
  R4 none of the above (report as is)
"""
import json, sys, time
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root (script moved to scripts/diagnostics/)
from p3audit.config import load_config
from p3audit.experiments.context import Context
from p3audit.experiments.shu_audit import shu_subject
from p3audit.metrics.intervention import crop
from p3audit.models.registry import build_model
from p3audit.training.trainer import predict_logits, resolve_device, train_model
from p3audit.utils.common import balanced_accuracy

cfg = load_config()
ctx = Context.create(cfg)
device = resolve_device(cfg["training"]["device"])
assert str(device).startswith("cuda"), f"GPU required, got {device}"
t0 = time.time()
data = {s: shu_subject(ctx, s) for s in range(1, 26)}
ch = next(iter(data.values()))[4]; pad, L = next(iter(data.values()))[2], next(iter(data.values()))[3]
subs = np.random.default_rng(cfg["seeds"]["split"]).permutation(list(data))
test = np.array_split(subs, 5)[0]
train = [s for s in subs if s not in set(test)]
es, tr = train[:2], train[2:]
cat = lambda ss: (np.concatenate([data[s][0] for s in ss]), np.concatenate([data[s][1] for s in ss]))
Xtr, ytr = cat(tr); Xes, yes = cat(es); Xte, yte = cat(test)
ctr = crop(Xtr, pad, L)
scale = cfg["foundation"]["labram"]["input_scale"]
info = {"fold": 0, "seed": 0, "test_subjects": [int(s) for s in test], "es_subjects": [int(s) for s in es],
        "n_train": int(len(ytr)), "n_es": int(len(yes)), "n_test": int(len(yte)),
        "label_share_1": {"train": float(ytr.mean()), "es": float(yes.mean()), "test": float(yte.mean())},
        "L_samples": int(L), "n_patches": int(L // cfg["foundation"]["labram"]["patch_size"]), "channels": list(ch),
        "raw_std_median": float(np.median(ctr[:500].std(axis=-1))),
        "raw_absmax_p99": float(np.percentile(np.abs(ctr[:500]), 99)),
        "model_input_std_median": float(np.median(ctr[:500].std(axis=-1)) * scale)}
print(json.dumps(info, indent=1))

m = build_model("labram", cfg, len(ch), L, ch)
tr_info = train_model(m, ctr, ytr, crop(Xes, pad, L), yes, cfg, 0, device, None, verbose=True)
hist = tr_info.get("history", [])
p_es = predict_logits(m, crop(Xes, pad, L), device).argmax(1)
p_te = predict_logits(m, crop(Xte, pad, L), device).argmax(1)
idx = np.random.default_rng(0).permutation(len(ytr))[:2000]
p_tr = predict_logits(m, ctr[idx], device).argmax(1)
res = {"info": info, "history": hist, "best_val_ba": tr_info.get("best_val_ba"), "epochs": tr_info.get("epochs"),
       "best_epoch": int(np.argmax([h["val_ba"] for h in hist])) if hist else None,
       "restored_model": {"train_subset_ba": balanced_accuracy(ytr[idx], p_tr), "train_pred1_share": float(p_tr.mean()),
                          "es_ba": balanced_accuracy(yes, p_es), "es_pred1_share": float(p_es.mean()),
                          "test_ba": balanced_accuracy(yte, p_te), "test_pred1_share": float(p_te.mean())},
       "seconds": time.time() - t0}
losses = [h["loss"] for h in hist]; vals = [h["val_ba"] for h in hist]
constant = max(res["restored_model"]["test_pred1_share"], 1 - res["restored_model"]["test_pred1_share"]) > 0.95
if min(losses) >= 0.68: verdict = "R1 optimisation failure"
elif max(vals) <= 0.55: verdict = "R2 generalisation / early-stopping failure"
elif constant: verdict = "R3 early-stopping-set artefact"
else: verdict = "R4 none of the above"
res["verdict"] = verdict
out = Path("results/diag"); out.mkdir(parents=True, exist_ok=True)
json.dump(res, open(out / "labram_shu_diag.json", "w"), indent=1)
print("VERDICT:", verdict); print(json.dumps(res["restored_model"], indent=1))
