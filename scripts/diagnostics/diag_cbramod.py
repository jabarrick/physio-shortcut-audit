"""Why is CBraMod's BA exactly 0.500 in every P4 training?  (2026-09-20, written BEFORE running)

Facts established before writing this script (read from D:/project/CBraMod, the official repo):
  * input scale: official PhysioNet-MI preprocessing takes epochs.get_data(units='uV') and the
    dataset returns data/100  -> our input_scale 0.01 on µV input MATCHES (待核实 -> verified).
  * official fine-tune recipe (finetune_main.py defaults / finetune_trainer.py):
      AdamW, backbone lr 1e-4, head lr 1e-3*sqrt(batch/256), weight_decay 5e-2, grad-norm clip 1.0,
      CosineAnnealingLR(T_max = epochs*steps, eta_min 1e-6), label_smoothing 0.1, dropout 0.1,
      classifier 'all_patch_reps' (flatten c*s*d -> 800 -> 200 -> n_cls); 'avgpooling_patch_reps'
      (mean over c,s -> Linear) is an official alternative.  Weights loaded with strict=True.
  * ours (trainer.train_model): ONE AdamW group at lr 1e-3 (10x the official backbone lr),
    wd 0, no clipping, no schedule, no label smoothing; mean-pool head; strict=False loading.

Arms (same data as P4: alpha cell, p = 0.5, s = 1.333 = strongest grid point; seed 0; fixed
N epochs, no early stopping, val BA printed every epoch on the P4 early-stopping slice):
  A  current trainer settings, current mean-pool head
  B  official optimiser recipe, current mean-pool head        (isolates the optimiser)
  C  official optimiser recipe, official all_patch_reps head  (isolates the head)

Pre-declared readings:
  R0  missing/unexpected keys non-empty                      -> pretrained weights were NOT loaded.
  R1  A: one class predicted for >= 95 % of windows and loss stuck near ln2 = 0.693
                                                              -> degenerate collapse confirmed.
  R2  B learns (non-constant predictions, loss falls, val BA > 0.55 at some epoch)
                                                              -> optimiser is the cause.
  R3  B collapses, C learns                                   -> pooling head is the cause.
  R4  all collapse                                            -> neither; look at data/patching.
The choice to follow the official recipe is made on principle (it is the published fine-tuning
protocol) and is NOT conditional on which arm scores highest; this script only establishes
whether the current setup is broken and whether the official recipe trains at all.
Run:   python diag_cbramod.py [--epochs 8]
Out:   results/pilots/cbramod_diag.json
"""
from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from p3audit.config import load_config
from p3audit.experiments.calibration import _roles
from p3audit.experiments.context import Context
from p3audit.generator.design import Cell
from p3audit.generator.semisynth import compose
from p3audit.models.registry import build_model
from p3audit.training.trainer import predict_logits, resolve_device, set_normalizer
from p3audit.utils.common import balanced_accuracy, seed_everything, write_json


def key_report(cfg, n_ch, n_t):
    """Reload the checkpoint strictly-compared against a fresh backbone."""
    import sys
    fc = cfg["foundation"]["cbramod"]
    sys.path.insert(0, str(Path(fc["repo_path"]).expanduser()))
    from models.cbramod import CBraMod
    bb = CBraMod(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12, nhead=8)
    state = torch.load(Path(fc["checkpoint"]).expanduser(), map_location="cpu")
    r = bb.load_state_dict(state, strict=False)
    return {"n_ckpt_keys": len(state), "missing": list(r.missing_keys), "unexpected": list(r.unexpected_keys)}


class AllPatchHead(nn.Module):
    """Official 'all_patch_reps' classifier; embed = 200-d input of the last Linear."""

    def __init__(self, n_ch, n_patch, dropout=0.1):
        super().__init__()
        self.body = nn.Sequential(nn.Flatten(), nn.Linear(n_ch * n_patch * 200, n_patch * 200), nn.ELU(),
                                  nn.Dropout(dropout), nn.Linear(n_patch * 200, 200), nn.ELU(), nn.Dropout(dropout))

    def forward(self, h):
        return self.body(h)


def make(cfg, ctx, n_t, head: str):
    m = build_model("cbramod", cfg, len(ctx.ch_names), n_t, ctx.ch_names)
    if head == "all_patch":
        n_patch = n_t // m.patch
        m.pre = AllPatchHead(len(ctx.ch_names), n_patch)
        bb = m.backbone

        def features(x, _m=m):
            B, C, T = x.shape
            return _m.pre(bb(x.reshape(B, C, T // _m.patch, _m.patch)))
        m.features = features
    return m


def run_arm(arm, cfg, ctx, tr, es, n_t, epochs, seed, device):
    seed_everything(seed)
    head = "all_patch" if arm == "C" else "mean"
    m = make(cfg, ctx, n_t, head).to(device)
    Xtr, ytr, Xes, yes = tr.materialize(), tr.y, es.materialize(), es.y
    set_normalizer(m, Xtr, cfg, seed=seed)
    bs = cfg["training"]["batch_size"]
    steps = int(np.ceil(len(Xtr) / bs))
    if arm == "A":
        opt = torch.optim.AdamW(m.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
        sched, clip, loss_fn = None, 0.0, nn.CrossEntropyLoss()
    else:
        bbp = [p for n, p in m.named_parameters() if n.startswith("backbone.")]
        oth = [p for n, p in m.named_parameters() if not n.startswith("backbone.")]
        opt = torch.optim.AdamW([{"params": bbp, "lr": 1e-4},
                                 {"params": oth, "lr": 1e-3 * (bs / 256) ** 0.5}], weight_decay=5e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps, eta_min=1e-6)
        clip, loss_fn = 1.0, nn.CrossEntropyLoss(label_smoothing=0.1)
    rng = np.random.default_rng(seed)
    hist = []
    for ep in range(epochs):
        m.train(); tot = 0.0; t0 = time.time()
        for b in np.array_split(rng.permutation(len(Xtr)), steps):
            if len(b) < 2:
                continue
            b = np.sort(b)
            xb = torch.as_tensor(Xtr[b], dtype=torch.float32, device=device)
            yb = torch.as_tensor(ytr[b], dtype=torch.long, device=device)
            opt.zero_grad()
            loss = loss_fn(m(xb), yb)
            loss.backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(m.parameters(), clip)
            opt.step()
            if sched:
                sched.step()
            tot += loss.item() * len(b)
        pred = predict_logits(m, Xes, device).argmax(1)
        row = {"epoch": ep, "loss": tot / len(Xtr), "val_ba": float(balanced_accuracy(yes, pred)),
               "frac_pred_1": float(pred.mean()), "sec": round(time.time() - t0, 1)}
        hist.append(row)
        print(f"[{arm}] ep{ep} loss {row['loss']:.4f} valBA {row['val_ba']:.3f} "
              f"pred1 {row['frac_pred_1']:.2f} ({row['sec']} s)", flush=True)
    # input scale actually seen by the backbone
    with torch.no_grad():
        xs = m.norm(torch.as_tensor(Xtr[:256], dtype=torch.float32, device=device)).cpu().numpy()
    del m; torch.cuda.empty_cache()
    return {"history": hist, "scaled_input_std": float(xs.std()), "scaled_input_absmax_p99": float(np.percentile(np.abs(xs), 99))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--arms", default="ABC")
    ap.add_argument("--s", type=float, default=1.333)
    a = ap.parse_args()
    cfg = load_config()
    ctx = Context.create(cfg)
    device = resolve_device(cfg["training"]["device"])
    tr_s, es_s, _ = _roles(ctx)            # same roles as P4; validation subjects are not touched
    caches = ctx.caches(tr_s + es_s)
    cell = Cell("alpha", 0.5, a.s, 1.0)
    tr = compose(caches, tr_s, cell, cfg, cell.p, key=(cell.cid, "cal_train", 0))
    es = compose(caches, es_s, cell, cfg, cell.p, key=(cell.cid, "cal_es", 0))
    out = {"cell": cell.cid, "n_train": len(tr), "n_es": len(es), "keys": key_report(cfg, len(ctx.ch_names), tr.core_len)}
    print("checkpoint:", {k: (v if isinstance(v, int) else len(v)) for k, v in out["keys"].items()},
          "missing[:5]", out["keys"]["missing"][:5], "unexpected[:5]", out["keys"]["unexpected"][:5])
    for arm in a.arms:
        out[arm] = run_arm(arm, cfg, ctx, tr, es, tr.core_len, a.epochs, 0, device)
        write_json(ctx.results / "pilots" / "cbramod_diag.json", out)
    print("written results/pilots/cbramod_diag.json")


if __name__ == "__main__":
    main()
