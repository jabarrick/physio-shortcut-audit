"""Training with early stopping on the 10 validation subjects (3.2, 3.5)."""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
from torch import nn

from ..models.registry import is_foundation
from ..utils.common import balanced_accuracy, get_logger, seed_everything

log = get_logger("p3audit.train")


def resolve_device(pref: str = "auto") -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pref.startswith("cuda") and not torch.cuda.is_available():
        # PILOT_LOG 13.9: fail at once with the reason, instead of at the first .to(device)
        raise RuntimeError("training.device is 'cuda' but CUDA is unavailable.  If nvidia-smi says "
                           "'GPU is lost', reboot (PILOT_LOG 13.8).  Nothing was computed on the CPU.")
    return torch.device(pref)


def set_normalizer(model, X_train: np.ndarray, cfg, max_windows: int = 2000, seed: int = 0) -> None:
    """Global training statistics per channel (3.3).  Foundation models keep their
    native µV scaling (x · input_scale) instead."""
    if is_foundation(model):
        name = model.name.split("-")[0]
        scale = cfg["foundation"].get(name, {}).get("input_scale", 0.01)
        model.norm.set_stats(np.zeros(model.n_ch), np.full(model.n_ch, 1.0 / scale))
        return
    idx = np.random.default_rng(seed).permutation(len(X_train))[:max_windows]
    sample = np.asarray(X_train[np.sort(idx)], dtype=np.float64)
    model.norm.set_stats(sample.mean(axis=(0, 2)), sample.std(axis=(0, 2)))


def subject_zscore(X: np.ndarray, subj: np.ndarray) -> np.ndarray:
    """Transductive sensitivity condition (3.3): per-subject, per-channel z-score."""
    out = np.empty_like(X, dtype=np.float32)
    for s in np.unique(subj):
        m = subj == s
        mu = X[m].mean(axis=(0, 2), keepdims=True)
        sd = X[m].std(axis=(0, 2), keepdims=True) + 1e-6
        out[m] = (X[m] - mu) / sd
    return out


def _batches(n, bs, rng=None):
    idx = rng.permutation(n) if rng is not None else np.arange(n)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


@torch.no_grad()
def predict_logits(model, X, device=None, batch_size: int = 256) -> np.ndarray:
    device = device or next(model.parameters()).device
    model.eval()
    out = []
    for b in _batches(len(X), batch_size):
        xb = torch.as_tensor(np.asarray(X[b]), dtype=torch.float32, device=device)
        out.append(model(xb).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def embeddings(model, X, device=None, batch_size: int = 256) -> np.ndarray:
    device = device or next(model.parameters()).device
    model.eval()
    out = []
    for b in _batches(len(X), batch_size):
        xb = torch.as_tensor(np.asarray(X[b]), dtype=torch.float32, device=device)
        out.append(model.embed(xb).cpu().numpy())
    return np.concatenate(out)


def evaluate_ba(model, X, y, device=None) -> float:
    return balanced_accuracy(y, predict_logits(model, X, device).argmax(1))


def labram_layer_id(name: str, n_blocks: int) -> int:
    """Official optim_factory.get_num_layer_for_vit on the backbone-relative name; our adapter
    prefixes the backbone with 'backbone.m.'; everything outside it (head) is the top layer."""
    num_max = n_blocks + 2
    if not name.startswith("backbone.m."):
        return num_max - 1
    v = name[len("backbone.m."):]
    if v in ("cls_token", "mask_token", "pos_embed") or v.startswith("patch_embed"):
        return 0
    if v.startswith("blocks."):
        return int(v.split(".")[1]) + 1
    return num_max - 1


def labram_param_groups(model, ft: dict, n_blocks: int) -> list[dict]:
    """Official get_parameter_groups: no weight decay for ndim <= 1, biases and
    {pos_embed, cls_token, time_embed}; lr_scale = layer_decay ** (num_layers + 1 - layer_id)."""
    skip = {"backbone.m.pos_embed", "backbone.m.cls_token", "backbone.m.time_embed"}
    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        nd = p.ndim <= 1 or n.endswith(".bias") or n in skip
        lid = labram_layer_id(n, n_blocks)
        key = (lid, nd)
        if key not in groups:
            groups[key] = {"params": [], "weight_decay": 0.0 if nd else ft["weight_decay"],
                           "lr_scale": ft["layer_decay"] ** (n_blocks + 1 - lid), "names": []}
        groups[key]["params"].append(p); groups[key]["names"].append(n)
    return list(groups.values())


def labram_optimizer(model, cfg, total_steps: int, steps_per_epoch: int | None = None):
    """PILOT_LOG 15.26: official LaBraM-Base fine-tuning recipe (paper table 5 / README TUAB command):
    AdamW (0.9, 0.999), peak lr 5e-4 with layer-wise decay 0.65, 5 warmup epochs (linear from 0),
    cosine to 1e-6 per step, weight decay 0.05, no gradient clipping.  Binary task -> no label smoothing
    (official binary tasks use BCE on one logit; two-logit CE is the equivalent)."""
    import math
    ft = cfg["foundation"]["labram_finetune"]
    tc = cfg["training"]
    n_blocks = len(model.backbone.m.blocks)
    groups = labram_param_groups(model, ft, n_blocks)
    opt = torch.optim.AdamW([{k: v for k, v in g.items() if k != "names"} for g in groups], lr=ft["lr"],
                            betas=(0.9, 0.999), eps=1e-8)
    steps_per_epoch = steps_per_epoch or max(1, total_steps // max(1, tc["max_epochs"]))
    warm = ft["warmup_epochs"] * steps_per_epoch
    lo = ft["min_lr"] / ft["lr"]

    def f(step):
        if step < warm:
            return step / max(1, warm)
        q = (step - warm) / max(1, total_steps - warm)
        return lo + 0.5 * (1 - lo) * (1 + math.cos(math.pi * min(1.0, q)))

    lam = [(lambda st, s_=g["lr_scale"]: s_ * f(st)) for g in groups]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lam)
    return opt, sched, 0.0, nn.CrossEntropyLoss(label_smoothing=ft["label_smoothing"])


def make_optimizer(model, cfg, total_steps: int, steps_per_epoch: int | None = None):
    """Conv nets: one AdamW group from `training`.  Foundation models: the official fine-tuning
    recipe from `foundation.finetune` (PILOT_LOG 11.2 / 11.7) — backbone lr, head lr
    head_lr_base*sqrt(batch/ref), weight decay, grad-norm clip, cosine schedule, label smoothing."""
    tc = cfg["training"]
    if is_foundation(model) and getattr(model, "name", "") == "labram":
        return labram_optimizer(model, cfg, total_steps, steps_per_epoch)
    ft = cfg["foundation"].get("finetune") if is_foundation(model) else None
    if not ft:
        opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
        return opt, None, 0.0, nn.CrossEntropyLoss()
    bb = [p for n, p in model.named_parameters() if n.startswith("backbone.")]
    other = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    head_lr = ft["head_lr_base"] * (tc["batch_size"] / ft["head_lr_ref_batch"]) ** 0.5
    opt = torch.optim.AdamW([{"params": bb, "lr": ft["backbone_lr"]}, {"params": other, "lr": head_lr}],
                            weight_decay=ft["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_steps), eta_min=ft["cosine_eta_min"])
    return opt, sched, float(ft["clip_grad_norm"]), nn.CrossEntropyLoss(label_smoothing=ft["label_smoothing"])


def train_model(model, X_tr, y_tr, X_va, y_va, cfg, seed: int, device=None, max_epochs: int | None = None,
                verbose: bool = True) -> dict:
    tc = cfg["training"]
    device = device or resolve_device(tc["device"])
    seed_everything(seed)
    model.to(device)
    set_normalizer(model, X_tr, cfg, seed=seed)
    epochs = max_epochs or tc["max_epochs"]
    steps = max(1, int(np.ceil(len(X_tr) / tc["batch_size"])))
    opt, sched, clip, loss_fn = make_optimizer(model, cfg, epochs * steps, steps)
    rng = np.random.default_rng(seed)
    best, best_state, bad, hist = -np.inf, None, 0, []
    t_start = time.time()
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for b in _batches(len(X_tr), tc["batch_size"], rng):
            if len(b) < 2:
                continue
            xb = torch.as_tensor(np.asarray(X_tr[np.sort(b)]), dtype=torch.float32, device=device)
            yb = torch.as_tensor(np.asarray(y_tr)[np.sort(b)], dtype=torch.long, device=device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            if sched is not None:
                sched.step()
            tot += loss.item() * len(b)
        va = evaluate_ba(model, X_va, y_va, device)
        hist.append({"epoch": ep, "loss": tot / len(X_tr), "val_ba": va})
        if va > best + 1e-4:
            best, best_state, bad = va, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
        if verbose:
            log.info(f"{model.name} ep{ep:3d} loss {tot / len(X_tr):.4f} val BA {va:.3f}")
        if bad >= tc["patience"]:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val_ba": float(best), "epochs": len(hist), "seconds": time.time() - t_start, "history": hist}
