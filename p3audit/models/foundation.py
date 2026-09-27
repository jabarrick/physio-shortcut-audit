"""Foundation-model adapters (CBraMod core model; LaBraM supplementary, 3.5).

Both official models take 200-Hz input cut into 1-s patches (B, C, P, 200)
in units of 100 µV (the official fine-tuning code divides µV by 100) —
[待核实 against each repo's current README before P3].

The official code is *not* vendored.  Point `foundation.<name>.repo_path`
at a local clone and `checkpoint` at the pretrained weights; the builders
below import from there.  If the import fails they raise with instructions.
`PatchTransformerStandIn` is a small randomly initialised patch transformer
used only for pipeline tests / dry runs; it is NOT a foundation model.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

from .base import EEGModel


class FoundationAdapter(EEGModel):
    uses_raw_scale = True   # trainer sets Normalizer to x * input_scale instead of z-scoring

    def __init__(self, name: str, backbone: nn.Module, n_ch: int, n_times: int, embed_dim: int,
                 patch: int = 200, n_classes: int = 2, forward_kwargs: dict | None = None,
                 head_type: str = "avgpooling_patch_reps", dropout: float = 0.1):
        super().__init__(n_ch, n_times, n_classes)
        if n_times % patch:
            raise ValueError(f"window of {n_times} samples is not a multiple of the {patch}-sample patch")
        self.name = name
        self.backbone = backbone
        self.patch = patch
        self.forward_kwargs = forward_kwargs or {}
        self.head_type = head_type
        n_patch = n_times // patch
        if head_type == "all_patch_reps":
            # official CBraMod default (models/model_for_physio.py): flatten (c, s, d) ->
            # Linear(c*s*d, s*200) -> ELU -> Dropout -> Linear(s*200, 200) -> ELU -> Dropout -> Linear(200, n)
            # `embed` is the 200-d input of the final Linear (same definition as the conv nets).
            self.pre = nn.Sequential(nn.Flatten(), nn.Linear(n_ch * n_patch * embed_dim, n_patch * 200), nn.ELU(),
                                     nn.Dropout(dropout), nn.Linear(n_patch * 200, 200), nn.ELU(), nn.Dropout(dropout))
            self.head = nn.Linear(200, n_classes)
        elif head_type == "avgpooling_patch_reps":
            self.pre = None
            self.head = nn.Linear(embed_dim, n_classes)
        else:
            raise ValueError(f"unknown foundation head {head_type!r}")

    def features(self, x):
        B, C, T = x.shape
        h = self.backbone(x.reshape(B, C, T // self.patch, self.patch), **self.forward_kwargs)
        if self.pre is not None:
            if h.dim() != 4:
                raise ValueError("all_patch_reps needs (B, C, P, D) backbone output")
            return self.pre(h)
        if h.dim() == 4:        # (B, C, P, D) -> mean over channels and patches
            return h.mean(dim=(1, 2))
        if h.dim() == 3:        # (B, tokens, D)
            return h.mean(dim=1)
        return h


class PatchTransformerStandIn(nn.Module):
    """(B, C, P, 200) -> (B, C, P, D).  For tests only."""

    def __init__(self, n_ch: int, patch: int = 200, d: int = 64, layers: int = 2, heads: int = 4, max_p: int = 30):
        super().__init__()
        self.proj = nn.Linear(patch, d)
        self.ch_emb = nn.Parameter(torch.zeros(1, n_ch, 1, d))
        self.t_emb = nn.Parameter(torch.zeros(1, 1, max_p, d))
        enc = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.d = d

    def forward(self, x):
        B, C, P, L = x.shape
        h = self.proj(x) + self.ch_emb + self.t_emb[:, :, :P]
        h = self.enc(h.reshape(B, C * P, self.d))
        return h.reshape(B, C, P, self.d)


def _add_repo(repo_path):
    if not repo_path:
        raise RuntimeError("foundation.<model>.repo_path is not set; clone the official repository and set it")
    p = str(Path(repo_path).expanduser())
    if p not in sys.path:
        sys.path.insert(0, p)


def build_cbramod(cfg, n_ch: int, n_times: int, ch_names) -> FoundationAdapter:
    fc = cfg["foundation"]["cbramod"]
    _add_repo(fc["repo_path"])
    try:
        from models.cbramod import CBraMod  # official repo layout [待核实]
    except ImportError as e:
        raise RuntimeError(f"cannot import CBraMod from {fc['repo_path']}: {e}") from e
    bb = CBraMod(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12, nhead=8)
    if fc["checkpoint"]:
        state = torch.load(Path(fc["checkpoint"]).expanduser(), map_location="cpu", weights_only=True)
        bb.load_state_dict(state, strict=True)   # official code loads strictly; 211/211 keys verified 2026-09-20
    if hasattr(bb, "proj_out"):
        bb.proj_out = nn.Identity()
    hc = cfg["foundation"].get("cbramod_head", {"type": "avgpooling_patch_reps"})
    return FoundationAdapter("cbramod", bb, n_ch, n_times, embed_dim=200, patch=fc["patch_size"],
                             head_type=hc["type"], dropout=hc.get("dropout", 0.1))


# LaBraM expected checkpoint deltas (PILOT_LOG 15.26; verified on labram-base.pth, repo commit c431221e):
# the pretraining checkpoint carries the student under 'student.' plus pretraining-only heads.
LABRAM_EXPECTED_MISSING = {"fc_norm.weight", "fc_norm.bias"}           # mean-pooling norm, new at fine-tuning
LABRAM_EXPECTED_UNEXPECTED = {"mask_token", "lm_head.weight", "lm_head.bias", "norm.weight", "norm.bias"}


def labram_standard_1020(repo_path) -> list[str]:
    """Read `standard_1020` from the official utils.py WITHOUT importing it (it imports h5py,
    tensorboardX, ... which the analysis environment does not need)."""
    import ast
    src = (Path(repo_path).expanduser() / "utils.py").read_text(encoding="utf-8")
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "standard_1020":
            return list(ast.literal_eval(node.value))
    raise RuntimeError("standard_1020 not found in LaBraM utils.py")


def labram_input_chans(ch_names, std) -> list[int]:
    """Official utils.get_input_chans: [0 (cls)] + (index in standard_1020 + 1), names upper-cased."""
    up = [c.upper().rstrip(".") for c in ch_names]
    missing = [c for c in up if c not in std]
    if missing:
        raise RuntimeError(f"channels not in LaBraM standard_1020: {missing}")
    return [0] + [std.index(c) + 1 for c in up]


def labram_filter_checkpoint(state: dict) -> dict:
    """Official run_class_finetuning.py: take 'model', keep 'student.*' with the prefix removed,
    drop relative_position_index entries."""
    state = state.get("model", state)
    out = {k[8:]: v for k, v in state.items() if k.startswith("student.")}
    return {k: v for k, v in out.items() if "relative_position_index" not in k}


def build_labram(cfg, n_ch: int, n_times: int, ch_names) -> FoundationAdapter:
    """PILOT_LOG 15.26: official LaBraM-Base fine-tuning setup (README TUAB command + run_class_finetuning
    defaults): no qkv bias, no relative position bias, absolute position embedding, layer scale 0.1,
    drop path 0.1, mean pooling + fc_norm, head init scaled by 0.001."""
    fc = cfg["foundation"]["labram"]
    mk = fc.get("model_kwargs", {})
    _add_repo(fc["repo_path"])
    try:
        import modeling_finetune
    except ImportError as e:
        raise RuntimeError(f"cannot import LaBraM from {fc['repo_path']}: {e}") from e
    bb = modeling_finetune.labram_base_patch200_200(
        pretrained=False, num_classes=0, drop_path_rate=mk.get("drop_path", 0.1),
        use_mean_pooling=mk.get("use_mean_pooling", True), init_scale=mk.get("init_scale", 0.001),
        use_rel_pos_bias=mk.get("use_rel_pos_bias", False), use_abs_pos_emb=mk.get("use_abs_pos_emb", True),
        init_values=mk.get("init_values", 0.1), qkv_bias=mk.get("qkv_bias", False))
    if not fc["checkpoint"]:
        raise RuntimeError("foundation.labram.checkpoint is not set (official checkpoints/labram-base.pth)")
    state = torch.load(Path(fc["checkpoint"]).expanduser(), map_location="cpu", weights_only=False)
    r = bb.load_state_dict(labram_filter_checkpoint(state), strict=False)
    if set(r.missing_keys) != LABRAM_EXPECTED_MISSING or set(r.unexpected_keys) != LABRAM_EXPECTED_UNEXPECTED:
        raise RuntimeError(f"LaBraM checkpoint mismatch: missing {r.missing_keys}, unexpected {r.unexpected_keys}")
    input_chans = labram_input_chans(ch_names, labram_standard_1020(fc["repo_path"]))

    class _Wrap(nn.Module):
        def __init__(self, m):
            super().__init__(); self.m = m

        def forward(self, x):
            return self.m.forward_features(x, input_chans=input_chans)

    ad = FoundationAdapter("labram", _Wrap(bb), n_ch, n_times, embed_dim=bb.embed_dim, patch=fc["patch_size"],
                           head_type="avgpooling_patch_reps")
    # official head: trunc_normal(std .02) then weight and bias multiplied by init_scale
    nn.init.trunc_normal_(ad.head.weight, std=0.02); nn.init.zeros_(ad.head.bias)
    with torch.no_grad():
        ad.head.weight.mul_(mk.get("init_scale", 0.001))
    return ad


def build_standin(cfg, n_ch: int, n_times: int, ch_names, name="standin") -> FoundationAdapter:
    hc = (cfg["foundation"].get("cbramod_head", {}) if name.startswith("cbramod") else {}) or {}
    return FoundationAdapter(name, PatchTransformerStandIn(n_ch), n_ch, n_times, embed_dim=64,
                             head_type=hc.get("type", "avgpooling_patch_reps"), dropout=hc.get("dropout", 0.1))
