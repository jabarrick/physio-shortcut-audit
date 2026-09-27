"""Common model interface.

Every model maps µV input (B, C, T) -> logits (B, 2) and exposes
`embed(x)` = penultimate representation (B, D) and `head` = final Linear.
Normalisation (global training statistics, 3.3) lives *inside* the model as
fixed buffers, so interventions and attributions operate in the µV domain.
"""
from __future__ import annotations

import torch
from torch import nn


class Normalizer(nn.Module):
    def __init__(self, n_ch: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(1, n_ch, 1))
        self.register_buffer("std", torch.ones(1, n_ch, 1))

    def set_stats(self, mean, std):
        self.mean.copy_(torch.as_tensor(mean, dtype=torch.float32).view(1, -1, 1))
        self.std.copy_(torch.as_tensor(std, dtype=torch.float32).clamp_min(1e-6).view(1, -1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


class EEGModel(nn.Module):
    name = "base"

    def __init__(self, n_ch: int, n_times: int, n_classes: int = 2):
        super().__init__()
        self.n_ch, self.n_times, self.n_classes = n_ch, n_times, n_classes
        self.norm = Normalizer(n_ch)
        self.head: nn.Linear

    def features(self, x: torch.Tensor) -> torch.Tensor:  # normalised input -> embedding
        raise NotImplementedError

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(self.norm(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x))

    @property
    def embed_dim(self) -> int:
        return self.head.in_features


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
