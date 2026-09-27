"""EEGNet-8,2 (Lawhern 2018), ShallowConvNet (Schirrmeister 2017) and the
CSOANet *placeholder*, all configured for 200 Hz input."""
from __future__ import annotations

import torch
from torch import nn

from .base import EEGModel


class EEGNet(EEGModel):
    """EEGNet-8,2: F1=8, D=2, F2=16; temporal kernel = sfreq/2 (100 @ 200 Hz)."""
    name = "eegnet"

    def __init__(self, n_ch, n_times, n_classes=2, sfreq=200.0, F1=8, D=2, F2=16, dropout=0.5):
        super().__init__(n_ch, n_times, n_classes)
        k = int(sfreq // 2)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, k), padding=(0, k // 2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(dropout))
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False),
            nn.BatchNorm2d(F2), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(dropout))
        with torch.no_grad():
            d = self.features(torch.zeros(1, n_ch, n_times)).shape[1]
        self.head = nn.Linear(d, n_classes)

    def features(self, x):
        return self.block2(self.block1(x.unsqueeze(1))).flatten(1)


class _Square(nn.Module):
    def forward(self, x):
        return x * x


class _SafeLog(nn.Module):
    def forward(self, x):
        return torch.log(torch.clamp(x, min=1e-6))


class ShallowConvNet(EEGModel):
    """40 temporal (k=20 ≈ 25 @ 250 Hz) → 40 spatial → square → avg-pool (60, stride 12) → log."""
    name = "shallow"

    def __init__(self, n_ch, n_times, n_classes=2, sfreq=200.0, n_filters=40, dropout=0.5):
        super().__init__(n_ch, n_times, n_classes)
        scale = sfreq / 250.0
        k, pool, stride = int(round(25 * scale)), int(round(75 * scale)), int(round(15 * scale))
        self.net = nn.Sequential(
            nn.Conv2d(1, n_filters, (1, k)),
            nn.Conv2d(n_filters, n_filters, (n_ch, 1), bias=False),
            nn.BatchNorm2d(n_filters), _Square(), nn.AvgPool2d((1, pool), (1, stride)), _SafeLog(),
            nn.Dropout(dropout))
        with torch.no_grad():
            d = self.features(torch.zeros(1, n_ch, n_times)).shape[1]
        self.head = nn.Linear(d, n_classes)

    def features(self, x):
        return self.net(x.unsqueeze(1)).flatten(1)


class CSOANetPlaceholder(EEGModel):
    """PLACEHOLDER for CSOANet — replace with the published implementation.

    Only the property stated in the outline (3.5) is reproduced: three temporal
    branches whose receptive fields are ≈72, 200 and 776 ms at 200 Hz, feeding a
    compact depthwise-spatial / separable stage (~3k parameters).  The interface
    (`features`, `head`) is what the rest of the pipeline depends on; swap the
    body for the real architecture and keep that interface.
    """
    name = "csoanet"

    def __init__(self, n_ch, n_times, n_classes=2, sfreq=200.0, F=4, D=2, F2=16, dropout=0.25,
                 rf_ms=(72.0, 200.0, 776.0)):
        super().__init__(n_ch, n_times, n_classes)
        ks = [max(3, int(round(r / 1000 * sfreq)) | 1) for r in rf_ms]      # odd kernels: 15, 41, 155
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(1, F, (1, k), padding=(0, k // 2), bias=False), nn.BatchNorm2d(F))
            for k in ks])
        C = F * len(ks)
        self.spatial = nn.Sequential(
            nn.Conv2d(C, C * D, (n_ch, 1), groups=C, bias=False), nn.BatchNorm2d(C * D), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(dropout))
        self.sep = nn.Sequential(
            nn.Conv2d(C * D, C * D, (1, 16), padding=(0, 8), groups=C * D, bias=False),
            nn.Conv2d(C * D, F2, 1, bias=False), nn.BatchNorm2d(F2), nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 8)), nn.Dropout(dropout))
        with torch.no_grad():
            d = self.features(torch.zeros(1, n_ch, n_times)).shape[1]
        self.head = nn.Linear(d, n_classes)

    def features(self, x):
        x = x.unsqueeze(1)
        x = torch.cat([b(x) for b in self.branches], dim=1)
        return self.sep(self.spatial(x)).flatten(1)


class RawFeatureLinear(EEGModel):
    """Upper reference (3.5): log band power per channel × band -> linear.
    FFT-based and differentiable, so interventions and IG apply unchanged."""
    name = "linear_raw"

    def __init__(self, n_ch, n_times, n_classes=2, sfreq=200.0,
                 bands=((4, 8), (8, 13), (13, 20), (20, 30), (30, 45))):
        super().__init__(n_ch, n_times, n_classes)
        f = torch.fft.rfftfreq(n_times, 1 / sfreq)
        masks = torch.stack([((f >= lo) & (f < hi)).float() for lo, hi in bands])
        self.register_buffer("masks", masks / masks.sum(1, keepdim=True))
        self.bn = nn.BatchNorm1d(n_ch * len(bands), affine=False)
        self.head = nn.Linear(n_ch * len(bands), n_classes)

    def features(self, x):
        p = torch.fft.rfft(x, dim=-1).abs() ** 2
        bp = torch.einsum("bcf,kf->bck", p, self.masks)
        return self.bn(torch.log(bp + 1e-6).flatten(1))
