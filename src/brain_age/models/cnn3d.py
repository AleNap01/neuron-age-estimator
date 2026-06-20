"""
cnn3d.py — Architettura della CNN 3D per Brain Age Estimation.

Rete leggera (≈300k parametri) progettata per essere addestrabile su CPU:
4 blocchi convoluzionali con pooling aggressivo riducono il volume da 64^3
a 4^3 prima del Global Average Pooling, mantenendo il numero di parametri
contenuto e il training fattibile senza GPU dedicata.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Blocco Conv3D + BatchNorm + ReLU + MaxPool (dimezza lo spazio)."""
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool3d(2),
    )


class BrainAgeCNN3D(nn.Module):
    """
    CNN 3D per la stima dell'età da volumi MRI 64x64x64 a canale singolo.

    Architettura:
        Input (1, 64, 64, 64)
          -> conv_block(1,   16)  -> (16, 32, 32, 32)
          -> conv_block(16,  32)  -> (32, 16, 16, 16)
          -> conv_block(32,  64)  -> (64,  8,  8,  8)
          -> conv_block(64, 128)  -> (128, 4,  4,  4)
          -> AdaptiveAvgPool3d(1) -> (128, 1,  1,  1)
          -> Flatten + Linear(128->64) + ReLU + Dropout + Linear(64->1)
    """

    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()

        self.features = nn.Sequential(
            _conv_block(1, 16),
            _conv_block(16, 32),
            _conv_block(32, 64),
            _conv_block(64, 128),
        )
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x)
        x = self.regressor(x)
        return x.squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
