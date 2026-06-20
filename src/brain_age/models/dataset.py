"""
dataset.py — Dataset PyTorch per i volumi MRI usati dalla CNN 3D.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class BrainMRIDataset(Dataset):
    """
    Dataset PyTorch sui volumi MRI già caricati in memoria (array numpy).

    Le età vengono normalizzate (z-score) usando media/std calcolate sul
    solo training set, per evitare data leakage dal validation/test set.
    Quando augment=True, applica un flip casuale lungo l'asse sagittale,
    sfruttando l'approssimativa simmetria bilaterale del cervello.
    """

    def __init__(
        self,
        volumes: np.ndarray,
        ages: np.ndarray,
        indices: np.ndarray,
        age_mean: float,
        age_std: float,
        augment: bool = False,
    ) -> None:
        self.volumes = volumes
        self.ages = ages
        self.indices = indices
        self.age_mean = age_mean
        self.age_std = age_std
        self.augment = augment

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        real_idx = self.indices[idx]
        volume = self.volumes[real_idx].copy()
        age = self.ages[real_idx]

        if self.augment and np.random.rand() > 0.5:
            volume = np.flip(volume, axis=0).copy()

        volume = volume[np.newaxis, :, :, :]  # canale singolo
        age_normalized = (age - self.age_mean) / self.age_std

        return (
            torch.from_numpy(volume).float(),
            torch.tensor(age_normalized, dtype=torch.float32),
        )

    def denormalize(self, age_normalized: torch.Tensor) -> torch.Tensor:
        """Riporta un'età normalizzata (z-score) alla scala originale in anni."""
        return age_normalized * self.age_std + self.age_mean
