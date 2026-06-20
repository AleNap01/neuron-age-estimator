"""
run_train_cnn.py — Entry point: addestramento della CNN 3D.

Uso:
    python scripts/run_train_cnn.py
"""

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from brain_age.config import (
    AGES_TRAIN_CNN,
    CNN_MODEL_PATH,
    RANDOM_STATE,
    VOLUMES_TRAIN_CNN,
)
from brain_age.models.dataset import BrainMRIDataset
from brain_age.training.train_cnn import TrainingConfig, train_cnn3d


def main() -> None:
    volumes = np.load(VOLUMES_TRAIN_CNN)
    ages = np.load(AGES_TRAIN_CNN)
    print(f"Volumes: {volumes.shape} | Ages: {ages.shape}")

    idx_train, idx_val = train_test_split(
        np.arange(len(ages)), test_size=0.15, random_state=RANDOM_STATE,
    )
    print(f"Train: {len(idx_train)} | Validation: {len(idx_val)}")

    age_mean = ages[idx_train].mean()
    age_std = ages[idx_train].std()

    train_dataset = BrainMRIDataset(volumes, ages, idx_train, age_mean, age_std, augment=True)
    val_dataset = BrainMRIDataset(volumes, ages, idx_val, age_mean, age_std, augment=False)

    config = TrainingConfig(
        batch_size=8,
        epochs=40,
        learning_rate=1e-3,
        device=torch.device("cpu"),  # DirectML non supporta Conv3d
    )

    model, history = train_cnn3d(train_dataset, val_dataset, age_std, CNN_MODEL_PATH, config)

    print(f"\n=== Training completato — Best Val MAE: {history.best_val_mae:.4f} anni ===")
    print(f"Modello salvato in: {CNN_MODEL_PATH}")


if __name__ == "__main__":
    main()
