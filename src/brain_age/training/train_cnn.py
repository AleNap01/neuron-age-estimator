"""
train_cnn.py — Loop di addestramento per la CNN 3D.

Il training è forzato su CPU: DirectML (necessario per usare GPU AMD su
Windows senza CUDA) non supporta l'operazione Conv3d, quindi anche
disponendo di una GPU compatibile DirectML non sarebbe possibile usarla
per questa architettura.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from brain_age.models.cnn3d import BrainAgeCNN3D
from brain_age.models.dataset import BrainMRIDataset


@dataclass
class TrainingConfig:
    batch_size: int = 8
    epochs: int = 40
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    early_stop_patience: int = 10
    lr_scheduler_patience: int = 4
    lr_scheduler_factor: float = 0.5
    device: torch.device = torch.device("cpu")


@dataclass
class TrainingHistory:
    train_mae: list[float]
    val_mae: list[float]
    best_val_mae: float
    stopped_early: bool
    epochs_run: int


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    age_std: float,
    device: torch.device,
) -> float:
    """Esegue un'epoca di training, ritorna il MAE (in anni) sul training set."""
    model.train()
    losses = []
    for volumes, ages in loader:
        volumes, ages = volumes.to(device), ages.to(device)

        optimizer.zero_grad()
        predictions = model(volumes)
        loss = criterion(predictions, ages)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return float(np.mean(losses)) * age_std


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    age_std: float,
    device: torch.device,
) -> float:
    """Valuta il modello su un loader, ritorna il MAE (in anni)."""
    model.eval()
    losses = []
    for volumes, ages in loader:
        volumes, ages = volumes.to(device), ages.to(device)
        predictions = model(volumes)
        loss = criterion(predictions, ages)
        losses.append(loss.item())

    return float(np.mean(losses)) * age_std


def train_cnn3d(
    train_dataset: BrainMRIDataset,
    val_dataset: BrainMRIDataset,
    age_std: float,
    save_path: Path,
    config: TrainingConfig | None = None,
    verbose: bool = True,
) -> tuple[BrainAgeCNN3D, TrainingHistory]:
    """
    Addestra la CNN 3D con early stopping e learning rate scheduling.

    Il modello con il miglior MAE di validazione viene salvato su disco
    durante il training (non solo al termine), così un'interruzione non
    fa perdere il progresso.
    """
    if config is None:
        config = TrainingConfig()

    torch.manual_seed(42)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                               shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                             shuffle=False, num_workers=0)

    model = BrainAgeCNN3D().to(config.device)
    criterion = nn.L1Loss()  # L1 Loss = MAE direttamente
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=config.lr_scheduler_factor,
        patience=config.lr_scheduler_patience,
    )

    history = {"train_mae": [], "val_mae": []}
    best_val_mae = float("inf")
    patience_counter = 0
    stopped_early = False
    epochs_run = 0

    for epoch in range(1, config.epochs + 1):
        t0 = time.time()

        train_mae = train_one_epoch(model, train_loader, criterion, optimizer, age_std, config.device)
        val_mae = evaluate(model, val_loader, criterion, age_std, config.device)
        scheduler.step(val_mae)

        history["train_mae"].append(train_mae)
        history["val_mae"].append(val_mae)
        epochs_run = epoch

        if verbose:
            elapsed = time.time() - t0
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:3d}/{config.epochs} | "
                  f"Train MAE: {train_mae:.3f} | Val MAE: {val_mae:.3f} | "
                  f"LR: {current_lr:.1e} | Tempo: {elapsed:.1f}s")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            if verbose:
                print(f"  -> Nuovo miglior modello salvato (Val MAE: {best_val_mae:.3f})")
        else:
            patience_counter += 1
            if patience_counter >= config.early_stop_patience:
                stopped_early = True
                if verbose:
                    print(f"\nEarly stopping all'epoca {epoch}.")
                break

    result_history = TrainingHistory(
        train_mae=history["train_mae"],
        val_mae=history["val_mae"],
        best_val_mae=best_val_mae,
        stopped_early=stopped_early,
        epochs_run=epochs_run,
    )

    # Ricarica i pesi del modello migliore (non necessariamente l'ultimo)
    model.load_state_dict(torch.load(save_path, map_location=config.device))

    return model, result_history
