"""
05_train_cnn3d.py — CNN 3D per Brain Age Estimation (CPU-only)
Brain Age Estimation - Progetto Enterprise

Architettura leggera pensata per training su CPU:
- 4 blocchi Conv3D + BatchNorm + ReLU + MaxPool (64 -> 32 -> 16 -> 8 -> 4)
- Global Average Pooling + 2 layer Fully Connected
- Dropout per regolarizzazione

Split: 85% train / 15% validation (dal training set originale)
"""

import numpy as np
import pandas as pd
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────
# 0. PATH E CONFIG
# ──────────────────────────────────────────────
BASE = r"C:\Users\napol\Desktop\Archivio Generale\10--Work and projects\Neuron_Age_Estimator"
OUT_DIR = os.path.join(BASE, "outputs")

VOLUME_SIZE = 64
BATCH_SIZE = 8
EPOCHS = 40
LR = 1e-3
DEVICE = torch.device("cpu")  # forzato CPU (DirectML non supporta Conv3d)

torch.manual_seed(42)
np.random.seed(42)

print(f"Device: {DEVICE}")
print(f"CPU threads disponibili: {torch.get_num_threads()}")

# ──────────────────────────────────────────────
# 1. CARICA DATI
# ──────────────────────────────────────────────
print("\nCaricamento volumi downsampled...")
volumes = np.load(os.path.join(OUT_DIR, f"volumes_train_{VOLUME_SIZE}.npy"))
ages = np.load(os.path.join(OUT_DIR, f"ages_train_{VOLUME_SIZE}.npy"))

print(f"Volumes shape: {volumes.shape}")
print(f"Ages shape: {ages.shape}")

# Split train/validation (85/15)
idx_train, idx_val = train_test_split(
    np.arange(len(ages)), test_size=0.15, random_state=42
)

print(f"Train: {len(idx_train)} soggetti | Validation: {len(idx_val)} soggetti")

# Normalizzazione età (z-score) — aiuta la convergenza della rete
age_mean = ages[idx_train].mean()
age_std = ages[idx_train].std()
print(f"Age mean: {age_mean:.2f}, std: {age_std:.2f}")

# ──────────────────────────────────────────────
# 2. DATASET E DATALOADER
# ──────────────────────────────────────────────


class BrainMRIDataset(Dataset):
    def __init__(self, volumes, ages, indices, age_mean, age_std, augment=False):
        self.volumes = volumes
        self.ages = ages
        self.indices = indices
        self.age_mean = age_mean
        self.age_std = age_std
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        vol = self.volumes[real_idx].copy()
        age = self.ages[real_idx]

        # Data augmentation leggera: flip random sull'asse sagittale (simmetria cerebrale)
        if self.augment and np.random.rand() > 0.5:
            vol = np.flip(vol, axis=0).copy()

        vol = vol[np.newaxis, :, :, :]  # canale singolo
        age_norm = (age - self.age_mean) / self.age_std

        return torch.from_numpy(vol).float(), torch.tensor(age_norm, dtype=torch.float32)


train_dataset = BrainMRIDataset(
    volumes, ages, idx_train, age_mean, age_std, augment=True)
val_dataset = BrainMRIDataset(
    volumes, ages, idx_val,   age_mean, age_std, augment=False)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader = DataLoader(
    val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ──────────────────────────────────────────────
# 3. ARCHITETTURA CNN 3D
# ──────────────────────────────────────────────


class BrainAgeCNN3D(nn.Module):
    def __init__(self):
        super().__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv3d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(2),  # dimezza la dimensione spaziale
            )

        self.features = nn.Sequential(
            conv_block(1,  16),   # 64 -> 32
            conv_block(16, 32),   # 32 -> 16
            conv_block(32, 64),   # 16 -> 8
            conv_block(64, 128),  # 8  -> 4
        )

        self.global_pool = nn.AdaptiveAvgPool3d(1)  # 4x4x4 -> 1x1x1

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.regressor(x)
        return x.squeeze(-1)


model = BrainAgeCNN3D().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"\nModello creato — parametri totali: {n_params:,}")

# ──────────────────────────────────────────────
# 4. TRAINING LOOP
# ──────────────────────────────────────────────
criterion = nn.L1Loss()  # L1 Loss = MAE direttamente
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=4)

history = {"train_mae": [], "val_mae": []}
best_val_mae = float("inf")
best_model_path = os.path.join(OUT_DIR, "best_cnn3d.pth")
patience_counter = 0
EARLY_STOP_PATIENCE = 10

print("\n" + "=" * 50)
print(f"Training CNN 3D — {EPOCHS} epoche, batch size {BATCH_SIZE}")
print("=" * 50)

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()

    # --- TRAIN ---
    model.train()
    train_losses = []
    for volumes_batch, ages_batch in train_loader:
        volumes_batch, ages_batch = volumes_batch.to(
            DEVICE), ages_batch.to(DEVICE)

        optimizer.zero_grad()
        preds = model(volumes_batch)
        loss = criterion(preds, ages_batch)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

    train_mae_norm = np.mean(train_losses)
    train_mae_years = train_mae_norm * age_std  # de-normalizza in anni

    # --- VALIDATION ---
    model.eval()
    val_losses = []
    with torch.no_grad():
        for volumes_batch, ages_batch in val_loader:
            volumes_batch, ages_batch = volumes_batch.to(
                DEVICE), ages_batch.to(DEVICE)
            preds = model(volumes_batch)
            loss = criterion(preds, ages_batch)
            val_losses.append(loss.item())

    val_mae_norm = np.mean(val_losses)
    val_mae_years = val_mae_norm * age_std

    scheduler.step(val_mae_years)

    history["train_mae"].append(train_mae_years)
    history["val_mae"].append(val_mae_years)

    elapsed = time.time() - t0
    current_lr = optimizer.param_groups[0]["lr"]

    print(f"Epoch {epoch:3d}/{EPOCHS} | "
          f"Train MAE: {train_mae_years:.3f} | "
          f"Val MAE: {val_mae_years:.3f} | "
          f"LR: {current_lr:.1e} | "
          f"Tempo: {elapsed:.1f}s")

    # Early stopping + salvataggio modello migliore
    if val_mae_years < best_val_mae:
        best_val_mae = val_mae_years
        patience_counter = 0
        torch.save(model.state_dict(), best_model_path)
        print(
            f"  -> Nuovo miglior modello salvato (Val MAE: {best_val_mae:.3f})")
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(
                f"\nEarly stopping all'epoca {epoch} (nessun miglioramento da {EARLY_STOP_PATIENCE} epoche)")
            break

print(f"\n=== Training completato — Best Val MAE: {best_val_mae:.4f} anni ===")

# ──────────────────────────────────────────────
# 5. GRAFICO CURVE DI TRAINING
# ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(history["train_mae"], label="Train MAE", color="#4C72B0", linewidth=2)
ax.plot(history["val_mae"],   label="Validation MAE",
        color="#DD4444", linewidth=2)
ax.axhline(best_val_mae, color="gray", linestyle="--",
           alpha=0.7, label=f"Best Val MAE = {best_val_mae:.2f}")
ax.set_title("CNN 3D — Curve di apprendimento")
ax.set_xlabel("Epoca")
ax.set_ylabel("MAE (anni)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "cnn3d_training_curves.png"), dpi=150)
plt.show()
print("[OK] Salvato: cnn3d_training_curves.png")

print("\n=== Script completato ===")
