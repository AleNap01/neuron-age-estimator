"""
04_downsample_volumes.py — Downsampling dei volumi MRI da 128^3 a 64^3
Brain Age Estimation - Progetto Enterprise (CNN 3D su CPU)

Riduciamo la risoluzione per rendere il training della CNN 3D fattibile
su CPU. Il downsampling avviene una sola volta; i volumi ridotti vengono
salvati in un unico file .npy (memoria-mappabile) per training rapidi.
"""

import numpy as np
import pandas as pd
import nibabel as nib
import os
from scipy.ndimage import zoom
from tqdm import tqdm

# ──────────────────────────────────────────────
# 0. PATH
# ──────────────────────────────────────────────
BASE = r"C:\Users\napol\Desktop\Archivio Generale\10--Work and projects\Neuron_Age_Estimator"
DATA_DIR = os.path.join(BASE, "data")
OUT_DIR = os.path.join(BASE, "outputs")

TARGET_SIZE = 64  # da 128^3 a 64^3


def fix_path(p):
    parts = p.replace("/", os.sep).split(os.sep)
    split = parts[1]
    fname = parts[2].replace(".nii.gz", ".nii")
    # doppio livello train/train
    return os.path.join(DATA_DIR, split, split, fname)


def downsample(volume, target_size=64):
    """Riduce un volume 3D a target_size^3 con interpolazione lineare."""
    factor = target_size / volume.shape[0]
    # order=1 = interpolazione lineare (rapida)
    return zoom(volume, zoom=factor, order=1)


# ──────────────────────────────────────────────
# 1. CARICA CSV
# ──────────────────────────────────────────────
train_df = pd.read_csv(os.path.join(BASE, "train.csv"))
test_df = pd.read_csv(os.path.join(BASE, "test.csv"))

print("=" * 50)
print(f"Downsampling volumi: 128^3 -> {TARGET_SIZE}^3")
print(f"Train: {len(train_df)} soggetti | Test: {len(test_df)} soggetti")

# ──────────────────────────────────────────────
# 2. DOWNSAMPLE TRAIN SET
# ──────────────────────────────────────────────
print("\nProcesso TRAIN set...")
volumes_train = np.zeros(
    (len(train_df), TARGET_SIZE, TARGET_SIZE, TARGET_SIZE), dtype=np.float32)
ages_train = np.zeros(len(train_df), dtype=np.float32)

for i, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Train"):
    vol = np.asarray(nib.load(fix_path(row["path"])).dataobj, dtype=np.float32)
    volumes_train[i] = downsample(vol, TARGET_SIZE)
    ages_train[i] = row["AGE"]

print(f"volumes_train shape: {volumes_train.shape}")
print(f"Memoria occupata: {volumes_train.nbytes / 1e9:.2f} GB")

# ──────────────────────────────────────────────
# 3. DOWNSAMPLE TEST SET
# ──────────────────────────────────────────────
print("\nProcesso TEST set...")
volumes_test = np.zeros(
    (len(test_df), TARGET_SIZE, TARGET_SIZE, TARGET_SIZE), dtype=np.float32)

for i, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Test"):
    vol = np.asarray(nib.load(fix_path(row["path"])).dataobj, dtype=np.float32)
    volumes_test[i] = downsample(vol, TARGET_SIZE)

print(f"volumes_test shape: {volumes_test.shape}")

# ──────────────────────────────────────────────
# 4. SALVA SU DISCO
# ──────────────────────────────────────────────
np.save(os.path.join(
    OUT_DIR, f"volumes_train_{TARGET_SIZE}.npy"), volumes_train)
np.save(os.path.join(
    OUT_DIR, f"volumes_test_{TARGET_SIZE}.npy"),  volumes_test)
np.save(os.path.join(OUT_DIR, f"ages_train_{TARGET_SIZE}.npy"),    ages_train)

print("\n[OK] Salvati:")
print(
    f"  volumes_train_{TARGET_SIZE}.npy → {volumes_train.shape}  ({volumes_train.nbytes/1e9:.2f} GB)")
print(
    f"  volumes_test_{TARGET_SIZE}.npy  → {volumes_test.shape}  ({volumes_test.nbytes/1e9:.2f} GB)")
print(f"  ages_train_{TARGET_SIZE}.npy    → {ages_train.shape}")
print("\n=== Downsampling completato ===")
