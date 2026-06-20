"""
02_feature_extraction.py — Feature Extraction da volumi MRI 3D
Brain Age Estimation - UniNa ML 25/26

Per ogni volume (128x128x128) estraiamo:
- Statistiche globali (mean, std, percentili, skewness, kurtosis)
- Statistiche per ottanti (8 sotto-regioni del volume)
- Statistiche per slice assiale (layer superiore, medio, inferiore)
Totale: ~130 feature per soggetto
"""

import numpy as np
import pandas as pd
import nibabel as nib
import os
from scipy import stats
from tqdm import tqdm

# ──────────────────────────────────────────────
# 0. PATH
# ──────────────────────────────────────────────
BASE = r"C:\Users\napol\Desktop\uni-na-ml-25-26-project-brain-age-estimation"
DATA_DIR = os.path.join(BASE, "data", "data")
OUT_DIR = BASE


def fix_path(p):
    parts = p.replace("/", os.sep).split(os.sep)
    split = parts[1]
    fname = parts[2].replace(".nii.gz", ".nii")
    return os.path.join(DATA_DIR, split, split, fname)


train_df = pd.read_csv(os.path.join(BASE, "train.csv"))
test_df = pd.read_csv(os.path.join(BASE, "test.csv"))

# ──────────────────────────────────────────────
# 1. FUNZIONE DI ESTRAZIONE FEATURE
# ──────────────────────────────────────────────


def extract_features(volume):
    """
    Estrae feature statistiche da un volume MRI 3D (128x128x128).
    Considera solo i voxel non-zero (cervello, non background).
    """
    feats = []

    # --- Feature globali (su tutto il volume) ---
    brain = volume[volume > 0.01]  # maschera cervello (escludi background)

    feats.append(brain.mean())
    feats.append(brain.std())
    feats.append(np.percentile(brain, 10))
    feats.append(np.percentile(brain, 25))
    feats.append(np.percentile(brain, 50))
    feats.append(np.percentile(brain, 75))
    feats.append(np.percentile(brain, 90))
    feats.append(np.percentile(brain, 95))
    feats.append(brain.max())
    feats.append(brain.min())
    feats.append(stats.skew(brain))
    feats.append(stats.kurtosis(brain))
    # frazione di voxel cerebrali (volume cerebrale relativo)
    feats.append(len(brain) / volume.size)

    # --- Feature per 8 ottanti (dividi il volume in 8 cubi 64x64x64) ---
    h = volume.shape[0] // 2
    octants = [
        volume[:h, :h, :h],
        volume[:h, :h, h:],
        volume[:h, h:, :h],
        volume[:h, h:, h:],
        volume[h:, :h, :h],
        volume[h:, :h, h:],
        volume[h:, h:, :h],
        volume[h:, h:, h:],
    ]
    for oct in octants:
        b = oct[oct > 0.01]
        if len(b) > 0:
            feats.append(b.mean())
            feats.append(b.std())
            feats.append(np.percentile(b, 25))
            feats.append(np.percentile(b, 75))
        else:
            feats.extend([0, 0, 0, 0])

    # --- Feature per 3 fasce assiali (inferiore / medio / superiore) ---
    t = volume.shape[0] // 3
    slabs = [
        volume[:t, :, :],
        volume[t:2*t, :, :],
        volume[2*t:, :, :],
    ]
    for slab in slabs:
        b = slab[slab > 0.01]
        if len(b) > 0:
            feats.append(b.mean())
            feats.append(b.std())
            feats.append(np.percentile(b, 25))
            feats.append(np.percentile(b, 75))
            feats.append(stats.skew(b))
        else:
            feats.extend([0, 0, 0, 0, 0])

    return np.array(feats, dtype=np.float32)


# ──────────────────────────────────────────────
# 2. ESTRAI FEATURE DAL TRAIN SET
# ──────────────────────────────────────────────
print("=" * 50)
print(f"Estrazione feature TRAIN ({len(train_df)} soggetti)...")
print("Potrebbe richiedere qualche minuto...")

X_train = []
y_train = []

for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Train"):
    path = fix_path(row["path"])
    vol = np.asarray(nib.load(path).dataobj, dtype=np.float32)
    feat = extract_features(vol)
    X_train.append(feat)
    y_train.append(row["AGE"])

X_train = np.array(X_train)
y_train = np.array(y_train, dtype=np.float32)

print(f"\nX_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"Feature estratte per soggetto: {X_train.shape[1]}")

# ──────────────────────────────────────────────
# 3. ESTRAI FEATURE DAL TEST SET
# ──────────────────────────────────────────────
print(f"\nEstrazione feature TEST ({len(test_df)} soggetti)...")

X_test = []

for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Test"):
    path = fix_path(row["path"])
    vol = np.asarray(nib.load(path).dataobj, dtype=np.float32)
    feat = extract_features(vol)
    X_test.append(feat)

X_test = np.array(X_test)
print(f"X_test shape: {X_test.shape}")

# ──────────────────────────────────────────────
# 4. SALVA SU DISCO
# ──────────────────────────────────────────────
np.save(os.path.join(OUT_DIR, "features_train.npy"), X_train)
np.save(os.path.join(OUT_DIR, "features_test.npy"),  X_test)
np.save(os.path.join(OUT_DIR, "labels_train.npy"),   y_train)

print("\n[OK] Salvati:")
print(f"  features_train.npy  → {X_train.shape}")
print(f"  features_test.npy   → {X_test.shape}")
print(f"  labels_train.npy    → {y_train.shape}")
print("\n=== Feature extraction completata ===")
