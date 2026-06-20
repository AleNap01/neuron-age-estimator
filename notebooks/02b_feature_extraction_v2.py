"""
02b_feature_extraction_v2.py — Feature Extraction MIGLIORATA (v2)
Brain Age Estimation - Progetto Enterprise

Rispetto a v1 (60 feature), aggiungiamo:
- Istogramma di intensità (10 bin) — forma della distribuzione voxel
- Gradiente spaziale (Sobel 3D) — cattura bordi/texture, utile per materia grigia/bianca
- 6 fasce assiali invece di 3 — risoluzione spaziale maggiore
- Rapporto centro/periferia — cattura espansione ventricoli con l'età
- Feature sui 3 piani centrali (assiale, coronale, sagittale) separatamente

Totale: ~150 feature per soggetto
"""

import numpy as np
import pandas as pd
import nibabel as nib
import os
from scipy import stats, ndimage
from tqdm import tqdm

# ──────────────────────────────────────────────
# 0. PATH
# ──────────────────────────────────────────────
BASE = r"C:\Users\napol\Desktop\brain_age_enterprise"
DATA_DIR = os.path.join(BASE, "data")
OUT_DIR = os.path.join(BASE, "outputs")


def fix_path(p):
    parts = p.replace("/", os.sep).split(os.sep)
    split = parts[1]
    fname = parts[2].replace(".nii.gz", ".nii")
    return os.path.join(DATA_DIR, split, split, fname)


train_df = pd.read_csv(os.path.join(BASE, "train.csv"))
test_df = pd.read_csv(os.path.join(BASE, "test.csv"))

# ──────────────────────────────────────────────
# 1. FUNZIONI DI SUPPORTO
# ──────────────────────────────────────────────


def basic_stats(arr, prefix_list):
    """Statistiche base su un array 1D di voxel (esclude background)."""
    b = arr[arr > 0.01]
    if len(b) == 0:
        return [0.0] * len(prefix_list)
    funcs = {
        "mean": b.mean(), "std": b.std(),
        "p10": np.percentile(b, 10), "p25": np.percentile(b, 25),
        "p50": np.percentile(b, 50), "p75": np.percentile(b, 75),
        "p90": np.percentile(b, 90),
        "skew": stats.skew(b), "kurt": stats.kurtosis(b),
    }
    return [funcs[k] for k in prefix_list]


STAT_KEYS = ["mean", "std", "p10", "p25", "p50", "p75", "p90", "skew", "kurt"]


def extract_features_v2(volume):
    feats = []
    brain = volume[volume > 0.01]

    # --- 1. Statistiche globali (9) ---
    feats.extend(basic_stats(volume, STAT_KEYS))
    feats.append(brain.max())
    feats.append(brain.min())
    feats.append(len(brain) / volume.size)  # volume cerebrale relativo

    # --- 2. Istogramma di intensità, 10 bin (10) ---
    hist, _ = np.histogram(brain, bins=10, range=(0, 1), density=True)
    feats.extend(hist.tolist())

    # --- 3. Gradiente spaziale (Sobel 3D) — texture/bordi (9) ---
    gx = ndimage.sobel(volume, axis=0)
    gy = ndimage.sobel(volume, axis=1)
    gz = ndimage.sobel(volume, axis=2)
    grad_mag = np.sqrt(gx**2 + gy**2 + gz**2)
    grad_brain = grad_mag[volume > 0.01]
    if len(grad_brain) > 0:
        feats.append(grad_brain.mean())
        feats.append(grad_brain.std())
        feats.append(np.percentile(grad_brain, 90))
    else:
        feats.extend([0, 0, 0])

    # --- 4. 8 ottanti (9 stat * 8 = 72) ---
    h = volume.shape[0] // 2
    octants = [
        volume[:h, :h, :h], volume[:h, :h, h:],
        volume[:h, h:, :h], volume[:h, h:, h:],
        volume[h:, :h, :h], volume[h:, :h, h:],
        volume[h:, h:, :h], volume[h:, h:, h:],
    ]
    for oct_vol in octants:
        feats.extend(basic_stats(oct_vol, STAT_KEYS))

    # --- 5. 6 fasce assiali (9 stat * 6 = 54) ---
    t = volume.shape[0] // 6
    for i in range(6):
        slab = volume[i*t:(i+1)*t, :, :]
        feats.extend(basic_stats(slab, STAT_KEYS))

    # --- 6. Centro vs periferia — proxy ventricoli/atrofia (4) ---
    c = volume.shape[0] // 4
    center = volume[c:3*c, c:3*c, c:3*c]
    center_b = center[center > 0.01]
    periph_b = brain  # tutto il cervello come riferimento periferia+centro

    if len(center_b) > 0:
        feats.append(center_b.mean())
        feats.append(center_b.std())
        # densità cervello al centro
        feats.append(len(center_b) / center.size)
        # rapporto centro/globale
        feats.append(center_b.mean() / (brain.mean() + 1e-6))
    else:
        feats.extend([0, 0, 0, 0])

    return np.array(feats, dtype=np.float32)


# ──────────────────────────────────────────────
# 2. ESTRAZIONE TRAIN
# ──────────────────────────────────────────────
print("=" * 50)
print(f"Estrazione feature v2 TRAIN ({len(train_df)} soggetti)...")

X_train, y_train = [], []
for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Train"):
    vol = np.asarray(nib.load(fix_path(row["path"])).dataobj, dtype=np.float32)
    X_train.append(extract_features_v2(vol))
    y_train.append(row["AGE"])

X_train = np.array(X_train)
y_train = np.array(y_train, dtype=np.float32)
print(
    f"\nX_train shape: {X_train.shape}  (feature per soggetto: {X_train.shape[1]})")

# ──────────────────────────────────────────────
# 3. ESTRAZIONE TEST
# ──────────────────────────────────────────────
print(f"\nEstrazione feature v2 TEST ({len(test_df)} soggetti)...")

X_test = []
for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Test"):
    vol = np.asarray(nib.load(fix_path(row["path"])).dataobj, dtype=np.float32)
    X_test.append(extract_features_v2(vol))

X_test = np.array(X_test)
print(f"X_test shape: {X_test.shape}")

# ──────────────────────────────────────────────
# 4. SALVA
# ──────────────────────────────────────────────
np.save(os.path.join(OUT_DIR, "features_train_v2.npy"), X_train)
np.save(os.path.join(OUT_DIR, "features_test_v2.npy"),  X_test)
np.save(os.path.join(OUT_DIR, "labels_train_v2.npy"),   y_train)

print("\n[OK] Salvati:")
print(f"  features_train_v2.npy → {X_train.shape}")
print(f"  features_test_v2.npy  → {X_test.shape}")
print(f"  labels_train_v2.npy   → {y_train.shape}")
print("\n=== Feature extraction v2 completata ===")
