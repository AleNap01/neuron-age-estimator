"""
01_eda.py — Exploratory Data Analysis
Brain Age Estimation - UniNa ML 25/26
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
import os

# ──────────────────────────────────────────────
# 0. PATH
# ──────────────────────────────────────────────
BASE = r"C:\Users\napol\Desktop\uni-na-ml-25-26-project-brain-age-estimation"
DATA_DIR = os.path.join(BASE, "data", "data")

# Il CSV contiene path tipo "data/train/subj_2183.nii.gz"
# La struttura reale è:  DATA_DIR\train\train\subj_2183.nii.gz


def fix_path(p):
    parts = p.replace("/", os.sep).split(os.sep)
    split = parts[1]   # "train" o "test"
    fname = parts[2].replace(".nii.gz", ".nii")  # rimuove .gz
    return os.path.join(DATA_DIR, split, split, fname)


train_df = pd.read_csv(os.path.join(BASE, "train.csv"))
test_df = pd.read_csv(os.path.join(BASE, "test.csv"))

print("=" * 50)
print(f"Train samples : {len(train_df)}")
print(f"Test  samples : {len(test_df)}")
print("\nTrain head:")
print(train_df.head())
print("\nStatistiche AGE:")
print(train_df["AGE"].describe())

# ──────────────────────────────────────────────
# 1. DISTRIBUZIONE ETÀ
# ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(train_df["AGE"], bins=40, color="#4C72B0", edgecolor="white")
axes[0].axvline(train_df["AGE"].mean(), color="red",
                linestyle="--", label=f'Mean={train_df["AGE"].mean():.1f}')
axes[0].axvline(train_df["AGE"].median(), color="orange",
                linestyle="--", label=f'Median={train_df["AGE"].median():.1f}')
axes[0].set_title("Distribuzione età - Train set")
axes[0].set_xlabel("Età (anni)")
axes[0].set_ylabel("Conteggio")
axes[0].legend()

axes[1].boxplot(train_df["AGE"], patch_artist=True,
                boxprops=dict(facecolor="#4C72B0", alpha=0.7))
axes[1].set_title("Boxplot età - Train set")
axes[1].set_ylabel("Età (anni)")

plt.tight_layout()
plt.savefig(os.path.join(BASE, "eda_age_distribution.png"), dpi=150)
plt.show()
print("[OK] Salvato: eda_age_distribution.png")

# ──────────────────────────────────────────────
# 2. FASCE D'ETÀ
# ──────────────────────────────────────────────
bins_age = [0, 12, 18, 30, 45, 60, 100]
labels = ["0-12", "13-18", "19-30", "31-45", "46-60", "60+"]
train_df["age_group"] = pd.cut(train_df["AGE"], bins=bins_age, labels=labels)

counts = train_df["age_group"].value_counts().sort_index()
print("\nFasce d'età nel training set:")
print(counts.to_string())

fig, ax = plt.subplots(figsize=(8, 5))
counts.plot(kind="bar", ax=ax, color="#4C72B0", edgecolor="white")
ax.set_title("Soggetti per fascia d'età")
ax.set_xlabel("Fascia d'età")
ax.set_ylabel("Conteggio")
ax.tick_params(axis='x', rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(BASE, "eda_age_groups.png"), dpi=150)
plt.show()
print("[OK] Salvato: eda_age_groups.png")

# ──────────────────────────────────────────────
# 3. ESPLORA UNA MRI DI ESEMPIO
# ──────────────────────────────────────────────
sample_path = fix_path(train_df["path"].iloc[0])
print(f"\nCarico MRI di esempio: {sample_path}")
print(f"File esiste: {os.path.exists(sample_path)}")

img = nib.load(sample_path)
data = np.asarray(img.dataobj, dtype=np.float32)

print(f"Shape volume  : {data.shape}")
print(f"Voxel min/max : {data.min():.4f} / {data.max():.4f}")
print(f"Voxel mean    : {data.mean():.4f}")
print(f"Voxel std     : {data.std():.4f}")
print(f"Voxel size    : {img.header.get_zooms()}")

cx, cy, cz = data.shape[0]//2, data.shape[1]//2, data.shape[2]//2

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(data[cx, :, :], cmap="gray", origin="lower")
axes[0].set_title(f"Slice assiale  (x={cx})")
axes[1].imshow(data[:, cy, :], cmap="gray", origin="lower")
axes[1].set_title(f"Slice coronale (y={cy})")
axes[2].imshow(data[:, :, cz], cmap="gray", origin="lower")
axes[2].set_title(f"Slice sagittale (z={cz})")

for ax in axes:
    ax.axis("off")

sogg = train_df["anon_id"].iloc[0]
eta = train_df["AGE"].iloc[0]
fig.suptitle(f"MRI esempio — {sogg}, età {eta} anni", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(BASE, "eda_mri_slices.png"), dpi=150)
plt.show()
print("[OK] Salvato: eda_mri_slices.png")

# ──────────────────────────────────────────────
# 4. VERIFICA SHAPE SU 10 VOLUMI
# ──────────────────────────────────────────────
print("\nVerifica shape su 10 soggetti casuali...")
shapes = []
for _, row in train_df.sample(10, random_state=42).iterrows():
    p = fix_path(row["path"])
    s = nib.load(p).shape
    shapes.append(s)
    print(f"  {row['anon_id']:12s}  età={row['AGE']:5.1f}  shape={s}")

unique_shapes = set(shapes)
print(f"\nShape uniche trovate: {unique_shapes}")
if len(unique_shapes) == 1:
    print("[OK] Tutti i volumi hanno la stessa shape — nessun resize necessario.")
else:
    print("[ATTENZIONE] Shape diverse — potrebbe servire un resize/padding.")

print("\n=== EDA completata ===")
