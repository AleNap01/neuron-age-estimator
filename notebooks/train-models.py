"""
03_train_models.py — Training e valutazione modelli ML
Brain Age Estimation - UniNa ML 25/26

Modelli:
  1. Ridge Regression (baseline lineare)
  2. Random Forest Regressor
  3. SVR (Support Vector Regression)

Valutazione: 5-fold Cross Validation con MAE
"""

import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error
import joblib

# ──────────────────────────────────────────────
# 0. PATH E CARICAMENTO DATI
# ──────────────────────────────────────────────
BASE = r"C:\Users\napol\Desktop\uni-na-ml-25-26-project-brain-age-estimation"

X_train = np.load(os.path.join(BASE, "features_train.npy"))
X_test = np.load(os.path.join(BASE, "features_test.npy"))
y_train = np.load(os.path.join(BASE, "labels_train.npy"))

print("=" * 50)
print(f"X_train : {X_train.shape}")
print(f"X_test  : {X_test.shape}")
print(f"y_train : {y_train.shape}")
print(f"Age range: {y_train.min():.1f} - {y_train.max():.1f} anni")

# ──────────────────────────────────────────────
# 1. DEFINIZIONE MODELLI (con pipeline Scaler + PCA + Model)
# ──────────────────────────────────────────────
models = {
    "Ridge": Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=30)),
        ("model",  Ridge(alpha=10.0)),
    ]),

    "RandomForest": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        )),
    ]),

    "SVR": Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=30)),
        ("model",  SVR(kernel="rbf", C=100, gamma="scale", epsilon=0.5)),
    ]),

    "GradientBoosting": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  GradientBoostingRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )),
    ]),
}

# ──────────────────────────────────────────────
# 2. CROSS VALIDATION (5-fold)
# ──────────────────────────────────────────────
kf = KFold(n_splits=5, shuffle=True, random_state=42)

results = {}
print("\n" + "=" * 50)
print("Cross Validation (5-fold) — MAE per modello")
print("=" * 50)

for name, pipeline in models.items():
    print(f"\n[{name}] Training...")
    t0 = time.time()

    scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=kf,
        scoring="neg_mean_absolute_error",
        n_jobs=-1 if name != "GradientBoosting" else 1,
    )
    mae_scores = -scores
    elapsed = time.time() - t0

    results[name] = {
        "mae_mean": mae_scores.mean(),
        "mae_std":  mae_scores.std(),
        "mae_folds": mae_scores,
    }

    print(
        f"  MAE: {mae_scores.mean():.4f} ± {mae_scores.std():.4f}  ({elapsed:.1f}s)")

# ──────────────────────────────────────────────
# 3. RIEPILOGO E GRAFICO
# ──────────────────────────────────────────────
print("\n" + "=" * 50)
print("RIEPILOGO RISULTATI")
print("=" * 50)
print(f"{'Modello':<20} {'MAE Mean':>10} {'MAE Std':>10}")
print("-" * 42)

best_name = min(results, key=lambda k: results[k]["mae_mean"])
for name, res in sorted(results.items(), key=lambda x: x[1]["mae_mean"]):
    marker = " ← BEST" if name == best_name else ""
    print(
        f"{name:<20} {res['mae_mean']:>10.4f} {res['mae_std']:>10.4f}{marker}")

# Grafico confronto
fig, ax = plt.subplots(figsize=(10, 5))
names = list(results.keys())
means = [results[n]["mae_mean"] for n in names]
stds = [results[n]["mae_std"] for n in names]
colors = ["#4C72B0" if n != best_name else "#DD4444" for n in names]

bars = ax.bar(names, means, yerr=stds, capsize=6,
              color=colors, edgecolor="white", alpha=0.85)
ax.set_title("Confronto MAE (5-fold CV) — Brain Age Estimation")
ax.set_ylabel("MAE (anni)")
ax.set_xlabel("Modello")
for bar, mean in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f"{mean:.2f}", ha="center", va="bottom", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(BASE, "model_comparison.png"), dpi=150)
plt.show()
print("[OK] Salvato: model_comparison.png")

# ──────────────────────────────────────────────
# 4. TRAIN BEST MODEL SU TUTTO IL TRAIN SET
# ──────────────────────────────────────────────
print(f"\nTraining {best_name} su tutto il training set...")
best_pipeline = models[best_name]
best_pipeline.fit(X_train, y_train)

# Salva il modello
model_path = os.path.join(BASE, f"best_model_{best_name}.pkl")
joblib.dump(best_pipeline, model_path)
print(f"[OK] Modello salvato: best_model_{best_name}.pkl")

# MAE sul training set (overfitting check)
y_pred_train = best_pipeline.predict(X_train)
train_mae = mean_absolute_error(y_train, y_pred_train)
print(f"MAE su train set (overfitting check): {train_mae:.4f}")

# ──────────────────────────────────────────────
# 5. PREDIZIONI SUL TEST SET
# ──────────────────────────────────────────────
print(f"\nGenerazione predizioni test set...")
test_df = pd.read_csv(os.path.join(BASE, "test.csv"))

y_pred_test = best_pipeline.predict(X_test)

submission = pd.DataFrame({
    "anon_id": test_df["anon_id"],
    "AGE":     np.round(y_pred_test, 2),
})

submission_path = os.path.join(BASE, f"submission_{best_name}.csv")
submission.to_csv(submission_path, index=False)
print(f"[OK] Submission salvata: submission_{best_name}.csv")
print(submission.head(10))

# Distribuzione predizioni
fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(y_train, bins=40, alpha=0.6,
        label="Train (ground truth)", color="#4C72B0")
ax.hist(y_pred_test, bins=40, alpha=0.6,
        label="Test (predizioni)", color="#DD4444")
ax.set_title("Distribuzione età: train vs predizioni test")
ax.set_xlabel("Età (anni)")
ax.set_ylabel("Conteggio")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(BASE, "pred_distribution.png"), dpi=150)
plt.show()
print("[OK] Salvato: pred_distribution.png")

print("\n=== Training completato ===")
