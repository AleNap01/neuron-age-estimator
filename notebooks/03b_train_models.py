"""
03b_train_models_v2.py — Training con feature v2 (155) + tuning iperparametri
Brain Age Estimation - Progetto Enterprise
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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error
import joblib

# ──────────────────────────────────────────────
# 0. PATH E DATI
# ──────────────────────────────────────────────
BASE = r"C:\Users\napol\Desktop\Archivio Generale\10--Work and projects\Neuron_Age_Estimator"
OUT_DIR = os.path.join(BASE, "outputs")

X_train = np.load(os.path.join(OUT_DIR, "features_train_v2.npy"))
X_test = np.load(os.path.join(OUT_DIR, "features_test_v2.npy"))
y_train = np.load(os.path.join(OUT_DIR, "labels_train_v2.npy"))

print("=" * 50)
print(f"X_train : {X_train.shape}")
print(f"X_test  : {X_test.shape}")
print(f"y_train : {y_train.shape}")

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# ──────────────────────────────────────────────
# 1. GRIDSEARCH SU SVR (modello migliore in v1)
# ──────────────────────────────────────────────
print("\n" + "=" * 50)
print("GridSearch su SVR...")
t0 = time.time()

svr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca",    PCA()),
    ("model",  SVR(kernel="rbf")),
])

svr_param_grid = {
    "pca__n_components": [30, 50, 70],
    "model__C":       [10, 50, 100, 200],
    "model__epsilon":  [0.2, 0.5, 1.0],
    "model__gamma":   ["scale", "auto"],
}

svr_search = GridSearchCV(
    svr_pipeline, svr_param_grid,
    cv=kf, scoring="neg_mean_absolute_error",
    n_jobs=-1, verbose=1,
)
svr_search.fit(X_train, y_train)

print(f"\n[SVR] Tempo: {time.time()-t0:.1f}s")
print(f"[SVR] Best params: {svr_search.best_params_}")
print(f"[SVR] Best MAE (CV): {-svr_search.best_score_:.4f}")

# ──────────────────────────────────────────────
# 2. GRIDSEARCH SU RANDOM FOREST
# ──────────────────────────────────────────────
print("\n" + "=" * 50)
print("GridSearch su Random Forest...")
t0 = time.time()

rf_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  RandomForestRegressor(n_jobs=-1, random_state=42)),
])

rf_param_grid = {
    "model__n_estimators":    [300, 500],
    "model__max_depth":       [10, 20, None],
    "model__min_samples_leaf": [1, 2, 4],
}

rf_search = GridSearchCV(
    rf_pipeline, rf_param_grid,
    cv=kf, scoring="neg_mean_absolute_error",
    n_jobs=-1, verbose=1,
)
rf_search.fit(X_train, y_train)

print(f"\n[RF] Tempo: {time.time()-t0:.1f}s")
print(f"[RF] Best params: {rf_search.best_params_}")
print(f"[RF] Best MAE (CV): {-rf_search.best_score_:.4f}")

# ──────────────────────────────────────────────
# 3. ENSEMBLE (VotingRegressor) — media pesata SVR + RF
# ──────────────────────────────────────────────
print("\n" + "=" * 50)
print("Costruzione Ensemble (SVR + RandomForest)...")

ensemble = VotingRegressor([
    ("svr", svr_search.best_estimator_),
    ("rf",  rf_search.best_estimator_),
], weights=[0.6, 0.4])  # più peso a SVR, che storicamente è il migliore

ensemble_scores = cross_val_score(
    ensemble, X_train, y_train,
    cv=kf, scoring="neg_mean_absolute_error", n_jobs=-1,
)
ensemble_mae = -ensemble_scores.mean()
print(f"[Ensemble] MAE (CV): {ensemble_mae:.4f} ± {ensemble_scores.std():.4f}")

# ──────────────────────────────────────────────
# 4. CONFRONTO FINALE E SCELTA MODELLO MIGLIORE
# ──────────────────────────────────────────────
results = {
    "SVR (tuned)": -svr_search.best_score_,
    "RandomForest (tuned)": -rf_search.best_score_,
    "Ensemble":              ensemble_mae,
}

print("\n" + "=" * 50)
print("RIEPILOGO FINALE")
print("=" * 50)
for name, mae in sorted(results.items(), key=lambda x: x[1]):
    print(f"  {name:<25} MAE = {mae:.4f}")

best_name = min(results, key=results.get)
print(f"\nMiglior modello: {best_name} (MAE={results[best_name]:.4f})")

# Grafico confronto v1 vs v2
fig, ax = plt.subplots(figsize=(10, 5))
names = list(results.keys())
values = list(results.values())
colors = ["#DD4444" if n == best_name else "#4C72B0" for n in names]
bars = ax.bar(names, values, color=colors, edgecolor="white", alpha=0.85)
ax.axhline(5.38, color="gray", linestyle="--",
           label="v1 baseline (SVR, MAE=5.38)")
ax.set_title("Confronto modelli v2 (feature avanzate + tuning)")
ax.set_ylabel("MAE (anni)")
for bar, val in zip(bars, values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1, f"{val:.2f}",
            ha="center", va="bottom")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "model_comparison_v2.png"), dpi=150)
plt.show()
print("[OK] Salvato: model_comparison_v2.png")

# ──────────────────────────────────────────────
# 5. TRAINING FINALE SU TUTTO IL TRAIN SET
# ──────────────────────────────────────────────
print(f"\nTraining finale: {best_name}...")

if best_name == "Ensemble":
    final_model = ensemble
elif best_name == "SVR (tuned)":
    final_model = svr_search.best_estimator_
else:
    final_model = rf_search.best_estimator_

final_model.fit(X_train, y_train)

y_pred_train = final_model.predict(X_train)
train_mae = mean_absolute_error(y_train, y_pred_train)
print(f"MAE su train set: {train_mae:.4f}")

joblib.dump(final_model, os.path.join(OUT_DIR, "best_model_v2.pkl"))
print("[OK] Modello salvato: best_model_v2.pkl")

# ──────────────────────────────────────────────
# 6. SUBMISSION
# ──────────────────────────────────────────────
test_df = pd.read_csv(os.path.join(BASE, "test.csv"))
y_pred_test = final_model.predict(X_test)

submission = pd.DataFrame({
    "anon_id": test_df["anon_id"],
    "AGE":     np.round(y_pred_test, 2),
})
submission.to_csv(os.path.join(OUT_DIR, "submission_v2.csv"), index=False)
print("[OK] Submission salvata: submission_v2.csv")
print(submission.head(10))

print("\n=== Training v2 completato ===")
