"""
train_classical.py — Addestramento e valutazione dei modelli ML classici.

Esegue 5-fold cross-validation su tutte le pipeline definite in
brain_age.models.classical, sceglie il modello con MAE più basso,
lo addestra sull'intero training set e genera la submission sul test set.
"""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score

from brain_age.config import RANDOM_STATE
from brain_age.models.classical import (
    build_ensemble,
    build_gradient_boosting_pipeline,
    build_random_forest_pipeline,
    build_ridge_pipeline,
    build_svr_pipeline,
)


def evaluate_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_splits: int = 5,
    verbose: bool = True,
) -> dict[str, dict]:
    """
    Valuta in cross-validation tutte le pipeline classiche disponibili.

    Returns
    -------
    dict
        Mappa {nome_modello: {"pipeline": ..., "mae_mean": ..., "mae_std": ...}}
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    candidates = {
        "Ridge": build_ridge_pipeline(),
        "RandomForest": build_random_forest_pipeline(),
        "SVR": build_svr_pipeline(),
        "GradientBoosting": build_gradient_boosting_pipeline(),
    }

    results = {}
    for name, pipeline in candidates.items():
        t0 = time.time()
        scores = cross_val_score(
            pipeline, X_train, y_train, cv=kf,
            scoring="neg_mean_absolute_error", n_jobs=-1,
        )
        mae_scores = -scores
        elapsed = time.time() - t0

        results[name] = {
            "pipeline": pipeline,
            "mae_mean": mae_scores.mean(),
            "mae_std": mae_scores.std(),
        }
        if verbose:
            print(f"[{name}] MAE = {mae_scores.mean():.4f} ± {mae_scores.std():.4f} "
                  f"({elapsed:.1f}s)")

    # Ensemble SVR + RandomForest
    t0 = time.time()
    ensemble = build_ensemble(candidates["SVR"], candidates["RandomForest"])
    ensemble_scores = -cross_val_score(
        ensemble, X_train, y_train, cv=kf,
        scoring="neg_mean_absolute_error", n_jobs=-1,
    )
    results["Ensemble"] = {
        "pipeline": ensemble,
        "mae_mean": ensemble_scores.mean(),
        "mae_std": ensemble_scores.std(),
    }
    if verbose:
        print(f"[Ensemble] MAE = {ensemble_scores.mean():.4f} ± "
              f"{ensemble_scores.std():.4f} ({time.time()-t0:.1f}s)")

    return results


def select_best_model(results: dict[str, dict]) -> tuple[str, dict]:
    """Restituisce (nome, info) del modello con MAE medio più basso."""
    best_name = min(results, key=lambda name: results[name]["mae_mean"])
    return best_name, results[best_name]


def train_final_model(
    pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    save_path: Path,
) -> float:
    """Addestra la pipeline sull'intero training set e la salva su disco."""
    pipeline.fit(X_train, y_train)
    train_mae = mean_absolute_error(y_train, pipeline.predict(X_train))
    joblib.dump(pipeline, save_path)
    return train_mae


def generate_submission(
    pipeline,
    X_test: np.ndarray,
    test_df: pd.DataFrame,
    save_path: Path,
) -> pd.DataFrame:
    """Genera il file di submission nel formato richiesto da Kaggle."""
    predictions = pipeline.predict(X_test)
    submission = pd.DataFrame({
        "anon_id": test_df["anon_id"],
        "AGE": np.round(predictions, 2),
    })
    submission.to_csv(save_path, index=False)
    return submission
