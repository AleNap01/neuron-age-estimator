"""
classical.py — Pipeline di Machine Learning classico per Brain Age Estimation.

Definisce le pipeline (StandardScaler + PCA + modello) per i diversi
algoritmi testati, e una funzione di utilità per costruire l'ensemble
finale (SVR + Random Forest) risultato il migliore in validazione.
"""

from __future__ import annotations

from sklearn.decomposition import PCA
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from brain_age.config import RANDOM_STATE


def build_ridge_pipeline(n_components: int = 30, alpha: float = 10.0) -> Pipeline:
    """Baseline lineare con regolarizzazione L2."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=RANDOM_STATE)),
        ("model", Ridge(alpha=alpha)),
    ])


def build_random_forest_pipeline(
    n_estimators: int = 500,
    max_depth: int | None = 20,
    min_samples_leaf: int = 1,
) -> Pipeline:
    """Ensemble di alberi di decisione, robusto a feature correlate."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )),
    ])


def build_svr_pipeline(
    n_components: int = 70,
    C: float = 200.0,
    epsilon: float = 1.0,
    gamma: str = "scale",
) -> Pipeline:
    """SVR con kernel RBF — il modello classico più performante nei test."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=RANDOM_STATE)),
        ("model", SVR(kernel="rbf", C=C, epsilon=epsilon, gamma=gamma)),
    ])


def build_gradient_boosting_pipeline(
    n_estimators: int = 300,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
) -> Pipeline:
    """Ensemble sequenziale di alberi (boosting)."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            random_state=RANDOM_STATE,
        )),
    ])


def build_ensemble(svr_pipeline: Pipeline, rf_pipeline: Pipeline,
                    weights: tuple[float, float] = (0.6, 0.4)) -> VotingRegressor:
    """
    Combina SVR e Random Forest tramite media pesata.

    Pesi di default (0.6, 0.4) favoriscono SVR, storicamente il modello
    classico più accurato nei test di cross-validation.
    """
    return VotingRegressor(
        [("svr", svr_pipeline), ("rf", rf_pipeline)],
        weights=list(weights),
    )
