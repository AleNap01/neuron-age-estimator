"""
config.py — Configurazione centrale del progetto.

Tutti i path e le costanti condivise vivono qui, in modo che il resto del
codice non contenga mai un path scritto a mano (niente più
"C:\\Users\\napol\\..." sparsi in dieci file diversi).

Il path di base si può sovrascrivere con la variabile d'ambiente
BRAIN_AGE_HOME, utile per eseguire il progetto su un'altra macchina
(es. Kaggle, server) senza modificare il codice.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────
# PATH DI BASE
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(
    os.environ.get("BRAIN_AGE_HOME", Path(__file__).resolve().parents[2])
)

DATA_DIR    = PROJECT_ROOT / "data"
OUTPUT_DIR  = PROJECT_ROOT / "outputs"
TRAIN_CSV   = PROJECT_ROOT / "train.csv"
TEST_CSV    = PROJECT_ROOT / "test.csv"
SAMPLE_SUB  = PROJECT_ROOT / "sampleSubmission.csv"

# Sottocartelle MRI — la struttura reale ha un doppio livello (es. train/train)
# dovuto all'estrazione dello zip Kaggle; lo gestiamo qui in un unico punto.
MRI_TRAIN_DIR = DATA_DIR / "train" / "train"
MRI_TEST_DIR  = DATA_DIR / "test" / "test"

# Assicura che la cartella di output esista
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# COSTANTI DATASET
# ──────────────────────────────────────────────
ORIGINAL_VOLUME_SIZE = 128   # dimensione originale dei volumi MRI (128^3)
CNN_VOLUME_SIZE       = 64   # dimensione dopo downsampling per la CNN 3D

RANDOM_STATE = 42            # seed unico per riproducibilità in tutto il progetto

# ──────────────────────────────────────────────
# FILE GENERATI (feature, modelli, predizioni)
# ──────────────────────────────────────────────
# Feature extraction v1 (60 feature - usata per la submission d'esame)
FEATURES_TRAIN_V1 = OUTPUT_DIR / "features_train.npy"
FEATURES_TEST_V1  = OUTPUT_DIR / "features_test.npy"
LABELS_TRAIN_V1   = OUTPUT_DIR / "labels_train.npy"

# Feature extraction v2 (155 feature - versione migliorata)
FEATURES_TRAIN_V2 = OUTPUT_DIR / "features_train_v2.npy"
FEATURES_TEST_V2  = OUTPUT_DIR / "features_test_v2.npy"
LABELS_TRAIN_V2   = OUTPUT_DIR / "labels_train_v2.npy"

# Volumi downsampled per la CNN 3D
VOLUMES_TRAIN_CNN = OUTPUT_DIR / f"volumes_train_{CNN_VOLUME_SIZE}.npy"
VOLUMES_TEST_CNN  = OUTPUT_DIR / f"volumes_test_{CNN_VOLUME_SIZE}.npy"
AGES_TRAIN_CNN    = OUTPUT_DIR / f"ages_train_{CNN_VOLUME_SIZE}.npy"

# Modelli salvati
MODEL_V1_PATH      = OUTPUT_DIR / "best_model_SVR.pkl"
MODEL_V2_PATH      = OUTPUT_DIR / "best_model_v2.pkl"
CNN_MODEL_PATH     = OUTPUT_DIR / "best_cnn3d.pth"

# Submission
SUBMISSION_V1_PATH = OUTPUT_DIR / "submission_SVR.csv"
SUBMISSION_V2_PATH = OUTPUT_DIR / "submission_v2.csv"
