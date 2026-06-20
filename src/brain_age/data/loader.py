"""
loader.py — Funzioni di caricamento dati: CSV e volumi MRI (NIfTI).
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from brain_age.config import MRI_TEST_DIR, MRI_TRAIN_DIR, TEST_CSV, TRAIN_CSV


def load_train_df() -> pd.DataFrame:
    """Carica train.csv (colonne: anon_id, path, AGE)."""
    return pd.read_csv(TRAIN_CSV)


def load_test_df() -> pd.DataFrame:
    """Carica test.csv (colonne: anon_id, path)."""
    return pd.read_csv(TEST_CSV)


def resolve_mri_path(relative_path: str, split: str) -> Path:
    """
    Convertejhe il path relativo presente nel CSV (es. "data/train/subj_2183.nii.gz")
    nel path assoluto reale sul filesystem.

    Il dataset Kaggle, una volta estratto, presenta una struttura con
    doppio livello di cartella (es. data/train/train/subj_2183.nii) a causa
    di come è organizzato lo zip originale. Questa funzione centralizza
    quella conoscenza in un solo punto, eliminando le manipolazioni di
    stringa sparse nei vari script.

    Parameters
    ----------
    relative_path : str
        Path come riportato nel CSV (può usare "/" o "\\" come separatore).
    split : str
        "train" oppure "test".

    Returns
    -------
    Path
        Path assoluto al file .nii sul filesystem.
    """
    filename = Path(relative_path.replace("\\", "/")).name
    filename = filename.replace(".nii.gz", ".nii")

    base_dir = MRI_TRAIN_DIR if split == "train" else MRI_TEST_DIR
    return base_dir / filename


def load_mri_volume(path: Path) -> np.ndarray:
    """Carica un singolo volume MRI da file NIfTI come array float32."""
    img = nib.load(str(path))
    return np.asarray(img.dataobj, dtype=np.float32)


def load_subject_volume(relative_path: str, split: str) -> np.ndarray:
    """Scorciatoia: risolve il path e carica direttamente il volume."""
    path = resolve_mri_path(relative_path, split)
    return load_mri_volume(path)
