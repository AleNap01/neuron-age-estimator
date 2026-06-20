"""
preprocessing.py — Operazioni di preprocessing sui volumi MRI.
"""

import numpy as np
from scipy.ndimage import zoom


def downsample_volume(volume: np.ndarray, target_size: int) -> np.ndarray:
    """
    Riduce la risoluzione di un volume 3D cubico mediante interpolazione lineare.

    Usato per rendere fattibile il training di una CNN 3D su CPU: un volume
    128^3 ha oltre 2 milioni di voxel, troppi per un training rapido senza GPU;
    un volume 64^3 riduce i calcoli di un fattore 8 con una perdita di
    dettaglio contenuta.

    Parameters
    ----------
    volume : np.ndarray
        Volume 3D di shape (N, N, N).
    target_size : int
        Dimensione desiderata per ciascun asse (es. 64).

    Returns
    -------
    np.ndarray
        Volume ridotto di shape (target_size, target_size, target_size).
    """
    factor = target_size / volume.shape[0]
    return zoom(volume, zoom=factor, order=1)


def mask_background(volume: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """
    Restituisce solo i voxel "cerebrali" (intensità sopra soglia), escludendo
    il background nero che circonda il cervello nei volumi già sottoposti a
    skull-stripping.
    """
    return volume[volume > threshold]
