"""
extraction.py — Feature engineering per il modello ML classico (v2, 155 feature)
              + estensione v3 con regioni anatomiche approssimate.

Ogni volume MRI 3D (128^3 ≈ 2M voxel) viene riassunto in un vettore di feature
statistiche, evitando di passare i voxel grezzi a modelli classici (Random
Forest, SVR) che andrebbero in overfitting con così tante dimensioni e così
pochi campioni (2328 soggetti).

Le feature v2 sono organizzate in 6 famiglie, motivate dal fatto che
l'invecchiamento cerebrale si manifesta come atrofia tissutale e dilatazione
ventricolare, fenomeni che alterano sia l'intensità media dei voxel sia la
loro distribuzione spaziale:

1. Statistiche globali        — quadro d'insieme dell'intero volume
2. Istogramma di intensità    — forma della distribuzione (non solo media/std)
3. Gradiente spaziale (Sobel) — texture/bordi, proxy del contrasto materia
                                  grigia / materia bianca
4. Statistiche per ottante    — 8 sotto-cubi, per rilevare asimmetrie locali
5. Statistiche per fascia     — 6 fasce lungo l'asse assiale
6. Centro vs periferia        — proxy della dilatazione ventricolare

Le feature v3 aggiungono region-of-interest APPROSSIMATE per posizione
(ventricoli laterali, ippocampo bilaterale) — si tratta di bounding-box
basati su conoscenza neuroanatomica generale, NON di un atlante registrato
(es. Harvard-Oxford/MNI). Un atlante vero richiederebbe la registrazione di
ogni volume nello spazio MNI152 tramite strumenti come FSL o ANTs, non
disponibili nell'ambiente di sviluppo (Windows, senza WSL configurato).
Queste feature vanno quindi interpretate come un'approssimazione euristica,
non come una segmentazione anatomicamente validata.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage, stats

# Le 9 statistiche descrittive calcolate ripetutamente su diverse sotto-regioni.
STAT_KEYS = ["mean", "std", "p10", "p25", "p50", "p75", "p90", "skew", "kurt"]

N_FEATURES_V2 = 155  # numero totale di feature prodotte da extract_features_v2
N_FEATURES_V3 = N_FEATURES_V2 + 9 * 3  # v2 (155) + 3 regioni approssimate × 9 statistiche = 182


def _basic_stats(region: np.ndarray, background_threshold: float = 0.01) -> list[float]:
    """Calcola le STAT_KEYS su una sotto-regione, escludendo il background."""
    voxels = region[region > background_threshold]
    if len(voxels) == 0:
        return [0.0] * len(STAT_KEYS)

    values = {
        "mean": voxels.mean(),
        "std": voxels.std(),
        "p10": np.percentile(voxels, 10),
        "p25": np.percentile(voxels, 25),
        "p50": np.percentile(voxels, 50),
        "p75": np.percentile(voxels, 75),
        "p90": np.percentile(voxels, 90),
        "skew": stats.skew(voxels),
        "kurt": stats.kurtosis(voxels),
    }
    return [values[key] for key in STAT_KEYS]


def _split_octants(volume: np.ndarray) -> list[np.ndarray]:
    """Divide il volume in 8 sotto-cubi (ottanti) di dimensione dimezzata."""
    h = volume.shape[0] // 2
    return [
        volume[:h, :h, :h], volume[:h, :h, h:],
        volume[:h, h:, :h], volume[:h, h:, h:],
        volume[h:, :h, :h], volume[h:, :h, h:],
        volume[h:, h:, :h], volume[h:, h:, h:],
    ]


def _split_axial_slabs(volume: np.ndarray, n_slabs: int = 6) -> list[np.ndarray]:
    """Divide il volume in n_slabs fasce lungo l'asse assiale (primo asse)."""
    t = volume.shape[0] // n_slabs
    return [volume[i * t:(i + 1) * t, :, :] for i in range(n_slabs)]


def extract_features_v2(volume: np.ndarray, background_threshold: float = 0.01) -> np.ndarray:
    """
    Estrae 155 feature statistiche da un volume MRI 3D.

    Parameters
    ----------
    volume : np.ndarray
        Volume MRI di shape (128, 128, 128), intensità normalizzate in [0, 1].
    background_threshold : float
        Soglia sotto la quale un voxel è considerato background (non cerebrale).

    Returns
    -------
    np.ndarray
        Vettore di 155 feature, dtype float32.
    """
    feats: list[float] = []
    brain = volume[volume > background_threshold]

    # 1. Statistiche globali (9 + max/min/volume relativo = 12)
    feats.extend(_basic_stats(volume, background_threshold))
    feats.append(float(brain.max()) if len(brain) else 0.0)
    feats.append(float(brain.min()) if len(brain) else 0.0)
    feats.append(len(brain) / volume.size)  # proxy del volume cerebrale relativo

    # 2. Istogramma di intensità, 10 bin
    if len(brain) > 0:
        hist, _ = np.histogram(brain, bins=10, range=(0, 1), density=True)
        hist = np.nan_to_num(hist, nan=0.0)  # densità non definita se un bin è vuoto
    else:
        hist = np.zeros(10)
    feats.extend(hist.tolist())

    # 3. Gradiente spaziale (Sobel 3D) — texture / bordi
    gx = ndimage.sobel(volume, axis=0)
    gy = ndimage.sobel(volume, axis=1)
    gz = ndimage.sobel(volume, axis=2)
    grad_mag = np.sqrt(gx**2 + gy**2 + gz**2)
    grad_brain = grad_mag[volume > background_threshold]
    if len(grad_brain) > 0:
        feats.append(float(grad_brain.mean()))
        feats.append(float(grad_brain.std()))
        feats.append(float(np.percentile(grad_brain, 90)))
    else:
        feats.extend([0.0, 0.0, 0.0])

    # 4. Statistiche per 8 ottanti (9 * 8 = 72)
    for octant in _split_octants(volume):
        feats.extend(_basic_stats(octant, background_threshold))

    # 5. Statistiche per 6 fasce assiali (9 * 6 = 54)
    for slab in _split_axial_slabs(volume, n_slabs=6):
        feats.extend(_basic_stats(slab, background_threshold))

    # 6. Centro vs periferia — proxy dilatazione ventricolare (4)
    c = volume.shape[0] // 4
    center = volume[c:3 * c, c:3 * c, c:3 * c]
    center_brain = center[center > background_threshold]

    if len(center_brain) > 0:
        global_mean = brain.mean() if len(brain) else 1e-6
        feats.append(float(center_brain.mean()))
        feats.append(float(center_brain.std()))
        feats.append(len(center_brain) / center.size)
        feats.append(float(center_brain.mean() / (global_mean + 1e-6)))
    else:
        feats.extend([0.0, 0.0, 0.0, 0.0])

    return np.array(feats, dtype=np.float32)


# ──────────────────────────────────────────────
# REGIONI ANATOMICHE APPROSSIMATE (v3)
# ──────────────────────────────────────────────
#
# Bounding-box espressi come FRAZIONI delle dimensioni del volume (0.0-1.0),
# così da funzionare a qualunque risoluzione (128^3, 64^3, ...), assumendo che
# il volume sia centrato e orientato in modo standard (come nei volumi forniti
# dalla competizione, già "minimally pre-processed" e "rigidly registered to
# a common space" secondo la descrizione del dataset).
#
# Convenzione assi: asse 0 = assiale (inferiore -> superiore),
#                    asse 1 = coronale (posteriore -> anteriore),
#                    asse 2 = sagittale (sinistra -> destra).
#
# ATTENZIONE: queste sono APPROSSIMAZIONI per posizione, non il risultato di
# una segmentazione anatomica validata. Vanno usate e descritte come tali.

_VENTRICLES_BBOX = {
    # I ventricoli laterali si trovano in posizione centrale, leggermente
    # superiore al centro del volume, attorno al piano sagittale mediano.
    "axial": (0.45, 0.65),
    "coronal": (0.35, 0.65),
    "sagittal": (0.40, 0.60),
}

_HIPPOCAMPUS_LEFT_BBOX = {
    # L'ippocampo si trova in posizione mediale-temporale, inferiore rispetto
    # ai ventricoli, in un emisfero (qui: "sinistro" lungo l'asse sagittale).
    "axial": (0.30, 0.50),
    "coronal": (0.35, 0.55),
    "sagittal": (0.55, 0.75),
}

_HIPPOCAMPUS_RIGHT_BBOX = {
    "axial": (0.30, 0.50),
    "coronal": (0.35, 0.55),
    "sagittal": (0.25, 0.45),
}


def _extract_bbox_region(volume: np.ndarray, bbox: dict) -> np.ndarray:
    """Estrae la sotto-regione del volume corrispondente a un bounding-box frazionario."""
    nx, ny, nz = volume.shape
    ax_lo, ax_hi = bbox["axial"]
    co_lo, co_hi = bbox["coronal"]
    sa_lo, sa_hi = bbox["sagittal"]

    return volume[
        int(ax_lo * nx):int(ax_hi * nx),
        int(co_lo * ny):int(co_hi * ny),
        int(sa_lo * nz):int(sa_hi * nz),
    ]


def extract_approximate_roi_features(volume: np.ndarray, background_threshold: float = 0.01) -> np.ndarray:
    """
    Estrae 27 feature (9 statistiche x 3 regioni) da bounding-box anatomici
    approssimati: ventricoli laterali, ippocampo sinistro, ippocampo destro.

    Queste regioni sono scelte perché note in letteratura come tra le più
    sensibili all'invecchiamento (dilatazione ventricolare, atrofia
    ippocampale), ma qui sono solo bounding-box posizionali — NON il
    risultato di una segmentazione anatomica con atlante registrato.
    """
    ventricles = _extract_bbox_region(volume, _VENTRICLES_BBOX)
    hippocampus_left = _extract_bbox_region(volume, _HIPPOCAMPUS_LEFT_BBOX)
    hippocampus_right = _extract_bbox_region(volume, _HIPPOCAMPUS_RIGHT_BBOX)

    feats: list[float] = []
    for region in (ventricles, hippocampus_left, hippocampus_right):
        feats.extend(_basic_stats(region, background_threshold))

    return np.array(feats, dtype=np.float32)


def extract_features_v3(volume: np.ndarray, background_threshold: float = 0.01) -> np.ndarray:
    """
    Estensione di extract_features_v2 con 27 feature aggiuntive da regioni
    anatomiche approssimate (ventricoli, ippocampo bilaterale).

    Totale: 155 (v2) + 27 = 182 feature.
    """
    base_features = extract_features_v2(volume, background_threshold)
    roi_features = extract_approximate_roi_features(volume, background_threshold)
    return np.concatenate([base_features, roi_features])
