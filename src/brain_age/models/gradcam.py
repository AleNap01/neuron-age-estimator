"""
gradcam.py — Grad-CAM 3D per l'interpretabilità della CNN.

Grad-CAM (Gradient-weighted Class Activation Mapping) produce una mappa di
salienza che evidenzia quali regioni del volume di input hanno influenzato
maggiormente la predizione del modello. Per un task di regressione (età),
usiamo il gradiente dell'output scalare rispetto alle feature map dell'ultimo
blocco convoluzionale.

Riferimento concettuale: Selvaraju et al., "Grad-CAM: Visual Explanations
from Deep Networks via Gradient-based Localization" (2017), qui adattato
da classificazione 2D a regressione 3D.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from brain_age.models.cnn3d import BrainAgeCNN3D


class GradCAM3D:
    """
    Calcola mappe Grad-CAM 3D per BrainAgeCNN3D.

    L'hook viene posizionato sull'ultimo blocco convoluzionale
    (model.features[-1]), che nella nostra architettura produce feature map
    di shape (128, 4, 4, 4) — la risoluzione spaziale più bassa ma anche la
    rappresentazione più "semantica" della rete.
    """

    def __init__(self, model: BrainAgeCNN3D) -> None:
        self.model = model
        self.model.eval()

        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

        target_layer = self.model.features[-1]
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input_, output):
        self._activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def generate(self, volume_tensor: torch.Tensor) -> tuple[np.ndarray, float]:
        """
        Calcola la mappa Grad-CAM per un singolo volume.

        Parameters
        ----------
        volume_tensor : torch.Tensor
            Tensore di shape (1, 1, D, H, W) (batch singolo, canale singolo).

        Returns
        -------
        cam : np.ndarray
            Mappa di salienza, shape (D, H, W), valori normalizzati in [0, 1],
            ricampionata alla risoluzione del volume di input.
        prediction : float
            Predizione del modello (età normalizzata, z-score) per questo volume.
        """
        volume_tensor = volume_tensor.clone().requires_grad_(True)

        prediction = self.model(volume_tensor)  # shape (1,)
        self.model.zero_grad()
        prediction.backward()

        # Pesi dei canali: media spaziale del gradiente per ciascun canale
        # (questo è il passo "weighted" di Grad-CAM: i canali con gradiente
        # medio più alto hanno contribuito di più alla predizione finale).
        weights = self._gradients.mean(dim=(2, 3, 4), keepdim=True)  # (1, C, 1, 1, 1)

        # Combinazione pesata delle feature map, seguita da ReLU (Grad-CAM
        # originale considera solo i contributi positivi alla predizione)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, d, h, w)
        cam = F.relu(cam)

        # Upsampling alla risoluzione originale del volume di input
        target_size = volume_tensor.shape[2:]
        cam = F.interpolate(cam, size=target_size, mode="trilinear", align_corners=False)

        cam = cam.squeeze().detach().numpy()

        # Normalizzazione in [0, 1] per la visualizzazione
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, prediction.item()


def overlay_gradcam_on_slice(
    brain_slice: np.ndarray,
    cam_slice: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Sovrappone una slice 2D della mappa Grad-CAM su una slice 2D del cervello,
    producendo un'immagine RGB pronta per la visualizzazione.

    Usa una colormap "calore" semplice (rosso = alta salienza) per evitare
    una dipendenza diretta da matplotlib in questo modulo.

    Parameters
    ----------
    brain_slice : np.ndarray
        Slice 2D del volume MRI, valori in [0, 1].
    cam_slice : np.ndarray
        Slice 2D della mappa Grad-CAM corrispondente, valori in [0, 1].
    alpha : float
        Opacità della sovrapposizione della heatmap.

    Returns
    -------
    np.ndarray
        Immagine RGB, shape (H, W, 3), valori in [0, 1].
    """
    gray = np.clip(brain_slice, 0, 1)
    base_rgb = np.stack([gray, gray, gray], axis=-1)

    # Colormap "calore" minimale: rosso cresce con la salienza, verde/blu calano
    heat_rgb = np.stack([
        np.ones_like(cam_slice),
        1 - cam_slice,
        1 - cam_slice,
    ], axis=-1)

    cam_3d = cam_slice[..., np.newaxis]
    blended = base_rgb * (1 - alpha * cam_3d) + heat_rgb * (alpha * cam_3d)

    return np.clip(blended, 0, 1)
