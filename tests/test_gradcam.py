"""Test per brain_age.models.gradcam."""

import numpy as np
import torch

from brain_age.models.cnn3d import BrainAgeCNN3D
from brain_age.models.gradcam import GradCAM3D, overlay_gradcam_on_slice


def _make_minimally_trained_model() -> BrainAgeCNN3D:
    """
    Restituisce un modello con pesi leggermente allontanati dall'inizializzazione.

    Con pesi puramente random, i gradienti di Grad-CAM possono risultare
    quasi uniformi e finire interamente azzerati dal ReLU finale: un
    comportamento corretto, ma che non distingue un Grad-CAM "funzionante"
    da uno "rotto". Pochi step di training rendono il test più significativo.
    """
    torch.manual_seed(123)
    model = BrainAgeCNN3D()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for _ in range(5):
        x = torch.randn(2, 1, 64, 64, 64)
        y = torch.randn(2)
        optimizer.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        optimizer.step()

    return model


class TestGradCAM3D:
    def test_output_shape_matches_input(self):
        model = _make_minimally_trained_model()
        gradcam = GradCAM3D(model)

        volume = torch.randn(1, 1, 64, 64, 64)
        cam, _ = gradcam.generate(volume)

        assert cam.shape == (64, 64, 64)

    def test_cam_is_normalized(self):
        model = _make_minimally_trained_model()
        gradcam = GradCAM3D(model)

        volume = torch.randn(1, 1, 64, 64, 64)
        cam, _ = gradcam.generate(volume)

        assert cam.min() >= 0.0
        assert cam.max() <= 1.0

    def test_cam_has_meaningful_variation(self):
        """Con un modello minimamente addestrato, il CAM non deve essere costante."""
        model = _make_minimally_trained_model()
        gradcam = GradCAM3D(model)

        volume = torch.randn(1, 1, 64, 64, 64)
        cam, _ = gradcam.generate(volume)

        assert cam.std() > 0.0

    def test_prediction_is_finite_scalar(self):
        model = _make_minimally_trained_model()
        gradcam = GradCAM3D(model)

        volume = torch.randn(1, 1, 64, 64, 64)
        _, prediction = gradcam.generate(volume)

        assert np.isfinite(prediction)


class TestOverlayGradcamOnSlice:
    def test_output_shape(self):
        brain_slice = np.random.uniform(0, 1, size=(64, 64)).astype(np.float32)
        cam_slice = np.random.uniform(0, 1, size=(64, 64)).astype(np.float32)

        overlay = overlay_gradcam_on_slice(brain_slice, cam_slice)

        assert overlay.shape == (64, 64, 3)

    def test_output_in_valid_range(self):
        brain_slice = np.random.uniform(0, 1, size=(64, 64)).astype(np.float32)
        cam_slice = np.random.uniform(0, 1, size=(64, 64)).astype(np.float32)

        overlay = overlay_gradcam_on_slice(brain_slice, cam_slice)

        assert overlay.min() >= 0.0
        assert overlay.max() <= 1.0

    def test_zero_saliency_returns_grayscale(self):
        """Con CAM tutto a zero, l'overlay deve coincidere con l'immagine originale in grigio."""
        brain_slice = np.full((32, 32), 0.5, dtype=np.float32)
        cam_slice = np.zeros((32, 32), dtype=np.float32)

        overlay = overlay_gradcam_on_slice(brain_slice, cam_slice)

        np.testing.assert_allclose(overlay[..., 0], brain_slice, atol=1e-6)
        np.testing.assert_allclose(overlay[..., 1], brain_slice, atol=1e-6)
        np.testing.assert_allclose(overlay[..., 2], brain_slice, atol=1e-6)
