"""Test per brain_age.features.extraction."""

import numpy as np
import pytest

from brain_age.features.extraction import (
    N_FEATURES_V2,
    N_FEATURES_V3,
    extract_approximate_roi_features,
    extract_features_v2,
    extract_features_v3,
)


@pytest.fixture
def fake_volume() -> np.ndarray:
    """Volume 64^3 sintetico: sfera di intensità ~0.5 su sfondo nero, + rumore."""
    rng = np.random.default_rng(42)
    size = 64
    volume = rng.normal(loc=0.5, scale=0.05, size=(size, size, size)).astype(np.float32)
    volume = np.clip(volume, 0, 1)

    # Azzera gli angoli per simulare il background nero post skull-stripping
    volume[:10, :10, :10] = 0.0
    return volume


def test_extract_features_v2_output_shape(fake_volume):
    """Il numero di feature prodotte deve corrispondere a N_FEATURES_V2."""
    features = extract_features_v2(fake_volume)
    assert features.shape == (N_FEATURES_V2,)


def test_extract_features_v2_dtype(fake_volume):
    """Le feature devono essere float32 per coerenza con il resto della pipeline."""
    features = extract_features_v2(fake_volume)
    assert features.dtype == np.float32


def test_extract_features_v2_no_nan_or_inf(fake_volume):
    """Nessuna feature deve essere NaN o infinita, anche con volumi degeneri."""
    features = extract_features_v2(fake_volume)
    assert np.all(np.isfinite(features))


def test_extract_features_v2_handles_all_background():
    """Un volume completamente vuoto (solo background) non deve crashare."""
    empty_volume = np.zeros((64, 64, 64), dtype=np.float32)
    features = extract_features_v2(empty_volume)
    assert features.shape == (N_FEATURES_V2,)
    assert np.all(np.isfinite(features))


def test_extract_features_v2_deterministic(fake_volume):
    """La stessa estrazione applicata due volte deve dare risultati identici."""
    features_1 = extract_features_v2(fake_volume)
    features_2 = extract_features_v2(fake_volume)
    np.testing.assert_array_equal(features_1, features_2)


class TestApproximateRoiFeatures:
    """
    Test per le regioni anatomiche APPROSSIMATE (ventricoli, ippocampo).

    Promemoria: sono bounding-box posizionali, non un atlante registrato;
    questi test verificano solo la correttezza tecnica dell'estrazione, non
    la validità anatomica delle regioni.
    """

    def test_output_shape(self, fake_volume):
        roi_features = extract_approximate_roi_features(fake_volume)
        assert roi_features.shape == (27,)  # 3 regioni x 9 statistiche

    def test_no_nan_or_inf(self, fake_volume):
        roi_features = extract_approximate_roi_features(fake_volume)
        assert np.all(np.isfinite(roi_features))

    def test_handles_empty_volume(self):
        empty_volume = np.zeros((128, 128, 128), dtype=np.float32)
        roi_features = extract_approximate_roi_features(empty_volume)
        assert roi_features.shape == (27,)
        assert np.all(np.isfinite(roi_features))

    def test_works_at_different_resolutions(self):
        """I bounding-box sono frazionari: devono funzionare anche a 64^3."""
        rng = np.random.default_rng(0)
        volume_64 = rng.uniform(0, 1, size=(64, 64, 64)).astype(np.float32)
        roi_features = extract_approximate_roi_features(volume_64)
        assert roi_features.shape == (27,)
        assert np.all(np.isfinite(roi_features))


class TestExtractFeaturesV3:
    def test_output_shape_matches_constant(self, fake_volume):
        features = extract_features_v3(fake_volume)
        assert features.shape == (N_FEATURES_V3,)

    def test_v3_extends_v2(self, fake_volume):
        """Le prime N_FEATURES_V2 feature di v3 devono coincidere con v2."""
        features_v2 = extract_features_v2(fake_volume)
        features_v3 = extract_features_v3(fake_volume)
        np.testing.assert_array_equal(features_v3[:N_FEATURES_V2], features_v2)

    def test_no_nan_or_inf(self, fake_volume):
        features = extract_features_v3(fake_volume)
        assert np.all(np.isfinite(features))
