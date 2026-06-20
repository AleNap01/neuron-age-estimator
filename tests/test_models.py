"""Test per brain_age.models.cnn3d e dataset."""

import numpy as np
import torch

from brain_age.models.cnn3d import BrainAgeCNN3D
from brain_age.models.dataset import BrainMRIDataset


class TestBrainAgeCNN3D:
    def test_forward_pass_output_shape(self):
        model = BrainAgeCNN3D()
        batch = torch.randn(4, 1, 64, 64, 64)  # batch=4, canale=1, volume 64^3
        output = model(batch)
        assert output.shape == (4,)

    def test_single_sample_forward_pass(self):
        model = BrainAgeCNN3D()
        sample = torch.randn(1, 1, 64, 64, 64)
        output = model(sample)
        assert output.shape == (1,)

    def test_parameter_count_is_reasonable(self):
        """La rete deve restare leggera (training su CPU)."""
        model = BrainAgeCNN3D()
        n_params = model.count_parameters()
        assert 0 < n_params < 1_000_000


class TestBrainMRIDataset:
    def test_dataset_length(self):
        volumes = np.random.rand(10, 64, 64, 64).astype(np.float32)
        ages = np.random.uniform(6, 88, size=10).astype(np.float32)
        indices = np.array([0, 2, 4, 6])

        dataset = BrainMRIDataset(volumes, ages, indices, age_mean=33.0, age_std=21.0)
        assert len(dataset) == 4

    def test_getitem_shapes(self):
        volumes = np.random.rand(5, 64, 64, 64).astype(np.float32)
        ages = np.random.uniform(6, 88, size=5).astype(np.float32)
        indices = np.array([0, 1, 2])

        dataset = BrainMRIDataset(volumes, ages, indices, age_mean=33.0, age_std=21.0)
        volume_tensor, age_tensor = dataset[0]

        assert volume_tensor.shape == (1, 64, 64, 64)
        assert age_tensor.shape == ()

    def test_age_normalization_roundtrip(self):
        volumes = np.random.rand(3, 64, 64, 64).astype(np.float32)
        ages = np.array([20.0, 40.0, 60.0], dtype=np.float32)
        indices = np.array([0, 1, 2])
        age_mean, age_std = ages.mean(), ages.std()

        dataset = BrainMRIDataset(volumes, ages, indices, age_mean, age_std)
        _, age_normalized = dataset[1]  # età reale = 40.0

        recovered_age = dataset.denormalize(age_normalized)
        assert abs(recovered_age.item() - 40.0) < 1e-4
