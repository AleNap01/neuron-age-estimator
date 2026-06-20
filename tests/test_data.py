"""Test per brain_age.data (loader e preprocessing)."""

import numpy as np
import pytest

from brain_age.data.loader import resolve_mri_path
from brain_age.data.preprocessing import downsample_volume, mask_background


class TestResolveMriPath:
    def test_unix_style_path_train(self):
        path = resolve_mri_path("data/train/subj_2183.nii.gz", split="train")
        assert path.name == "subj_2183.nii"
        assert "train" in str(path)

    def test_windows_style_path_test(self):
        path = resolve_mri_path("data\\test\\subj_511.nii.gz", split="test")
        assert path.name == "subj_511.nii"
        assert "test" in str(path)

    def test_strips_gz_extension(self):
        path = resolve_mri_path("data/train/subj_1.nii.gz", split="train")
        assert path.suffix == ".nii"
        assert not str(path).endswith(".gz")


class TestDownsampleVolume:
    def test_downsample_128_to_64(self):
        volume = np.random.rand(128, 128, 128).astype(np.float32)
        result = downsample_volume(volume, target_size=64)
        assert result.shape == (64, 64, 64)

    def test_downsample_preserves_value_range(self):
        volume = np.random.uniform(0, 1, size=(128, 128, 128)).astype(np.float32)
        result = downsample_volume(volume, target_size=64)
        # L'interpolazione lineare non deve generare valori fuori range
        assert result.min() >= -0.01
        assert result.max() <= 1.01


class TestMaskBackground:
    def test_excludes_zero_voxels(self):
        volume = np.array([0.0, 0.0, 0.5, 0.8, 0.005])
        result = mask_background(volume, threshold=0.01)
        np.testing.assert_array_equal(result, np.array([0.5, 0.8]))

    def test_empty_when_all_background(self):
        volume = np.zeros(100)
        result = mask_background(volume)
        assert len(result) == 0
