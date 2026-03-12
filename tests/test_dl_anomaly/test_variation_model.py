"""Tests for variation_model.core.variation_model -- VariationModel class.

Validates the Welford online algorithm, mean/std computation, threshold
preparation, and save/load persistence.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from variation_model.core.variation_model import VariationModel


# ------------------------------------------------------------------
# train_incremental (Welford algorithm)
# ------------------------------------------------------------------

class TestTrainIncremental:
    """Tests for the online Welford update."""

    def test_single_image_not_trained(self) -> None:
        """A model with only 1 image should not be considered trained (need >= 2)."""
        model = VariationModel()
        model.train_incremental(np.ones((32, 32), dtype=np.uint8))
        assert model.is_trained is False
        assert model.count == 1

    def test_two_images_is_trained(self) -> None:
        """A model with 2 images should be considered trained."""
        model = VariationModel()
        model.train_incremental(np.zeros((32, 32), dtype=np.uint8))
        model.train_incremental(np.ones((32, 32), dtype=np.uint8) * 255)
        assert model.is_trained is True
        assert model.count == 2

    def test_count_increments(self) -> None:
        """Each call to train_incremental should increment the count by 1."""
        model = VariationModel()
        for i in range(5):
            model.train_incremental(np.zeros((16, 16), dtype=np.uint8))
        assert model.count == 5


# ------------------------------------------------------------------
# Mean / std computation with known data
# ------------------------------------------------------------------

class TestMeanStd:
    """Tests for mean and std_image with analytically known values."""

    def test_mean_of_identical_images(self) -> None:
        """Mean of N identical images should equal that image."""
        img = np.full((32, 32), 100.0, dtype=np.float64)
        model = VariationModel()
        for _ in range(5):
            model.train_incremental(img.astype(np.uint8))

        np.testing.assert_allclose(model.mean_image, 100.0, atol=1e-10)

    def test_std_of_identical_images_is_zero(self) -> None:
        """Standard deviation of identical images should be zero."""
        img = np.full((32, 32), 100, dtype=np.uint8)
        model = VariationModel()
        for _ in range(5):
            model.train_incremental(img)

        np.testing.assert_allclose(model.std_image, 0.0, atol=1e-10)

    def test_mean_of_two_values(self) -> None:
        """Mean of images with values 100 and 200 should be 150."""
        model = VariationModel()
        model.train_incremental(np.full((16, 16), 100, dtype=np.uint8))
        model.train_incremental(np.full((16, 16), 200, dtype=np.uint8))

        np.testing.assert_allclose(model.mean_image, 150.0, atol=1e-10)

    def test_std_of_two_values(self) -> None:
        """Std of values [100, 200] with population formula: sqrt(((100-150)^2+(200-150)^2)/2) = 50."""
        model = VariationModel()
        model.train_incremental(np.full((16, 16), 100, dtype=np.uint8))
        model.train_incremental(np.full((16, 16), 200, dtype=np.uint8))

        # The VariationModel uses population std: sqrt(m2 / count)
        expected_std = 50.0
        np.testing.assert_allclose(model.std_image, expected_std, atol=1e-10)

    def test_std_none_before_training(self) -> None:
        """std_image should return None when count < 2."""
        model = VariationModel()
        assert model.std_image is None

        model.train_incremental(np.zeros((8, 8), dtype=np.uint8))
        assert model.std_image is None

    def test_mean_with_known_sequence(self) -> None:
        """Mean of [10, 20, 30] should be 20."""
        model = VariationModel()
        for v in [10, 20, 30]:
            model.train_incremental(np.full((8, 8), v, dtype=np.uint8))

        np.testing.assert_allclose(model.mean_image, 20.0, atol=1e-10)

    def test_std_with_known_sequence(self) -> None:
        """Population std of [10, 20, 30] = sqrt(200/3) ~ 8.165."""
        model = VariationModel()
        for v in [10, 20, 30]:
            model.train_incremental(np.full((8, 8), v, dtype=np.uint8))

        expected = np.sqrt(((10 - 20) ** 2 + (20 - 20) ** 2 + (30 - 20) ** 2) / 3.0)
        np.testing.assert_allclose(model.std_image, expected, atol=1e-10)


# ------------------------------------------------------------------
# prepare (threshold images)
# ------------------------------------------------------------------

class TestPrepare:
    """Tests for threshold preparation."""

    def test_prepare_raises_when_untrained(self) -> None:
        """prepare() should raise RuntimeError on an untrained model."""
        model = VariationModel()
        with pytest.raises(RuntimeError, match="尚未訓練"):
            model.prepare()

    def test_prepare_creates_upper_lower(self) -> None:
        """After prepare(), upper and lower threshold images should exist."""
        model = VariationModel()
        for v in [100, 120, 110]:
            model.train_incremental(np.full((16, 16), v, dtype=np.uint8))

        model.prepare(abs_threshold=10, var_threshold=3.0)
        images = model.get_model_images()
        assert images["upper"] is not None
        assert images["lower"] is not None

    def test_upper_greater_than_lower(self) -> None:
        """Upper threshold should always be >= lower threshold."""
        model = VariationModel()
        for v in [50, 100, 150, 200]:
            model.train_incremental(np.full((16, 16), v, dtype=np.uint8))

        model.prepare(abs_threshold=5, var_threshold=2.0)
        images = model.get_model_images()
        assert np.all(images["upper"] >= images["lower"])


# ------------------------------------------------------------------
# Save / Load round-trip
# ------------------------------------------------------------------

class TestSaveLoad:
    """Tests for model persistence."""

    def test_save_load_round_trip(self, tmp_path: Path) -> None:
        """Saving and loading should preserve mean, m2, and count."""
        model = VariationModel()
        for v in [80, 120, 100]:
            model.train_incremental(np.full((16, 16), v, dtype=np.uint8))
        model.prepare(abs_threshold=10, var_threshold=3.0)

        save_path = tmp_path / "model.npz"
        model.save(save_path)
        assert save_path.exists()

        loaded = VariationModel.load(save_path)
        assert loaded.count == model.count
        np.testing.assert_allclose(loaded.mean_image, model.mean_image)
        np.testing.assert_allclose(loaded.std_image, model.std_image)

    def test_save_raises_when_untrained(self, tmp_path: Path) -> None:
        """Saving an untrained model should raise RuntimeError."""
        model = VariationModel()
        with pytest.raises(RuntimeError, match="尚未訓練"):
            model.save(tmp_path / "bad.npz")

    def test_load_nonexistent_raises(self, tmp_path: Path) -> None:
        """Loading from a nonexistent path should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            VariationModel.load(tmp_path / "does_not_exist.npz")

    def test_load_auto_appends_npz(self, tmp_path: Path) -> None:
        """If path has no .npz suffix but the .npz file exists, it should still load."""
        model = VariationModel()
        for v in [90, 110]:
            model.train_incremental(np.full((8, 8), v, dtype=np.uint8))

        save_path = tmp_path / "model.npz"
        model.save(save_path)

        # Load without extension
        loaded = VariationModel.load(tmp_path / "model")
        assert loaded.count == 2
