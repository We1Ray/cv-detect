"""Tests for dl_anomaly.core.anomaly_scorer -- AnomalyScorer class."""

from __future__ import annotations

import numpy as np
import pytest

from dl_anomaly.core.anomaly_scorer import AnomalyScorer


# ------------------------------------------------------------------
# compute_pixel_error
# ------------------------------------------------------------------

class TestComputePixelError:
    """Tests for the static compute_pixel_error method."""

    def test_identical_images_zero_error(self, sample_image: np.ndarray) -> None:
        """Error between an image and itself should be exactly zero."""
        err = AnomalyScorer.compute_pixel_error(sample_image, sample_image)
        assert err.shape == (256, 256)
        assert err.dtype == np.float32
        np.testing.assert_allclose(err, 0.0, atol=1e-7)

    def test_max_error_black_white(self) -> None:
        """Error between all-black and all-white images should be 1.0."""
        black = np.zeros((64, 64, 3), dtype=np.uint8)
        white = np.full((64, 64, 3), 255, dtype=np.uint8)
        err = AnomalyScorer.compute_pixel_error(black, white)
        np.testing.assert_allclose(err, 1.0, atol=1e-5)

    def test_grayscale_input(self, sample_grayscale: np.ndarray) -> None:
        """2D (H, W) grayscale arrays should be handled correctly."""
        err = AnomalyScorer.compute_pixel_error(sample_grayscale, sample_grayscale)
        assert err.shape == (256, 256)
        np.testing.assert_allclose(err, 0.0, atol=1e-7)

    def test_known_pixel_value(self) -> None:
        """Compute error for a known single-pixel difference."""
        a = np.array([[100]], dtype=np.uint8)
        b = np.array([[200]], dtype=np.uint8)
        err = AnomalyScorer.compute_pixel_error(a, b)
        expected = (100.0 ** 2) / (255.0 ** 2)
        np.testing.assert_allclose(err[0, 0], expected, atol=1e-6)

    def test_output_range_zero_to_one(self, sample_image: np.ndarray) -> None:
        """Error map values should be in [0, 1]."""
        rng = np.random.RandomState(99)
        other = rng.randint(0, 256, sample_image.shape, dtype=np.uint8)
        err = AnomalyScorer.compute_pixel_error(sample_image, other)
        assert err.min() >= 0.0
        assert err.max() <= 1.0


# ------------------------------------------------------------------
# compute_image_score
# ------------------------------------------------------------------

class TestComputeImageScore:
    """Tests for the static compute_image_score method."""

    def test_zero_map_gives_zero(self) -> None:
        """An all-zero error map should produce a score of 0."""
        err = np.zeros((64, 64), dtype=np.float32)
        assert AnomalyScorer.compute_image_score(err) == pytest.approx(0.0)

    def test_uniform_map_gives_value(self) -> None:
        """A uniform error map should produce a score equal to that value."""
        err = np.full((64, 64), 0.5, dtype=np.float32)
        assert AnomalyScorer.compute_image_score(err) == pytest.approx(0.5)

    def test_returns_float(self) -> None:
        """Score should always be a Python float."""
        err = np.ones((10, 10), dtype=np.float32)
        score = AnomalyScorer.compute_image_score(err)
        assert isinstance(score, float)


# ------------------------------------------------------------------
# classify / fit_threshold
# ------------------------------------------------------------------

class TestClassify:
    """Tests for threshold fitting and classification."""

    def test_classify_above_threshold(self) -> None:
        """A score above the fitted threshold should be classified as anomalous."""
        scorer = AnomalyScorer(device="cpu")
        scorer.fit_threshold([0.1, 0.2, 0.3, 0.4, 0.5], percentile=95.0)
        assert scorer.classify(1.0) is True

    def test_classify_below_threshold(self) -> None:
        """A score well below the threshold should be classified as normal."""
        scorer = AnomalyScorer(device="cpu")
        scorer.fit_threshold([0.1, 0.2, 0.3, 0.4, 0.5], percentile=95.0)
        assert scorer.classify(0.0) is False

    def test_classify_raises_without_fit(self) -> None:
        """Calling classify before fit_threshold should raise RuntimeError."""
        scorer = AnomalyScorer(device="cpu")
        with pytest.raises(RuntimeError, match="Threshold has not been fitted"):
            scorer.classify(0.5)

    def test_fit_threshold_empty_list_raises(self) -> None:
        """fit_threshold with an empty list should raise ValueError."""
        scorer = AnomalyScorer(device="cpu")
        with pytest.raises(ValueError, match="empty"):
            scorer.fit_threshold([], percentile=95.0)

    def test_fit_threshold_returns_float(self) -> None:
        """fit_threshold should return the computed threshold as a float."""
        scorer = AnomalyScorer(device="cpu")
        t = scorer.fit_threshold([0.1, 0.2, 0.3], percentile=50.0)
        assert isinstance(t, float)
        assert t == pytest.approx(0.2, abs=0.01)


# ------------------------------------------------------------------
# create_anomaly_map
# ------------------------------------------------------------------

class TestCreateAnomalyMap:
    """Tests for Gaussian-smoothed anomaly map creation."""

    def test_output_shape_matches_input(self) -> None:
        """Output anomaly map should have the same shape as the input error map."""
        scorer = AnomalyScorer(device="cpu")
        err = np.random.rand(64, 64).astype(np.float32)
        amap = scorer.create_anomaly_map(err, gaussian_sigma=4.0)
        assert amap.shape == (64, 64)

    def test_output_normalised_zero_to_one(self) -> None:
        """Output should be min-max normalised to [0, 1]."""
        scorer = AnomalyScorer(device="cpu")
        err = np.random.rand(64, 64).astype(np.float32) * 0.5
        amap = scorer.create_anomaly_map(err, gaussian_sigma=4.0)
        assert amap.min() >= -1e-6
        assert amap.max() <= 1.0 + 1e-6

    def test_uniform_interior_is_near_constant(self) -> None:
        """On a uniform error map, the interior (away from edges) should be near-constant.

        Edge effects from zero-padded convolution are expected, so we check
        only a central crop for near-uniformity.
        """
        scorer = AnomalyScorer(device="cpu")
        err = np.full((128, 128), 0.3, dtype=np.float32)
        amap = scorer.create_anomaly_map(err, gaussian_sigma=4.0)
        # The central region should have very low variance
        center = amap[32:96, 32:96]
        assert center.std() < 1e-4

    def test_dtype_is_float32(self) -> None:
        """Output should be float32."""
        scorer = AnomalyScorer(device="cpu")
        err = np.random.rand(32, 32).astype(np.float32)
        amap = scorer.create_anomaly_map(err, gaussian_sigma=2.0)
        assert amap.dtype == np.float32
