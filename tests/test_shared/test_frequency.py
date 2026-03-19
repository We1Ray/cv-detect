"""Tests for shared.core.frequency -- FFT and frequency domain processing."""

from __future__ import annotations

import numpy as np
import pytest

from shared.core.frequency import (
    FFTResult,
    apply_frequency_filter,
    compute_fft,
    create_bandpass_filter,
    create_butterworth_filter,
    create_gaussian_filter,
    create_notch_filter,
    inverse_fft,
    remove_periodic_pattern,
)


# ------------------------------------------------------------------
# compute_fft / inverse_fft
# ------------------------------------------------------------------


class TestComputeFFT:
    """Tests for the compute_fft function."""

    def test_returns_fft_result(self, sample_grayscale: np.ndarray) -> None:
        """compute_fft should return an FFTResult dataclass."""
        result = compute_fft(sample_grayscale)
        assert isinstance(result, FFTResult)

    def test_magnitude_shape_at_least_as_large_as_input(
        self, sample_grayscale: np.ndarray
    ) -> None:
        """Magnitude shape should be >= input shape (due to optimal DFT padding)."""
        result = compute_fft(sample_grayscale)
        h, w = sample_grayscale.shape[:2]
        assert result.magnitude.shape[0] >= h
        assert result.magnitude.shape[1] >= w

    def test_magnitude_is_uint8(self, sample_grayscale: np.ndarray) -> None:
        """Magnitude spectrum should be uint8 for display."""
        result = compute_fft(sample_grayscale)
        assert result.magnitude.dtype == np.uint8

    def test_magnitude_non_negative(self, sample_grayscale: np.ndarray) -> None:
        """All magnitude values should be >= 0."""
        result = compute_fft(sample_grayscale)
        assert np.all(result.magnitude >= 0)

    def test_phase_range(self, sample_grayscale: np.ndarray) -> None:
        """Phase values should be in [-pi, pi]."""
        result = compute_fft(sample_grayscale)
        assert result.phase.min() >= -np.pi - 1e-6
        assert result.phase.max() <= np.pi + 1e-6

    def test_original_shape_preserved(self, sample_grayscale: np.ndarray) -> None:
        """FFTResult.shape should record the original image dimensions."""
        result = compute_fft(sample_grayscale)
        assert result.shape == sample_grayscale.shape[:2]

    def test_accepts_color_image(self, sample_image: np.ndarray) -> None:
        """compute_fft should accept a BGR image by converting to grayscale."""
        result = compute_fft(sample_image)
        assert isinstance(result, FFTResult)

    def test_roundtrip_reconstruction(self, sample_grayscale: np.ndarray) -> None:
        """FFT followed by inverse FFT should roughly reconstruct the image."""
        fft_result = compute_fft(sample_grayscale)
        reconstructed = inverse_fft(fft_result)
        assert reconstructed.shape == sample_grayscale.shape
        assert reconstructed.dtype == np.uint8


# ------------------------------------------------------------------
# Gaussian filter
# ------------------------------------------------------------------


class TestGaussianFilter:
    """Tests for create_gaussian_filter."""

    def test_lowpass_shape(self) -> None:
        """Output shape should match the requested shape."""
        shape = (128, 128)
        filt = create_gaussian_filter(shape, sigma=30.0, filter_type="lowpass")
        assert filt.shape == shape

    def test_lowpass_values_in_unit_range(self) -> None:
        """Lowpass filter values should be in [0, 1]."""
        filt = create_gaussian_filter((128, 128), sigma=30.0, filter_type="lowpass")
        assert filt.min() >= 0.0 - 1e-9
        assert filt.max() <= 1.0 + 1e-9

    def test_highpass_values_in_unit_range(self) -> None:
        """Highpass filter values should be in [0, 1]."""
        filt = create_gaussian_filter((128, 128), sigma=30.0, filter_type="highpass")
        assert filt.min() >= 0.0 - 1e-9
        assert filt.max() <= 1.0 + 1e-9

    def test_lowpass_center_is_max(self) -> None:
        """The center of a lowpass Gaussian filter should have the highest value."""
        filt = create_gaussian_filter((128, 128), sigma=30.0, filter_type="lowpass")
        center = filt[64, 64]
        assert center == pytest.approx(1.0, abs=1e-6)

    def test_highpass_center_is_min(self) -> None:
        """The center of a highpass Gaussian filter should be near zero."""
        filt = create_gaussian_filter((128, 128), sigma=30.0, filter_type="highpass")
        center = filt[64, 64]
        assert center == pytest.approx(0.0, abs=1e-6)

    def test_invalid_filter_type_raises(self) -> None:
        """An invalid filter_type should raise ValueError."""
        with pytest.raises(ValueError, match="filter_type"):
            create_gaussian_filter((128, 128), sigma=30.0, filter_type="bandpass")


# ------------------------------------------------------------------
# Butterworth filter
# ------------------------------------------------------------------


class TestButterworthFilter:
    """Tests for create_butterworth_filter."""

    def test_lowpass_shape(self) -> None:
        """Output shape should match the requested shape."""
        shape = (64, 64)
        filt = create_butterworth_filter(shape, cutoff=20.0, order=2, filter_type="lowpass")
        assert filt.shape == shape

    def test_lowpass_values_in_unit_range(self) -> None:
        """Filter values should be in [0, 1]."""
        filt = create_butterworth_filter((64, 64), cutoff=20.0, order=2, filter_type="lowpass")
        assert filt.min() >= 0.0 - 1e-9
        assert filt.max() <= 1.0 + 1e-9

    def test_highpass_shape(self) -> None:
        """Highpass filter should have the same shape as requested."""
        shape = (64, 64)
        filt = create_butterworth_filter(shape, cutoff=20.0, order=2, filter_type="highpass")
        assert filt.shape == shape

    def test_higher_order_sharper_rolloff(self) -> None:
        """A higher-order filter should have a sharper transition at cutoff."""
        shape = (128, 128)
        low_order = create_butterworth_filter(shape, cutoff=30.0, order=1, filter_type="lowpass")
        high_order = create_butterworth_filter(shape, cutoff=30.0, order=5, filter_type="lowpass")
        # At 2x cutoff distance, higher order should attenuate more
        test_point = (64 + 60, 64)  # well beyond cutoff
        assert high_order[test_point] < low_order[test_point]

    def test_invalid_filter_type_raises(self) -> None:
        """An invalid filter_type should raise ValueError."""
        with pytest.raises(ValueError, match="filter_type"):
            create_butterworth_filter((64, 64), cutoff=20.0, filter_type="invalid")


# ------------------------------------------------------------------
# Bandpass filter
# ------------------------------------------------------------------


class TestBandpassFilter:
    """Tests for create_bandpass_filter."""

    def test_shape_matches(self) -> None:
        """Output shape should match the requested shape."""
        shape = (128, 128)
        filt = create_bandpass_filter(shape, low_cutoff=10.0, high_cutoff=40.0)
        assert filt.shape == shape

    def test_values_in_unit_range(self) -> None:
        """Filter values should be in [0, 1]."""
        filt = create_bandpass_filter((128, 128), low_cutoff=10.0, high_cutoff=40.0)
        assert filt.min() >= 0.0 - 1e-9
        assert filt.max() <= 1.0 + 1e-9

    def test_dc_is_near_zero(self) -> None:
        """The DC component (center) should be suppressed in a bandpass filter."""
        filt = create_bandpass_filter((128, 128), low_cutoff=20.0, high_cutoff=50.0)
        center = filt[64, 64]
        assert center < 0.1


# ------------------------------------------------------------------
# Notch filter
# ------------------------------------------------------------------


class TestNotchFilter:
    """Tests for create_notch_filter."""

    def test_shape_matches(self) -> None:
        """Output shape should match the requested shape."""
        shape = (128, 128)
        filt = create_notch_filter(shape, centers=[(10, 10)], radius=5)
        assert filt.shape == shape

    def test_values_in_unit_range(self) -> None:
        """Filter values should be in [0, 1]."""
        filt = create_notch_filter((128, 128), centers=[(10, 10)], radius=5)
        assert filt.min() >= 0.0 - 1e-9
        assert filt.max() <= 1.0 + 1e-9

    def test_notch_center_is_zero(self) -> None:
        """The filter value at and near the notch center should be zero."""
        shape = (128, 128)
        centers = [(20, 20)]
        filt = create_notch_filter(shape, centers=centers, radius=10)
        crow, ccol = shape[0] // 2, shape[1] // 2
        # Value at the notch center (relative to spectrum center)
        val = filt[crow + 20, ccol + 20]
        assert val == 0.0

    def test_symmetric_notch_is_also_zero(self) -> None:
        """The symmetric counterpart of a notch should also be zero."""
        shape = (128, 128)
        centers = [(15, 15)]
        filt = create_notch_filter(shape, centers=centers, radius=10)
        crow, ccol = shape[0] // 2, shape[1] // 2
        val = filt[crow - 15, ccol - 15]
        assert val == 0.0

    def test_far_from_notch_is_one(self) -> None:
        """Points far from any notch should remain at 1.0."""
        shape = (128, 128)
        filt = create_notch_filter(shape, centers=[(50, 50)], radius=3)
        # Check a point far from the notch
        assert filt[0, 0] == 1.0


# ------------------------------------------------------------------
# apply_frequency_filter
# ------------------------------------------------------------------


class TestApplyFilter:
    """Tests for apply_frequency_filter."""

    def test_output_same_shape_as_input(self, sample_grayscale: np.ndarray) -> None:
        """Filtered image should have the same shape as the input."""
        filt = create_gaussian_filter(sample_grayscale.shape, sigma=30.0)
        result = apply_frequency_filter(sample_grayscale, filt)
        assert result.shape == sample_grayscale.shape

    def test_output_is_uint8(self, sample_grayscale: np.ndarray) -> None:
        """Output should be uint8."""
        filt = create_gaussian_filter(sample_grayscale.shape, sigma=30.0)
        result = apply_frequency_filter(sample_grayscale, filt)
        assert result.dtype == np.uint8

    def test_lowpass_produces_valid_output(self) -> None:
        """A lowpass filter should produce a valid uint8 image without crashing."""
        rng = np.random.RandomState(99)
        noisy = rng.randint(50, 200, (64, 64), dtype=np.uint8)
        filt = create_gaussian_filter((64, 64), sigma=10.0, filter_type="lowpass")
        filtered = apply_frequency_filter(noisy, filt)
        assert filtered.dtype == np.uint8
        assert filtered.shape == noisy.shape
        # The filtered image should not be all zeros or all 255
        assert filtered.min() < filtered.max()

    def test_accepts_color_image(self, sample_image: np.ndarray) -> None:
        """apply_frequency_filter should accept BGR images."""
        filt = create_gaussian_filter(sample_image.shape[:2], sigma=30.0)
        result = apply_frequency_filter(sample_image, filt)
        assert result.shape == sample_image.shape[:2]


# ------------------------------------------------------------------
# remove_periodic_pattern
# ------------------------------------------------------------------


class TestRemovePeriodicPattern:
    """Tests for remove_periodic_pattern."""

    def test_output_shape_matches_input(self, sample_grayscale: np.ndarray) -> None:
        """Output shape should match the input image shape."""
        result = remove_periodic_pattern(sample_grayscale)
        assert result.shape == sample_grayscale.shape

    def test_output_is_uint8(self, sample_grayscale: np.ndarray) -> None:
        """Output should be uint8."""
        result = remove_periodic_pattern(sample_grayscale)
        assert result.dtype == np.uint8

    def test_no_crash_on_uniform_image(self) -> None:
        """Should not crash on a uniform (all-constant) image."""
        uniform = np.full((64, 64), 128, dtype=np.uint8)
        result = remove_periodic_pattern(uniform)
        assert result.shape == (64, 64)
        assert result.dtype == np.uint8

    def test_handles_synthetic_periodic_pattern(self) -> None:
        """Should run without error on an image with a synthetic periodic pattern."""
        rows, cols = 128, 128
        x = np.arange(cols)
        y = np.arange(rows)
        X, Y = np.meshgrid(x, y)
        # Add a sinusoidal pattern
        pattern = 128 + 60 * np.sin(2 * np.pi * X / 16.0)
        img = np.clip(pattern, 0, 255).astype(np.uint8)
        result = remove_periodic_pattern(img)
        assert result.shape == (rows, cols)
