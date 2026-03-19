"""Tests for shared.core.color_inspect -- color inspection and measurement."""

from __future__ import annotations

import numpy as np
import pytest

from shared.core.color_inspect import (
    ColorSample,
    DeltaEResult,
    build_color_palette,
    check_color_tolerance,
    check_color_uniformity,
    classify_color,
    compute_delta_e,
    compute_delta_e_map,
    delta_e_cie76,
    delta_e_ciede2000,
    rgb_to_lab,
    sample_color,
)


# ------------------------------------------------------------------
# rgb_to_lab
# ------------------------------------------------------------------


class TestRgbToLab:
    """Tests for the rgb_to_lab conversion."""

    def test_output_shape_matches_input(self, sample_image: np.ndarray) -> None:
        """Lab image should have the same H, W, C shape as the input."""
        lab = rgb_to_lab(sample_image)
        assert lab.shape == sample_image.shape

    def test_output_dtype_is_float32(self, sample_image: np.ndarray) -> None:
        """Lab image should be float32."""
        lab = rgb_to_lab(sample_image)
        assert lab.dtype == np.float32

    def test_l_channel_range(self, sample_image: np.ndarray) -> None:
        """L channel should be in [0, 100] range (OpenCV convention: 0-255 mapped)."""
        lab = rgb_to_lab(sample_image)
        l_channel = lab[:, :, 0]
        # OpenCV Lab L is in [0, 255] range mapped from [0, 100]
        assert l_channel.min() >= 0.0
        assert l_channel.max() <= 255.0

    def test_accepts_grayscale(self) -> None:
        """Should accept a single-channel grayscale image."""
        gray = np.full((64, 64), 128, dtype=np.uint8)
        lab = rgb_to_lab(gray)
        assert lab.ndim == 3
        assert lab.shape[2] == 3


# ------------------------------------------------------------------
# Delta-E functions
# ------------------------------------------------------------------


class TestDeltaE:
    """Tests for delta_e_cie76 and delta_e_ciede2000."""

    def test_cie76_identical_is_zero(self) -> None:
        """CIE76 of identical colors should be zero."""
        lab = (50.0, 20.0, -10.0)
        assert delta_e_cie76(lab, lab) == pytest.approx(0.0)

    def test_cie76_different_is_positive(self) -> None:
        """CIE76 of different colors should be positive."""
        lab1 = (50.0, 20.0, -10.0)
        lab2 = (60.0, 30.0, 0.0)
        de = delta_e_cie76(lab1, lab2)
        assert de > 0.0

    def test_ciede2000_identical_is_zero(self) -> None:
        """CIEDE2000 of identical colors should be zero."""
        lab = (50.0, 20.0, -10.0)
        assert delta_e_ciede2000(lab, lab) == pytest.approx(0.0, abs=1e-6)

    def test_ciede2000_different_is_positive(self) -> None:
        """CIEDE2000 of different colors should be positive."""
        lab1 = (50.0, 20.0, -10.0)
        lab2 = (60.0, 30.0, 0.0)
        de = delta_e_ciede2000(lab1, lab2)
        assert de > 0.0

    def test_cie76_is_euclidean(self) -> None:
        """CIE76 should equal the Euclidean distance in Lab space."""
        import math

        lab1 = (50.0, 0.0, 0.0)
        lab2 = (53.0, 4.0, 0.0)
        expected = math.sqrt(9.0 + 16.0)
        assert delta_e_cie76(lab1, lab2) == pytest.approx(expected, abs=1e-6)

    def test_ciede2000_symmetry(self) -> None:
        """CIEDE2000 should be symmetric: dE(a,b) == dE(b,a)."""
        lab1 = (50.0, 25.0, -10.0)
        lab2 = (70.0, -15.0, 30.0)
        assert delta_e_ciede2000(lab1, lab2) == pytest.approx(
            delta_e_ciede2000(lab2, lab1), abs=1e-6
        )

    def test_compute_delta_e_cie76_method(self) -> None:
        """compute_delta_e with CIE76 method should return a DeltaEResult."""
        s1 = ColorSample(lab=(50.0, 20.0, -10.0), rgb=(0, 0, 0), hsv=(0, 0, 0), std=(0, 0, 0), area=1)
        s2 = ColorSample(lab=(50.0, 20.0, -10.0), rgb=(0, 0, 0), hsv=(0, 0, 0), std=(0, 0, 0), area=1)
        result = compute_delta_e(s1, s2, method="CIE76")
        assert isinstance(result, DeltaEResult)
        assert result.delta_e == pytest.approx(0.0)
        assert result.pass_fail is True

    def test_compute_delta_e_pass_fail(self) -> None:
        """compute_delta_e should set pass_fail=False when delta_e exceeds tolerance."""
        s1 = ColorSample(lab=(50.0, 0.0, 0.0), rgb=(0, 0, 0), hsv=(0, 0, 0), std=(0, 0, 0), area=1)
        s2 = ColorSample(lab=(90.0, 0.0, 0.0), rgb=(0, 0, 0), hsv=(0, 0, 0), std=(0, 0, 0), area=1)
        result = compute_delta_e(s1, s2, method="CIE76", tolerance=3.0)
        assert result.delta_e > 3.0
        assert result.pass_fail is False


# ------------------------------------------------------------------
# sample_color
# ------------------------------------------------------------------


class TestSampleColor:
    """Tests for the sample_color function."""

    def test_returns_color_sample(self, sample_image: np.ndarray) -> None:
        """sample_color should return a ColorSample dataclass."""
        result = sample_color(sample_image)
        assert isinstance(result, ColorSample)

    def test_lab_has_three_components(self, sample_image: np.ndarray) -> None:
        """Lab tuple should have exactly 3 components."""
        result = sample_color(sample_image)
        assert len(result.lab) == 3

    def test_rgb_values_are_valid(self, sample_image: np.ndarray) -> None:
        """RGB values should be integers in [0, 255]."""
        result = sample_color(sample_image)
        for v in result.rgb:
            assert 0 <= v <= 255

    def test_area_matches_full_image(self, sample_image: np.ndarray) -> None:
        """Area should equal total pixels when no ROI is given."""
        result = sample_color(sample_image)
        h, w = sample_image.shape[:2]
        assert result.area == h * w

    def test_roi_reduces_area(self, sample_image: np.ndarray) -> None:
        """Using an ROI should result in a smaller area."""
        roi = (10, 10, 50, 50)
        result = sample_color(sample_image, roi=roi)
        assert result.area == 50 * 50

    def test_uniform_image_low_std(self) -> None:
        """A uniform-color image should have near-zero standard deviation."""
        uniform = np.full((64, 64, 3), [100, 150, 200], dtype=np.uint8)
        result = sample_color(uniform)
        for s in result.std:
            assert s < 1.0


# ------------------------------------------------------------------
# build_color_palette
# ------------------------------------------------------------------


class TestColorPalette:
    """Tests for build_color_palette."""

    def test_returns_list_of_color_samples(self, sample_image: np.ndarray) -> None:
        """Should return a list of ColorSample objects."""
        palette = build_color_palette(sample_image, n_colors=4)
        assert isinstance(palette, list)
        assert all(isinstance(s, ColorSample) for s in palette)

    def test_correct_number_of_colors(self, sample_image: np.ndarray) -> None:
        """Should return at most n_colors clusters."""
        n = 5
        palette = build_color_palette(sample_image, n_colors=n)
        assert len(palette) <= n
        assert len(palette) >= 1

    def test_rgb_values_valid(self, sample_image: np.ndarray) -> None:
        """All palette RGB values should be in [0, 255]."""
        palette = build_color_palette(sample_image, n_colors=3)
        for sample in palette:
            for v in sample.rgb:
                assert 0 <= v <= 255

    def test_sorted_by_area_descending(self, sample_image: np.ndarray) -> None:
        """Palette should be sorted by area in descending order."""
        palette = build_color_palette(sample_image, n_colors=4)
        if len(palette) >= 2:
            areas = [s.area for s in palette]
            assert areas == sorted(areas, reverse=True)


# ------------------------------------------------------------------
# check_color_uniformity
# ------------------------------------------------------------------


class TestColorUniformity:
    """Tests for check_color_uniformity."""

    def test_uniform_image_passes(self) -> None:
        """A uniform image should be classified as uniform."""
        uniform = np.full((64, 64, 3), [100, 150, 200], dtype=np.uint8)
        result = check_color_uniformity(uniform, max_std=5.0)
        assert result["uniform"] is True

    def test_non_uniform_image_fails(self) -> None:
        """An image with high color variance should fail uniformity check."""
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img[:32, :, :] = [0, 0, 255]
        img[32:, :, :] = [255, 0, 0]
        result = check_color_uniformity(img, max_std=5.0)
        assert result["uniform"] is False

    def test_returns_expected_keys(self, sample_image: np.ndarray) -> None:
        """Result dict should contain all expected keys."""
        result = check_color_uniformity(sample_image)
        assert "uniform" in result
        assert "std_l" in result
        assert "std_a" in result
        assert "std_b" in result
        assert "mean_lab" in result

    def test_tolerance_threshold_overrides_max_std(self) -> None:
        """The tolerance_threshold parameter should override max_std."""
        uniform = np.full((64, 64, 3), [100, 150, 200], dtype=np.uint8)
        result = check_color_uniformity(uniform, max_std=0.001, tolerance_threshold=50.0)
        assert result["uniform"] is True


# ------------------------------------------------------------------
# check_color_tolerance
# ------------------------------------------------------------------


class TestColorTolerance:
    """Tests for check_color_tolerance."""

    def test_same_color_passes(self) -> None:
        """An image of a single color should pass tolerance check for that color."""
        # Create a uniform blue image and get its Lab value
        img = np.full((32, 32, 3), [200, 100, 50], dtype=np.uint8)
        sample = sample_color(img)
        mask = check_color_tolerance(img, reference_lab=sample.lab, tolerance=5.0, method="CIE76")
        assert mask.dtype == np.uint8
        # Most pixels should be within tolerance
        assert np.sum(mask == 255) > 0

    def test_different_color_fails(self) -> None:
        """An image far from the reference color should mostly fail."""
        img = np.full((32, 32, 3), [0, 0, 255], dtype=np.uint8)
        # Reference is black in Lab
        mask = check_color_tolerance(img, reference_lab=(0.0, 0.0, 0.0), tolerance=1.0, method="CIE76")
        # Almost no pixel should pass
        assert np.sum(mask == 255) < mask.size

    def test_output_is_binary_mask(self, sample_image: np.ndarray) -> None:
        """Output should be a binary mask with only 0 and 255."""
        mask = check_color_tolerance(
            sample_image, reference_lab=(50.0, 0.0, 0.0), tolerance=50.0, method="CIE76"
        )
        unique = set(np.unique(mask))
        assert unique.issubset({0, 255})


# ------------------------------------------------------------------
# classify_color
# ------------------------------------------------------------------


class TestClassifyColor:
    """Tests for classify_color."""

    def test_returns_result(self) -> None:
        """classify_color should return a ColorClassResult."""
        sample = ColorSample(
            lab=(100.0, 0.0, 0.0), rgb=(255, 255, 255),
            hsv=(0, 0, 255), std=(0, 0, 0), area=1,
        )
        result = classify_color(sample)
        assert result.class_name != ""
        assert 0.0 <= result.confidence <= 1.0
        assert result.delta_e >= 0.0

    def test_white_classifies_near_white(self) -> None:
        """A white sample should classify close to the white reference."""
        sample = ColorSample(
            lab=(100.0, 0.0, 0.0), rgb=(255, 255, 255),
            hsv=(0, 0, 255), std=(0, 0, 0), area=1,
        )
        result = classify_color(sample)
        # The default reference for white is (100, 0, 0)
        assert result.delta_e < 5.0


# ------------------------------------------------------------------
# compute_delta_e_map
# ------------------------------------------------------------------


class TestComputeDeltaEMap:
    """Tests for compute_delta_e_map."""

    def test_output_shape(self) -> None:
        """Delta-E map should have shape (H, W)."""
        img = np.full((32, 32, 3), [128, 128, 128], dtype=np.uint8)
        de_map = compute_delta_e_map(img, reference_color=(50.0, 0.0, 0.0), method="CIE76")
        assert de_map.shape == (32, 32)

    def test_output_dtype_float32(self) -> None:
        """Delta-E map should be float32."""
        img = np.full((32, 32, 3), [128, 128, 128], dtype=np.uint8)
        de_map = compute_delta_e_map(img, reference_color=(50.0, 0.0, 0.0), method="CIE76")
        assert de_map.dtype == np.float32

    def test_values_non_negative(self) -> None:
        """All Delta-E values should be non-negative."""
        img = np.full((32, 32, 3), [128, 128, 128], dtype=np.uint8)
        de_map = compute_delta_e_map(img, reference_color=(50.0, 0.0, 0.0), method="CIE76")
        assert np.all(de_map >= 0.0)
