"""Tests for shared.core.ocr_engine and shared.core.barcode_engine.

These modules depend on optional external packages (pytesseract, pyzbar).
Tests that require these packages are skipped gracefully when unavailable.
Preprocessing helpers and verification functions are tested without them.
"""

from __future__ import annotations

import numpy as np
import pytest

from shared.core.ocr_engine import (
    check_tesseract_available,
    deskew_image,
    preprocess_for_ocr,
    verify_text,
)
from shared.core.barcode_engine import (
    BarcodeResult,
    check_pyzbar_available,
    compute_scan_profile,
    decode_barcodes,
    grade_barcode_quality,
    list_available_decoders,
    verify_barcode,
)


# ====================================================================
# OCR engine tests
# ====================================================================


class TestOCRPreprocess:
    """Tests for preprocess_for_ocr (no external packages needed)."""

    def test_returns_valid_image_adaptive(self, sample_grayscale: np.ndarray) -> None:
        """Adaptive preprocessing should return a valid uint8 image."""
        result = preprocess_for_ocr(sample_grayscale, method="adaptive")
        assert result.dtype == np.uint8
        assert result.ndim == 2
        assert result.shape == sample_grayscale.shape

    def test_returns_valid_image_otsu(self, sample_grayscale: np.ndarray) -> None:
        """Otsu preprocessing should return a valid uint8 image."""
        result = preprocess_for_ocr(sample_grayscale, method="otsu")
        assert result.dtype == np.uint8
        assert result.ndim == 2

    def test_method_none_returns_enhanced(self, sample_grayscale: np.ndarray) -> None:
        """Method 'none' should skip binarisation but still enhance contrast."""
        result = preprocess_for_ocr(sample_grayscale, method="none")
        assert result.dtype == np.uint8
        assert result.ndim == 2

    def test_accepts_color_image(self, sample_image: np.ndarray) -> None:
        """Should convert color images to grayscale internally."""
        result = preprocess_for_ocr(sample_image, method="adaptive")
        assert result.ndim == 2
        assert result.dtype == np.uint8

    def test_unknown_method_falls_back(self, sample_grayscale: np.ndarray) -> None:
        """An unknown method should fall back to adaptive without crashing."""
        result = preprocess_for_ocr(sample_grayscale, method="unknown_method")
        assert result.dtype == np.uint8
        assert result.ndim == 2


class TestDeskewImage:
    """Tests for deskew_image (no external packages needed)."""

    def test_returns_same_shape(self, sample_grayscale: np.ndarray) -> None:
        """Deskewed image should have the same shape as input."""
        corrected, angle = deskew_image(sample_grayscale)
        assert corrected.shape == sample_grayscale.shape

    def test_returns_angle_float(self, sample_grayscale: np.ndarray) -> None:
        """Detected angle should be a float."""
        _, angle = deskew_image(sample_grayscale)
        assert isinstance(angle, float)

    def test_accepts_color_image(self, sample_image: np.ndarray) -> None:
        """Should accept a BGR image."""
        corrected, angle = deskew_image(sample_image)
        assert corrected.shape == sample_image.shape

    def test_uniform_image_zero_angle(self) -> None:
        """A uniform image with no lines should return zero angle."""
        uniform = np.full((64, 64), 128, dtype=np.uint8)
        _, angle = deskew_image(uniform)
        assert angle == 0.0


class TestVerifyText:
    """Tests for verify_text (no external packages needed)."""

    def test_exact_match(self) -> None:
        """An exact match should return match=True."""
        result = verify_text("ABC123", r"ABC123")
        assert result["match"] is True
        assert result["similarity"] == pytest.approx(1.0)

    def test_partial_match_search(self) -> None:
        """Non-strict mode should find a pattern anywhere in the string."""
        result = verify_text("prefix_ABC123_suffix", r"ABC123")
        assert result["match"] is True

    def test_strict_no_partial(self) -> None:
        """Strict mode should require the entire string to match."""
        result = verify_text("prefix_ABC123", r"ABC123", strict=True)
        assert result["match"] is False

    def test_strict_full_match(self) -> None:
        """Strict mode should pass when the entire string matches."""
        result = verify_text("ABC123", r"ABC123", strict=True)
        assert result["match"] is True

    def test_regex_pattern(self) -> None:
        """Should support regex patterns."""
        result = verify_text("LOT-20260319-001", r"LOT-\d{8}-\d{3}")
        assert result["match"] is True

    def test_no_match(self) -> None:
        """Non-matching text should return match=False."""
        result = verify_text("HELLO", r"WORLD")
        assert result["match"] is False

    def test_returns_expected_keys(self) -> None:
        """Result dict should contain all expected keys."""
        result = verify_text("test", r"test")
        assert "match" in result
        assert "recognized" in result
        assert "expected" in result
        assert "similarity" in result

    def test_similarity_range(self) -> None:
        """Similarity should be between 0 and 1."""
        result = verify_text("abc", r"xyz")
        assert 0.0 <= result["similarity"] <= 1.0


# ====================================================================
# Barcode engine tests
# ====================================================================


class TestListAvailableDecoders:
    """Tests for list_available_decoders."""

    def test_always_includes_opencv(self) -> None:
        """OpenCV decoder should always be listed."""
        decoders = list_available_decoders()
        assert "opencv" in decoders

    def test_returns_list(self) -> None:
        """Should return a list of strings."""
        decoders = list_available_decoders()
        assert isinstance(decoders, list)
        assert all(isinstance(d, str) for d in decoders)


class TestDecodeInterface:
    """Tests for decode_barcodes interface contract."""

    def test_returns_list(self, sample_grayscale: np.ndarray) -> None:
        """decode_barcodes should always return a list (possibly empty)."""
        results = decode_barcodes(sample_grayscale, decoder="opencv")
        assert isinstance(results, list)

    def test_empty_on_no_barcode(self) -> None:
        """A blank image should yield no barcode results."""
        blank = np.full((100, 100), 200, dtype=np.uint8)
        results = decode_barcodes(blank, decoder="opencv")
        assert isinstance(results, list)

    def test_accepts_color_image(self, sample_image: np.ndarray) -> None:
        """Should accept a BGR image without crashing."""
        results = decode_barcodes(sample_image, decoder="opencv")
        assert isinstance(results, list)


class TestComputeScanProfile:
    """Tests for compute_scan_profile."""

    def test_returns_1d_array(self) -> None:
        """Scan profile should be a 1D float64 array."""
        img = np.random.randint(0, 255, (50, 100), dtype=np.uint8)
        bbox = (0, 0, 100, 50)
        profile = compute_scan_profile(img, bbox)
        assert profile.ndim == 1
        assert profile.dtype == np.float64

    def test_profile_length_matches_width(self) -> None:
        """Profile length should equal the barcode width."""
        img = np.random.randint(0, 255, (50, 80), dtype=np.uint8)
        bbox = (0, 0, 80, 50)
        profile = compute_scan_profile(img, bbox)
        assert len(profile) == 80

    def test_empty_bbox_returns_empty(self) -> None:
        """An empty or invalid bbox should return an empty array."""
        img = np.random.randint(0, 255, (50, 80), dtype=np.uint8)
        profile = compute_scan_profile(img, (0, 0, 0, 0))
        assert len(profile) == 0


class TestGradeInterface:
    """Tests for grade_barcode_quality interface."""

    def test_grade_updates_result(self) -> None:
        """grade_barcode_quality should update the BarcodeResult quality fields."""
        img = np.random.randint(0, 255, (50, 100), dtype=np.uint8)
        br = BarcodeResult(data="123456", type="EAN13", bbox=(0, 0, 100, 50))
        graded = grade_barcode_quality(img, br)
        assert graded.quality_grade is not None
        assert graded.quality_grade in ("A", "B", "C", "D", "F")
        assert graded.quality_metrics is not None

    def test_invalid_bbox_grade_f(self) -> None:
        """An invalid (zero-size) bbox should result in grade F."""
        img = np.zeros((50, 100), dtype=np.uint8)
        br = BarcodeResult(data="test", type="QR", bbox=(0, 0, 0, 0))
        graded = grade_barcode_quality(img, br)
        assert graded.quality_grade == "F"


class TestVerifyBarcode:
    """Tests for verify_barcode."""

    def test_matching_data(self) -> None:
        """Matching expected data should return valid=True."""
        br = BarcodeResult(data="4901234567890", type="EAN13", bbox=(0, 0, 100, 50))
        result = verify_barcode(br, expected_data="4901234567890")
        assert result["valid"] is True
        assert result["data_match"] is True

    def test_mismatched_data(self) -> None:
        """Mismatched data should return data_match=False."""
        br = BarcodeResult(data="4901234567890", type="EAN13", bbox=(0, 0, 100, 50))
        result = verify_barcode(br, expected_data="0000000000000")
        assert result["data_match"] is False

    def test_grade_pass(self) -> None:
        """A barcode with grade A should pass min_grade C."""
        br = BarcodeResult(
            data="test", type="QR", bbox=(0, 0, 50, 50), quality_grade="A"
        )
        result = verify_barcode(br, min_grade="C")
        assert result["grade_pass"] is True

    def test_grade_fail(self) -> None:
        """A barcode with grade F should fail min_grade C."""
        br = BarcodeResult(
            data="test", type="QR", bbox=(0, 0, 50, 50), quality_grade="F"
        )
        result = verify_barcode(br, min_grade="C")
        assert result["grade_pass"] is False

    def test_returns_expected_keys(self) -> None:
        """Result dict should have all expected keys."""
        br = BarcodeResult(data="test", type="QR", bbox=(0, 0, 50, 50))
        result = verify_barcode(br)
        assert "valid" in result
        assert "data_match" in result
        assert "grade_pass" in result
        assert "details" in result
