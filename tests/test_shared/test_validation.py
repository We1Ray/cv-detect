"""Tests for shared.validation -- input validation helpers."""

from __future__ import annotations

import numpy as np
import pytest

from shared.validation import (
    ImageValidationError,
    validate_image,
    validate_kernel_size,
    validate_positive,
    validate_range,
)


# ------------------------------------------------------------------
# validate_image
# ------------------------------------------------------------------

class TestValidateImage:
    """Tests for the validate_image function."""

    def test_accepts_2d_array(self) -> None:
        """A 2D uint8 array (grayscale) should pass validation."""
        img = np.zeros((64, 64), dtype=np.uint8)
        validate_image(img)  # Should not raise

    def test_accepts_3d_array(self) -> None:
        """A 3D uint8 array (H, W, C) should pass validation."""
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        validate_image(img)  # Should not raise

    def test_rejects_none(self) -> None:
        """None should be rejected with ImageValidationError."""
        with pytest.raises(ImageValidationError, match="None"):
            validate_image(None)

    def test_rejects_non_ndarray(self) -> None:
        """A plain Python list should be rejected."""
        with pytest.raises(ImageValidationError, match="numpy.ndarray"):
            validate_image([[1, 2], [3, 4]])

    def test_rejects_1d_array(self) -> None:
        """A 1D array is not a valid image."""
        with pytest.raises(ImageValidationError, match="維度"):
            validate_image(np.zeros(10))

    def test_rejects_4d_array(self) -> None:
        """A 4D array is not a valid image."""
        with pytest.raises(ImageValidationError, match="維度"):
            validate_image(np.zeros((1, 3, 64, 64)))

    def test_rejects_empty_array(self) -> None:
        """An empty (size=0) array should be rejected."""
        with pytest.raises(ImageValidationError, match="空"):
            validate_image(np.zeros((0, 64), dtype=np.uint8))

    def test_custom_name_in_error_message(self) -> None:
        """The name parameter should appear in the error message."""
        with pytest.raises(ImageValidationError, match="my_img"):
            validate_image(None, name="my_img")


# ------------------------------------------------------------------
# validate_positive
# ------------------------------------------------------------------

class TestValidatePositive:
    """Tests for the validate_positive function."""

    def test_accepts_positive_int(self) -> None:
        """A positive integer should pass."""
        validate_positive(5)  # Should not raise

    def test_accepts_positive_float(self) -> None:
        """A positive float should pass."""
        validate_positive(0.001)  # Should not raise

    def test_rejects_zero(self) -> None:
        """Zero is not positive and should be rejected."""
        with pytest.raises(ImageValidationError):
            validate_positive(0)

    def test_rejects_negative(self) -> None:
        """Negative numbers should be rejected."""
        with pytest.raises(ImageValidationError):
            validate_positive(-3.5)

    def test_rejects_string(self) -> None:
        """A string is not a number and should be rejected."""
        with pytest.raises(ImageValidationError, match="數值"):
            validate_positive("5")


# ------------------------------------------------------------------
# validate_kernel_size
# ------------------------------------------------------------------

class TestValidateKernelSize:
    """Tests for the validate_kernel_size function."""

    @pytest.mark.parametrize("k", [1, 3, 5, 7, 11])
    def test_accepts_positive_odd(self, k: int) -> None:
        """Positive odd integers should pass."""
        validate_kernel_size(k)  # Should not raise

    def test_rejects_even(self) -> None:
        """Even kernel sizes should be rejected."""
        with pytest.raises(ImageValidationError, match="奇數"):
            validate_kernel_size(4)

    def test_rejects_zero(self) -> None:
        """Zero should be rejected."""
        with pytest.raises(ImageValidationError):
            validate_kernel_size(0)

    def test_rejects_negative(self) -> None:
        """Negative values should be rejected."""
        with pytest.raises(ImageValidationError):
            validate_kernel_size(-3)


# ------------------------------------------------------------------
# validate_range
# ------------------------------------------------------------------

class TestValidateRange:
    """Tests for the validate_range function."""

    def test_accepts_value_in_range(self) -> None:
        """A value within [lo, hi] should pass."""
        validate_range(0.5, 0.0, 1.0)  # Should not raise

    def test_accepts_boundary_values(self) -> None:
        """Boundary values (lo and hi) should pass."""
        validate_range(0.0, 0.0, 1.0)
        validate_range(1.0, 0.0, 1.0)

    def test_rejects_below_range(self) -> None:
        """A value below lo should be rejected."""
        with pytest.raises(ImageValidationError, match="範圍"):
            validate_range(-0.1, 0.0, 1.0)

    def test_rejects_above_range(self) -> None:
        """A value above hi should be rejected."""
        with pytest.raises(ImageValidationError, match="範圍"):
            validate_range(1.1, 0.0, 1.0)

    def test_rejects_non_numeric(self) -> None:
        """Non-numeric types should be rejected."""
        with pytest.raises(ImageValidationError, match="數值"):
            validate_range("abc", 0.0, 1.0)
