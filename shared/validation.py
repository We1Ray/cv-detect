"""Input validation helpers for image-processing functions."""

from __future__ import annotations

import numpy as np

from shared.i18n import t


class ImageValidationError(ValueError):
    """Raised when an image array fails validation."""


def validate_image(img, name: str = "image") -> None:
    """Ensure *img* is a valid NumPy image array."""
    if img is None:
        raise ImageValidationError(t("validation.none_error", name=name))
    if not isinstance(img, np.ndarray):
        raise ImageValidationError(
            t("validation.ndarray_error", name=name, actual=type(img).__name__)
        )
    if img.ndim not in (2, 3):
        raise ImageValidationError(
            t("validation.ndim_error", name=name, ndim=img.ndim)
        )
    if img.size == 0:
        raise ImageValidationError(t("validation.empty_error", name=name))


def validate_kernel_size(k, name: str = "kernel size") -> None:
    """Ensure *k* is a positive odd integer."""
    if not isinstance(k, (int, float)) or int(k) != k:
        raise ImageValidationError(t("validation.integer_error", name=name, value=repr(k)))
    k = int(k)
    if k < 1:
        raise ImageValidationError(t("validation.min_error", name=name, min=1, value=k))
    if k % 2 == 0:
        raise ImageValidationError(t("validation.odd_error", name=name, value=k))


def validate_positive(val, name: str = "value") -> None:
    """Ensure *val* is a positive number."""
    if not isinstance(val, (int, float)):
        raise ImageValidationError(t("validation.numeric_error", name=name, actual=type(val).__name__))
    if val <= 0:
        raise ImageValidationError(t("validation.positive_error", name=name, value=val))


def validate_range(val, lo: float, hi: float, name: str = "value") -> None:
    """Ensure ``lo <= val <= hi``."""
    if not isinstance(val, (int, float)):
        raise ImageValidationError(t("validation.numeric_error", name=name, actual=type(val).__name__))
    if val < lo or val > hi:
        raise ImageValidationError(
            t("validation.range_error", name=name, min=lo, max=hi, value=val)
        )
