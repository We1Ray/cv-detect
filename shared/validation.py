"""Input validation helpers for image-processing functions."""

from __future__ import annotations

import numpy as np


class ImageValidationError(ValueError):
    """Raised when an image array fails validation."""


def validate_image(img, name: str = "image") -> None:
    """Ensure *img* is a valid NumPy image array."""
    if img is None:
        raise ImageValidationError(f"{name} 不可為 None")
    if not isinstance(img, np.ndarray):
        raise ImageValidationError(
            f"{name} 必須是 numpy.ndarray，收到 {type(img).__name__}"
        )
    if img.ndim not in (2, 3):
        raise ImageValidationError(
            f"{name} 維度必須為 2 或 3，收到 ndim={img.ndim}"
        )
    if img.size == 0:
        raise ImageValidationError(f"{name} 不可為空 (size=0)")


def validate_kernel_size(k, name: str = "kernel size") -> None:
    """Ensure *k* is a positive odd integer."""
    if not isinstance(k, (int, float)) or int(k) != k:
        raise ImageValidationError(f"{name} 必須是整數，收到 {k!r}")
    k = int(k)
    if k < 1:
        raise ImageValidationError(f"{name} 必須 >= 1，收到 {k}")
    if k % 2 == 0:
        raise ImageValidationError(f"{name} 必須是奇數，收到 {k}")


def validate_positive(val, name: str = "value") -> None:
    """Ensure *val* is a positive number."""
    if not isinstance(val, (int, float)):
        raise ImageValidationError(f"{name} 必須是數值，收到 {type(val).__name__}")
    if val <= 0:
        raise ImageValidationError(f"{name} 必須 > 0，收到 {val}")


def validate_range(val, lo: float, hi: float, name: str = "value") -> None:
    """Ensure ``lo <= val <= hi``."""
    if not isinstance(val, (int, float)):
        raise ImageValidationError(f"{name} 必須是數值，收到 {type(val).__name__}")
    if val < lo or val > hi:
        raise ImageValidationError(
            f"{name} 必須在 [{lo}, {hi}] 範圍內，收到 {val}"
        )
