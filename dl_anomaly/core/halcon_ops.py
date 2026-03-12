"""
core/halcon_ops.py - Comprehensive HALCON-style operator library using OpenCV.

Provides a wide range of image processing operations modelled after
MVTec HALCON operators, fully implemented on top of OpenCV and NumPy.

Categories:
    1.  Image Arithmetic
    2.  Filters
    3.  Edge Detection
    4.  Gray Morphology
    5.  Geometric Transforms
    6.  Color Space
    7.  Texture / Feature Images
    8.  Contours (XLD)
    9.  Matching
    10. Measurement
    11. Drawing
    12. Barcode / QR Code
    13. Segmentation
    14. Feature Points
    15. Line / Circle Detection (Hough)
    16. Miscellaneous
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..', '..'))
from shared.validation import validate_image, validate_positive
from shared.op_logger import log_operation

logger = logging.getLogger(__name__)

# ====================================================================== #
#  Named constants                                                        #
# ====================================================================== #

DEFAULT_KSIZE = 3
DEFAULT_SIGMA = 1.0
DEFAULT_CANNY_LOW = 50.0
DEFAULT_CANNY_HIGH = 150.0
DEFAULT_MORPH_KSIZE = 5
DEFAULT_BILATERAL_D = 9
DEFAULT_BILATERAL_SIGMA = 75.0
DEFAULT_HOUGH_THRESHOLD = 100
DEFAULT_NCC_THRESHOLD = 0.8
DEFAULT_CLAHE_CLIP = 2.0
DEFAULT_CLAHE_TILE = 8

# ====================================================================== #
#  Lightweight Region dataclass                                           #
# ====================================================================== #


@dataclass
class Region:
    """A binary region (mask) with cached geometric properties.

    Attributes:
        mask:   uint8 binary mask (0 or 255) with the same spatial extent
                as the canvas it was created on.
        labels: optional label image from ``connectedComponents``.
        area:   total number of non-zero pixels.
        cx:     centroid x coordinate.
        cy:     centroid y coordinate.
    """

    mask: np.ndarray
    labels: Optional[np.ndarray] = field(default=None, repr=False)
    area: int = 0
    cx: float = 0.0
    cy: float = 0.0

    def __post_init__(self) -> None:
        if self.area == 0 and self.mask is not None:
            self.area = int(np.count_nonzero(self.mask))
            if self.area > 0:
                ys, xs = np.nonzero(self.mask)
                self.cx = float(np.mean(xs))
                self.cy = float(np.mean(ys))


# ====================================================================== #
#  Internal helpers                                                       #
# ====================================================================== #


def _ensure_gray(img: np.ndarray) -> np.ndarray:
    """Convert image to single-channel grayscale if it is not already."""
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[2] == 1:
        return img[:, :, 0]
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _ensure_uint8(img: np.ndarray) -> np.ndarray:
    """Clip and convert an array to ``uint8``."""
    return np.clip(img, 0, 255).astype(np.uint8)


def _odd(k: int) -> int:
    """Return *k* if odd, otherwise *k + 1*."""
    k = max(int(k), 1)
    return k if k % 2 == 1 else k + 1


def _ksize_from_sigma(sigma: float) -> int:
    """Compute a suitable odd kernel size from a Gaussian sigma."""
    k = int(math.ceil(sigma * 6)) | 1  # at least 6*sigma, forced odd
    return max(k, 3)


def _wrap_cv2_error(context: str, exc: cv2.error) -> None:
    """Log an OpenCV error with context and re-raise as ``RuntimeError``."""
    msg = f"{context}: OpenCV error - {exc.err}" if hasattr(exc, 'err') else f"{context}: {exc}"
    logger.error(msg)
    raise RuntimeError(msg) from exc


# ====================================================================== #
#  1. Image Arithmetic                                                    #
# ====================================================================== #


def add_image(
    img1: np.ndarray,
    img2: np.ndarray,
    mult: float = 1.0,
    add: float = 0.0,
) -> np.ndarray:
    """Weighted addition: ``result = img1 * mult + img2 * mult + add``.

    Uses ``cv2.addWeighted`` under the hood.
    """
    result = cv2.addWeighted(
        img1.astype(np.float64),
        mult,
        img2.astype(np.float64),
        mult,
        add,
    )
    return _ensure_uint8(result)


def sub_image(
    img1: np.ndarray,
    img2: np.ndarray,
    mult: float = 1.0,
    add: float = 128.0,
) -> np.ndarray:
    """Subtract with offset: ``result = (img1 - img2) * mult + add``."""
    diff = (img1.astype(np.float64) - img2.astype(np.float64)) * mult + add
    return _ensure_uint8(diff)


def mult_image(
    img1: np.ndarray,
    img2: np.ndarray,
    mult: float = 1.0,
    add: float = 0.0,
) -> np.ndarray:
    """Pixel-wise multiplication: ``result = img1 * img2 * mult / 255 + add``."""
    product = img1.astype(np.float64) * img2.astype(np.float64) / 255.0 * mult + add
    return _ensure_uint8(product)


def abs_image(img: np.ndarray) -> np.ndarray:
    """Absolute value of each pixel, clipped to ``uint8``."""
    return _ensure_uint8(np.abs(img.astype(np.float64)))


def invert_image(img: np.ndarray) -> np.ndarray:
    """Invert a ``uint8`` image: ``255 - img``."""
    return (255 - img.astype(np.int16)).clip(0, 255).astype(np.uint8)


def scale_image(img: np.ndarray, mult: float, add: float) -> np.ndarray:
    """Scale gray values: ``result = img * mult + add``."""
    validate_image(img)
    result = img.astype(np.float64) * mult + add
    return _ensure_uint8(result)


def log_image(img: np.ndarray, base: str = "e") -> np.ndarray:
    """Logarithmic gray-value transform for enhancing dark regions.

    ``result = log_base(1 + img) * (255 / log_base(256))``

    Args:
        base: ``"e"`` (natural log), ``"2"``, or ``"10"``.
    """
    f = img.astype(np.float64)
    if base == "2":
        log_val = np.log2(1.0 + f)
        scale = 255.0 / np.log2(256.0)
    elif base == "10":
        log_val = np.log10(1.0 + f)
        scale = 255.0 / np.log10(256.0)
    else:  # natural log
        log_val = np.log(1.0 + f)
        scale = 255.0 / np.log(256.0)
    return _ensure_uint8(log_val * scale)


def exp_image(img: np.ndarray, base: str = "e") -> np.ndarray:
    """Exponential gray-value transform for enhancing bright regions.

    ``result = (base^(img/255) - 1) / (base - 1) * 255``

    Args:
        base: ``"e"`` (natural exp), ``"2"``, or ``"10"``.
    """
    f = img.astype(np.float64) / 255.0
    base_val = {"2": 2.0, "10": 10.0}.get(base, math.e)
    exp_val = np.power(base_val, f) - 1.0
    exp_val = exp_val / (base_val - 1.0) * 255.0
    return _ensure_uint8(exp_val)


def gamma_image(img: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Gamma correction: ``result = 255 * (img / 255) ^ gamma``.

    Args:
        gamma: Gamma value.  ``< 1`` brightens dark regions,
               ``> 1`` darkens bright regions.
    """
    validate_image(img)
    validate_positive(gamma, "gamma")
    f = img.astype(np.float64) / 255.0
    corrected = np.power(f, gamma) * 255.0
    return _ensure_uint8(corrected)


def min_image(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """Pixel-wise minimum of two images."""
    return np.minimum(img1, img2)


def max_image(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """Pixel-wise maximum of two images."""
    return np.maximum(img1, img2)


def crop_rectangle(
    img: np.ndarray, x: int, y: int, w: int, h: int
) -> np.ndarray:
    """Crop a rectangular region ``img[y:y+h, x:x+w]``."""
    return img[y : y + h, x : x + w].copy()


# ====================================================================== #
#  1b. Domain Operations                                                  #
# ====================================================================== #


def _extract_mask(region) -> np.ndarray:
    """Extract a binary uint8 mask from either Region class.

    Supports:
    - ``halcon_ops.Region`` (has ``.mask``).
    - ``core.region.Region`` (has ``.to_binary_mask()`` or ``.labels``).
    """
    if hasattr(region, "mask") and region.mask is not None:
        return region.mask
    if hasattr(region, "to_binary_mask"):
        return region.to_binary_mask()
    if hasattr(region, "labels") and region.labels is not None:
        return ((region.labels > 0).astype(np.uint8)) * 255
    raise TypeError(
        f"Cannot extract mask from region of type {type(region).__name__}"
    )


def reduce_domain(
    img: np.ndarray, region, fill_value: int = 0
) -> np.ndarray:
    """Restrict an image to the domain defined by *region*.

    Pixels outside the region are set to *fill_value*.  Analogous to
    HALCON ``reduce_domain``.

    Args:
        img:        Source image (any number of channels).
        region:     Region object providing the binary mask.
        fill_value: Gray value for pixels outside the region (default 0).

    Returns:
        A copy of *img* where pixels outside the mask equal *fill_value*.
    """
    mask = _extract_mask(region)
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(
            mask, (img.shape[1], img.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    result = np.full_like(img, fill_value)
    result[mask > 0] = img[mask > 0]
    return result


def crop_domain(img: np.ndarray, region) -> np.ndarray:
    """Crop an image to the bounding box of *region*.

    Analogous to HALCON ``crop_domain``.

    Returns:
        Cropped image containing only the bounding-box area.
    """
    mask = _extract_mask(region)
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(
            mask, (img.shape[1], img.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return img[:1, :1].copy()
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    return img[y1:y2, x1:x2].copy()


# ====================================================================== #
#  2. Filters                                                             #
# ====================================================================== #


def mean_image(img: np.ndarray, ksize: int = DEFAULT_KSIZE) -> np.ndarray:
    """Mean (box) filter."""
    validate_image(img)
    k = _odd(ksize)
    return cv2.blur(img, (k, k))


def median_image(img: np.ndarray, ksize: int = DEFAULT_KSIZE) -> np.ndarray:
    """Median filter (kernel size forced to be odd)."""
    validate_image(img)
    k = _odd(ksize)
    return cv2.medianBlur(img, k)


def gauss_filter(img: np.ndarray, sigma: float = DEFAULT_SIGMA) -> np.ndarray:
    """Gaussian low-pass filter.  Kernel size is derived from *sigma*."""
    validate_image(img)
    validate_positive(sigma, "sigma")
    k = _ksize_from_sigma(sigma)
    return cv2.GaussianBlur(img, (k, k), sigmaX=sigma, sigmaY=sigma)


def gauss_blur(img: np.ndarray, ksize: int = DEFAULT_KSIZE) -> np.ndarray:
    """Gaussian blur with explicit kernel size (sigma auto-computed by OpenCV)."""
    validate_image(img)
    k = _odd(ksize)
    return cv2.GaussianBlur(img, (k, k), 0)


def binomial_filter(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Binomial filter (Gaussian with ``sigma=0``, so OpenCV auto-computes sigma)."""
    k = _odd(ksize)
    return cv2.GaussianBlur(img, (k, k), sigmaX=0, sigmaY=0)


@log_operation(logger)
def bilateral_filter(
    img: np.ndarray,
    d: int = DEFAULT_BILATERAL_D,
    sigma_color: float = DEFAULT_BILATERAL_SIGMA,
    sigma_space: float = DEFAULT_BILATERAL_SIGMA,
) -> np.ndarray:
    """Edge-preserving bilateral filter."""
    validate_image(img)
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


def sharpen_image(img: np.ndarray, amount: float = 1.0) -> np.ndarray:
    """Unsharp-mask sharpening: ``img + amount * (img - blurred)``."""
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=3)
    sharpened = img.astype(np.float64) + amount * (
        img.astype(np.float64) - blurred.astype(np.float64)
    )
    return _ensure_uint8(sharpened)


def emphasize(img: np.ndarray, ksize: int = 7, factor: float = 1.0) -> np.ndarray:
    """High-frequency emphasis: ``img + factor * (img - lowpass)``."""
    k = _odd(ksize)
    lowpass = cv2.GaussianBlur(img, (k, k), 0)
    result = img.astype(np.float64) + factor * (
        img.astype(np.float64) - lowpass.astype(np.float64)
    )
    return _ensure_uint8(result)


def laplace_filter(img: np.ndarray) -> np.ndarray:
    """Laplacian filter (``CV_64F``), absolute value clipped to ``uint8``."""
    gray = _ensure_gray(img)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return _ensure_uint8(np.abs(lap))


def sobel_filter(img: np.ndarray, direction: str = "both") -> np.ndarray:
    """Sobel edge filter.

    Args:
        direction: ``"x"``, ``"y"``, or ``"both"`` (magnitude).
    """
    validate_image(img)
    gray = _ensure_gray(img)
    if direction == "x":
        return _ensure_uint8(np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)))
    if direction == "y":
        return _ensure_uint8(np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)))
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sx ** 2 + sy ** 2)
    return _ensure_uint8(mag)


def prewitt_filter(img: np.ndarray) -> np.ndarray:
    """Prewitt edge magnitude using manual 3x3 kernels."""
    validate_image(img)
    gray = _ensure_gray(img).astype(np.float64)
    kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
    ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float64)
    gx = cv2.filter2D(gray, cv2.CV_64F, kx)
    gy = cv2.filter2D(gray, cv2.CV_64F, ky)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return _ensure_uint8(mag)


def derivative_gauss(
    img: np.ndarray, sigma: float = 1.0, component: str = "x"
) -> np.ndarray:
    """Gaussian smoothing followed by first derivative (Sobel).

    Args:
        component: ``"x"`` or ``"y"``.
    """
    gray = _ensure_gray(img)
    blurred = gauss_filter(gray, sigma)
    if component == "y":
        deriv = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    else:
        deriv = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    return _ensure_uint8(np.abs(deriv))


# ====================================================================== #
#  3. Edge Detection                                                      #
# ====================================================================== #


@log_operation(logger)
def edges_canny(
    img: np.ndarray,
    low: float = DEFAULT_CANNY_LOW,
    high: float = DEFAULT_CANNY_HIGH,
    sigma: float = DEFAULT_SIGMA,
) -> np.ndarray:
    """Canny edge detector with optional Gaussian pre-smoothing."""
    validate_image(img)
    gray = _ensure_gray(img)
    if sigma > 0:
        k = _ksize_from_sigma(sigma)
        gray = cv2.GaussianBlur(gray, (k, k), sigma)
    return cv2.Canny(gray, low, high)


def edges_sobel(img: np.ndarray, direction: str = "both") -> np.ndarray:
    """Sobel edge magnitude (alias of :func:`sobel_filter`)."""
    return sobel_filter(img, direction)


def zero_crossing(img: np.ndarray) -> np.ndarray:
    """Detect zero-crossings in the Laplacian of the image.

    Returns a binary ``uint8`` mask (0 / 255) marking locations where the
    Laplacian changes sign.
    """
    gray = _ensure_gray(img)
    lap = cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F)

    # A zero-crossing occurs where neighbouring pixels have different signs.
    # We check four-connected neighbours.
    mask = np.zeros(lap.shape, dtype=np.uint8)
    # Horizontal neighbour sign change
    sign_change_h = (lap[:, :-1] * lap[:, 1:]) < 0
    mask[:, :-1] |= (sign_change_h.astype(np.uint8) * 255)
    mask[:, 1:] |= (sign_change_h.astype(np.uint8) * 255)
    # Vertical neighbour sign change
    sign_change_v = (lap[:-1, :] * lap[1:, :]) < 0
    mask[:-1, :] |= (sign_change_v.astype(np.uint8) * 255)
    mask[1:, :] |= (sign_change_v.astype(np.uint8) * 255)
    return mask


# ====================================================================== #
#  4. Gray Morphology                                                     #
# ====================================================================== #


_SHAPE_MAP = {
    "rectangle": cv2.MORPH_RECT,
    "rect": cv2.MORPH_RECT,
    "ellipse": cv2.MORPH_ELLIPSE,
    "circle": cv2.MORPH_ELLIPSE,
    "octagon": cv2.MORPH_ELLIPSE,  # approximate with ellipse
    "cross": cv2.MORPH_CROSS,
}


def _morph_kernel(ksize: int, shape: str = "rectangle") -> np.ndarray:
    k = _odd(ksize)
    morph = _SHAPE_MAP.get(shape, cv2.MORPH_RECT)
    return cv2.getStructuringElement(morph, (k, k))


def gray_erosion(img: np.ndarray, ksize: int = DEFAULT_KSIZE) -> np.ndarray:
    """Grayscale erosion (minimum filter)."""
    validate_image(img)
    gray = _ensure_gray(img)
    return cv2.erode(gray, _morph_kernel(ksize))


def gray_dilation(img: np.ndarray, ksize: int = DEFAULT_KSIZE) -> np.ndarray:
    """Grayscale dilation (maximum filter)."""
    validate_image(img)
    gray = _ensure_gray(img)
    return cv2.dilate(gray, _morph_kernel(ksize))


def gray_opening(img: np.ndarray, ksize: int = DEFAULT_KSIZE) -> np.ndarray:
    """Grayscale morphological opening (erosion then dilation)."""
    validate_image(img)
    gray = _ensure_gray(img)
    return cv2.morphologyEx(gray, cv2.MORPH_OPEN, _morph_kernel(ksize))


def gray_closing(img: np.ndarray, ksize: int = DEFAULT_KSIZE) -> np.ndarray:
    """Grayscale morphological closing (dilation then erosion)."""
    validate_image(img)
    gray = _ensure_gray(img)
    return cv2.morphologyEx(gray, cv2.MORPH_CLOSE, _morph_kernel(ksize))


def top_hat(img: np.ndarray, ksize: int = DEFAULT_MORPH_KSIZE) -> np.ndarray:
    """Top-hat (white hat): ``img - opening(img)``."""
    gray = _ensure_gray(img)
    return cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, _morph_kernel(ksize))


def bottom_hat(img: np.ndarray, ksize: int = DEFAULT_MORPH_KSIZE) -> np.ndarray:
    """Bottom-hat (black hat): ``closing(img) - img``."""
    gray = _ensure_gray(img)
    return cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, _morph_kernel(ksize))


def gray_opening_shape(
    img: np.ndarray, kw: int = 7, kh: int = 7, shape: str = "octagon",
) -> np.ndarray:
    """Gray opening with configurable kernel size and shape.

    Equivalent to HALCON ``gray_opening_shape``.
    """
    gray = _ensure_gray(img)
    kernel = _morph_kernel(max(kw, kh), shape)
    return cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)


def gray_closing_shape(
    img: np.ndarray, kw: int = 7, kh: int = 7, shape: str = "octagon",
) -> np.ndarray:
    """Gray closing with configurable kernel size and shape.

    Equivalent to HALCON ``gray_closing_shape``.
    """
    gray = _ensure_gray(img)
    kernel = _morph_kernel(max(kw, kh), shape)
    return cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)


@log_operation(logger)
def dyn_threshold(
    img_light: np.ndarray,
    img_dark: np.ndarray,
    offset: float = 75,
    mode: str = "not_equal",
) -> np.ndarray:
    """Dynamic threshold segmentation.

    Compares *img_light* and *img_dark* pixel-by-pixel.

    Parameters
    ----------
    img_light, img_dark : ndarray
        Two single-channel images of the same size (e.g. opening / closing).
    offset : float
        Threshold offset.  The absolute difference must exceed this value.
    mode : str
        ``"light"`` — keep pixels where img_light > img_dark + offset.
        ``"dark"``  — keep pixels where img_dark > img_light + offset.
        ``"not_equal"`` — keep pixels where |diff| > offset.
        ``"equal"`` — keep pixels where |diff| <= offset.

    Returns
    -------
    ndarray
        Binary uint8 mask (0 / 255).
    """
    validate_image(img_light, "img_light")
    validate_image(img_dark, "img_dark")
    a = img_light.astype(np.float32)
    b = img_dark.astype(np.float32)
    diff = a - b

    if mode == "light":
        mask = diff > offset
    elif mode == "dark":
        mask = diff < -offset
    elif mode == "not_equal":
        mask = np.abs(diff) > offset
    elif mode == "equal":
        mask = np.abs(diff) <= offset
    else:
        mask = np.abs(diff) > offset

    return (mask.astype(np.uint8)) * 255


@log_operation(logger)
def var_threshold(
    img: np.ndarray,
    width: int = 15,
    height: int = 15,
    std_mult: float = 0.2,
    absolute_threshold: float = 2.0,
    light_dark: str = "dark",
) -> np.ndarray:
    """Variable threshold segmentation for uneven illumination.

    Uses local mean and local standard deviation to compute per-pixel
    thresholds, similar to HALCON ``var_threshold``.

    For ``light_dark='dark'``:
        pixel is foreground if ``pixel < local_mean - std_mult * local_std``
        **and** ``local_mean - pixel > absolute_threshold``.
    For ``light_dark='light'``:
        pixel is foreground if ``pixel > local_mean + std_mult * local_std``
        **and** ``pixel - local_mean > absolute_threshold``.
    For ``light_dark='equal'``:
        pixel is foreground if ``|pixel - local_mean| <= std_mult * local_std``
        **or** ``|pixel - local_mean| <= absolute_threshold``.
    For ``light_dark='not_equal'``:
        pixel is foreground if ``|pixel - local_mean| > std_mult * local_std``
        **and** ``|pixel - local_mean| > absolute_threshold``.

    Returns:
        Binary uint8 mask (0 / 255).
    """
    validate_image(img)
    gray = _ensure_gray(img).astype(np.float64)
    kw = _odd(width)
    kh = _odd(height)

    local_mean = cv2.blur(gray, (kw, kh))
    local_sq = cv2.blur(gray ** 2, (kw, kh))
    local_std = np.sqrt(np.maximum(local_sq - local_mean ** 2, 0.0))

    diff = gray - local_mean
    thresh_std = std_mult * local_std

    if light_dark == "dark":
        mask = (diff < -thresh_std) & (np.abs(diff) > absolute_threshold)
    elif light_dark == "light":
        mask = (diff > thresh_std) & (np.abs(diff) > absolute_threshold)
    elif light_dark == "equal":
        mask = (np.abs(diff) <= thresh_std) | (np.abs(diff) <= absolute_threshold)
    else:  # not_equal
        mask = (np.abs(diff) > thresh_std) & (np.abs(diff) > absolute_threshold)

    return (mask.astype(np.uint8)) * 255


@log_operation(logger)
def local_threshold(
    img: np.ndarray,
    method: str = "adapted_std_deviation",
    light_dark: str = "dark",
    ksize: int = 15,
    scale: float = 0.2,
) -> np.ndarray:
    """Local (adaptive) threshold using local statistics.

    A convenience wrapper around different local thresholding strategies.

    Args:
        method: ``"adapted_std_deviation"`` uses mean +/- scale * std.
                ``"mean"`` uses mean +/- scale * 255.
        light_dark: ``"dark"`` segments dark objects, ``"light"`` segments
                    bright objects, ``"not_equal"`` segments both.
        ksize: Local neighbourhood size (forced odd).
        scale: Scale factor applied to the local statistic.

    Returns:
        Binary uint8 mask (0 / 255).
    """
    gray = _ensure_gray(img).astype(np.float64)
    k = _odd(ksize)

    local_mean = cv2.blur(gray, (k, k))

    if method == "mean":
        offset = scale * 255.0
        if light_dark == "dark":
            mask = gray < (local_mean - offset)
        elif light_dark == "light":
            mask = gray > (local_mean + offset)
        else:  # not_equal
            mask = np.abs(gray - local_mean) > offset
    else:  # adapted_std_deviation
        local_sq = cv2.blur(gray ** 2, (k, k))
        local_std = np.sqrt(np.maximum(local_sq - local_mean ** 2, 0.0))
        offset = scale * local_std
        if light_dark == "dark":
            mask = gray < (local_mean - offset)
        elif light_dark == "light":
            mask = gray > (local_mean + offset)
        else:  # not_equal
            mask = np.abs(gray - local_mean) > offset

    return (mask.astype(np.uint8)) * 255


# ====================================================================== #
#  FFT / Frequency Domain                                                 #
# ====================================================================== #


def fft_image(img: np.ndarray) -> np.ndarray:
    """Compute the FFT magnitude spectrum of a grayscale image.

    Returns a ``uint8`` image of the log-scaled magnitude spectrum,
    with the DC component shifted to the centre.
    """
    gray = _ensure_gray(img).astype(np.float64)
    dft = np.fft.fft2(gray)
    dft_shift = np.fft.fftshift(dft)
    magnitude = np.abs(dft_shift)
    # Log scale for better visibility
    magnitude = np.log1p(magnitude)
    # Normalise to 0..255
    m_min, m_max = magnitude.min(), magnitude.max()
    if m_max - m_min > 0:
        magnitude = (magnitude - m_min) / (m_max - m_min) * 255.0
    return _ensure_uint8(magnitude)


def gen_gauss_filter(
    shape: Tuple[int, int], sigma: float
) -> np.ndarray:
    """Generate a 2-D Gaussian low-pass filter in the frequency domain.

    Args:
        shape: ``(rows, cols)`` of the filter (must match image size).
        sigma: Standard deviation of the Gaussian.

    Returns:
        float64 array with values in ``[0, 1]``.
    """
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    y = np.arange(rows) - crow
    x = np.arange(cols) - ccol
    xx, yy = np.meshgrid(x, y)
    d_sq = xx.astype(np.float64) ** 2 + yy.astype(np.float64) ** 2
    gauss = np.exp(-d_sq / (2.0 * sigma * sigma))
    return gauss


def freq_filter(
    img: np.ndarray, filter_type: str = "lowpass", cutoff: float = 30.0
) -> np.ndarray:
    """Apply a frequency-domain Gaussian filter.

    Args:
        filter_type: ``"lowpass"`` or ``"highpass"``.
        cutoff: Standard deviation (sigma) of the Gaussian in frequency
                domain — controls the cut-off frequency.

    Returns:
        Filtered ``uint8`` image.
    """
    gray = _ensure_gray(img).astype(np.float64)
    rows, cols = gray.shape

    # Forward FFT
    dft = np.fft.fft2(gray)
    dft_shift = np.fft.fftshift(dft)

    # Build filter
    gauss = gen_gauss_filter((rows, cols), cutoff)
    if filter_type == "highpass":
        gauss = 1.0 - gauss

    # Apply filter
    filtered_dft = dft_shift * gauss

    # Inverse FFT
    dft_ishift = np.fft.ifftshift(filtered_dft)
    result = np.fft.ifft2(dft_ishift)
    result = np.abs(result)
    return _ensure_uint8(result)


# ====================================================================== #
#  5. Geometric Transforms                                                #
# ====================================================================== #


def rotate_image(
    img: np.ndarray, angle: float, mode: str = "constant"
) -> np.ndarray:
    """Rotate image around its centre.

    Args:
        angle: Rotation angle in degrees (counter-clockwise positive).
        mode:  Border mode -- ``"constant"`` (black) or ``"replicate"``.
    """
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    mat = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    border = cv2.BORDER_REPLICATE if mode == "replicate" else cv2.BORDER_CONSTANT
    return cv2.warpAffine(img, mat, (w, h), borderMode=border)


def mirror_image(img: np.ndarray, axis: str = "horizontal") -> np.ndarray:
    """Mirror (flip) an image.

    Args:
        axis: ``"horizontal"`` (left-right), ``"vertical"`` (top-bottom),
              or ``"both"``.
    """
    code_map = {"horizontal": 1, "vertical": 0, "both": -1}
    flip_code = code_map.get(axis, 1)
    return cv2.flip(img, flip_code)


def zoom_image(
    img: np.ndarray, factor_x: float = 1.0, factor_y: float = 1.0
) -> np.ndarray:
    """Resize (zoom) an image by independent x / y scale factors."""
    h, w = img.shape[:2]
    new_w = max(1, int(round(w * factor_x)))
    new_h = max(1, int(round(h * factor_y)))
    interp = cv2.INTER_AREA if (factor_x < 1.0 or factor_y < 1.0) else cv2.INTER_LINEAR
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


def affine_trans_image(img: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply an affine transformation using a 2x3 matrix."""
    h, w = img.shape[:2]
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.shape != (2, 3):
        raise ValueError(f"Affine matrix must be 2x3, got {mat.shape}")
    return cv2.warpAffine(img, mat, (w, h), flags=cv2.INTER_LINEAR)


def projective_trans_image(img: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a projective (perspective) transformation using a 3x3 matrix."""
    h, w = img.shape[:2]
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.shape != (3, 3):
        raise ValueError(f"Perspective matrix must be 3x3, got {mat.shape}")
    return cv2.warpPerspective(img, mat, (w, h), flags=cv2.INTER_LINEAR)


def polar_trans_image(
    img: np.ndarray,
    cx: float,
    cy: float,
    max_radius: Optional[float] = None,
) -> np.ndarray:
    """Log-polar / linear-polar transform around centre ``(cx, cy)``.

    Args:
        max_radius: Maximum radius in pixels.  Defaults to the distance
                    from the centre to the farthest image corner.
    """
    h, w = img.shape[:2]
    if max_radius is None:
        # Distance to the farthest corner
        corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float64)
        dists = np.sqrt((corners[:, 0] - cx) ** 2 + (corners[:, 1] - cy) ** 2)
        max_radius = float(np.max(dists))
    flags = cv2.WARP_FILL_OUTLIERS + cv2.WARP_POLAR_LINEAR
    return cv2.warpPolar(img, (w, h), (cx, cy), max_radius, flags)


# ====================================================================== #
#  6. Color Space                                                         #
# ====================================================================== #


def rgb_to_gray(img: np.ndarray) -> np.ndarray:
    """Convert a colour image to grayscale.  Handles both BGR and already-gray."""
    if img.ndim == 2:
        return img
    if img.shape[2] == 1:
        return img[:, :, 0]
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def rgb_to_hsv(img: np.ndarray) -> np.ndarray:
    """Convert BGR image to HSV colour space."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def rgb_to_hls(img: np.ndarray) -> np.ndarray:
    """Convert BGR image to HLS colour space."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2HLS)


def decompose3(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a 3-channel image into three single-channel images."""
    if img.ndim == 2:
        return img.copy(), img.copy(), img.copy()
    ch0, ch1, ch2 = cv2.split(img)
    return ch0, ch1, ch2


def compose3(
    ch1: np.ndarray, ch2: np.ndarray, ch3: np.ndarray
) -> np.ndarray:
    """Merge three single-channel images into one 3-channel image."""
    return cv2.merge([ch1, ch2, ch3])


def histogram_eq(img: np.ndarray) -> np.ndarray:
    """Histogram equalisation.

    For grayscale images ``cv2.equalizeHist`` is used directly.  For colour
    images the Y channel in YCrCb space is equalised.
    """
    if img.ndim == 2:
        return cv2.equalizeHist(img)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def illuminate(
    img: np.ndarray, ksize: int = 51, factor: float = 1.0
) -> np.ndarray:
    """Background illumination correction.

    Estimates the background via a large-kernel mean filter, then normalises:
    ``result = img / background * 128 * factor``.
    """
    k = _odd(ksize)
    gray = _ensure_gray(img).astype(np.float64)
    background = cv2.blur(gray, (k, k)).astype(np.float64)
    # Avoid division by zero
    background = np.where(background < 1.0, 1.0, background)
    corrected = gray / background * 128.0 * factor
    return _ensure_uint8(corrected)


# ====================================================================== #
#  7. Texture / Feature Images                                            #
# ====================================================================== #


def texture_laws(
    img: np.ndarray,
    filter_type: str = "L5E5",
    shift: int = 2,
    ksize: int = 5,
) -> np.ndarray:
    """Laws texture energy measure.

    One-dimensional vectors::

        L5 = [1, 4, 6, 4, 1]
        E5 = [-1, -2, 0, 2, 1]
        S5 = [-1, 0, 2, 0, -1]
        R5 = [1, -4, 6, -4, 1]
        W5 = [-1, 2, 0, -2, 1]

    The 2-D filter kernel is the outer product of two selected vectors
    (e.g. ``"L5E5"`` -> L5 outer E5).  The image is convolved with this
    kernel, the absolute response is taken, and a local mean with
    ``(2*shift+1)`` kernel smooths it into a texture energy image.
    """
    vectors: Dict[str, np.ndarray] = {
        "L5": np.array([1, 4, 6, 4, 1], dtype=np.float64),
        "E5": np.array([-1, -2, 0, 2, 1], dtype=np.float64),
        "S5": np.array([-1, 0, 2, 0, -1], dtype=np.float64),
        "R5": np.array([1, -4, 6, -4, 1], dtype=np.float64),
        "W5": np.array([-1, 2, 0, -2, 1], dtype=np.float64),
    }

    # Parse filter_type like "L5E5" into two vector names
    name = filter_type.upper()
    if len(name) != 4:
        raise ValueError(
            f"filter_type must be 4 characters (e.g. 'L5E5'), got '{filter_type}'"
        )
    row_name = name[:2]
    col_name = name[2:]
    if row_name not in vectors or col_name not in vectors:
        raise ValueError(
            f"Unknown Laws vectors: '{row_name}', '{col_name}'. "
            f"Choose from {list(vectors.keys())}."
        )

    row_vec = vectors[row_name]
    col_vec = vectors[col_name]
    kernel = np.outer(row_vec, col_vec)

    gray = _ensure_gray(img).astype(np.float64)
    filtered = cv2.filter2D(gray, cv2.CV_64F, kernel)
    energy = np.abs(filtered)

    # Smooth the energy map
    smooth_k = 2 * shift + 1
    energy = cv2.blur(energy, (smooth_k, smooth_k))
    return _ensure_uint8(energy)


def entropy_image(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Local entropy computed over a sliding window.

    Uses skimage.filters.rank.entropy for efficient C-level computation
    instead of pure-Python loops.  Falls back to a vectorised numpy
    implementation if skimage is unavailable.
    """
    gray = _ensure_gray(img).astype(np.uint8)
    k = _odd(ksize)

    try:
        from skimage.filters.rank import entropy as _sk_entropy
        from skimage.morphology import disk

        radius = k // 2
        selem = disk(radius)
        # skimage entropy returns values in bits, float64
        entropy_map = _sk_entropy(gray, selem).astype(np.float64)
    except ImportError:
        # Fallback: vectorised sliding-window via stride tricks
        half = k // 2
        h, w = gray.shape
        padded = cv2.copyMakeBorder(gray, half, half, half, half, cv2.BORDER_REFLECT)
        # Use as_strided to create (H, W, k, k) view
        shape = (h, w, k, k)
        strides = padded.strides + padded.strides
        patches = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
        patches_flat = patches.reshape(h, w, -1).astype(np.uint8)

        # Batch histogram via apply_along_axis replacement
        entropy_map = np.zeros((h, w), dtype=np.float64)
        for row in range(h):
            for col in range(w):
                hist = np.bincount(patches_flat[row, col], minlength=256).astype(np.float64)
                hist = hist / hist.sum()
                nonzero = hist[hist > 0]
                entropy_map[row, col] = -np.sum(nonzero * np.log2(nonzero))

    # Normalise to 0..255
    emin, emax = entropy_map.min(), entropy_map.max()
    if emax - emin > 0:
        entropy_map = (entropy_map - emin) / (emax - emin) * 255.0
    return _ensure_uint8(entropy_map)


def deviation_image(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Local standard deviation image.

    Uses the identity ``std = sqrt(E[X^2] - E[X]^2)`` computed via box
    filters for efficiency.
    """
    gray = _ensure_gray(img).astype(np.float64)
    k = _odd(ksize)
    mean_val = cv2.blur(gray, (k, k))
    mean_sq = cv2.blur(gray ** 2, (k, k))
    variance = np.maximum(mean_sq - mean_val ** 2, 0.0)
    std_map = np.sqrt(variance)
    return _ensure_uint8(std_map)


def local_min(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Local minimum filter (equivalent to grayscale erosion)."""
    return cv2.erode(img, _morph_kernel(ksize))


def local_max(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Local maximum filter (equivalent to grayscale dilation)."""
    return cv2.dilate(img, _morph_kernel(ksize))


def mean_curvature(img: np.ndarray) -> np.ndarray:
    """Approximate mean curvature flow using the Laplacian.

    The Laplacian of a smoothed image approximates the mean curvature of the
    intensity surface.  Result is absolute-valued and scaled to ``uint8``.
    """
    gray = _ensure_gray(img).astype(np.float64)
    smooth = cv2.GaussianBlur(gray, (5, 5), 1.0)
    lap = cv2.Laplacian(smooth, cv2.CV_64F)
    # Normalise to 0..255
    abs_lap = np.abs(lap)
    lmax = abs_lap.max()
    if lmax > 0:
        abs_lap = abs_lap / lmax * 255.0
    return _ensure_uint8(abs_lap)


# ====================================================================== #
#  8. Contours (XLD)                                                      #
# ====================================================================== #


def find_contours(
    region_or_mask: Union[np.ndarray, "Region"],
) -> List[np.ndarray]:
    """Find contours in a binary mask or :class:`Region`.

    Returns:
        List of contour arrays (each Nx1x2, int32).
    """
    if isinstance(region_or_mask, Region):
        mask = region_or_mask.mask
    else:
        mask = region_or_mask

    if mask.ndim == 3:
        mask = _ensure_gray(mask)
    # Ensure binary
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return list(contours)


def fit_line(contour: np.ndarray) -> Tuple[float, float, float, float]:
    """Fit a line to a contour.

    Returns:
        ``(vx, vy, x0, y0)`` -- direction vector and a point on the line.
    """
    line = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x0, y0 = line.flatten().tolist()
    return (vx, vy, x0, y0)


def fit_circle(contour: np.ndarray) -> Tuple[float, float, float]:
    """Fit a minimum enclosing circle to a contour.

    Returns:
        ``(cx, cy, radius)``.
    """
    (cx, cy), r = cv2.minEnclosingCircle(contour)
    return (float(cx), float(cy), float(r))


def fit_ellipse(
    contour: np.ndarray,
) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
    """Fit an ellipse to a contour (requires >= 5 points).

    Returns:
        ``((cx, cy), (major_axis, minor_axis), angle)``.
    """
    if len(contour) < 5:
        raise ValueError(
            f"fit_ellipse requires at least 5 points, got {len(contour)}"
        )
    ellipse = cv2.fitEllipse(contour)
    return ellipse  # type: ignore[return-value]


def contour_length(contour: np.ndarray) -> float:
    """Return the arc length (perimeter) of a contour."""
    return float(cv2.arcLength(contour, closed=False))


def convex_hull(contour: np.ndarray) -> np.ndarray:
    """Compute the convex hull of a contour."""
    return cv2.convexHull(contour)


def select_contours(
    contours: List[np.ndarray],
    feature: str,
    min_val: float,
    max_val: float,
) -> List[np.ndarray]:
    """Select contours whose *feature* value is within ``[min_val, max_val]``.

    Supported features:
        - ``"length"``: arc length.
        - ``"area"``: contour area.
        - ``"circularity"``: ``4 * pi * area / perimeter^2``.
    """
    selected: List[np.ndarray] = []
    for c in contours:
        if feature == "length":
            val = cv2.arcLength(c, closed=True)
        elif feature == "area":
            val = cv2.contourArea(c)
        elif feature == "circularity":
            area = cv2.contourArea(c)
            peri = cv2.arcLength(c, closed=True)
            val = (4.0 * math.pi * area / (peri * peri)) if peri > 0 else 0.0
        else:
            raise ValueError(f"Unknown contour feature: '{feature}'")
        if min_val <= val <= max_val:
            selected.append(c)
    return selected


# ====================================================================== #
#  9. Matching                                                            #
# ====================================================================== #


@log_operation(logger)
def template_match_ncc(
    img: np.ndarray,
    template: np.ndarray,
    threshold: float = DEFAULT_NCC_THRESHOLD,
) -> List[Tuple[int, int, float]]:
    """Normalised cross-correlation template matching.

    Args:
        threshold: Minimum correlation score to report a match.

    Returns:
        List of ``(x, y, score)`` for each match above *threshold*.
    """
    validate_image(img)
    validate_image(template, "template")
    gray_img = _ensure_gray(img).astype(np.float32)
    gray_tpl = _ensure_gray(template).astype(np.float32)
    try:
        result = cv2.matchTemplate(gray_img, gray_tpl, cv2.TM_CCORR_NORMED)
    except cv2.error as exc:
        _wrap_cv2_error("template_match_ncc", exc)

    matches: List[Tuple[int, int, float]] = []
    th, tw = gray_tpl.shape[:2]
    # Use a copy to allow suppression
    res = result.copy()
    while True:
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val < threshold:
            break
        x, y = max_loc
        matches.append((x, y, float(max_val)))
        # Suppress the detected region
        x1 = max(0, x - tw // 2)
        y1 = max(0, y - th // 2)
        x2 = min(res.shape[1], x + tw // 2 + 1)
        y2 = min(res.shape[0], y + th // 2 + 1)
        res[y1:y2, x1:x2] = 0.0
    return matches


@log_operation(logger)
def match_shape(
    img: np.ndarray,
    template: np.ndarray,
    levels: int = 3,
) -> List[Tuple[int, int, float, float]]:
    """Simplified multi-scale edge-based shape matching.

    Builds a Gaussian pyramid of *levels* for both the image and the
    template.  At each pyramid level, ``TM_CCOEFF_NORMED`` matching is
    performed.  The best match from the coarsest level seeds the search
    at finer levels.

    Returns:
        List of ``(x, y, score, angle)`` -- angle is always 0 in this
        simplified implementation (no rotation search).
    """
    gray_img = _ensure_gray(img).astype(np.uint8)
    gray_tpl = _ensure_gray(template).astype(np.uint8)

    # Build pyramids
    pyr_img = [gray_img]
    pyr_tpl = [gray_tpl]
    for _ in range(levels - 1):
        pyr_img.append(cv2.pyrDown(pyr_img[-1]))
        pyr_tpl.append(cv2.pyrDown(pyr_tpl[-1]))

    best_x, best_y, best_score = 0, 0, -1.0

    # Coarse to fine
    for lvl in range(levels - 1, -1, -1):
        cur_img = pyr_img[lvl]
        cur_tpl = pyr_tpl[lvl]
        th, tw = cur_tpl.shape[:2]
        ih, iw = cur_img.shape[:2]

        # Skip if template is larger than image at this level
        if th > ih or tw > iw:
            continue

        result = cv2.matchTemplate(
            cur_img.astype(np.float32),
            cur_tpl.astype(np.float32),
            cv2.TM_CCOEFF_NORMED,
        )
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val > best_score:
            scale = 2 ** lvl
            best_x = int(max_loc[0] * scale)
            best_y = int(max_loc[1] * scale)
            best_score = float(max_val)

    if best_score < 0:
        return []
    return [(best_x, best_y, best_score, 0.0)]


@log_operation(logger)
def match_template(
    img: np.ndarray,
    template: np.ndarray,
    method: str = "ccoeff_normed",
) -> Tuple[int, int, float]:
    """Single-best template match with user-selected method.

    Args:
        method: One of ``"sqdiff"``, ``"sqdiff_normed"``, ``"ccorr"``,
                ``"ccorr_normed"``, ``"ccoeff"``, ``"ccoeff_normed"``.

    Returns:
        ``(best_x, best_y, best_score)``.
    """
    method_map: Dict[str, int] = {
        "sqdiff": cv2.TM_SQDIFF,
        "sqdiff_normed": cv2.TM_SQDIFF_NORMED,
        "ccorr": cv2.TM_CCORR,
        "ccorr_normed": cv2.TM_CCORR_NORMED,
        "ccoeff": cv2.TM_CCOEFF,
        "ccoeff_normed": cv2.TM_CCOEFF_NORMED,
    }
    cv_method = method_map.get(method.lower())
    if cv_method is None:
        raise ValueError(
            f"Unknown template matching method: '{method}'. "
            f"Choose from {list(method_map.keys())}."
        )

    gray_img = _ensure_gray(img).astype(np.float32)
    gray_tpl = _ensure_gray(template).astype(np.float32)
    result = cv2.matchTemplate(gray_img, gray_tpl, cv_method)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # For SQDIFF methods, the best match is the minimum
    if cv_method in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED):
        best_x, best_y = min_loc
        best_score = float(min_val)
    else:
        best_x, best_y = max_loc
        best_score = float(max_val)

    return (best_x, best_y, best_score)


# ====================================================================== #
#  10. Measurement                                                        #
# ====================================================================== #


def measure_line_profile(
    img: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> Dict[str, Any]:
    """Sample gray values along a line from ``(x1,y1)`` to ``(x2,y2)``.

    Returns:
        Dictionary with keys:
        - ``"positions"``: 1-D array of cumulative arc-length positions.
        - ``"values"``: 1-D array of gray values along the line.
        - ``"length"``: total Euclidean length of the line.
    """
    gray = _ensure_gray(img).astype(np.float64)
    length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    num_pts = max(int(round(length)), 2)
    xs = np.linspace(x1, x2, num_pts)
    ys = np.linspace(y1, y2, num_pts)

    h, w = gray.shape
    # Clip to image bounds
    xs_clip = np.clip(xs, 0, w - 1).astype(np.int32)
    ys_clip = np.clip(ys, 0, h - 1).astype(np.int32)

    values = gray[ys_clip, xs_clip]
    positions = np.linspace(0.0, length, num_pts)

    return {
        "positions": positions,
        "values": values,
        "length": length,
    }


def distance_pp(
    x1: float, y1: float, x2: float, y2: float
) -> float:
    """Euclidean distance between two points."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def angle_ll(
    line1: Tuple[float, float, float, float],
    line2: Tuple[float, float, float, float],
) -> float:
    """Angle between two lines in degrees.

    Each line is given as ``(vx, vy, x0, y0)`` (direction vector + point).
    Returns the acute angle in the range ``[0, 180)``.
    """
    vx1, vy1 = line1[0], line1[1]
    vx2, vy2 = line2[0], line2[1]
    dot = vx1 * vx2 + vy1 * vy2
    mag1 = math.sqrt(vx1 ** 2 + vy1 ** 2)
    mag2 = math.sqrt(vx2 ** 2 + vy2 ** 2)
    if mag1 == 0 or mag2 == 0:
        return 0.0
    cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(abs(cos_angle)))


def area_center(
    region: Union[np.ndarray, "Region"],
) -> Tuple[int, float, float]:
    """Compute the area and centroid of a region or binary mask.

    Args:
        region: A :class:`Region` object or a ``uint8`` binary mask.

    Returns:
        ``(area, cx, cy)`` -- pixel count and centroid coordinates.
    """
    if isinstance(region, Region):
        return (region.area, region.cx, region.cy)

    mask = region
    if mask.ndim == 3:
        mask = _ensure_gray(mask)
    area = int(np.count_nonzero(mask))
    if area == 0:
        return (0, 0.0, 0.0)
    ys, xs = np.nonzero(mask)
    cx = float(np.mean(xs))
    cy = float(np.mean(ys))
    return (area, cx, cy)


# ====================================================================== #
#  11. Drawing                                                            #
# ====================================================================== #


def gen_rectangle(
    x: int, y: int, w: int, h: int, shape: Tuple[int, ...]
) -> Region:
    """Generate a :class:`Region` that is a filled rectangle.

    Args:
        x, y: Top-left corner.
        w, h: Width and height.
        shape: Canvas shape ``(rows, cols)`` or ``(rows, cols, channels)``.
    """
    canvas = np.zeros(shape[:2], dtype=np.uint8)
    cv2.rectangle(canvas, (x, y), (x + w - 1, y + h - 1), 255, thickness=-1)
    _, labels = cv2.connectedComponents(canvas)
    return Region(mask=canvas, labels=labels)


def gen_circle(
    cx: int, cy: int, r: int, shape: Tuple[int, ...]
) -> Region:
    """Generate a :class:`Region` that is a filled circle."""
    canvas = np.zeros(shape[:2], dtype=np.uint8)
    cv2.circle(canvas, (cx, cy), r, 255, thickness=-1)
    _, labels = cv2.connectedComponents(canvas)
    return Region(mask=canvas, labels=labels)


def gen_ellipse(
    cx: int,
    cy: int,
    a: int,
    b: int,
    angle: float,
    shape: Tuple[int, ...],
) -> Region:
    """Generate a :class:`Region` that is a filled, rotated ellipse.

    Args:
        a, b: Semi-major and semi-minor axis lengths.
        angle: Rotation angle in degrees.
    """
    canvas = np.zeros(shape[:2], dtype=np.uint8)
    cv2.ellipse(canvas, (cx, cy), (a, b), angle, 0, 360, 255, thickness=-1)
    _, labels = cv2.connectedComponents(canvas)
    return Region(mask=canvas, labels=labels)


def gen_region_polygon(
    points: Sequence[Tuple[int, int]], shape: Tuple[int, ...]
) -> Region:
    """Generate a :class:`Region` from polygon vertices.

    Args:
        points: Sequence of ``(x, y)`` vertices.
        shape:  Canvas shape ``(rows, cols, ...)``.
    """
    canvas = np.zeros(shape[:2], dtype=np.uint8)
    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(canvas, [pts], 255)
    _, labels = cv2.connectedComponents(canvas)
    return Region(mask=canvas, labels=labels)


def paint_region(
    region: Union[np.ndarray, "Region"],
    img: np.ndarray,
    color: Tuple[int, ...] = (255, 0, 0),
) -> np.ndarray:
    """Paint a region onto an image copy with the specified colour.

    If the input image is grayscale it is converted to BGR first.
    """
    out = img.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    mask = region.mask if isinstance(region, Region) else region
    if mask.ndim == 3:
        mask = _ensure_gray(mask)

    out[mask > 0] = color
    return out


def draw_text(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    size: float = 1.0,
    color: Tuple[int, ...] = (255, 255, 255),
    thickness: int = 1,
) -> np.ndarray:
    """Draw text onto a copy of the image."""
    out = img.copy()
    cv2.putText(
        out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness
    )
    return out


def draw_line(
    img: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: Tuple[int, ...] = (255, 0, 0),
    thickness: int = 1,
) -> np.ndarray:
    """Draw a line segment onto a copy of the image."""
    out = img.copy()
    cv2.line(out, (x1, y1), (x2, y2), color, thickness)
    return out


def draw_rectangle(
    img: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    color: Tuple[int, ...] = (0, 255, 0),
    thickness: int = 1,
) -> np.ndarray:
    """Draw a rectangle onto a copy of the image."""
    out = img.copy()
    cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
    return out


def draw_circle(
    img: np.ndarray,
    cx: int,
    cy: int,
    r: int,
    color: Tuple[int, ...] = (0, 0, 255),
    thickness: int = 1,
) -> np.ndarray:
    """Draw a circle onto a copy of the image."""
    out = img.copy()
    cv2.circle(out, (cx, cy), r, color, thickness)
    return out


def draw_arrow(
    img: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: Tuple[int, ...] = (255, 0, 0),
    thickness: int = 1,
    tip_length: float = 0.1,
) -> np.ndarray:
    """Draw an arrowed line onto a copy of the image."""
    out = img.copy()
    cv2.arrowedLine(out, (x1, y1), (x2, y2), color, thickness, tipLength=tip_length)
    return out


def draw_cross(
    img: np.ndarray,
    cx: int,
    cy: int,
    size: int = 10,
    color: Tuple[int, ...] = (255, 0, 0),
    thickness: int = 1,
) -> np.ndarray:
    """Draw a cross-hair (+ shape) at ``(cx, cy)`` onto a copy of the image."""
    out = img.copy()
    half = size // 2
    cv2.line(out, (cx - half, cy), (cx + half, cy), color, thickness)
    cv2.line(out, (cx, cy - half), (cx, cy + half), color, thickness)
    return out


# ====================================================================== #
#  12. Barcode / QR Code                                                  #
# ====================================================================== #


@log_operation(logger)
def find_barcode(img: np.ndarray) -> List[Dict[str, Any]]:
    """Detect and decode barcodes using ``cv2.barcode.BarcodeDetector``.

    Returns:
        List of dicts with keys ``"type"``, ``"data"``, ``"points"``.
        Returns an empty list if the detector is not available.
    """
    results: List[Dict[str, Any]] = []
    try:
        detector = cv2.barcode.BarcodeDetector()
    except AttributeError:
        logger.warning(
            "cv2.barcode.BarcodeDetector is not available in this OpenCV build."
        )
        return results

    gray = _ensure_gray(img)

    # Use detectAndDecodeWithType (returns 4 values) if available,
    # otherwise fall back to detectAndDecode (returns 3 values).
    try:
        ok, decoded_info, decoded_type, points = detector.detectAndDecodeWithType(gray)
    except AttributeError:
        try:
            retval, points, straight_code = detector.detectAndDecode(gray)
        except cv2.error as exc:
            logger.warning("find_barcode: OpenCV error - %s", exc)
            return results
        if retval and isinstance(retval, str) and retval:
            return [{"type": "unknown", "data": retval, "points": points if points is not None else np.array([])}]
        return results
    except cv2.error as exc:
        logger.warning("find_barcode: OpenCV error - %s", exc)
        return results

    if not ok or decoded_info is None:
        return results

    for i, data in enumerate(decoded_info):
        if data:  # non-empty string
            entry: Dict[str, Any] = {
                "type": decoded_type[i] if decoded_type is not None else "unknown",
                "data": data,
                "points": points[i] if points is not None else np.array([]),
            }
            results.append(entry)
    return results


@log_operation(logger)
def find_qrcode(img: np.ndarray) -> List[Dict[str, Any]]:
    """Detect and decode QR codes using ``cv2.QRCodeDetector``.

    Returns:
        List of dicts with keys ``"data"`` and ``"points"``.
    """
    results: List[Dict[str, Any]] = []
    detector = cv2.QRCodeDetector()
    gray = _ensure_gray(img)

    # Try multi-detector first (OpenCV >= 4.x)
    try:
        retvals = detector.detectAndDecodeMulti(gray)
        # OpenCV >= 4.8 returns (ok, decoded_info, points, straight_code)
        # Some versions return (ok, decoded_info, points)
        ok = retvals[0]
        decoded_info = retvals[1]
        points = retvals[2] if len(retvals) > 2 else None
        if ok and decoded_info is not None:
            for i, data in enumerate(decoded_info):
                if data:
                    entry: Dict[str, Any] = {
                        "data": data,
                        "points": points[i] if points is not None else np.array([]),
                    }
                    results.append(entry)
            return results
    except (cv2.error, AttributeError):
        pass

    # Fallback: single QR code detection
    data, pts, _ = detector.detectAndDecode(gray)
    if data:
        results.append({
            "data": data,
            "points": pts if pts is not None else np.array([]),
        })
    return results


@log_operation(logger)
def find_datamatrix(img: np.ndarray) -> List[Dict[str, Any]]:
    """Attempt to detect Data Matrix codes.

    Uses ``cv2.barcode.BarcodeDetector`` which can recognise some 2-D
    symbologies depending on the OpenCV build.  Falls back to an empty
    list if the detector is unavailable.

    Returns:
        List of dicts with keys ``"type"``, ``"data"``, ``"points"``.
    """
    results: List[Dict[str, Any]] = []
    try:
        detector = cv2.barcode.BarcodeDetector()
    except AttributeError:
        logger.warning(
            "cv2.barcode.BarcodeDetector is not available; "
            "cannot detect Data Matrix codes."
        )
        return results

    gray = _ensure_gray(img)

    # Use detectAndDecodeWithType (returns 4 values) if available,
    # otherwise fall back to detectAndDecode (returns 3 values).
    try:
        ok, decoded_info, decoded_type, points = detector.detectAndDecodeWithType(gray)
    except AttributeError:
        retval, points, straight_code = detector.detectAndDecode(gray)
        if retval and isinstance(retval, str) and retval:
            return [{"type": "datamatrix", "data": retval, "points": points if points is not None else np.array([])}]
        return results

    if not ok or decoded_info is None:
        return results

    for i, data in enumerate(decoded_info):
        if data:
            entry: Dict[str, Any] = {
                "type": decoded_type[i] if decoded_type is not None else "datamatrix",
                "data": data,
                "points": points[i] if points is not None else np.array([]),
            }
            results.append(entry)
    return results


# ====================================================================== #
#  13. Segmentation                                                       #
# ====================================================================== #


@log_operation(logger)
def watersheds(img: np.ndarray, marker_thresh: float = 0.5) -> np.ndarray:
    """Marker-based watershed segmentation.

    Computes a distance transform on a binarised version of the image,
    applies a threshold to create markers, then runs ``cv2.watershed``
    to produce segmented region boundaries.

    Args:
        marker_thresh: Fraction of the max distance value used to seed
                       markers (0..1).

    Returns:
        BGR visualisation with watershed boundaries drawn in red.
    """
    gray = _ensure_gray(img)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Morphological opening to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    # Sure background
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # Distance transform
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    # Sure foreground
    _, sure_fg = cv2.threshold(dist, marker_thresh * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    unknown = cv2.subtract(sure_bg, sure_fg)
    # Markers
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    # Watershed needs BGR input
    if img.ndim == 2:
        color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        color = img.copy()
    cv2.watershed(color, markers)
    color[markers == -1] = [0, 0, 255]
    return color


def distance_transform(img: np.ndarray, method: str = "L2") -> np.ndarray:
    """Compute the distance transform of a binary image.

    Args:
        method: ``"L1"``, ``"L2"``, or ``"C"`` (Chebyshev).

    Returns:
        Normalised ``uint8`` distance map.
    """
    dist_map = {"L1": cv2.DIST_L1, "L2": cv2.DIST_L2, "C": cv2.DIST_C}
    dt = dist_map.get(method.upper(), cv2.DIST_L2)
    gray = _ensure_gray(img)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    dist = cv2.distanceTransform(binary, dt, 5)
    # Normalise to 0..255
    d_max = dist.max()
    if d_max > 0:
        dist = dist / d_max * 255.0
    return _ensure_uint8(dist)


def skeleton(img: np.ndarray) -> np.ndarray:
    """Morphological skeletonisation (thinning).

    Iteratively erodes and subtracts until convergence, producing a
    one-pixel-wide skeleton of the foreground.

    Returns:
        Binary ``uint8`` skeleton image (0 / 255).
    """
    gray = _ensure_gray(img)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    skel = np.zeros_like(binary)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(binary, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(binary, temp)
        skel = cv2.bitwise_or(skel, temp)
        binary = eroded.copy()
        if cv2.countNonZero(binary) == 0:
            break
    return skel


# ====================================================================== #
#  14. Feature Points                                                     #
# ====================================================================== #


def points_harris(
    img: np.ndarray,
    block_size: int = 2,
    ksize: int = 3,
    k: float = 0.04,
    threshold: float = 0.01,
) -> np.ndarray:
    """Harris corner detection.

    Args:
        block_size: Neighbourhood size for corner detection.
        ksize: Aperture parameter for the Sobel operator.
        k: Harris detector free parameter.
        threshold: Fraction of the max response to consider a corner.

    Returns:
        BGR image with detected corners marked as red circles.
    """
    gray = _ensure_gray(img).astype(np.float32)
    harris = cv2.cornerHarris(gray, block_size, ksize, k)
    harris = cv2.dilate(harris, None)
    if img.ndim == 2:
        out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        out = img.copy()
    out[harris > threshold * harris.max()] = [0, 0, 255]
    return out


def points_shi_tomasi(
    img: np.ndarray,
    max_corners: int = 100,
    quality: float = 0.01,
    min_distance: float = 10.0,
) -> np.ndarray:
    """Shi-Tomasi good features to track.

    Args:
        max_corners: Maximum number of corners to detect.
        quality: Minimum quality level (0..1).
        min_distance: Minimum Euclidean distance between corners.

    Returns:
        BGR image with detected feature points marked as green circles.
    """
    gray = _ensure_gray(img)
    corners = cv2.goodFeaturesToTrack(gray, max_corners, quality, min_distance)
    if img.ndim == 2:
        out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        out = img.copy()
    if corners is not None:
        for c in corners:
            x, y = c.ravel()
            cv2.circle(out, (int(x), int(y)), 4, (0, 255, 0), -1)
    return out


# ====================================================================== #
#  15. Line / Circle Detection (Hough)                                    #
# ====================================================================== #


@log_operation(logger)
def hough_lines(
    img: np.ndarray,
    rho: float = 1.0,
    theta_deg: float = 1.0,
    threshold: int = DEFAULT_HOUGH_THRESHOLD,
) -> np.ndarray:
    """Detect straight lines via the probabilistic Hough transform.

    Args:
        rho: Distance resolution in pixels.
        theta_deg: Angle resolution in degrees.
        threshold: Accumulator threshold for ``HoughLinesP``.

    Returns:
        BGR image with detected lines drawn in green.
    """
    validate_image(img)
    gray = _ensure_gray(img)
    edges = cv2.Canny(gray, DEFAULT_CANNY_LOW, DEFAULT_CANNY_HIGH)
    theta = theta_deg * np.pi / 180.0
    lines = cv2.HoughLinesP(edges, rho, theta, threshold,
                            minLineLength=30, maxLineGap=10)
    if img.ndim == 2:
        out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        out = img.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return out


@log_operation(logger)
def hough_circles(
    img: np.ndarray,
    dp: float = 1.2,
    min_dist: float = 30.0,
    param1: float = 50.0,
    param2: float = 30.0,
    min_radius: int = 0,
    max_radius: int = 0,
) -> np.ndarray:
    """Detect circles via the Hough gradient method.

    Args:
        dp: Inverse ratio of accumulator resolution.
        min_dist: Minimum distance between circle centres.
        param1: Upper Canny threshold (lower is half).
        param2: Accumulator threshold for circle centres.
        min_radius: Minimum circle radius (0 = no limit).
        max_radius: Maximum circle radius (0 = no limit).

    Returns:
        BGR image with detected circles drawn in cyan.
    """
    validate_image(img)
    gray = _ensure_gray(img)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp, min_dist,
        param1=param1, param2=param2,
        minRadius=min_radius, maxRadius=max_radius,
    )
    if img.ndim == 2:
        out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        out = img.copy()
    if circles is not None:
        circles_rounded = np.uint16(np.around(circles))
        for cx, cy, r in circles_rounded[0]:
            cv2.circle(out, (cx, cy), r, (255, 255, 0), 2)
            cv2.circle(out, (cx, cy), 2, (0, 0, 255), 3)
    return out


# ====================================================================== #
#  16. Miscellaneous                                                      #
# ====================================================================== #


@log_operation(logger)
def optical_flow(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """Dense optical flow (Farneback) between two frames.

    Returns:
        BGR visualisation where hue encodes flow direction and value
        encodes flow magnitude.
    """
    gray1 = _ensure_gray(img1)
    gray2 = _ensure_gray(img2)
    try:
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0,
        )
    except cv2.error as exc:
        _wrap_cv2_error("optical_flow", exc)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((*gray1.shape, 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def gen_gauss_pyramid(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """Build a Gaussian image pyramid.

    Args:
        levels: Number of pyramid levels (including the original).

    Returns:
        List of images from finest (original) to coarsest.
    """
    pyramid: List[np.ndarray] = [img.copy()]
    current = img
    for _ in range(levels - 1):
        current = cv2.pyrDown(current)
        pyramid.append(current)
    return pyramid


def estimate_noise(img: np.ndarray) -> float:
    """Estimate the noise standard deviation of an image.

    Uses the robust median estimator on the Laplacian response
    (Immerkaer method).

    Returns:
        Estimated noise sigma (float).
    """
    gray = _ensure_gray(img).astype(np.float64)
    h, w = gray.shape
    # 3x3 Laplacian kernel
    kernel = np.array([[1, -2, 1],
                       [-2, 4, -2],
                       [1, -2, 1]], dtype=np.float64)
    lap = cv2.filter2D(gray, cv2.CV_64F, kernel)
    sigma = np.median(np.abs(lap)) * 1.4826 / math.sqrt(6)
    return float(sigma)


def abs_diff_image(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """Absolute pixel-wise difference between two images."""
    return cv2.absdiff(img1, img2)


def grab_image(device: int = 0) -> Optional[np.ndarray]:
    """Capture a single frame from a camera device.

    Args:
        device: Camera index (default 0 for the first webcam).

    Returns:
        Captured BGR image, or ``None`` if the camera could not be opened.
    """
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        logger.warning("Cannot open camera device %d", device)
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        logger.warning("Failed to grab frame from device %d", device)
        return None
    return frame


def clahe(
    img: np.ndarray,
    clip_limit: float = DEFAULT_CLAHE_CLIP,
    tile_size: int = DEFAULT_CLAHE_TILE,
) -> np.ndarray:
    """CLAHE (Contrast Limited Adaptive Histogram Equalisation).

    Args:
        clip_limit: Threshold for contrast limiting.
        tile_size: Size of the grid tiles (square).

    Returns:
        Enhanced ``uint8`` image.
    """
    validate_image(img)
    validate_positive(clip_limit, "clip_limit")
    validate_positive(tile_size, "tile_size")
    cl = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    if img.ndim == 2:
        return cl.apply(img)
    # For colour images, apply CLAHE to the L channel in LAB space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cl.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
