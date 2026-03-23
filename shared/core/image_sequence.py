"""Image sequence accumulation and temporal processing.

Provides operations for multi-frame averaging, noise reduction,
background modeling, and temporal analysis of image sequences.

Key components
--------------
- ``ImageAccumulator`` -- online mean/variance computation (Welford).
- ``BackgroundModel`` -- adaptive background subtraction.
- ``TemporalFilter`` -- frame-to-frame temporal smoothing.
- Utility functions for sequence statistics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AccumulatorStats:
    """Statistics from image accumulation."""
    count: int
    mean: np.ndarray         # (H, W) or (H, W, C) mean image
    variance: np.ndarray     # (H, W) or (H, W, C) per-pixel variance
    std: np.ndarray          # (H, W) or (H, W, C) standard deviation
    min_image: np.ndarray    # (H, W) or (H, W, C) per-pixel minimum
    max_image: np.ndarray    # (H, W) or (H, W, C) per-pixel maximum

    def snr_map(self) -> np.ndarray:
        """Compute signal-to-noise ratio map (mean / std)."""
        safe_std = np.where(self.std > 1e-10, self.std, 1e-10)
        return (self.mean / safe_std).astype(np.float32)

    def range_image(self) -> np.ndarray:
        """Pixel-wise (max - min) range."""
        return (self.max_image - self.min_image).astype(np.float32)


class ImageAccumulator:
    """Online image averaging with Welford's algorithm.

    Computes running mean and variance without storing all frames.
    Memory usage is O(H*W) regardless of sequence length.

    Usage::

        acc = ImageAccumulator()
        for frame in capture_loop():
            acc.add(frame)
        stats = acc.get_stats()
        denoised = stats.mean.astype(np.uint8)
    """

    def __init__(self) -> None:
        self._count: int = 0
        self._mean: Optional[np.ndarray] = None
        self._m2: Optional[np.ndarray] = None
        self._min: Optional[np.ndarray] = None
        self._max: Optional[np.ndarray] = None

    @property
    def count(self) -> int:
        return self._count

    def add(self, image: np.ndarray) -> None:
        """Add a frame to the accumulator."""
        img = image.astype(np.float64)

        if self._count == 0:
            self._mean = img.copy()
            self._m2 = np.zeros_like(img)
            self._min = img.copy()
            self._max = img.copy()
        else:
            delta = img - self._mean
            self._mean += delta / (self._count + 1)
            delta2 = img - self._mean
            self._m2 += delta * delta2
            np.minimum(self._min, img, out=self._min)
            np.maximum(self._max, img, out=self._max)

        self._count += 1

    def get_stats(self) -> AccumulatorStats:
        """Return current accumulation statistics."""
        if self._count == 0:
            raise ValueError("No images accumulated.")

        variance = self._m2 / max(self._count - 1, 1)
        return AccumulatorStats(
            count=self._count,
            mean=self._mean.astype(np.float32),
            variance=variance.astype(np.float32),
            std=np.sqrt(variance).astype(np.float32),
            min_image=self._min.astype(np.float32),
            max_image=self._max.astype(np.float32),
        )

    def get_mean_image(self) -> np.ndarray:
        """Return the current mean as uint8 (denoised image)."""
        if self._mean is None:
            raise ValueError("No images accumulated.")
        return np.clip(self._mean, 0, 255).astype(np.uint8)

    def reset(self) -> None:
        """Reset the accumulator."""
        self._count = 0
        self._mean = None
        self._m2 = None
        self._min = None
        self._max = None


class BackgroundModel:
    """Adaptive background modeling and subtraction.

    Supports:
    - Running average (exponential moving average)
    - Gaussian mixture model (MOG2 via OpenCV)
    - Median-based background

    Parameters
    ----------
    method : str
        "running_avg", "mog2", or "median".
    learning_rate : float
        Update rate for running average / MOG2 (0.0-1.0).
    history : int
        Number of frames for MOG2 history.
    """

    def __init__(
        self,
        method: str = "running_avg",
        learning_rate: float = 0.01,
        history: int = 500,
    ) -> None:
        self._method = method
        self._lr = learning_rate
        self._bg: Optional[np.ndarray] = None
        self._mog2 = None
        self._median_buffer: List[np.ndarray] = []
        self._median_size = min(history, 50)

        if method == "mog2":
            self._mog2 = cv2.createBackgroundSubtractorMOG2(
                history=history, varThreshold=16, detectShadows=False,
            )

    def update(self, frame: np.ndarray) -> np.ndarray:
        """Update background and return foreground mask.

        Returns
        -------
        mask : ndarray (H, W) uint8, 255 = foreground
        """
        if self._method == "running_avg":
            return self._update_running_avg(frame)
        elif self._method == "mog2":
            return self._mog2.apply(frame, learningRate=self._lr)
        elif self._method == "median":
            return self._update_median(frame)
        else:
            raise ValueError(f"Unknown method: {self._method}")

    def get_background(self) -> Optional[np.ndarray]:
        """Return the current background model image."""
        if self._method == "mog2":
            return self._mog2.getBackgroundImage() if self._mog2 else None
        return self._bg

    def _update_running_avg(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        fg = gray.astype(np.float32)

        if self._bg is None:
            self._bg = fg.copy()
            return np.zeros_like(gray)

        cv2.accumulateWeighted(fg, self._bg, self._lr)
        diff = cv2.absdiff(fg, self._bg)
        _, mask = cv2.threshold(diff.astype(np.uint8), 25, 255, cv2.THRESH_BINARY)
        return mask

    def _update_median(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        self._median_buffer.append(gray.astype(np.float32))
        if len(self._median_buffer) > self._median_size:
            self._median_buffer.pop(0)

        if len(self._median_buffer) < 3:
            self._bg = gray.astype(np.float32)
            return np.zeros_like(gray)

        self._bg = np.median(np.stack(self._median_buffer), axis=0).astype(np.float32)
        diff = cv2.absdiff(gray.astype(np.float32), self._bg)
        _, mask = cv2.threshold(diff.astype(np.uint8), 25, 255, cv2.THRESH_BINARY)
        return mask


class TemporalFilter:
    """Frame-to-frame temporal smoothing filter.

    Reduces temporal noise while preserving edges and fast changes.

    Parameters
    ----------
    method : str
        "ema" (exponential moving average) or "kalman".
    alpha : float
        Smoothing factor for EMA (0.0-1.0, higher = more responsive).
    """

    def __init__(self, method: str = "ema", alpha: float = 0.3) -> None:
        self._method = method
        self._alpha = alpha
        self._prev: Optional[np.ndarray] = None

    def filter(self, frame: np.ndarray) -> np.ndarray:
        """Apply temporal filtering to a frame."""
        f = frame.astype(np.float32)
        if self._prev is None:
            self._prev = f.copy()
            return frame

        if self._method == "ema":
            self._prev = self._alpha * f + (1.0 - self._alpha) * self._prev
        else:
            # Simple Kalman-like adaptive blending
            diff = np.abs(f - self._prev)
            adaptive_alpha = np.where(diff > 30.0, 0.8, self._alpha)
            self._prev = adaptive_alpha * f + (1.0 - adaptive_alpha) * self._prev

        return np.clip(self._prev, 0, 255).astype(np.uint8)

    def reset(self) -> None:
        self._prev = None


# ======================================================================
# Utility functions
# ======================================================================

def mean_image(images: List[np.ndarray]) -> np.ndarray:
    """Compute the pixel-wise mean of a list of images."""
    if not images:
        raise ValueError("Empty image list")
    acc = images[0].astype(np.float64)
    for img in images[1:]:
        acc += img.astype(np.float64)
    return (acc / len(images)).astype(np.uint8)


def median_image(images: List[np.ndarray]) -> np.ndarray:
    """Compute the pixel-wise median of a list of images."""
    if not images:
        raise ValueError("Empty image list")
    stack = np.stack([img.astype(np.float32) for img in images], axis=0)
    return np.median(stack, axis=0).astype(np.uint8)


def std_image(images: List[np.ndarray]) -> np.ndarray:
    """Compute the pixel-wise standard deviation."""
    if len(images) < 2:
        raise ValueError("Need at least 2 images for std")
    stack = np.stack([img.astype(np.float64) for img in images], axis=0)
    return np.std(stack, axis=0).astype(np.float32)


def max_image(images: List[np.ndarray]) -> np.ndarray:
    """Compute the pixel-wise maximum."""
    result = images[0].copy()
    for img in images[1:]:
        np.maximum(result, img, out=result)
    return result


def min_image(images: List[np.ndarray]) -> np.ndarray:
    """Compute the pixel-wise minimum."""
    result = images[0].copy()
    for img in images[1:]:
        np.minimum(result, img, out=result)
    return result


def temporal_denoise(images: List[np.ndarray], method: str = "mean") -> np.ndarray:
    """Denoise by combining multiple frames.

    Parameters
    ----------
    method : "mean", "median", or "weighted_mean" (center-weighted).
    """
    if method == "mean":
        return mean_image(images)
    elif method == "median":
        return median_image(images)
    elif method == "weighted_mean":
        n = len(images)
        weights = np.exp(-0.5 * ((np.arange(n) - n // 2) / (n / 4)) ** 2)
        weights /= weights.sum()
        acc = np.zeros_like(images[0], dtype=np.float64)
        for img, w in zip(images, weights):
            acc += img.astype(np.float64) * w
        return np.clip(acc, 0, 255).astype(np.uint8)
    else:
        raise ValueError(f"Unknown method: {method}")
