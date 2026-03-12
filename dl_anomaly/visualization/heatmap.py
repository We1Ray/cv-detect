"""Heatmap and overlay visualisation utilities.

All functions accept and return NumPy arrays (uint8 BGR or RGB depending on
context).  The convention throughout is **RGB** order, which matches both
matplotlib and Pillow; callers that display with OpenCV should convert with
``cv2.cvtColor``.
"""

from __future__ import annotations

import cv2
import numpy as np


def create_error_heatmap(
    error_map: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Convert a float ``(H, W)`` error map in [0, 1] to a coloured uint8 ``(H, W, 3)`` image (RGB)."""
    map_u8 = (np.clip(error_map, 0.0, 1.0) * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(map_u8, colormap)
    return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)


def create_defect_overlay(
    original: np.ndarray,
    error_map: np.ndarray,
    threshold: float = 0.5,
    alpha: float = 0.5,
) -> np.ndarray:
    """Overlay a semi-transparent red mask on *original* where ``error_map > threshold``.

    Parameters
    ----------
    original : (H, W, 3) uint8 RGB image.
    error_map : (H, W) float32 in [0, 1].
    threshold : Threshold above which the overlay is applied.
    alpha : Blending factor for the overlay (0 = fully transparent).

    Returns
    -------
    (H, W, 3) uint8 RGB image.
    """
    if original.ndim == 2:
        original = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)

    overlay = original.copy()
    mask = error_map > threshold

    # Red channel highlight
    red = np.zeros_like(original)
    red[:, :, 0] = 255

    overlay[mask] = cv2.addWeighted(original, 1.0 - alpha, red, alpha, 0)[mask]
    return overlay


def create_reconstruction_comparison(
    original: np.ndarray,
    reconstruction: np.ndarray,
    gap: int = 4,
) -> np.ndarray:
    """Return a side-by-side ``(H, W*2+gap, 3)`` image.

    Both inputs must have the same spatial size.  Grayscale images are
    converted to 3-channel for consistency.
    """
    def _ensure_rgb(img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img

    a = _ensure_rgb(original)
    b = _ensure_rgb(reconstruction)
    h, w, c = a.shape
    separator = np.full((h, gap, c), 200, dtype=np.uint8)
    return np.concatenate([a, separator, b], axis=1)


def create_composite_result(
    original: np.ndarray,
    reconstruction: np.ndarray,
    heatmap: np.ndarray,
    mask: np.ndarray,
    gap: int = 4,
) -> np.ndarray:
    """Create a 2x2 composite image.

    Layout::

        +-------------+--+-----------------+
        |  Original   |  | Reconstruction  |
        +-------------+--+-----------------+
        |  Heatmap    |  | Defect mask     |
        +-------------+--+-----------------+

    All four panels are resized to match ``original``'s spatial dimensions.
    """

    def _ensure_rgb(img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img

    h, w = original.shape[:2]

    panels = []
    for img in (original, reconstruction, heatmap, mask):
        img = _ensure_rgb(img)
        if img.shape[0] != h or img.shape[1] != w:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        panels.append(img)

    sep_h = np.full((h, gap, 3), 200, dtype=np.uint8)
    sep_v = np.full((gap, w * 2 + gap, 3), 200, dtype=np.uint8)

    top_row = np.concatenate([panels[0], sep_h, panels[1]], axis=1)
    bot_row = np.concatenate([panels[2], sep_h, panels[3]], axis=1)
    composite = np.concatenate([top_row, sep_v, bot_row], axis=0)

    return composite
