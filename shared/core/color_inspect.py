"""
core/color_inspect.py - Color inspection and measurement for industrial QC.

Provides CIE Lab color-difference calculations, color classification, and
color-consistency checking for defect-detection pipelines.

Categories:
    1. Data Classes (ColorSample, DeltaEResult, ColorClassResult)
    2. Color Space Conversion
    3. Color Sampling
    4. Color Difference (Delta-E: CIE76 / CIEDE2000)
    5. Color Classification
    6. Color Consistency / Uniformity
    7. Visualization Helpers
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from shared.validation import validate_image
from shared.op_logger import log_operation

logger = logging.getLogger(__name__)

# ====================================================================== #
#  Built-in reference colours (CIE L*a*b*)                                #
# ====================================================================== #

DEFAULT_COLORS: Dict[str, Tuple[float, float, float]] = {
    "\u7d05": (53.2, 80.1, 67.2),
    "\u6a59": (74.9, 23.9, 78.9),
    "\u9ec3": (97.1, -21.6, 94.5),
    "\u7da0": (46.2, -51.7, 49.9),
    "\u9752": (91.1, -48.1, -14.1),
    "\u85cd": (32.3, 79.2, -107.9),
    "\u7d2b": (29.8, 58.9, -36.5),
    "\u7c89": (76.0, 32.3, 3.8),
    "\u767d": (100.0, 0.0, 0.0),
    "\u7070": (53.6, 0.0, 0.0),
    "\u9ed1": (0.0, 0.0, 0.0),
    "\u68d5": (37.0, 15.0, 16.0),
}

# ====================================================================== #
#  Data classes                                                            #
# ====================================================================== #


@dataclass
class ColorSample:
    """A colour measurement from a region of an image."""

    lab: Tuple[float, float, float]
    rgb: Tuple[int, int, int]
    hsv: Tuple[int, int, int]
    std: Tuple[float, float, float]
    area: int


@dataclass
class DeltaEResult:
    """Result of a colour difference measurement."""

    delta_e: float
    delta_l: float
    delta_a: float
    delta_b: float
    pass_fail: bool
    tolerance: float
    method: str


@dataclass
class ColorClassResult:
    """Result of colour classification."""

    class_name: str
    confidence: float
    lab: Tuple[float, float, float]
    nearest_reference: str
    delta_e: float


# ====================================================================== #
#  Color space conversion                                                  #
# ====================================================================== #


@log_operation("rgb_to_lab")
def rgb_to_lab(image: np.ndarray) -> np.ndarray:
    """Convert an RGB (or BGR) image to CIE L*a*b* float32."""
    validate_image(image)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    return lab.astype(np.float32)


@log_operation("lab_to_rgb")
def lab_to_rgb(lab_image: np.ndarray) -> np.ndarray:
    """Convert a CIE L*a*b* image back to BGR uint8."""
    validate_image(lab_image)
    lab_u8 = np.clip(lab_image, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab_u8, cv2.COLOR_Lab2BGR)


@log_operation("rgb_to_hsv_float")
def rgb_to_hsv_float(image: np.ndarray) -> np.ndarray:
    """Convert BGR image to HSV with float32 (H: 0-360, S: 0-1, V: 0-1)."""
    validate_image(image)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    f32 = image.astype(np.float32) / 255.0
    hsv = cv2.cvtColor(f32, cv2.COLOR_BGR2HSV)
    return hsv


# ====================================================================== #
#  Color sampling                                                          #
# ====================================================================== #


def _crop_roi(image: np.ndarray, roi: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
    """Return the ROI sub-image, or the full image if *roi* is ``None``."""
    if roi is None:
        return image
    x, y, w, h = roi
    return image[y:y + h, x:x + w]


@log_operation("sample_color")
def sample_color(image: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None) -> ColorSample:
    """Sample the average colour from *image* (or an ROI within it).

    Parameters
    ----------
    image : np.ndarray
        BGR uint8 image.
    roi : tuple, optional
        ``(x, y, w, h)`` rectangle.  ``None`` uses the full image.

    Returns
    -------
    ColorSample
    """
    validate_image(image)
    patch = _crop_roi(image, roi)
    area = patch.shape[0] * patch.shape[1]

    # Lab
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2Lab).astype(np.float32)
    mean_lab = tuple(float(v) for v in cv2.mean(lab)[:3])
    std_lab = tuple(float(np.std(lab[:, :, c])) for c in range(3))

    # RGB (stored as BGR internally, report in RGB order)
    mean_bgr = cv2.mean(patch)[:3]
    mean_rgb = (int(round(mean_bgr[2])), int(round(mean_bgr[1])), int(round(mean_bgr[0])))

    # HSV
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    mean_hsv = tuple(int(round(v)) for v in cv2.mean(hsv)[:3])

    return ColorSample(
        lab=mean_lab,  # type: ignore[arg-type]
        rgb=mean_rgb,  # type: ignore[arg-type]
        hsv=mean_hsv,  # type: ignore[arg-type]
        std=std_lab,  # type: ignore[arg-type]
        area=area,
    )


@log_operation("sample_colors_grid")
def sample_colors_grid(
    image: np.ndarray,
    grid_rows: int = 3,
    grid_cols: int = 3,
) -> List[ColorSample]:
    """Divide *image* into a grid and sample each cell.

    Useful for uniformity analysis across regions.
    """
    validate_image(image)
    h, w = image.shape[:2]
    cell_h = h // grid_rows
    cell_w = w // grid_cols

    samples: List[ColorSample] = []
    for r in range(grid_rows):
        for c in range(grid_cols):
            roi = (c * cell_w, r * cell_h, cell_w, cell_h)
            samples.append(sample_color(image, roi=roi))
    return samples


# ====================================================================== #
#  Delta-E colour difference                                               #
# ====================================================================== #


def delta_e_cie76(
    lab1: Tuple[float, float, float],
    lab2: Tuple[float, float, float],
) -> float:
    """CIE76 Delta-E: Euclidean distance in L*a*b* space."""
    return float(math.sqrt(
        (lab1[0] - lab2[0]) ** 2
        + (lab1[1] - lab2[1]) ** 2
        + (lab1[2] - lab2[2]) ** 2
    ))


def delta_e_ciede2000(
    lab1: Tuple[float, float, float],
    lab2: Tuple[float, float, float],
    kL: float = 1.0,
    kC: float = 1.0,
    kH: float = 1.0,
) -> float:
    """CIEDE2000 Delta-E (the industry-standard perceptual colour metric).

    Implements the full algorithm with SL, SC, SH, and RT corrections as
    described in *Sharma, Wu, Dalal (2005)*.
    """
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2

    # Step 1 -- calculate C'ab, h'ab
    C1 = math.sqrt(a1 ** 2 + b1 ** 2)
    C2 = math.sqrt(a2 ** 2 + b2 ** 2)
    C_avg = (C1 + C2) / 2.0
    C_avg7 = C_avg ** 7
    G = 0.5 * (1.0 - math.sqrt(C_avg7 / (C_avg7 + 25.0 ** 7)))

    a1p = a1 * (1.0 + G)
    a2p = a2 * (1.0 + G)

    C1p = math.sqrt(a1p ** 2 + b1 ** 2)
    C2p = math.sqrt(a2p ** 2 + b2 ** 2)

    h1p = math.degrees(math.atan2(b1, a1p)) % 360.0
    h2p = math.degrees(math.atan2(b2, a2p)) % 360.0

    # Step 2 -- calculate delta L', delta C', delta H'
    dLp = L2 - L1
    dCp = C2p - C1p

    if C1p * C2p == 0.0:
        dhp = 0.0
    elif abs(h2p - h1p) <= 180.0:
        dhp = h2p - h1p
    elif h2p - h1p > 180.0:
        dhp = h2p - h1p - 360.0
    else:
        dhp = h2p - h1p + 360.0

    dHp = 2.0 * math.sqrt(C1p * C2p) * math.sin(math.radians(dhp / 2.0))

    # Step 3 -- calculate CIEDE2000
    Lp_avg = (L1 + L2) / 2.0
    Cp_avg = (C1p + C2p) / 2.0

    if C1p * C2p == 0.0:
        hp_avg = h1p + h2p
    elif abs(h1p - h2p) <= 180.0:
        hp_avg = (h1p + h2p) / 2.0
    elif h1p + h2p < 360.0:
        hp_avg = (h1p + h2p + 360.0) / 2.0
    else:
        hp_avg = (h1p + h2p - 360.0) / 2.0

    T = (
        1.0
        - 0.17 * math.cos(math.radians(hp_avg - 30.0))
        + 0.24 * math.cos(math.radians(2.0 * hp_avg))
        + 0.32 * math.cos(math.radians(3.0 * hp_avg + 6.0))
        - 0.20 * math.cos(math.radians(4.0 * hp_avg - 63.0))
    )

    SL = 1.0 + 0.015 * (Lp_avg - 50.0) ** 2 / math.sqrt(20.0 + (Lp_avg - 50.0) ** 2)
    SC = 1.0 + 0.045 * Cp_avg
    SH = 1.0 + 0.015 * Cp_avg * T

    Cp_avg7 = Cp_avg ** 7
    RT = (
        -2.0
        * math.sqrt(Cp_avg7 / (Cp_avg7 + 25.0 ** 7))
        * math.sin(math.radians(60.0 * math.exp(-((hp_avg - 275.0) / 25.0) ** 2)))
    )

    dE = math.sqrt(
        (dLp / (kL * SL)) ** 2
        + (dCp / (kC * SC)) ** 2
        + (dHp / (kH * SH)) ** 2
        + RT * (dCp / (kC * SC)) * (dHp / (kH * SH))
    )
    return float(dE)


@log_operation("compute_delta_e")
def compute_delta_e(
    sample1: ColorSample,
    sample2: ColorSample,
    method: str = "CIEDE2000",
    tolerance: float = 3.0,
) -> DeltaEResult:
    """Compare two :class:`ColorSample` objects and return a :class:`DeltaEResult`.

    Parameters
    ----------
    method : str
        ``"CIE76"`` or ``"CIEDE2000"``.
    tolerance : float
        Maximum acceptable Delta-E.  Typical values:
        1.0 = imperceptible, 3.0 = noticeable, 6.0 = large.
    """
    lab1, lab2 = sample1.lab, sample2.lab
    dl = lab2[0] - lab1[0]
    da = lab2[1] - lab1[1]
    db = lab2[2] - lab1[2]

    if method.upper() == "CIE76":
        de = delta_e_cie76(lab1, lab2)
    elif method.upper() == "CIEDE2000":
        de = delta_e_ciede2000(lab1, lab2)
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'CIE76' or 'CIEDE2000'.")

    return DeltaEResult(
        delta_e=de,
        delta_l=dl,
        delta_a=da,
        delta_b=db,
        pass_fail=de <= tolerance,
        tolerance=tolerance,
        method=method.upper(),
    )


@log_operation("compute_delta_e_map")
def compute_delta_e_map(
    image: np.ndarray,
    reference_color: Tuple[float, float, float],
    method: str = "CIE76",
) -> np.ndarray:
    """Per-pixel Delta-E map against a reference L*a*b* colour.

    Returns a float32 array with shape ``(H, W)`` where each value is the
    Delta-E of that pixel relative to *reference_color*.

    Parameters
    ----------
    image : np.ndarray
        BGR uint8 image.
    reference_color : tuple
        ``(L, a, b)`` reference colour.
    method : str
        ``"CIE76"`` (fast, vectorised) or ``"CIEDE2000"`` (more perceptually
        accurate but significantly slower for large images).
    """
    validate_image(image)
    lab = rgb_to_lab(image)
    ref = np.array(reference_color, dtype=np.float32).reshape(1, 1, 3)

    if method.upper() == "CIE76":
        diff = lab - ref
        de_map = np.sqrt(np.sum(diff ** 2, axis=2))
    elif method.upper() == "CIEDE2000":
        h, w = lab.shape[:2]
        de_map = np.empty((h, w), dtype=np.float32)
        ref_t = (float(reference_color[0]), float(reference_color[1]), float(reference_color[2]))
        for r in range(h):
            for c in range(w):
                pixel_lab = (float(lab[r, c, 0]), float(lab[r, c, 1]), float(lab[r, c, 2]))
                de_map[r, c] = delta_e_ciede2000(pixel_lab, ref_t)
    else:
        raise ValueError(
            f"Unknown method: {method!r}. Use 'CIE76' or 'CIEDE2000'."
        )

    return de_map.astype(np.float32)


# ====================================================================== #
#  Color classification                                                    #
# ====================================================================== #


@log_operation("classify_color")
def classify_color(
    sample: ColorSample,
    reference_colors: Optional[Dict[str, Tuple[float, float, float]]] = None,
) -> ColorClassResult:
    """Classify *sample* against a set of named reference colours.

    If *reference_colors* is ``None`` the built-in :data:`DEFAULT_COLORS`
    table is used.
    """
    if reference_colors is None:
        reference_colors = DEFAULT_COLORS

    best_name = ""
    best_de = float("inf")
    for name, ref_lab in reference_colors.items():
        de = delta_e_cie76(sample.lab, ref_lab)
        if de < best_de:
            best_de = de
            best_name = name

    # Confidence: map Delta-E to a 0-1 score.  DE=0 -> 1.0, DE>=100 -> 0.0
    confidence = max(0.0, 1.0 - best_de / 100.0)

    return ColorClassResult(
        class_name=best_name,
        confidence=confidence,
        lab=sample.lab,
        nearest_reference=best_name,
        delta_e=best_de,
    )


@log_operation("build_color_palette")
def build_color_palette(image: np.ndarray, n_colors: int = 8) -> List[ColorSample]:
    """Extract *n_colors* dominant colours via K-means clustering in Lab space.

    Returns a list of :class:`ColorSample` sorted by area (most frequent
    first).
    """
    validate_image(image)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab).astype(np.float32)
    pixels = lab.reshape(-1, 3)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centres = cv2.kmeans(
        pixels, n_colors, None, criteria, 3, cv2.KMEANS_PP_CENTERS,
    )

    labels_flat = labels.flatten()
    total_pixels = len(labels_flat)

    samples: List[ColorSample] = []
    for i in range(n_colors):
        mask = labels_flat == i
        count = int(np.sum(mask))
        if count == 0:
            continue

        cluster_lab = pixels[mask]
        mean_l = float(np.mean(cluster_lab[:, 0]))
        mean_a = float(np.mean(cluster_lab[:, 1]))
        mean_b = float(np.mean(cluster_lab[:, 2]))
        std_l = float(np.std(cluster_lab[:, 0]))
        std_a = float(np.std(cluster_lab[:, 1]))
        std_b = float(np.std(cluster_lab[:, 2]))

        # Convert centre back to BGR then RGB
        centre_lab = np.array([[[mean_l, mean_a, mean_b]]], dtype=np.float32)
        centre_u8 = np.clip(centre_lab, 0, 255).astype(np.uint8)
        centre_bgr = cv2.cvtColor(centre_u8, cv2.COLOR_Lab2BGR)[0, 0]
        rgb = (int(centre_bgr[2]), int(centre_bgr[1]), int(centre_bgr[0]))

        centre_hsv = cv2.cvtColor(
            np.array([[centre_bgr]], dtype=np.uint8), cv2.COLOR_BGR2HSV,
        )[0, 0]
        hsv = (int(centre_hsv[0]), int(centre_hsv[1]), int(centre_hsv[2]))

        samples.append(ColorSample(
            lab=(mean_l, mean_a, mean_b),
            rgb=rgb,
            hsv=hsv,
            std=(std_l, std_a, std_b),
            area=count,
        ))

    samples.sort(key=lambda s: s.area, reverse=True)
    return samples


# ====================================================================== #
#  Color consistency / uniformity                                          #
# ====================================================================== #


@log_operation("check_color_uniformity")
def check_color_uniformity(
    image: np.ndarray,
    roi: Optional[Tuple[int, int, int, int]] = None,
    max_std: float = 5.0,
    tolerance_threshold: Optional[float] = None,
) -> Dict:
    """Check whether colour is uniform within an ROI.

    Parameters
    ----------
    max_std : float
        Maximum acceptable standard deviation per L/a/b channel.
    tolerance_threshold : float, optional
        If provided, overrides *max_std* as the uniformity threshold.
        This is a convenience alias so callers can use a more descriptive
        parameter name.

    Returns
    -------
    dict
        ``{"uniform": bool, "std_l": float, "std_a": float, "std_b": float,
        "mean_lab": tuple}``
    """
    validate_image(image)
    threshold = tolerance_threshold if tolerance_threshold is not None else max_std
    patch = _crop_roi(image, roi)
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2Lab).astype(np.float32)

    std_l = float(np.std(lab[:, :, 0]))
    std_a = float(np.std(lab[:, :, 1]))
    std_b = float(np.std(lab[:, :, 2]))
    mean_lab = tuple(float(v) for v in cv2.mean(lab)[:3])

    uniform = std_l <= threshold and std_a <= threshold and std_b <= threshold

    return {
        "uniform": uniform,
        "std_l": std_l,
        "std_a": std_a,
        "std_b": std_b,
        "mean_lab": mean_lab,
    }


@log_operation("check_color_tolerance")
def check_color_tolerance(
    image: np.ndarray,
    reference_lab: Tuple[float, float, float],
    tolerance: float = 3.0,
    method: str = "CIEDE2000",
) -> np.ndarray:
    """Return a binary mask of pixels within *tolerance* of *reference_lab*.

    Pixels within tolerance are set to 255; those outside are 0.

    Parameters
    ----------
    method : str
        ``"CIE76"`` or ``"CIEDE2000"``.  The chosen method is forwarded to
        :func:`compute_delta_e_map` for the per-pixel calculation.  Note
        that ``"CIEDE2000"`` is significantly slower for large images.
    """
    validate_image(image)
    de_map = compute_delta_e_map(image, reference_lab, method=method)
    mask = np.zeros_like(de_map, dtype=np.uint8)
    mask[de_map <= tolerance] = 255
    return mask


# ====================================================================== #
#  Visualization helpers                                                   #
# ====================================================================== #


@log_operation("draw_delta_e_map")
def draw_delta_e_map(delta_e_map: np.ndarray, max_de: float = 10.0) -> np.ndarray:
    """Colour-map a Delta-E map: green=0, yellow=mid, red=max_de.

    Returns a BGR uint8 image.
    """
    normalised = np.clip(delta_e_map / max_de, 0.0, 1.0)
    u8 = (normalised * 255).astype(np.uint8)
    # COLORMAP_JET: blue(low) -> green -> yellow -> red(high)
    # We want green(low) -> red(high), so use a custom LUT or flip.
    coloured = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
    return coloured


@log_operation("draw_color_palette")
def draw_color_palette(
    palette: List[ColorSample],
    width: int = 400,
    height: int = 60,
) -> np.ndarray:
    """Draw a horizontal bar showing palette colours proportional to area."""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    total_area = sum(s.area for s in palette)
    if total_area == 0:
        return canvas

    x = 0
    for sample in palette:
        bar_w = max(1, int(round(sample.area / total_area * width)))
        if x + bar_w > width:
            bar_w = width - x
        bgr = (sample.rgb[2], sample.rgb[1], sample.rgb[0])
        cv2.rectangle(canvas, (x, 0), (x + bar_w, height), bgr, cv2.FILLED)
        x += bar_w
        if x >= width:
            break
    return canvas


@log_operation("draw_color_info")
def draw_color_info(
    image: np.ndarray,
    sample: ColorSample,
    position: Tuple[int, int] = (10, 30),
) -> np.ndarray:
    """Overlay colour info text (Lab, RGB, HSV) on *image*.

    Returns a copy with the text drawn.
    """
    out = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    colour = (255, 255, 255)
    thickness = 1
    line_gap = 20
    x, y = position

    lines = [
        f"Lab: ({sample.lab[0]:.1f}, {sample.lab[1]:.1f}, {sample.lab[2]:.1f})",
        f"RGB: {sample.rgb}",
        f"HSV: {sample.hsv}",
        f"Std: ({sample.std[0]:.1f}, {sample.std[1]:.1f}, {sample.std[2]:.1f})",
    ]
    for i, line in enumerate(lines):
        cv2.putText(out, line, (x, y + i * line_gap), font, scale, colour, thickness, cv2.LINE_AA)
    return out


@log_operation("annotate_color_regions")
def annotate_color_regions(
    image: np.ndarray,
    samples: List[ColorSample],
    rois: List[Tuple[int, int, int, int]],
) -> np.ndarray:
    """Draw ROI rectangles with colour info labels on each region.

    Parameters
    ----------
    samples : list[ColorSample]
        One sample per ROI.
    rois : list[tuple]
        Each ROI as ``(x, y, w, h)``.  Must match length of *samples*.
    """
    out = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.4
    thickness = 1

    for sample, roi in zip(samples, rois):
        x, y, w, h = roi
        # Draw rectangle
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Label
        label = (
            f"L={sample.lab[0]:.1f} a={sample.lab[1]:.1f} b={sample.lab[2]:.1f}"
        )
        cv2.putText(out, label, (x, y - 5), font, scale, (0, 255, 0), thickness, cv2.LINE_AA)
    return out
