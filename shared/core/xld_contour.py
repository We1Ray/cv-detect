"""
core/xld_contour.py - XLD (eXtended Line Description) sub-pixel contour operations.

Provides Halcon-style XLD contour extraction, processing, fitting, and feature
computation for sub-pixel-accurate contour analysis in industrial vision
pipelines.  All geometric computations use float64 arrays for maximum precision.

Categories:
    1. Data Classes (XLDContour, XLDContourSet)
    2. Contour Extraction
    3. Contour Processing
    4. Contour Fitting
    5. Contour Features
    6. Contour Geometry
    7. Visualization
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

logger = logging.getLogger(__name__)

# ====================================================================== #
#  Named constants                                                        #
# ====================================================================== #

DEFAULT_SIGMA = 1.0
DEFAULT_LOW = 20.0
DEFAULT_HIGH = 40.0
MIN_CONTOUR_LEN = 3
IRLS_MAX_ITER = 50
IRLS_TOL = 1e-6


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


def _ensure_bgr(img: np.ndarray) -> np.ndarray:
    """Return a 3-channel BGR copy suitable for colour drawing."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img.copy()


def _ksize_from_sigma(sigma: float) -> int:
    """Compute a suitable odd kernel size from a Gaussian sigma."""
    k = int(math.ceil(sigma * 6)) | 1
    return max(k, 3)


# ====================================================================== #
#  Data Classes                                                           #
# ====================================================================== #


@dataclass
class XLDContour:
    """A single sub-pixel contour (eXtended Line Description).

    Attributes:
        points:     Nx2 float64 array of (x, y) sub-pixel coordinates.
        attributes: Arbitrary per-contour metadata (e.g. edge contrast).
        is_closed:  Whether the contour forms a closed loop.
    """

    points: np.ndarray
    attributes: Dict[str, float] = field(default_factory=dict)
    is_closed: bool = False

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=np.float64)
        if self.points.ndim != 2 or self.points.shape[1] != 2:
            raise ValueError(
                f"points must be Nx2, got shape {self.points.shape}"
            )

    def __len__(self) -> int:
        return len(self.points)

    def copy(self) -> XLDContour:
        return XLDContour(
            points=self.points.copy(),
            attributes=dict(self.attributes),
            is_closed=self.is_closed,
        )


@dataclass
class XLDContourSet:
    """An ordered collection of XLD contours with set operations.

    Attributes:
        contours: List of XLDContour instances.
    """

    contours: List[XLDContour] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.contours)

    def __getitem__(self, idx: int) -> XLDContour:
        return self.contours[idx]

    def __iter__(self):
        return iter(self.contours)

    def append(self, contour: XLDContour) -> None:
        self.contours.append(contour)

    def extend(self, other: XLDContourSet) -> None:
        self.contours.extend(other.contours)

    def copy(self) -> XLDContourSet:
        return XLDContourSet(contours=[c.copy() for c in self.contours])

    @property
    def total_points(self) -> int:
        """Total number of sub-pixel points across all contours."""
        return sum(len(c) for c in self.contours)


# ====================================================================== #
#  Contour Extraction                                                     #
# ====================================================================== #


def edges_sub_pix(
    image: np.ndarray,
    sigma: float = DEFAULT_SIGMA,
    low: float = DEFAULT_LOW,
    high: float = DEFAULT_HIGH,
) -> XLDContourSet:
    """Sub-pixel edge detection producing XLD contours.

    Applies Gaussian smoothing followed by Canny edge detection, then
    extracts sub-pixel contour chains from the binary edge map via
    ``cv2.findContours`` and refines to sub-pixel with ``cv2.cornerSubPix``
    on each contour point.

    Args:
        image: Input image (grayscale or colour).
        sigma: Gaussian smoothing sigma.
        low:   Canny low threshold.
        high:  Canny high threshold.

    Returns:
        XLDContourSet containing the detected sub-pixel edge contours.
    """
    gray = _ensure_gray(image)
    gray_f = gray.astype(np.float64)

    # Gaussian smoothing
    ksize = _ksize_from_sigma(sigma)
    smoothed = cv2.GaussianBlur(gray, (ksize, ksize), sigma)

    # Canny edge detection
    edges = cv2.Canny(smoothed, low, high, L2gradient=True)

    # Extract contours from binary edge map
    contours_cv, _ = cv2.findContours(
        edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
    )

    result = XLDContourSet()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

    for cnt in contours_cv:
        if len(cnt) < MIN_CONTOUR_LEN:
            continue
        # Squeeze to Nx2 float
        pts = cnt.squeeze().astype(np.float32)
        if pts.ndim != 2 or pts.shape[1] != 2:
            continue

        # Sub-pixel refinement
        try:
            refined = cv2.cornerSubPix(
                gray, pts, winSize=(3, 3), zeroZone=(-1, -1), criteria=criteria
            )
        except cv2.error:
            refined = pts

        sub_pts = refined.astype(np.float64)
        is_closed = np.linalg.norm(sub_pts[0] - sub_pts[-1]) < 2.0
        result.append(XLDContour(points=sub_pts, is_closed=is_closed))

    logger.info("edges_sub_pix: extracted %d contours", len(result))
    return result


def threshold_sub_pix(
    image: np.ndarray,
    threshold: float,
) -> XLDContourSet:
    """Extract sub-pixel iso-value contours at a given threshold.

    Computes a binary mask at the given threshold, finds contours, then
    performs linear interpolation along each edge to place the contour
    points at the exact sub-pixel iso-level.

    Args:
        image:     Input grayscale image.
        threshold: Iso-value threshold for contour extraction.

    Returns:
        XLDContourSet of sub-pixel iso-value contours.
    """
    gray = _ensure_gray(image).astype(np.float64)
    h, w = gray.shape

    # Binary threshold
    mask = (gray >= threshold).astype(np.uint8) * 255
    contours_cv, _ = cv2.findContours(
        mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
    )

    result = XLDContourSet()
    for cnt in contours_cv:
        if len(cnt) < MIN_CONTOUR_LEN:
            continue
        pts_int = cnt.squeeze()
        if pts_int.ndim != 2 or pts_int.shape[1] != 2:
            continue

        # Sub-pixel interpolation along gradient direction
        sub_pts = np.empty_like(pts_int, dtype=np.float64)
        for k, (px, py) in enumerate(pts_int):
            fx, fy = float(px), float(py)
            # Gradient-based sub-pixel shift
            ix, iy = int(round(fx)), int(round(fy))
            if 1 <= ix < w - 1 and 1 <= iy < h - 1:
                gx = (gray[iy, ix + 1] - gray[iy, ix - 1]) / 2.0
                gy = (gray[iy + 1, ix] - gray[iy - 1, ix]) / 2.0
                g_mag = math.sqrt(gx * gx + gy * gy)
                if g_mag > 1e-6:
                    val = gray[iy, ix]
                    offset = (threshold - val) / g_mag
                    offset = max(-0.5, min(0.5, offset))
                    fx += offset * (gx / g_mag)
                    fy += offset * (gy / g_mag)
            sub_pts[k, 0] = fx
            sub_pts[k, 1] = fy

        is_closed = np.linalg.norm(sub_pts[0] - sub_pts[-1]) < 2.0
        result.append(XLDContour(points=sub_pts, is_closed=is_closed))

    logger.info("threshold_sub_pix: extracted %d contours at level %.1f", len(result), threshold)
    return result


# ====================================================================== #
#  Contour Processing                                                     #
# ====================================================================== #


def smooth_contours_xld(
    contour_set: XLDContourSet,
    sigma: float = DEFAULT_SIGMA,
) -> XLDContourSet:
    """Gaussian smoothing of XLD contour point coordinates.

    Each coordinate dimension (x, y) is independently smoothed with a 1-D
    Gaussian filter.  Closed contours use ``wrap`` boundary mode.

    Args:
        contour_set: Input contours.
        sigma:       Gaussian smoothing sigma (in contour-point index space).

    Returns:
        New XLDContourSet with smoothed coordinates.
    """
    result = XLDContourSet()
    for c in contour_set:
        mode = "wrap" if c.is_closed else "reflect"
        sx = gaussian_filter1d(c.points[:, 0], sigma=sigma, mode=mode)
        sy = gaussian_filter1d(c.points[:, 1], sigma=sigma, mode=mode)
        pts = np.column_stack([sx, sy])
        result.append(XLDContour(points=pts, attributes=dict(c.attributes), is_closed=c.is_closed))
    return result


def segment_contours_xld(
    contour_set: XLDContourSet,
    max_curvature: float = 0.5,
) -> XLDContourSet:
    """Split contours at high-curvature points (corners).

    Points where the local curvature exceeds ``max_curvature`` are treated
    as split points.  Each segment between split points becomes a new
    contour in the result set.

    Args:
        contour_set:   Input contours.
        max_curvature: Curvature threshold for splitting (1/pixels).

    Returns:
        New XLDContourSet with segmented contours.
    """
    result = XLDContourSet()
    for c in contour_set:
        if len(c) < MIN_CONTOUR_LEN:
            result.append(c.copy())
            continue

        kappa = _curvature_array(c.points)
        split_indices = list(np.where(np.abs(kappa) > max_curvature)[0])

        if not split_indices:
            result.append(c.copy())
            continue

        # Ensure 0 and end are in split points
        if split_indices[0] != 0:
            split_indices.insert(0, 0)
        if split_indices[-1] != len(c) - 1:
            split_indices.append(len(c) - 1)

        for i in range(len(split_indices) - 1):
            s, e = split_indices[i], split_indices[i + 1] + 1
            if e - s >= MIN_CONTOUR_LEN:
                result.append(
                    XLDContour(
                        points=c.points[s:e].copy(),
                        attributes=dict(c.attributes),
                        is_closed=False,
                    )
                )
    return result


def select_contours_xld(
    contour_set: XLDContourSet,
    feature: Literal["length", "circularity", "num_points"] = "length",
    min_value: float = 0.0,
    max_value: float = float("inf"),
) -> XLDContourSet:
    """Filter contours by a scalar feature value.

    Args:
        contour_set: Input contours.
        feature:     Feature to filter on.
        min_value:   Minimum accepted value (inclusive).
        max_value:   Maximum accepted value (inclusive).

    Returns:
        New XLDContourSet containing only the contours that pass the filter.
    """
    result = XLDContourSet()
    for c in contour_set:
        if feature == "length":
            val = length_xld(c)
        elif feature == "circularity":
            val = circularity_xld(c)
        elif feature == "num_points":
            val = float(len(c))
        else:
            raise ValueError(f"Unknown feature '{feature}'")

        if min_value <= val <= max_value:
            result.append(c.copy())
    return result


def union_contours_xld(
    contour_set: XLDContourSet,
    max_gap: float = 5.0,
) -> XLDContourSet:
    """Merge contour segments whose endpoints are within ``max_gap`` pixels.

    Performs a greedy nearest-neighbour merge: repeatedly joins the pair of
    open contours whose endpoint distance is smallest, until no pair is
    closer than ``max_gap``.

    Args:
        contour_set: Input contours.
        max_gap:     Maximum allowed gap in pixels for merging endpoints.

    Returns:
        New XLDContourSet with merged contours.
    """
    segments = [c.copy() for c in contour_set if not c.is_closed]
    closed = [c.copy() for c in contour_set if c.is_closed]

    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(segments):
            best_j = -1
            best_dist = max_gap
            best_flip_i = False
            best_flip_j = False

            end_i = segments[i].points[-1]
            start_i = segments[i].points[0]

            for j in range(i + 1, len(segments)):
                start_j = segments[j].points[0]
                end_j = segments[j].points[-1]

                # end_i -> start_j
                d = np.linalg.norm(end_i - start_j)
                if d < best_dist:
                    best_dist, best_j = d, j
                    best_flip_i, best_flip_j = False, False

                # end_i -> end_j (flip j)
                d = np.linalg.norm(end_i - end_j)
                if d < best_dist:
                    best_dist, best_j = d, j
                    best_flip_i, best_flip_j = False, True

                # start_i -> start_j (flip i)
                d = np.linalg.norm(start_i - start_j)
                if d < best_dist:
                    best_dist, best_j = d, j
                    best_flip_i, best_flip_j = True, False

                # start_i -> end_j
                d = np.linalg.norm(start_i - end_j)
                if d < best_dist:
                    best_dist, best_j = d, j
                    best_flip_i, best_flip_j = True, True

            if best_j >= 0:
                pts_i = segments[i].points
                pts_j = segments[best_j].points
                if best_flip_i:
                    pts_i = pts_i[::-1]
                if best_flip_j:
                    pts_j = pts_j[::-1]

                merged = np.vstack([pts_i, pts_j])
                segments[i] = XLDContour(points=merged, is_closed=False)
                segments.pop(best_j)
                changed = True
            else:
                i += 1

    result = XLDContourSet(contours=closed)
    result.contours.extend(segments)
    return result


def close_contours_xld(
    contour_set: XLDContourSet,
    max_gap: float = 10.0,
) -> XLDContourSet:
    """Close open contours whose start and end points are within ``max_gap``.

    A closing segment is added between the last and first point.

    Args:
        contour_set: Input contours.
        max_gap:     Maximum gap to close (pixels).

    Returns:
        New XLDContourSet with eligible contours closed.
    """
    result = XLDContourSet()
    for c in contour_set:
        nc = c.copy()
        if not nc.is_closed and len(nc) >= MIN_CONTOUR_LEN:
            gap = np.linalg.norm(nc.points[0] - nc.points[-1])
            if gap <= max_gap:
                nc.points = np.vstack([nc.points, nc.points[0:1]])
                nc.is_closed = True
        result.append(nc)
    return result


# ====================================================================== #
#  Contour Fitting                                                        #
# ====================================================================== #


def fit_line_contour_xld(
    contour: XLDContour,
    algorithm: Literal["least_squares", "huber"] = "least_squares",
) -> Tuple[float, float, float, float, float]:
    """Fit a line to an XLD contour.

    Args:
        contour:   Input contour.
        algorithm: ``"least_squares"`` or ``"huber"`` (robust).

    Returns:
        Tuple of ``(vx, vy, x0, y0, residual)`` where ``(vx, vy)`` is the
        unit direction vector, ``(x0, y0)`` is a point on the line, and
        ``residual`` is the RMS fitting error in pixels.
    """
    pts = contour.points
    if len(pts) < 2:
        raise ValueError("At least 2 points required for line fitting")

    if algorithm == "huber":
        dist_type = cv2.DIST_HUBER
    else:
        dist_type = cv2.DIST_L2

    pts_f32 = pts.astype(np.float32).reshape(-1, 1, 2)
    line = cv2.fitLine(pts_f32, dist_type, 0, 0.01, 0.01)
    vx, vy, x0, y0 = line.ravel()

    # Compute RMS residual
    dx = pts[:, 0] - x0
    dy = pts[:, 1] - y0
    dist = np.abs(dx * vy - dy * vx)
    residual = float(np.sqrt(np.mean(dist ** 2)))

    return (float(vx), float(vy), float(x0), float(y0), residual)


def fit_circle_contour_xld(
    contour: XLDContour,
) -> Tuple[float, float, float, float]:
    """Fit a circle to an XLD contour using algebraic least squares.

    Uses the Kasa method: minimises the algebraic distance
    ``x^2 + y^2 + Dx + Ey + F = 0``.

    Args:
        contour: Input contour (at least 3 points).

    Returns:
        Tuple of ``(cx, cy, radius, residual)`` where ``residual`` is the
        RMS radial fitting error in pixels.
    """
    pts = contour.points
    if len(pts) < 3:
        raise ValueError("At least 3 points required for circle fitting")

    x = pts[:, 0]
    y = pts[:, 1]
    A = np.column_stack([x, y, np.ones(len(pts))])
    b = -(x ** 2 + y ** 2)
    result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    cx = -result[0] / 2.0
    cy = -result[1] / 2.0
    r = math.sqrt(max(cx ** 2 + cy ** 2 - result[2], 0.0))

    # RMS radial residual
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - r
    residual = float(np.sqrt(np.mean(dist ** 2)))

    return (cx, cy, r, residual)


def fit_ellipse_contour_xld(
    contour: XLDContour,
) -> Tuple[float, float, float, float, float, float]:
    """Fit an ellipse to an XLD contour via OpenCV.

    Args:
        contour: Input contour (at least 5 points).

    Returns:
        Tuple of ``(cx, cy, semi_a, semi_b, angle_deg, residual)`` where
        ``semi_a >= semi_b`` are the semi-axes, ``angle_deg`` is the
        orientation in degrees, and ``residual`` is the RMS fitting error.
    """
    pts = contour.points
    if len(pts) < 5:
        raise ValueError("At least 5 points required for ellipse fitting")

    pts_f32 = pts.astype(np.float32)
    ((cx, cy), (d1, d2), angle) = cv2.fitEllipse(pts_f32)
    semi_a = max(d1, d2) / 2.0
    semi_b = min(d1, d2) / 2.0
    if d2 > d1:
        angle = (angle + 90.0) % 180.0

    # Compute RMS residual to fitted ellipse
    theta = np.deg2rad(angle)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    u = cos_t * dx + sin_t * dy
    v = -sin_t * dx + cos_t * dy
    # Algebraic distance to ellipse boundary
    if semi_a > 0 and semi_b > 0:
        ellipse_val = (u / semi_a) ** 2 + (v / semi_b) ** 2
        residual = float(np.sqrt(np.mean((ellipse_val - 1.0) ** 2)))
    else:
        residual = float("inf")

    return (float(cx), float(cy), semi_a, semi_b, float(angle), residual)


# ====================================================================== #
#  Contour Features                                                       #
# ====================================================================== #


def length_xld(contour: XLDContour) -> float:
    """Compute the arc length of an XLD contour.

    Sums the Euclidean distances between consecutive sub-pixel points.
    """
    if len(contour) < 2:
        return 0.0
    diffs = np.diff(contour.points, axis=0)
    return float(np.sum(np.sqrt(np.sum(diffs ** 2, axis=1))))


def area_center_xld(contour: XLDContour) -> Tuple[float, float, float]:
    """Compute the signed area and centroid of an XLD contour.

    Uses the shoelace formula.  The contour need not be explicitly closed;
    the last-to-first edge is included automatically.

    Returns:
        Tuple of ``(area, cx, cy)``.  Area is positive for counter-clockwise
        contours and negative for clockwise ones.
    """
    pts = contour.points
    n = len(pts)
    if n < 3:
        cx = float(np.mean(pts[:, 0])) if n > 0 else 0.0
        cy = float(np.mean(pts[:, 1])) if n > 0 else 0.0
        return (0.0, cx, cy)

    x = pts[:, 0]
    y = pts[:, 1]
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)

    cross = x * y_next - x_next * y
    area = float(np.sum(cross)) / 2.0

    if abs(area) < 1e-12:
        return (0.0, float(np.mean(x)), float(np.mean(y)))

    cx = float(np.sum((x + x_next) * cross)) / (6.0 * area)
    cy = float(np.sum((y + y_next) * cross)) / (6.0 * area)
    return (area, cx, cy)


def moments_xld(contour: XLDContour) -> Dict[str, float]:
    """Compute central moments Mu11, Mu20, Mu02 of an XLD contour.

    These are computed relative to the centroid of the contour points
    (not the enclosed-area centroid).

    Returns:
        Dict with keys ``"Mu20"``, ``"Mu02"``, ``"Mu11"``.
    """
    pts = contour.points
    n = len(pts)
    if n < 2:
        return {"Mu20": 0.0, "Mu02": 0.0, "Mu11": 0.0}

    cx = np.mean(pts[:, 0])
    cy = np.mean(pts[:, 1])
    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy

    mu20 = float(np.sum(dx ** 2)) / n
    mu02 = float(np.sum(dy ** 2)) / n
    mu11 = float(np.sum(dx * dy)) / n

    return {"Mu20": mu20, "Mu02": mu02, "Mu11": mu11}


def circularity_xld(contour: XLDContour) -> float:
    """Compute the isoperimetric ratio (circularity) of an XLD contour.

    Defined as ``4 * pi * |area| / perimeter^2``.  Returns 1.0 for a
    perfect circle, 0.0 for degenerate contours.
    """
    area, _, _ = area_center_xld(contour)
    perimeter = length_xld(contour)
    if perimeter < 1e-12:
        return 0.0
    return float(4.0 * math.pi * abs(area) / (perimeter ** 2))


def curvature_xld(contour: XLDContour) -> np.ndarray:
    """Compute per-point curvature of an XLD contour.

    Uses the discrete curvature formula based on the circumscribed circle
    through three consecutive points.  Boundary points receive zero curvature.

    Returns:
        1-D float64 array of curvature values (1/pixels), same length as
        the contour.
    """
    return _curvature_array(contour.points)


def _curvature_array(pts: np.ndarray) -> np.ndarray:
    """Compute per-point curvature for an Nx2 array of points (vectorized)."""
    n = len(pts)
    if n < 3:
        return np.zeros(n, dtype=np.float64)

    # Vectorized computation for interior points
    p_prev = pts[:-2]  # (n-2, 2)
    p_curr = pts[1:-1]  # (n-2, 2)
    p_next = pts[2:]    # (n-2, 2)

    v1 = p_prev - p_curr  # d10
    v2 = p_next - p_curr  # d12

    # 2D cross product
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]

    # Distances
    d1 = np.linalg.norm(v1, axis=1)
    d2 = np.linalg.norm(v2, axis=1)
    d3 = np.linalg.norm(p_next - p_prev, axis=1)

    denom = d1 * d2 * d3
    interior = np.where(denom > 1e-12, 2.0 * cross / denom, 0.0)

    kappa = np.zeros(n, dtype=np.float64)
    kappa[1:-1] = interior
    return kappa


# ====================================================================== #
#  Contour Geometry                                                       #
# ====================================================================== #


def dist_contours_xld(
    contour_a: XLDContour,
    contour_b: XLDContour,
) -> float:
    """Compute the Hausdorff distance between two XLD contours.

    The Hausdorff distance is the maximum over all points in one contour
    of the minimum distance to the other contour, taken symmetrically.

    Args:
        contour_a: First contour.
        contour_b: Second contour.

    Returns:
        Hausdorff distance in pixels (float).
    """
    pts_a = contour_a.points
    pts_b = contour_b.points

    if len(pts_a) == 0 or len(pts_b) == 0:
        return float("inf")

    # Chunked Hausdorff to avoid O(Na*Nb) memory allocation
    CHUNK = 10000

    def _directed_hausdorff(a: np.ndarray, b: np.ndarray) -> float:
        max_min_dist = 0.0
        for i in range(0, len(a), CHUNK):
            chunk_a = a[i:i + CHUNK]
            min_dists = np.full(len(chunk_a), np.inf)
            for j in range(0, len(b), CHUNK):
                chunk_b = b[j:j + CHUNK]
                diff = chunk_a[:, np.newaxis, :] - chunk_b[np.newaxis, :, :]
                dists = np.sqrt(np.sum(diff ** 2, axis=2))
                min_dists = np.minimum(min_dists, dists.min(axis=1))
            max_min_dist = max(max_min_dist, float(min_dists.max()))
        return max_min_dist

    d_a2b = _directed_hausdorff(pts_a, pts_b)
    d_b2a = _directed_hausdorff(pts_b, pts_a)

    return max(d_a2b, d_b2a)


def affine_trans_contour_xld(
    contour_set: XLDContourSet,
    matrix: np.ndarray,
) -> XLDContourSet:
    """Apply a 2x3 affine transformation to all contours in a set.

    Args:
        contour_set: Input contours.
        matrix:      2x3 affine transformation matrix.

    Returns:
        New XLDContourSet with transformed coordinates.
    """
    M = np.asarray(matrix, dtype=np.float64)
    if M.shape != (2, 3):
        raise ValueError(f"Expected 2x3 affine matrix, got shape {M.shape}")

    result = XLDContourSet()
    for c in contour_set:
        pts = c.points
        # Homogeneous: (N, 3) = [x, y, 1]
        ones = np.ones((len(pts), 1), dtype=np.float64)
        pts_h = np.hstack([pts, ones])
        # Transform: (N, 2) = (N, 3) @ (3, 2)
        transformed = pts_h @ M.T
        result.append(
            XLDContour(
                points=transformed,
                attributes=dict(c.attributes),
                is_closed=c.is_closed,
            )
        )
    return result


# ====================================================================== #
#  Visualization                                                          #
# ====================================================================== #


def draw_xld(
    image: np.ndarray,
    contour_set: XLDContourSet,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 1,
    draw_points: bool = False,
    point_radius: int = 2,
) -> np.ndarray:
    """Draw XLD contours with sub-pixel accuracy on an image.

    Uses ``cv2.polylines`` for the contour path and optionally draws
    individual sub-pixel points as small circles.

    Args:
        image:        Input image (will not be modified).
        contour_set:  Contours to draw.
        color:        BGR colour tuple.
        thickness:    Line thickness in pixels.
        draw_points:  If True, draw individual contour points.
        point_radius: Radius for point circles.

    Returns:
        BGR image with contours drawn on it.
    """
    canvas = _ensure_bgr(image)

    for c in contour_set:
        if len(c) < 2:
            continue

        # Draw polyline with sub-pixel precision via LINE_AA
        pts_int = np.round(c.points).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(
            canvas,
            [pts_int],
            isClosed=c.is_closed,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

        if draw_points:
            for px, py in c.points:
                cv2.circle(
                    canvas,
                    (int(round(px)), int(round(py))),
                    point_radius,
                    color,
                    -1,
                    lineType=cv2.LINE_AA,
                )

    return canvas
