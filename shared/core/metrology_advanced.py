"""Advanced metrology extensions for geometric dimensioning & tolerancing.

Extends :mod:`shared.core.metrology` with additional measurement operators
commonly required in industrial vision applications -- arc measurement,
thickness gauging, GD&T form tolerances (roundness, straightness,
parallelism, perpendicularity, concentricity, symmetry), and B-spline
contour fitting.

All angle results are in **radians** unless stated otherwise.  Distance
results are in **pixel units**; apply a calibration factor for physical
units.

Dependencies:
    - numpy (required)
    - scipy.interpolate (for spline fitting; optional with graceful fallback)
    - cv2 (for contour utilities; optional)

Usage::

    from shared.core.metrology_advanced import (
        measure_roundness,
        measure_parallelism,
        measure_thickness,
    )

    roundness = measure_roundness(contour_points)
    parallelism = measure_parallelism(line_a, line_b)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ====================================================================== #
#  Constants                                                               #
# ====================================================================== #
_EPS = 1e-12
_SPLINE_DEFAULT_SMOOTHING = 0.0
_SPLINE_DEFAULT_DEGREE = 3


# ====================================================================== #
#  Data classes                                                            #
# ====================================================================== #

@dataclass
class MeasureArc:
    """Arc-shaped measurement region.

    Defines an annular sector on the image for sub-pixel edge detection
    along arcs (similar to Halcon's ``gen_measure_arc``).

    Attributes
    ----------
    center_row : float
        Centre Y coordinate.
    center_col : float
        Centre X coordinate.
    radius : float
        Radius of the arc's centre line.
    angle_start : float
        Start angle in radians (measured counter-clockwise from +X axis).
    angle_extent : float
        Angular extent in radians (positive = counter-clockwise).
    annulus_radius : float
        Half-width of the measurement band perpendicular to the arc.
    """

    center_row: float
    center_col: float
    radius: float
    angle_start: float
    angle_extent: float
    annulus_radius: float


@dataclass
class GDT_Result:
    """Geometric Dimensioning & Tolerancing result.

    Attributes
    ----------
    tolerance_type : str
        GD&T symbol name (e.g. ``"roundness"``, ``"parallelism"``).
    value : float
        Measured deviation value (pixels or radians, depending on type).
    tolerance : float
        Specified tolerance limit (0 if not provided).
    is_pass : bool
        Whether *value* <= *tolerance* (always True when tolerance is 0).
    unit : str
        Unit label (``"px"``, ``"rad"``, ``"ratio"``).
    details : Dict[str, Any]
        Additional measurement details specific to the tolerance type.
    """

    tolerance_type: str
    value: float
    tolerance: float = 0.0
    is_pass: bool = True
    unit: str = "px"
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.tolerance > 0:
            self.is_pass = self.value <= self.tolerance


@dataclass
class SplineFitResult:
    """Result of B-spline fitting to contour points.

    Attributes
    ----------
    fitted_points : np.ndarray
        Array of shape ``(N, 2)`` with fitted ``(x, y)`` coordinates.
    residuals : np.ndarray
        Per-point distances from original to fitted spline.
    rms_error : float
        Root-mean-square fitting error.
    max_error : float
        Maximum fitting error.
    tck : Any
        Scipy ``tck`` tuple for further evaluation (None if scipy unavailable).
    """

    fitted_points: np.ndarray
    residuals: np.ndarray
    rms_error: float
    max_error: float
    tck: Any = None


@dataclass
class ThicknessResult:
    """Wall / material thickness measurement result.

    Attributes
    ----------
    thickness_values : np.ndarray
        Array of measured thicknesses at each measurement position.
    mean_thickness : float
        Mean thickness value.
    min_thickness : float
        Minimum thickness.
    max_thickness : float
        Maximum thickness.
    std_thickness : float
        Standard deviation of thickness values.
    edge_pairs : List[Tuple[Tuple[float, float], Tuple[float, float]]]
        List of opposing edge-point pairs used for measurement.
    """

    thickness_values: np.ndarray
    mean_thickness: float
    min_thickness: float
    max_thickness: float
    std_thickness: float
    edge_pairs: List[Tuple[Tuple[float, float], Tuple[float, float]]] = field(
        default_factory=list
    )


# ====================================================================== #
#  Arc measurement region                                                  #
# ====================================================================== #

def gen_measure_arc(
    center_row: float,
    center_col: float,
    radius: float,
    angle_start: float,
    angle_extent: float,
    annulus_radius: float,
    num_samples: int = 100,
) -> Tuple[MeasureArc, np.ndarray]:
    """Generate an arc-shaped measurement region.

    Returns the arc descriptor and an array of sample points along the
    arc centre-line suitable for profile extraction.

    Parameters
    ----------
    center_row, center_col : float
        Arc centre coordinates.
    radius : float
        Arc radius.
    angle_start : float
        Start angle in radians.
    angle_extent : float
        Angular extent in radians.
    annulus_radius : float
        Half-width of the measurement band.
    num_samples : int
        Number of sample points along the arc.

    Returns
    -------
    Tuple[MeasureArc, np.ndarray]
        ``(arc_descriptor, sample_points)`` where *sample_points* has
        shape ``(num_samples, 2)`` with columns ``(row, col)``.
    """
    if radius <= 0:
        raise ValueError("radius must be positive.")
    if annulus_radius <= 0:
        raise ValueError("annulus_radius must be positive.")

    arc = MeasureArc(
        center_row=center_row,
        center_col=center_col,
        radius=radius,
        angle_start=angle_start,
        angle_extent=angle_extent,
        annulus_radius=annulus_radius,
    )

    angles = np.linspace(angle_start, angle_start + angle_extent, num_samples)
    rows = center_row + radius * np.sin(angles)
    cols = center_col + radius * np.cos(angles)
    sample_points = np.column_stack([rows, cols])

    return arc, sample_points


# ====================================================================== #
#  Thickness measurement                                                   #
# ====================================================================== #

def measure_thickness(
    contour_outer: np.ndarray,
    contour_inner: np.ndarray,
    num_rays: int = 36,
    center: Optional[Tuple[float, float]] = None,
) -> ThicknessResult:
    """Measure wall/material thickness between two opposing contours.

    Projects rays from a centre point outward and finds intersections
    with both contours to compute thickness at each ray angle.

    Parameters
    ----------
    contour_outer : np.ndarray
        Outer contour points, shape ``(N, 2)`` as ``(x, y)``.
    contour_inner : np.ndarray
        Inner contour points, shape ``(M, 2)`` as ``(x, y)``.
    num_rays : int
        Number of angular measurement rays (evenly spaced).
    center : Tuple[float, float], optional
        Centre point ``(x, y)`` for rays.  If None, computed as the
        centroid of the inner contour.

    Returns
    -------
    ThicknessResult
        Thickness statistics and per-ray measurements.
    """
    if contour_outer.ndim != 2 or contour_outer.shape[1] != 2:
        raise ValueError("contour_outer must have shape (N, 2).")
    if contour_inner.ndim != 2 or contour_inner.shape[1] != 2:
        raise ValueError("contour_inner must have shape (M, 2).")

    if center is None:
        cx = float(np.mean(contour_inner[:, 0]))
        cy = float(np.mean(contour_inner[:, 1]))
    else:
        cx, cy = center

    angles = np.linspace(0, 2 * math.pi, num_rays, endpoint=False)
    thicknesses: List[float] = []
    edge_pairs: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []

    for angle in angles:
        dx = math.cos(angle)
        dy = math.sin(angle)

        inner_pt = _find_contour_intersection(contour_inner, cx, cy, dx, dy)
        outer_pt = _find_contour_intersection(contour_outer, cx, cy, dx, dy)

        if inner_pt is not None and outer_pt is not None:
            dist = math.hypot(outer_pt[0] - inner_pt[0], outer_pt[1] - inner_pt[1])
            thicknesses.append(dist)
            edge_pairs.append((inner_pt, outer_pt))

    if not thicknesses:
        logger.warning("measure_thickness: no valid ray intersections found.")
        empty = np.array([])
        return ThicknessResult(
            thickness_values=empty,
            mean_thickness=0.0,
            min_thickness=0.0,
            max_thickness=0.0,
            std_thickness=0.0,
            edge_pairs=[],
        )

    arr = np.array(thicknesses)
    return ThicknessResult(
        thickness_values=arr,
        mean_thickness=float(np.mean(arr)),
        min_thickness=float(np.min(arr)),
        max_thickness=float(np.max(arr)),
        std_thickness=float(np.std(arr)),
        edge_pairs=edge_pairs,
    )


# ====================================================================== #
#  Spline fitting                                                          #
# ====================================================================== #

def fit_spline_contour_xld(
    points: np.ndarray,
    smoothing: float = _SPLINE_DEFAULT_SMOOTHING,
    degree: int = _SPLINE_DEFAULT_DEGREE,
    num_output_points: int = 0,
    periodic: bool = False,
) -> SplineFitResult:
    """Fit a B-spline curve to contour points.

    Parameters
    ----------
    points : np.ndarray
        Contour points of shape ``(N, 2)`` as ``(x, y)``.
    smoothing : float
        Smoothing factor for ``scipy.interpolate.splprep``.  Use 0 for
        interpolating spline.
    degree : int
        Spline degree (1-5).
    num_output_points : int
        Number of points to sample on the fitted spline.  If 0,
        defaults to ``len(points)``.
    periodic : bool
        If True, fit a closed (periodic) spline.

    Returns
    -------
    SplineFitResult
        Fitted spline points, residuals, and error statistics.
    """
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N, 2).")
    n = len(points)
    if n < degree + 1:
        raise ValueError(
            f"Need at least {degree + 1} points for degree-{degree} spline, "
            f"got {n}."
        )

    if num_output_points <= 0:
        num_output_points = n

    try:
        from scipy.interpolate import splprep, splev

        x = points[:, 0]
        y = points[:, 1]

        tck, u = splprep(
            [x, y],
            s=smoothing,
            k=degree,
            per=periodic,
        )

        u_new = np.linspace(0, 1, num_output_points)
        x_fit, y_fit = splev(u_new, tck)
        fitted = np.column_stack([x_fit, y_fit])

        # Compute per-point residuals against original points.
        u_orig = u
        x_orig_fit, y_orig_fit = splev(u_orig, tck)
        dx = points[:, 0] - x_orig_fit
        dy = points[:, 1] - y_orig_fit
        residuals = np.sqrt(dx ** 2 + dy ** 2)

        rms = float(np.sqrt(np.mean(residuals ** 2)))
        max_err = float(np.max(residuals))

        return SplineFitResult(
            fitted_points=fitted,
            residuals=residuals,
            rms_error=rms,
            max_error=max_err,
            tck=tck,
        )

    except ImportError:
        logger.warning(
            "scipy not available -- falling back to linear interpolation."
        )
        # Fallback: simple linear interpolation.
        t_orig = np.linspace(0, 1, n)
        t_new = np.linspace(0, 1, num_output_points)
        x_fit = np.interp(t_new, t_orig, points[:, 0])
        y_fit = np.interp(t_new, t_orig, points[:, 1])
        fitted = np.column_stack([x_fit, y_fit])

        # Residuals are zero for linear interp at original parameter values.
        residuals = np.zeros(n)

        return SplineFitResult(
            fitted_points=fitted,
            residuals=residuals,
            rms_error=0.0,
            max_error=0.0,
            tck=None,
        )


# ====================================================================== #
#  Line / form tolerance measurements                                      #
# ====================================================================== #

def measure_parallelism(
    line_a: Tuple[float, float, float, float],
    line_b: Tuple[float, float, float, float],
    tolerance: float = 0.0,
) -> GDT_Result:
    """Measure parallelism deviation between two lines.

    Parameters
    ----------
    line_a : Tuple[float, float, float, float]
        First line as ``(x1, y1, x2, y2)``.
    line_b : Tuple[float, float, float, float]
        Second line as ``(x1, y1, x2, y2)``.
    tolerance : float
        Tolerance limit in radians (0 = no limit check).

    Returns
    -------
    GDT_Result
        Angular deviation in radians.
    """
    angle_a = _line_angle(line_a)
    angle_b = _line_angle(line_b)
    deviation = abs(_angle_diff(angle_a, angle_b))

    return GDT_Result(
        tolerance_type="parallelism",
        value=deviation,
        tolerance=tolerance,
        unit="rad",
        details={
            "angle_a_rad": angle_a,
            "angle_b_rad": angle_b,
            "deviation_deg": math.degrees(deviation),
        },
    )


def measure_perpendicularity(
    line_a: Tuple[float, float, float, float],
    line_b: Tuple[float, float, float, float],
    tolerance: float = 0.0,
) -> GDT_Result:
    """Measure perpendicularity deviation between two lines.

    The ideal angle between perpendicular lines is pi/2 radians.
    The result reports the deviation from that ideal.

    Parameters
    ----------
    line_a, line_b : Tuple[float, float, float, float]
        Lines as ``(x1, y1, x2, y2)``.
    tolerance : float
        Tolerance limit in radians.

    Returns
    -------
    GDT_Result
        Deviation from 90 degrees in radians.
    """
    angle_a = _line_angle(line_a)
    angle_b = _line_angle(line_b)
    angle_between = abs(_angle_diff(angle_a, angle_b))
    deviation = abs(angle_between - math.pi / 2)

    return GDT_Result(
        tolerance_type="perpendicularity",
        value=deviation,
        tolerance=tolerance,
        unit="rad",
        details={
            "angle_between_rad": angle_between,
            "angle_between_deg": math.degrees(angle_between),
            "deviation_deg": math.degrees(deviation),
        },
    )


def measure_concentricity(
    circle_a: Tuple[float, float, float],
    circle_b: Tuple[float, float, float],
    tolerance: float = 0.0,
) -> GDT_Result:
    """Measure concentricity of two circles.

    Concentricity is the distance between the centres of the two circles.

    Parameters
    ----------
    circle_a : Tuple[float, float, float]
        First circle as ``(cx, cy, radius)``.
    circle_b : Tuple[float, float, float]
        Second circle as ``(cx, cy, radius)``.
    tolerance : float
        Tolerance limit in pixels.

    Returns
    -------
    GDT_Result
        Centre-to-centre distance in pixels.
    """
    dx = circle_a[0] - circle_b[0]
    dy = circle_a[1] - circle_b[1]
    offset = math.hypot(dx, dy)

    return GDT_Result(
        tolerance_type="concentricity",
        value=offset,
        tolerance=tolerance,
        unit="px",
        details={
            "center_a": (circle_a[0], circle_a[1]),
            "center_b": (circle_b[0], circle_b[1]),
            "radius_a": circle_a[2],
            "radius_b": circle_b[2],
            "offset_x": dx,
            "offset_y": dy,
        },
    )


def measure_roundness(
    contour: np.ndarray,
    tolerance: float = 0.0,
) -> GDT_Result:
    """Measure roundness of a contour (deviation from best-fit circle).

    Roundness is defined as the difference between the maximum and
    minimum radial distances from the fitted circle centre.

    Parameters
    ----------
    contour : np.ndarray
        Contour points of shape ``(N, 2)`` as ``(x, y)``.
    tolerance : float
        Tolerance limit in pixels.

    Returns
    -------
    GDT_Result
        Roundness value (max_radius - min_radius) in pixels.
    """
    if contour.ndim != 2 or contour.shape[1] != 2:
        raise ValueError("contour must have shape (N, 2).")
    if len(contour) < 3:
        raise ValueError("Need at least 3 points to measure roundness.")

    cx, cy, r_fit = _fit_circle_lsq(contour)
    radii = np.sqrt((contour[:, 0] - cx) ** 2 + (contour[:, 1] - cy) ** 2)
    r_min = float(np.min(radii))
    r_max = float(np.max(radii))
    roundness = r_max - r_min

    return GDT_Result(
        tolerance_type="roundness",
        value=roundness,
        tolerance=tolerance,
        unit="px",
        details={
            "center": (cx, cy),
            "fitted_radius": r_fit,
            "min_radius": r_min,
            "max_radius": r_max,
            "num_points": len(contour),
        },
    )


def measure_straightness(
    points: np.ndarray,
    tolerance: float = 0.0,
) -> GDT_Result:
    """Measure straightness of a set of points (max deviation from fitted line).

    Parameters
    ----------
    points : np.ndarray
        Points of shape ``(N, 2)`` as ``(x, y)``.
    tolerance : float
        Tolerance limit in pixels.

    Returns
    -------
    GDT_Result
        Maximum deviation from the best-fit line in pixels.
    """
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N, 2).")
    if len(points) < 2:
        raise ValueError("Need at least 2 points for straightness.")

    # Fit a line using SVD (total least squares).
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]  # direction of maximum variance

    # Perpendicular distances.
    normal = np.array([-direction[1], direction[0]])
    distances = np.abs(centered @ normal)
    max_dev = float(np.max(distances))
    mean_dev = float(np.mean(distances))

    return GDT_Result(
        tolerance_type="straightness",
        value=max_dev,
        tolerance=tolerance,
        unit="px",
        details={
            "mean_deviation": mean_dev,
            "max_deviation": max_dev,
            "line_direction": tuple(direction),
            "centroid": tuple(centroid),
            "num_points": len(points),
        },
    )


def measure_symmetry(
    contour: np.ndarray,
    axis_point: Tuple[float, float],
    axis_direction: Tuple[float, float],
    tolerance: float = 0.0,
) -> GDT_Result:
    """Measure symmetry of a contour about a given axis.

    For each contour point, computes the reflected point across the axis
    and finds the nearest contour point to the reflection.  Symmetry
    deviation is the maximum of these nearest-neighbour distances.

    Parameters
    ----------
    contour : np.ndarray
        Contour points of shape ``(N, 2)`` as ``(x, y)``.
    axis_point : Tuple[float, float]
        A point on the symmetry axis.
    axis_direction : Tuple[float, float]
        Direction vector of the symmetry axis (will be normalised).
    tolerance : float
        Tolerance limit in pixels.

    Returns
    -------
    GDT_Result
        Maximum symmetry deviation in pixels.
    """
    if contour.ndim != 2 or contour.shape[1] != 2:
        raise ValueError("contour must have shape (N, 2).")

    ax_pt = np.array(axis_point, dtype=np.float64)
    ax_dir = np.array(axis_direction, dtype=np.float64)
    ax_dir = ax_dir / (np.linalg.norm(ax_dir) + _EPS)

    # Reflect all contour points across the axis at once.
    pts = contour.astype(np.float64)
    v = pts - ax_pt
    proj_lengths = v @ ax_dir
    projections = ax_pt + proj_lengths[:, np.newaxis] * ax_dir
    reflected = 2.0 * projections - pts

    # For each reflected point, find nearest original point using cKDTree.
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(pts)
        distances, _ = tree.query(reflected, k=1)
    except ImportError:
        # Fallback: vectorized pairwise distance (O(N^2) memory but no scipy)
        diff = reflected[:, np.newaxis, :] - pts[np.newaxis, :, :]
        distances = np.min(np.sqrt(np.sum(diff ** 2, axis=2)), axis=1)

    max_dev = float(distances.max())
    mean_dev = float(distances.mean())

    return GDT_Result(
        tolerance_type="symmetry",
        value=max_dev,
        tolerance=tolerance,
        unit="px",
        details={
            "mean_deviation": mean_dev,
            "max_deviation": max_dev,
            "axis_point": axis_point,
            "axis_direction": tuple(ax_dir),
            "num_points": len(contour),
        },
    )


# ====================================================================== #
#  Internal helpers                                                        #
# ====================================================================== #

def _line_angle(line: Tuple[float, float, float, float]) -> float:
    """Compute the angle of a line segment in radians [0, pi)."""
    dx = line[2] - line[0]
    dy = line[3] - line[1]
    angle = math.atan2(dy, dx)
    # Normalise to [0, pi) since line direction is ambiguous.
    if angle < 0:
        angle += math.pi
    return angle


def _angle_diff(a: float, b: float) -> float:
    """Compute the smallest signed angular difference in [-pi/2, pi/2]."""
    diff = a - b
    # Wrap to [-pi, pi].
    while diff > math.pi:
        diff -= math.pi
    while diff < -math.pi:
        diff += math.pi
    # For line angles, direction is ambiguous, so wrap to [-pi/2, pi/2].
    if diff > math.pi / 2:
        diff -= math.pi
    elif diff < -math.pi / 2:
        diff += math.pi
    return diff


def _fit_circle_lsq(points: np.ndarray) -> Tuple[float, float, float]:
    """Fit a circle to 2D points using algebraic least squares.

    Uses the Kasa method (linear least squares on the equation
    ``x^2 + y^2 + Dx + Ey + F = 0``).

    Returns
    -------
    Tuple[float, float, float]
        ``(cx, cy, radius)``
    """
    x = points[:, 0]
    y = points[:, 1]

    A = np.column_stack([x, y, np.ones(len(x))])
    b = -(x ** 2 + y ** 2)

    result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    D, E, F = result

    cx = -D / 2.0
    cy = -E / 2.0
    r = math.sqrt(max(cx ** 2 + cy ** 2 - F, 0))

    return float(cx), float(cy), float(r)


def _find_contour_intersection(
    contour: np.ndarray,
    cx: float,
    cy: float,
    dx: float,
    dy: float,
) -> Optional[Tuple[float, float]]:
    """Find the intersection of a ray with a contour.

    The ray starts at ``(cx, cy)`` in direction ``(dx, dy)``.
    Returns the nearest intersection point in the positive ray
    direction, or None if no intersection is found.

    Parameters
    ----------
    contour : np.ndarray
        Contour points of shape ``(N, 2)`` as ``(x, y)``.
    cx, cy : float
        Ray origin.
    dx, dy : float
        Ray direction (need not be normalised).

    Returns
    -------
    Tuple[float, float] or None
        Intersection point ``(x, y)`` or None.
    """
    n = len(contour)
    if n < 2:
        return None

    best_t = float("inf")
    best_pt: Optional[Tuple[float, float]] = None

    for i in range(n):
        j = (i + 1) % n
        # Segment from contour[i] to contour[j].
        sx, sy = contour[i]
        ex, ey = contour[j]
        seg_dx = ex - sx
        seg_dy = ey - sy

        denom = dx * seg_dy - dy * seg_dx
        if abs(denom) < _EPS:
            continue

        t = ((sx - cx) * seg_dy - (sy - cy) * seg_dx) / denom
        u = ((sx - cx) * dy - (sy - cy) * dx) / denom

        if t > _EPS and 0.0 <= u <= 1.0 and t < best_t:
            best_t = t
            best_pt = (cx + t * dx, cy + t * dy)

    return best_pt
