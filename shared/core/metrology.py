"""
core/metrology.py - Sub-pixel measurement and metrology operators.

Provides HALCON-style measurement and geometric fitting operators for
precision metrology tasks in defect detection pipelines.

Categories:
    1. Data Classes (SubPixelEdge, MeasureRectangle, MeasurePair, etc.)
    2. Sub-pixel Edge Detection
    3. Measurement Rectangle Operations
    4. Geometric Fitting (line, circle, ellipse)
    5. Distance / Angle Measurements
    6. Drawing Helpers
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from shared.validation import validate_image, validate_positive
from shared.op_logger import log_operation

logger = logging.getLogger(__name__)

# ====================================================================== #
#  Named constants                                                        #
# ====================================================================== #

DEFAULT_ALPHA = 1.0
DEFAULT_LOW = 20.0
DEFAULT_HIGH = 40.0
DEFAULT_SIGMA = 1.0
DEFAULT_THRESHOLD = 30.0
IRLS_MAX_ITER = 50
IRLS_TOL = 1e-6
PARALLEL_TOL = 1e-12


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


def _ensure_uint8(img: np.ndarray) -> np.ndarray:
    """Clip and convert an array to ``uint8``."""
    return np.clip(img, 0, 255).astype(np.uint8)


def _ksize_from_sigma(sigma: float) -> int:
    """Compute a suitable odd kernel size from a Gaussian sigma."""
    k = int(math.ceil(sigma * 6)) | 1
    return max(k, 3)


def _bilinear_sample(
    img: np.ndarray, x: np.ndarray, y: np.ndarray,
) -> np.ndarray:
    """Sample *img* at sub-pixel positions via bilinear interpolation.

    Parameters
    ----------
    img : np.ndarray
        2-D grayscale image (float64 or uint8).
    x, y : np.ndarray
        1-D arrays of column and row coordinates respectively.

    Returns
    -------
    np.ndarray
        Interpolated intensity values (float64).
    """
    h, w = img.shape[:2]
    img_f = img.astype(np.float64)

    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    x0c = np.clip(x0, 0, w - 1)
    x1c = np.clip(x1, 0, w - 1)
    y0c = np.clip(y0, 0, h - 1)
    y1c = np.clip(y1, 0, h - 1)

    dx = x - x0.astype(np.float64)
    dy = y - y0.astype(np.float64)

    return (
        img_f[y0c, x0c] * (1.0 - dx) * (1.0 - dy)
        + img_f[y0c, x1c] * dx * (1.0 - dy)
        + img_f[y1c, x0c] * (1.0 - dx) * dy
        + img_f[y1c, x1c] * dx * dy
    )


# ====================================================================== #
#  Data classes                                                           #
# ====================================================================== #


@dataclass
class SubPixelEdge:
    """A single edge location with sub-pixel accuracy.

    Attributes:
        row:       Sub-pixel Y coordinate.
        col:       Sub-pixel X coordinate.
        angle:     Edge direction in radians.
        amplitude: Edge strength (gradient magnitude).
        type:      ``"rising"`` or ``"falling"``.
    """

    row: float
    col: float
    angle: float
    amplitude: float
    type: str  # "rising" or "falling"


@dataclass
class MeasureRectangle:
    """Measurement rectangle (ROI for 1-D edge measurement).

    Attributes:
        row:     Centre Y.
        col:     Centre X.
        phi:     Rotation angle in radians.
        length1: Half-width along the main axis.
        length2: Half-height perpendicular to the main axis.
    """

    row: float
    col: float
    phi: float
    length1: float
    length2: float


@dataclass
class MeasurePair:
    """A pair of edges (rising + falling) with the distance between them.

    Attributes:
        edge1:    First edge (typically rising).
        edge2:    Second edge (typically falling).
        distance: Euclidean distance between the two edges.
    """

    edge1: SubPixelEdge
    edge2: SubPixelEdge
    distance: float


@dataclass
class FitResult:
    """Result of a geometric primitive fit.

    Attributes:
        type:        ``"line"``, ``"circle"``, or ``"ellipse"``.
        params:      Dictionary of fitted parameters (type-dependent).

                     * **line**: ``row1, col1, row2, col2, angle, distance``
                       (Hesse normal form distance from origin).
                     * **circle**: ``row, col, radius``.
                     * **ellipse**: ``row, col, phi, ra, rb``
                       (centre, angle, semi-major, semi-minor).

        error:       RMS fitting error.
        points_used: Number of inlier points used in the fit.
    """

    type: str
    params: Dict[str, float]
    error: float
    points_used: int


@dataclass
class MeasurementResult:
    """A scalar measurement with optional endpoint annotation.

    Attributes:
        value:  The measured numeric value.
        unit:   Unit string, e.g. ``"px"`` or ``"mm"``.
        type:   Measurement kind: ``"distance"``, ``"angle"``, ``"radius"``, etc.
        point1: Optional first reference point ``(row, col)``.
        point2: Optional second reference point ``(row, col)``.
    """

    value: float
    unit: str
    type: str
    point1: Optional[Tuple[float, float]] = None
    point2: Optional[Tuple[float, float]] = None


# ====================================================================== #
#  2. Sub-pixel edge detection                                            #
# ====================================================================== #


@log_operation(logger)
def edges_sub_pix(
    image: np.ndarray,
    alpha: float = DEFAULT_ALPHA,
    low: float = DEFAULT_LOW,
    high: float = DEFAULT_HIGH,
) -> List[SubPixelEdge]:
    """Detect edges with sub-pixel accuracy using Canny-style processing.

    Algorithm
    ---------
    1. Gaussian smoothing (sigma = *alpha*).
    2. Sobel gradient computation (Gx, Gy).
    3. Gradient magnitude and direction.
    4. Non-maximum suppression along the gradient direction.
    5. For each surviving edge pixel, refine position by fitting a parabola
       to the three gradient-magnitude values perpendicular to the edge
       direction.
    6. Hysteresis thresholding (*low*, *high*).

    Parameters
    ----------
    image : np.ndarray
        Input image (grayscale or colour).
    alpha : float
        Gaussian smoothing sigma.
    low : float
        Low hysteresis threshold on gradient magnitude.
    high : float
        High hysteresis threshold on gradient magnitude.

    Returns
    -------
    List[SubPixelEdge]
    """
    validate_image(image)
    validate_positive(alpha, "alpha")

    gray = _ensure_gray(image).astype(np.float64)
    h, w = gray.shape

    # 1. Gaussian smoothing
    ksize = _ksize_from_sigma(alpha)
    smoothed = cv2.GaussianBlur(gray, (ksize, ksize), alpha)

    # 2. Sobel gradients
    gx = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=3)

    # 3. Magnitude and direction
    mag = np.hypot(gx, gy)
    angle = np.arctan2(gy, gx)

    # 4. Non-maximum suppression (vectorised over 4 quantised directions)
    nms = np.zeros_like(mag)
    direction = (np.round(angle / (np.pi / 4.0)) % 4).astype(np.int32)

    offsets = [(0, 1), (1, 1), (1, 0), (1, -1)]
    for d, (dr, dc) in enumerate(offsets):
        mask_d = direction[1:-1, 1:-1] == d
        r_idx, c_idx = np.where(mask_d)
        r_idx += 1
        c_idx += 1
        m_center = mag[r_idx, c_idx]
        m_pos = mag[r_idx + dr, c_idx + dc]
        m_neg = mag[r_idx - dr, c_idx - dc]
        survive = (m_center >= m_pos) & (m_center >= m_neg)
        nms[r_idx[survive], c_idx[survive]] = mag[
            r_idx[survive], c_idx[survive]
        ]

    # 6. Hysteresis thresholding
    strong = nms >= high
    weak = (nms >= low) & (~strong)

    # BFS-style hysteresis: dilate strong mask into weak pixels
    edge_map = strong.copy()
    kernel_3x3 = np.ones((3, 3), np.uint8)
    prev_count = 0
    while True:
        dilated = cv2.dilate(
            edge_map.astype(np.uint8), kernel_3x3,
        ).astype(bool)
        edge_map = edge_map | (dilated & weak)
        cur_count = int(np.count_nonzero(edge_map))
        if cur_count == prev_count:
            break
        prev_count = cur_count

    # 5. Sub-pixel refinement via parabola fit along gradient direction
    ey, ex = np.where(edge_map)
    if ey.size == 0:
        logger.debug("edges_sub_pix: no edges found")
        return []

    dirs = angle[ey, ex]
    cos_d = np.cos(dirs)
    sin_d = np.sin(dirs)

    # Neighbour coordinates along gradient direction
    ry_m = np.clip(np.round(ey - sin_d).astype(int), 0, h - 1)
    rx_m = np.clip(np.round(ex - cos_d).astype(int), 0, w - 1)
    ry_p = np.clip(np.round(ey + sin_d).astype(int), 0, h - 1)
    rx_p = np.clip(np.round(ex + cos_d).astype(int), 0, w - 1)

    f_m = mag[ry_m, rx_m]
    f_0 = mag[ey, ex]
    f_p = mag[ry_p, rx_p]

    denom = f_m - 2.0 * f_0 + f_p
    safe = np.abs(denom) > 1e-12
    offset = np.zeros(ey.size, dtype=np.float64)
    offset[safe] = 0.5 * (f_m[safe] - f_p[safe]) / denom[safe]
    offset = np.clip(offset, -0.5, 0.5)

    sub_row = ey.astype(np.float64) + offset * sin_d
    sub_col = ex.astype(np.float64) + offset * cos_d

    # Classify edge type by sign of the gradient projection
    grad_along = gx[ey, ex] * cos_d + gy[ey, ex] * sin_d
    types = np.where(grad_along >= 0, "rising", "falling")

    edges: List[SubPixelEdge] = []
    for i in range(ey.size):
        edges.append(
            SubPixelEdge(
                row=float(sub_row[i]),
                col=float(sub_col[i]),
                angle=float(dirs[i]),
                amplitude=float(f_0[i]),
                type=str(types[i]),
            )
        )

    logger.debug("edges_sub_pix: found %d sub-pixel edges", len(edges))
    return edges


# ====================================================================== #
#  3. Measurement rectangle operations                                    #
# ====================================================================== #


def gen_measure_rect2(
    row: float,
    col: float,
    phi: float,
    length1: float,
    length2: float,
) -> MeasureRectangle:
    """Create a measurement rectangle (ROI for 1-D edge measurement).

    Parameters
    ----------
    row : float
        Centre Y coordinate.
    col : float
        Centre X coordinate.
    phi : float
        Rotation angle in radians.
    length1 : float
        Half-width along the main axis.
    length2 : float
        Half-height perpendicular to the main axis.

    Returns
    -------
    MeasureRectangle
    """
    validate_positive(length1, "length1")
    validate_positive(length2, "length2")
    return MeasureRectangle(
        row=float(row),
        col=float(col),
        phi=float(phi),
        length1=float(length1),
        length2=float(length2),
    )


@log_operation(logger)
def measure_pos(
    image: np.ndarray,
    measure_rect: MeasureRectangle,
    sigma: float = DEFAULT_SIGMA,
    threshold: float = DEFAULT_THRESHOLD,
    transition: str = "all",
    select: str = "all",
) -> List[SubPixelEdge]:
    """Find sub-pixel edge positions along a measurement rectangle.

    Extracts 1-D profiles perpendicular to the rectangle's main axis,
    averages them for noise robustness, then detects edges as local
    extrema of the smoothed derivative with parabola-based sub-pixel
    refinement.

    Parameters
    ----------
    image : np.ndarray
        Input image (grayscale or colour).
    measure_rect : MeasureRectangle
        The measurement rectangle ROI.
    sigma : float
        Gaussian sigma for 1-D profile smoothing.
    threshold : float
        Minimum derivative amplitude to accept an edge.
    transition : str
        ``"positive"`` (rising), ``"negative"`` (falling), or ``"all"``.
    select : str
        ``"first"``, ``"last"``, or ``"all"``.

    Returns
    -------
    List[SubPixelEdge]
        Edges in image coordinates, sorted by position along the main axis.
    """
    validate_image(image)
    validate_positive(sigma, "sigma")

    gray = _ensure_gray(image)
    gray = _ensure_uint8(gray)
    mr = measure_rect

    cos_phi = math.cos(mr.phi)
    sin_phi = math.sin(mr.phi)
    # Perpendicular direction
    perp_cos = -sin_phi
    perp_sin = cos_phi

    # Number of samples along the main axis
    n_samples = max(int(round(2.0 * mr.length1)) + 1, 3)
    # Number of profile lines perpendicular to main axis
    n_lines = max(int(round(2.0 * mr.length2)) + 1, 1)

    t_main = np.linspace(-mr.length1, mr.length1, n_samples)
    t_perp = np.linspace(-mr.length2, mr.length2, n_lines)

    # Accumulate profiles via bilinear sampling
    profile = np.zeros(n_samples, dtype=np.float64)
    valid_count = 0
    for tp in t_perp:
        sample_cols = mr.col + t_main * cos_phi + tp * perp_cos
        sample_rows = mr.row + t_main * sin_phi + tp * perp_sin
        profile += _bilinear_sample(gray, sample_cols, sample_rows)
        valid_count += 1

    if valid_count == 0:
        logger.warning("measure_pos: no valid profiles extracted")
        return []
    profile /= valid_count

    # Smooth the 1-D profile
    if sigma > 0.3:
        ks = _ksize_from_sigma(sigma)
        profile = cv2.GaussianBlur(
            profile.reshape(1, -1), (ks, 1), sigma,
        ).ravel()

    # Central-difference derivative
    deriv = np.zeros_like(profile)
    deriv[1:-1] = (profile[2:] - profile[:-2]) / 2.0

    # Find edges as local extrema of the derivative that exceed threshold
    edges: List[SubPixelEdge] = []
    for i in range(1, len(deriv) - 1):
        if abs(deriv[i]) < threshold:
            continue
        # Must be a local extremum
        if abs(deriv[i]) < abs(deriv[i - 1]) or abs(deriv[i]) < abs(deriv[i + 1]):
            continue

        # Determine transition type
        if deriv[i] > 0:
            edge_type = "rising"
        else:
            edge_type = "falling"

        # Filter by transition
        if transition == "positive" and edge_type != "rising":
            continue
        if transition == "negative" and edge_type != "falling":
            continue

        # Parabolic sub-pixel refinement on derivative
        dm = deriv[i - 1]
        d0 = deriv[i]
        dp = deriv[i + 1]
        denom = 2.0 * (dm - 2.0 * d0 + dp)
        if abs(denom) > 1e-10:
            offset = (dm - dp) / denom
        else:
            offset = 0.0
        offset = max(-0.5, min(0.5, offset))

        # Sub-pixel position along the profile
        t_sub = t_main[0] + (i + offset) * (t_main[-1] - t_main[0]) / (n_samples - 1)

        # Transform back to image coordinates
        edge_col = mr.col + t_sub * cos_phi
        edge_row = mr.row + t_sub * sin_phi

        edges.append(
            SubPixelEdge(
                row=float(edge_row),
                col=float(edge_col),
                angle=mr.phi,
                amplitude=float(abs(d0)),
                type=edge_type,
            )
        )

    # Filter by select
    if select == "first" and edges:
        edges = [edges[0]]
    elif select == "last" and edges:
        edges = [edges[-1]]

    logger.debug("measure_pos: found %d edge(s)", len(edges))
    return edges


@log_operation(logger)
def measure_pairs(
    image: np.ndarray,
    measure_rect: MeasureRectangle,
    sigma: float = DEFAULT_SIGMA,
    threshold: float = DEFAULT_THRESHOLD,
    transition: str = "all",
    select: str = "all",
) -> List[MeasurePair]:
    """Find edge pairs (rising + falling) along a measurement rectangle.

    Similar to :func:`measure_pos` but groups consecutive edges into pairs
    of rising followed by falling transitions.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    measure_rect : MeasureRectangle
        Measurement rectangle ROI.
    sigma : float
        Gaussian sigma for 1-D smoothing.
    threshold : float
        Minimum derivative amplitude.
    transition : str
        ``"positive"``, ``"negative"``, or ``"all"``.
    select : str
        ``"first"``, ``"last"``, or ``"all"``.

    Returns
    -------
    List[MeasurePair]
        Each pair contains a rising and falling edge with the distance
        between them.
    """
    all_edges = measure_pos(
        image, measure_rect,
        sigma=sigma, threshold=threshold,
        transition="all", select="all",
    )

    # Group into pairs: consecutive rising -> falling
    pairs: List[MeasurePair] = []
    i = 0
    while i < len(all_edges) - 1:
        e1 = all_edges[i]
        e2 = all_edges[i + 1]
        if e1.type == "rising" and e2.type == "falling":
            dist = math.hypot(e2.row - e1.row, e2.col - e1.col)
            pairs.append(MeasurePair(edge1=e1, edge2=e2, distance=dist))
            i += 2
        else:
            i += 1

    # Filter by transition (applied to pair orientation)
    if transition == "positive":
        pairs = [p for p in pairs if p.edge1.type == "rising"]
    elif transition == "negative":
        pairs = [p for p in pairs if p.edge1.type == "falling"]

    # Filter by select
    if select == "first" and pairs:
        pairs = [pairs[0]]
    elif select == "last" and pairs:
        pairs = [pairs[-1]]

    logger.debug("measure_pairs: found %d pair(s)", len(pairs))
    return pairs


# ====================================================================== #
#  4. Geometric fitting                                                   #
# ====================================================================== #


def _huber_weight(residuals: np.ndarray, c: float) -> np.ndarray:
    """Huber weighting function for robust fitting."""
    w = np.ones_like(residuals)
    mask = np.abs(residuals) > c
    w[mask] = c / np.abs(residuals[mask])
    return w


def _tukey_weight(residuals: np.ndarray, c: float) -> np.ndarray:
    """Tukey bisquare weighting function for robust fitting."""
    w = np.zeros_like(residuals)
    mask = np.abs(residuals) < c
    u = residuals[mask] / c
    w[mask] = (1.0 - u ** 2) ** 2
    return w


def _robust_weights(
    residuals: np.ndarray,
    algorithm: str,
    clipping_factor: float,
) -> np.ndarray:
    """Compute robust weights from residuals based on the algorithm."""
    mad = np.median(np.abs(residuals - np.median(residuals)))
    sigma_est = 1.4826 * mad if mad > 1e-12 else 1.0
    normalised = residuals / sigma_est

    if algorithm == "huber":
        return _huber_weight(normalised, clipping_factor)
    elif algorithm == "tukey":
        return _tukey_weight(normalised, clipping_factor)
    return np.ones_like(residuals)


@log_operation(logger)
def fit_line_contour_xld(
    points: Union[Sequence[Tuple[float, float]], np.ndarray],
    algorithm: str = "tukey",
    max_num_points: int = -1,
    clipping_factor: float = 2.0,
) -> FitResult:
    """Fit a line to a set of 2-D contour points.

    Parameters
    ----------
    points : sequence of (row, col)
        Input contour points.
    algorithm : str
        ``"regression"`` (ordinary least squares), ``"tukey"`` (robust
        M-estimator with Tukey bisquare weights), or ``"huber"`` (robust
        M-estimator with Huber weights).
    max_num_points : int
        Maximum points to use (``-1`` = all).
    clipping_factor : float
        Outlier clipping factor for robust methods.

    Returns
    -------
    FitResult
        ``type="line"`` with params ``row1, col1, row2, col2, angle,
        distance`` (Hesse normal form distance from origin).

    Raises
    ------
    ValueError
        If fewer than 2 points are provided or an unknown algorithm is
        specified.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(-1, 2)
    if pts.shape[0] < 2:
        raise ValueError("fit_line_contour_xld requires at least 2 points")

    if max_num_points > 0 and len(pts) > max_num_points:
        idx = np.random.choice(len(pts), max_num_points, replace=False)
        pts = pts[idx]

    if algorithm not in ("regression", "tukey", "huber"):
        raise ValueError(
            f"fit_line_contour_xld: unknown algorithm '{algorithm}', "
            f"expected 'regression', 'tukey', or 'huber'"
        )

    n = len(pts)
    rows = pts[:, 0]
    cols = pts[:, 1]
    weights = np.ones(n, dtype=np.float64)

    max_iter = 1 if algorithm == "regression" else IRLS_MAX_ITER

    nx = ny = 0.0
    dir_r = dir_c = 0.0
    cx = cy = 0.0

    for iteration in range(max_iter):
        sw = np.sum(weights)
        cx = float(np.sum(weights * cols) / sw)
        cy = float(np.sum(weights * rows) / sw)

        dc = cols - cx
        dr = rows - cy

        # Weighted covariance matrix
        cov = np.array([
            [np.sum(weights * dc * dc), np.sum(weights * dc * dr)],
            [np.sum(weights * dc * dr), np.sum(weights * dr * dr)],
        ])

        _, _, vt = np.linalg.svd(cov)
        # Direction = first singular vector (largest variance)
        dir_c, dir_r = float(vt[0, 0]), float(vt[0, 1])
        # Normal = second singular vector (least variance)
        nx, ny = float(vt[1, 0]), float(vt[1, 1])

        # Signed residuals (perpendicular distances)
        residuals = (cols - cx) * nx + (rows - cy) * ny

        if algorithm != "regression":
            weights = _robust_weights(residuals, algorithm, clipping_factor)
            weights = np.maximum(weights, 1e-12)

            # Check convergence
            if iteration > 0:
                new_rms = float(np.sqrt(np.mean(residuals ** 2)))
                if iteration > 1 and abs(new_rms - _prev_rms) < IRLS_TOL:
                    break
                _prev_rms = new_rms
            else:
                _prev_rms = float(np.sqrt(np.mean(residuals ** 2)))

    # Ensure consistent normal direction
    if nx < 0 or (abs(nx) < 1e-12 and ny < 0):
        nx, ny = -nx, -ny
        dir_c, dir_r = -dir_c, -dir_r

    # Hesse normal form distance from origin
    hesse_dist = float(abs(cx * nx + cy * ny))

    # Line angle w.r.t. horizontal (angle of the direction vector)
    line_angle = float(math.atan2(dir_r, dir_c))

    # Project points onto the direction to find endpoints
    projections = (cols - cx) * dir_c + (rows - cy) * dir_r
    t_min = float(np.min(projections))
    t_max = float(np.max(projections))

    row1 = cy + t_min * dir_r
    col1 = cx + t_min * dir_c
    row2 = cy + t_max * dir_r
    col2 = cx + t_max * dir_c

    # Final RMS error
    residuals = (cols - cx) * nx + (rows - cy) * ny
    rms = float(np.sqrt(np.mean(residuals ** 2)))

    points_used = int(np.sum(weights > 1e-6)) if algorithm != "regression" else n

    logger.debug(
        "fit_line_contour_xld (%s): angle=%.4f rad, rms=%.4f",
        algorithm, line_angle, rms,
    )
    return FitResult(
        type="line",
        params={
            "row1": float(row1),
            "col1": float(col1),
            "row2": float(row2),
            "col2": float(col2),
            "angle": line_angle,
            "distance": hesse_dist,
        },
        error=rms,
        points_used=points_used,
    )


@log_operation(logger)
def fit_circle_contour_xld(
    points: Union[Sequence[Tuple[float, float]], np.ndarray],
    algorithm: str = "algebraic",
    max_num_points: int = -1,
    clipping_factor: float = 2.0,
) -> FitResult:
    """Fit a circle to a set of 2-D contour points.

    Parameters
    ----------
    points : sequence of (row, col)
        Input contour points.
    algorithm : str
        ``"algebraic"`` (Kasa method) or ``"geometric"`` (iterative
        refinement with robust weighting).
    max_num_points : int
        Maximum points to use (``-1`` = all).
    clipping_factor : float
        Outlier clipping factor for the geometric method.

    Returns
    -------
    FitResult
        ``type="circle"`` with params ``row, col, radius``.

    Raises
    ------
    ValueError
        If fewer than 3 points are provided.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(-1, 2)
    if pts.shape[0] < 3:
        raise ValueError("fit_circle_contour_xld requires at least 3 points")

    if max_num_points > 0 and len(pts) > max_num_points:
        idx = np.random.choice(len(pts), max_num_points, replace=False)
        pts = pts[idx]

    n = len(pts)
    rows = pts[:, 0]
    cols = pts[:, 1]

    # Kasa algebraic method: solve  [2c, 2r, 1] * [cc, cr, d]^T = c^2 + r^2
    A = np.column_stack([2.0 * cols, 2.0 * rows, np.ones(n)])
    b = cols ** 2 + rows ** 2
    result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cc = float(result[0])
    cr = float(result[1])
    r_sq = cc ** 2 + cr ** 2 - result[2]
    radius = math.sqrt(max(float(r_sq), 0.0))

    if algorithm == "geometric":
        # Iterative refinement with robust weighting
        for _ in range(30):
            dist = np.sqrt((cols - cc) ** 2 + (rows - cr) ** 2)
            residuals = dist - radius

            weights = _robust_weights(residuals, "tukey", clipping_factor)
            weights = np.maximum(weights, 1e-12)
            sw = np.sum(weights)

            cc = float(np.sum(weights * cols) / sw)
            cr = float(np.sum(weights * rows) / sw)
            dist = np.sqrt((cols - cc) ** 2 + (rows - cr) ** 2)
            radius = float(np.sum(weights * dist) / sw)

        points_used = int(np.sum(weights > 1e-6))
    else:
        points_used = n

    # RMS error
    dists = np.sqrt((cols - cc) ** 2 + (rows - cr) ** 2)
    rms = float(np.sqrt(np.mean((dists - radius) ** 2)))

    logger.debug(
        "fit_circle_contour_xld (%s): row=%.2f, col=%.2f, r=%.2f, rms=%.4f",
        algorithm, cr, cc, radius, rms,
    )
    return FitResult(
        type="circle",
        params={
            "row": cr,
            "col": cc,
            "radius": radius,
        },
        error=rms,
        points_used=points_used,
    )


@log_operation(logger)
def fit_ellipse_contour_xld(
    points: Union[Sequence[Tuple[float, float]], np.ndarray],
    algorithm: str = "fitzgibbon",
    max_num_points: int = -1,
    clipping_factor: float = 2.0,
) -> FitResult:
    """Fit an ellipse to a set of 2-D contour points.

    Uses the direct least-squares fitting of ellipses method by
    Fitzgibbon, Pilu, and Fisher (1999), which solves a constrained
    eigenvalue problem guaranteeing an ellipse result.

    Parameters
    ----------
    points : sequence of (row, col)
        Input contour points.
    algorithm : str
        ``"fitzgibbon"`` (direct least squares).
    max_num_points : int
        Maximum points to use (``-1`` = all).
    clipping_factor : float
        Outlier clipping factor (reserved for future use).

    Returns
    -------
    FitResult
        ``type="ellipse"`` with params ``row, col`` (centre), ``phi``
        (angle in radians), ``ra`` (semi-major), ``rb`` (semi-minor).

    Raises
    ------
    ValueError
        If fewer than 5 points are provided or the conic is degenerate.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(-1, 2)
    if pts.shape[0] < 5:
        raise ValueError("fit_ellipse_contour_xld requires at least 5 points")

    if max_num_points > 0 and len(pts) > max_num_points:
        idx = np.random.choice(len(pts), max_num_points, replace=False)
        pts = pts[idx]

    n = len(pts)
    rows = pts[:, 0]
    cols = pts[:, 1]

    # Centre and scale data for numerical stability
    mean_c = float(np.mean(cols))
    mean_r = float(np.mean(rows))
    std_c = float(np.std(cols)) if np.std(cols) > 1e-10 else 1.0
    std_r = float(np.std(rows)) if np.std(rows) > 1e-10 else 1.0
    xn = (cols - mean_c) / std_c
    yn = (rows - mean_r) / std_r

    # Design matrix: [x^2, xy, y^2, x, y, 1]
    D = np.column_stack([xn ** 2, xn * yn, yn ** 2, xn, yn, np.ones(n)])

    # Scatter matrix
    S = D.T @ D

    # Constraint matrix for 4ac - b^2 = 1 (ellipse constraint)
    C = np.zeros((6, 6), dtype=np.float64)
    C[0, 2] = 2.0
    C[2, 0] = 2.0
    C[1, 1] = -1.0

    # Generalised eigenvalue problem: S a = lambda C a
    try:
        eigvals, eigvecs = np.linalg.eig(np.linalg.inv(S) @ C)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(S) @ C)

    # Find the eigenvector satisfying the ellipse constraint 4ac - b^2 > 0
    real_mask = np.isreal(eigvals)
    best_idx = -1
    best_val = -np.inf
    for i in range(6):
        if not real_mask[i]:
            continue
        ev = eigvals[i].real
        vec = eigvecs[:, i].real
        cond = 4.0 * vec[0] * vec[2] - vec[1] ** 2
        if cond > 0 and ev > best_val:
            best_val = ev
            best_idx = i

    if best_idx < 0:
        # Fallback: smallest positive eigenvalue
        for i in range(6):
            if not real_mask[i]:
                continue
            ev = eigvals[i].real
            if ev > 0 and (best_idx < 0 or ev < best_val):
                best_val = ev
                best_idx = i

    if best_idx < 0:
        raise ValueError("fit_ellipse_contour_xld: no valid ellipse found")

    coeffs = eigvecs[:, best_idx].real
    a, b, c, d, e, f = coeffs

    # Un-normalise coefficients to original coordinate space
    A2 = a / (std_c * std_c)
    B2 = b / (std_c * std_r)
    C2 = c / (std_r * std_r)
    D2 = d / std_c - 2.0 * a * mean_c / (std_c * std_c) - b * mean_r / (std_c * std_r)
    E2 = e / std_r - 2.0 * c * mean_r / (std_r * std_r) - b * mean_c / (std_c * std_r)
    F2 = (
        a * mean_c ** 2 / (std_c * std_c)
        + b * mean_c * mean_r / (std_c * std_r)
        + c * mean_r ** 2 / (std_r * std_r)
        - d * mean_c / std_c
        - e * mean_r / std_r
        + f
    )

    a, b, c, d, e, f = A2, B2, C2, D2, E2, F2

    # Extract ellipse parameters from general conic
    det_M = a * c - (b / 2.0) ** 2
    if abs(det_M) < 1e-15:
        raise ValueError("fit_ellipse_contour_xld: degenerate conic (det ~ 0)")

    denom_4 = 4.0 * a * c - b ** 2
    col_c = float((b * e - 2.0 * c * d) / denom_4)
    row_c = float((b * d - 2.0 * a * e) / denom_4)

    # Rotation angle
    if abs(a - c) < 1e-15:
        phi = math.pi / 4.0 if b > 0 else -math.pi / 4.0
    else:
        phi = 0.5 * math.atan2(b, a - c)

    # Semi-axes
    cos_t = math.cos(phi)
    sin_t = math.sin(phi)
    a_rot = a * cos_t ** 2 + b * cos_t * sin_t + c * sin_t ** 2
    c_rot = a * sin_t ** 2 - b * cos_t * sin_t + c * cos_t ** 2

    num = a * col_c ** 2 + b * col_c * row_c + c * row_c ** 2 - f

    if abs(a_rot) < 1e-15 or abs(c_rot) < 1e-15:
        raise ValueError("fit_ellipse_contour_xld: degenerate axes")

    ra_sq = num / a_rot
    rb_sq = num / c_rot

    ra = math.sqrt(max(ra_sq, 0.0))
    rb = math.sqrt(max(rb_sq, 0.0))

    # Ensure ra >= rb (semi-major >= semi-minor)
    if rb > ra:
        ra, rb = rb, ra
        phi += math.pi / 2.0

    # Normalise angle to [-pi/2, pi/2]
    phi = math.atan2(math.sin(phi), math.cos(phi))
    if phi > math.pi / 2.0:
        phi -= math.pi
    elif phi < -math.pi / 2.0:
        phi += math.pi

    # RMS error (approximate geometric distance to ellipse)
    algebraic = (
        a * cols ** 2 + b * cols * rows + c * rows ** 2
        + d * cols + e * rows + f
    )
    grad_x = 2.0 * a * cols + b * rows + d
    grad_y = b * cols + 2.0 * c * rows + e
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    grad_mag = np.where(grad_mag < 1e-10, 1.0, grad_mag)
    geo_dist = algebraic / grad_mag
    rms = float(np.sqrt(np.mean(geo_dist ** 2)))

    logger.debug(
        "fit_ellipse_contour_xld: row=%.2f, col=%.2f, ra=%.2f, rb=%.2f, "
        "phi=%.4f rad, rms=%.4f",
        row_c, col_c, ra, rb, phi, rms,
    )
    return FitResult(
        type="ellipse",
        params={
            "row": float(row_c),
            "col": float(col_c),
            "phi": float(phi),
            "ra": float(ra),
            "rb": float(rb),
        },
        error=rms,
        points_used=n,
    )


# ====================================================================== #
#  5. Distance / angle measurements                                       #
# ====================================================================== #


def distance_pp(
    row1: float, col1: float, row2: float, col2: float,
) -> float:
    """Euclidean distance between two points.

    Parameters
    ----------
    row1, col1 : float
        First point coordinates.
    row2, col2 : float
        Second point coordinates.

    Returns
    -------
    float
        Distance in pixels.
    """
    return math.hypot(row2 - row1, col2 - col1)


def distance_pl(
    row_point: float,
    col_point: float,
    row1_line: float,
    col1_line: float,
    row2_line: float,
    col2_line: float,
) -> float:
    """Perpendicular distance from a point to a line defined by two points.

    Parameters
    ----------
    row_point, col_point : float
        The query point.
    row1_line, col1_line : float
        First point on the line.
    row2_line, col2_line : float
        Second point on the line.

    Returns
    -------
    float
        Unsigned perpendicular distance.
    """
    dr = row2_line - row1_line
    dc = col2_line - col1_line
    length = math.hypot(dr, dc)
    if length < 1e-12:
        return math.hypot(row_point - row1_line, col_point - col1_line)
    return abs(
        dc * (row1_line - row_point) - dr * (col1_line - col_point)
    ) / length


def distance_cc(
    contour1: Union[Sequence[Tuple[float, float]], np.ndarray],
    contour2: Union[Sequence[Tuple[float, float]], np.ndarray],
    mode: str = "point_to_point",
) -> float:
    """Minimum distance between two contours.

    Parameters
    ----------
    contour1 : sequence of (row, col)
        First contour.
    contour2 : sequence of (row, col)
        Second contour.
    mode : str
        ``"point_to_point"`` (exhaustive pairwise minimum).

    Returns
    -------
    float
        Minimum distance between any point pair across the two contours.

    Raises
    ------
    ValueError
        If either contour is empty.
    """
    c1 = np.asarray(contour1, dtype=np.float64)
    c2 = np.asarray(contour2, dtype=np.float64)

    if c1.size == 0 or c2.size == 0:
        raise ValueError("distance_cc: contours must not be empty")

    # Pairwise distances via broadcasting: (N, M, 2)
    diff = c1[:, np.newaxis, :] - c2[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    return float(np.min(dists))


def angle_ll(
    row1_l1: float, col1_l1: float, row2_l1: float, col2_l1: float,
    row1_l2: float, col1_l2: float, row2_l2: float, col2_l2: float,
) -> float:
    """Angle between two lines in radians.

    Parameters
    ----------
    row1_l1, col1_l1, row2_l1, col2_l1 : float
        Endpoints of the first line.
    row1_l2, col1_l2, row2_l2, col2_l2 : float
        Endpoints of the second line.

    Returns
    -------
    float
        Angle between the lines in radians ``[0, pi/2]``.
    """
    dr1 = row2_l1 - row1_l1
    dc1 = col2_l1 - col1_l1
    dr2 = row2_l2 - row1_l2
    dc2 = col2_l2 - col1_l2

    len1 = math.hypot(dr1, dc1)
    len2 = math.hypot(dr2, dc2)
    if len1 < 1e-12 or len2 < 1e-12:
        return 0.0

    cos_angle = (dc1 * dc2 + dr1 * dr2) / (len1 * len2)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.acos(abs(cos_angle))


def angle_lx(
    row1: float, col1: float, row2: float, col2: float,
) -> float:
    """Angle of a line with respect to the horizontal axis.

    Parameters
    ----------
    row1, col1 : float
        First endpoint.
    row2, col2 : float
        Second endpoint.

    Returns
    -------
    float
        Angle in radians ``(-pi, pi]``.
    """
    return math.atan2(row2 - row1, col2 - col1)


# ====================================================================== #
#  6. Drawing helpers                                                     #
# ====================================================================== #


def draw_measure_rect(
    image: np.ndarray,
    rect: MeasureRectangle,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 1,
) -> np.ndarray:
    """Draw a measurement rectangle on the image.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    rect : MeasureRectangle
        The measurement rectangle to draw.
    color : tuple of int
        BGR colour tuple.
    thickness : int
        Line thickness.

    Returns
    -------
    np.ndarray
        Image with the rectangle drawn (BGR, 3-channel).
    """
    validate_image(image)
    canvas = _ensure_bgr(image)

    cos_p = math.cos(rect.phi)
    sin_p = math.sin(rect.phi)

    # Four corners relative to centre
    corners_local = [
        (-rect.length1, -rect.length2),
        (rect.length1, -rect.length2),
        (rect.length1, rect.length2),
        (-rect.length1, rect.length2),
    ]

    pts = []
    for dx, dy in corners_local:
        c = rect.col + dx * cos_p - dy * sin_p
        r = rect.row + dx * sin_p + dy * cos_p
        pts.append((int(round(c)), int(round(r))))

    pts_arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(
        canvas, [pts_arr], isClosed=True, color=color, thickness=thickness,
    )

    # Draw the main axis as a centre line
    ax1_col = int(round(rect.col - rect.length1 * cos_p))
    ax1_row = int(round(rect.row - rect.length1 * sin_p))
    ax2_col = int(round(rect.col + rect.length1 * cos_p))
    ax2_row = int(round(rect.row + rect.length1 * sin_p))
    cv2.line(
        canvas,
        (ax1_col, ax1_row),
        (ax2_col, ax2_row),
        color,
        max(1, thickness - 1),
    )

    return canvas


def draw_edges(
    image: np.ndarray,
    edges: List[SubPixelEdge],
    color: Tuple[int, int, int] = (255, 0, 0),
    radius: int = 3,
) -> np.ndarray:
    """Draw sub-pixel edges as small circles with direction indicators.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    edges : list of SubPixelEdge
        Edges to draw.
    color : tuple of int
        BGR colour tuple.
    radius : int
        Circle radius in pixels.

    Returns
    -------
    np.ndarray
        Image with edges drawn (BGR, 3-channel).
    """
    validate_image(image)
    canvas = _ensure_bgr(image)

    for e in edges:
        centre = (int(round(e.col)), int(round(e.row)))
        cv2.circle(canvas, centre, radius, color, -1)
        # Direction indicator
        end_col = int(round(e.col + radius * 2.0 * math.cos(e.angle)))
        end_row = int(round(e.row + radius * 2.0 * math.sin(e.angle)))
        cv2.line(canvas, centre, (end_col, end_row), color, 1)

    return canvas


def draw_fit_result(
    image: np.ndarray,
    fit: FitResult,
    color: Tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Draw a fitted geometric primitive (line, circle, or ellipse).

    Parameters
    ----------
    image : np.ndarray
        Input image.
    fit : FitResult
        The fit result to visualise.
    color : tuple of int
        BGR colour tuple.
    thickness : int
        Line thickness.

    Returns
    -------
    np.ndarray
        Image with the primitive drawn (BGR, 3-channel).
    """
    validate_image(image)
    canvas = _ensure_bgr(image)
    p = fit.params

    if fit.type == "line":
        pt1 = (int(round(p["col1"])), int(round(p["row1"])))
        pt2 = (int(round(p["col2"])), int(round(p["row2"])))
        cv2.line(canvas, pt1, pt2, color, thickness)

    elif fit.type == "circle":
        centre = (int(round(p["col"])), int(round(p["row"])))
        r = int(round(p["radius"]))
        cv2.circle(canvas, centre, r, color, thickness)

    elif fit.type == "ellipse":
        centre = (int(round(p["col"])), int(round(p["row"])))
        axes = (int(round(p["ra"])), int(round(p["rb"])))
        angle_deg = math.degrees(p["phi"])
        cv2.ellipse(canvas, centre, axes, angle_deg, 0, 360, color, thickness)

    return canvas


def draw_measurement(
    image: np.ndarray,
    measurement: MeasurementResult,
    color: Tuple[int, int, int] = (255, 255, 0),
    thickness: int = 1,
) -> np.ndarray:
    """Draw a measurement annotation (line between points with label).

    Parameters
    ----------
    image : np.ndarray
        Input image.
    measurement : MeasurementResult
        The measurement to visualise.
    color : tuple of int
        BGR colour tuple.
    thickness : int
        Line thickness.

    Returns
    -------
    np.ndarray
        Image with measurement annotation drawn (BGR, 3-channel).
    """
    validate_image(image)
    canvas = _ensure_bgr(image)

    label = f"{measurement.value:.2f} {measurement.unit}"

    if measurement.point1 is not None and measurement.point2 is not None:
        pt1 = (int(round(measurement.point1[1])), int(round(measurement.point1[0])))
        pt2 = (int(round(measurement.point2[1])), int(round(measurement.point2[0])))
        cv2.line(canvas, pt1, pt2, color, thickness)

        # Crosshair markers at endpoints
        for pt in (pt1, pt2):
            cv2.drawMarker(
                canvas, pt, color,
                markerType=cv2.MARKER_CROSS, markerSize=8, thickness=1,
            )

        # Label at midpoint
        mid_x = (pt1[0] + pt2[0]) // 2
        mid_y = (pt1[1] + pt2[1]) // 2
        cv2.putText(
            canvas, label,
            (mid_x + 5, mid_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )

    elif measurement.point1 is not None:
        pt = (int(round(measurement.point1[1])), int(round(measurement.point1[0])))
        cv2.putText(
            canvas, label,
            (pt[0] + 5, pt[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )

    return canvas
