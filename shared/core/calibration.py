"""
core/calibration.py - Camera calibration and pixel-to-world coordinate mapping.

Wraps OpenCV calibration APIs with a HALCON-style interface for industrial
measurement pipelines.  Supports full camera calibration from chessboard or
circle-grid patterns, as well as simplified pixel-to-mm mapping when a full
calibration is not required.

Categories:
    1. Data Classes (CalibrationResult, WorldMapping)
    2. Corner / Pattern Detection
    3. Full Camera Calibration
    4. Simple Pixel-to-mm Mapping
    5. Coordinate Conversion & Measurement
    6. Drawing Helpers
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from shared.validation import validate_image
from shared.op_logger import log_operation

logger = logging.getLogger(__name__)


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


# ====================================================================== #
#  Data Classes                                                           #
# ====================================================================== #


@dataclass
class CalibrationResult:
    """Full camera calibration result.

    Attributes:
        camera_matrix:  3x3 intrinsic camera matrix.
        dist_coeffs:    Distortion coefficients ``(k1, k2, p1, p2, k3)``.
        rvecs:          Rotation vectors for each calibration image.
        tvecs:          Translation vectors for each calibration image.
        rms_error:      Overall reprojection error in pixels.
        image_size:     ``(width, height)`` of the calibration images.
        num_images:     Number of images used for calibration.
        pattern_size:   ``(cols, rows)`` of the calibration pattern.
        square_size:    Physical size of each square in millimetres.
    """

    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    rvecs: List[np.ndarray]
    tvecs: List[np.ndarray]
    rms_error: float
    image_size: Tuple[int, int]
    num_images: int
    pattern_size: Tuple[int, int]
    square_size: float


@dataclass
class WorldMapping:
    """Pixel-to-world coordinate mapping for planar measurements.

    Attributes:
        pixels_per_mm_x:  Horizontal resolution (px/mm).
        pixels_per_mm_y:  Vertical resolution (px/mm).
        pixels_per_mm:    Average resolution (px/mm).
        origin_px:        Origin of the world coordinate system in pixel
                          coordinates ``(x, y)``.
        rotation_deg:     Rotation of the world axes relative to the image
                          axes in degrees.
        method:           How the mapping was obtained: ``"calibration"``,
                          ``"known_distance"``, or ``"known_object"``.
    """

    pixels_per_mm_x: float
    pixels_per_mm_y: float
    pixels_per_mm: float
    origin_px: Tuple[float, float]
    rotation_deg: float
    method: str


# ====================================================================== #
#  Corner / Pattern Detection                                             #
# ====================================================================== #


@log_operation(logger)
def find_chessboard_corners(
    image: np.ndarray,
    pattern_size: Tuple[int, int] = (9, 6),
    flags: Optional[int] = None,
) -> Optional[np.ndarray]:
    """Detect chessboard corners with sub-pixel refinement.

    Parameters
    ----------
    image : np.ndarray
        Input image (grayscale or colour).
    pattern_size : tuple of int
        ``(cols, rows)`` of interior corners in the chessboard pattern.
    flags : int or None
        Optional flags forwarded to :func:`cv2.findChessboardCorners`.
        When *None* the default combination
        ``CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE`` is used.

    Returns
    -------
    np.ndarray or None
        Nx2 array of sub-pixel corner positions, or *None* if the pattern
        was not found.
    """
    validate_image(image)
    gray = _ensure_gray(image)

    if flags is None:
        flags = (
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        )

    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not found or corners is None:
        logger.warning(
            "Chessboard pattern %s not found in image.", pattern_size
        )
        return None

    # Sub-pixel refinement
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # Reshape to Nx2
    corners_2d = corners.reshape(-1, 2)
    logger.info(
        "Found %d chessboard corners for pattern %s.",
        len(corners_2d),
        pattern_size,
    )
    return corners_2d


@log_operation(logger)
def find_circle_grid(
    image: np.ndarray,
    pattern_size: Tuple[int, int] = (4, 11),
    symmetric: bool = False,
) -> Optional[np.ndarray]:
    """Detect circle grid centres.

    Parameters
    ----------
    image : np.ndarray
        Input image (grayscale or colour).
    pattern_size : tuple of int
        ``(cols, rows)`` of circles in the grid.
    symmetric : bool
        If *True* use :data:`cv2.CALIB_CB_SYMMETRIC_GRID`, otherwise
        use :data:`cv2.CALIB_CB_ASYMMETRIC_GRID`.

    Returns
    -------
    np.ndarray or None
        Nx2 array of circle centres, or *None* if the pattern was not found.
    """
    validate_image(image)
    gray = _ensure_gray(image)

    grid_flag = (
        cv2.CALIB_CB_SYMMETRIC_GRID
        if symmetric
        else cv2.CALIB_CB_ASYMMETRIC_GRID
    )

    found, centres = cv2.findCirclesGrid(gray, pattern_size, None, grid_flag)
    if not found or centres is None:
        logger.warning(
            "Circle grid pattern %s not found in image.", pattern_size
        )
        return None

    centres_2d = centres.reshape(-1, 2)
    logger.info(
        "Found %d circle centres for pattern %s.",
        len(centres_2d),
        pattern_size,
    )
    return centres_2d


# ====================================================================== #
#  Full Camera Calibration                                                #
# ====================================================================== #


@log_operation(logger)
def calibrate_camera(
    images_or_corners: Union[List[np.ndarray], List[np.ndarray]],
    pattern_size: Tuple[int, int] = (9, 6),
    square_size: float = 1.0,
) -> CalibrationResult:
    """Perform full camera calibration from chessboard images or corners.

    Parameters
    ----------
    images_or_corners : list
        Either a list of images (corner detection will be run automatically)
        or a list of pre-detected corner arrays (each Nx2).
    pattern_size : tuple of int
        ``(cols, rows)`` of interior corners.
    square_size : float
        Physical size of each square in millimetres.

    Returns
    -------
    CalibrationResult
        Calibration parameters and reprojection error.

    Raises
    ------
    ValueError
        If no valid corner sets could be obtained.
    """
    if len(images_or_corners) == 0:
        raise ValueError("At least one image or corner set is required.")

    # Determine if we received images or pre-detected corners
    first = images_or_corners[0]
    is_image = first.ndim >= 2 and (
        first.ndim == 2 or (first.ndim == 3 and first.shape[2] in (1, 3, 4))
    )

    corner_sets: List[np.ndarray] = []
    image_size: Optional[Tuple[int, int]] = None

    if is_image:
        for idx, img in enumerate(images_or_corners):
            validate_image(img)
            h, w = img.shape[:2]
            image_size = (w, h)
            corners = find_chessboard_corners.__wrapped__(
                img, pattern_size
            ) if hasattr(find_chessboard_corners, '__wrapped__') else _find_corners_raw(img, pattern_size)
            if corners is not None:
                corner_sets.append(corners)
            else:
                logger.warning(
                    "Skipping image %d/%d: pattern not found.",
                    idx + 1,
                    len(images_or_corners),
                )
    else:
        for corners in images_or_corners:
            corner_sets.append(
                corners.reshape(-1, 2)
                if corners.ndim != 2
                else corners
            )
        # Without images we cannot infer image_size; caller should supply
        # a reasonable default or we leave it as (0, 0).
        image_size = (0, 0)

    if len(corner_sets) == 0:
        raise ValueError(
            "No valid corner sets obtained.  Check images and pattern_size."
        )

    # Build 3-D object points
    cols, rows = pattern_size
    objp = np.zeros((cols * rows, 3), dtype=np.float32)
    objp[:, :2] = (
        np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size
    )
    obj_points = [objp] * len(corner_sets)

    # Format corners for OpenCV (Nx1x2 float32)
    img_points = [
        c.reshape(-1, 1, 2).astype(np.float32) for c in corner_sets
    ]

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points,
        img_points,
        image_size,  # image_size is already (w, h) which is what OpenCV expects
        None,
        None,
    )

    logger.info(
        "Calibration complete: RMS reprojection error = %.4f px "
        "(%d images).",
        rms,
        len(corner_sets),
    )

    return CalibrationResult(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        rvecs=list(rvecs),
        tvecs=list(tvecs),
        rms_error=rms,
        image_size=image_size,  # type: ignore[arg-type]
        num_images=len(corner_sets),
        pattern_size=pattern_size,
        square_size=square_size,
    )


def _find_corners_raw(
    image: np.ndarray,
    pattern_size: Tuple[int, int],
) -> Optional[np.ndarray]:
    """Internal corner detection without the logging decorator."""
    gray = _ensure_gray(image)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not found or corners is None:
        return None
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return corners.reshape(-1, 2)


@log_operation(logger)
def undistort_image(
    image: np.ndarray,
    calibration: CalibrationResult,
) -> np.ndarray:
    """Apply lens distortion correction.

    Parameters
    ----------
    image : np.ndarray
        Distorted input image.
    calibration : CalibrationResult
        Previously obtained calibration result.

    Returns
    -------
    np.ndarray
        Undistorted image with the same shape as the input.
    """
    validate_image(image)
    h, w = image.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        calibration.camera_matrix,
        calibration.dist_coeffs,
        (w, h),
        alpha=1,
        newImgSize=(w, h),
    )
    undistorted = cv2.undistort(
        image,
        calibration.camera_matrix,
        calibration.dist_coeffs,
        None,
        new_camera_matrix,
    )
    # Optionally crop to valid ROI
    x, y, rw, rh = roi
    if rw > 0 and rh > 0:
        undistorted = undistorted[y : y + rh, x : x + rw]

    logger.info("Undistorted image to %dx%d.", undistorted.shape[1], undistorted.shape[0])
    return undistorted


# ====================================================================== #
#  Calibration Persistence                                                #
# ====================================================================== #


@log_operation(logger)
def save_calibration(calibration: CalibrationResult, path: Union[str, Path]) -> None:
    """Save a calibration result to a JSON file.

    Parameters
    ----------
    calibration : CalibrationResult
        The calibration to persist.
    path : str or Path
        Destination file path.
    """
    data = {
        "camera_matrix": calibration.camera_matrix.tolist(),
        "dist_coeffs": calibration.dist_coeffs.tolist(),
        "rvecs": [r.tolist() for r in calibration.rvecs],
        "tvecs": [t.tolist() for t in calibration.tvecs],
        "rms_error": calibration.rms_error,
        "image_size": list(calibration.image_size),
        "num_images": calibration.num_images,
        "pattern_size": list(calibration.pattern_size),
        "square_size": calibration.square_size,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    logger.info("Saved calibration to %s.", path)


@log_operation(logger)
def load_calibration(path: Union[str, Path]) -> CalibrationResult:
    """Load a calibration result from a JSON file.

    Parameters
    ----------
    path : str or Path
        Source file path.

    Returns
    -------
    CalibrationResult
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    result = CalibrationResult(
        camera_matrix=np.array(data["camera_matrix"], dtype=np.float64),
        dist_coeffs=np.array(data["dist_coeffs"], dtype=np.float64),
        rvecs=[np.array(r, dtype=np.float64) for r in data["rvecs"]],
        tvecs=[np.array(t, dtype=np.float64) for t in data["tvecs"]],
        rms_error=float(data["rms_error"]),
        image_size=tuple(data["image_size"]),  # type: ignore[arg-type]
        num_images=int(data["num_images"]),
        pattern_size=tuple(data["pattern_size"]),  # type: ignore[arg-type]
        square_size=float(data["square_size"]),
    )
    logger.info("Loaded calibration from %s (RMS=%.4f).", path, result.rms_error)
    return result


# ====================================================================== #
#  Simple Pixel-to-mm Mapping                                             #
# ====================================================================== #


@log_operation(logger)
def calibrate_from_known_distance(
    pixel_distance: float,
    real_distance_mm: float,
    axis: str = "both",
) -> WorldMapping:
    """Create a world mapping from a single known distance measurement.

    Parameters
    ----------
    pixel_distance : float
        Measured distance in pixels.
    real_distance_mm : float
        Corresponding real-world distance in millimetres.
    axis : str
        Which axis the measurement corresponds to: ``"x"``, ``"y"``, or
        ``"both"`` (isotropic).

    Returns
    -------
    WorldMapping
    """
    if pixel_distance <= 0:
        raise ValueError("pixel_distance must be positive.")
    if real_distance_mm <= 0:
        raise ValueError("real_distance_mm must be positive.")
    if axis not in ("x", "y", "both"):
        raise ValueError("axis must be 'x', 'y', or 'both'.")

    ppm = pixel_distance / real_distance_mm

    if axis == "x":
        ppm_x, ppm_y = ppm, ppm  # best guess for y
    elif axis == "y":
        ppm_x, ppm_y = ppm, ppm
    else:
        ppm_x, ppm_y = ppm, ppm

    mapping = WorldMapping(
        pixels_per_mm_x=ppm_x,
        pixels_per_mm_y=ppm_y,
        pixels_per_mm=ppm,
        origin_px=(0.0, 0.0),
        rotation_deg=0.0,
        method="known_distance",
    )
    logger.info(
        "Known-distance mapping: %.4f px/mm (%.4f mm/px).",
        ppm,
        1.0 / ppm,
    )
    return mapping


@log_operation(logger)
def calibrate_from_known_object(
    image: np.ndarray,
    object_width_mm: float,
    object_height_mm: Optional[float] = None,
    roi: Optional[Tuple[int, int, int, int]] = None,
) -> WorldMapping:
    """Create a world mapping by detecting an object of known dimensions.

    The largest contour inside the ROI (or the full image) is assumed to be
    the reference object.

    Parameters
    ----------
    image : np.ndarray
        Input image containing the reference object.
    object_width_mm : float
        Real-world width of the reference object in millimetres.
    object_height_mm : float or None
        Real-world height.  When *None* isotropic scaling is assumed.
    roi : tuple or None
        ``(x, y, w, h)`` region of interest.  When *None* the whole image
        is used.

    Returns
    -------
    WorldMapping
    """
    validate_image(image)
    if object_width_mm <= 0:
        raise ValueError("object_width_mm must be positive.")

    gray = _ensure_gray(image)

    if roi is not None:
        rx, ry, rw, rh = roi
        gray = gray[ry : ry + rh, rx : rx + rw]
    else:
        rx, ry = 0, 0

    # Threshold and find contours
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        raise ValueError("No contours found in the image/ROI.")

    # Pick the largest contour by area
    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    (cx, cy), (w_px, h_px), angle = rect

    # Ensure width >= height for consistent orientation
    if w_px < h_px:
        w_px, h_px = h_px, w_px
        angle += 90.0

    ppm_x = w_px / object_width_mm
    if object_height_mm is not None and object_height_mm > 0:
        ppm_y = h_px / object_height_mm
    else:
        ppm_y = ppm_x

    ppm_avg = (ppm_x + ppm_y) / 2.0

    mapping = WorldMapping(
        pixels_per_mm_x=ppm_x,
        pixels_per_mm_y=ppm_y,
        pixels_per_mm=ppm_avg,
        origin_px=(float(cx + rx), float(cy + ry)),
        rotation_deg=float(angle),
        method="known_object",
    )
    logger.info(
        "Known-object mapping: %.4f x %.4f px/mm (object %.1f x %.1f px).",
        ppm_x,
        ppm_y,
        w_px,
        h_px,
    )
    return mapping


@log_operation(logger)
def calibrate_from_chessboard(
    image: np.ndarray,
    pattern_size: Tuple[int, int],
    square_size_mm: float,
) -> WorldMapping:
    """Create a world mapping from a single chessboard image.

    Detects the chessboard and computes the average pixels-per-mm from the
    known square size.  More accurate than :func:`calibrate_from_known_distance`
    because it averages over many point pairs.

    Parameters
    ----------
    image : np.ndarray
        Image containing a visible chessboard pattern.
    pattern_size : tuple of int
        ``(cols, rows)`` of interior corners.
    square_size_mm : float
        Physical size of each chessboard square in millimetres.

    Returns
    -------
    WorldMapping
    """
    validate_image(image)
    if square_size_mm <= 0:
        raise ValueError("square_size_mm must be positive.")

    corners = _find_corners_raw(image, pattern_size)
    if corners is None:
        raise ValueError(
            "Chessboard pattern %s not found in image." % (pattern_size,)
        )

    cols, rows = pattern_size

    # Compute average horizontal spacing (along columns)
    h_dists: List[float] = []
    for r in range(rows):
        for c in range(cols - 1):
            idx0 = r * cols + c
            idx1 = r * cols + c + 1
            d = float(np.linalg.norm(corners[idx1] - corners[idx0]))
            h_dists.append(d)

    # Compute average vertical spacing (along rows)
    v_dists: List[float] = []
    for r in range(rows - 1):
        for c in range(cols):
            idx0 = r * cols + c
            idx1 = (r + 1) * cols + c
            d = float(np.linalg.norm(corners[idx1] - corners[idx0]))
            v_dists.append(d)

    avg_h = float(np.mean(h_dists)) if h_dists else 1.0
    avg_v = float(np.mean(v_dists)) if v_dists else 1.0

    ppm_x = avg_h / square_size_mm
    ppm_y = avg_v / square_size_mm
    ppm_avg = (ppm_x + ppm_y) / 2.0

    # Estimate rotation from first row of corners
    dx = corners[1][0] - corners[0][0]
    dy = corners[1][1] - corners[0][1]
    rotation_deg = float(math.degrees(math.atan2(dy, dx)))

    # Origin at the first detected corner
    origin = (float(corners[0][0]), float(corners[0][1]))

    mapping = WorldMapping(
        pixels_per_mm_x=ppm_x,
        pixels_per_mm_y=ppm_y,
        pixels_per_mm=ppm_avg,
        origin_px=origin,
        rotation_deg=rotation_deg,
        method="calibration",
    )
    logger.info(
        "Chessboard mapping: %.4f x %.4f px/mm, rotation=%.2f deg.",
        ppm_x,
        ppm_y,
        rotation_deg,
    )
    return mapping


# ====================================================================== #
#  Coordinate Conversion & Measurement                                    #
# ====================================================================== #


@log_operation(logger)
def pixel_to_world(
    px_x: float,
    px_y: float,
    mapping: WorldMapping,
) -> Tuple[float, float]:
    """Convert pixel coordinates to world coordinates (mm).

    Parameters
    ----------
    px_x, px_y : float
        Pixel coordinates.
    mapping : WorldMapping
        Active pixel-to-world mapping.

    Returns
    -------
    tuple of float
        ``(world_x_mm, world_y_mm)``
    """
    dx = px_x - mapping.origin_px[0]
    dy = px_y - mapping.origin_px[1]

    theta = math.radians(-mapping.rotation_deg)
    rx = dx * math.cos(theta) - dy * math.sin(theta)
    ry = dx * math.sin(theta) + dy * math.cos(theta)

    world_x = rx / mapping.pixels_per_mm_x
    world_y = ry / mapping.pixels_per_mm_y
    return (world_x, world_y)


@log_operation(logger)
def world_to_pixel(
    world_x: float,
    world_y: float,
    mapping: WorldMapping,
) -> Tuple[float, float]:
    """Convert world coordinates (mm) to pixel coordinates.

    Parameters
    ----------
    world_x, world_y : float
        World coordinates in millimetres.
    mapping : WorldMapping
        Active pixel-to-world mapping.

    Returns
    -------
    tuple of float
        ``(px_x, px_y)``
    """
    rx = world_x * mapping.pixels_per_mm_x
    ry = world_y * mapping.pixels_per_mm_y

    theta = math.radians(mapping.rotation_deg)
    dx = rx * math.cos(theta) - ry * math.sin(theta)
    dy = rx * math.sin(theta) + ry * math.cos(theta)

    px_x = dx + mapping.origin_px[0]
    px_y = dy + mapping.origin_px[1]
    return (px_x, px_y)


@log_operation(logger)
def measure_distance_mm(
    px1: float,
    py1: float,
    px2: float,
    py2: float,
    mapping: WorldMapping,
) -> float:
    """Compute the real-world distance between two pixel points.

    Parameters
    ----------
    px1, py1 : float
        First point in pixel coordinates.
    px2, py2 : float
        Second point in pixel coordinates.
    mapping : WorldMapping
        Active pixel-to-world mapping.

    Returns
    -------
    float
        Distance in millimetres.
    """
    w1 = pixel_to_world.__wrapped__(px1, py1, mapping) if hasattr(pixel_to_world, '__wrapped__') else pixel_to_world(px1, py1, mapping)
    w2 = pixel_to_world.__wrapped__(px2, py2, mapping) if hasattr(pixel_to_world, '__wrapped__') else pixel_to_world(px2, py2, mapping)
    dist = math.sqrt((w2[0] - w1[0]) ** 2 + (w2[1] - w1[1]) ** 2)
    return dist


@log_operation(logger)
def measure_area_mm2(
    contour_or_mask: np.ndarray,
    mapping: WorldMapping,
) -> float:
    """Compute the real-world area from a contour or binary mask.

    Parameters
    ----------
    contour_or_mask : np.ndarray
        Either a contour (Nx1x2 or Nx2) or a binary mask image.
    mapping : WorldMapping
        Active pixel-to-world mapping.

    Returns
    -------
    float
        Area in square millimetres.
    """
    if contour_or_mask.ndim == 2 and contour_or_mask.shape[1] != 2:
        # Binary mask
        area_px = float(cv2.countNonZero(contour_or_mask))
    else:
        # Contour
        contour = contour_or_mask
        if contour.ndim == 2 and contour.shape[1] == 2:
            contour = contour.reshape(-1, 1, 2)
        area_px = float(cv2.contourArea(contour))

    area_mm2 = area_px / (mapping.pixels_per_mm_x * mapping.pixels_per_mm_y)
    return area_mm2


@log_operation(logger)
def measure_length_mm(
    contour: np.ndarray,
    mapping: WorldMapping,
) -> float:
    """Compute the real-world arc length of a contour.

    Parameters
    ----------
    contour : np.ndarray
        Contour array (Nx1x2 or Nx2).
    mapping : WorldMapping
        Active pixel-to-world mapping.

    Returns
    -------
    float
        Arc length in millimetres.
    """
    if contour.ndim == 2 and contour.shape[1] == 2:
        contour = contour.reshape(-1, 1, 2)

    length_px = float(cv2.arcLength(contour, closed=False))
    length_mm = length_px / mapping.pixels_per_mm
    return length_mm


# ====================================================================== #
#  Drawing Helpers                                                        #
# ====================================================================== #


@log_operation(logger)
def draw_calibration_corners(
    image: np.ndarray,
    corners: np.ndarray,
    pattern_size: Tuple[int, int],
    found: bool = True,
) -> np.ndarray:
    """Draw detected calibration corners on an image.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    corners : np.ndarray
        Corner positions (Nx2 or Nx1x2).
    pattern_size : tuple of int
        ``(cols, rows)`` of the pattern.
    found : bool
        Whether the full pattern was successfully found.

    Returns
    -------
    np.ndarray
        Image with corners drawn (BGR).
    """
    validate_image(image)
    vis = _ensure_bgr(image)

    # OpenCV expects Nx1x2 float32
    c = corners.reshape(-1, 1, 2).astype(np.float32)
    cv2.drawChessboardCorners(vis, pattern_size, c, found)
    return vis


@log_operation(logger)
def draw_world_grid(
    image: np.ndarray,
    mapping: WorldMapping,
    grid_spacing_mm: float = 10.0,
    color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Overlay a world-coordinate grid on an image.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    mapping : WorldMapping
        Active pixel-to-world mapping.
    grid_spacing_mm : float
        Spacing between grid lines in millimetres.
    color : tuple of int
        BGR colour for grid lines.

    Returns
    -------
    np.ndarray
        Image with grid overlay (BGR).
    """
    validate_image(image)
    vis = _ensure_bgr(image)
    h, w = vis.shape[:2]

    if grid_spacing_mm <= 0:
        raise ValueError("grid_spacing_mm must be positive.")

    # Determine the world extent of the image
    corners_world = [
        pixel_to_world.__wrapped__(0, 0, mapping) if hasattr(pixel_to_world, '__wrapped__') else pixel_to_world(0, 0, mapping),
        pixel_to_world.__wrapped__(w, 0, mapping) if hasattr(pixel_to_world, '__wrapped__') else pixel_to_world(w, 0, mapping),
        pixel_to_world.__wrapped__(0, h, mapping) if hasattr(pixel_to_world, '__wrapped__') else pixel_to_world(0, h, mapping),
        pixel_to_world.__wrapped__(w, h, mapping) if hasattr(pixel_to_world, '__wrapped__') else pixel_to_world(w, h, mapping),
    ]
    xs = [c[0] for c in corners_world]
    ys = [c[1] for c in corners_world]

    x_min = math.floor(min(xs) / grid_spacing_mm) * grid_spacing_mm
    x_max = math.ceil(max(xs) / grid_spacing_mm) * grid_spacing_mm
    y_min = math.floor(min(ys) / grid_spacing_mm) * grid_spacing_mm
    y_max = math.ceil(max(ys) / grid_spacing_mm) * grid_spacing_mm

    _w2p = pixel_to_world.__wrapped__ if hasattr(pixel_to_world, '__wrapped__') else pixel_to_world
    _wp = world_to_pixel.__wrapped__ if hasattr(world_to_pixel, '__wrapped__') else world_to_pixel

    # Draw vertical grid lines
    gx = x_min
    while gx <= x_max:
        pt_top = _wp(gx, y_min, mapping)
        pt_bot = _wp(gx, y_max, mapping)
        p1 = (int(round(pt_top[0])), int(round(pt_top[1])))
        p2 = (int(round(pt_bot[0])), int(round(pt_bot[1])))
        cv2.line(vis, p1, p2, color, 1, cv2.LINE_AA)
        # Label
        if 0 <= p1[0] < w and 0 <= p1[1] < h:
            cv2.putText(
                vis,
                f"{gx:.0f}",
                (p1[0] + 2, p1[1] + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                color,
                1,
                cv2.LINE_AA,
            )
        gx += grid_spacing_mm

    # Draw horizontal grid lines
    gy = y_min
    while gy <= y_max:
        pt_left = _wp(x_min, gy, mapping)
        pt_right = _wp(x_max, gy, mapping)
        p1 = (int(round(pt_left[0])), int(round(pt_left[1])))
        p2 = (int(round(pt_right[0])), int(round(pt_right[1])))
        cv2.line(vis, p1, p2, color, 1, cv2.LINE_AA)
        if 0 <= p1[0] < w and 0 <= p1[1] < h:
            cv2.putText(
                vis,
                f"{gy:.0f}",
                (p1[0] + 2, p1[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                color,
                1,
                cv2.LINE_AA,
            )
        gy += grid_spacing_mm

    return vis


@log_operation(logger)
def draw_ruler(
    image: np.ndarray,
    mapping: WorldMapping,
    position: str = "bottom",
    length_mm: Optional[float] = None,
) -> np.ndarray:
    """Draw a scale ruler/bar on the image.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    mapping : WorldMapping
        Active pixel-to-world mapping.
    position : str
        Where to draw the ruler: ``"bottom"``, ``"top"``, ``"left"``, or
        ``"right"``.
    length_mm : float or None
        Length of the ruler in millimetres.  When *None* an appropriate
        length is chosen automatically.

    Returns
    -------
    np.ndarray
        Image with ruler overlay (BGR).
    """
    validate_image(image)
    vis = _ensure_bgr(image)
    h, w = vis.shape[:2]

    # Auto-select ruler length (~20% of image width)
    if length_mm is None:
        image_width_mm = w / mapping.pixels_per_mm_x
        # Round to a nice number
        raw = image_width_mm * 0.2
        magnitude = 10 ** math.floor(math.log10(max(raw, 1e-6)))
        length_mm = round(raw / magnitude) * magnitude
        if length_mm <= 0:
            length_mm = 1.0

    ruler_px = length_mm * mapping.pixels_per_mm

    margin = 20
    bar_thickness = 6
    tick_height = 10

    if position in ("bottom", "top"):
        x_start = margin
        x_end = int(round(x_start + ruler_px))
        if position == "bottom":
            y_pos = h - margin
        else:
            y_pos = margin + tick_height + 15

        # Main bar
        cv2.rectangle(
            vis,
            (x_start, y_pos - bar_thickness // 2),
            (x_end, y_pos + bar_thickness // 2),
            (255, 255, 255),
            -1,
        )
        cv2.rectangle(
            vis,
            (x_start, y_pos - bar_thickness // 2),
            (x_end, y_pos + bar_thickness // 2),
            (0, 0, 0),
            1,
        )

        # End ticks
        for x in (x_start, x_end):
            cv2.line(
                vis,
                (x, y_pos - tick_height),
                (x, y_pos + tick_height),
                (0, 0, 0),
                1,
            )

        # Label
        label = f"{length_mm:.4g} mm"
        text_x = x_start + (x_end - x_start) // 2 - len(label) * 3
        text_y = y_pos - tick_height - 4
        cv2.putText(
            vis,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    elif position in ("left", "right"):
        y_start = margin
        y_end = int(round(y_start + ruler_px))
        if position == "left":
            x_pos = margin + 15
        else:
            x_pos = w - margin - 15

        cv2.rectangle(
            vis,
            (x_pos - bar_thickness // 2, y_start),
            (x_pos + bar_thickness // 2, y_end),
            (255, 255, 255),
            -1,
        )
        cv2.rectangle(
            vis,
            (x_pos - bar_thickness // 2, y_start),
            (x_pos + bar_thickness // 2, y_end),
            (0, 0, 0),
            1,
        )

        for y in (y_start, y_end):
            cv2.line(
                vis,
                (x_pos - tick_height, y),
                (x_pos + tick_height, y),
                (0, 0, 0),
                1,
            )

        label = f"{length_mm:.4g} mm"
        cv2.putText(
            vis,
            label,
            (x_pos + tick_height + 4, (y_start + y_end) // 2 + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            label,
            (x_pos + tick_height + 4, (y_start + y_end) // 2 + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return vis


@log_operation(logger)
def draw_measurement_annotation(
    image: np.ndarray,
    pt1_px: Tuple[float, float],
    pt2_px: Tuple[float, float],
    mapping: WorldMapping,
    color: Tuple[int, int, int] = (0, 255, 255),
) -> np.ndarray:
    """Draw a dimensioning line between two points with a mm label.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    pt1_px, pt2_px : tuple of float
        Endpoints in pixel coordinates ``(x, y)``.
    mapping : WorldMapping
        Active pixel-to-world mapping.
    color : tuple of int
        BGR colour for the annotation.

    Returns
    -------
    np.ndarray
        Annotated image (BGR).
    """
    validate_image(image)
    vis = _ensure_bgr(image)

    p1 = (int(round(pt1_px[0])), int(round(pt1_px[1])))
    p2 = (int(round(pt2_px[0])), int(round(pt2_px[1])))

    # Main measurement line
    cv2.line(vis, p1, p2, color, 1, cv2.LINE_AA)

    # Endpoint markers
    marker_size = 5
    cv2.drawMarker(
        vis, p1, color, cv2.MARKER_CROSS, marker_size, 1, cv2.LINE_AA
    )
    cv2.drawMarker(
        vis, p2, color, cv2.MARKER_CROSS, marker_size, 1, cv2.LINE_AA
    )

    # Compute distance
    dist_mm = measure_distance_mm.__wrapped__(
        pt1_px[0], pt1_px[1], pt2_px[0], pt2_px[1], mapping
    ) if hasattr(measure_distance_mm, '__wrapped__') else _measure_dist_raw(
        pt1_px[0], pt1_px[1], pt2_px[0], pt2_px[1], mapping
    )

    # Place label at the midpoint, offset slightly
    mid_x = (p1[0] + p2[0]) // 2
    mid_y = (p1[1] + p2[1]) // 2
    label = f"{dist_mm:.2f} mm"

    # Background for readability
    (tw, th), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
    )
    cv2.rectangle(
        vis,
        (mid_x - 2, mid_y - th - 4),
        (mid_x + tw + 2, mid_y + 2),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        vis,
        label,
        (mid_x, mid_y - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )

    return vis


def _measure_dist_raw(
    px1: float,
    py1: float,
    px2: float,
    py2: float,
    mapping: WorldMapping,
) -> float:
    """Internal distance computation without decorator overhead."""
    dx = px1 - mapping.origin_px[0]
    dy = py1 - mapping.origin_px[1]
    theta = math.radians(-mapping.rotation_deg)
    rx1 = dx * math.cos(theta) - dy * math.sin(theta)
    ry1 = dx * math.sin(theta) + dy * math.cos(theta)
    w1x = rx1 / mapping.pixels_per_mm_x
    w1y = ry1 / mapping.pixels_per_mm_y

    dx = px2 - mapping.origin_px[0]
    dy = py2 - mapping.origin_px[1]
    rx2 = dx * math.cos(theta) - dy * math.sin(theta)
    ry2 = dx * math.sin(theta) + dy * math.cos(theta)
    w2x = rx2 / mapping.pixels_per_mm_x
    w2y = ry2 / mapping.pixels_per_mm_y

    return math.sqrt((w2x - w1x) ** 2 + (w2y - w1y) ** 2)
