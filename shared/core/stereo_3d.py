"""
core/stereo_3d.py - Stereo Vision, Structured Light, and 3D Reconstruction.

Provides comprehensive 3D reconstruction and advanced matching operators for
industrial inspection pipelines.  All algorithms are implemented using NumPy,
OpenCV, and SciPy.

Categories:
    1. Binocular Stereo Vision (disparity, depth, rectification)
    2. Structured Light (Gray code, sinusoidal phase-shift, reconstruction)
    3. Sheet-of-Light / Laser Triangulation (line extraction, height profiling)
    4. Auto-Focus Quality (focus metrics, best-focus selection)
    5. Descriptor-Based Matching (ORB / SIFT / AKAZE + homography)
    6. Anisotropic Shape Model (independent X/Y scale matching)
    7. Component-Based Matching (multi-part with geometric constraints)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from scipy import ndimage, signal

from shared.op_logger import log_operation
from shared.validation import validate_image, validate_positive

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


# ====================================================================== #
#  1. Binocular Stereo Vision                                             #
# ====================================================================== #


@dataclass
class StereoConfig:
    """Configuration for binocular stereo disparity computation.

    Attributes:
        baseline:        Distance between two camera centres (mm).
        focal_length:    Camera focal length in pixels.
        min_disparity:   Minimum disparity value (typically 0).
        max_disparity:   Maximum disparity (must be divisible by 16).
        block_size:      SAD window size (odd, >= 3).
        use_sgbm:        Use StereoSGBM (True) or StereoBM (False).
        uniqueness_ratio: Margin by which the best cost must beat second-best (%).
        speckle_window:  Max size of smooth disparity regions for speckle filter.
        speckle_range:   Max disparity variation within a connected component.
        p1:              SGBM penalty for disparity changes of +/- 1.
        p2:              SGBM penalty for larger disparity changes.
    """

    baseline: float = 60.0
    focal_length: float = 700.0
    min_disparity: int = 0
    max_disparity: int = 128
    block_size: int = 5
    use_sgbm: bool = True
    uniqueness_ratio: int = 10
    speckle_window: int = 100
    speckle_range: int = 2
    p1: Optional[int] = None
    p2: Optional[int] = None


@dataclass
class StereoResult:
    """Output from stereo disparity / depth computation.

    Attributes:
        disparity_map:  (H, W) float32 disparity in pixels.
        depth_map:      (H, W) float32 metric depth in same units as baseline.
        point_cloud:    (H, W, 3) float32 XYZ coordinates (optional).
        confidence_map: (H, W) float32 in [0, 1] confidence per pixel.
    """

    disparity_map: np.ndarray
    depth_map: np.ndarray
    point_cloud: Optional[np.ndarray] = None
    confidence_map: Optional[np.ndarray] = None


@log_operation(logger)
def compute_stereo_disparity(
    left: np.ndarray,
    right: np.ndarray,
    config: Optional[StereoConfig] = None,
) -> np.ndarray:
    """Compute disparity map from a rectified stereo pair.

    Args:
        left:   Left rectified image (grayscale or BGR).
        right:  Right rectified image (same size/type as *left*).
        config: Stereo matching parameters.  Defaults to ``StereoConfig()``.

    Returns:
        (H, W) float32 disparity map in pixels.
    """
    validate_image(left, "left")
    validate_image(right, "right")
    cfg = config or StereoConfig()

    left_g = _ensure_gray(left)
    right_g = _ensure_gray(right)

    num_disp = cfg.max_disparity - cfg.min_disparity
    # OpenCV requires num_disparities divisible by 16
    num_disp = max(16, ((num_disp + 15) // 16) * 16)

    if cfg.use_sgbm:
        p1 = cfg.p1 or 8 * cfg.block_size * cfg.block_size
        p2 = cfg.p2 or 32 * cfg.block_size * cfg.block_size
        matcher = cv2.StereoSGBM_create(
            minDisparity=cfg.min_disparity,
            numDisparities=num_disp,
            blockSize=cfg.block_size,
            P1=p1,
            P2=p2,
            uniquenessRatio=cfg.uniqueness_ratio,
            speckleWindowSize=cfg.speckle_window,
            speckleRange=cfg.speckle_range,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
    else:
        matcher = cv2.StereoBM_create(
            numDisparities=num_disp,
            blockSize=cfg.block_size,
        )
        matcher.setUniquenessRatio(cfg.uniqueness_ratio)
        matcher.setSpeckleWindowSize(cfg.speckle_window)
        matcher.setSpeckleRange(cfg.speckle_range)

    raw = matcher.compute(left_g, right_g)
    # OpenCV returns disparity * 16 as int16
    disparity = raw.astype(np.float32) / 16.0
    logger.debug("Disparity range: [%.1f, %.1f]", disparity.min(), disparity.max())
    return disparity


@log_operation(logger)
def disparity_to_depth(
    disparity: np.ndarray,
    baseline: float,
    focal_length: float,
    *,
    invalid_depth: float = 0.0,
) -> np.ndarray:
    """Convert disparity map to metric depth.

    depth = (baseline * focal_length) / disparity

    Args:
        disparity:      (H, W) float32 disparity in pixels.
        baseline:       Distance between cameras (same units as desired depth).
        focal_length:   Focal length in pixels.
        invalid_depth:  Value assigned where disparity <= 0.

    Returns:
        (H, W) float32 depth map.
    """
    validate_positive(baseline, "baseline")
    validate_positive(focal_length, "focal_length")

    valid = disparity > 0
    depth = np.full_like(disparity, invalid_depth, dtype=np.float32)
    depth[valid] = (baseline * focal_length) / disparity[valid]
    return depth


@log_operation(logger)
def stereo_rectify(
    left: np.ndarray,
    right: np.ndarray,
    camera_matrix_left: np.ndarray,
    dist_coeffs_left: np.ndarray,
    camera_matrix_right: np.ndarray,
    dist_coeffs_right: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rectify a stereo image pair so that epipolar lines become horizontal.

    Args:
        left / right:            Input images (same size).
        camera_matrix_left/right: 3x3 intrinsic matrices.
        dist_coeffs_left/right:  Distortion coefficient vectors.
        R:                       3x3 rotation between cameras.
        T:                       3x1 translation between cameras.

    Returns:
        Tuple of (rectified_left, rectified_right, Q) where Q is the 4x4
        disparity-to-depth mapping matrix.
    """
    validate_image(left, "left")
    validate_image(right, "right")
    h, w = left.shape[:2]

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        camera_matrix_left, dist_coeffs_left,
        camera_matrix_right, dist_coeffs_right,
        (w, h), R, T,
        alpha=0,
    )

    map1l, map2l = cv2.initUndistortRectifyMap(
        camera_matrix_left, dist_coeffs_left, R1, P1, (w, h), cv2.CV_32FC1,
    )
    map1r, map2r = cv2.initUndistortRectifyMap(
        camera_matrix_right, dist_coeffs_right, R2, P2, (w, h), cv2.CV_32FC1,
    )

    rect_left = cv2.remap(left, map1l, map2l, cv2.INTER_LINEAR)
    rect_right = cv2.remap(right, map1r, map2r, cv2.INTER_LINEAR)

    logger.info("Stereo rectification complete (%dx%d).", w, h)
    return rect_left, rect_right, Q


# ====================================================================== #
#  2. Structured Light                                                    #
# ====================================================================== #


class StructuredLightMethod(str, Enum):
    """Enumeration of supported structured-light coding methods."""
    GRAY_CODE = "gray_code"
    SINUSOIDAL_3STEP = "sinusoidal_3step"
    SINUSOIDAL_4STEP = "sinusoidal_4step"


@dataclass
class StructuredLightConfig:
    """Configuration for structured-light pattern generation and decoding.

    Attributes:
        num_patterns: Number of bit-planes for Gray code, or frequency count.
        width:        Projector horizontal resolution.
        height:       Projector vertical resolution.
        method:       Coding method to use.
    """

    num_patterns: int = 10
    width: int = 1024
    height: int = 768
    method: StructuredLightMethod = StructuredLightMethod.GRAY_CODE


@log_operation(logger)
def generate_gray_code_patterns(
    config: Optional[StructuredLightConfig] = None,
) -> List[np.ndarray]:
    """Generate a sequence of Gray-code stripe patterns.

    Each pattern is a (height, width) uint8 image with values 0 or 255.

    Args:
        config: Structured-light configuration.

    Returns:
        List of *num_patterns* Gray-code pattern images.
    """
    cfg = config or StructuredLightConfig()
    patterns: List[np.ndarray] = []

    cols = np.arange(cfg.width, dtype=np.uint32)
    for bit in range(cfg.num_patterns):
        period = cfg.width / (2 ** (bit + 1))
        binary_vals = (cols / period).astype(np.uint32)
        gray_vals = binary_vals ^ (binary_vals >> 1)
        mask = ((gray_vals & 1) * 255).astype(np.uint8)
        pattern = np.zeros((cfg.height, cfg.width), dtype=np.uint8)
        pattern[:] = mask[np.newaxis, :]
        patterns.append(pattern)

    logger.info("Generated %d Gray-code patterns (%dx%d).",
                len(patterns), cfg.width, cfg.height)
    return patterns


@log_operation(logger)
def generate_sinusoidal_patterns(
    config: Optional[StructuredLightConfig] = None,
    num_frequencies: int = 3,
) -> List[np.ndarray]:
    """Generate phase-shifted sinusoidal fringe patterns.

    For a 3-step method, each frequency produces 3 patterns shifted by
    0, 2pi/3, 4pi/3.  For 4-step, shifts are 0, pi/2, pi, 3pi/2.

    Args:
        config:           Configuration with width, height, method fields.
        num_frequencies:  Number of frequency octaves to generate.

    Returns:
        List of uint8 pattern images.
    """
    cfg = config or StructuredLightConfig()
    steps = 4 if cfg.method == StructuredLightMethod.SINUSOIDAL_4STEP else 3
    patterns: List[np.ndarray] = []

    x = np.arange(cfg.width, dtype=np.float64)

    for freq_idx in range(num_frequencies):
        period = cfg.width / (2 ** freq_idx) if freq_idx > 0 else cfg.width
        for step in range(steps):
            phase_shift = 2.0 * np.pi * step / steps
            row = 127.5 + 127.5 * np.cos(2.0 * np.pi * x / period + phase_shift)
            pattern = np.tile(row.astype(np.uint8), (cfg.height, 1))
            patterns.append(pattern)

    logger.info("Generated %d sinusoidal patterns (%d freq x %d steps).",
                len(patterns), num_frequencies, steps)
    return patterns


@log_operation(logger)
def decode_gray_code(
    captures: List[np.ndarray],
    threshold: int = 30,
) -> np.ndarray:
    """Decode Gray-code captured images into a column index map.

    Args:
        captures:   List of captured images matching the Gray-code sequence.
                    Each must be grayscale or will be converted.
        threshold:  Intensity threshold to distinguish 0/1 stripes.

    Returns:
        (H, W) int32 array of decoded column indices.  Pixels that could
        not be decoded reliably are set to -1.
    """
    if not captures:
        raise ValueError("At least one captured image is required.")
    validate_image(captures[0], "captures[0]")

    gray_caps = [_ensure_gray(c) for c in captures]
    h, w = gray_caps[0].shape
    num_bits = len(gray_caps)

    # Decode Gray code bit by bit
    gray_code = np.zeros((h, w), dtype=np.int32)
    valid = np.ones((h, w), dtype=bool)

    for bit_idx, cap in enumerate(gray_caps):
        bright = cap.astype(np.float32)
        bit_mask = (bright > threshold).astype(np.int32)
        gray_code |= bit_mask << (num_bits - 1 - bit_idx)
        # Mark low-contrast pixels as invalid
        valid &= np.abs(bright - 127.5) > (threshold / 2.0)

    # Gray to binary conversion
    binary_code = gray_code.copy()
    shift = 1
    while shift < num_bits:
        binary_code ^= (binary_code >> shift)
        shift <<= 1

    binary_code[~valid] = -1
    logger.info("Gray-code decoded: %d valid pixels.", int(np.sum(valid)))
    return binary_code


@log_operation(logger)
def decode_phase_shift(
    captures: List[np.ndarray],
    steps: int = 3,
) -> np.ndarray:
    """Decode wrapped phase from phase-shifted sinusoidal captures.

    Args:
        captures: List of captured images (length must equal *steps*).
        steps:    Number of phase-shift steps (3 or 4).

    Returns:
        (H, W) float64 wrapped phase in [-pi, pi].
    """
    if len(captures) != steps:
        raise ValueError(f"Expected {steps} captures, got {len(captures)}.")

    imgs = [_ensure_gray(c).astype(np.float64) for c in captures]

    if steps == 3:
        # 3-step: shifts at 0, 2pi/3, 4pi/3
        numerator = np.sqrt(3.0) * (imgs[0] - imgs[2])
        denominator = 2.0 * imgs[1] - imgs[0] - imgs[2]
        phase = np.arctan2(numerator, denominator)
    elif steps == 4:
        # 4-step: shifts at 0, pi/2, pi, 3pi/2
        phase = np.arctan2(imgs[3] - imgs[1], imgs[0] - imgs[2])
    else:
        raise ValueError(f"Unsupported step count: {steps}")

    return phase


@log_operation(logger)
def unwrap_phase(
    wrapped: np.ndarray,
    reference: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Temporal phase unwrapping using a coarse-to-fine reference.

    If *reference* is provided (an already-unwrapped lower-frequency phase),
    the unwrapping uses the reference to resolve 2*pi ambiguities.  Otherwise
    a simple row-wise cumulative unwrap is applied.

    Args:
        wrapped:    (H, W) float64 wrapped phase in [-pi, pi].
        reference:  Optional (H, W) float64 unwrapped coarser-frequency phase.

    Returns:
        (H, W) float64 unwrapped phase.
    """
    if reference is not None:
        # Temporal unwrapping: find integer k such that
        #   unwrapped = wrapped + 2*pi*k  is closest to reference
        k = np.round((reference - wrapped) / (2.0 * np.pi))
        return wrapped + 2.0 * np.pi * k

    # Fallback: row-wise spatial unwrap
    unwrapped = np.zeros_like(wrapped)
    for row in range(wrapped.shape[0]):
        unwrapped[row, :] = np.unwrap(wrapped[row, :])
    return unwrapped


@log_operation(logger)
def reconstruct_structured_light(
    column_map: np.ndarray,
    camera_matrix: np.ndarray,
    projector_matrix: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
) -> np.ndarray:
    """Triangulate 3D points from decoded column/phase mapping and calibration.

    For each valid pixel (row, col) in the camera image, the corresponding
    projector column is given by *column_map*.  A ray-plane intersection
    (triangulation) yields the 3D point.

    Args:
        column_map:       (H, W) float64 decoded projector column coordinate.
        camera_matrix:    3x3 camera intrinsic matrix.
        projector_matrix: 3x3 projector intrinsic matrix.
        R:                3x3 rotation from camera to projector frame.
        T:                3x1 translation from camera to projector frame.

    Returns:
        (N, 3) float64 point cloud of reconstructed 3D points.
    """
    h, w = column_map.shape[:2]
    valid_mask = column_map >= 0
    if np.issubdtype(column_map.dtype, np.floating):
        valid_mask = np.isfinite(column_map) & (column_map >= 0)

    cam_rows, cam_cols = np.where(valid_mask)
    proj_cols = column_map[cam_rows, cam_cols].astype(np.float64)

    # Camera ray for each pixel
    fx_c, fy_c = camera_matrix[0, 0], camera_matrix[1, 1]
    cx_c, cy_c = camera_matrix[0, 2], camera_matrix[1, 2]
    fx_p = projector_matrix[0, 0]
    cx_p = projector_matrix[0, 2]

    # Camera rays: direction in camera frame
    x_c = (cam_cols.astype(np.float64) - cx_c) / fx_c
    y_c = (cam_rows.astype(np.float64) - cy_c) / fy_c
    cam_dirs = np.stack([x_c, y_c, np.ones_like(x_c)], axis=-1)  # (N, 3)

    # Projector plane equation: for each projector column, the plane normal
    # in projector frame is [1, 0, -(proj_col - cx_p)/fx_p]
    n_proj = np.stack([
        np.ones_like(proj_cols),
        np.zeros_like(proj_cols),
        -(proj_cols - cx_p) / fx_p,
    ], axis=-1)  # (N, 3)

    # Transform plane normal to camera frame: n_cam = R^T @ n_proj
    R_inv = R.T
    T_vec = T.flatten()
    n_cam = (R_inv @ n_proj.T).T  # (N, 3)

    # Plane offset in camera frame: d = n_proj . T
    d = np.sum(n_proj * T_vec[np.newaxis, :], axis=-1)  # (N,)

    # Ray-plane intersection: t = d / (n_cam . cam_dir)
    denom = np.sum(n_cam * cam_dirs, axis=-1)
    valid_denom = np.abs(denom) > 1e-8
    t = np.zeros_like(d)
    t[valid_denom] = d[valid_denom] / denom[valid_denom]

    points = cam_dirs * t[:, np.newaxis]
    points[~valid_denom] = 0.0

    logger.info("Structured-light reconstruction: %d points.", int(np.sum(valid_denom)))
    return points[valid_denom]


# ====================================================================== #
#  3. Sheet-of-Light / Laser Triangulation                                #
# ====================================================================== #


@dataclass
class SheetOfLightConfig:
    """Configuration for laser sheet-of-light triangulation.

    Attributes:
        laser_thickness:  Expected laser line width in pixels.
        angle:            Angle between camera optical axis and laser plane (rad).
        pixel_size:       Camera pixel size in mm/pixel.
        baseline:         Distance from camera to laser origin in mm.
        roi_top:          Top row of the ROI to search for the laser line.
        roi_bottom:       Bottom row of the ROI (0 = full height).
        intensity_thresh: Minimum intensity to consider as laser candidate.
    """

    laser_thickness: float = 10.0
    angle: float = math.radians(45.0)
    pixel_size: float = 0.01
    baseline: float = 100.0
    roi_top: int = 0
    roi_bottom: int = 0
    intensity_thresh: int = 50


@log_operation(logger)
def extract_laser_line(
    image: np.ndarray,
    config: Optional[SheetOfLightConfig] = None,
    *,
    method: str = "gaussian",
) -> np.ndarray:
    """Extract sub-pixel laser line centroid for each column.

    For each column, the row position of the laser line peak is found with
    sub-pixel accuracy using a Gaussian fit or centre-of-gravity method.

    Args:
        image:  Grayscale image containing a laser stripe.
        config: Sheet-of-light configuration.
        method: ``"gaussian"`` for Gaussian peak fit, ``"cog"`` for
                centre-of-gravity.

    Returns:
        (W,) float64 array of sub-pixel row positions.  Columns where no
        laser line was detected contain ``np.nan``.
    """
    validate_image(image, "image")
    cfg = config or SheetOfLightConfig()
    gray = _ensure_gray(image).astype(np.float64)
    h, w = gray.shape

    top = cfg.roi_top
    bot = cfg.roi_bottom if cfg.roi_bottom > 0 else h
    roi = gray[top:bot, :]

    centroids = np.full(w, np.nan, dtype=np.float64)

    # Vectorized: find peak rows and valid columns in bulk
    peak_rows = np.argmax(roi, axis=0)          # (W,)
    peak_vals = roi[peak_rows, np.arange(w)]    # (W,)
    valid_cols = peak_vals >= cfg.intensity_thresh

    if method == "cog":
        # Centre-of-gravity: fully vectorized
        rows_arr = np.arange(roi.shape[0], dtype=np.float64)[:, np.newaxis]  # (H_roi, 1)
        weights = np.maximum(roi - cfg.intensity_thresh, 0)   # (H_roi, W)
        totals = weights.sum(axis=0)                          # (W,)
        cog_valid = valid_cols & (totals > 0)
        centroids[cog_valid] = (
            (rows_arr * weights).sum(axis=0)[cog_valid] / totals[cog_valid] + top
        )
    else:
        # Gaussian: per-column parabola fit on log-intensity near peak
        valid_idx = np.where(valid_cols)[0]
        roi_h = roi.shape[0]
        for col in valid_idx:
            peak_row = int(peak_rows[col])
            lo = max(0, peak_row - 2)
            hi = min(roi_h, peak_row + 3)
            segment = roi[lo:hi, col].copy()
            if segment.min() <= 0:
                segment = np.maximum(segment, 1.0)
            log_seg = np.log(segment)
            rows_local = np.arange(lo, lo + len(segment), dtype=np.float64)
            if len(rows_local) >= 3:
                coeffs = np.polyfit(rows_local, log_seg, 2)
                if coeffs[0] < 0:
                    centroid = -coeffs[1] / (2.0 * coeffs[0])
                    if lo <= centroid <= hi:
                        centroids[col] = centroid + top
                        continue
            # Fallback to simple peak
            centroids[col] = float(peak_row) + top

    detected = int(np.sum(np.isfinite(centroids)))
    logger.debug("Laser line: %d / %d columns detected.", detected, w)
    return centroids


@log_operation(logger)
def laser_to_height(
    centroids: np.ndarray,
    reference_row: float,
    config: Optional[SheetOfLightConfig] = None,
) -> np.ndarray:
    """Convert laser centroid positions to height values via triangulation.

    height = (reference_row - centroid) * pixel_size * baseline /
             (baseline * cos(angle) + (reference_row - centroid) * pixel_size * sin(angle))

    Simplified form for Scheimpflug-like geometry.

    Args:
        centroids:      (W,) float64 sub-pixel row positions from
                        :func:`extract_laser_line`.
        reference_row:  Row position corresponding to zero height.
        config:         Triangulation parameters.

    Returns:
        (W,) float64 height values in mm.  NaN where centroid is NaN.
    """
    cfg = config or SheetOfLightConfig()
    delta = (reference_row - centroids) * cfg.pixel_size  # displacement in mm
    sin_a = math.sin(cfg.angle)
    cos_a = math.cos(cfg.angle)

    denom = cfg.baseline * cos_a + delta * sin_a
    safe = np.abs(denom) > 1e-12
    height = np.full_like(centroids, np.nan, dtype=np.float64)
    height[safe] = (delta[safe] * cfg.baseline) / denom[safe]
    return height


@log_operation(logger)
def scan_to_profile(
    image: np.ndarray,
    reference_row: float,
    config: Optional[SheetOfLightConfig] = None,
) -> np.ndarray:
    """Extract a height profile from a single laser-line image.

    Convenience function that chains :func:`extract_laser_line` and
    :func:`laser_to_height`.

    Returns:
        (W,) float64 height profile in mm.
    """
    centroids = extract_laser_line(image, config)
    return laser_to_height(centroids, reference_row, config)


@log_operation(logger)
def profiles_to_surface(
    profiles: List[np.ndarray],
    scan_step: float = 1.0,
) -> np.ndarray:
    """Accumulate multiple height profiles into a 3D surface map.

    Args:
        profiles:   List of (W,) height profile arrays from successive scans.
        scan_step:  Physical distance between consecutive scan lines (mm).

    Returns:
        (num_profiles, W, 3) float64 surface map where the three channels are
        (X, Y, Z).  X is column * pixel pitch, Y is profile_index * scan_step,
        and Z is the height value.
    """
    if not profiles:
        raise ValueError("At least one profile is required.")

    num = len(profiles)
    w = profiles[0].shape[0]
    surface = np.zeros((num, w, 3), dtype=np.float64)

    x_coords = np.arange(w, dtype=np.float64)

    for i, prof in enumerate(profiles):
        surface[i, :, 0] = x_coords
        surface[i, :, 1] = i * scan_step
        surface[i, :, 2] = prof

    logger.info("Surface map: %d profiles x %d columns.", num, w)
    return surface


# ====================================================================== #
#  4. Auto-Focus Quality                                                  #
# ====================================================================== #


@log_operation(logger)
def focus_measure(
    image: np.ndarray,
    method: str = "laplacian_variance",
) -> float:
    """Compute a scalar focus quality metric for the given image.

    Supported methods:
        - ``"laplacian_variance"``: Variance of the Laplacian.
        - ``"tenengrad"``: Mean of squared Sobel gradient magnitude.
        - ``"modified_laplacian"``: Sum of absolute horizontal and vertical
          second derivatives.
        - ``"vollath_f4"``: Vollath's F4 auto-correlation measure.
        - ``"brenner"``: Brenner gradient (horizontal difference squared).

    Args:
        image:  Input image (grayscale or colour).
        method: Focus metric name.

    Returns:
        Non-negative float representing focus sharpness (higher = sharper).
    """
    validate_image(image, "image")
    gray = _ensure_gray(image).astype(np.float64)

    if method == "laplacian_variance":
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        return float(lap.var())

    if method == "tenengrad":
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        return float(np.mean(gx ** 2 + gy ** 2))

    if method == "modified_laplacian":
        kernel_h = np.array([[0, 0, 0], [-1, 2, -1], [0, 0, 0]], dtype=np.float64)
        kernel_v = kernel_h.T
        lh = np.abs(cv2.filter2D(gray, cv2.CV_64F, kernel_h))
        lv = np.abs(cv2.filter2D(gray, cv2.CV_64F, kernel_v))
        return float(np.mean(lh + lv))

    if method == "vollath_f4":
        h, w = gray.shape
        # F4 = sum(I(x,y)*I(x+1,y)) - sum(I(x,y)*I(x+2,y))
        f1 = np.sum(gray[:, :-1] * gray[:, 1:])
        f2 = np.sum(gray[:, :-2] * gray[:, 2:]) if w > 2 else 0.0
        return float(f1 - f2)

    if method == "brenner":
        diff = gray[:, 2:] - gray[:, :-2]
        return float(np.mean(diff ** 2))

    raise ValueError(f"Unknown focus method: {method!r}")


@log_operation(logger)
def find_best_focus(
    images: List[np.ndarray],
    positions: Optional[List[float]] = None,
    method: str = "laplacian_variance",
) -> Tuple[int, float, float]:
    """Find the sharpest image from a focal stack.

    Optionally performs sub-step parabolic interpolation when *positions*
    are provided and the best index is not at an endpoint.

    Args:
        images:    List of images captured at different focus positions.
        positions: Optional list of focus positions (e.g., Z-axis values).
        method:    Focus metric (see :func:`focus_measure`).

    Returns:
        Tuple of ``(best_index, best_position, best_score)``.
        If *positions* is ``None``, *best_position* equals *best_index*.
    """
    if not images:
        raise ValueError("At least one image is required.")

    scores = [focus_measure(img, method) for img in images]
    best_idx = int(np.argmax(scores))
    best_score = scores[best_idx]

    pos = positions if positions is not None else list(range(len(images)))

    # Sub-step parabolic interpolation
    if 0 < best_idx < len(scores) - 1 and positions is not None:
        s_prev = scores[best_idx - 1]
        s_curr = scores[best_idx]
        s_next = scores[best_idx + 1]
        denom = 2.0 * (2.0 * s_curr - s_prev - s_next)
        if abs(denom) > 1e-12:
            offset = (s_prev - s_next) / denom
            p_prev = pos[best_idx - 1]
            p_curr = pos[best_idx]
            p_next = pos[best_idx + 1]
            step = (p_next - p_prev) / 2.0
            best_pos = p_curr + offset * step
            logger.info("Best focus: idx=%d, pos=%.3f (interpolated), score=%.2f",
                        best_idx, best_pos, best_score)
            return best_idx, best_pos, best_score

    best_pos = pos[best_idx]
    logger.info("Best focus: idx=%d, pos=%.3f, score=%.2f",
                best_idx, best_pos, best_score)
    return best_idx, float(best_pos), best_score


# ====================================================================== #
#  5. Descriptor-Based Matching                                           #
# ====================================================================== #


@dataclass
class DescriptorModel:
    """Model for descriptor-based matching (uncalibrated/perspective scenes).

    Attributes:
        keypoints:     Serialised keypoint data (list of tuples).
        descriptors:   (N, D) uint8 or float32 descriptor matrix.
        template_size: (height, width) of the original template image.
        detector_type: Name of the feature detector used.
        homography:    Optional reference homography (identity for creation).
    """

    keypoints: List[Tuple[float, float, float, float, float, int, int]]
    descriptors: np.ndarray
    template_size: Tuple[int, int]
    detector_type: str
    homography: Optional[np.ndarray] = None


def _keypoints_to_list(
    kps: Sequence[cv2.KeyPoint],
) -> List[Tuple[float, float, float, float, float, int, int]]:
    """Serialise OpenCV KeyPoints into a plain list of tuples."""
    return [
        (kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
        for kp in kps
    ]


def _list_to_keypoints(
    data: List[Tuple[float, float, float, float, float, int, int]],
) -> List[cv2.KeyPoint]:
    """Deserialise plain tuples back into OpenCV KeyPoints."""
    return [
        cv2.KeyPoint(x=d[0], y=d[1], size=d[2], angle=d[3],
                      response=d[4], octave=d[5], class_id=d[6])
        for d in data
    ]


@log_operation(logger)
def create_descriptor_model(
    template: np.ndarray,
    detector_type: str = "ORB",
    max_keypoints: int = 1000,
) -> DescriptorModel:
    """Create a descriptor model from a template image.

    Args:
        template:       Template image (grayscale or colour).
        detector_type:  ``"ORB"``, ``"SIFT"``, or ``"AKAZE"``.
        max_keypoints:  Maximum number of keypoints to retain.

    Returns:
        A :class:`DescriptorModel` ready for matching.
    """
    validate_image(template, "template")
    gray = _ensure_gray(template)

    dt = detector_type.upper()
    if dt == "ORB":
        detector = cv2.ORB_create(nfeatures=max_keypoints)
    elif dt == "SIFT":
        detector = cv2.SIFT_create(nfeatures=max_keypoints)
    elif dt == "AKAZE":
        detector = cv2.AKAZE_create()
    else:
        raise ValueError(f"Unsupported detector: {detector_type!r}")

    kps, desc = detector.detectAndCompute(gray, None)
    if desc is None or len(kps) == 0:
        raise RuntimeError("No features detected in template image.")

    logger.info("Descriptor model: %d keypoints (%s).", len(kps), dt)
    return DescriptorModel(
        keypoints=_keypoints_to_list(kps),
        descriptors=desc,
        template_size=(gray.shape[0], gray.shape[1]),
        detector_type=dt,
        homography=np.eye(3, dtype=np.float64),
    )


@log_operation(logger)
def find_descriptor_match(
    image: np.ndarray,
    model: DescriptorModel,
    *,
    ratio_threshold: float = 0.75,
    min_matches: int = 10,
    ransac_threshold: float = 5.0,
) -> Optional[Dict]:
    """Match descriptor model against a target image with homography.

    Uses Lowe's ratio test and RANSAC-based homography estimation.

    Args:
        image:            Target image to search.
        model:            Descriptor model from :func:`create_descriptor_model`.
        ratio_threshold:  Lowe's ratio test threshold.
        min_matches:      Minimum good matches required.
        ransac_threshold: RANSAC reprojection threshold in pixels.

    Returns:
        Dict with keys ``"homography"`` (3x3), ``"num_matches"``,
        ``"inlier_ratio"``, ``"corners"`` (projected template corners),
        or ``None`` if matching fails.
    """
    validate_image(image, "image")
    gray = _ensure_gray(image)

    dt = model.detector_type
    if dt == "ORB":
        detector = cv2.ORB_create(nfeatures=2000)
    elif dt == "SIFT":
        detector = cv2.SIFT_create(nfeatures=2000)
    elif dt == "AKAZE":
        detector = cv2.AKAZE_create()
    else:
        raise ValueError(f"Unsupported detector: {dt!r}")

    kps_img, desc_img = detector.detectAndCompute(gray, None)
    if desc_img is None or len(kps_img) < min_matches:
        logger.warning("Too few keypoints in target image.")
        return None

    # Matcher selection
    if dt == "ORB":
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    raw_matches = matcher.knnMatch(model.descriptors, desc_img, k=2)

    # Lowe's ratio test
    good = []
    for m_pair in raw_matches:
        if len(m_pair) == 2 and m_pair[0].distance < ratio_threshold * m_pair[1].distance:
            good.append(m_pair[0])

    if len(good) < min_matches:
        logger.info("Descriptor match: only %d good matches (need %d).",
                     len(good), min_matches)
        return None

    # Homography via RANSAC
    kps_model = _list_to_keypoints(model.keypoints)
    pts_model = np.float32([kps_model[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_img = np.float32([kps_img[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts_model, pts_img, cv2.RANSAC, ransac_threshold)
    if H is None:
        logger.warning("Homography estimation failed.")
        return None

    inliers = int(mask.sum()) if mask is not None else 0
    inlier_ratio = inliers / len(good) if good else 0.0

    # Project template corners
    th, tw = model.template_size
    corners_model = np.float32([
        [0, 0], [tw, 0], [tw, th], [0, th],
    ]).reshape(-1, 1, 2)
    corners_img = cv2.perspectiveTransform(corners_model, H)

    logger.info("Descriptor match: %d inliers / %d matches (%.1f%%).",
                inliers, len(good), inlier_ratio * 100)
    return {
        "homography": H,
        "num_matches": len(good),
        "inlier_ratio": inlier_ratio,
        "corners": corners_img.reshape(-1, 2),
    }


# ====================================================================== #
#  6. Anisotropic Shape Model                                             #
# ====================================================================== #


@dataclass
class AnisotropicModel:
    """Shape model supporting independent X and Y scale ranges.

    Attributes:
        template:         Grayscale template image.
        edges:            Edge image of the template.
        contour_points:   List of (x, y) edge positions relative to origin.
        contour_angles:   Gradient angles at each contour point.
        origin:           (cx, cy) model centre.
        angle_range:      (min_angle, max_angle) in radians.
        angle_step:       Angle search step in radians.
        scale_x_range:    (min_sx, max_sx) horizontal scale range.
        scale_y_range:    (min_sy, max_sy) vertical scale range.
        scale_step:       Scale search increment.
        num_levels:       Pyramid levels for coarse-to-fine.
        min_contrast:     Minimum gradient magnitude for edge selection.
    """

    template: np.ndarray
    edges: np.ndarray
    contour_points: np.ndarray
    contour_angles: np.ndarray
    origin: Tuple[float, float]
    angle_range: Tuple[float, float] = (-math.radians(30), math.radians(30))
    angle_step: float = math.radians(1)
    scale_x_range: Tuple[float, float] = (0.8, 1.2)
    scale_y_range: Tuple[float, float] = (0.8, 1.2)
    scale_step: float = 0.05
    num_levels: int = 3
    min_contrast: int = 30


@log_operation(logger)
def create_aniso_shape_model(
    template: np.ndarray,
    *,
    angle_range: Tuple[float, float] = (-math.radians(30), math.radians(30)),
    angle_step: float = math.radians(1),
    scale_x_range: Tuple[float, float] = (0.8, 1.2),
    scale_y_range: Tuple[float, float] = (0.8, 1.2),
    scale_step: float = 0.05,
    num_levels: int = 3,
    min_contrast: int = 30,
    max_points: int = 1500,
) -> AnisotropicModel:
    """Create a shape model with independent X/Y scale ranges.

    Similar to ``create_shape_model`` but allows anisotropic scaling during
    the search phase, which is useful when objects may be stretched or
    compressed differently along each axis (e.g., perspective distortion,
    elastic deformation).

    Args:
        template:       Template image.
        angle_range:    Min/max rotation angle in radians.
        angle_step:     Rotation search step.
        scale_x_range:  Min/max horizontal scale factor.
        scale_y_range:  Min/max vertical scale factor.
        scale_step:     Scale search step (used for both axes).
        num_levels:     Pyramid levels.
        min_contrast:   Edge contrast threshold.
        max_points:     Maximum number of contour points to retain.

    Returns:
        An :class:`AnisotropicModel`.
    """
    validate_image(template, "template")
    gray = _ensure_gray(template).astype(np.float32)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    angles = np.arctan2(gy, gx)
    edges = (mag > min_contrast).astype(np.uint8) * 255

    ys, xs = np.where(mag > min_contrast)
    if len(xs) == 0:
        raise RuntimeError("No edges detected in template (try lowering min_contrast).")

    # Sub-sample if too many points
    if len(xs) > max_points:
        idx = np.linspace(0, len(xs) - 1, max_points, dtype=int)
        xs, ys = xs[idx], ys[idx]

    cx = gray.shape[1] / 2.0
    cy = gray.shape[0] / 2.0
    contour_pts = np.stack([xs - cx, ys - cy], axis=-1).astype(np.float32)
    contour_ang = angles[ys, xs]

    logger.info("Aniso model: %d contour points, origin=(%.1f, %.1f).",
                len(contour_pts), cx, cy)
    return AnisotropicModel(
        template=gray.astype(np.uint8) if gray.max() <= 255 else gray,
        edges=edges,
        contour_points=contour_pts,
        contour_angles=contour_ang,
        origin=(cx, cy),
        angle_range=angle_range,
        angle_step=angle_step,
        scale_x_range=scale_x_range,
        scale_y_range=scale_y_range,
        scale_step=scale_step,
        num_levels=num_levels,
        min_contrast=min_contrast,
    )


@log_operation(logger)
def find_aniso_shape_model(
    image: np.ndarray,
    model: AnisotropicModel,
    *,
    min_score: float = 0.5,
    greediness: float = 0.9,
    max_results: int = 1,
) -> List[Dict]:
    """Search for the anisotropic shape model in an image.

    Performs a coarse-to-fine search over (x, y, angle, scale_x, scale_y)
    and returns matching instances sorted by score.

    Args:
        image:       Search image.
        model:       Anisotropic model from :func:`create_aniso_shape_model`.
        min_score:   Minimum normalised match score in [0, 1].
        greediness:  Early-termination aggressiveness in [0, 1].
        max_results: Maximum number of matches to return.

    Returns:
        List of dicts with keys ``"row"``, ``"col"``, ``"angle"``,
        ``"scale_x"``, ``"scale_y"``, ``"score"``.
    """
    validate_image(image, "image")
    gray = _ensure_gray(image).astype(np.float32)
    h, w = gray.shape

    gx_img = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy_img = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag_img = np.sqrt(gx_img ** 2 + gy_img ** 2)
    ang_img = np.arctan2(gy_img, gx_img)

    pts = model.contour_points
    pt_angles = model.contour_angles
    n_pts = len(pts)
    greedy_thresh = 1.0 - greediness

    results: List[Dict] = []

    # Generate candidate parameters
    sx_vals = np.arange(model.scale_x_range[0], model.scale_x_range[1] + 1e-9,
                        model.scale_step)
    sy_vals = np.arange(model.scale_y_range[0], model.scale_y_range[1] + 1e-9,
                        model.scale_step)
    a_vals = np.arange(model.angle_range[0], model.angle_range[1] + 1e-9,
                       model.angle_step)

    # Coarse search on downsampled image for speed
    scale_factor = 2 ** (model.num_levels - 1)
    small_gray = cv2.resize(gray, (w // scale_factor, h // scale_factor))
    small_gx = cv2.Sobel(small_gray, cv2.CV_32F, 1, 0, ksize=3)
    small_gy = cv2.Sobel(small_gray, cv2.CV_32F, 0, 1, ksize=3)
    small_ang = np.arctan2(small_gy, small_gx)
    small_mag = np.sqrt(small_gx ** 2 + small_gy ** 2)
    sh, sw = small_gray.shape

    # Sub-sample points for coarse level
    coarse_step = max(1, n_pts // 100)
    coarse_idx = np.arange(0, n_pts, coarse_step)
    coarse_pts = pts[coarse_idx]
    coarse_ang = pt_angles[coarse_idx]
    n_coarse = len(coarse_idx)

    candidates = []

    for angle in a_vals[::max(1, len(a_vals) // 10)]:
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        for sx in sx_vals[::max(1, len(sx_vals) // 3)]:
            for sy in sy_vals[::max(1, len(sy_vals) // 3)]:
                # Transform coarse points
                tx = (coarse_pts[:, 0] * sx * cos_a - coarse_pts[:, 1] * sy * sin_a)
                ty = (coarse_pts[:, 0] * sx * sin_a + coarse_pts[:, 1] * sy * cos_a)
                tx_s = tx / scale_factor
                ty_s = ty / scale_factor

                # Slide over coarse image
                step_c = max(2, scale_factor)
                for cy_off in range(0, sh, step_c):
                    for cx_off in range(0, sw, step_c):
                        px = (cx_off + tx_s).astype(int)
                        py = (cy_off + ty_s).astype(int)
                        valid = (px >= 0) & (px < sw) & (py >= 0) & (py < sh)
                        if valid.sum() < n_coarse * 0.5:
                            continue
                        img_a = small_ang[py[valid], px[valid]]
                        mod_a = coarse_ang[valid] + angle
                        cos_diff = np.cos(img_a - mod_a)
                        score = cos_diff.mean()
                        if score > min_score * 0.7:
                            candidates.append((
                                cx_off * scale_factor,
                                cy_off * scale_factor,
                                angle, sx, sy, score,
                            ))

    # Refine top candidates at full resolution
    candidates.sort(key=lambda c: -c[5])
    seen: List[Tuple[float, float]] = []

    for cx_c, cy_c, angle_c, sx_c, sy_c, _ in candidates[:200]:
        cos_a, sin_a = math.cos(angle_c), math.sin(angle_c)
        tx = (pts[:, 0] * sx_c * cos_a - pts[:, 1] * sy_c * sin_a)
        ty = (pts[:, 0] * sx_c * sin_a + pts[:, 1] * sy_c * cos_a)

        # Search small window around coarse position
        best_local = 0.0
        best_pos = (cx_c, cy_c)
        for dy in range(-scale_factor, scale_factor + 1, max(1, scale_factor // 2)):
            for dx in range(-scale_factor, scale_factor + 1, max(1, scale_factor // 2)):
                px = (cx_c + dx + tx).astype(int)
                py = (cy_c + dy + ty).astype(int)
                valid = (px >= 0) & (px < w) & (py >= 0) & (py < h)
                if valid.sum() < n_pts * 0.3:
                    continue
                img_a = ang_img[py[valid], px[valid]]
                mod_a = pt_angles[valid] + angle_c
                cos_diff = np.cos(img_a - mod_a)

                # Greedy early termination
                partial = np.cumsum(cos_diff)
                n_checked = np.arange(1, len(cos_diff) + 1)
                running_score = partial / n_pts
                max_possible = running_score + (n_pts - n_checked) / n_pts
                if np.any(max_possible < min_score * greedy_thresh):
                    continue

                score = partial[-1] / n_pts if len(partial) > 0 else 0.0
                if score > best_local:
                    best_local = score
                    best_pos = (cx_c + dx, cy_c + dy)

        if best_local >= min_score:
            # NMS: skip if too close to existing result
            too_close = False
            for sx_p, sy_p in seen:
                if abs(best_pos[0] - sx_p) < 20 and abs(best_pos[1] - sy_p) < 20:
                    too_close = True
                    break
            if not too_close:
                results.append({
                    "row": float(best_pos[1]),
                    "col": float(best_pos[0]),
                    "angle": float(angle_c),
                    "scale_x": float(sx_c),
                    "scale_y": float(sy_c),
                    "score": float(best_local),
                })
                seen.append(best_pos)
                if len(results) >= max_results:
                    break

    results.sort(key=lambda r: -r["score"])
    logger.info("Aniso search: %d matches found.", len(results))
    return results[:max_results]


# ====================================================================== #
#  7. Component-Based Matching                                            #
# ====================================================================== #


@dataclass
class _SubModel:
    """A single component within a composite model.

    Attributes:
        template:       Grayscale template for this component.
        contour_points: Edge points relative to component centre.
        contour_angles: Gradient angles at contour points.
        origin:         (cx, cy) component centre.
        offset:         (dx, dy) expected offset from root component centre.
        tolerance:      Allowed displacement from expected offset (pixels).
    """

    template: np.ndarray
    contour_points: np.ndarray
    contour_angles: np.ndarray
    origin: Tuple[float, float]
    offset: Tuple[float, float]
    tolerance: float


@dataclass
class ComponentModel:
    """Multi-part model with relative geometry constraints.

    The first sub-model is the *root* component.  All other components have
    offsets defined relative to the root.

    Attributes:
        sub_models:   List of sub-model definitions.
        num_levels:   Pyramid levels for search.
        min_contrast: Edge contrast threshold for all components.
    """

    sub_models: List[_SubModel] = field(default_factory=list)
    num_levels: int = 3
    min_contrast: int = 30


@log_operation(logger)
def create_component_model(
    templates: List[np.ndarray],
    offsets: List[Tuple[float, float]],
    *,
    tolerances: Optional[List[float]] = None,
    min_contrast: int = 30,
    num_levels: int = 3,
    max_points_per: int = 500,
) -> ComponentModel:
    """Create a component-based model from multiple template patches.

    Args:
        templates:       List of template images (one per component).
        offsets:         Expected (dx, dy) offset of each component centre
                         relative to the *first* component's centre.
        tolerances:      Per-component displacement tolerance in pixels.
                         Defaults to 20 px for every component.
        min_contrast:    Edge contrast threshold.
        num_levels:      Pyramid levels for coarse-to-fine search.
        max_points_per:  Maximum contour points per component.

    Returns:
        A :class:`ComponentModel`.
    """
    if len(templates) != len(offsets):
        raise ValueError("templates and offsets must have the same length.")
    if not templates:
        raise ValueError("At least one template is required.")

    tols = tolerances or [20.0] * len(templates)
    sub_models: List[_SubModel] = []

    for i, (tmpl, offset, tol) in enumerate(zip(templates, offsets, tols)):
        validate_image(tmpl, f"templates[{i}]")
        gray = _ensure_gray(tmpl).astype(np.float32)

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx ** 2 + gy ** 2)
        angles = np.arctan2(gy, gx)

        ys, xs = np.where(mag > min_contrast)
        if len(xs) == 0:
            raise RuntimeError(f"No edges in template[{i}].")

        if len(xs) > max_points_per:
            idx = np.linspace(0, len(xs) - 1, max_points_per, dtype=int)
            xs, ys = xs[idx], ys[idx]

        cx = gray.shape[1] / 2.0
        cy = gray.shape[0] / 2.0
        pts = np.stack([xs - cx, ys - cy], axis=-1).astype(np.float32)
        angs = angles[ys, xs]

        sub_models.append(_SubModel(
            template=gray.astype(np.uint8),
            contour_points=pts,
            contour_angles=angs,
            origin=(cx, cy),
            offset=offset,
            tolerance=tol,
        ))

    logger.info("Component model: %d sub-models created.", len(sub_models))
    return ComponentModel(
        sub_models=sub_models,
        num_levels=num_levels,
        min_contrast=min_contrast,
    )


def _match_single_component(
    ang_img: np.ndarray,
    mag_img: np.ndarray,
    sub: _SubModel,
    search_cx: float,
    search_cy: float,
    search_radius: float,
    min_score: float,
) -> Optional[Dict]:
    """Search for one component near a predicted position.

    Returns dict with ``row``, ``col``, ``score`` or None.
    """
    h, w = ang_img.shape
    pts = sub.contour_points
    pt_angles = sub.contour_angles
    n_pts = len(pts)

    best_score = 0.0
    best_pos: Optional[Tuple[float, float]] = None
    r = int(search_radius)

    for dy in range(-r, r + 1, max(1, r // 5)):
        for dx in range(-r, r + 1, max(1, r // 5)):
            cx = search_cx + dx
            cy = search_cy + dy
            px = (cx + pts[:, 0]).astype(int)
            py = (cy + pts[:, 1]).astype(int)
            valid = (px >= 0) & (px < w) & (py >= 0) & (py < h)
            if valid.sum() < n_pts * 0.3:
                continue
            img_a = ang_img[py[valid], px[valid]]
            mod_a = pt_angles[valid]
            score = float(np.mean(np.cos(img_a - mod_a)))
            if score > best_score:
                best_score = score
                best_pos = (cx, cy)

    if best_pos is not None and best_score >= min_score:
        return {"row": best_pos[1], "col": best_pos[0], "score": best_score}
    return None


@log_operation(logger)
def find_component_model(
    image: np.ndarray,
    model: ComponentModel,
    *,
    min_score: float = 0.5,
    max_results: int = 5,
) -> List[Dict]:
    """Find instances of a component model in an image.

    First searches for the root component, then verifies all child
    components at their expected relative positions (within tolerance).

    Args:
        image:       Search image.
        model:       Component model from :func:`create_component_model`.
        min_score:   Minimum score for each component to count as a match.
        max_results: Maximum number of complete matches to return.

    Returns:
        List of dicts, each with ``"root_row"``, ``"root_col"``,
        ``"components"`` (list of per-component dicts with row/col/score),
        and ``"overall_score"`` (mean of component scores).
    """
    validate_image(image, "image")
    gray = _ensure_gray(image).astype(np.float32)
    h, w = gray.shape

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag_img = np.sqrt(gx ** 2 + gy ** 2)
    ang_img = np.arctan2(gy, gx)

    root = model.sub_models[0]
    root_pts = root.contour_points
    root_angles = root.contour_angles
    n_root = len(root_pts)

    # Coarse search for root component
    root_candidates: List[Tuple[float, float, float]] = []
    step = max(3, 2 ** (model.num_levels - 1))

    for cy_off in range(0, h, step):
        for cx_off in range(0, w, step):
            px = (cx_off + root_pts[:, 0]).astype(int)
            py = (cy_off + root_pts[:, 1]).astype(int)
            valid = (px >= 0) & (px < w) & (py >= 0) & (py < h)
            if valid.sum() < n_root * 0.3:
                continue
            img_a = ang_img[py[valid], px[valid]]
            mod_a = root_angles[valid]
            score = float(np.mean(np.cos(img_a - mod_a)))
            if score > min_score * 0.7:
                root_candidates.append((cx_off, cy_off, score))

    root_candidates.sort(key=lambda c: -c[2])

    results: List[Dict] = []
    seen_roots: List[Tuple[float, float]] = []

    for rcx, rcy, rscore in root_candidates[:50]:
        # NMS for root
        too_close = False
        for sx, sy in seen_roots:
            if abs(rcx - sx) < step and abs(rcy - sy) < step:
                too_close = True
                break
        if too_close:
            continue

        # Refine root position
        root_match = _match_single_component(
            ang_img, mag_img, root, rcx, rcy,
            search_radius=float(step), min_score=min_score,
        )
        if root_match is None:
            continue

        root_col = root_match["col"]
        root_row = root_match["row"]

        # Verify all child components
        comp_results = [root_match]
        all_ok = True

        for sub in model.sub_models[1:]:
            expected_cx = root_col + sub.offset[0]
            expected_cy = root_row + sub.offset[1]
            child = _match_single_component(
                ang_img, mag_img, sub,
                expected_cx, expected_cy,
                search_radius=sub.tolerance,
                min_score=min_score,
            )
            if child is None:
                all_ok = False
                break
            comp_results.append(child)

        if not all_ok:
            continue

        overall = float(np.mean([c["score"] for c in comp_results]))
        results.append({
            "root_row": root_row,
            "root_col": root_col,
            "components": comp_results,
            "overall_score": overall,
        })
        seen_roots.append((root_col, root_row))

        if len(results) >= max_results:
            break

    results.sort(key=lambda r: -r["overall_score"])
    logger.info("Component search: %d complete matches found.", len(results))
    return results[:max_results]
