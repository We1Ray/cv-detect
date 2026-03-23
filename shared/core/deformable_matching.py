"""
core/deformable_matching.py - Deformable (Non-Rigid) Template Matching.

Provides deformable template matching operators for flexible / soft materials
where rigid shape matching is insufficient.  Two deformation models are
supported:

    1. **Thin-Plate Spline (TPS)** - smooth global deformation driven by
       matched feature-point pairs.
    2. **Grid-Based Deformation** - piecewise-affine model that divides the
       template into a regular grid and estimates a local affine transform
       per cell.

The matching pipeline is:
    a. Extract keypoints and descriptors from template (ORB or SIFT).
    b. Match features between template and target image.
    c. Estimate a deformation field from the correspondences.
    d. Warp the template and compute a quality score.

Features:
    - Partial occlusion handling via robust matching & inlier ratio scoring
    - Match quality scoring (structural similarity + inlier ratio)
    - Visualisation helpers for matches, deformation field, and warped overlay

Dependencies: OpenCV, NumPy (no external deep-learning library required).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from shared.op_logger import log_operation
from shared.validation import validate_image

logger = logging.getLogger(__name__)

# ====================================================================== #
#  Named constants                                                        #
# ====================================================================== #

_DEFAULT_MAX_FEATURES: int = 1000
_DEFAULT_MATCH_RATIO: float = 0.75
_DEFAULT_RANSAC_REPROJ: float = 5.0
_DEFAULT_GRID_ROWS: int = 8
_DEFAULT_GRID_COLS: int = 8
_TPS_REGULARISATION: float = 1e-3
_MIN_GOOD_MATCHES: int = 8


class FeatureType(str, Enum):
    """Supported feature detector / descriptor types."""
    ORB = "orb"
    SIFT = "sift"


# ====================================================================== #
#  Data classes                                                           #
# ====================================================================== #


@dataclass
class DeformableModel:
    """Internal representation of a deformable template model.

    Attributes:
        template:     Grayscale template image (uint8).
        keypoints:    Detected keypoint coordinates (N, 2) as ``(x, y)``.
        descriptors:  Feature descriptors (N, D) array.
        feature_type: Detector used (``ORB`` or ``SIFT``).
        mask:         Optional binary mask for the template ROI.
    """

    template: np.ndarray
    keypoints: np.ndarray
    descriptors: np.ndarray
    feature_type: FeatureType = FeatureType.ORB
    mask: Optional[np.ndarray] = None


@dataclass
class MatchResult:
    """Result of a deformable matching operation.

    Attributes:
        score:           Overall match quality in ``[0, 1]``.
        warped_template: Template warped to align with the target image.
        src_pts:         Matched source (template) keypoint positions (M, 2).
        dst_pts:         Matched target keypoint positions (M, 2).
        inlier_mask:     Boolean mask indicating geometric inliers.
        deformation:     Deformation field (H, W, 2) mapping, or ``None``.
    """

    score: float
    warped_template: np.ndarray
    src_pts: np.ndarray
    dst_pts: np.ndarray
    inlier_mask: np.ndarray
    deformation: Optional[np.ndarray] = None


# ====================================================================== #
#  Internal helpers                                                       #
# ====================================================================== #


def _ensure_gray(img: np.ndarray) -> np.ndarray:
    """Convert to single-channel grayscale if needed."""
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[2] == 1:
        return img[:, :, 0]
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _ensure_bgr(img: np.ndarray) -> np.ndarray:
    """Return a 3-channel BGR copy."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img.copy()


def _build_detector(
    feature_type: FeatureType,
    max_features: int,
) -> cv2.Feature2D:
    """Instantiate a feature detector/descriptor."""
    if feature_type == FeatureType.SIFT:
        return cv2.SIFT_create(nfeatures=max_features)
    return cv2.ORB_create(nfeatures=max_features)


def _build_matcher(feature_type: FeatureType) -> cv2.DescriptorMatcher:
    """Instantiate a brute-force matcher appropriate for the descriptor."""
    if feature_type == FeatureType.SIFT:
        return cv2.BFMatcher(cv2.NORM_L2)
    return cv2.BFMatcher(cv2.NORM_HAMMING)


def _kp_to_array(kps: List[cv2.KeyPoint]) -> np.ndarray:
    """Convert a list of OpenCV KeyPoints to (N, 2) float array."""
    return np.array([kp.pt for kp in kps], dtype=np.float64)


# ====================================================================== #
#  Model creation                                                         #
# ====================================================================== #


@log_operation(logger)
def create_deformable_model(
    template: np.ndarray,
    feature_type: Union[str, FeatureType] = FeatureType.ORB,
    max_features: int = _DEFAULT_MAX_FEATURES,
    mask: Optional[np.ndarray] = None,
) -> DeformableModel:
    """Create a deformable template model by extracting keypoints.

    Parameters:
        template:      Template image (grayscale or BGR).
        feature_type:  ``"orb"`` or ``"sift"``.
        max_features:  Maximum number of keypoints to detect.
        mask:          Optional uint8 mask (255 = ROI, 0 = ignore).

    Returns:
        A :class:`DeformableModel` ready for matching.
    """
    validate_image(template, "template")
    gray = _ensure_gray(template)
    ft = FeatureType(feature_type) if isinstance(feature_type, str) else feature_type

    detector = _build_detector(ft, max_features)
    kps, descs = detector.detectAndCompute(gray, mask)

    if kps is None or len(kps) < _MIN_GOOD_MATCHES:
        raise ValueError(
            f"Only {0 if kps is None else len(kps)} keypoints detected; "
            f"need at least {_MIN_GOOD_MATCHES}"
        )

    pts = _kp_to_array(kps)
    logger.info("Created deformable model: %d keypoints (%s)",
                len(kps), ft.value)
    return DeformableModel(
        template=gray,
        keypoints=pts,
        descriptors=descs,
        feature_type=ft,
        mask=mask,
    )


# ====================================================================== #
#  Feature matching with ratio test                                       #
# ====================================================================== #


def _match_features(
    model: DeformableModel,
    target_gray: np.ndarray,
    max_features: int = _DEFAULT_MAX_FEATURES,
    ratio: float = _DEFAULT_MATCH_RATIO,
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect features in *target_gray* and match against the model.

    Returns:
        ``(src_pts, dst_pts)`` each (M, 2).  Raises ``ValueError`` if too
        few good matches are found.
    """
    detector = _build_detector(model.feature_type, max_features)
    kps_t, descs_t = detector.detectAndCompute(target_gray, None)

    if kps_t is None or descs_t is None or len(kps_t) < _MIN_GOOD_MATCHES:
        raise ValueError("Insufficient keypoints in target image")

    matcher = _build_matcher(model.feature_type)
    raw_matches = matcher.knnMatch(model.descriptors, descs_t, k=2)

    good: List[cv2.DMatch] = []
    for m_pair in raw_matches:
        if len(m_pair) == 2:
            m, n = m_pair
            if m.distance < ratio * n.distance:
                good.append(m)

    if len(good) < _MIN_GOOD_MATCHES:
        raise ValueError(
            f"Only {len(good)} good matches; need at least {_MIN_GOOD_MATCHES}"
        )

    pts_t = _kp_to_array(kps_t)
    src_pts = model.keypoints[[m.queryIdx for m in good]]
    dst_pts = pts_t[[m.trainIdx for m in good]]

    logger.info("Feature matching: %d / %d raw -> %d good matches",
                len(good), len(raw_matches), len(good))
    return src_pts, dst_pts


# ====================================================================== #
#  Thin-Plate Spline (TPS) warping                                        #
# ====================================================================== #


def _tps_kernel(r: np.ndarray) -> np.ndarray:
    """Thin-plate spline radial basis: U(r) = r^2 * log(r), with U(0) = 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(r > 0, r ** 2 * np.log(r), 0.0)
    return result


def _solve_tps(
    src: np.ndarray,
    dst: np.ndarray,
    regularisation: float = _TPS_REGULARISATION,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve for TPS warp coefficients mapping *src* -> *dst*.

    Parameters:
        src: (M, 2) control points in source space.
        dst: (M, 2) corresponding points in destination space.

    Returns:
        ``(weights, affine)`` where *weights* is (M, 2) and *affine* is (3, 2).
    """
    m = len(src)
    # Pairwise distances
    diff = src[:, np.newaxis, :] - src[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    K = _tps_kernel(dists)

    # Build system: [K P; P^T 0] [w; a] = [dst; 0]
    P = np.hstack([np.ones((m, 1)), src])  # (M, 3)
    L = np.zeros((m + 3, m + 3), dtype=np.float64)
    L[:m, :m] = K + regularisation * np.eye(m)
    L[:m, m:] = P
    L[m:, :m] = P.T

    rhs = np.zeros((m + 3, 2), dtype=np.float64)
    rhs[:m] = dst

    # Use lstsq for numerical robustness (TPS matrix can be near-singular
    # when control points are very close together).
    params, _, _, _ = np.linalg.lstsq(L, rhs, rcond=None)
    weights = params[:m]   # (M, 2)
    affine = params[m:]    # (3, 2)
    return weights, affine


def _apply_tps(
    points: np.ndarray,
    control_src: np.ndarray,
    weights: np.ndarray,
    affine: np.ndarray,
) -> np.ndarray:
    """Apply a TPS warp to an array of (N, 2) query points."""
    diff = points[:, np.newaxis, :] - control_src[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))  # (N, M)
    K = _tps_kernel(dists)

    P = np.hstack([np.ones((len(points), 1)), points])  # (N, 3)
    result = K @ weights + P @ affine
    return result


@log_operation(logger)
def _warp_image_tps(
    image: np.ndarray,
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    output_shape: Tuple[int, int],
    regularisation: float = _TPS_REGULARISATION,
) -> Tuple[np.ndarray, np.ndarray]:
    """Warp *image* from source to destination using a TPS model.

    Returns:
        ``(warped_image, deformation_field)``
    """
    h, w = output_shape
    weights, affine = _solve_tps(dst_pts, src_pts, regularisation)

    # Build grid of destination pixel coordinates
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float64),
                                 np.arange(h, dtype=np.float64))
    query = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    mapped = _apply_tps(query, dst_pts, weights, affine)
    map_x = mapped[:, 0].reshape(h, w).astype(np.float32)
    map_y = mapped[:, 1].reshape(h, w).astype(np.float32)

    warped = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    deformation = np.stack([
        map_x - grid_x.astype(np.float32),
        map_y - grid_y.astype(np.float32),
    ], axis=-1)

    return warped, deformation


# ====================================================================== #
#  Grid-based deformation model                                           #
# ====================================================================== #


@log_operation(logger)
def _warp_image_grid(
    image: np.ndarray,
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    output_shape: Tuple[int, int],
    grid_rows: int = _DEFAULT_GRID_ROWS,
    grid_cols: int = _DEFAULT_GRID_COLS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Warp *image* using a piecewise-affine grid deformation.

    The output region is divided into a grid.  For each cell, a local
    affine transform is estimated from the matched points that fall inside
    (or nearby).  Cells with insufficient matches fall back to the global
    affine transform.

    Returns:
        ``(warped_image, deformation_field)``
    """
    h, w = output_shape
    cell_h = h / grid_rows
    cell_w = w / grid_cols

    # Global affine fallback
    global_H, _ = cv2.estimateAffine2D(
        dst_pts.astype(np.float32), src_pts.astype(np.float32),
        method=cv2.RANSAC, ransacReprojThreshold=_DEFAULT_RANSAC_REPROJ,
    )
    if global_H is None:
        global_H = np.eye(2, 3, dtype=np.float64)

    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)

    for r in range(grid_rows):
        for c in range(grid_cols):
            y0 = int(r * cell_h)
            y1 = int((r + 1) * cell_h) if r < grid_rows - 1 else h
            x0 = int(c * cell_w)
            x1 = int((c + 1) * cell_w) if c < grid_cols - 1 else w

            # Find points near this cell (with margin)
            margin = max(cell_h, cell_w) * 0.5
            in_cell = (
                (dst_pts[:, 0] >= x0 - margin) &
                (dst_pts[:, 0] < x1 + margin) &
                (dst_pts[:, 1] >= y0 - margin) &
                (dst_pts[:, 1] < y1 + margin)
            )

            if in_cell.sum() >= 3:
                local_H, _ = cv2.estimateAffine2D(
                    dst_pts[in_cell].astype(np.float32),
                    src_pts[in_cell].astype(np.float32),
                    method=cv2.RANSAC,
                    ransacReprojThreshold=_DEFAULT_RANSAC_REPROJ,
                )
                if local_H is None:
                    local_H = global_H
            else:
                local_H = global_H

            # Fill in the map for this cell
            gx, gy = np.meshgrid(
                np.arange(x0, x1, dtype=np.float64),
                np.arange(y0, y1, dtype=np.float64),
            )
            pts_cell = np.stack([gx.ravel(), gy.ravel(), np.ones(gx.size)],
                                axis=1)
            mapped = (local_H @ pts_cell.T).T  # (K, 2)
            map_x[y0:y1, x0:x1] = mapped[:, 0].reshape(y1 - y0, x1 - x0).astype(np.float32)
            map_y[y0:y1, x0:x1] = mapped[:, 1].reshape(y1 - y0, x1 - x0).astype(np.float32)

    warped = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32),
                                 np.arange(h, dtype=np.float32))
    deformation = np.stack([map_x - grid_x, map_y - grid_y], axis=-1)

    return warped, deformation


# ====================================================================== #
#  High-level deformable matching                                         #
# ====================================================================== #


@log_operation(logger)
def find_deformable_match(
    model: DeformableModel,
    target: np.ndarray,
    method: str = "tps",
    max_features: int = _DEFAULT_MAX_FEATURES,
    match_ratio: float = _DEFAULT_MATCH_RATIO,
    ransac_reproj: float = _DEFAULT_RANSAC_REPROJ,
    grid_rows: int = _DEFAULT_GRID_ROWS,
    grid_cols: int = _DEFAULT_GRID_COLS,
) -> MatchResult:
    """Find and score a deformable match of *model* in *target*.

    Parameters:
        model:         A :class:`DeformableModel` from :func:`create_deformable_model`.
        target:        Target image (grayscale or BGR).
        method:        ``"tps"`` for Thin-Plate Spline or ``"grid"`` for
                       piecewise-affine grid deformation.
        max_features:  Max features to detect in the target.
        match_ratio:   Lowe's ratio test threshold.
        ransac_reproj: RANSAC reprojection threshold for inlier filtering.
        grid_rows:     Grid rows (only for ``"grid"`` method).
        grid_cols:     Grid cols (only for ``"grid"`` method).

    Returns:
        A :class:`MatchResult` with the score, warped template, and details.
    """
    validate_image(target, "target")
    target_gray = _ensure_gray(target)

    # Step 1: Feature matching
    src_pts, dst_pts = _match_features(model, target_gray, max_features,
                                       match_ratio)

    # Step 2: Geometric inlier filtering
    _, inlier_mask = cv2.estimateAffine2D(
        src_pts.astype(np.float32),
        dst_pts.astype(np.float32),
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_reproj,
    )
    if inlier_mask is None:
        inlier_mask = np.ones(len(src_pts), dtype=np.uint8)
    inlier_mask = inlier_mask.ravel().astype(bool)

    inlier_src = src_pts[inlier_mask]
    inlier_dst = dst_pts[inlier_mask]

    if len(inlier_src) < _MIN_GOOD_MATCHES:
        raise ValueError(
            f"Only {len(inlier_src)} inliers after RANSAC; "
            f"need at least {_MIN_GOOD_MATCHES}"
        )

    # Step 3: Warp template
    output_shape = target_gray.shape[:2]
    if method == "tps":
        warped, deformation = _warp_image_tps(
            model.template, inlier_src, inlier_dst, output_shape,
        )
    elif method == "grid":
        warped, deformation = _warp_image_grid(
            model.template, inlier_src, inlier_dst, output_shape,
            grid_rows, grid_cols,
        )
    else:
        raise ValueError(f"Unknown method '{method}'; use 'tps' or 'grid'")

    # Step 4: Compute quality score
    score = _compute_match_score(warped, target_gray, inlier_mask, model)

    logger.info("Deformable match (%s): score=%.4f, inliers=%d/%d",
                method, score, inlier_mask.sum(), len(src_pts))
    return MatchResult(
        score=score,
        warped_template=warped,
        src_pts=src_pts,
        dst_pts=dst_pts,
        inlier_mask=inlier_mask,
        deformation=deformation,
    )


def _compute_match_score(
    warped: np.ndarray,
    target_gray: np.ndarray,
    inlier_mask: np.ndarray,
    model: DeformableModel,
) -> float:
    """Compute a composite match quality score in [0, 1].

    The score combines:
        - **Inlier ratio** (proportion of matches that are geometric inliers).
        - **Normalised cross-correlation** between the warped template and the
          target in the overlapping region.
    """
    # Inlier ratio component
    inlier_ratio = float(inlier_mask.sum()) / max(len(inlier_mask), 1)

    # NCC component (only where warped is non-zero)
    overlap = warped > 0
    if overlap.sum() < 100:
        ncc = 0.0
    else:
        w_vals = warped[overlap].astype(np.float64)
        t_vals = target_gray[overlap].astype(np.float64)
        w_vals -= w_vals.mean()
        t_vals -= t_vals.mean()
        denom = np.sqrt((w_vals ** 2).sum() * (t_vals ** 2).sum())
        ncc = float((w_vals * t_vals).sum() / denom) if denom > 1e-12 else 0.0
        ncc = max(ncc, 0.0)

    # Weighted combination
    score = 0.4 * inlier_ratio + 0.6 * ncc
    return float(np.clip(score, 0.0, 1.0))


# ====================================================================== #
#  Visualisation                                                          #
# ====================================================================== #


@log_operation(logger)
def draw_matches(
    model: DeformableModel,
    target: np.ndarray,
    result: MatchResult,
    draw_inliers_only: bool = True,
) -> np.ndarray:
    """Draw matched keypoint pairs side-by-side.

    Returns:
        BGR visualisation image.
    """
    tmpl_bgr = _ensure_bgr(model.template)
    tgt_bgr = _ensure_bgr(target)

    h1, w1 = tmpl_bgr.shape[:2]
    h2, w2 = tgt_bgr.shape[:2]
    h_out = max(h1, h2)
    canvas = np.zeros((h_out, w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = tmpl_bgr
    canvas[:h2, w1:] = tgt_bgr

    mask = result.inlier_mask if draw_inliers_only else np.ones(len(result.src_pts), dtype=bool)
    for src, dst, is_inlier in zip(result.src_pts, result.dst_pts, mask):
        if not is_inlier and draw_inliers_only:
            continue
        color = (0, 255, 0) if is_inlier else (0, 0, 255)
        pt1 = (int(src[0]), int(src[1]))
        pt2 = (int(dst[0]) + w1, int(dst[1]))
        cv2.circle(canvas, pt1, 3, color, -1)
        cv2.circle(canvas, pt2, 3, color, -1)
        cv2.line(canvas, pt1, pt2, color, 1, cv2.LINE_AA)

    return canvas


@log_operation(logger)
def draw_deformation_field(
    target: np.ndarray,
    result: MatchResult,
    step: int = 16,
    scale: float = 1.0,
) -> np.ndarray:
    """Overlay a deformation vector field on the target image.

    Parameters:
        target: Target image (grayscale or BGR).
        result: Match result containing the deformation field.
        step:   Grid spacing for the arrows (pixels).
        scale:  Arrow length multiplier.

    Returns:
        BGR visualisation image.
    """
    vis = _ensure_bgr(target)
    if result.deformation is None:
        logger.warning("No deformation field available for visualisation")
        return vis

    h, w = result.deformation.shape[:2]
    for y in range(0, h, step):
        for x in range(0, w, step):
            dx = result.deformation[y, x, 0] * scale
            dy = result.deformation[y, x, 1] * scale
            mag = np.sqrt(dx * dx + dy * dy)
            if mag < 0.5:
                continue
            pt1 = (x, y)
            pt2 = (int(x + dx), int(y + dy))
            cv2.arrowedLine(vis, pt1, pt2, (0, 255, 255), 1, cv2.LINE_AA,
                            tipLength=0.3)

    return vis


@log_operation(logger)
def draw_warped_overlay(
    target: np.ndarray,
    result: MatchResult,
    alpha: float = 0.5,
) -> np.ndarray:
    """Blend the warped template over the target as a semi-transparent overlay.

    Parameters:
        target: Target image.
        result: Match result containing the warped template.
        alpha:  Blend factor (0 = target only, 1 = warped only).

    Returns:
        BGR blended image.
    """
    tgt_bgr = _ensure_bgr(target)
    warped_bgr = _ensure_bgr(result.warped_template)

    # Only blend where warped pixels are non-zero
    mask = result.warped_template > 0
    if mask.ndim == 2:
        mask_3c = np.stack([mask] * 3, axis=-1)
    else:
        mask_3c = mask

    blended = tgt_bgr.copy()
    blended[mask_3c] = cv2.addWeighted(
        warped_bgr, alpha, tgt_bgr, 1.0 - alpha, 0.0,
    )[mask_3c]

    return blended
