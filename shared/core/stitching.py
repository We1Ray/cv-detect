"""
core/stitching.py - Image Stitching for Large Product Inspection.

Combines multiple overlapping views into a single panoramic image.  Supports
three stitching modes:

    * **panorama** -- General-purpose stitching via OpenCV's built-in Stitcher
      with optional feature-based refinement.
    * **scan** -- Strip-by-strip stitching for line-scan cameras where images
      arrive in sequential order with known approximate overlap.  Uses phase
      correlation for fast sub-pixel alignment.
    * **grid** -- 2-D grid arrangement where the row/column layout is known in
      advance.  Rows are stitched first, then the resulting strips are stitched
      together vertically.

Blending strategies:

    * **multiband** -- Laplacian-pyramid blending for visually seamless results.
    * **feather** -- Distance-from-edge weighted alpha blending.
    * **none** -- Hard cut (no blending).

All public functions are decorated with ``@log_operation(logger)`` and perform
input validation via ``shared.validation.validate_image``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from shared.validation import validate_image
from shared.op_logger import log_operation

logger = logging.getLogger(__name__)

# ====================================================================== #
#  Named constants                                                        #
# ====================================================================== #

_MIN_MATCH_COUNT: int = 10
_LAPLACIAN_LEVELS: int = 5
_DEFAULT_OVERLAP_RATIO: float = 0.3
_SUPPORTED_IMAGE_EXTS: Tuple[str, ...] = (
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff",
)


# ====================================================================== #
#  Data classes                                                           #
# ====================================================================== #


@dataclass
class StitchResult:
    """Result container returned by all stitching functions.

    Attributes:
        panorama:      Stitched output image (BGR or grayscale ndarray).
        num_images:    Number of input images that were processed.
        status:        One of ``"success"``, ``"partial"``, ``"failed"``.
        homographies:  List of 3x3 homography matrices used for warping.
        overlap_mask:  Binary mask indicating regions where two or more
                       source images overlap in the final panorama.
        confidence:    Average feature-matching confidence in ``[0, 1]``.
        message:       Human-readable status description.
    """

    panorama: np.ndarray
    num_images: int
    status: str
    homographies: List[np.ndarray] = field(default_factory=list)
    overlap_mask: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.uint8),
    )
    confidence: float = 0.0
    message: str = ""


# ====================================================================== #
#  Internal helpers                                                       #
# ====================================================================== #


def _ensure_bgr(img: np.ndarray) -> np.ndarray:
    """Convert a grayscale image to BGR; return BGR images unchanged."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def _ensure_gray(img: np.ndarray) -> np.ndarray:
    """Convert image to single-channel uint8 grayscale if needed."""
    if img.ndim == 2:
        out = img
    elif img.ndim == 3 and img.shape[2] == 1:
        out = img[:, :, 0]
    elif img.ndim == 3 and img.shape[2] == 4:
        out = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    elif img.ndim == 3:
        out = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        out = img
    return out.astype(np.uint8) if out.dtype != np.uint8 else out


def _build_laplacian_pyramid(
    img: np.ndarray, levels: int,
) -> List[np.ndarray]:
    """Build a Laplacian pyramid with *levels* levels."""
    gaussian = img.copy().astype(np.float32)
    gp: List[np.ndarray] = [gaussian]
    for _ in range(levels - 1):
        gaussian = cv2.pyrDown(gaussian)
        gp.append(gaussian)

    lp: List[np.ndarray] = []
    for i in range(levels - 1):
        expanded = cv2.pyrUp(gp[i + 1], dstsize=(gp[i].shape[1], gp[i].shape[0]))
        lp.append(gp[i] - expanded)
    lp.append(gp[-1])
    return lp


def _reconstruct_from_laplacian(pyramid: List[np.ndarray]) -> np.ndarray:
    """Reconstruct image from a Laplacian pyramid."""
    img = pyramid[-1].copy()
    for i in range(len(pyramid) - 2, -1, -1):
        img = cv2.pyrUp(img, dstsize=(pyramid[i].shape[1], pyramid[i].shape[0]))
        img = img + pyramid[i]
    return img


def _multiband_blend(
    img1: np.ndarray,
    img2: np.ndarray,
    mask: np.ndarray,
    levels: int = _LAPLACIAN_LEVELS,
) -> np.ndarray:
    """Blend two images using Laplacian pyramid (multiband) blending.

    Parameters:
        img1:   First image (float32 or uint8).
        img2:   Second image (same size as *img1*).
        mask:   Binary mask -- 255 where *img1* should dominate, 0 for *img2*.
        levels: Number of pyramid levels.
    """
    img1_f = img1.astype(np.float32)
    img2_f = img2.astype(np.float32)
    mask_f = mask.astype(np.float32) / 255.0

    if mask_f.ndim == 2 and img1_f.ndim == 3:
        mask_f = np.stack([mask_f] * img1_f.shape[2], axis=-1)

    lp1 = _build_laplacian_pyramid(img1_f, levels)
    lp2 = _build_laplacian_pyramid(img2_f, levels)

    # Gaussian pyramid for the mask.
    gp_mask: List[np.ndarray] = [mask_f]
    for _ in range(levels - 1):
        gp_mask.append(cv2.pyrDown(gp_mask[-1]))

    blended_pyramid: List[np.ndarray] = []
    for la, lb, gm in zip(lp1, lp2, gp_mask):
        # Resize mask level if sizes mismatch after pyrDown rounding.
        if gm.shape[:2] != la.shape[:2]:
            gm = cv2.resize(gm, (la.shape[1], la.shape[0]))
        if gm.ndim == 2 and la.ndim == 3:
            gm = np.stack([gm] * la.shape[2], axis=-1)
        blended_pyramid.append(la * gm + lb * (1.0 - gm))

    result = _reconstruct_from_laplacian(blended_pyramid)
    return np.clip(result, 0, 255).astype(np.uint8)


def _feather_blend(
    img1: np.ndarray,
    img2: np.ndarray,
    mask1: np.ndarray,
    mask2: np.ndarray,
) -> np.ndarray:
    """Blend two images using distance-from-edge weighting (feathering).

    Parameters:
        img1, img2:   Images on the same canvas (same size).
        mask1, mask2: Binary masks indicating valid pixels for each image.
    """
    dist1 = cv2.distanceTransform(mask1, cv2.DIST_L2, 5).astype(np.float32)
    dist2 = cv2.distanceTransform(mask2, cv2.DIST_L2, 5).astype(np.float32)

    total = dist1 + dist2
    total[total == 0] = 1.0  # avoid division by zero

    weight1 = dist1 / total
    weight2 = dist2 / total

    if weight1.ndim == 2 and img1.ndim == 3:
        weight1 = np.stack([weight1] * img1.shape[2], axis=-1)
        weight2 = np.stack([weight2] * img1.shape[2], axis=-1)

    blended = (img1.astype(np.float32) * weight1 +
               img2.astype(np.float32) * weight2)
    return np.clip(blended, 0, 255).astype(np.uint8)


def _is_valid_homography(H: np.ndarray) -> bool:
    """Check that a homography matrix is not degenerate."""
    if H is None or H.shape != (3, 3):
        return False
    det = np.linalg.det(H)
    if abs(det) < 1e-6 or abs(det) > 1e6:
        return False
    # Check that the homography does not flip the image excessively.
    if H[2, 0] ** 2 + H[2, 1] ** 2 > 0.01:
        return False
    return True


# ====================================================================== #
#  Low-level building blocks                                              #
# ====================================================================== #


@log_operation(logger)
def detect_and_match_features(
    img1: np.ndarray,
    img2: np.ndarray,
    method: str = "orb",
    ratio_threshold: float = 0.75,
) -> Tuple[List, List, List]:
    """Detect keypoints, compute descriptors, and match between two images.

    Parameters:
        img1:            First image (BGR or grayscale).
        img2:            Second image (BGR or grayscale).
        method:          Feature detector -- ``"orb"``, ``"sift"``, or
                         ``"akaze"``.
        ratio_threshold: Lowe's ratio test threshold.  Matches whose distance
                         ratio exceeds this value are rejected.

    Returns:
        ``(keypoints1, keypoints2, good_matches)`` where *good_matches* is a
        list of ``cv2.DMatch`` objects that passed the ratio test.
    """
    validate_image(img1, "img1")
    validate_image(img2, "img2")

    gray1 = _ensure_gray(img1)
    gray2 = _ensure_gray(img2)

    method_lower = method.lower()
    if method_lower == "sift":
        detector = cv2.SIFT_create()
        norm_type = cv2.NORM_L2
    elif method_lower == "akaze":
        detector = cv2.AKAZE_create()
        norm_type = cv2.NORM_HAMMING
    else:
        detector = cv2.ORB_create(nfeatures=2000)
        norm_type = cv2.NORM_HAMMING

    kp1, des1 = detector.detectAndCompute(gray1, None)
    kp2, des2 = detector.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        logger.warning(
            "detect_and_match_features: insufficient descriptors "
            "(img1=%d, img2=%d)",
            0 if des1 is None else len(des1),
            0 if des2 is None else len(des2),
        )
        return list(kp1), list(kp2), []

    # Use FLANN for float descriptors, BFMatcher for binary.
    if norm_type == cv2.NORM_L2:
        index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    else:
        matcher = cv2.BFMatcher(norm_type)

    raw_matches = matcher.knnMatch(des1, des2, k=2)

    good_matches: List[cv2.DMatch] = []
    for pair in raw_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)

    logger.info(
        "Feature matching (%s): %d/%d good matches (ratio=%.2f)",
        method_lower,
        len(good_matches),
        len(raw_matches),
        ratio_threshold,
    )
    return list(kp1), list(kp2), good_matches


@log_operation(logger)
def estimate_homography(
    matches: List,
    kp1: List,
    kp2: List,
    method: str = "ransac",
    reproj_threshold: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a 3x3 homography from matched keypoints.

    Parameters:
        matches:          List of ``cv2.DMatch`` from feature matching.
        kp1, kp2:         Keypoint lists for image 1 and image 2.
        method:           Outlier rejection method (``"ransac"`` or
                          ``"lmeds"``).
        reproj_threshold: Maximum reprojection error for RANSAC inliers.

    Returns:
        ``(H, inlier_mask)`` -- *H* is the 3x3 homography mapping points
        from *img2* to *img1*'s coordinate frame; *inlier_mask* is a
        uint8 array of the same length as *matches* indicating inliers.

    Raises:
        ValueError: If fewer than ``_MIN_MATCH_COUNT`` matches are provided.
    """
    if len(matches) < _MIN_MATCH_COUNT:
        raise ValueError(
            f"estimate_homography: 需要至少 {_MIN_MATCH_COUNT} 個匹配點，"
            f"目前只有 {len(matches)} 個"
        )

    src_pts = np.float32(
        [kp1[m.queryIdx].pt for m in matches],
    ).reshape(-1, 1, 2)
    dst_pts = np.float32(
        [kp2[m.trainIdx].pt for m in matches],
    ).reshape(-1, 1, 2)

    cv_method = cv2.RANSAC if method.lower() == "ransac" else cv2.LMEDS
    H, mask = cv2.findHomography(dst_pts, src_pts, cv_method, reproj_threshold)

    if H is None:
        logger.error("estimate_homography: findHomography returned None")
        H = np.eye(3, dtype=np.float64)
        mask = np.zeros((len(matches), 1), dtype=np.uint8)

    if not _is_valid_homography(H):
        logger.warning(
            "estimate_homography: degenerate homography detected "
            "(det=%.4f), falling back to identity",
            float(np.linalg.det(H)),
        )
        H = np.eye(3, dtype=np.float64)

    inliers = int(mask.sum()) if mask is not None else 0
    logger.info(
        "Homography estimated: %d/%d inliers",
        inliers,
        len(matches),
    )
    return H, mask


@log_operation(logger)
def warp_and_blend(
    img1: np.ndarray,
    img2: np.ndarray,
    homography: np.ndarray,
    blend_mode: str = "multiband",
) -> np.ndarray:
    """Warp *img2* into *img1*'s coordinate frame and blend the overlap.

    Parameters:
        img1:       Reference image.
        img2:       Image to be warped.
        homography: 3x3 matrix mapping *img2* coordinates to *img1*.
        blend_mode: ``"multiband"``, ``"feather"``, or ``"none"``.

    Returns:
        Blended panorama image (uint8).
    """
    validate_image(img1, "img1")
    validate_image(img2, "img2")

    img1 = _ensure_bgr(img1)
    img2 = _ensure_bgr(img2)

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Compute the bounding box of img2 warped into img1's frame.
    corners_img2 = np.float32(
        [[0, 0], [w2, 0], [w2, h2], [0, h2]],
    ).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners_img2, homography)

    all_corners = np.concatenate(
        [
            np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2),
            warped_corners,
        ],
        axis=0,
    )
    x_min, y_min = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    x_max, y_max = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    # Translation to shift everything into positive coordinates.
    translation = np.array(
        [[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64,
    )
    canvas_w = int(x_max - x_min)
    canvas_h = int(y_max - y_min)

    # Warp img2 onto canvas.
    warped2 = cv2.warpPerspective(
        img2, translation @ homography, (canvas_w, canvas_h),
    )

    # Place img1 onto canvas.
    canvas1 = np.zeros_like(warped2)
    canvas1[-y_min:-y_min + h1, -x_min:-x_min + w1] = img1

    # Masks for valid pixels.
    mask1 = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    mask1[-y_min:-y_min + h1, -x_min:-x_min + w1] = 255

    gray_warped = cv2.cvtColor(warped2, cv2.COLOR_BGR2GRAY)
    mask2 = (gray_warped > 0).astype(np.uint8) * 255

    overlap = cv2.bitwise_and(mask1, mask2)
    has_overlap = np.any(overlap > 0)

    if blend_mode == "multiband" and has_overlap:
        result = _multiband_blend(canvas1, warped2, mask1)
    elif blend_mode == "feather" and has_overlap:
        result = _feather_blend(canvas1, warped2, mask1, mask2)
    else:
        # Hard cut -- img1 takes priority.
        result = warped2.copy()
        roi = mask1 > 0
        result[roi] = canvas1[roi]

    # Fill any remaining black pixels from the other image.
    black_pixels = np.all(result == 0, axis=-1)
    result[black_pixels & (mask2 > 0)] = warped2[black_pixels & (mask2 > 0)]

    return result


@log_operation(logger)
def stitch_strip(
    images: List[np.ndarray],
    overlap_ratio: float = _DEFAULT_OVERLAP_RATIO,
    direction: str = "horizontal",
) -> StitchResult:
    """Stitch sequential strip images using phase correlation.

    Designed for line-scan setups where images arrive in order with a
    known approximate overlap.

    Parameters:
        images:        Ordered list of strip images.
        overlap_ratio: Approximate fraction of overlap between consecutive
                       frames (``0.0`` to ``1.0``).
        direction:     ``"horizontal"`` or ``"vertical"``.

    Returns:
        :class:`StitchResult`.
    """
    if not images:
        return StitchResult(
            panorama=np.array([], dtype=np.uint8),
            num_images=0,
            status="failed",
            message="No images provided",
        )

    for idx, img in enumerate(images):
        validate_image(img, f"images[{idx}]")

    if len(images) == 1:
        return StitchResult(
            panorama=images[0].copy(),
            num_images=1,
            status="success",
            homographies=[np.eye(3, dtype=np.float64)],
            overlap_mask=np.zeros(images[0].shape[:2], dtype=np.uint8),
            confidence=1.0,
            message="Single image -- no stitching required",
        )

    is_vertical = direction.lower() == "vertical"
    homographies: List[np.ndarray] = [np.eye(3, dtype=np.float64)]
    panorama = _ensure_bgr(images[0]).copy()
    total_confidence = 0.0

    for i in range(1, len(images)):
        current = _ensure_bgr(images[i])
        gray_prev = _ensure_gray(images[i - 1])
        gray_curr = _ensure_gray(current)

        # Extract overlapping regions based on direction and ratio.
        if is_vertical:
            overlap_px = int(gray_prev.shape[0] * overlap_ratio)
            roi_prev = gray_prev[-overlap_px:, :]
            roi_curr = gray_curr[:overlap_px, :]
        else:
            overlap_px = int(gray_prev.shape[1] * overlap_ratio)
            roi_prev = gray_prev[:, -overlap_px:]
            roi_curr = gray_curr[:, :overlap_px]

        # Phase correlation for sub-pixel shift estimation.
        roi_prev_f = roi_prev.astype(np.float32)
        roi_curr_f = roi_curr.astype(np.float32)
        shift, response = cv2.phaseCorrelate(roi_prev_f, roi_curr_f)

        dx, dy = shift
        if is_vertical:
            # Overall shift: previous image height minus overlap plus correction.
            total_dy = gray_prev.shape[0] - overlap_px + dy
            H = np.array(
                [[1, 0, dx], [0, 1, total_dy], [0, 0, 1]], dtype=np.float64,
            )
        else:
            total_dx = gray_prev.shape[1] - overlap_px + dx
            H = np.array(
                [[1, 0, total_dx], [0, 1, dy], [0, 0, 1]], dtype=np.float64,
            )

        # Accumulate homography.
        accumulated_H = homographies[-1] @ H
        homographies.append(accumulated_H)
        total_confidence += float(response)

        # Warp and place on canvas.
        panorama = warp_and_blend(panorama, current, accumulated_H, "feather")

    avg_confidence = total_confidence / max(len(images) - 1, 1)
    overlap_mask = np.zeros(panorama.shape[:2], dtype=np.uint8)

    return StitchResult(
        panorama=panorama,
        num_images=len(images),
        status="success",
        homographies=homographies,
        overlap_mask=overlap_mask,
        confidence=min(avg_confidence, 1.0),
        message=f"Strip stitching completed ({direction}, {len(images)} images)",
    )


@log_operation(logger)
def stitch_grid(
    images: List[np.ndarray],
    grid_shape: Tuple[int, int],
    overlap_ratio: float = 0.2,
) -> StitchResult:
    """Stitch images arranged in a known 2-D grid.

    Parameters:
        images:        Flat list of images in row-major order.
        grid_shape:    ``(rows, cols)`` describing the grid layout.
        overlap_ratio: Approximate overlap fraction between adjacent tiles.

    Returns:
        :class:`StitchResult`.
    """
    rows, cols = grid_shape
    expected = rows * cols
    if len(images) != expected:
        return StitchResult(
            panorama=np.array([], dtype=np.uint8),
            num_images=len(images),
            status="failed",
            message=(
                f"Expected {expected} images for grid {grid_shape}, "
                f"got {len(images)}"
            ),
        )

    for idx, img in enumerate(images):
        validate_image(img, f"images[{idx}]")

    # Phase 1 -- stitch each row horizontally.
    row_strips: List[np.ndarray] = []
    all_homographies: List[np.ndarray] = []

    for r in range(rows):
        row_images = images[r * cols:(r + 1) * cols]
        if cols == 1:
            row_strips.append(_ensure_bgr(row_images[0]))
            all_homographies.append(np.eye(3, dtype=np.float64))
        else:
            result = stitch_strip(row_images, overlap_ratio, direction="horizontal")
            if result.status == "failed":
                return StitchResult(
                    panorama=np.array([], dtype=np.uint8),
                    num_images=len(images),
                    status="failed",
                    homographies=all_homographies,
                    message=f"Failed to stitch row {r}: {result.message}",
                )
            row_strips.append(result.panorama)
            all_homographies.extend(result.homographies)

    # Phase 2 -- stitch row strips vertically.
    if rows == 1:
        final = row_strips[0]
    else:
        vert_result = stitch_strip(row_strips, overlap_ratio, direction="vertical")
        if vert_result.status == "failed":
            return StitchResult(
                panorama=np.array([], dtype=np.uint8),
                num_images=len(images),
                status="failed",
                homographies=all_homographies,
                message=f"Failed to stitch rows vertically: {vert_result.message}",
            )
        final = vert_result.panorama
        all_homographies.extend(vert_result.homographies)

    return StitchResult(
        panorama=final,
        num_images=len(images),
        status="success",
        homographies=all_homographies,
        overlap_mask=np.zeros(final.shape[:2], dtype=np.uint8),
        confidence=1.0,
        message=f"Grid stitching completed ({rows}x{cols})",
    )


# ====================================================================== #
#  High-level API                                                         #
# ====================================================================== #


@log_operation(logger)
def stitch_images(
    images: List[np.ndarray],
    mode: str = "panorama",
    blend: str = "multiband",
    confidence_threshold: float = 0.3,
) -> StitchResult:
    """Stitch a list of images into a single panoramic image.

    Parameters:
        images:               List of input images (BGR or grayscale ndarrays).
        mode:                 ``"panorama"`` (general OpenCV stitcher),
                              ``"scan"`` (strip-by-strip), or ``"grid"``
                              (2-D grid -- requires images count to be a
                              perfect rectangle; falls back to panorama).
        blend:                ``"multiband"``, ``"feather"``, or ``"none"``.
        confidence_threshold: Minimum matching confidence; pairs below this
                              threshold are rejected.

    Returns:
        :class:`StitchResult`.
    """
    if not images:
        return StitchResult(
            panorama=np.array([], dtype=np.uint8),
            num_images=0,
            status="failed",
            message="No images provided",
        )

    for idx, img in enumerate(images):
        validate_image(img, f"images[{idx}]")

    if len(images) == 1:
        return StitchResult(
            panorama=images[0].copy(),
            num_images=1,
            status="success",
            homographies=[np.eye(3, dtype=np.float64)],
            overlap_mask=np.zeros(images[0].shape[:2], dtype=np.uint8),
            confidence=1.0,
            message="Single image -- no stitching required",
        )

    mode_lower = mode.lower()

    # ---- scan mode --------------------------------------------------- #
    if mode_lower == "scan":
        return stitch_strip(images, direction="horizontal")

    # ---- grid mode --------------------------------------------------- #
    if mode_lower == "grid":
        n = len(images)
        # Try to infer a reasonable grid shape.
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        if rows * cols != n:
            logger.warning(
                "stitch_images(grid): %d images cannot form a perfect grid; "
                "falling back to panorama mode.",
                n,
            )
        else:
            return stitch_grid(images, (rows, cols))

    # ---- panorama mode (default) ------------------------------------- #
    bgr_images = [_ensure_bgr(img) for img in images]

    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    stitcher.setPanoConfidenceThresh(confidence_threshold)

    status_code, pano = stitcher.stitch(bgr_images)

    if status_code == cv2.Stitcher_OK:
        logger.info(
            "OpenCV Stitcher succeeded (%d images -> %dx%d)",
            len(images),
            pano.shape[1],
            pano.shape[0],
        )
        return StitchResult(
            panorama=pano,
            num_images=len(images),
            status="success",
            homographies=[],
            overlap_mask=np.zeros(pano.shape[:2], dtype=np.uint8),
            confidence=1.0,
            message="Panorama stitching completed via OpenCV Stitcher",
        )

    # OpenCV stitcher failed -- fall back to pairwise approach.
    status_map = {
        cv2.Stitcher_ERR_NEED_MORE_IMGS: "ERR_NEED_MORE_IMGS",
        cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "ERR_HOMOGRAPHY_EST_FAIL",
        cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "ERR_CAMERA_PARAMS_ADJUST_FAIL",
    }
    err_name = status_map.get(status_code, f"UNKNOWN({status_code})")
    logger.warning(
        "OpenCV Stitcher failed (%s); attempting pairwise fallback",
        err_name,
    )

    return _pairwise_stitch(bgr_images, blend, confidence_threshold)


def _pairwise_stitch(
    images: List[np.ndarray],
    blend: str,
    confidence_threshold: float,
) -> StitchResult:
    """Fallback pairwise stitching when the OpenCV Stitcher fails."""
    panorama = images[0].copy()
    homographies: List[np.ndarray] = [np.eye(3, dtype=np.float64)]
    total_confidence = 0.0
    failed_pairs = 0

    for i in range(1, len(images)):
        try:
            kp1, kp2, good_matches = detect_and_match_features(
                panorama, images[i], method="sift",
            )
            if len(good_matches) < _MIN_MATCH_COUNT:
                logger.warning(
                    "Pairwise stitch: too few matches for image %d "
                    "(%d < %d); skipping.",
                    i, len(good_matches), _MIN_MATCH_COUNT,
                )
                failed_pairs += 1
                continue

            H, mask = estimate_homography(good_matches, kp1, kp2)
            inlier_ratio = float(mask.sum()) / len(good_matches) if mask is not None else 0.0

            if inlier_ratio < confidence_threshold:
                logger.warning(
                    "Pairwise stitch: low inlier ratio for image %d "
                    "(%.2f < %.2f); skipping.",
                    i, inlier_ratio, confidence_threshold,
                )
                failed_pairs += 1
                continue

            homographies.append(H)
            total_confidence += inlier_ratio
            panorama = warp_and_blend(panorama, images[i], H, blend)

        except Exception:
            logger.exception("Pairwise stitch failed for image %d", i)
            failed_pairs += 1

    if failed_pairs == len(images) - 1:
        return StitchResult(
            panorama=images[0].copy(),
            num_images=len(images),
            status="failed",
            homographies=homographies,
            confidence=0.0,
            message="All pairwise stitches failed",
        )

    stitched_count = len(images) - failed_pairs
    avg_conf = total_confidence / max(stitched_count - 1, 1)
    status = "success" if failed_pairs == 0 else "partial"

    return StitchResult(
        panorama=panorama,
        num_images=len(images),
        status=status,
        homographies=homographies,
        overlap_mask=np.zeros(panorama.shape[:2], dtype=np.uint8),
        confidence=min(avg_conf, 1.0),
        message=(
            f"Pairwise stitching: {stitched_count}/{len(images)} images "
            f"stitched ({failed_pairs} skipped)"
        ),
    )


@log_operation(logger)
def stitch_from_directory(
    image_dir: str,
    mode: str = "panorama",
    sort_by: str = "name",
) -> StitchResult:
    """Load all images from a directory and stitch them.

    Parameters:
        image_dir: Path to directory containing image files.
        mode:      Stitching mode (see :func:`stitch_images`).
        sort_by:   ``"name"`` (alphabetical), ``"date"`` (modification time),
                   or ``"custom"`` (no sorting, filesystem order).

    Returns:
        :class:`StitchResult`.
    """
    dir_path = Path(image_dir)
    if not dir_path.is_dir():
        return StitchResult(
            panorama=np.array([], dtype=np.uint8),
            num_images=0,
            status="failed",
            message=f"Directory does not exist: {image_dir}",
        )

    files = [
        f for f in dir_path.iterdir()
        if f.is_file() and f.suffix.lower() in _SUPPORTED_IMAGE_EXTS
    ]

    if not files:
        return StitchResult(
            panorama=np.array([], dtype=np.uint8),
            num_images=0,
            status="failed",
            message=f"No image files found in {image_dir}",
        )

    sort_by_lower = sort_by.lower()
    if sort_by_lower == "name":
        files.sort(key=lambda f: f.name)
    elif sort_by_lower == "date":
        files.sort(key=lambda f: f.stat().st_mtime)
    # "custom" -- keep filesystem order.

    images: List[np.ndarray] = []
    for fp in files:
        img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if img is None:
            logger.warning("Failed to read image: %s", fp)
            continue
        images.append(img)

    logger.info(
        "Loaded %d images from %s (sort=%s)",
        len(images),
        image_dir,
        sort_by,
    )
    return stitch_images(images, mode=mode)


# ====================================================================== #
#  Utility functions                                                      #
# ====================================================================== #


@log_operation(logger)
def estimate_overlap(img1: np.ndarray, img2: np.ndarray) -> float:
    """Estimate the overlap percentage between two images.

    Uses feature matching to find corresponding points and computes the
    fraction of *img1* that overlaps with *img2*.

    Parameters:
        img1, img2: Input images.

    Returns:
        Overlap fraction in ``[0.0, 1.0]``.
    """
    validate_image(img1, "img1")
    validate_image(img2, "img2")

    kp1, kp2, good_matches = detect_and_match_features(
        img1, img2, method="orb",
    )

    if len(good_matches) < 4:
        return 0.0

    try:
        H, mask = estimate_homography(good_matches, kp1, kp2)
    except ValueError:
        return 0.0

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners = np.float32(
        [[0, 0], [w2, 0], [w2, h2], [0, h2]],
    ).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(corners, H)

    # Intersection with img1's bounding rectangle.
    warped_pts = warped.reshape(-1, 2)
    x_min = max(0, float(warped_pts[:, 0].min()))
    y_min = max(0, float(warped_pts[:, 1].min()))
    x_max = min(w1, float(warped_pts[:, 0].max()))
    y_max = min(h1, float(warped_pts[:, 1].max()))

    if x_max <= x_min or y_max <= y_min:
        return 0.0

    overlap_area = (x_max - x_min) * (y_max - y_min)
    img1_area = w1 * h1
    return min(overlap_area / img1_area, 1.0)


@log_operation(logger)
def compute_seam(
    img1: np.ndarray,
    img2: np.ndarray,
    overlap_region: np.ndarray,
) -> np.ndarray:
    """Find an optimal seam through the overlap region.

    The seam minimises the colour difference between the two images along
    its path using dynamic programming.

    Parameters:
        img1, img2:      Images on the same canvas (same size).
        overlap_region:  Binary mask (uint8) marking the overlap area.

    Returns:
        Binary mask (uint8, same size as inputs) -- 255 on the side of
        *img1*, 0 on the side of *img2*.
    """
    validate_image(img1, "img1")
    validate_image(img2, "img2")

    diff = cv2.absdiff(
        img1.astype(np.float32),
        img2.astype(np.float32),
    )
    if diff.ndim == 3:
        diff = diff.sum(axis=2)

    # Restrict cost to overlap region.
    cost = diff.copy()
    cost[overlap_region == 0] = 1e9

    h, w = cost.shape

    # Forward pass (top to bottom).
    dp = cost.copy()
    for r in range(1, h):
        for c in range(w):
            left = dp[r - 1, max(c - 1, 0)]
            center = dp[r - 1, c]
            right = dp[r - 1, min(c + 1, w - 1)]
            dp[r, c] += min(left, center, right)

    # Backtrack to find the seam.
    seam_cols = np.zeros(h, dtype=np.int32)
    seam_cols[-1] = int(np.argmin(dp[-1]))
    for r in range(h - 2, -1, -1):
        c = seam_cols[r + 1]
        c_left = max(c - 1, 0)
        c_right = min(c + 1, w - 1)
        candidates = [c_left, c, c_right]
        vals = [dp[r, cc] for cc in candidates]
        seam_cols[r] = candidates[int(np.argmin(vals))]

    # Build mask: 255 left of seam (img1 side), 0 right (img2 side).
    mask = np.zeros((h, w), dtype=np.uint8)
    for r in range(h):
        mask[r, :seam_cols[r]] = 255

    return mask


@log_operation(logger)
def crop_black_borders(image: np.ndarray, threshold: int = 5) -> np.ndarray:
    """Remove black borders that arise from perspective warping.

    Detects the largest axis-aligned rectangle of non-black pixels.

    Parameters:
        image:     Input image (possibly with black borders).
        threshold: Pixel intensity below which a pixel is considered black.

    Returns:
        Cropped image with black borders removed.
    """
    validate_image(image, "image")

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Find bounding rectangle of non-zero pixels.
    coords = cv2.findNonZero(binary)
    if coords is None:
        logger.warning("crop_black_borders: entire image is below threshold")
        return image

    x, y, w, h = cv2.boundingRect(coords)
    cropped = image[y:y + h, x:x + w]

    logger.info(
        "Cropped black borders: (%d, %d, %d, %d) -> %dx%d",
        x, y, w, h, cropped.shape[1], cropped.shape[0],
    )
    return cropped


# ====================================================================== #
#  Drawing / visualisation                                                #
# ====================================================================== #


@log_operation(logger)
def draw_matches(
    img1: np.ndarray,
    img2: np.ndarray,
    kp1: List,
    kp2: List,
    matches: List,
    max_draw: int = 50,
) -> np.ndarray:
    """Visualise feature matches between two images side by side.

    Parameters:
        img1, img2:   Source images.
        kp1, kp2:     Keypoint lists.
        matches:      List of ``cv2.DMatch``.
        max_draw:     Maximum number of matches to draw.

    Returns:
        Side-by-side visualisation image (BGR uint8).
    """
    validate_image(img1, "img1")
    validate_image(img2, "img2")

    draw_matches_subset = matches[:max_draw]
    vis = cv2.drawMatches(
        _ensure_bgr(img1),
        kp1,
        _ensure_bgr(img2),
        kp2,
        draw_matches_subset,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    return vis


@log_operation(logger)
def draw_stitch_overview(
    images: List[np.ndarray],
    homographies: List[np.ndarray],
    canvas_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Show where each image maps in the final panorama.

    Draws coloured outlines for each image's projected boundary on a
    shared canvas.

    Parameters:
        images:       List of source images.
        homographies: Corresponding homography for each image.
        canvas_size:  ``(width, height)``; auto-computed if ``None``.

    Returns:
        Overview image (BGR uint8).
    """
    if not images or not homographies:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    n = min(len(images), len(homographies))

    # Determine canvas bounds.
    all_corners: List[np.ndarray] = []
    for i in range(n):
        h, w = images[i].shape[:2]
        corners = np.float32(
            [[0, 0], [w, 0], [w, h], [0, h]],
        ).reshape(-1, 1, 2)
        if homographies[i] is not None:
            warped = cv2.perspectiveTransform(corners, homographies[i])
        else:
            warped = corners
        all_corners.append(warped)

    all_pts = np.concatenate(all_corners, axis=0)
    x_min, y_min = all_pts.min(axis=0).ravel()
    x_max, y_max = all_pts.max(axis=0).ravel()

    if canvas_size is not None:
        cw, ch = canvas_size
    else:
        cw = int(x_max - x_min + 20)
        ch = int(y_max - y_min + 20)

    canvas = np.zeros((ch, cw, 3), dtype=np.uint8)
    offset_x = int(-x_min + 10)
    offset_y = int(-y_min + 10)

    # Colour palette for outlines.
    colours = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
        (0, 255, 255), (255, 0, 255), (128, 255, 0), (0, 128, 255),
    ]

    for i in range(n):
        pts = all_corners[i].reshape(-1, 2)
        pts[:, 0] += offset_x
        pts[:, 1] += offset_y
        pts_int = pts.astype(np.int32).reshape((-1, 1, 2))
        colour = colours[i % len(colours)]
        cv2.polylines(canvas, [pts_int], isClosed=True, color=colour, thickness=2)

        # Label.
        centroid = pts.mean(axis=0).astype(int)
        cv2.putText(
            canvas,
            f"#{i}",
            tuple(centroid),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            colour,
            1,
            cv2.LINE_AA,
        )

    return canvas
