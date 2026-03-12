"""
core/shape_matching.py - Shape-Based Matching (HALCON-style) using OpenCV.

Provides gradient-direction-based shape matching modelled after MVTec HALCON's
``create_shape_model`` / ``find_shape_model`` operators.  The implementation
relies entirely on OpenCV and NumPy.

Algorithm overview:
    1. A *ShapeModel* is created from a template image by extracting edge
       contour points and their associated gradient directions.
    2. A Gaussian pyramid is built so that the search can proceed coarse-to-fine.
    3. At each candidate position/angle/scale the match score is computed as
       the mean cosine similarity between the model gradient directions and
       the image gradient directions at the corresponding (transformed) pixel
       locations.
    4. Greediness-based early termination, non-maximum suppression, and
       sub-pixel refinement are applied.

Features:
    - Multi-scale pyramid search (coarse-to-fine)
    - Rotation invariant matching
    - Scale invariant matching
    - Sub-pixel position refinement via parabolic interpolation
    - Greedy early-termination for fast rejection
    - Score based on normalised gradient direction cosine similarity

Reference:
    Steger, C.  "Similarity Measures for Occlusion, Clutter, and Illumination
    Invariant Object Recognition".  *DAGM 2001*.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from shared.validation import validate_image, validate_positive
from shared.op_logger import log_operation

logger = logging.getLogger(__name__)

# ====================================================================== #
#  Named constants                                                        #
# ====================================================================== #

_DEFAULT_MIN_CONTRAST: int = 30
_DEFAULT_NUM_LEVELS: int = 4
_DEFAULT_ANGLE_STEP: float = math.radians(1)
_DEFAULT_GREEDINESS: float = 0.9
_MAX_CONTOUR_POINTS: int = 2000


# ====================================================================== #
#  Data classes                                                           #
# ====================================================================== #


@dataclass
class ShapeModel:
    """Internal representation of a shape model for gradient-direction matching.

    Created by :func:`create_shape_model`.  Stores edge gradient information
    at multiple pyramid levels so that :func:`find_shape_model` can perform
    an efficient coarse-to-fine search.

    Attributes:
        edges:           Edge magnitude image (float32, same size as template).
        gradient_x:      Horizontal Sobel gradient of the template (float32).
        gradient_y:      Vertical Sobel gradient of the template (float32).
        angles:          Gradient direction image in radians (float32).
        contour_points:  List of ``(x, y)`` pixel coordinates where the
                         gradient magnitude exceeds the contrast threshold.
        contour_angles:  Gradient angle at each contour point (1-D float32
                         array, ``len == len(contour_points)``).
        origin:          Model origin ``(cx, cy)`` used as the rotation/scale
                         centre (typically the template centre).
        bounding_box:    Tight bounding box around contour points
                         ``(x, y, w, h)``.
        pyramid:         Multi-scale representation.  Each element is a dict
                         with keys ``edges``, ``gradient_x``, ``gradient_y``,
                         ``contour_points``, ``contour_angles``.  Index 0 is
                         the finest (original) level.
        angle_start:     Start of the allowed rotation range (radians).
        angle_extent:    Extent of the allowed rotation range (radians).
        scale_min:       Minimum allowed scale factor.
        scale_max:       Maximum allowed scale factor.
        num_levels:      Number of pyramid levels.
    """

    edges: np.ndarray
    gradient_x: np.ndarray
    gradient_y: np.ndarray
    angles: np.ndarray
    contour_points: List[Tuple[int, int]]
    contour_angles: np.ndarray
    origin: Tuple[float, float]
    bounding_box: Tuple[int, int, int, int]
    pyramid: List[Dict] = field(default_factory=list, repr=False)
    angle_start: float = 0.0
    angle_extent: float = 2.0 * math.pi
    scale_min: float = 1.0
    scale_max: float = 1.0
    num_levels: int = _DEFAULT_NUM_LEVELS


@dataclass
class MatchResult:
    """A single match returned by :func:`find_shape_model`.

    Attributes:
        row:   Sub-pixel Y position of the match in the search image.
        col:   Sub-pixel X position of the match in the search image.
        angle: Rotation angle of the best match (radians).
        scale: Scale factor of the best match.
        score: Normalised match score in ``[0, 1]``.
    """

    row: float
    col: float
    angle: float
    scale: float
    score: float


# ====================================================================== #
#  Internal helpers                                                       #
# ====================================================================== #


def _ensure_gray(img: np.ndarray) -> np.ndarray:
    """Convert image to single-channel uint8 grayscale if it is not already."""
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


def _compute_gradients(
    image: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Sobel gradients, magnitude, and direction.

    Parameters:
        image: Single-channel uint8 image.

    Returns:
        ``(gradient_x, gradient_y, magnitude, angle)`` -- all float32.
        *angle* is in radians (range ``[-pi, pi]``).
    """
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    ang = np.arctan2(gy, gx)
    return gx, gy, mag, ang


def _build_pyramid(image: np.ndarray, num_levels: int) -> List[np.ndarray]:
    """Build a Gaussian pyramid with *num_levels* levels.

    Level 0 is the original image; each subsequent level is halved in size
    via ``cv2.pyrDown``.

    Parameters:
        image:      Single-channel image (uint8 or float32).
        num_levels: Total number of levels including the original.

    Returns:
        List of images from finest (index 0) to coarsest (index
        ``num_levels - 1``).
    """
    pyr: List[np.ndarray] = [image]
    for _ in range(num_levels - 1):
        pyr.append(cv2.pyrDown(pyr[-1]))
    return pyr


def _rotate_points(
    points: np.ndarray,
    angle: float,
    origin: Tuple[float, float],
) -> np.ndarray:
    """Rotate *points* by *angle* radians around *origin*.

    Parameters:
        points: ``(N, 2)`` array of ``(x, y)`` coordinates.
        angle:  Rotation angle in radians (counter-clockwise positive).
        origin: ``(cx, cy)`` centre of rotation.

    Returns:
        ``(N, 2)`` array of rotated coordinates (float64).
    """
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    ox, oy = origin

    translated = points.astype(np.float64) - np.array([[ox, oy]])
    rotated = np.empty_like(translated)
    rotated[:, 0] = translated[:, 0] * cos_a - translated[:, 1] * sin_a
    rotated[:, 1] = translated[:, 0] * sin_a + translated[:, 1] * cos_a
    rotated += np.array([[ox, oy]])
    return rotated


def _extract_contour_points(
    magnitude: np.ndarray,
    angle: np.ndarray,
    min_contrast: float,
    max_points: int = _MAX_CONTOUR_POINTS,
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """Extract edge contour points where gradient magnitude > *min_contrast*.

    If the number of qualifying pixels exceeds *max_points*, a uniform
    sub-sampling is performed to keep the count manageable.

    Returns:
        ``(contour_points, contour_angles)`` where *contour_points* is a list
        of ``(x, y)`` tuples and *contour_angles* is a float32 1-D array.
    """
    ys, xs = np.where(magnitude > min_contrast)
    if len(xs) == 0:
        return [], np.array([], dtype=np.float32)

    if len(xs) > max_points:
        step = max(1, len(xs) // max_points)
        indices = np.arange(0, len(xs), step)[:max_points]
        xs = xs[indices]
        ys = ys[indices]

    contour_pts: List[Tuple[int, int]] = list(zip(xs.tolist(), ys.tolist()))
    contour_ang = angle[ys, xs].astype(np.float32)
    return contour_pts, contour_ang


# ====================================================================== #
#  Public API -- model creation                                           #
# ====================================================================== #


@log_operation(logger)
def create_shape_model(
    template: np.ndarray,
    num_levels: int = _DEFAULT_NUM_LEVELS,
    angle_start: float = 0.0,
    angle_extent: float = 2.0 * math.pi,
    angle_step: float = _DEFAULT_ANGLE_STEP,
    scale_min: float = 1.0,
    scale_max: float = 1.0,
    scale_step: float = 0.01,
    min_contrast: int = _DEFAULT_MIN_CONTRAST,
) -> ShapeModel:
    """Create a shape model from a template image.

    The model stores gradient-direction information at multiple pyramid
    levels so that :func:`find_shape_model` can perform an efficient
    coarse-to-fine search.

    Modelled after HALCON's ``create_shape_model`` operator.

    Parameters:
        template:      Reference image (grayscale or BGR).
        num_levels:    Number of Gaussian pyramid levels (``>= 1``).
        angle_start:   Start of the rotation search range (radians).
        angle_extent:  Extent of the rotation search range (radians).
        angle_step:    Angular resolution for the search (radians, ``> 0``).
        scale_min:     Minimum scale factor.
        scale_max:     Maximum scale factor.
        scale_step:    Scale resolution (``> 0``).
        min_contrast:  Minimum gradient magnitude to accept an edge pixel.

    Returns:
        A :class:`ShapeModel` ready for use with :func:`find_shape_model`.

    Raises:
        shared.validation.ImageValidationError: If *template* is invalid.
    """
    validate_image(template, "template")
    validate_positive(num_levels, "num_levels")
    validate_positive(min_contrast, "min_contrast")

    gray = _ensure_gray(template)
    gx, gy, mag, ang = _compute_gradients(gray)

    contour_pts, contour_ang = _extract_contour_points(
        mag, ang, float(min_contrast),
    )

    if len(contour_pts) == 0:
        logger.warning(
            "create_shape_model: no edge pixels exceed min_contrast=%d; "
            "consider lowering the threshold.",
            min_contrast,
        )

    # Model origin is the template centre.
    h, w = gray.shape[:2]
    origin = (w / 2.0, h / 2.0)

    # Bounding box around contour points.
    if contour_pts:
        pts_arr = np.array(contour_pts, dtype=np.int32)
        x_min, y_min = int(pts_arr[:, 0].min()), int(pts_arr[:, 1].min())
        x_max, y_max = int(pts_arr[:, 0].max()), int(pts_arr[:, 1].max())
        bbox: Tuple[int, int, int, int] = (
            x_min, y_min, x_max - x_min + 1, y_max - y_min + 1,
        )
    else:
        bbox = (0, 0, w, h)

    # Build pyramid data -- one dict per level.
    pyramid: List[Dict] = []
    pyr_images = _build_pyramid(gray, num_levels)
    for lvl, pyr_img in enumerate(pyr_images):
        lvl_gx, lvl_gy, lvl_mag, lvl_ang = _compute_gradients(pyr_img)
        # Scale the contrast threshold proportionally (edges weaken at
        # coarser levels).
        lvl_contrast = max(min_contrast / (2 ** lvl), 5.0)
        lvl_pts, lvl_ang_arr = _extract_contour_points(
            lvl_mag, lvl_ang, lvl_contrast,
        )
        pyramid.append({
            "edges": lvl_mag,
            "gradient_x": lvl_gx,
            "gradient_y": lvl_gy,
            "contour_points": lvl_pts,
            "contour_angles": lvl_ang_arr,
        })

    model = ShapeModel(
        edges=mag,
        gradient_x=gx,
        gradient_y=gy,
        angles=ang,
        contour_points=contour_pts,
        contour_angles=contour_ang,
        origin=origin,
        bounding_box=bbox,
        pyramid=pyramid,
        angle_start=angle_start,
        angle_extent=angle_extent,
        scale_min=scale_min,
        scale_max=scale_max,
        num_levels=num_levels,
    )
    logger.info(
        "Shape model created: %d contour points, %d pyramid levels, "
        "angle=[%.1f, %.1f] deg, scale=[%.2f, %.2f]",
        len(contour_pts),
        num_levels,
        math.degrees(angle_start),
        math.degrees(angle_start + angle_extent),
        scale_min,
        scale_max,
    )
    return model


# ====================================================================== #
#  Internal scoring                                                       #
# ====================================================================== #


def _score_candidate(
    img_ang: np.ndarray,
    model_pts: np.ndarray,
    model_angles: np.ndarray,
    row: float,
    col: float,
    angle: float,
    scale: float,
    origin: Tuple[float, float],
    greediness: float,
    min_score: float,
) -> float:
    """Compute the gradient-direction match score for a single candidate.

    The score is the mean of ``cos(model_angle[i] + angle - image_angle[i])``
    evaluated at each transformed contour point.

    Early termination is applied when partial scoring indicates the final
    score cannot exceed *min_score* (controlled by *greediness*).

    Parameters:
        img_ang:       Gradient direction image (float32, radians).
        model_pts:     ``(N, 2)`` model contour points ``(x, y)``.
        model_angles:  ``(N,)`` gradient angles at model contour points.
        row:           Candidate Y position.
        col:           Candidate X position.
        angle:         Candidate rotation angle (radians).
        scale:         Candidate scale factor.
        origin:        Model origin ``(cx, cy)``.
        greediness:    Greediness in ``[0, 1]`` for early termination.
        min_score:     Minimum acceptable score.

    Returns:
        Score in ``[0, 1]``, or ``0.0`` if early termination triggers.
    """
    n = len(model_pts)
    if n == 0:
        return 0.0

    pts = model_pts.astype(np.float64)
    ox, oy = origin

    # Scale around origin, then rotate, then translate.
    pts_centered = pts - np.array([[ox, oy]])
    pts_scaled = pts_centered * scale

    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    rotated = np.empty_like(pts_scaled)
    rotated[:, 0] = pts_scaled[:, 0] * cos_a - pts_scaled[:, 1] * sin_a
    rotated[:, 1] = pts_scaled[:, 0] * sin_a + pts_scaled[:, 1] * cos_a

    # Translate to candidate position.
    tx = col - ox * scale
    ty = row - oy * scale
    transformed = rotated + np.array([[ox + tx, oy + ty]])

    # Round to integer pixel coordinates and clip to image bounds.
    ix = np.round(transformed[:, 0]).astype(np.intp)
    iy = np.round(transformed[:, 1]).astype(np.intp)

    h, w = img_ang.shape[:2]
    valid = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    n_valid = int(valid.sum())
    if n_valid == 0:
        return 0.0

    # Cosine similarity between rotated model angles and image angles.
    img_angles_at = img_ang[iy[valid], ix[valid]]
    model_angles_at = model_angles[valid] + angle

    cos_diff = np.cos(model_angles_at - img_angles_at)

    # Greedy early termination: check at fractional checkpoints whether
    # the achievable score can still beat min_score.
    if greediness > 0 and n_valid > 20:
        threshold = greediness * min_score
        quarter = n_valid // 4
        if float(np.mean(cos_diff[:quarter])) < threshold:
            return 0.0
        half = n_valid // 2
        if float(np.mean(cos_diff[:half])) < threshold:
            return 0.0

    # Penalise if many points fell outside the image.
    coverage = n_valid / n
    score = float(np.mean(cos_diff)) * coverage
    return max(score, 0.0)


# ====================================================================== #
#  Public API -- model search                                             #
# ====================================================================== #


@log_operation(logger)
def find_shape_model(
    image: np.ndarray,
    model: ShapeModel,
    angle_start: Optional[float] = None,
    angle_extent: Optional[float] = None,
    min_score: float = 0.5,
    num_matches: int = 1,
    max_overlap: float = 0.5,
    sub_pixel: str = "least_squares",
    greediness: float = _DEFAULT_GREEDINESS,
) -> List[MatchResult]:
    """Find occurrences of a shape model in a search image.

    Modelled after HALCON's ``find_shape_model`` operator.

    The search proceeds coarse-to-fine through the model's pyramid levels.
    At each level the gradient-direction score is evaluated for a grid of
    candidate positions, angles, and scales.  Promising candidates are
    refined at finer levels.  Sub-pixel refinement and non-maximum
    suppression are applied at the finest level.

    Parameters:
        image:        Search image (grayscale or BGR).
        model:        Shape model created by :func:`create_shape_model`.
        angle_start:  Override the model's angle range start (radians).
                      ``None`` uses the model's stored value.
        angle_extent: Override the model's angle range extent (radians).
                      ``None`` uses the model's stored value.
        min_score:    Minimum acceptable match score in ``[0, 1]``.
        num_matches:  Maximum number of matches to return.
        max_overlap:  Maximum allowed overlap between matches (distance-
                      based NMS; ``0.0`` = no overlap, ``1.0`` = no NMS).
        sub_pixel:    Sub-pixel refinement strategy.  ``"least_squares"``
                      applies parabolic interpolation; ``"none"`` skips it.
        greediness:   Early-termination aggressiveness in ``[0, 1]``.
                      Higher values skip unpromising candidates sooner.

    Returns:
        List of :class:`MatchResult` sorted by score (descending), at most
        *num_matches* entries.

    Raises:
        shared.validation.ImageValidationError: If *image* is invalid.
    """
    validate_image(image, "image")

    if angle_start is None:
        angle_start = model.angle_start
    if angle_extent is None:
        angle_extent = model.angle_extent

    gray = _ensure_gray(image)
    num_levels = model.num_levels

    # Build image pyramid and pre-compute gradient directions.
    img_pyr = _build_pyramid(gray, num_levels)
    img_ang_pyr: List[np.ndarray] = []
    for pyr_img in img_pyr:
        _, _, _, ang = _compute_gradients(pyr_img)
        img_ang_pyr.append(ang)

    # Angle / scale search bounds.
    a_start = angle_start
    a_end = angle_start + angle_extent
    s_min = model.scale_min
    s_max = model.scale_max

    # Pre-compute angle steps at the coarsest level -- wider steps that
    # will be refined at finer levels.
    coarse_angle_step = _DEFAULT_ANGLE_STEP * (2 ** (num_levels - 1))
    angle_steps_coarse = np.arange(a_start, a_end, coarse_angle_step)
    if len(angle_steps_coarse) == 0:
        angle_steps_coarse = np.array([a_start])

    # Pre-compute sin/cos lookup for the coarse angle steps.
    _cos_lut = np.cos(angle_steps_coarse)
    _sin_lut = np.sin(angle_steps_coarse)

    scale_step_coarse = 0.05
    if s_min >= s_max:
        scale_steps = np.array([s_min])
    else:
        scale_steps = np.arange(
            s_min, s_max + scale_step_coarse * 0.5, scale_step_coarse,
        )
        if len(scale_steps) == 0:
            scale_steps = np.array([s_min])

    # ------------------------------------------------------------------ #
    #  Coarsest level: exhaustive search                                  #
    # ------------------------------------------------------------------ #
    coarsest = num_levels - 1
    coarse_img_ang = img_ang_pyr[coarsest]
    coarse_h, coarse_w = coarse_img_ang.shape[:2]

    coarse_data = (
        model.pyramid[coarsest]
        if coarsest < len(model.pyramid)
        else model.pyramid[-1]
    )
    coarse_pts = coarse_data["contour_points"]
    coarse_angs = coarse_data["contour_angles"]

    if len(coarse_pts) == 0:
        logger.warning(
            "find_shape_model: model has no contour points at the coarsest "
            "pyramid level."
        )
        return []

    coarse_pts_arr = np.array(coarse_pts, dtype=np.float64)
    scale_factor = 1.0 / (2 ** coarsest)
    coarse_origin = (
        model.origin[0] * scale_factor,
        model.origin[1] * scale_factor,
    )

    # Search step in pixels at the coarsest level.
    search_step = max(2, int(round(4 * scale_factor)))

    # Each candidate: (score, row, col, angle, scale)
    candidates: List[Tuple[float, float, float, float, float]] = []

    coarse_threshold = min_score * 0.5

    for s in scale_steps:
        for a in angle_steps_coarse:
            for r in range(0, coarse_h, search_step):
                for c in range(0, coarse_w, search_step):
                    sc = _score_candidate(
                        coarse_img_ang,
                        coarse_pts_arr,
                        coarse_angs,
                        float(r),
                        float(c),
                        a,
                        s,
                        coarse_origin,
                        greediness,
                        coarse_threshold,
                    )
                    if sc >= coarse_threshold:
                        candidates.append((sc, float(r), float(c), a, s))

    if not candidates:
        logger.info("find_shape_model: no candidates found at coarsest level.")
        return []

    # Sort by score descending and keep a bounded set for refinement.
    candidates.sort(key=lambda t: t[0], reverse=True)
    max_candidates = max(num_matches * 50, 200)
    candidates = candidates[:max_candidates]

    # ------------------------------------------------------------------ #
    #  Refine through pyramid levels (coarse -> fine)                     #
    # ------------------------------------------------------------------ #
    for lvl in range(coarsest - 1, -1, -1):
        lvl_img_ang = img_ang_pyr[lvl]
        lvl_data = (
            model.pyramid[lvl]
            if lvl < len(model.pyramid)
            else model.pyramid[-1]
        )
        lvl_pts = lvl_data["contour_points"]
        lvl_angs = lvl_data["contour_angles"]

        if len(lvl_pts) == 0:
            # Propagate candidates to the next (finer) level by doubling
            # spatial coordinates.
            candidates = [
                (sc, r * 2.0, c * 2.0, a, s)
                for sc, r, c, a, s in candidates
            ]
            continue

        lvl_pts_arr = np.array(lvl_pts, dtype=np.float64)
        lvl_scale = 1.0 / (2 ** lvl)
        lvl_origin = (
            model.origin[0] * lvl_scale,
            model.origin[1] * lvl_scale,
        )

        # Angle refinement step for this level.
        lvl_angle_step = _DEFAULT_ANGLE_STEP * (2 ** lvl)
        search_radius = 3
        lvl_threshold = min_score * (0.5 + 0.5 * (coarsest - lvl) / max(coarsest, 1))

        refined: List[Tuple[float, float, float, float, float]] = []
        for _, r, c, a, s in candidates:
            # Up-scale position from the previous (coarser) level.
            base_r = r * 2.0
            base_c = c * 2.0

            best_sc = 0.0
            best_r, best_c, best_a, best_s = base_r, base_c, a, s

            for dr in range(-search_radius, search_radius + 1):
                for dc in range(-search_radius, search_radius + 1):
                    for da_mult in (-1, 0, 1):
                        da = a + da_mult * lvl_angle_step
                        cur_sc = _score_candidate(
                            lvl_img_ang,
                            lvl_pts_arr,
                            lvl_angs,
                            base_r + dr,
                            base_c + dc,
                            da,
                            s,
                            lvl_origin,
                            greediness,
                            lvl_threshold,
                        )
                        if cur_sc > best_sc:
                            best_sc = cur_sc
                            best_r = base_r + dr
                            best_c = base_c + dc
                            best_a = da
                            best_s = s

            if best_sc >= lvl_threshold:
                refined.append((best_sc, best_r, best_c, best_a, best_s))

        candidates = refined
        if not candidates:
            break

        # Sort and trim.
        candidates.sort(key=lambda t: t[0], reverse=True)
        candidates = candidates[: max(num_matches * 10, 50)]

    if not candidates:
        return []

    # ------------------------------------------------------------------ #
    #  Sub-pixel refinement at the finest level                           #
    # ------------------------------------------------------------------ #
    finest_img_ang = img_ang_pyr[0]
    finest_data = model.pyramid[0] if model.pyramid else {
        "contour_points": model.contour_points,
        "contour_angles": model.contour_angles,
    }
    f_pts = finest_data["contour_points"]
    f_angs = finest_data["contour_angles"]

    sub_refined: List[Tuple[float, float, float, float, float]] = []

    if len(f_pts) > 0 and sub_pixel != "none":
        f_pts_arr = np.array(f_pts, dtype=np.float64)
        f_origin = model.origin

        for sc, r, c, a, s in candidates:
            # Parabola fitting: evaluate at -0.5, 0, +0.5 offsets along
            # each axis independently and fit a quadratic to locate the
            # sub-pixel peak.
            scores_r: List[float] = []
            for dr in (-0.5, 0.0, 0.5):
                scores_r.append(_score_candidate(
                    finest_img_ang, f_pts_arr, f_angs,
                    r + dr, c, a, s, f_origin, 0.0, 0.0,
                ))

            scores_c: List[float] = []
            for dc in (-0.5, 0.0, 0.5):
                scores_c.append(_score_candidate(
                    finest_img_ang, f_pts_arr, f_angs,
                    r, c + dc, a, s, f_origin, 0.0, 0.0,
                ))

            best_r = r
            denom_r = 2.0 * (scores_r[0] - 2.0 * scores_r[1] + scores_r[2])
            if abs(denom_r) > 1e-12:
                offset_r = 0.5 * (scores_r[0] - scores_r[2]) / denom_r
                best_r = r + max(-0.5, min(0.5, offset_r))

            best_c = c
            denom_c = 2.0 * (scores_c[0] - 2.0 * scores_c[1] + scores_c[2])
            if abs(denom_c) > 1e-12:
                offset_c = 0.5 * (scores_c[0] - scores_c[2]) / denom_c
                best_c = c + max(-0.5, min(0.5, offset_c))

            # Re-score at the refined position.
            best_sc = _score_candidate(
                finest_img_ang, f_pts_arr, f_angs,
                best_r, best_c, a, s, f_origin, 0.0, 0.0,
            )
            sub_refined.append((best_sc, best_r, best_c, a, s))
    else:
        sub_refined = list(candidates)

    # ------------------------------------------------------------------ #
    #  Filter by min_score                                                #
    # ------------------------------------------------------------------ #
    candidates = [
        (sc, r, c, a, s) for sc, r, c, a, s in sub_refined if sc >= min_score
    ]
    if not candidates:
        return []

    candidates.sort(key=lambda t: t[0], reverse=True)

    # ------------------------------------------------------------------ #
    #  Non-maximum suppression (distance-based)                           #
    # ------------------------------------------------------------------ #
    if max_overlap < 1.0:
        bw = model.bounding_box[2]
        bh = model.bounding_box[3]
        suppress_dist = math.sqrt(bw * bw + bh * bh) * (1.0 - max_overlap)

        suppressed: List[Tuple[float, float, float, float, float]] = []
        for cand in candidates:
            sc, r, c, a, s = cand
            too_close = False
            for kept in suppressed:
                dr = r - kept[1]
                dc = c - kept[2]
                if math.sqrt(dr * dr + dc * dc) < suppress_dist:
                    too_close = True
                    break
            if not too_close:
                suppressed.append(cand)
        candidates = suppressed

    # ------------------------------------------------------------------ #
    #  Build results                                                      #
    # ------------------------------------------------------------------ #
    results: List[MatchResult] = []
    for sc, r, c, a, s in candidates[:num_matches]:
        results.append(MatchResult(row=r, col=c, angle=a, scale=s, score=sc))

    logger.info(
        "find_shape_model: found %d match(es), best score=%.3f",
        len(results),
        results[0].score if results else 0.0,
    )
    return results


# ====================================================================== #
#  Public API -- visualisation                                            #
# ====================================================================== #


@log_operation(logger)
def draw_shape_matches(
    image: np.ndarray,
    matches: List[MatchResult],
    model: Optional[ShapeModel] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw match results on top of the image.

    For each match a crosshair is drawn at the match position.  If *model*
    is provided, the model's bounding box is drawn rotated and scaled at
    the match location.  The match score is labelled next to each marker.

    Parameters:
        image:     Background image (will not be modified in place).
        matches:   List of :class:`MatchResult` from
                   :func:`find_shape_model`.
        model:     Optional shape model used to draw the rotated bounding
                   box overlay.
        color:     BGR colour tuple for annotations.
        thickness: Line thickness in pixels.

    Returns:
        Annotated copy of the image as a BGR uint8 array.
    """
    validate_image(image, "image")

    vis = image.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    cross_size = 15

    for m in matches:
        cx_int = int(round(m.col))
        cy_int = int(round(m.row))

        # Crosshair.
        cv2.line(
            vis,
            (cx_int - cross_size, cy_int),
            (cx_int + cross_size, cy_int),
            color,
            thickness,
        )
        cv2.line(
            vis,
            (cx_int, cy_int - cross_size),
            (cx_int, cy_int + cross_size),
            color,
            thickness,
        )

        # Rotated bounding box.
        if model is not None:
            bx, by, bw, bh = model.bounding_box
            ox, oy = model.origin

            # Corners relative to model origin.
            corners = np.array(
                [
                    [bx - ox, by - oy],
                    [bx + bw - ox, by - oy],
                    [bx + bw - ox, by + bh - oy],
                    [bx - ox, by + bh - oy],
                ],
                dtype=np.float64,
            )
            corners *= m.scale

            cos_a = math.cos(m.angle)
            sin_a = math.sin(m.angle)
            rot_corners = np.empty_like(corners)
            rot_corners[:, 0] = corners[:, 0] * cos_a - corners[:, 1] * sin_a
            rot_corners[:, 1] = corners[:, 0] * sin_a + corners[:, 1] * cos_a

            rot_corners[:, 0] += m.col
            rot_corners[:, 1] += m.row

            pts_draw = rot_corners.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(
                vis, [pts_draw], isClosed=True, color=color, thickness=thickness,
            )

        # Score label.
        label = f"{m.score:.3f}"
        cv2.putText(
            vis,
            label,
            (cx_int + cross_size + 4, cy_int - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    return vis
