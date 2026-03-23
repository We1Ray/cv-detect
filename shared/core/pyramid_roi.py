"""Pyramid ROI and shape model inspection.

Provides automatic region-of-interest determination from shape models at
multiple Gaussian pyramid levels, enabling efficient coarse-to-fine search
with progressive ROI refinement.

Key components
--------------
- ``PyramidROI`` -- dataclass mapping each pyramid level to its ROI bounds.
- ``inspect_shape_model`` -- analyse a shape model to determine the effective
  ROI at every pyramid level.
- ``auto_roi_from_model`` -- compute an optimal search ROI based on model extents.
- ``restrict_search_roi`` -- narrow the fine-level ROI using a coarse-level match.
- ``pyramid_inspect`` -- full coarse-to-fine search with automatic ROI refinement.
- ``gen_pyramid_rois`` -- pre-compute scaled ROIs for all levels.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, replace as dataclass_replace
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from shared.validation import validate_image
from shared.op_logger import log_operation

logger = logging.getLogger(__name__)

# ====================================================================== #
#  Type aliases                                                           #
# ====================================================================== #

# (x, y, width, height) -- all in pixel coordinates at the given level.
ROIBounds = Tuple[int, int, int, int]


# ====================================================================== #
#  Data classes                                                           #
# ====================================================================== #

@dataclass
class PyramidROI:
    """ROI bounds at each pyramid level.

    Attributes:
        level_rois: Mapping from pyramid level (0 = finest) to
                    ``(x, y, width, height)`` in that level's coordinate
                    system.
    """

    level_rois: Dict[int, ROIBounds] = field(default_factory=dict)

    @property
    def num_levels(self) -> int:
        return len(self.level_rois)

    def roi_at(self, level: int) -> ROIBounds:
        """Return ROI bounds for the given level.

        Raises ``KeyError`` if *level* is not present.
        """
        return self.level_rois[level]

    def to_dict(self) -> Dict[str, object]:
        return {
            "num_levels": self.num_levels,
            "level_rois": {str(k): list(v) for k, v in self.level_rois.items()},
        }


# ====================================================================== #
#  Internal helpers                                                       #
# ====================================================================== #

def _clamp_roi(
    roi: ROIBounds,
    img_w: int,
    img_h: int,
) -> ROIBounds:
    """Clamp an ROI to the image boundaries."""
    x, y, w, h = roi
    x = max(0, x)
    y = max(0, y)
    w = min(w, img_w - x)
    h = min(h, img_h - y)
    return (x, y, max(w, 1), max(h, 1))


def _scale_roi(roi: ROIBounds, factor: float) -> ROIBounds:
    """Scale an ROI by *factor* (used when moving between pyramid levels)."""
    x, y, w, h = roi
    return (
        int(round(x * factor)),
        int(round(y * factor)),
        max(1, int(round(w * factor))),
        max(1, int(round(h * factor))),
    )


def _expand_roi(roi: ROIBounds, margin: int) -> ROIBounds:
    """Expand an ROI by *margin* pixels on every side."""
    x, y, w, h = roi
    return (x - margin, y - margin, w + 2 * margin, h + 2 * margin)


# ====================================================================== #
#  Public API                                                             #
# ====================================================================== #

@log_operation(logger)
def inspect_shape_model(
    shape_model: object,
    image: np.ndarray,
) -> PyramidROI:
    """Analyse a shape model to determine the effective ROI at each level.

    The model is expected to have the attributes ``bounding_box``,
    ``origin``, and ``num_levels`` (as produced by
    :func:`shared.core.shape_matching.create_shape_model`).

    Parameters
    ----------
    shape_model : ShapeModel
        A shape model with ``bounding_box``, ``origin``, and ``num_levels``.
    image : ndarray
        The reference image used to clamp ROI bounds.

    Returns
    -------
    PyramidROI
        ROI bounds at each pyramid level.
    """
    validate_image(image, "image")

    bbox: Tuple[int, int, int, int] = getattr(shape_model, "bounding_box", (0, 0, 0, 0))
    num_levels: int = getattr(shape_model, "num_levels", 1)
    img_h, img_w = image.shape[:2]

    # The base ROI is the model's bounding box with a margin.
    margin = max(bbox[2], bbox[3]) // 4
    base_roi = _expand_roi(bbox, margin)
    base_roi = _clamp_roi(base_roi, img_w, img_h)

    rois: Dict[int, ROIBounds] = {}
    for level in range(num_levels):
        scale = 1.0 / (2 ** level)
        lvl_w = max(1, int(round(img_w * scale)))
        lvl_h = max(1, int(round(img_h * scale)))
        lvl_roi = _scale_roi(base_roi, scale)
        lvl_roi = _clamp_roi(lvl_roi, lvl_w, lvl_h)
        rois[level] = lvl_roi

    result = PyramidROI(level_rois=rois)
    logger.info(
        "inspect_shape_model: %d levels, base ROI=%s",
        num_levels, base_roi,
    )
    return result


@log_operation(logger)
def auto_roi_from_model(shape_model: object) -> ROIBounds:
    """Compute an optimal search ROI based on a shape model's extents.

    The returned ROI is centred on the model origin and padded by half
    the model's bounding-box dimensions to allow for positional variance.

    Parameters
    ----------
    shape_model : ShapeModel
        A shape model with ``bounding_box`` and ``origin``.

    Returns
    -------
    ROIBounds
        ``(x, y, width, height)`` in the original (finest) coordinate system.
    """
    bbox: Tuple[int, int, int, int] = getattr(shape_model, "bounding_box", (0, 0, 0, 0))
    origin: Tuple[float, float] = getattr(shape_model, "origin", (0.0, 0.0))

    bx, by, bw, bh = bbox
    ox, oy = origin

    # Pad by 50 % of model extent on each side.
    pad_x = bw // 2
    pad_y = bh // 2
    roi_x = bx - pad_x
    roi_y = by - pad_y
    roi_w = bw + 2 * pad_x
    roi_h = bh + 2 * pad_y

    logger.info(
        "auto_roi_from_model: origin=(%.1f, %.1f), bbox=%s -> ROI=(%d, %d, %d, %d)",
        ox, oy, bbox, roi_x, roi_y, roi_w, roi_h,
    )
    return (roi_x, roi_y, roi_w, roi_h)


@log_operation(logger)
def restrict_search_roi(
    image: np.ndarray,
    model: object,
    coarse_result: object,
    margin_factor: float = 0.3,
) -> ROIBounds:
    """Use a coarse-level match to restrict the fine-level search area.

    Given a match result from a coarser pyramid level (with ``row``, ``col``,
    ``scale`` attributes), this function computes a tight ROI around the
    predicted object position at the finest resolution.

    Parameters
    ----------
    image : ndarray
        The full-resolution image (used for boundary clamping).
    model : ShapeModel
        The shape model (for ``bounding_box``).
    coarse_result : MatchResult
        A match result with ``row``, ``col``, ``angle``, ``scale`` attributes.
    margin_factor : float
        Extra margin as a fraction of the model extent (default 0.3).

    Returns
    -------
    ROIBounds
        ``(x, y, width, height)`` in original-resolution coordinates.
    """
    validate_image(image, "image")

    bbox: Tuple[int, int, int, int] = getattr(model, "bounding_box", (0, 0, 0, 0))
    bw, bh = bbox[2], bbox[3]

    match_col: float = getattr(coarse_result, "col", 0.0)
    match_row: float = getattr(coarse_result, "row", 0.0)
    match_scale: float = getattr(coarse_result, "scale", 1.0)

    # Effective extent of the model at the matched scale.
    eff_w = int(round(bw * match_scale))
    eff_h = int(round(bh * match_scale))

    margin_x = max(1, int(round(eff_w * margin_factor)))
    margin_y = max(1, int(round(eff_h * margin_factor)))

    roi_x = int(round(match_col)) - eff_w // 2 - margin_x
    roi_y = int(round(match_row)) - eff_h // 2 - margin_y
    roi_w = eff_w + 2 * margin_x
    roi_h = eff_h + 2 * margin_y

    img_h, img_w = image.shape[:2]
    roi = _clamp_roi((roi_x, roi_y, roi_w, roi_h), img_w, img_h)

    logger.info(
        "restrict_search_roi: coarse match at (%.1f, %.1f) -> ROI=%s",
        match_col, match_row, roi,
    )
    return roi


@log_operation(logger)
def pyramid_inspect(
    image: np.ndarray,
    model: object,
    search_roi: Optional[ROIBounds] = None,
    min_score: float = 0.5,
    greediness: float = 0.9,
) -> List[object]:
    """Full coarse-to-fine search with automatic ROI refinement.

    Delegates to :func:`shared.core.shape_matching.find_shape_model` at each
    pyramid level, progressively narrowing the search ROI using the
    best match from the previous (coarser) level.

    Parameters
    ----------
    image : ndarray
        Search image (grayscale or BGR).
    model : ShapeModel
        Shape model created by
        :func:`shared.core.shape_matching.create_shape_model`.
    search_roi : ROIBounds, optional
        Initial search region at the finest level.  ``None`` searches the
        full image.
    min_score : float
        Minimum acceptable match score.
    greediness : float
        Greedy early-termination parameter in ``[0, 1]``.

    Returns
    -------
    list of MatchResult
        Matches found at the finest level.
    """
    validate_image(image, "image")

    # Lazy import to avoid circular dependency.
    from shared.core.shape_matching import find_shape_model

    num_levels: int = getattr(model, "num_levels", 1)
    img_h, img_w = image.shape[:2]

    # If no ROI given, use auto ROI or full image.
    if search_roi is None:
        search_roi = (0, 0, img_w, img_h)

    # Pre-compute the ROI at each level.
    proi = gen_pyramid_rois((img_w, img_h), num_levels, search_roi)

    coarsest = num_levels - 1
    best_coarse = None

    # Search from coarsest to finest.
    for level in range(coarsest, -1, -1):
        roi = proi.roi_at(level)
        scale = 1.0 / (2 ** level)
        lvl_w = max(1, int(round(img_w * scale)))
        lvl_h = max(1, int(round(img_h * scale)))

        # Downsample image to this level.
        lvl_img = image
        for _ in range(level):
            lvl_img = cv2.pyrDown(lvl_img)

        # Extract the ROI sub-image.
        rx, ry, rw, rh = _clamp_roi(roi, lvl_w, lvl_h)
        sub_img = lvl_img[ry : ry + rh, rx : rx + rw]

        if sub_img.size == 0:
            logger.warning("pyramid_inspect: empty sub-image at level %d", level)
            continue

        # Run the shape matcher on the sub-image.
        matches = find_shape_model(
            sub_img,
            model,
            min_score=min_score * 0.7 if level > 0 else min_score,
            num_matches=1,
            greediness=greediness,
        )

        if not matches:
            logger.info("pyramid_inspect: no match at level %d", level)
            if level > 0:
                continue
            return []

        # Translate match coordinates back to full-image coordinates at
        # the current level.
        best = matches[0]
        # Adjust row/col to account for the ROI offset and level scaling.
        adjusted_col = (best.col + rx) / scale
        adjusted_row = (best.row + ry) / scale

        if level > 0 and best_coarse is None:
            best_coarse = best
            # Restrict the next level's ROI using this coarse match.
            # Use dataclass replace if available, otherwise fall back to type constructor.
            if hasattr(best, '__dataclass_fields__'):
                adjusted_match = dataclass_replace(
                    best,
                    row=adjusted_row,
                    col=adjusted_col,
                    score=best.score,
                )
            else:
                adjusted_match = type(best)(
                    row=adjusted_row,
                    col=adjusted_col,
                    angle=best.angle,
                    scale=best.scale,
                    score=best.score,
                )
            fine_roi = restrict_search_roi(image, model, adjusted_match)
            proi = gen_pyramid_rois((img_w, img_h), level, fine_roi)

    # Final pass on the full-resolution image within the restricted ROI.
    final_roi = proi.roi_at(0) if 0 in proi.level_rois else search_roi
    rx, ry, rw, rh = _clamp_roi(final_roi, img_w, img_h)
    sub_img = image[ry : ry + rh, rx : rx + rw]

    if sub_img.size == 0:
        return []

    final_matches = find_shape_model(
        sub_img,
        model,
        min_score=min_score,
        num_matches=1,
        greediness=greediness,
    )

    # Translate coordinates back to full image.
    from shared.core.shape_matching import MatchResult
    results: List[MatchResult] = []
    for m in final_matches:
        results.append(MatchResult(
            row=m.row + ry,
            col=m.col + rx,
            angle=m.angle,
            scale=m.scale,
            score=m.score,
        ))

    logger.info(
        "pyramid_inspect: %d match(es), best score=%.3f",
        len(results), results[0].score if results else 0.0,
    )
    return results


@log_operation(logger)
def gen_pyramid_rois(
    image_size: Tuple[int, int],
    levels: int,
    roi: ROIBounds,
) -> PyramidROI:
    """Pre-compute ROI bounds at each pyramid level with proper scaling.

    Parameters
    ----------
    image_size : tuple of int
        ``(width, height)`` of the original image.
    levels : int
        Total number of pyramid levels (``>= 1``).
    roi : ROIBounds
        ROI at the finest (level 0) resolution.

    Returns
    -------
    PyramidROI
        ROI bounds for every level from ``0`` to ``levels - 1``.
    """
    img_w, img_h = image_size
    if levels < 1:
        raise ValueError(f"levels must be >= 1, got {levels}.")

    base_roi = _clamp_roi(roi, img_w, img_h)
    rois: Dict[int, ROIBounds] = {0: base_roi}

    for level in range(1, levels):
        scale = 1.0 / (2 ** level)
        lvl_w = max(1, int(round(img_w * scale)))
        lvl_h = max(1, int(round(img_h * scale)))
        lvl_roi = _scale_roi(base_roi, scale)
        lvl_roi = _clamp_roi(lvl_roi, lvl_w, lvl_h)
        rois[level] = lvl_roi

    result = PyramidROI(level_rois=rois)
    logger.info(
        "gen_pyramid_rois: %d levels, base ROI=%s",
        levels, base_roi,
    )
    return result
