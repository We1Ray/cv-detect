"""Region operations for vision-style image processing.

Provides threshold, binary_threshold, connection, select_shape,
compute_region_properties, and region_to_display_image functions
that operate on images and Region objects.
"""
from __future__ import annotations

import math
from typing import List, Optional, Set, Tuple

import cv2
import numpy as np

from dl_anomaly.core.region import Region, RegionProperties


# ------------------------------------------------------------------
# Threshold operations
# ------------------------------------------------------------------

def threshold(
    image: np.ndarray,
    min_gray: int,
    max_gray: int,
) -> Region:
    """Manual gray-value threshold.  Returns a Region with one label
    for each connected component whose pixels fall within
    [min_gray, max_gray].

    The input *image* should be single-channel uint8.
    """
    gray = _ensure_gray_uint8(image)
    mask = cv2.inRange(gray, int(min_gray), int(max_gray))
    return _mask_to_region(mask, source_image=gray)


def binary_threshold(
    image: np.ndarray,
    method: str = "otsu",
    block_size: int = 11,
    c_value: int = 2,
) -> Region:
    """Automatic binary threshold.

    Parameters
    ----------
    image : ndarray
        Single-channel or BGR image.
    method : str
        ``"otsu"`` or ``"adaptive"``.
    block_size : int
        Block size for adaptive thresholding (must be odd and > 1).
    c_value : int
        Constant subtracted from the mean for adaptive thresholding.
    """
    gray = _ensure_gray_uint8(image)
    if method == "otsu":
        _, mask = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
    elif method == "adaptive":
        if block_size % 2 == 0:
            block_size += 1
        if block_size < 3:
            block_size = 3
        mask = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            c_value,
        )
    else:
        raise ValueError(f"Unknown threshold method: {method}")
    return _mask_to_region(mask, source_image=gray)


# ------------------------------------------------------------------
# Connection (connected-component labeling)
# ------------------------------------------------------------------

def connection(region: Region) -> Region:
    """Re-label a region so every connected component gets its own label.

    This is analogous to the ``connection`` operator in industrial vision.
    """
    mask = region.to_binary_mask()
    num, labels = cv2.connectedComponents(mask, connectivity=8)
    labels = labels.astype(np.int32)
    n = num - 1  # exclude background label 0
    props = compute_region_properties(labels, region.source_image)
    return Region(
        labels=labels,
        num_regions=n,
        properties=props,
        source_image=region.source_image,
        source_shape=region.source_shape or mask.shape[:2],
    )


# ------------------------------------------------------------------
# select_shape (feature-based filtering)
# ------------------------------------------------------------------

def select_shape(
    region: Region,
    feature: str,
    min_val: float,
    max_val: float,
) -> Region:
    """Keep only those connected regions whose *feature* value lies in
    [min_val, max_val].

    Delegates to ``Region.filter_by`` but recomputes properties if they
    are missing.
    """
    if not region.properties and region.num_regions > 0:
        props = compute_region_properties(region.labels, region.source_image)
        region = Region(
            labels=region.labels,
            num_regions=region.num_regions,
            properties=props,
            source_image=region.source_image,
            source_shape=region.source_shape,
        )
    return region.filter_by(feature, min_val, max_val)


# ------------------------------------------------------------------
# compute_region_properties
# ------------------------------------------------------------------

def compute_region_properties(
    labels: np.ndarray,
    source_image: Optional[np.ndarray] = None,
) -> List[RegionProperties]:
    """Compute geometric and photometric properties for every region
    in *labels* (1-based labeling, 0 = background).

    Uses ``cv2.connectedComponentsWithStats`` for bulk stats when possible,
    and only falls back to per-region contour analysis for regions that need
    detailed shape descriptors.
    """
    unique_labels = [int(v) for v in np.unique(labels) if v != 0]
    if not unique_labels:
        return []

    gray: Optional[np.ndarray] = None
    if source_image is not None:
        gray = _ensure_gray_uint8(source_image)

    # Pre-compute bulk stats from the labels array itself.
    # connectedComponentsWithStats gives us area, bbox, centroid in O(H*W).
    binary = (labels > 0).astype(np.uint8) * 255
    num_cc, cc_labels, cc_stats, cc_centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8,
    )

    # Build a map from our label value -> CC stat index
    # Since our labels and cc_labels may differ, build a lookup.
    # For each cc index (1..num_cc-1), find the dominant original label.
    cc_to_orig: dict = {}
    for cc_idx in range(1, num_cc):
        # Sample a pixel belonging to this CC to get the original label
        ys, xs = np.where(cc_labels == cc_idx)
        if len(ys) == 0:
            continue
        orig_label = int(labels[ys[0], xs[0]])
        if orig_label > 0:
            cc_to_orig[cc_idx] = orig_label

    # Build a map from original label -> cc_stats row for fast lookup
    orig_to_cc: dict = {}
    for cc_idx, orig_label in cc_to_orig.items():
        orig_to_cc.setdefault(orig_label, []).append(cc_idx)

    # For gray-value stats, compute a flat lookup for speed
    gray_flat = gray.ravel() if gray is not None else None
    labels_flat = labels.ravel()

    props_list: List[RegionProperties] = []

    for idx in unique_labels:
        # Use CC stats if available for this label
        cc_indices = orig_to_cc.get(idx, [])

        # Fast path: get area and bbox from CC stats
        if cc_indices:
            # Merge stats from all CCs belonging to this label
            total_area = sum(int(cc_stats[ci, cv2.CC_STAT_AREA]) for ci in cc_indices)
            if total_area == 0:
                continue
            # Use the CC with the largest area for bbox/centroid
            main_cc = max(cc_indices, key=lambda ci: cc_stats[ci, cv2.CC_STAT_AREA])
            bx = int(cc_stats[main_cc, cv2.CC_STAT_LEFT])
            by = int(cc_stats[main_cc, cv2.CC_STAT_TOP])
            bw = int(cc_stats[main_cc, cv2.CC_STAT_WIDTH])
            bh = int(cc_stats[main_cc, cv2.CC_STAT_HEIGHT])
            cx = float(cc_centroids[main_cc, 0])
            cy = float(cc_centroids[main_cc, 1])
            area = total_area
        else:
            # Fallback: compute from mask directly
            mask = (labels == idx).astype(np.uint8)
            area = int(mask.sum())
            if area == 0:
                continue
            ys, xs = np.where(mask)
            bx, by = int(xs.min()), int(ys.min())
            bw = int(xs.max()) - bx + 1
            bh = int(ys.max()) - by + 1
            cx = float(xs.mean())
            cy = float(ys.mean())

        # Extract contour only from the ROI (bounding box) for speed
        roi_mask = (labels[by:by+bh, bx:bx+bw] == idx).astype(np.uint8)
        contours, _ = cv2.findContours(
            roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
            offset=(bx, by),
        )

        if contours:
            cnt = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(cnt, True)

            # Circularity
            if perimeter > 0:
                circularity = min(4.0 * math.pi * area / (perimeter * perimeter), 1.0)
            else:
                circularity = 0.0

            # Convexity
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            convexity = area / hull_area if hull_area > 0 else 0.0

            # Orientation
            if len(cnt) >= 5:
                try:
                    _, _, angle = cv2.fitEllipse(cnt)
                except cv2.error:
                    angle = 0.0
            else:
                angle = 0.0
        else:
            perimeter = 0.0
            circularity = 0.0
            convexity = 0.0
            angle = 0.0

        # Rectangularity
        bbox_area = bw * bh
        rectangularity = area / bbox_area if bbox_area > 0 else 0.0

        # Aspect ratio
        min_side = min(bw, bh)
        aspect_ratio = max(bw, bh) / min_side if min_side > 0 else 1.0

        # Compactness
        compactness = (perimeter * perimeter) / area if area > 0 else 0.0

        # Gray-value statistics — use vectorized indexing
        mean_value = 0.0
        min_value = 0.0
        max_value = 0.0
        if gray_flat is not None:
            region_pixels = gray_flat[labels_flat == idx]
            if len(region_pixels) > 0:
                mean_value = float(region_pixels.mean())
                min_value = float(region_pixels.min())
                max_value = float(region_pixels.max())

        props_list.append(RegionProperties(
            index=idx,
            area=area,
            centroid=(cx, cy),
            bbox=(bx, by, bw, bh),
            width=bw,
            height=bh,
            circularity=round(circularity, 4),
            rectangularity=round(rectangularity, 4),
            aspect_ratio=round(aspect_ratio, 4),
            compactness=round(compactness, 4),
            convexity=round(convexity, 4),
            perimeter=round(perimeter, 2),
            orientation=round(angle, 2),
            mean_value=round(mean_value, 2),
            min_value=round(min_value, 2),
            max_value=round(max_value, 2),
        ))

    return props_list


# ------------------------------------------------------------------
# region_to_display_image  (visualisation helper)
# ------------------------------------------------------------------

def region_to_display_image(
    region: Region,
    source_image: Optional[np.ndarray] = None,
    *,
    show_labels: bool = True,
    show_bbox: bool = True,
    show_cross: bool = True,
    alpha: float = 0.45,
    highlight_indices: Optional[List[int]] = None,
    highlight_color: Tuple[int, int, int] = (255, 255, 0),
) -> np.ndarray:
    """Render *region* overlaid on *source_image*.

    Each region is coloured with a distinct hue; optionally draw
    bounding boxes, centroid crosses and label numbers.

    Uses a vectorized colour LUT instead of per-region pixel masking
    for significantly better performance with many regions.

    Parameters
    ----------
    highlight_indices : list[int], optional
        1-based region indices to render with *highlight_color*.
    highlight_color : tuple
        BGR colour for highlighted regions.

    Returns
    -------
    np.ndarray
        BGR uint8 image with overlays.
    """
    if source_image is not None:
        base = source_image.copy()
    elif region.source_image is not None:
        base = region.source_image.copy()
    else:
        h, w = region.labels.shape[:2]
        base = np.zeros((h, w, 3), dtype=np.uint8)

    if base.dtype != np.uint8:
        mn, mx = base.min(), base.max()
        if mx - mn > 0:
            base = ((base - mn) / (mx - mn) * 255).astype(np.uint8)
        else:
            base = np.zeros_like(base, dtype=np.uint8)

    if base.ndim == 2:
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    elif base.ndim == 3 and base.shape[2] == 4:
        base = cv2.cvtColor(base, cv2.COLOR_BGRA2BGR)

    hl_set: Set[int] = set(highlight_indices) if highlight_indices else set()
    n = region.num_regions

    # Build colour LUT: index 0 = background (transparent), 1..N = region colours
    # LUT shape: (N+1, 3), dtype uint8
    lut = np.zeros((n + 1, 3), dtype=np.uint8)
    # Pre-compute all hues at once
    if n > 0:
        hues = ((np.arange(n) * 180) // max(n, 1)) % 180
        hsv_strip = np.zeros((1, n, 3), dtype=np.uint8)
        hsv_strip[0, :, 0] = hues.astype(np.uint8)
        hsv_strip[0, :, 1] = 200
        hsv_strip[0, :, 2] = 220
        bgr_strip = cv2.cvtColor(hsv_strip, cv2.COLOR_HSV2BGR)
        lut[1:] = bgr_strip[0]  # shape (N, 3)

    # Override highlight colours
    for hi in hl_set:
        if 1 <= hi <= n:
            lut[hi] = highlight_color

    # Vectorised overlay: use labels as index into LUT
    labels_clipped = np.clip(region.labels, 0, n).astype(np.int32)
    overlay = lut[labels_clipped]  # (H, W, 3)

    # Blend only foreground pixels
    fg_mask = (labels_clipped > 0)
    result = base.copy()
    if np.any(fg_mask):
        result[fg_mask] = cv2.addWeighted(
            overlay[fg_mask].reshape(-1, 1, 3), alpha,
            base[fg_mask].reshape(-1, 1, 3), 1 - alpha, 0,
        ).reshape(-1, 3)

    # Draw annotations on top of the blended image
    # Pre-compute annotation colours from the same LUT
    for p in region.properties:
        bx, by, bw, bh = p.bbox
        cx_i, cy_i = int(p.centroid[0]), int(p.centroid[1])

        idx = min(p.index, n)
        ann_color: Tuple[int, ...] = tuple(int(c) for c in lut[idx])

        if show_bbox:
            cv2.rectangle(result, (bx, by), (bx + bw, by + bh), ann_color, 1)

        if show_cross:
            arm = 6
            cv2.line(
                result, (cx_i - arm, cy_i), (cx_i + arm, cy_i), ann_color, 1,
            )
            cv2.line(
                result, (cx_i, cy_i - arm), (cx_i, cy_i + arm), ann_color, 1,
            )

        if show_labels:
            cv2.putText(
                result, str(p.index),
                (bx, by - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
                cv2.LINE_AA,
            )

    return result


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _ensure_gray_uint8(image: np.ndarray) -> np.ndarray:
    """Convert arbitrary image to single-channel uint8."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    if gray.dtype != np.uint8:
        mn, mx = gray.min(), gray.max()
        if mx - mn > 0:
            gray = ((gray - mn) / (mx - mn) * 255).astype(np.uint8)
        else:
            gray = np.zeros_like(gray, dtype=np.uint8)
    return gray


def _mask_to_region(
    mask: np.ndarray,
    source_image: Optional[np.ndarray] = None,
) -> Region:
    """Convert a binary uint8 mask to a Region with connected-component
    labeling and properties."""
    num, labels = cv2.connectedComponents(mask, connectivity=8)
    labels = labels.astype(np.int32)
    n = num - 1
    props = compute_region_properties(labels, source_image)
    return Region(
        labels=labels,
        num_regions=n,
        properties=props,
        source_image=source_image,
        source_shape=mask.shape[:2],
    )
