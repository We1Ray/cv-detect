"""Region data structures for HALCON-style region operations.

Provides Region and RegionProperties dataclasses that model labeled
connected-component regions with rich geometric and photometric properties.
Supports set operations (union, intersection, difference, complement),
filtering by feature values, and visualization helpers.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class RegionProperties:
    """Properties of a single connected region.

    Attributes:
        index: 1-based region label inside the parent labels array.
        area: Total pixel count belonging to this region.
        centroid: Center of mass as (cx, cy).
        bbox: Bounding box as (x, y, w, h).
        width: Bounding box width.
        height: Bounding box height.
        circularity: Isoperimetric quotient 4*pi*area / perimeter**2.
        rectangularity: area / (bbox_w * bbox_h).
        aspect_ratio: max(w, h) / min(w, h).
        compactness: perimeter**2 / area.
        convexity: area / convex_hull_area.
        perimeter: Arc length of the outer contour.
        orientation: Angle in degrees from cv2.fitEllipse (0-180).
        mean_value: Mean gray-level intensity inside the region.
        min_value: Minimum gray-level intensity inside the region.
        max_value: Maximum gray-level intensity inside the region.
    """

    index: int
    area: int
    centroid: Tuple[float, float]
    bbox: Tuple[int, int, int, int]
    width: int
    height: int
    circularity: float
    rectangularity: float
    aspect_ratio: float
    compactness: float
    convexity: float
    perimeter: float
    orientation: float
    mean_value: float
    min_value: float
    max_value: float


@dataclass
class Region:
    """A labeled region set, analogous to HALCON's Region concept.

    ``labels`` is an int32 ndarray of shape (H, W) where 0 represents the
    background and values 1..N identify distinct connected regions.

    Attributes:
        labels: int32 array, shape (H, W).  0 = background, 1..N = regions.
        num_regions: Number of distinct foreground regions.
        properties: List of RegionProperties, one per region.
        source_image: Optional reference to the grayscale source image.
        source_shape: (H, W) tuple of the original image dimensions.
    """

    labels: np.ndarray
    num_regions: int
    properties: List[RegionProperties] = field(default_factory=list)
    source_image: Optional[np.ndarray] = None
    source_shape: Optional[Tuple[int, int]] = None

    # ------------------------------------------------------------------
    # Mask helpers
    # ------------------------------------------------------------------

    def to_binary_mask(self) -> np.ndarray:
        """Return a uint8 binary mask (0/255) of all regions combined."""
        return ((self.labels > 0) * 255).astype(np.uint8)

    def to_color_mask(self, alpha: float = 0.6) -> np.ndarray:
        """Render each region in a unique colour spread across the HSV wheel.

        Returns an RGB uint8 image of shape (H, W, 3).  Background pixels
        are set to black (0, 0, 0).
        """
        h, w = self.labels.shape[:2]
        color_img = np.zeros((h, w, 3), dtype=np.uint8)
        if self.num_regions == 0:
            return color_img

        for i in range(1, self.num_regions + 1):
            hue = int((i - 1) * 180 / max(self.num_regions, 1)) % 180
            mask = self.labels == i
            color_img[mask] = [hue, 200, 220]

        color_img = cv2.cvtColor(color_img, cv2.COLOR_HSV2BGR)
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

        # Zero out background pixels that might have been coloured.
        bg = self.labels == 0
        color_img[bg] = 0
        return color_img

    # ------------------------------------------------------------------
    # Single-region extraction
    # ------------------------------------------------------------------

    def get_single_region(self, index: int) -> "Region":
        """Extract a single region by its 1-based label *index*.

        The returned Region has ``num_regions`` equal to 1 (or 0 if the
        requested index is not present) and its label array contains only
        values 0 and 1.
        """
        new_labels = np.zeros_like(self.labels)
        new_labels[self.labels == index] = 1

        new_props: List[RegionProperties] = []
        for p in self.properties:
            if p.index == index:
                p_copy = copy.copy(p)
                p_copy.index = 1
                new_props.append(p_copy)

        return Region(
            labels=new_labels,
            num_regions=1 if np.any(new_labels) else 0,
            properties=new_props,
            source_image=self.source_image,
            source_shape=self.source_shape,
        )

    # ------------------------------------------------------------------
    # Feature-based filtering
    # ------------------------------------------------------------------

    def filter_by(
        self, feature: str, min_val: float, max_val: float
    ) -> "Region":
        """Return a new Region keeping only regions whose *feature* falls
        within [*min_val*, *max_val*].
        """
        passing: List[int] = []
        for p in self.properties:
            val = getattr(p, feature, None)
            if val is not None and min_val <= val <= max_val:
                passing.append(p.index)
        return self._keep_indices(passing)

    # ------------------------------------------------------------------
    # Set operations
    # ------------------------------------------------------------------

    def union(self, other: "Region") -> "Region":
        """Pixel-wise union of two region sets."""
        from dl_anomaly.core.region_ops import compute_region_properties

        combined = (
            ((self.labels > 0) | (other.labels > 0)).astype(np.uint8) * 255
        )
        num, new_labels = cv2.connectedComponents(combined, connectivity=8)
        new_labels = new_labels.astype(np.int32)
        props = compute_region_properties(new_labels, self.source_image)
        return Region(
            labels=new_labels,
            num_regions=num - 1,
            properties=props,
            source_image=self.source_image,
            source_shape=self.source_shape,
        )

    def intersection(self, other: "Region") -> "Region":
        """Pixel-wise intersection of two region sets."""
        from dl_anomaly.core.region_ops import compute_region_properties

        combined = (
            ((self.labels > 0) & (other.labels > 0)).astype(np.uint8) * 255
        )
        num, new_labels = cv2.connectedComponents(combined, connectivity=8)
        new_labels = new_labels.astype(np.int32)
        props = compute_region_properties(new_labels, self.source_image)
        return Region(
            labels=new_labels,
            num_regions=num - 1,
            properties=props,
            source_image=self.source_image,
            source_shape=self.source_shape,
        )

    def difference(self, other: "Region") -> "Region":
        """Pixel-wise difference: self minus other."""
        from dl_anomaly.core.region_ops import compute_region_properties

        combined = (
            ((self.labels > 0) & ~(other.labels > 0)).astype(np.uint8) * 255
        )
        num, new_labels = cv2.connectedComponents(combined, connectivity=8)
        new_labels = new_labels.astype(np.int32)
        props = compute_region_properties(new_labels, self.source_image)
        return Region(
            labels=new_labels,
            num_regions=num - 1,
            properties=props,
            source_image=self.source_image,
            source_shape=self.source_shape,
        )

    def complement(
        self, shape: Optional[Tuple[int, int]] = None
    ) -> "Region":
        """Complement region: all pixels that are *not* in any region."""
        from dl_anomaly.core.region_ops import compute_region_properties

        s = shape or self.source_shape or self.labels.shape[:2]
        inv = (self.labels == 0).astype(np.uint8) * 255
        # Crop / pad to requested shape if it differs.
        if inv.shape[:2] != s:
            canvas = np.zeros((s[0], s[1]), dtype=np.uint8)
            rh = min(s[0], inv.shape[0])
            rw = min(s[1], inv.shape[1])
            canvas[:rh, :rw] = inv[:rh, :rw]
            inv = canvas

        num, new_labels = cv2.connectedComponents(inv, connectivity=8)
        new_labels = new_labels.astype(np.int32)
        props = compute_region_properties(new_labels, self.source_image)
        return Region(
            labels=new_labels,
            num_regions=num - 1,
            properties=props,
            source_image=self.source_image,
            source_shape=self.source_shape,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _keep_indices(self, indices: List[int]) -> "Region":
        """Keep only the given label indices and re-label them 1..len(indices)."""
        new_labels = np.zeros_like(self.labels)
        new_props: List[RegionProperties] = []
        for new_idx, old_idx in enumerate(indices, 1):
            new_labels[self.labels == old_idx] = new_idx
            for p in self.properties:
                if p.index == old_idx:
                    p_copy = copy.copy(p)
                    p_copy.index = new_idx
                    new_props.append(p_copy)
        return Region(
            labels=new_labels,
            num_regions=len(indices),
            properties=new_props,
            source_image=self.source_image,
            source_shape=self.source_shape,
        )
