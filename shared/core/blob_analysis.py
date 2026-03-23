"""Advanced blob (region) analysis with Halcon-equivalent feature extraction.

Extends the basic :class:`~shared.core.region.RegionProperties` with a rich
set of shape descriptors including moments, Hu invariants, Feret diameters,
Euler number, and inscribed / circumscribed circle metrics.

Typical usage::

    from shared.core.blob_analysis import extract_blob_features, select_blobs

    features = extract_blob_features(labels, gray_image)
    large = select_blobs(features, {"area": (">", 500), "circularity": (">", 0.7)})
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class BlobFeatures:
    """Extended region properties analogous to Halcon's region features.

    All spatial quantities are expressed in **pixel** units unless otherwise
    noted.  Moment-based attributes follow the OpenCV naming convention.

    Attributes:
        index: 1-based region label.
        area: Pixel count of the region.
        centroid: Centre of mass as ``(cx, cy)``.
        bbox: Bounding box ``(x, y, w, h)``.
        perimeter: Arc length of the outer contour.
        orientation: Angle (degrees, 0-180) of the fitted ellipse major axis.
        circularity: ``4 * pi * area / perimeter**2``.
        rectangularity: ``area / (bbox_w * bbox_h)``.
        aspect_ratio: ``max(w, h) / min(w, h)``.
        compactness: ``perimeter**2 / area``.
        convexity: ``area / convex_hull_area``.
        mean_value: Mean intensity inside the region (NaN if no image given).
        min_value: Min intensity inside the region.
        max_value: Max intensity inside the region.

        # --- Ellipse ---
        elliptic_axis_ra: Semi-major axis length of the fitted ellipse.
        elliptic_axis_rb: Semi-minor axis length of the fitted ellipse.
        eccentricity: Eccentricity of the fitted ellipse (0 = circle, 1 = line).

        # --- Moments ---
        moments_m00: Raw spatial moment of order (0,0).
        moments_m10: Raw spatial moment of order (1,0).
        moments_m01: Raw spatial moment of order (0,1).
        moments_mu20: Central moment mu_20.
        moments_mu11: Central moment mu_11.
        moments_mu02: Central moment mu_02.
        moments_nu20: Normalised central moment nu_20.
        moments_nu11: Normalised central moment nu_11.
        moments_nu02: Normalised central moment nu_02.
        hu_moments: 7 Hu moment invariants as a list of floats.

        # --- Feret diameters ---
        feret_diameter_min: Minimum Feret (caliper) diameter.
        feret_diameter_max: Maximum Feret (caliper) diameter.
        feret_angle: Angle (degrees) of the maximum Feret diameter.

        # --- Topology ---
        euler_number: Euler number (connected components minus holes).

        # --- Shape ratios ---
        solidity: ``area / convex_hull_area``.
        extent: ``area / bbox_area``.
        equivalent_diameter: Diameter of a circle with the same area.

        # --- Inscribed / circumscribed circles ---
        inner_circle_radius: Radius of the largest inscribed circle.
        inner_circle_center: Centre of the largest inscribed circle.
        outer_circle_radius: Radius of the smallest enclosing circle.
        outer_circle_center: Centre of the smallest enclosing circle.
    """

    # Basic (mirroring RegionProperties) ------------------------------------
    index: int = 0
    area: int = 0
    centroid: Tuple[float, float] = (0.0, 0.0)
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    perimeter: float = 0.0
    orientation: float = 0.0
    circularity: float = 0.0
    rectangularity: float = 0.0
    aspect_ratio: float = 1.0
    compactness: float = 0.0
    convexity: float = 0.0
    mean_value: float = float("nan")
    min_value: float = float("nan")
    max_value: float = float("nan")

    # Ellipse ----------------------------------------------------------------
    elliptic_axis_ra: float = 0.0
    elliptic_axis_rb: float = 0.0
    eccentricity: float = 0.0

    # Raw moments ------------------------------------------------------------
    moments_m00: float = 0.0
    moments_m10: float = 0.0
    moments_m01: float = 0.0

    # Central moments (2nd order) -------------------------------------------
    moments_mu20: float = 0.0
    moments_mu11: float = 0.0
    moments_mu02: float = 0.0

    # Normalised central moments --------------------------------------------
    moments_nu20: float = 0.0
    moments_nu11: float = 0.0
    moments_nu02: float = 0.0

    # Hu moment invariants --------------------------------------------------
    hu_moments: List[float] = field(default_factory=lambda: [0.0] * 7)

    # Feret diameters -------------------------------------------------------
    feret_diameter_min: float = 0.0
    feret_diameter_max: float = 0.0
    feret_angle: float = 0.0

    # Topology --------------------------------------------------------------
    euler_number: int = 0

    # Shape ratios ----------------------------------------------------------
    solidity: float = 0.0
    extent: float = 0.0
    equivalent_diameter: float = 0.0

    # Inscribed / circumscribed circles -------------------------------------
    inner_circle_radius: float = 0.0
    inner_circle_center: Tuple[float, float] = (0.0, 0.0)
    outer_circle_radius: float = 0.0
    outer_circle_center: Tuple[float, float] = (0.0, 0.0)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise all features to a flat dictionary."""
        d: Dict[str, Any] = {}
        for k in self.__dataclass_fields__:
            v = getattr(self, k)
            if isinstance(v, np.ndarray):
                v = v.tolist()
            d[k] = v
        return d


# ---------------------------------------------------------------------------
# Feret diameter computation (rotating calipers)
# ---------------------------------------------------------------------------


def compute_feret_diameters(
    contour: np.ndarray,
) -> Tuple[float, float, float]:
    """Compute minimum and maximum Feret (caliper) diameters of a contour.

    Uses the convex hull and a rotating-calipers sweep to find the minimum
    and maximum projection widths.

    Parameters:
        contour: Nx1x2 or Nx2 int32 array of contour points.

    Returns:
        ``(feret_min, feret_max, feret_max_angle_degrees)``
    """
    contour = contour.reshape(-1, 2).astype(np.float64)
    if len(contour) < 2:
        return 0.0, 0.0, 0.0

    hull_indices = cv2.convexHull(contour.astype(np.float32), returnPoints=False)
    if hull_indices is None or len(hull_indices) < 2:
        return 0.0, 0.0, 0.0

    hull_pts = contour[hull_indices.ravel()]
    n = len(hull_pts)

    if n == 1:
        return 0.0, 0.0, 0.0
    if n == 2:
        d = float(np.linalg.norm(hull_pts[1] - hull_pts[0]))
        dx, dy = hull_pts[1] - hull_pts[0]
        angle = math.degrees(math.atan2(dy, dx)) % 180.0
        return d, d, angle

    # Sweep 180 degrees in 1-degree increments for robust min/max projection.
    min_width = float("inf")
    max_width = 0.0
    max_angle = 0.0

    for deg in range(180):
        theta = math.radians(deg)
        direction = np.array([math.cos(theta), math.sin(theta)])
        projections = hull_pts @ direction
        width = float(projections.max() - projections.min())
        if width < min_width:
            min_width = width
        if width > max_width:
            max_width = width
            max_angle = float(deg)

    return min_width, max_width, max_angle


# ---------------------------------------------------------------------------
# Inscribed circle (via distance transform)
# ---------------------------------------------------------------------------


def compute_inner_circle(
    binary_mask: np.ndarray,
) -> Tuple[float, float, float]:
    """Find the largest inscribed circle inside a binary region.

    Uses the distance transform to find the interior point farthest from
    the boundary.

    Parameters:
        binary_mask: uint8 binary image (0 / 255).

    Returns:
        ``(cx, cy, radius)``  -- centre and radius in pixels.  Returns
        ``(0, 0, 0)`` if the mask is empty.
    """
    if binary_mask is None or binary_mask.size == 0:
        return 0.0, 0.0, 0.0

    mask = binary_mask.astype(np.uint8)
    if mask.max() == 0:
        return 0.0, 0.0, 0.0

    # Ensure binary 0/255
    if mask.max() == 1:
        mask = mask * 255

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    _, max_val, _, max_loc = cv2.minMaxLoc(dist)
    cx, cy = float(max_loc[0]), float(max_loc[1])
    return cx, cy, float(max_val)


# ---------------------------------------------------------------------------
# Euler number
# ---------------------------------------------------------------------------


def compute_euler_number(binary_mask: np.ndarray) -> int:
    """Compute the Euler number of a binary region.

    Euler number = (number of connected components) - (number of holes).
    Uses 8-connectivity for the foreground.

    Parameters:
        binary_mask: uint8 binary image (0 / 255).

    Returns:
        The Euler number as an integer.
    """
    if binary_mask is None or binary_mask.size == 0:
        return 0

    mask = binary_mask.astype(np.uint8)
    if mask.max() == 1:
        mask = mask * 255

    # Connected components of the foreground (8-connectivity).
    num_fg, _ = cv2.connectedComponents(mask, connectivity=8)
    num_fg -= 1  # subtract background label

    # Connected components of the background holes (4-connectivity).
    inv = cv2.bitwise_not(mask)
    num_bg, labels_bg = cv2.connectedComponents(inv, connectivity=4)
    # One of the background components is the outer background -- not a hole.
    # Count only interior holes: bg components that do not touch the border.
    num_holes = 0
    h, w = mask.shape[:2]
    for i in range(1, num_bg):
        component = labels_bg == i
        # Check if this component touches any border.
        touches_border = (
            np.any(component[0, :])
            or np.any(component[-1, :])
            or np.any(component[:, 0])
            or np.any(component[:, -1])
        )
        if not touches_border:
            num_holes += 1

    return num_fg - num_holes


# ---------------------------------------------------------------------------
# Full feature extraction
# ---------------------------------------------------------------------------


def _extract_single_blob(
    label_val: int,
    labels: np.ndarray,
    image: Optional[np.ndarray],
) -> BlobFeatures:
    """Extract all blob features for a single label value."""
    mask = (labels == label_val).astype(np.uint8) * 255
    area = int(cv2.countNonZero(mask))

    if area == 0:
        return BlobFeatures(index=label_val, area=0)

    # Contour -----------------------------------------------------------------
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return BlobFeatures(index=label_val, area=area)
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, closed=True)

    # Moments -----------------------------------------------------------------
    m = cv2.moments(mask, binaryImage=True)
    if m["m00"] == 0:
        logger.debug("Skipping zero-area region %d", label_val)
        return BlobFeatures(index=label_val, area=0)
    m00 = m["m00"]
    cx = m["m10"] / m00
    cy = m["m01"] / m00
    hu = cv2.HuMoments(m).ravel().tolist()

    # Bounding box ------------------------------------------------------------
    x, y, bw, bh = cv2.boundingRect(contour)
    bbox_area = bw * bh if bw * bh > 0 else 1

    # Convex hull -------------------------------------------------------------
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    hull_area = hull_area if hull_area > 0 else 1.0

    # Ellipse -----------------------------------------------------------------
    ra, rb, orientation, eccentricity = 0.0, 0.0, 0.0, 0.0
    if len(contour) >= 5:
        try:
            (ex, ey), (ew, eh), eangle = cv2.fitEllipse(contour)
            # OpenCV fitEllipse returns (width, height) which may be in
            # either order; use max/min to get major/minor reliably.
            major_d = max(ew, eh)
            minor_d = min(ew, eh)
            ra = major_d / 2.0
            rb = minor_d / 2.0
            orientation = eangle
            if ra > 0:
                eccentricity = math.sqrt(1.0 - (rb / ra) ** 2) if rb <= ra else 0.0
        except cv2.error:
            pass

    # Circularity, rectangularity, etc. --------------------------------------
    circ = (4.0 * math.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0
    rect = area / bbox_area
    wh_min = min(bw, bh) if min(bw, bh) > 0 else 1
    asp = max(bw, bh) / wh_min
    comp = (perimeter ** 2) / area if area > 0 else 0.0
    convexity = area / hull_area
    solidity = area / hull_area
    extent = area / bbox_area
    eq_diam = math.sqrt(4.0 * area / math.pi)

    # Intensity ---------------------------------------------------------------
    mean_val, min_val, max_val = float("nan"), float("nan"), float("nan")
    if image is not None:
        roi = image[mask > 0]
        if roi.size > 0:
            mean_val = float(np.mean(roi))
            min_val = float(np.min(roi))
            max_val = float(np.max(roi))

    # Feret diameters ---------------------------------------------------------
    feret_min, feret_max, feret_angle = compute_feret_diameters(contour)

    # Inscribed circle --------------------------------------------------------
    ic_cx, ic_cy, ic_r = compute_inner_circle(mask)

    # Circumscribed (minimum enclosing) circle --------------------------------
    (oc_cx, oc_cy), oc_r = cv2.minEnclosingCircle(contour)

    # Euler number ------------------------------------------------------------
    euler = compute_euler_number(mask)

    return BlobFeatures(
        index=label_val,
        area=area,
        centroid=(cx, cy),
        bbox=(x, y, bw, bh),
        perimeter=perimeter,
        orientation=orientation,
        circularity=circ,
        rectangularity=rect,
        aspect_ratio=asp,
        compactness=comp,
        convexity=convexity,
        mean_value=mean_val,
        min_value=min_val,
        max_value=max_val,
        # Ellipse
        elliptic_axis_ra=ra,
        elliptic_axis_rb=rb,
        eccentricity=eccentricity,
        # Raw moments
        moments_m00=m["m00"],
        moments_m10=m["m10"],
        moments_m01=m["m01"],
        # Central moments
        moments_mu20=m["mu20"],
        moments_mu11=m["mu11"],
        moments_mu02=m["mu02"],
        # Normalised moments
        moments_nu20=m["nu20"],
        moments_nu11=m["nu11"],
        moments_nu02=m["nu02"],
        # Hu
        hu_moments=hu,
        # Feret
        feret_diameter_min=feret_min,
        feret_diameter_max=feret_max,
        feret_angle=feret_angle,
        # Topology
        euler_number=euler,
        # Shape ratios
        solidity=solidity,
        extent=extent,
        equivalent_diameter=eq_diam,
        # Circles
        inner_circle_radius=ic_r,
        inner_circle_center=(ic_cx, ic_cy),
        outer_circle_radius=float(oc_r),
        outer_circle_center=(float(oc_cx), float(oc_cy)),
    )


def extract_blob_features(
    labels: np.ndarray,
    image: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
) -> List[BlobFeatures]:
    """Extract all advanced blob features for every labeled region.

    Parameters:
        labels: int32 label array of shape ``(H, W)`` where 0 is background
            and values 1..N identify distinct regions.
        image: Optional single-channel grayscale image for intensity features.
        mask: Optional binary mask.  If provided, labels outside the mask
            are zeroed before feature extraction.

    Returns:
        A list of :class:`BlobFeatures`, one per region (sorted by index).
    """
    if labels is None or labels.size == 0:
        return []

    work_labels = labels.copy().astype(np.int32)

    if mask is not None:
        work_labels[mask == 0] = 0

    unique_labels = np.unique(work_labels)
    unique_labels = unique_labels[unique_labels > 0]

    features: List[BlobFeatures] = []
    for lbl in unique_labels:
        feat = _extract_single_blob(int(lbl), work_labels, image)
        features.append(feat)

    return features


# ---------------------------------------------------------------------------
# Blob selection (filtering)
# ---------------------------------------------------------------------------

_FILTER_OPS: Dict[str, Any] = {
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


def select_blobs(
    features: Sequence[BlobFeatures],
    criteria: Dict[str, Union[Tuple[str, float], Tuple[str, float, float]]],
) -> List[BlobFeatures]:
    """Filter blobs by arbitrary feature criteria.

    Parameters:
        features: List of :class:`BlobFeatures` to filter.
        criteria: Mapping of ``feature_name`` to a filter specification.

            * ``("op", value)`` -- e.g. ``(">" , 500)`` keeps blobs where
              the feature is > 500.
            * ``("between", lo, hi)`` -- keeps blobs where
              ``lo <= feature <= hi``.

    Returns:
        A new list containing only the blobs that satisfy **all** criteria.

    Example::

        big_round = select_blobs(feats, {
            "area": (">", 500),
            "circularity": (">=", 0.7),
        })
    """
    result: List[BlobFeatures] = []

    for blob in features:
        keep = True
        for feat_name, spec in criteria.items():
            val = getattr(blob, feat_name, None)
            if val is None:
                keep = False
                break

            if spec[0] == "between":
                lo, hi = spec[1], spec[2]
                if not (lo <= val <= hi):
                    keep = False
                    break
            else:
                op_str, threshold = spec[0], spec[1]
                op_fn = _FILTER_OPS.get(op_str)
                if op_fn is None:
                    raise ValueError(
                        f"Unknown operator '{op_str}'. "
                        f"Supported: {list(_FILTER_OPS.keys())} and 'between'."
                    )
                if not op_fn(val, threshold):
                    keep = False
                    break

        if keep:
            result.append(blob)

    return result
