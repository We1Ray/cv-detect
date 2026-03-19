"""ROI (Region of Interest) management for CV defect detection.

Provides industrial vision-style ROI creation, manipulation, and persistence.
Supports rectangle, rotated rectangle, circle, ellipse, polygon, and ring
ROI types with JSON serialisation, mask generation, and drawing utilities.
"""

from __future__ import annotations

import copy
import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from shared.op_logger import log_operation

from dl_anomaly.core.region import Region

logger = logging.getLogger(__name__)

_VALID_ROI_TYPES = frozenset(
    {"rectangle", "rotated_rectangle", "circle", "ellipse", "polygon", "ring"}
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_native(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays to Python native types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        converted = [_to_native(v) for v in obj]
        return type(obj)(converted) if isinstance(obj, tuple) else converted
    return obj


# ---------------------------------------------------------------------------
# ROI dataclass
# ---------------------------------------------------------------------------


@dataclass
class ROI:
    """A single region of interest with type-specific parameters.

    Attributes:
        roi_type: One of ``rectangle``, ``rotated_rectangle``, ``circle``,
            ``ellipse``, ``polygon``, ``ring``.
        params: Dictionary of type-specific geometric parameters.
        name: Human-readable label.
        color: BGR display colour.
        visible: Whether the ROI is drawn / included in masks.
        locked: If True, the ROI cannot be moved or resized.
    """

    roi_type: str
    params: Dict[str, Any]
    name: str = "ROI_0"
    color: Tuple[int, int, int] = (0, 255, 0)
    visible: bool = True
    locked: bool = False

    def __post_init__(self) -> None:
        if self.roi_type not in _VALID_ROI_TYPES:
            raise ValueError(
                f"Invalid roi_type '{self.roi_type}'. "
                f"Must be one of {sorted(_VALID_ROI_TYPES)}."
            )

    # ------------------------------------------------------------------
    # Mask / Region generation
    # ------------------------------------------------------------------

    @log_operation(logger)
    def to_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """Generate a binary uint8 mask (0/255) for this ROI.

        Parameters:
            shape: (height, width) of the output mask.

        Returns:
            uint8 ndarray with 255 inside the ROI and 0 outside.
        """
        mask = np.zeros(shape[:2], dtype=np.uint8)
        p = self.params

        if self.roi_type == "rectangle":
            x, y, w, h = int(p["x"]), int(p["y"]), int(p["width"]), int(p["height"])
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, thickness=-1)

        elif self.roi_type == "rotated_rectangle":
            cx, cy = float(p["cx"]), float(p["cy"])
            w, h = float(p["width"]), float(p["height"])
            angle = float(p["angle"])
            box = cv2.boxPoints(((cx, cy), (w, h), angle))
            pts = np.intp(np.round(box))
            cv2.fillConvexPoly(mask, pts, 255)

        elif self.roi_type == "circle":
            cx, cy, r = int(round(p["cx"])), int(round(p["cy"])), int(round(p["radius"]))
            cv2.circle(mask, (cx, cy), r, 255, thickness=-1)

        elif self.roi_type == "ellipse":
            cx, cy = int(round(p["cx"])), int(round(p["cy"]))
            rx, ry = int(round(p["rx"])), int(round(p["ry"]))
            angle = float(p["angle"])
            cv2.ellipse(mask, (cx, cy), (rx, ry), angle, 0, 360, 255, thickness=-1)

        elif self.roi_type == "polygon":
            points = np.array(p["points"], dtype=np.float64)
            pts = np.intp(np.round(points))
            cv2.fillPoly(mask, [pts], 255)

        elif self.roi_type == "ring":
            cx, cy = int(round(p["cx"])), int(round(p["cy"]))
            r_out = int(round(p["outer_radius"]))
            r_in = int(round(p["inner_radius"]))
            cv2.circle(mask, (cx, cy), r_out, 255, thickness=-1)
            cv2.circle(mask, (cx, cy), r_in, 0, thickness=-1)

        return mask

    def to_region(
        self, shape: Tuple[int, int], source_image: Optional[np.ndarray] = None
    ) -> Region:
        """Convert to a :class:`Region` object.

        Parameters:
            shape: (height, width) for the label array.
            source_image: Optional grayscale source image attached to the Region.

        Returns:
            A Region with a single labelled component (label 1).
        """
        mask = self.to_mask(shape)
        num, labels = cv2.connectedComponents(mask, connectivity=8)
        labels = labels.astype(np.int32)
        return Region(
            labels=labels,
            num_regions=num - 1,
            properties=[],
            source_image=source_image,
            source_shape=shape,
        )

    # ------------------------------------------------------------------
    # Geometry queries
    # ------------------------------------------------------------------

    def bounding_box(self) -> Tuple[int, int, int, int]:
        """Return the axis-aligned bounding box as ``(x, y, w, h)``."""
        p = self.params

        if self.roi_type == "rectangle":
            return (int(p["x"]), int(p["y"]), int(p["width"]), int(p["height"]))

        if self.roi_type == "rotated_rectangle":
            box = cv2.boxPoints(
                ((p["cx"], p["cy"]), (p["width"], p["height"]), p["angle"])
            )
            x_min, y_min = np.min(box, axis=0)
            x_max, y_max = np.max(box, axis=0)
            x, y = int(math.floor(x_min)), int(math.floor(y_min))
            w = int(math.ceil(x_max)) - x
            h = int(math.ceil(y_max)) - y
            return (x, y, w, h)

        if self.roi_type == "circle":
            cx, cy, r = p["cx"], p["cy"], p["radius"]
            x = int(math.floor(cx - r))
            y = int(math.floor(cy - r))
            d = int(math.ceil(2 * r))
            return (x, y, d, d)

        if self.roi_type == "ellipse":
            cx, cy = p["cx"], p["cy"]
            rx, ry = p["rx"], p["ry"]
            angle_rad = math.radians(p["angle"])
            cos_a, sin_a = abs(math.cos(angle_rad)), abs(math.sin(angle_rad))
            half_w = rx * cos_a + ry * sin_a
            half_h = rx * sin_a + ry * cos_a
            x = int(math.floor(cx - half_w))
            y = int(math.floor(cy - half_h))
            w = int(math.ceil(2 * half_w))
            h = int(math.ceil(2 * half_h))
            return (x, y, w, h)

        if self.roi_type == "polygon":
            pts = np.array(p["points"])
            x_min, y_min = np.min(pts, axis=0)
            x_max, y_max = np.max(pts, axis=0)
            x, y = int(math.floor(x_min)), int(math.floor(y_min))
            w = int(math.ceil(x_max)) - x
            h = int(math.ceil(y_max)) - y
            return (x, y, w, h)

        if self.roi_type == "ring":
            cx, cy, r = p["cx"], p["cy"], p["outer_radius"]
            x = int(math.floor(cx - r))
            y = int(math.floor(cy - r))
            d = int(math.ceil(2 * r))
            return (x, y, d, d)

        raise ValueError(f"Unknown roi_type: {self.roi_type}")

    def contains_point(self, x: float, y: float) -> bool:
        """Test whether the point ``(x, y)`` lies inside the ROI."""
        p = self.params

        if self.roi_type == "rectangle":
            return (
                p["x"] <= x <= p["x"] + p["width"]
                and p["y"] <= y <= p["y"] + p["height"]
            )

        if self.roi_type == "rotated_rectangle":
            cx, cy = p["cx"], p["cy"]
            angle_rad = math.radians(-p["angle"])
            cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
            dx, dy = x - cx, y - cy
            lx = abs(cos_a * dx - sin_a * dy)
            ly = abs(sin_a * dx + cos_a * dy)
            return lx <= p["width"] / 2 and ly <= p["height"] / 2

        if self.roi_type == "circle":
            return (x - p["cx"]) ** 2 + (y - p["cy"]) ** 2 <= p["radius"] ** 2

        if self.roi_type == "ellipse":
            cx, cy = p["cx"], p["cy"]
            angle_rad = math.radians(-p["angle"])
            cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
            dx, dy = x - cx, y - cy
            lx = cos_a * dx - sin_a * dy
            ly = sin_a * dx + cos_a * dy
            return (lx / p["rx"]) ** 2 + (ly / p["ry"]) ** 2 <= 1.0

        if self.roi_type == "polygon":
            result = cv2.pointPolygonTest(
                np.array(p["points"], dtype=np.float32), (x, y), False
            )
            return result >= 0

        if self.roi_type == "ring":
            d2 = (x - p["cx"]) ** 2 + (y - p["cy"]) ** 2
            return p["inner_radius"] ** 2 <= d2 <= p["outer_radius"] ** 2

        raise ValueError(f"Unknown roi_type: {self.roi_type}")

    def area(self) -> float:
        """Compute the geometric area of the ROI."""
        p = self.params

        if self.roi_type == "rectangle":
            return float(p["width"] * p["height"])

        if self.roi_type == "rotated_rectangle":
            return float(p["width"] * p["height"])

        if self.roi_type == "circle":
            return math.pi * p["radius"] ** 2

        if self.roi_type == "ellipse":
            return math.pi * p["rx"] * p["ry"]

        if self.roi_type == "polygon":
            pts = p["points"]
            n = len(pts)
            if n < 3:
                return 0.0
            # Shoelace formula
            s = 0.0
            for i in range(n):
                x0, y0 = pts[i]
                x1, y1 = pts[(i + 1) % n]
                s += x0 * y1 - x1 * y0
            return abs(s) / 2.0

        if self.roi_type == "ring":
            return math.pi * (p["outer_radius"] ** 2 - p["inner_radius"] ** 2)

        raise ValueError(f"Unknown roi_type: {self.roi_type}")

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the ROI to a JSON-compatible dict."""
        return _to_native(
            {
                "roi_type": self.roi_type,
                "params": self.params,
                "name": self.name,
                "color": list(self.color),
                "visible": self.visible,
                "locked": self.locked,
            }
        )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ROI":
        """Deserialize an ROI from a dict."""
        color = d.get("color", [0, 255, 0])
        return cls(
            roi_type=d["roi_type"],
            params=d["params"],
            name=d.get("name", "ROI_0"),
            color=tuple(color),  # type: ignore[arg-type]
            visible=d.get("visible", True),
            locked=d.get("locked", False),
        )


# ---------------------------------------------------------------------------
# ROIManager
# ---------------------------------------------------------------------------


class ROIManager:
    """Manages a collection of :class:`ROI` objects with persistence.

    Attributes:
        _rois: Internal list of ROI objects.
        _image_shape: Reference image dimensions (H, W).
        _next_id: Auto-incrementing counter used for default ROI names.
    """

    def __init__(self) -> None:
        self._rois: List[ROI] = []
        self._image_shape: Optional[Tuple[int, int]] = None
        self._next_id: int = 0

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def add_roi(self, roi: ROI) -> int:
        """Add an ROI and return its index."""
        if roi.name == "ROI_0" or roi.name == f"ROI_{self._next_id}":
            roi.name = f"ROI_{self._next_id}"
        self._rois.append(roi)
        idx = len(self._rois) - 1
        self._next_id += 1
        logger.info("Added ROI '%s' at index %d", roi.name, idx)
        return idx

    def remove_roi(self, index: int) -> None:
        """Remove an ROI by its index."""
        if not 0 <= index < len(self._rois):
            raise IndexError(f"ROI index {index} out of range (0..{len(self._rois) - 1})")
        removed = self._rois.pop(index)
        logger.info("Removed ROI '%s' from index %d", removed.name, index)

    def get_roi(self, index: int) -> ROI:
        """Retrieve an ROI by index."""
        if not 0 <= index < len(self._rois):
            raise IndexError(f"ROI index {index} out of range (0..{len(self._rois) - 1})")
        return self._rois[index]

    def get_all_rois(self) -> List[ROI]:
        """Return a shallow copy of the internal ROI list."""
        return list(self._rois)

    def clear(self) -> None:
        """Remove all ROIs."""
        self._rois.clear()
        self._next_id = 0
        logger.info("Cleared all ROIs")

    def set_image_shape(self, shape: Tuple[int, int]) -> None:
        """Set the reference image shape ``(height, width)``."""
        self._image_shape = shape

    def __len__(self) -> int:
        return len(self._rois)

    # ------------------------------------------------------------------
    # Mask operations
    # ------------------------------------------------------------------

    def _require_shape(self) -> Tuple[int, int]:
        if self._image_shape is None:
            raise RuntimeError(
                "Image shape not set. Call set_image_shape() first."
            )
        return self._image_shape

    @log_operation(logger)
    def get_combined_mask(self) -> np.ndarray:
        """Return the union mask of all visible ROIs (uint8, 0/255)."""
        shape = self._require_shape()
        mask = np.zeros(shape[:2], dtype=np.uint8)
        for roi in self._rois:
            if roi.visible:
                roi_mask = roi.to_mask(shape)
                mask = cv2.bitwise_or(mask, roi_mask)
        return mask

    @log_operation(logger)
    def get_inverse_mask(self) -> np.ndarray:
        """Return the inverse of the combined visible mask."""
        combined = self.get_combined_mask()
        return cv2.bitwise_not(combined)

    @log_operation(logger)
    def apply_roi_to_image(
        self, image: np.ndarray, index: int
    ) -> np.ndarray:
        """Mask *image* keeping only the area inside the specified ROI.

        Pixels outside the ROI are set to black (0).

        Parameters:
            image: Input image (grayscale or BGR).
            index: Index of the ROI to apply.

        Returns:
            A copy of *image* with pixels outside the ROI zeroed.
        """
        roi = self.get_roi(index)
        h, w = image.shape[:2]
        mask = roi.to_mask((h, w))
        result = image.copy()
        if result.ndim == 3:
            result[mask == 0] = 0
        else:
            result[mask == 0] = 0
        return result

    @log_operation(logger)
    def crop_roi(
        self, image: np.ndarray, index: int
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Crop *image* to the bounding box of the specified ROI.

        Parameters:
            image: Input image.
            index: Index of the ROI.

        Returns:
            Tuple of (cropped_image, (offset_x, offset_y)).
        """
        roi = self.get_roi(index)
        bx, by, bw, bh = roi.bounding_box()
        h, w = image.shape[:2]
        # Clamp to image bounds
        x1 = max(bx, 0)
        y1 = max(by, 0)
        x2 = min(bx + bw, w)
        y2 = min(by + bh, h)
        cropped = image[y1:y2, x1:x2].copy()
        return cropped, (x1, y1)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save all ROIs to a JSON file."""
        data = self.to_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info("Saved %d ROIs to %s", len(self._rois), path)

    def load(self, path: str) -> None:
        """Load ROIs from a JSON file, replacing current state."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        loaded = ROIManager.from_dict(data)
        self._rois = loaded._rois
        self._image_shape = loaded._image_shape
        self._next_id = loaded._next_id
        logger.info("Loaded %d ROIs from %s", len(self._rois), path)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the entire manager state to a dict."""
        return _to_native(
            {
                "image_shape": list(self._image_shape) if self._image_shape else None,
                "next_id": self._next_id,
                "rois": [roi.to_dict() for roi in self._rois],
            }
        )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ROIManager":
        """Deserialize an ROIManager from a dict."""
        mgr = cls()
        shape = d.get("image_shape")
        if shape is not None:
            mgr._image_shape = tuple(shape)  # type: ignore[assignment]
        mgr._next_id = d.get("next_id", 0)
        for roi_d in d.get("rois", []):
            mgr._rois.append(ROI.from_dict(roi_d))
        return mgr

    # ------------------------------------------------------------------
    # Transform operations
    # ------------------------------------------------------------------

    def move_roi(self, index: int, dx: float, dy: float) -> None:
        """Translate an ROI by ``(dx, dy)`` pixels."""
        roi = self.get_roi(index)
        if roi.locked:
            logger.warning("ROI '%s' is locked; move ignored.", roi.name)
            return
        p = roi.params

        if roi.roi_type == "rectangle":
            p["x"] = p["x"] + dx
            p["y"] = p["y"] + dy
        elif roi.roi_type in ("rotated_rectangle", "circle", "ellipse", "ring"):
            p["cx"] = p["cx"] + dx
            p["cy"] = p["cy"] + dy
        elif roi.roi_type == "polygon":
            p["points"] = [(x + dx, y + dy) for x, y in p["points"]]

        logger.info("Moved ROI '%s' by (%.1f, %.1f)", roi.name, dx, dy)

    def resize_roi(self, index: int, scale: float) -> None:
        """Scale an ROI uniformly by *scale* around its centre."""
        roi = self.get_roi(index)
        if roi.locked:
            logger.warning("ROI '%s' is locked; resize ignored.", roi.name)
            return
        if scale <= 0:
            raise ValueError("Scale must be positive.")
        p = roi.params

        if roi.roi_type == "rectangle":
            cx = p["x"] + p["width"] / 2.0
            cy = p["y"] + p["height"] / 2.0
            new_w = p["width"] * scale
            new_h = p["height"] * scale
            p["x"] = cx - new_w / 2.0
            p["y"] = cy - new_h / 2.0
            p["width"] = new_w
            p["height"] = new_h
        elif roi.roi_type == "rotated_rectangle":
            p["width"] = p["width"] * scale
            p["height"] = p["height"] * scale
        elif roi.roi_type == "circle":
            p["radius"] = p["radius"] * scale
        elif roi.roi_type == "ellipse":
            p["rx"] = p["rx"] * scale
            p["ry"] = p["ry"] * scale
        elif roi.roi_type == "polygon":
            pts = p["points"]
            cx = sum(x for x, _ in pts) / len(pts)
            cy = sum(y for _, y in pts) / len(pts)
            p["points"] = [
                (cx + (x - cx) * scale, cy + (y - cy) * scale) for x, y in pts
            ]
        elif roi.roi_type == "ring":
            p["inner_radius"] = p["inner_radius"] * scale
            p["outer_radius"] = p["outer_radius"] * scale

        logger.info("Resized ROI '%s' by factor %.3f", roi.name, scale)

    def duplicate_roi(self, index: int) -> int:
        """Duplicate an ROI and return the new index."""
        roi = self.get_roi(index)
        new_roi = ROI(
            roi_type=roi.roi_type,
            params=copy.deepcopy(roi.params),
            name=f"{roi.name}_copy",
            color=roi.color,
            visible=roi.visible,
            locked=False,
        )
        return self.add_roi(new_roi)


# ---------------------------------------------------------------------------
# Factory functions (vision-style)
# ---------------------------------------------------------------------------


def gen_rectangle1(row1: int, col1: int, row2: int, col2: int) -> ROI:
    """Create an axis-aligned rectangle from top-left ``(col1, row1)`` to
    bottom-right ``(col2, row2)``.

    Parameters follow vision convention: ``(row, col)``.
    """
    x = min(col1, col2)
    y = min(row1, row2)
    w = abs(col2 - col1)
    h = abs(row2 - row1)
    return ROI(
        roi_type="rectangle",
        params={"x": x, "y": y, "width": w, "height": h},
    )


def gen_rectangle2(
    row: float, col: float, phi: float, length1: float, length2: float
) -> ROI:
    """Create a rotated rectangle.

    Parameters:
        row, col: Centre point (vision convention: row first).
        phi: Rotation angle in radians.
        length1: Half-length along the major axis.
        length2: Half-length along the minor axis.
    """
    angle_deg = math.degrees(phi)
    return ROI(
        roi_type="rotated_rectangle",
        params={
            "cx": float(col),
            "cy": float(row),
            "width": float(2 * length1),
            "height": float(2 * length2),
            "angle": float(angle_deg),
        },
    )


def gen_circle(row: float, col: float, radius: float) -> ROI:
    """Create a circle ROI.

    Parameters:
        row, col: Centre point (vision convention).
        radius: Circle radius in pixels.
    """
    return ROI(
        roi_type="circle",
        params={"cx": float(col), "cy": float(row), "radius": float(radius)},
    )


def gen_ellipse(
    row: float, col: float, phi: float, ra: float, rb: float
) -> ROI:
    """Create an ellipse ROI.

    Parameters:
        row, col: Centre point (vision convention).
        phi: Orientation angle in radians.
        ra: Semi-axis length along the major axis.
        rb: Semi-axis length along the minor axis.
    """
    angle_deg = math.degrees(phi)
    return ROI(
        roi_type="ellipse",
        params={
            "cx": float(col),
            "cy": float(row),
            "rx": float(ra),
            "ry": float(rb),
            "angle": float(angle_deg),
        },
    )


def gen_region_polygon(
    rows: List[float], cols: List[float]
) -> ROI:
    """Create a polygon ROI from parallel lists of row and column coordinates.

    Parameters:
        rows: Y-coordinates of polygon vertices.
        cols: X-coordinates of polygon vertices.
    """
    if len(rows) != len(cols):
        raise ValueError("rows and cols must have the same length.")
    if len(rows) < 3:
        raise ValueError("A polygon requires at least 3 points.")
    points = [(float(c), float(r)) for r, c in zip(rows, cols)]
    return ROI(roi_type="polygon", params={"points": points})


def gen_ring(
    row: float, col: float, inner_radius: float, outer_radius: float
) -> ROI:
    """Create a ring (annulus) ROI.

    Parameters:
        row, col: Centre point (vision convention).
        inner_radius: Inner radius.
        outer_radius: Outer radius.
    """
    if inner_radius >= outer_radius:
        raise ValueError("inner_radius must be less than outer_radius.")
    return ROI(
        roi_type="ring",
        params={
            "cx": float(col),
            "cy": float(row),
            "inner_radius": float(inner_radius),
            "outer_radius": float(outer_radius),
        },
    )


# ---------------------------------------------------------------------------
# Drawing functions
# ---------------------------------------------------------------------------


def _draw_label(
    image: np.ndarray,
    text: str,
    position: Tuple[int, int],
    color: Tuple[int, int, int],
) -> None:
    """Draw a text label with a dark background for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    ox, oy = position
    # Background rectangle
    cv2.rectangle(image, (ox, oy - th - 4), (ox + tw + 4, oy + baseline), (0, 0, 0), -1)
    cv2.putText(image, text, (ox + 2, oy - 2), font, scale, color, thickness, cv2.LINE_AA)


def _draw_handle(
    image: np.ndarray, cx: int, cy: int, size: int = 4, color: Tuple[int, int, int] = (255, 255, 255)
) -> None:
    """Draw a small square handle at the given position."""
    cv2.rectangle(
        image,
        (cx - size, cy - size),
        (cx + size, cy + size),
        color,
        thickness=-1,
    )
    cv2.rectangle(
        image,
        (cx - size, cy - size),
        (cx + size, cy + size),
        (0, 0, 0),
        thickness=1,
    )


def draw_single_roi(
    image: np.ndarray,
    roi: ROI,
    show_handles: bool = False,
    thickness: int = 2,
) -> np.ndarray:
    """Draw a single ROI on *image* and return the annotated copy.

    Parameters:
        image: BGR image.
        roi: The ROI to draw.
        show_handles: If True, draw resize/move handles.
        thickness: Line thickness.

    Returns:
        Annotated copy of the image.
    """
    out = image.copy()
    if not roi.visible:
        return out

    color = roi.color
    p = roi.params

    if roi.roi_type == "rectangle":
        x, y, w, h = int(p["x"]), int(p["y"]), int(p["width"]), int(p["height"])
        cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
        if show_handles:
            for hx, hy in [(x, y), (x + w, y), (x, y + h), (x + w, y + h),
                            (x + w // 2, y), (x + w // 2, y + h),
                            (x, y + h // 2), (x + w, y + h // 2)]:
                _draw_handle(out, hx, hy)

    elif roi.roi_type == "rotated_rectangle":
        box = cv2.boxPoints(
            ((p["cx"], p["cy"]), (p["width"], p["height"]), p["angle"])
        )
        pts = np.intp(np.round(box))
        cv2.drawContours(out, [pts], 0, color, thickness)
        if show_handles:
            for pt in pts:
                _draw_handle(out, int(pt[0]), int(pt[1]))

    elif roi.roi_type == "circle":
        cx, cy, r = int(round(p["cx"])), int(round(p["cy"])), int(round(p["radius"]))
        cv2.circle(out, (cx, cy), r, color, thickness)
        if show_handles:
            _draw_handle(out, cx, cy)
            for hx, hy in [(cx + r, cy), (cx - r, cy), (cx, cy + r), (cx, cy - r)]:
                _draw_handle(out, hx, hy)

    elif roi.roi_type == "ellipse":
        cx, cy = int(round(p["cx"])), int(round(p["cy"]))
        rx, ry = int(round(p["rx"])), int(round(p["ry"]))
        angle = float(p["angle"])
        cv2.ellipse(out, (cx, cy), (rx, ry), angle, 0, 360, color, thickness)
        if show_handles:
            _draw_handle(out, cx, cy)
            angle_rad = math.radians(angle)
            cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
            for r_x, r_y in [(rx, 0), (-rx, 0), (0, ry), (0, -ry)]:
                hx = int(round(cx + cos_a * r_x - sin_a * r_y))
                hy = int(round(cy + sin_a * r_x + cos_a * r_y))
                _draw_handle(out, hx, hy)

    elif roi.roi_type == "polygon":
        pts = np.array(p["points"], dtype=np.float64)
        pts_int = np.intp(np.round(pts))
        cv2.polylines(out, [pts_int], isClosed=True, color=color, thickness=thickness)
        if show_handles:
            for pt in pts_int:
                _draw_handle(out, int(pt[0]), int(pt[1]))

    elif roi.roi_type == "ring":
        cx, cy = int(round(p["cx"])), int(round(p["cy"]))
        r_out = int(round(p["outer_radius"]))
        r_in = int(round(p["inner_radius"]))
        cv2.circle(out, (cx, cy), r_out, color, thickness)
        cv2.circle(out, (cx, cy), r_in, color, thickness)
        if show_handles:
            _draw_handle(out, cx, cy)
            for r in (r_in, r_out):
                for hx, hy in [(cx + r, cy), (cx - r, cy), (cx, cy + r), (cx, cy - r)]:
                    _draw_handle(out, hx, hy)

    return out


def draw_rois(
    image: np.ndarray,
    rois: List[ROI],
    show_names: bool = True,
    show_handles: bool = False,
    thickness: int = 2,
) -> np.ndarray:
    """Draw all ROIs on *image*.

    Parameters:
        image: BGR image.
        rois: List of ROI objects to draw.
        show_names: If True, render the ROI name near its bounding box.
        show_handles: If True, draw resize/move handles on each ROI.
        thickness: Line thickness.

    Returns:
        Annotated copy of the image.
    """
    out = image.copy()
    for roi in rois:
        if not roi.visible:
            continue
        out = draw_single_roi(out, roi, show_handles=show_handles, thickness=thickness)
        if show_names:
            bx, by, _, _ = roi.bounding_box()
            label_x = max(bx, 0)
            label_y = max(by - 4, 12)
            _draw_label(out, roi.name, (label_x, label_y), roi.color)
    return out
