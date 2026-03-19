"""
core/barcode_engine.py - Enhanced barcode / QR code detection with quality grading.

Extends the basic ``find_barcode`` / ``find_qrcode`` helpers in
``vision_ops.py`` with structured result types, multi-decoder support,
simplified ISO 15416 / 15415 quality grading, and visualisation helpers.

Categories:
    1. Data Classes (BarcodeResult)
    2. Decoder Availability Checks
    3. Detection & Decoding (OpenCV, pyzbar)
    4. Quality Grading (ISO 15416 / 15415 simplified)
    5. Verification
    6. Drawing
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from shared.validation import validate_image, validate_positive
from shared.op_logger import log_operation

logger = logging.getLogger(__name__)

# ====================================================================== #
#  Named constants                                                        #
# ====================================================================== #

GRADE_ORDER = ("A", "B", "C", "D", "F")

_GRADE_COLORS: Dict[str, Tuple[int, int, int]] = {
    "A": (0, 200, 0),     # green
    "B": (200, 100, 0),   # blue (BGR)
    "C": (0, 220, 220),   # yellow
    "D": (0, 140, 255),   # orange
    "F": (0, 0, 220),     # red
}

_1D_TYPES = frozenset({
    "EAN-13", "EAN-8", "EAN13", "EAN8",
    "UPC-A", "UPC-E", "UPCA", "UPCE",
    "CODE128", "CODE39", "CODE93", "Code128", "Code39", "Code93",
    "I25", "ITF", "I2of5", "CODABAR",
})

# ====================================================================== #
#  Data Classes                                                           #
# ====================================================================== #


@dataclass
class BarcodeResult:
    """A single detected and decoded barcode or 2D code."""
    data: str                                   # decoded data
    type: str                                   # "EAN-13", "Code128", "QR", etc.
    bbox: Tuple[int, int, int, int]             # (x, y, w, h)
    polygon: Optional[np.ndarray] = None        # corner points
    confidence: float = 1.0                     # decode confidence 0-1
    quality_grade: Optional[str] = None         # "A"-"F"
    quality_metrics: Optional[Dict[str, Any]] = field(default=None)


# ====================================================================== #
#  Decoder availability                                                   #
# ====================================================================== #


def check_pyzbar_available() -> bool:
    """Return True if pyzbar is importable."""
    try:
        import pyzbar.pyzbar  # noqa: F401
        return True
    except Exception:
        return False


def list_available_decoders() -> List[str]:
    """Return decoder back-end names available in this environment.

    ``"opencv"`` is always listed (ships with cv2).
    ``"pyzbar"`` is listed only when the package is installed.
    """
    decoders = ["opencv"]
    if check_pyzbar_available():
        decoders.append("pyzbar")
    return decoders


# ====================================================================== #
#  Internal helpers                                                       #
# ====================================================================== #


def _ensure_gray(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("Empty or None image provided.")
    if len(image.shape) == 3 and image.shape[2] >= 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _polygon_to_bbox(pts: np.ndarray) -> Tuple[int, int, int, int]:
    """Convert Nx2 polygon points to (x, y, w, h)."""
    pts = np.asarray(pts, dtype=np.int32).reshape(-1, 2)
    x, y, w, h = cv2.boundingRect(pts)
    return (x, y, w, h)


def _worse_grade(a: str, b: str) -> str:
    """Return the worse of two letter grades."""
    ia = GRADE_ORDER.index(a) if a in GRADE_ORDER else len(GRADE_ORDER) - 1
    ib = GRADE_ORDER.index(b) if b in GRADE_ORDER else len(GRADE_ORDER) - 1
    return GRADE_ORDER[max(ia, ib)]


def _value_to_grade(value: float, thresholds: Tuple[float, float, float, float]) -> str:
    """Map a 0-1 value to a letter grade given (A, B, C, D) thresholds.

    Values >= thresholds[0] get 'A', >= thresholds[1] get 'B', etc.
    Anything below thresholds[3] is 'F'.
    """
    for grade, thr in zip(GRADE_ORDER, thresholds):
        if value >= thr:
            return grade
    return "F"


# ====================================================================== #
#  3. Detection & Decoding                                                #
# ====================================================================== #


@log_operation(logger)
def decode_barcodes(
    image: np.ndarray,
    decoder: str = "auto",
    types: Optional[List[str]] = None,
) -> List[BarcodeResult]:
    """Detect and decode all barcodes / QR codes in *image*.

    Args:
        image: Input image (BGR or grayscale).
        decoder: ``"auto"`` (try pyzbar first, fall back to opencv),
            ``"pyzbar"``, or ``"opencv"``.
        types: Optional whitelist of symbology names to keep
            (e.g. ``["EAN13", "QRCODE"]``).

    Returns:
        List of :class:`BarcodeResult`.
    """
    results: List[BarcodeResult] = []

    if decoder == "auto":
        if check_pyzbar_available():
            results = decode_with_pyzbar(image)
        if not results:
            results = decode_with_opencv(image)
    elif decoder == "pyzbar":
        results = decode_with_pyzbar(image)
    elif decoder == "opencv":
        results = decode_with_opencv(image)
    else:
        logger.warning("Unknown decoder '%s'; falling back to auto.", decoder)
        return decode_barcodes(image, decoder="auto", types=types)

    # Filter by type if requested
    if types and results:
        upper_types = {t.upper() for t in types}
        results = [r for r in results if r.type.upper() in upper_types]

    logger.info("decode_barcodes found %d result(s).", len(results))
    return results


@log_operation(logger)
def decode_with_opencv(image: np.ndarray) -> List[BarcodeResult]:
    """Decode using OpenCV's barcode and QR detectors.

    Combines ``cv2.barcode.BarcodeDetector`` (1D) and
    ``cv2.QRCodeDetector`` (QR).
    """
    gray = _ensure_gray(image)
    results: List[BarcodeResult] = []

    # --- 1D barcodes ---
    try:
        detector = cv2.barcode.BarcodeDetector()
        try:
            ok, decoded_info, decoded_type, points = detector.detectAndDecodeWithType(gray)
        except AttributeError:
            retval, points, _ = detector.detectAndDecode(gray)
            if retval and isinstance(retval, str) and retval:
                bbox = _polygon_to_bbox(points) if points is not None else (0, 0, 0, 0)
                results.append(BarcodeResult(
                    data=retval, type="unknown", bbox=bbox,
                    polygon=points, confidence=1.0,
                ))
            ok = False

        if ok and decoded_info is not None:
            for i, data in enumerate(decoded_info):
                if not data:
                    continue
                pts = points[i] if points is not None else None
                bbox = _polygon_to_bbox(pts) if pts is not None else (0, 0, 0, 0)
                btype = decoded_type[i] if decoded_type is not None else "unknown"
                results.append(BarcodeResult(
                    data=data, type=btype, bbox=bbox,
                    polygon=pts, confidence=1.0,
                ))
    except AttributeError:
        logger.debug("cv2.barcode.BarcodeDetector not available.")
    except cv2.error as exc:
        logger.warning("OpenCV barcode detector error: %s", exc)

    # --- QR codes ---
    try:
        qr_det = cv2.QRCodeDetector()
        try:
            qr_ok, qr_info, qr_pts, _ = qr_det.detectAndDecodeMulti(gray)
        except (cv2.error, AttributeError):
            qr_data, qr_pts, _ = qr_det.detectAndDecode(gray)
            qr_ok = bool(qr_data)
            qr_info = (qr_data,) if qr_data else ()

        if qr_ok and qr_info:
            for i, data in enumerate(qr_info):
                if not data:
                    continue
                pts = qr_pts[i] if qr_pts is not None and i < len(qr_pts) else None
                bbox = _polygon_to_bbox(pts) if pts is not None else (0, 0, 0, 0)
                results.append(BarcodeResult(
                    data=data, type="QR", bbox=bbox,
                    polygon=pts, confidence=1.0,
                ))
    except cv2.error as exc:
        logger.warning("OpenCV QR detector error: %s", exc)

    return results


@log_operation(logger)
def decode_with_pyzbar(image: np.ndarray) -> List[BarcodeResult]:
    """Decode barcodes and 2D codes using pyzbar.

    pyzbar supports EAN-13, EAN-8, UPC-A, UPC-E, Code128, Code39,
    Code93, I2of5, QR, PDF417, and more.
    """
    try:
        from pyzbar import pyzbar as _pyzbar
    except ImportError:
        logger.error(
            "pyzbar is not installed.  "
            "Install with: pip install pyzbar  "
            "(also requires libzbar shared library)."
        )
        return []

    gray = _ensure_gray(image)
    decoded = _pyzbar.decode(gray)

    results: List[BarcodeResult] = []
    for obj in decoded:
        # obj.rect -> Rect(left, top, width, height)
        bbox = (obj.rect.left, obj.rect.top, obj.rect.width, obj.rect.height)
        polygon = np.array([(p.x, p.y) for p in obj.polygon], dtype=np.int32) if obj.polygon else None
        results.append(BarcodeResult(
            data=obj.data.decode("utf-8", errors="replace"),
            type=obj.type,
            bbox=bbox,
            polygon=polygon,
            confidence=1.0,
        ))

    return results


# ====================================================================== #
#  4. Quality Grading                                                     #
# ====================================================================== #


@log_operation(logger)
def grade_barcode_quality(
    image: np.ndarray,
    barcode_result: BarcodeResult,
) -> BarcodeResult:
    """Compute a simplified quality grade for a detected barcode.

    For 1D barcodes a simplified ISO 15416 assessment is applied
    (edge contrast, symbol contrast, modulation, defects).
    For 2D codes a simplified ISO 15415 assessment is applied.

    The *barcode_result* is updated in-place and also returned.
    """
    gray = _ensure_gray(image)
    x, y, w, h = barcode_result.bbox
    if w <= 0 or h <= 0:
        barcode_result.quality_grade = "F"
        barcode_result.quality_metrics = {"error": "invalid bbox"}
        return barcode_result

    # Clamp to image bounds
    ih, iw = gray.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(iw, x + w), min(ih, y + h)
    crop = gray[y1:y2, x1:x2]

    if crop.size == 0:
        barcode_result.quality_grade = "F"
        barcode_result.quality_metrics = {"error": "empty crop"}
        return barcode_result

    is_1d = barcode_result.type.upper().replace("-", "") in {
        t.upper().replace("-", "") for t in _1D_TYPES
    }

    if is_1d:
        metrics, grade = _grade_1d(crop)
    else:
        metrics, grade = _grade_2d(crop)

    barcode_result.quality_grade = grade
    barcode_result.quality_metrics = metrics
    logger.info(
        "Quality grade for '%s' (%s): %s",
        barcode_result.data[:20], barcode_result.type, grade,
    )
    return barcode_result


def _grade_1d(crop: np.ndarray) -> Tuple[Dict[str, Any], str]:
    """Simplified ISO 15416 grading for a 1D barcode crop."""
    profile = compute_scan_profile(crop, (0, 0, crop.shape[1], crop.shape[0]))
    p_min = float(np.min(profile))
    p_max = float(np.max(profile))
    p_mean = float(np.mean(profile))

    symbol_contrast = (p_max - p_min) / 255.0
    sc_grade = _value_to_grade(symbol_contrast, (0.70, 0.55, 0.40, 0.20))

    min_reflectance = p_min / 255.0
    # Lower minimum reflectance is better (dark bars are dark)
    mr_grade = _value_to_grade(1.0 - min_reflectance, (0.50, 0.40, 0.30, 0.20))

    # Edge contrast: average gradient magnitude along profile
    grad = np.abs(np.diff(profile.astype(np.float64)))
    edge_contrast = float(np.mean(grad)) / 255.0 if len(grad) > 0 else 0.0
    ec_grade = _value_to_grade(edge_contrast, (0.15, 0.10, 0.06, 0.03))

    # Modulation = symbol_contrast / max_reflectance
    max_refl = p_max / 255.0 if p_max > 0 else 1e-6
    modulation = symbol_contrast / max_refl
    mod_grade = _value_to_grade(modulation, (0.70, 0.60, 0.50, 0.40))

    # Defects: check for unexpected peaks in the profile derivative
    if len(grad) > 0:
        defect_score = 1.0 - float(np.std(grad)) / (float(np.max(grad)) + 1e-6)
        defect_score = max(0.0, min(1.0, defect_score))
    else:
        defect_score = 0.0
    def_grade = _value_to_grade(defect_score, (0.80, 0.60, 0.40, 0.25))

    overall = "A"
    for g in (sc_grade, mr_grade, ec_grade, mod_grade, def_grade):
        overall = _worse_grade(overall, g)

    metrics: Dict[str, Any] = {
        "symbol_contrast": round(symbol_contrast, 4),
        "symbol_contrast_grade": sc_grade,
        "min_reflectance": round(min_reflectance, 4),
        "min_reflectance_grade": mr_grade,
        "edge_contrast": round(edge_contrast, 4),
        "edge_contrast_grade": ec_grade,
        "modulation": round(modulation, 4),
        "modulation_grade": mod_grade,
        "defects": round(defect_score, 4),
        "defects_grade": def_grade,
    }
    return metrics, overall


def _grade_2d(crop: np.ndarray) -> Tuple[Dict[str, Any], str]:
    """Simplified ISO 15415 grading for a 2D code crop."""
    p_min = float(np.min(crop))
    p_max = float(np.max(crop))

    contrast = (p_max - p_min) / 255.0
    c_grade = _value_to_grade(contrast, (0.70, 0.55, 0.40, 0.20))

    # Modulation: ratio of local contrast to global contrast
    local_std = float(np.std(crop.astype(np.float64)))
    modulation = local_std / 128.0  # normalise
    modulation = min(modulation, 1.0)
    m_grade = _value_to_grade(modulation, (0.50, 0.40, 0.30, 0.20))

    # Fixed pattern damage: check uniformity of border (for QR)
    h, w = crop.shape[:2]
    if h >= 6 and w >= 6:
        border = np.concatenate([
            crop[0, :], crop[-1, :], crop[:, 0], crop[:, -1]
        ])
        border_std = float(np.std(border.astype(np.float64)))
        fp_score = 1.0 - min(border_std / 128.0, 1.0)
    else:
        fp_score = 0.5
    fp_grade = _value_to_grade(fp_score, (0.80, 0.60, 0.40, 0.25))

    # Decode already succeeded (we have data) -> grade A for decode
    decode_grade = "A"

    overall = "A"
    for g in (c_grade, m_grade, fp_grade, decode_grade):
        overall = _worse_grade(overall, g)

    metrics: Dict[str, Any] = {
        "contrast": round(contrast, 4),
        "contrast_grade": c_grade,
        "modulation": round(modulation, 4),
        "modulation_grade": m_grade,
        "fixed_pattern_damage": round(fp_score, 4),
        "fixed_pattern_grade": fp_grade,
        "decode_grade": decode_grade,
    }
    return metrics, overall


@log_operation(logger)
def compute_scan_profile(
    image: np.ndarray,
    barcode_bbox: Tuple[int, int, int, int],
    num_scans: int = 10,
) -> np.ndarray:
    """Extract and average 1D intensity profiles across a barcode region.

    Args:
        image: Input image (BGR or grayscale).
        barcode_bbox: ``(x, y, w, h)`` bounding box.
        num_scans: Number of horizontal scan lines to average.

    Returns:
        1D numpy array of averaged intensity values (float64),
        length equal to the barcode width.
    """
    gray = _ensure_gray(image)
    x, y, w, h = barcode_bbox
    ih, iw = gray.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(iw, x + w), min(ih, y + h)
    crop = gray[y1:y2, x1:x2]

    if crop.size == 0:
        return np.array([], dtype=np.float64)

    ch, cw = crop.shape[:2]
    num_scans = min(num_scans, ch)
    if num_scans <= 0:
        return np.array([], dtype=np.float64)

    step = max(1, ch // (num_scans + 1))
    profiles = []
    for i in range(1, num_scans + 1):
        row = min(i * step, ch - 1)
        profiles.append(crop[row, :].astype(np.float64))

    return np.mean(profiles, axis=0)


# ====================================================================== #
#  5. Verification                                                        #
# ====================================================================== #


@log_operation(logger)
def verify_barcode(
    result: BarcodeResult,
    expected_data: Optional[str] = None,
    min_grade: str = "C",
) -> Dict[str, Any]:
    """Verify a barcode against expected data and minimum quality.

    Args:
        result: The :class:`BarcodeResult` to verify.
        expected_data: If provided, check decoded data matches exactly.
        min_grade: Minimum acceptable quality grade (default ``"C"``).

    Returns:
        Dict with ``"valid"``, ``"data_match"``, ``"grade_pass"``,
        and ``"details"``.
    """
    data_match = True
    if expected_data is not None:
        data_match = result.data == expected_data

    grade_pass = True
    if result.quality_grade is not None:
        min_idx = GRADE_ORDER.index(min_grade) if min_grade in GRADE_ORDER else 2
        cur_idx = GRADE_ORDER.index(result.quality_grade) if result.quality_grade in GRADE_ORDER else 4
        grade_pass = cur_idx <= min_idx

    valid = data_match and grade_pass

    return {
        "valid": valid,
        "data_match": data_match,
        "grade_pass": grade_pass,
        "details": {
            "decoded_data": result.data,
            "expected_data": expected_data,
            "quality_grade": result.quality_grade,
            "min_grade": min_grade,
            "type": result.type,
        },
    }


# ====================================================================== #
#  6. Drawing                                                             #
# ====================================================================== #


@log_operation(logger)
def draw_barcode_results(
    image: np.ndarray,
    results: List[BarcodeResult],
    show_data: bool = True,
    show_grade: bool = True,
    thickness: int = 2,
) -> np.ndarray:
    """Draw detected barcode regions with labels onto an image.

    Bounding boxes are colour-coded by quality grade when available.

    Args:
        image: Input image (BGR).  A copy is made internally.
        results: List of :class:`BarcodeResult` to visualise.
        show_data: Show decoded data text.
        show_grade: Show quality grade letter.
        thickness: Line thickness.

    Returns:
        Annotated copy of the image.
    """
    canvas = image.copy()
    if len(canvas.shape) == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    for r in results:
        grade = r.quality_grade or "A"
        color = _GRADE_COLORS.get(grade, (0, 200, 0))

        # Draw polygon if available, otherwise rectangle
        if r.polygon is not None and len(r.polygon) >= 3:
            pts = r.polygon.reshape(-1, 1, 2).astype(np.int32)
            cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=thickness)
        else:
            x, y, w, h = r.bbox
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thickness)

        # Label
        parts: List[str] = []
        if show_data:
            display_data = r.data if len(r.data) <= 30 else r.data[:27] + "..."
            parts.append(display_data)
        if show_grade and r.quality_grade:
            parts.append(f"[{r.quality_grade}]")
        label = " ".join(parts)

        if label:
            x, y, w, h = r.bbox
            font_scale = 0.5
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            cv2.rectangle(
                canvas, (x, y - th - baseline - 4), (x + tw + 4, y), color, cv2.FILLED
            )
            cv2.putText(
                canvas, label, (x + 2, y - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA,
            )

    return canvas


@log_operation(logger)
def draw_scan_profile(
    profile: np.ndarray,
    width: int = 400,
    height: int = 200,
) -> np.ndarray:
    """Render a 1D scan profile as a line-graph image.

    Args:
        profile: 1D array of intensity values.
        width: Output image width in pixels.
        height: Output image height in pixels.

    Returns:
        BGR image with the profile drawn as a white line on black
        background.
    """
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    if profile is None or len(profile) == 0:
        return canvas

    # Normalise profile to canvas height
    p = profile.astype(np.float64)
    p_min, p_max = float(np.min(p)), float(np.max(p))
    if p_max - p_min < 1e-6:
        p_norm = np.full_like(p, height // 2, dtype=np.float64)
    else:
        p_norm = (p - p_min) / (p_max - p_min) * (height - 20) + 10

    # Map to pixel coordinates
    n = len(p_norm)
    x_coords = np.linspace(0, width - 1, n).astype(np.int32)
    y_coords = (height - 1 - p_norm).astype(np.int32)
    y_coords = np.clip(y_coords, 0, height - 1)

    pts = np.column_stack([x_coords, y_coords]).reshape(-1, 1, 2)
    cv2.polylines(canvas, [pts], isClosed=False, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

    return canvas
