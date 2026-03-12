"""
core/ocr_engine.py - OCR text recognition module for industrial inspection.

Wraps Tesseract (via pytesseract) and PaddleOCR to read date codes,
lot numbers, labels, and other printed text found on products or packaging.

Categories:
    1. Data Classes (OCRResult, TextLine)
    2. Engine Availability Checks
    3. Tesseract-based OCR
    4. PaddleOCR-based OCR
    5. Preprocessing Helpers
    6. Verification
    7. Drawing
"""

from __future__ import annotations

import logging
import math
import re
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

DEFAULT_PSM = 7          # single text line
DEFAULT_LANG = "eng"
DEFAULT_CLAHE_CLIP = 2.0
DEFAULT_CLAHE_TILE = 8
DEFAULT_MAX_SKEW = 15.0

# ====================================================================== #
#  Data Classes                                                           #
# ====================================================================== #


@dataclass
class OCRResult:
    """Single recognised text element (word or character)."""
    text: str                           # recognized text
    confidence: float                   # 0-1
    bbox: Tuple[int, int, int, int]     # (x, y, w, h)
    engine: str                         # "tesseract" or "paddleocr"


@dataclass
class TextLine:
    """A complete text line with optional per-character breakdown."""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]     # (x, y, w, h)
    char_results: List[OCRResult] = field(default_factory=list)


# ====================================================================== #
#  Engine availability checks                                             #
# ====================================================================== #


def check_tesseract_available() -> bool:
    """Return True if pytesseract is importable and tesseract binary is found."""
    try:
        import pytesseract
        # pytesseract.get_tesseract_version() raises if the binary is missing
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def check_paddleocr_available() -> bool:
    """Return True if paddleocr is importable."""
    try:
        import paddleocr  # noqa: F401
        return True
    except Exception:
        return False


def list_available_engines() -> List[str]:
    """Return a list of OCR engine names available in this environment."""
    engines: List[str] = []
    if check_tesseract_available():
        engines.append("tesseract")
    if check_paddleocr_available():
        engines.append("paddleocr")
    return engines


# ====================================================================== #
#  Internal helpers                                                       #
# ====================================================================== #


def _ensure_gray(image: np.ndarray) -> np.ndarray:
    """Convert to single-channel uint8 if necessary."""
    if image is None or image.size == 0:
        raise ValueError("Empty or None image provided.")
    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if len(image.shape) == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return image


def _crop_roi(image: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
    """Crop image to (x, y, w, h) ROI, clamped to image bounds."""
    h, w = image.shape[:2]
    x, y, rw, rh = roi
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w, x + rw)
    y2 = min(h, y + rh)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid ROI after clamping: ({x1},{y1},{x2},{y2})")
    return image[y1:y2, x1:x2].copy()


def _is_low_contrast(gray: np.ndarray, threshold: float = 30.0) -> bool:
    """Check if an image has low contrast based on standard deviation."""
    return float(np.std(gray)) < threshold


# ====================================================================== #
#  3. Tesseract-based OCR                                                 #
# ====================================================================== #


@log_operation(logger)
def ocr_tesseract(
    image: np.ndarray,
    lang: str = DEFAULT_LANG,
    config: str = f"--psm {DEFAULT_PSM}",
    roi: Optional[Tuple[int, int, int, int]] = None,
) -> List[OCRResult]:
    """Recognise text using Tesseract with word-level bounding boxes.

    Args:
        image: Input image (BGR or grayscale).
        lang: Tesseract language code (e.g. ``"eng"``, ``"chi_tra"``).
        config: Tesseract CLI config string.  Common PSM modes:
            7 = single text line, 6 = uniform block, 3 = fully automatic.
        roi: Optional ``(x, y, w, h)`` region of interest.  The image is
            cropped to this rectangle before recognition.

    Returns:
        List of :class:`OCRResult` with word-level bounding boxes.
    """
    try:
        import pytesseract
    except ImportError:
        logger.error(
            "pytesseract is not installed.  "
            "Install with: pip install pytesseract  "
            "(also requires the Tesseract binary)."
        )
        return []

    if roi is not None:
        image = _crop_roi(image, roi)

    gray = _ensure_gray(image)

    # Preprocess if contrast is low
    if _is_low_contrast(gray):
        logger.debug("Low contrast detected; applying adaptive preprocessing.")
        gray = preprocess_for_ocr(gray, method="adaptive")

    data = pytesseract.image_to_data(
        gray, lang=lang, config=config, output_type=pytesseract.Output.DICT
    )

    results: List[OCRResult] = []
    n_boxes = len(data["text"])
    for i in range(n_boxes):
        text = data["text"][i].strip()
        conf = float(data["conf"][i])
        if not text or conf < 0:
            continue
        confidence = conf / 100.0  # normalise to 0-1
        bbox = (
            int(data["left"][i]),
            int(data["top"][i]),
            int(data["width"][i]),
            int(data["height"][i]),
        )
        results.append(
            OCRResult(text=text, confidence=confidence, bbox=bbox, engine="tesseract")
        )

    logger.info("Tesseract found %d word(s).", len(results))
    return results


@log_operation(logger)
def ocr_tesseract_simple(
    image: np.ndarray,
    lang: str = DEFAULT_LANG,
    config: str = f"--psm {DEFAULT_PSM}",
) -> str:
    """Return recognised text as a plain string (no bounding-box data).

    This is a convenience wrapper around ``pytesseract.image_to_string``.
    """
    try:
        import pytesseract
    except ImportError:
        logger.error(
            "pytesseract is not installed.  "
            "Install with: pip install pytesseract"
        )
        return ""

    gray = _ensure_gray(image)
    if _is_low_contrast(gray):
        gray = preprocess_for_ocr(gray, method="adaptive")

    text: str = pytesseract.image_to_string(gray, lang=lang, config=config).strip()
    logger.info("Tesseract simple result: '%s'", text)
    return text


# ====================================================================== #
#  4. PaddleOCR-based OCR                                                 #
# ====================================================================== #


@log_operation(logger)
def ocr_paddle(
    image: np.ndarray,
    lang: str = "en",
    roi: Optional[Tuple[int, int, int, int]] = None,
) -> List[OCRResult]:
    """Recognise text using PaddleOCR (detection + recognition).

    PaddleOCR offers better support for Asian text (CJK) and complex
    layouts compared to Tesseract.

    Args:
        image: Input image (BGR or grayscale).
        lang: Language code (``"en"``, ``"ch"``, ``"japan"``, etc.).
        roi: Optional ``(x, y, w, h)`` region of interest.

    Returns:
        List of :class:`OCRResult`.
    """
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        logger.error(
            "paddleocr is not installed.  "
            "Install with: pip install paddleocr paddlepaddle"
        )
        return []

    if roi is not None:
        image = _crop_roi(image, roi)

    # Ensure BGR input for PaddleOCR
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
    raw_results = ocr.ocr(image, cls=True)

    results: List[OCRResult] = []
    if raw_results is None:
        return results

    for line_group in raw_results:
        if line_group is None:
            continue
        for line in line_group:
            # line = [[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], (text, conf)]
            box_pts, (text, conf) = line
            pts = np.array(box_pts, dtype=np.int32)
            x, y, w, h = cv2.boundingRect(pts)
            results.append(
                OCRResult(
                    text=text,
                    confidence=float(conf),
                    bbox=(x, y, w, h),
                    engine="paddleocr",
                )
            )

    logger.info("PaddleOCR found %d text region(s).", len(results))
    return results


# ====================================================================== #
#  5. Preprocessing helpers                                               #
# ====================================================================== #


@log_operation(logger)
def preprocess_for_ocr(
    image: np.ndarray,
    method: str = "adaptive",
) -> np.ndarray:
    """Preprocess an image to improve OCR accuracy.

    Steps:
        1. Convert to grayscale (if needed).
        2. Apply CLAHE for contrast enhancement.
        3. Binarize with the selected *method*.

    Args:
        image: Input image.
        method: ``"adaptive"`` (adaptive threshold), ``"otsu"``, or
            ``"none"`` (skip binarisation).

    Returns:
        Preprocessed single-channel uint8 image.
    """
    gray = _ensure_gray(image)

    # CLAHE contrast enhancement
    clahe = cv2.createCLAHE(
        clipLimit=DEFAULT_CLAHE_CLIP,
        tileGridSize=(DEFAULT_CLAHE_TILE, DEFAULT_CLAHE_TILE),
    )
    enhanced = clahe.apply(gray)

    if method == "adaptive":
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 10,
        )
    elif method == "otsu":
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "none":
        binary = enhanced
    else:
        logger.warning("Unknown preprocess method '%s'; using 'adaptive'.", method)
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 10,
        )

    return binary


@log_operation(logger)
def deskew_image(
    image: np.ndarray,
    max_angle: float = DEFAULT_MAX_SKEW,
) -> Tuple[np.ndarray, float]:
    """Detect and correct text skew.

    Uses Hough line transform on an edge map to estimate the dominant
    skew angle, then rotates the image to compensate.

    Args:
        image: Input image (BGR or grayscale).
        max_angle: Maximum skew angle (degrees) to correct.
            Lines exceeding this are ignored.

    Returns:
        ``(corrected_image, detected_angle_degrees)``.
    """
    gray = _ensure_gray(image)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=100,
        minLineLength=gray.shape[1] // 4, maxLineGap=10,
    )

    if lines is None or len(lines) == 0:
        logger.debug("No lines detected for deskew; returning original image.")
        return image.copy(), 0.0

    angles: List[float] = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if abs(angle) <= max_angle:
            angles.append(angle)

    if not angles:
        return image.copy(), 0.0

    median_angle = float(np.median(angles))
    logger.info("Detected skew angle: %.2f degrees.", median_angle)

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    corrected = cv2.warpAffine(
        image, rot_mat, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return corrected, median_angle


# ====================================================================== #
#  6. Verification                                                        #
# ====================================================================== #


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr_row.append(min(
                curr_row[j] + 1,        # insert
                prev_row[j + 1] + 1,    # delete
                prev_row[j] + cost,     # replace
            ))
        prev_row = curr_row
    return prev_row[-1]


@log_operation(logger)
def verify_text(
    recognized: str,
    expected_pattern: str,
    strict: bool = False,
) -> Dict[str, Any]:
    """Verify recognised text against an expected regex pattern.

    Args:
        recognized: The OCR-recognised string.
        expected_pattern: Regex pattern the text should match.
        strict: If True, the *entire* string must match (``re.fullmatch``).
            Otherwise ``re.search`` is used.

    Returns:
        Dict with keys ``"match"`` (bool), ``"recognized"`` (str),
        ``"expected"`` (str), ``"similarity"`` (float 0-1).
    """
    if strict:
        match = re.fullmatch(expected_pattern, recognized) is not None
    else:
        match = re.search(expected_pattern, recognized) is not None

    # Compute similarity against the pattern literal (strip regex meta chars)
    plain_expected = re.sub(r"[\\^$.*+?{}\[\]()|]", "", expected_pattern)
    if plain_expected:
        dist = _levenshtein_distance(recognized, plain_expected)
        max_len = max(len(recognized), len(plain_expected))
        similarity = 1.0 - (dist / max_len) if max_len > 0 else 1.0
    else:
        similarity = 1.0 if match else 0.0

    return {
        "match": match,
        "recognized": recognized,
        "expected": expected_pattern,
        "similarity": round(similarity, 4),
    }


# ====================================================================== #
#  7. Drawing                                                             #
# ====================================================================== #


@log_operation(logger)
def draw_ocr_results(
    image: np.ndarray,
    results: List[OCRResult],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes and labels for OCR results onto an image.

    Args:
        image: Input image (BGR).  A copy is made internally.
        results: List of :class:`OCRResult` to visualise.
        color: BGR colour for rectangles and text.
        thickness: Line thickness.

    Returns:
        Annotated copy of the image.
    """
    canvas = image.copy()
    if len(canvas.shape) == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    for r in results:
        x, y, w, h = r.bbox
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thickness)
        label = f"{r.text} ({r.confidence * 100:.0f}%)"
        font_scale = 0.5
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        # Background rectangle for readability
        cv2.rectangle(
            canvas, (x, y - th - baseline - 4), (x + tw, y), color, cv2.FILLED
        )
        cv2.putText(
            canvas, label, (x, y - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA,
        )

    return canvas
