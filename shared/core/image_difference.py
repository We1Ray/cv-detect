"""Image Difference -- registration-based defect detection via reference comparison.

Aligns a target image to a golden reference using selectable registration
methods (ECC, ORB feature matching, phase correlation, or none) and computes
a pixel-wise absolute difference map.  The difference map is thresholded and
morphologically cleaned to produce a binary defect mask.

Key components
--------------
- ``DifferenceResult`` -- dataclass holding diff map, mask, score, etc.
- ``ImageDifferencer`` -- stateful detector that caches a reference image and
  exposes both a rich ``compute_difference`` API and a pipeline-compatible
  ``detect`` interface.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from shared.validation import validate_image
from shared.op_logger import log_operation

logger = logging.getLogger(__name__)

# Valid registration methods.
_REGISTRATION_METHODS = ("ecc", "orb", "phase_correlation", "none")


# ====================================================================== #
#  Data Classes                                                            #
# ====================================================================== #

@dataclass
class DifferenceResult:
    """Container for image-difference defect detection results.

    Attributes:
        diff_map:          Float32 absolute difference map (blurred, 0-255
                           range).
        binary_mask:       Uint8 binary defect mask after thresholding and
                           morphological cleanup.
        score:             Overall anomaly score -- mean intensity inside
                           the defect mask, or 0.0 when no defect pixels
                           exist.
        aligned_image:     Target image after alignment to the reference.
        alignment_quality: Correlation coefficient between reference and
                           aligned target (1.0 = perfect, 0.0 = unrelated).
                           Set to 1.0 when registration is disabled.
    """

    diff_map: np.ndarray
    binary_mask: np.ndarray
    score: float
    aligned_image: np.ndarray
    alignment_quality: float


# ====================================================================== #
#  Internal helpers                                                        #
# ====================================================================== #

def _ensure_gray(img: np.ndarray) -> np.ndarray:
    """Convert to single-channel grayscale if needed."""
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _correlation_score(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised correlation coefficient between two same-sized gray images.

    Returns a value in ``[-1, 1]`` where 1.0 means identical.  If either
    image has zero variance the function returns 0.0.
    """
    a_f = a.astype(np.float64).ravel()
    b_f = b.astype(np.float64).ravel()
    a_f -= a_f.mean()
    b_f -= b_f.mean()
    denom = np.linalg.norm(a_f) * np.linalg.norm(b_f)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a_f, b_f) / denom)


# ====================================================================== #
#  ImageDifferencer                                                        #
# ====================================================================== #

class ImageDifferencer:
    """Registration-based image differencing for defect detection.

    Parameters:
        registration_method: Alignment strategy -- ``"ecc"`` (enhanced
            correlation coefficient), ``"orb"`` (ORB features + homography),
            ``"phase_correlation"`` (translation-only FFT), or ``"none"``.
        threshold:         Absolute-difference threshold for binarisation.
        blur_sigma:        Gaussian blur sigma applied before differencing
            to suppress sensor noise.
        morph_kernel_size: Size of the square structuring element used for
            morphological open/close cleanup.
        min_defect_area:   Minimum contour area (in pixels) to retain;
            smaller blobs are discarded.
    """

    def __init__(
        self,
        registration_method: str = "ecc",
        threshold: float = 30.0,
        blur_sigma: float = 2.0,
        morph_kernel_size: int = 5,
        min_defect_area: int = 20,
    ) -> None:
        method = registration_method.lower()
        if method not in _REGISTRATION_METHODS:
            raise ValueError(
                f"Unknown registration method {registration_method!r}; "
                f"choose from {_REGISTRATION_METHODS}"
            )
        self.registration_method: str = method
        self.threshold: float = threshold
        self.blur_sigma: float = blur_sigma
        self.morph_kernel_size: int = morph_kernel_size
        self.min_defect_area: int = min_defect_area

        self._reference: Optional[np.ndarray] = None
        self._reference_gray: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    #  Reference management                                                #
    # ------------------------------------------------------------------ #

    def set_reference(self, image: np.ndarray) -> None:
        """Cache a golden reference image.

        The reference is stored in both its original form and as grayscale
        for use by the registration and differencing stages.
        """
        validate_image(image, "reference")
        self._reference = image.copy()
        self._reference_gray = _ensure_gray(image)
        logger.info(
            "Reference image set (%dx%d, %d channels)",
            image.shape[1],
            image.shape[0],
            1 if image.ndim == 2 else image.shape[2],
        )

    @property
    def has_reference(self) -> bool:
        """Return ``True`` if a reference image has been cached."""
        return self._reference is not None

    # ------------------------------------------------------------------ #
    #  Registration                                                        #
    # ------------------------------------------------------------------ #

    @log_operation(logger)
    def register(
        self,
        reference: np.ndarray,
        target: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Align *target* to *reference* using the configured method.

        Returns:
            ``(aligned_image, quality_score)`` where *aligned_image* is the
            warped target and *quality_score* is the normalised correlation
            between *reference* and the aligned result.
        """
        ref_gray = _ensure_gray(reference)
        tgt_gray = _ensure_gray(target)

        if ref_gray.shape != tgt_gray.shape:
            logger.warning(
                "Shape mismatch: reference %s vs target %s -- resizing target",
                ref_gray.shape,
                tgt_gray.shape,
            )
            target = cv2.resize(target, (reference.shape[1], reference.shape[0]))
            tgt_gray = _ensure_gray(target)

        if self.registration_method == "none":
            quality = _correlation_score(ref_gray, tgt_gray)
            return target.copy(), quality

        if self.registration_method == "ecc":
            aligned, quality = self._register_ecc(ref_gray, tgt_gray, target)
        elif self.registration_method == "orb":
            aligned, quality = self._register_orb(ref_gray, tgt_gray, target)
        elif self.registration_method == "phase_correlation":
            aligned, quality = self._register_phase(ref_gray, tgt_gray, target)
        else:
            # Should never reach here due to __init__ validation.
            raise ValueError(f"Unsupported method: {self.registration_method}")

        logger.debug("Registration quality: %.4f", quality)
        return aligned, quality

    # -- ECC ----------------------------------------------------------- #

    def _register_ecc(
        self,
        ref_gray: np.ndarray,
        tgt_gray: np.ndarray,
        target: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Align using Enhanced Correlation Coefficient (ECC).

        Uses ``MOTION_EUCLIDEAN`` (translation + rotation) as a robust
        default.  Falls back to identity on convergence failure.
        """
        h, w = ref_gray.shape[:2]
        warp_matrix = np.eye(2, 3, dtype=np.float32)

        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            200,   # max iterations
            1e-6,  # epsilon
        )

        try:
            _, warp_matrix = cv2.findTransformECC(
                ref_gray,
                tgt_gray,
                warp_matrix,
                cv2.MOTION_EUCLIDEAN,
                criteria,
                inputMask=None,
                gaussFiltSize=5,
            )
        except cv2.error as exc:
            logger.warning("ECC failed to converge: %s -- using identity", exc)
            quality = _correlation_score(ref_gray, tgt_gray)
            return target.copy(), quality

        aligned = cv2.warpAffine(
            target,
            warp_matrix,
            (w, h),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE,
        )
        aligned_gray = _ensure_gray(aligned)
        quality = _correlation_score(ref_gray, aligned_gray)
        return aligned, quality

    # -- ORB ----------------------------------------------------------- #

    def _register_orb(
        self,
        ref_gray: np.ndarray,
        tgt_gray: np.ndarray,
        target: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Align using ORB features + RANSAC homography."""
        h, w = ref_gray.shape[:2]

        orb = cv2.ORB_create(nfeatures=2000)
        kp1, des1 = orb.detectAndCompute(ref_gray, None)
        kp2, des2 = orb.detectAndCompute(tgt_gray, None)

        if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
            logger.warning(
                "ORB: insufficient features (ref=%d, tgt=%d) -- using identity",
                0 if des1 is None else len(des1),
                0 if des2 is None else len(des2),
            )
            quality = _correlation_score(ref_gray, tgt_gray)
            return target.copy(), quality

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        raw_matches = matcher.knnMatch(des2, des1, k=2)

        # Lowe's ratio test.
        good: List[cv2.DMatch] = []
        for pair in raw_matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < 0.75 * n.distance:
                    good.append(m)

        if len(good) < 4:
            logger.warning(
                "ORB: only %d good matches (need >= 4) -- using identity",
                len(good),
            )
            quality = _correlation_score(ref_gray, tgt_gray)
            return target.copy(), quality

        src_pts = np.float32(
            [kp2[m.queryIdx].pt for m in good],
        ).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp1[m.trainIdx].pt for m in good],
        ).reshape(-1, 1, 2)

        homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if homography is None:
            logger.warning("ORB: RANSAC failed -- using identity")
            quality = _correlation_score(ref_gray, tgt_gray)
            return target.copy(), quality

        inliers = int(mask.sum()) if mask is not None else 0
        logger.debug("ORB: %d/%d inliers", inliers, len(good))

        aligned = cv2.warpPerspective(
            target,
            homography,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        aligned_gray = _ensure_gray(aligned)
        quality = _correlation_score(ref_gray, aligned_gray)
        return aligned, quality

    # -- Phase correlation --------------------------------------------- #

    def _register_phase(
        self,
        ref_gray: np.ndarray,
        tgt_gray: np.ndarray,
        target: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Align using phase correlation (translation only)."""
        h, w = ref_gray.shape[:2]

        ref_f = ref_gray.astype(np.float32)
        tgt_f = tgt_gray.astype(np.float32)

        shift, response = cv2.phaseCorrelate(ref_f, tgt_f)
        dx, dy = shift  # (x, y) translation

        logger.debug("Phase correlation shift: dx=%.2f, dy=%.2f, response=%.4f", dx, dy, response)

        warp_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        aligned = cv2.warpAffine(
            target,
            warp_matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        aligned_gray = _ensure_gray(aligned)
        quality = _correlation_score(ref_gray, aligned_gray)
        return aligned, quality

    # ------------------------------------------------------------------ #
    #  Difference computation                                              #
    # ------------------------------------------------------------------ #

    @log_operation(logger)
    def compute_difference(self, target: np.ndarray) -> DifferenceResult:
        """Compute the difference between the cached reference and *target*.

        Steps:
            1. Convert to grayscale if needed.
            2. Register (align) target to reference.
            3. Gaussian blur both images to suppress noise.
            4. Compute absolute pixel-wise difference.
            5. Threshold to produce a binary mask.
            6. Morphological open then close to clean the mask.
            7. Remove small blobs below ``min_defect_area``.
            8. Compute an anomaly score (mean diff inside the mask).

        Raises:
            RuntimeError: If no reference image has been set.
        """
        if self._reference is None or self._reference_gray is None:
            raise RuntimeError(
                "No reference image set. Call set_reference() first."
            )
        validate_image(target, "target")

        # 1. Grayscale conversion.
        tgt_gray = _ensure_gray(target)

        # 2. Registration.
        aligned, alignment_quality = self.register(self._reference, target)
        aligned_gray = _ensure_gray(aligned)

        # 3. Gaussian blur.
        ksize = 0  # auto-compute from sigma
        ref_blur = cv2.GaussianBlur(
            self._reference_gray, (ksize, ksize), self.blur_sigma,
        )
        tgt_blur = cv2.GaussianBlur(
            aligned_gray, (ksize, ksize), self.blur_sigma,
        )

        # 4. Absolute difference.
        diff_map = cv2.absdiff(
            ref_blur.astype(np.float32),
            tgt_blur.astype(np.float32),
        )

        # 5. Threshold.
        _, binary = cv2.threshold(
            diff_map.astype(np.uint8),
            int(self.threshold),
            255,
            cv2.THRESH_BINARY,
        )
        binary = binary.astype(np.uint8)

        # 6. Morphological cleanup: open (remove noise) then close (fill gaps).
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.morph_kernel_size, self.morph_kernel_size),
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # 7. Remove small blobs.
        if self.min_defect_area > 0:
            binary = self._filter_small_blobs(binary)

        # 8. Anomaly score.
        mask_pixels = binary > 0
        if mask_pixels.any():
            score = float(np.mean(diff_map[mask_pixels]))
        else:
            score = 0.0

        return DifferenceResult(
            diff_map=diff_map,
            binary_mask=binary,
            score=score,
            aligned_image=aligned,
            alignment_quality=alignment_quality,
        )

    def _filter_small_blobs(self, mask: np.ndarray) -> np.ndarray:
        """Remove connected components smaller than ``min_defect_area``."""
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        filtered = np.zeros_like(mask)
        for cnt in contours:
            if cv2.contourArea(cnt) >= self.min_defect_area:
                cv2.drawContours(filtered, [cnt], -1, 255, cv2.FILLED)
        return filtered

    # ------------------------------------------------------------------ #
    #  Pipeline-compatible detect interface                                #
    # ------------------------------------------------------------------ #

    @log_operation(logger)
    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """DetectStep-compatible interface returning a standard result dict.

        The returned dictionary contains:

        - ``anomaly_score``  (float):  Mean diff intensity in defect regions.
        - ``is_defective``   (bool):   ``True`` if any defect pixels remain.
        - ``defect_mask``    (ndarray): Binary uint8 mask.
        - ``defect_regions`` (list):    List of bounding-box dicts
          ``{"x", "y", "w", "h", "area"}``.
        - ``error_map``      (ndarray): Float32 difference map.
        - ``alignment_quality`` (float): Registration correlation score.
        """
        result = self.compute_difference(image)

        # Extract defect region bounding boxes.
        contours, _ = cv2.findContours(
            result.binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        defect_regions: List[Dict[str, int]] = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            defect_regions.append({
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "area": int(cv2.contourArea(cnt)),
            })

        return {
            "anomaly_score": result.score,
            "is_defective": bool(np.any(result.binary_mask > 0)),
            "defect_mask": result.binary_mask,
            "defect_regions": defect_regions,
            "error_map": result.diff_map,
            "alignment_quality": result.alignment_quality,
        }
