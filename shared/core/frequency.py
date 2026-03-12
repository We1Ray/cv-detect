"""
core/frequency.py - Advanced frequency domain processing for defect detection.

Provides comprehensive FFT tools specifically designed for industrial defect
detection, including filter generation, periodic pattern removal, and
frequency-based anomaly detection.  Complements the basic ``fft_image`` and
``freq_filter`` helpers in :pymod:`halcon_ops`.

Categories:
    1.  Core FFT Operations
    2.  Filter Generation (Gaussian, Butterworth, Band, Notch, Custom)
    3.  Applied Processing (pattern removal, anomaly detection)
    4.  Visualization Helpers
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from shared.validation import validate_image
from shared.op_logger import log_operation

logger = logging.getLogger(__name__)


# ====================================================================== #
#  Data Classes                                                            #
# ====================================================================== #

@dataclass
class FFTResult:
    """Container for the results of a forward FFT computation.

    Attributes:
        magnitude:        Log-scaled magnitude spectrum as ``uint8``
                          (displayable, normalized to 0-255).
        phase:            Phase spectrum in radians (``float64``).
        complex_spectrum: Full complex DFT with DC shifted to center
                          (for reconstruction via :func:`inverse_fft`).
        shape:            Original image height and width before any
                          zero-padding.
    """

    magnitude: np.ndarray
    phase: np.ndarray
    complex_spectrum: np.ndarray
    shape: Tuple[int, int]


# ====================================================================== #
#  Internal helpers                                                        #
# ====================================================================== #

def _ensure_gray(img: np.ndarray) -> np.ndarray:
    """Convert to single-channel grayscale if needed."""
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _ensure_uint8(img: np.ndarray) -> np.ndarray:
    """Clip and cast to ``uint8``."""
    return np.clip(img, 0, 255).astype(np.uint8)


def _distance_matrix(shape: Tuple[int, int]) -> np.ndarray:
    """Return a float64 matrix of Euclidean distances from the center."""
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    u = np.arange(rows) - crow
    v = np.arange(cols) - ccol
    U, V = np.meshgrid(v, u)  # note: meshgrid(x, y)
    return np.sqrt(U.astype(np.float64) ** 2 + V.astype(np.float64) ** 2)


# ====================================================================== #
#  1. Core FFT Operations                                                  #
# ====================================================================== #

@log_operation(logger)
def compute_fft(image: np.ndarray) -> FFTResult:
    """Compute the forward FFT of *image* with optimal zero-padding.

    The image is converted to grayscale ``float64``, padded to the next
    optimal DFT size, and transformed.  The returned :class:`FFTResult`
    contains both a displayable log-magnitude spectrum and the full complex
    data needed for reconstruction.

    Args:
        image: Input image (grayscale or BGR).

    Returns:
        :class:`FFTResult` with magnitude, phase, complex spectrum, and
        original shape.
    """
    validate_image(image)
    gray = _ensure_gray(image).astype(np.float64)
    orig_h, orig_w = gray.shape[:2]

    # Optimal DFT size (may be larger than input)
    opt_rows = cv2.getOptimalDFTSize(orig_h)
    opt_cols = cv2.getOptimalDFTSize(orig_w)

    # Zero-pad to optimal size
    padded = np.zeros((opt_rows, opt_cols), dtype=np.float64)
    padded[:orig_h, :orig_w] = gray

    # Forward FFT and shift DC to center
    dft = np.fft.fft2(padded)
    dft_shift = np.fft.fftshift(dft)

    # Magnitude: 20 * log10(1 + |F|)
    mag = np.abs(dft_shift)
    log_mag = 20.0 * np.log10(1.0 + mag)

    # Normalize to [0, 255] uint8 for display
    m_min, m_max = log_mag.min(), log_mag.max()
    if m_max - m_min > 0:
        mag_display = (log_mag - m_min) / (m_max - m_min) * 255.0
    else:
        mag_display = np.zeros_like(log_mag)
    mag_display = _ensure_uint8(mag_display)

    # Phase
    phase = np.angle(dft_shift)

    return FFTResult(
        magnitude=mag_display,
        phase=phase,
        complex_spectrum=dft_shift,
        shape=(orig_h, orig_w),
    )


@log_operation(logger)
def inverse_fft(fft_result: FFTResult) -> np.ndarray:
    """Reconstruct an image from a :class:`FFTResult`.

    Performs the inverse shift, inverse FFT, crops to the original shape,
    and normalizes to ``uint8``.

    Args:
        fft_result: A :class:`FFTResult` (possibly with a modified
                    ``complex_spectrum``).

    Returns:
        Reconstructed ``uint8`` image.
    """
    # Inverse shift
    f_ishift = np.fft.ifftshift(fft_result.complex_spectrum)
    # Inverse FFT
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    # Crop to original shape
    orig_h, orig_w = fft_result.shape
    img_back = img_back[:orig_h, :orig_w]

    # Normalize to uint8
    v_min, v_max = img_back.min(), img_back.max()
    if v_max - v_min > 0:
        img_back = (img_back - v_min) / (v_max - v_min) * 255.0
    else:
        img_back = np.zeros_like(img_back)
    return _ensure_uint8(img_back)


# ====================================================================== #
#  2. Filter Generation                                                    #
# ====================================================================== #

@log_operation(logger)
def create_gaussian_filter(
    shape: Tuple[int, int],
    sigma: float,
    filter_type: str = "lowpass",
) -> np.ndarray:
    """Create a Gaussian frequency-domain filter.

    Args:
        shape:       ``(rows, cols)`` of the frequency plane.
        sigma:       Standard deviation controlling the cutoff.
        filter_type: ``"lowpass"`` or ``"highpass"``.

    Returns:
        ``float64`` filter mask with values in [0, 1].
    """
    D = _distance_matrix(shape)
    H = np.exp(-(D ** 2) / (2.0 * sigma ** 2))

    if filter_type == "highpass":
        H = 1.0 - H
    elif filter_type != "lowpass":
        raise ValueError(
            f"filter_type must be 'lowpass' or 'highpass', got {filter_type!r}"
        )
    return H


@log_operation(logger)
def create_butterworth_filter(
    shape: Tuple[int, int],
    cutoff: float,
    order: int = 2,
    filter_type: str = "lowpass",
) -> np.ndarray:
    """Create a Butterworth frequency-domain filter.

    The lowpass transfer function is::

        H(u,v) = 1 / (1 + (D(u,v) / D0) ^ (2n))

    Args:
        shape:       ``(rows, cols)`` of the frequency plane.
        cutoff:      Cutoff frequency ``D0`` in pixels.
        order:       Filter order ``n`` (higher = sharper roll-off).
        filter_type: ``"lowpass"`` or ``"highpass"``.

    Returns:
        ``float64`` filter mask with values in [0, 1].
    """
    D = _distance_matrix(shape)
    # Avoid division by zero at DC
    D_safe = np.where(D == 0, 1e-10, D)
    H = 1.0 / (1.0 + (D_safe / cutoff) ** (2 * order))

    if filter_type == "highpass":
        H = 1.0 - H
    elif filter_type != "lowpass":
        raise ValueError(
            f"filter_type must be 'lowpass' or 'highpass', got {filter_type!r}"
        )
    return H


@log_operation(logger)
def create_bandpass_filter(
    shape: Tuple[int, int],
    low_cutoff: float,
    high_cutoff: float,
    order: int = 2,
) -> np.ndarray:
    """Create a Butterworth bandpass filter.

    Passes frequencies between *low_cutoff* and *high_cutoff*.

    Args:
        shape:       ``(rows, cols)`` of the frequency plane.
        low_cutoff:  Lower cutoff frequency.
        high_cutoff: Upper cutoff frequency.
        order:       Butterworth order.

    Returns:
        ``float64`` filter mask with values in [0, 1].
    """
    lp_high = create_butterworth_filter(shape, high_cutoff, order, "lowpass")
    lp_low = create_butterworth_filter(shape, low_cutoff, order, "lowpass")
    bp = lp_high - lp_low
    return np.clip(bp, 0.0, 1.0)


@log_operation(logger)
def create_bandstop_filter(
    shape: Tuple[int, int],
    low_cutoff: float,
    high_cutoff: float,
    order: int = 2,
) -> np.ndarray:
    """Create a Butterworth bandstop (band-reject) filter.

    Rejects frequencies between *low_cutoff* and *high_cutoff*.

    Args:
        shape:       ``(rows, cols)`` of the frequency plane.
        low_cutoff:  Lower cutoff frequency.
        high_cutoff: Upper cutoff frequency.
        order:       Butterworth order.

    Returns:
        ``float64`` filter mask with values in [0, 1].
    """
    bp = create_bandpass_filter(shape, low_cutoff, high_cutoff, order)
    return 1.0 - bp


@log_operation(logger)
def create_notch_filter(
    shape: Tuple[int, int],
    centers: Sequence[Tuple[int, int]],
    radius: int = 10,
) -> np.ndarray:
    """Create a notch reject filter to suppress specific frequency peaks.

    Each center ``(u, v)`` (relative to the shifted spectrum center) and its
    symmetric counterpart ``(-u, -v)`` are suppressed with a circular hole
    of the given *radius*.  This is ideal for removing periodic noise /
    patterns (e.g. moire, regular texture).

    Args:
        shape:   ``(rows, cols)`` of the frequency plane.
        centers: Sequence of ``(u, v)`` frequency coordinates to suppress.
        radius:  Radius of each notch hole in pixels.

    Returns:
        ``float64`` filter mask with values in [0, 1].
    """
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    H = np.ones((rows, cols), dtype=np.float64)

    u_coords = np.arange(rows) - crow
    v_coords = np.arange(cols) - ccol
    V, U = np.meshgrid(v_coords, u_coords)

    for cu, cv in centers:
        # Primary notch
        dist_primary = np.sqrt((U - cu) ** 2 + (V - cv) ** 2)
        H[dist_primary <= radius] = 0.0
        # Symmetric counterpart
        dist_sym = np.sqrt((U + cu) ** 2 + (V + cv) ** 2)
        H[dist_sym <= radius] = 0.0

    return H


@log_operation(logger)
def create_custom_mask(
    shape: Tuple[int, int],
    mask_image: np.ndarray,
) -> np.ndarray:
    """Convert a user-drawn binary mask image to a frequency filter.

    The mask is resized to *shape*, converted to grayscale, and normalized
    to [0, 1] ``float64``.  Useful for interactive masking in a GUI.

    Args:
        shape:      ``(rows, cols)`` target shape.
        mask_image: Binary or grayscale mask (any dtype).

    Returns:
        ``float64`` filter mask with values in [0, 1].
    """
    validate_image(mask_image)
    gray = _ensure_gray(mask_image)
    resized = cv2.resize(gray, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = resized.astype(np.float64)
    m_max = mask.max()
    if m_max > 0:
        mask /= m_max
    return mask


# ====================================================================== #
#  3. Applied Processing                                                   #
# ====================================================================== #

@log_operation(logger)
def apply_frequency_filter(
    image: np.ndarray,
    filter_mask: np.ndarray,
) -> np.ndarray:
    """Full pipeline: FFT -> multiply by filter mask -> inverse FFT.

    Args:
        image:       Input image (grayscale or BGR).
        filter_mask: ``float64`` frequency filter of matching shape, or it
                     will be resized to the padded FFT shape.

    Returns:
        Filtered ``uint8`` image.
    """
    validate_image(image)
    fft = compute_fft(image)
    spec = fft.complex_spectrum

    # Resize filter if shapes differ (e.g. user provided original-size mask)
    if filter_mask.shape != spec.shape:
        filter_mask = cv2.resize(
            filter_mask,
            (spec.shape[1], spec.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    # Apply filter in frequency domain
    filtered_spec = spec * filter_mask
    fft.complex_spectrum = filtered_spec

    return inverse_fft(fft)


@log_operation(logger)
def remove_periodic_pattern(
    image: np.ndarray,
    min_distance: int = 10,
    num_peaks: int = 5,
    radius: int = 8,
) -> np.ndarray:
    """Automatically detect and remove periodic patterns from an image.

    Workflow:
        1. Compute FFT magnitude spectrum.
        2. Suppress DC component area.
        3. Find the top *num_peaks* peaks at least *min_distance* pixels
           from the center.
        4. Create a notch filter at those peaks.
        5. Apply the filter and return the cleaned image.

    This is especially useful for textured surface inspection (fabric,
    metal grain, printed circuit boards, etc.).

    Args:
        image:        Input image (grayscale or BGR).
        min_distance: Minimum distance from DC to consider a peak.
        num_peaks:    Number of top peaks to suppress.
        radius:       Radius of each notch hole.

    Returns:
        Cleaned ``uint8`` image with periodic patterns removed.
    """
    validate_image(image)
    fft = compute_fft(image)
    spec_shape = fft.complex_spectrum.shape

    # Work on the magnitude for peak detection
    mag = np.abs(fft.complex_spectrum).astype(np.float64)
    mag = np.log1p(mag)

    crow, ccol = spec_shape[0] // 2, spec_shape[1] // 2

    # Zero-out DC neighborhood so it isn't picked as a peak
    dc_radius = max(min_distance, 5)
    y_lo = max(0, crow - dc_radius)
    y_hi = min(spec_shape[0], crow + dc_radius + 1)
    x_lo = max(0, ccol - dc_radius)
    x_hi = min(spec_shape[1], ccol + dc_radius + 1)
    mag_search = mag.copy()
    mag_search[y_lo:y_hi, x_lo:x_hi] = 0.0

    # Find peaks iteratively (simple greedy approach)
    centers: List[Tuple[int, int]] = []
    for _ in range(num_peaks):
        idx = np.unravel_index(np.argmax(mag_search), mag_search.shape)
        peak_val = mag_search[idx]
        if peak_val <= 0:
            break
        # Store relative to center
        cu = int(idx[0]) - crow
        cv = int(idx[1]) - ccol
        centers.append((cu, cv))
        # Suppress this peak area so it isn't found again
        r_lo = max(0, int(idx[0]) - radius)
        r_hi = min(spec_shape[0], int(idx[0]) + radius + 1)
        c_lo = max(0, int(idx[1]) - radius)
        c_hi = min(spec_shape[1], int(idx[1]) + radius + 1)
        mag_search[r_lo:r_hi, c_lo:c_hi] = 0.0

    if not centers:
        logger.warning("remove_periodic_pattern: no peaks found, returning original")
        return _ensure_uint8(_ensure_gray(image).astype(np.float64))

    logger.info("Detected %d periodic peaks for notch filtering", len(centers))

    # Create and apply notch filter
    notch = create_notch_filter(spec_shape, centers, radius=radius)
    fft.complex_spectrum = fft.complex_spectrum * notch

    return inverse_fft(fft)


@log_operation(logger)
def frequency_defect_detection(
    image: np.ndarray,
    reference_spectrum: Optional[np.ndarray] = None,
    threshold: float = 3.0,
) -> np.ndarray:
    """Detect frequency-domain anomalies by comparing against a reference.

    If *reference_spectrum* is ``None``, a simple self-referencing approach
    is used: local statistics of the magnitude spectrum are computed and
    pixels deviating more than ``threshold * std`` are flagged.

    When a *reference_spectrum* (complex, same shape) is provided the
    difference in log-magnitude is computed and thresholded.

    Args:
        image:              Input image.
        reference_spectrum: Optional complex reference spectrum (centered).
        threshold:          Number of standard deviations for anomaly
                            flagging.

    Returns:
        ``uint8`` difference / anomaly map (0 = normal, 255 = anomaly).
    """
    validate_image(image)
    fft = compute_fft(image)
    mag = np.log1p(np.abs(fft.complex_spectrum))

    if reference_spectrum is not None:
        # Resize reference if needed
        ref = reference_spectrum
        if ref.shape != fft.complex_spectrum.shape:
            # Cannot meaningfully resize complex; warn and fall back
            logger.warning(
                "Reference spectrum shape %s != image spectrum shape %s; "
                "falling back to self-referencing mode.",
                ref.shape,
                fft.complex_spectrum.shape,
            )
            reference_spectrum = None

    if reference_spectrum is not None:
        ref_mag = np.log1p(np.abs(reference_spectrum))
        diff = np.abs(mag - ref_mag)
    else:
        # Self-referencing: compare each pixel to global statistics
        mean_val = mag.mean()
        std_val = mag.std()
        diff = np.abs(mag - mean_val)

    # Threshold
    std_diff = diff.std()
    mean_diff = diff.mean()
    anomaly = np.zeros_like(diff, dtype=np.uint8)
    anomaly[diff > mean_diff + threshold * std_diff] = 255

    # Crop to original shape
    orig_h, orig_w = fft.shape
    anomaly = anomaly[:orig_h, :orig_w]

    return anomaly


@log_operation(logger)
def compute_power_spectrum(image: np.ndarray) -> np.ndarray:
    """Compute the power spectrum |F(u,v)|^2 of *image*.

    Args:
        image: Input image (grayscale or BGR).

    Returns:
        Normalized power spectrum as ``float64`` image in [0, 1].
    """
    validate_image(image)
    fft = compute_fft(image)
    power = np.abs(fft.complex_spectrum) ** 2
    # Log scale for visibility
    power = np.log1p(power)
    p_min, p_max = power.min(), power.max()
    if p_max - p_min > 0:
        power = (power - p_min) / (p_max - p_min)
    else:
        power = np.zeros_like(power)
    return power.astype(np.float64)


@log_operation(logger)
def compute_phase_correlation(
    image1: np.ndarray,
    image2: np.ndarray,
) -> Tuple[float, float]:
    """Estimate sub-pixel translation between two images via phase correlation.

    Args:
        image1: First image.
        image2: Second image (same size as *image1*).

    Returns:
        ``(dx, dy)`` shift in pixels (float).  Positive dx means *image2*
        is shifted right relative to *image1*.
    """
    validate_image(image1, "image1")
    validate_image(image2, "image2")

    gray1 = _ensure_gray(image1).astype(np.float64)
    gray2 = _ensure_gray(image2).astype(np.float64)

    # Ensure same shape
    h = min(gray1.shape[0], gray2.shape[0])
    w = min(gray1.shape[1], gray2.shape[1])
    gray1 = gray1[:h, :w]
    gray2 = gray2[:h, :w]

    # FFT of both
    F1 = np.fft.fft2(gray1)
    F2 = np.fft.fft2(gray2)

    # Cross-power spectrum
    cross = F1 * np.conj(F2)
    cross_mag = np.abs(cross)
    cross_mag[cross_mag == 0] = 1e-10
    normalized = cross / cross_mag

    # Inverse FFT to get correlation surface
    corr = np.fft.ifft2(normalized)
    corr = np.abs(corr)

    # Find peak
    peak_idx = np.unravel_index(np.argmax(corr), corr.shape)
    dy = float(peak_idx[0])
    dx = float(peak_idx[1])

    # Wrap around: if shift > half the image, it is negative
    if dy > h / 2:
        dy -= h
    if dx > w / 2:
        dx -= w

    logger.info("Phase correlation shift: dx=%.2f, dy=%.2f", dx, dy)
    return (dx, dy)


# ====================================================================== #
#  4. Visualization Helpers                                                #
# ====================================================================== #

@log_operation(logger)
def draw_spectrum(
    magnitude: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Apply a colormap to a magnitude spectrum for display.

    Draws a center crosshair and basic frequency axis labels.

    Args:
        magnitude: ``uint8`` magnitude spectrum image.
        colormap:  OpenCV colormap constant (default ``COLORMAP_JET``).

    Returns:
        BGR ``uint8`` visualization image.
    """
    validate_image(magnitude)
    gray = _ensure_gray(magnitude)
    colored = cv2.applyColorMap(gray, colormap)

    h, w = colored.shape[:2]
    cx, cy = w // 2, h // 2

    # Center crosshair
    color = (255, 255, 255)
    cv2.line(colored, (cx - 15, cy), (cx + 15, cy), color, 1, cv2.LINE_AA)
    cv2.line(colored, (cx, cy - 15), (cx, cy + 15), color, 1, cv2.LINE_AA)

    # Frequency axis labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(colored, "DC", (cx + 5, cy - 5), font, 0.4, color, 1, cv2.LINE_AA)
    cv2.putText(colored, "+u", (w - 30, cy - 5), font, 0.35, color, 1, cv2.LINE_AA)
    cv2.putText(colored, "-u", (5, cy - 5), font, 0.35, color, 1, cv2.LINE_AA)
    cv2.putText(colored, "+v", (cx + 5, h - 10), font, 0.35, color, 1, cv2.LINE_AA)
    cv2.putText(colored, "-v", (cx + 5, 15), font, 0.35, color, 1, cv2.LINE_AA)

    return colored


@log_operation(logger)
def draw_filter_response(filter_mask: np.ndarray) -> np.ndarray:
    """Visualize a frequency filter mask as a colored image.

    Args:
        filter_mask: ``float64`` filter mask with values in [0, 1].

    Returns:
        BGR ``uint8`` visualization image.
    """
    display = (filter_mask * 255.0).astype(np.uint8)
    return cv2.applyColorMap(display, cv2.COLORMAP_VIRIDIS)


@log_operation(logger)
def overlay_spectrum_on_image(
    image: np.ndarray,
    magnitude: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Blend a magnitude spectrum on top of the original image.

    Both are resized to match if needed, and the magnitude is colormapped
    before blending.

    Args:
        image:     Original image (grayscale or BGR).
        magnitude: ``uint8`` magnitude spectrum.
        alpha:     Blending weight for the spectrum overlay (0-1).

    Returns:
        BGR ``uint8`` blended image.
    """
    validate_image(image)
    validate_image(magnitude, "magnitude")

    # Ensure BGR
    if image.ndim == 2:
        base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        base = image.copy()

    # Colormap the magnitude
    mag_gray = _ensure_gray(magnitude)
    mag_color = cv2.applyColorMap(mag_gray, cv2.COLORMAP_JET)

    # Resize spectrum to match image
    h, w = base.shape[:2]
    if mag_color.shape[:2] != (h, w):
        mag_color = cv2.resize(mag_color, (w, h), interpolation=cv2.INTER_LINEAR)

    blended = cv2.addWeighted(base, 1.0 - alpha, mag_color, alpha, 0)
    return blended
