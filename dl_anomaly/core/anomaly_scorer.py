"""Anomaly scoring utilities.

Computes per-pixel error maps (MSE, SSIM, combined), image-level scores,
threshold fitting, and smoothed anomaly maps for visualisation.

When a GPU device is provided, expensive operations (SSIM, gaussian
smoothing) run on the GPU via PyTorch for significant speed-ups.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class AnomalyScorer:
    """Stateful scorer that learns a threshold from training-set errors
    and classifies new images as normal / anomalous.

    Parameters
    ----------
    device : str or torch.device, optional
        ``'cuda'`` or ``'cpu'``.  When ``'cuda'`` is available, SSIM and
        gaussian smoothing are computed on the GPU.
    """

    def __init__(self, device: str = "cpu") -> None:
        self.threshold: Optional[float] = None
        self.device = torch.device(device)

        # Cached gaussian kernels  (key: (window_size, channels))
        self._ssim_kernel_cache: dict[Tuple[int, int], torch.Tensor] = {}
        self._gauss_kernel_cache: dict[Tuple[int, float], torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_ssim_kernel(self, win_size: int, channels: int) -> torch.Tensor:
        """Return a cached gaussian kernel for SSIM computation."""
        key = (win_size, channels)
        if key not in self._ssim_kernel_cache:
            coords = torch.arange(win_size, dtype=torch.float32) - win_size // 2
            g1d = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
            g1d /= g1d.sum()
            g2d = g1d.unsqueeze(1) @ g1d.unsqueeze(0)
            kernel = g2d.expand(channels, 1, win_size, win_size).contiguous()
            self._ssim_kernel_cache[key] = kernel.to(self.device)
        return self._ssim_kernel_cache[key]

    def _get_gauss_kernel(self, sigma: float) -> torch.Tensor:
        """Return a cached 2D gaussian kernel for anomaly-map smoothing."""
        ksize = max(int(sigma * 6) | 1, 3)
        key = (ksize, sigma)
        if key not in self._gauss_kernel_cache:
            coords = torch.arange(ksize, dtype=torch.float32) - ksize // 2
            g1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
            g1d /= g1d.sum()
            g2d = g1d.unsqueeze(1) @ g1d.unsqueeze(0)
            kernel = g2d.unsqueeze(0).unsqueeze(0)  # (1, 1, K, K)
            self._gauss_kernel_cache[key] = kernel.to(self.device)
        return self._gauss_kernel_cache[key]

    # ------------------------------------------------------------------
    # Per-pixel error maps
    # ------------------------------------------------------------------

    @staticmethod
    def compute_pixel_error(original: np.ndarray, reconstruction: np.ndarray) -> np.ndarray:
        """Mean-squared error across the channel axis.

        Parameters
        ----------
        original, reconstruction:
            ``(H, W)`` or ``(H, W, C)`` uint8 arrays.

        Returns
        -------
        np.ndarray
            ``(H, W)`` error map in [0, 1].
        """
        diff = (original.astype(np.float32) - reconstruction.astype(np.float32)) ** 2
        if diff.ndim == 3:
            diff = diff.mean(axis=2)
        max_val = 255.0 ** 2 if original.max() > 1.0 else 1.0
        return (diff / max_val).astype(np.float32)

    def compute_ssim_map(self, original: np.ndarray, reconstruction: np.ndarray) -> np.ndarray:
        """Return ``1 - SSIM`` map computed on the GPU when available.

        Both inputs must be uint8 ``(H, W)`` or ``(H, W, C)``.
        """
        # Convert to float32 tensors
        orig_f = original.astype(np.float32) / 255.0
        recon_f = reconstruction.astype(np.float32) / 255.0

        if orig_f.ndim == 2:
            orig_f = orig_f[np.newaxis, np.newaxis, :, :]  # (1, 1, H, W)
            recon_f = recon_f[np.newaxis, np.newaxis, :, :]
            channels = 1
        else:
            # (H, W, C) -> (1, C, H, W)
            orig_f = np.transpose(orig_f, (2, 0, 1))[np.newaxis, :]
            recon_f = np.transpose(recon_f, (2, 0, 1))[np.newaxis, :]
            channels = orig_f.shape[1]

        x = torch.from_numpy(orig_f).to(self.device)
        y = torch.from_numpy(recon_f).to(self.device)

        min_dim = min(x.shape[2], x.shape[3])
        win_size = min(7, min_dim)
        if win_size % 2 == 0:
            win_size -= 1
        win_size = max(win_size, 3)

        kernel = self._get_ssim_kernel(win_size, channels)
        pad = win_size // 2

        mu_x = F.conv2d(x, kernel, padding=pad, groups=channels)
        mu_y = F.conv2d(y, kernel, padding=pad, groups=channels)

        sigma_x_sq = F.conv2d(x * x, kernel, padding=pad, groups=channels) - mu_x ** 2
        sigma_y_sq = F.conv2d(y * y, kernel, padding=pad, groups=channels) - mu_y ** 2
        sigma_xy = F.conv2d(x * y, kernel, padding=pad, groups=channels) - mu_x * mu_y

        C1 = (0.01) ** 2
        C2 = (0.03) ** 2

        ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
            (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x_sq + sigma_y_sq + C2)
        )

        # Average across channels, invert
        ssim_map = ssim_map.mean(dim=1, keepdim=True)  # (1, 1, H, W)
        anomaly_map = (1.0 - ssim_map).clamp(0.0, 1.0)

        return anomaly_map.squeeze().cpu().numpy().astype(np.float32)

    def compute_combined_error(
        self,
        original: np.ndarray,
        reconstruction: np.ndarray,
        ssim_weight: float = 0.5,
    ) -> np.ndarray:
        """Weighted combination of pixel MSE and (1-SSIM) maps."""
        mse_map = self.compute_pixel_error(original, reconstruction)
        if ssim_weight <= 0.0:
            return mse_map
        ssim_map = self.compute_ssim_map(original, reconstruction)
        combined = (1.0 - ssim_weight) * mse_map + ssim_weight * ssim_map
        return combined.astype(np.float32)

    # ------------------------------------------------------------------
    # Image-level score
    # ------------------------------------------------------------------

    @staticmethod
    def compute_image_score(error_map: np.ndarray) -> float:
        """Scalar anomaly score: mean of the error map."""
        return float(np.mean(error_map))

    # ------------------------------------------------------------------
    # Threshold
    # ------------------------------------------------------------------

    def fit_threshold(self, training_errors: List[float], percentile: float = 95.0) -> float:
        """Fit the anomaly threshold as the given percentile of *training_errors*.

        Returns the computed threshold and stores it internally.
        """
        if not training_errors:
            raise ValueError("Cannot fit threshold with an empty error list.")
        self.threshold = float(np.percentile(training_errors, percentile))
        logger.info(
            "Anomaly threshold set to %.6f (percentile=%.1f, n=%d)",
            self.threshold,
            percentile,
            len(training_errors),
        )
        return self.threshold

    def classify(self, score: float) -> bool:
        """Return *True* if *score* exceeds the fitted threshold (i.e. anomalous)."""
        if self.threshold is None:
            raise RuntimeError("Threshold has not been fitted yet. Call fit_threshold() first.")
        return score > self.threshold

    # ------------------------------------------------------------------
    # Anomaly map for visualisation
    # ------------------------------------------------------------------

    def create_anomaly_map(self, error_map: np.ndarray, gaussian_sigma: float = 4.0) -> np.ndarray:
        """Gaussian-smoothed and min-max-normalised anomaly map in [0, 1].

        Uses GPU-accelerated gaussian filtering when available.
        """
        kernel = self._get_gauss_kernel(gaussian_sigma)
        pad = kernel.shape[-1] // 2

        t = torch.from_numpy(error_map.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)
        smoothed = F.conv2d(t, kernel, padding=pad)
        smoothed = smoothed.squeeze()

        vmin = smoothed.min()
        vmax = smoothed.max()
        if (vmax - vmin) < 1e-8:
            return np.zeros_like(error_map, dtype=np.float32)
        normalised = (smoothed - vmin) / (vmax - vmin)
        return normalised.cpu().numpy().astype(np.float32)
