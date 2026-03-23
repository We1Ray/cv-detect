"""Photometric stereo for surface defect enhancement.

Uses images captured under different illumination directions to recover
surface normals and albedo, revealing subtle surface defects invisible
under single-directional lighting.

Key components
--------------
- ``LightDirection`` -- 3D light vector specification.
- ``PhotometricResult`` -- surface normals, albedo, gradient maps.
- ``PhotometricStereo`` -- core algorithm with multiple methods.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LightDirection:
    """3D light source direction (unit vector)."""
    x: float
    y: float
    z: float

    def to_array(self) -> np.ndarray:
        v = np.array([self.x, self.y, self.z], dtype=np.float64)
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    @classmethod
    def from_angle(cls, azimuth_deg: float, elevation_deg: float) -> "LightDirection":
        """Create from azimuth (0-360) and elevation (0-90) angles."""
        az = math.radians(azimuth_deg)
        el = math.radians(elevation_deg)
        return cls(
            x=math.cos(el) * math.cos(az),
            y=math.cos(el) * math.sin(az),
            z=math.sin(el),
        )


@dataclass
class PhotometricResult:
    """Result of photometric stereo computation."""
    normal_map: np.ndarray       # (H, W, 3) surface normals
    albedo_map: np.ndarray       # (H, W) reflectance/albedo
    gradient_x: np.ndarray       # (H, W) surface gradient in x
    gradient_y: np.ndarray       # (H, W) surface gradient in y
    depth_map: Optional[np.ndarray] = None  # (H, W) integrated depth
    mean_curvature: Optional[np.ndarray] = None  # (H, W) local curvature

    def normal_to_rgb(self) -> np.ndarray:
        """Convert normal map to displayable RGB image."""
        rgb = ((self.normal_map + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        return rgb

    def gradient_magnitude(self) -> np.ndarray:
        """Compute gradient magnitude (useful for defect detection)."""
        return np.sqrt(self.gradient_x ** 2 + self.gradient_y ** 2).astype(np.float32)

    def curvature_map(self) -> np.ndarray:
        """Compute mean curvature from gradient maps."""
        gxx = cv2.Sobel(self.gradient_x, cv2.CV_64F, 1, 0, ksize=3)
        gyy = cv2.Sobel(self.gradient_y, cv2.CV_64F, 0, 1, ksize=3)
        return ((gxx + gyy) / 2.0).astype(np.float32)


class PhotometricStereo:
    """Photometric stereo surface reconstruction.

    Parameters
    ----------
    method : str
        "least_squares" (classic), "robust" (Huber-weighted), or "rpca"
        (Robust PCA for specular highlights).
    integrate_depth : bool
        If True, integrate gradients to produce a depth map.
    """

    def __init__(
        self,
        method: str = "least_squares",
        integrate_depth: bool = False,
    ) -> None:
        if method not in ("least_squares", "robust", "rpca"):
            raise ValueError(f"Unknown method: {method}")
        self._method = method
        self._integrate_depth = integrate_depth

    def compute(
        self,
        images: List[np.ndarray],
        light_dirs: List[LightDirection],
        mask: Optional[np.ndarray] = None,
    ) -> PhotometricResult:
        """Compute photometric stereo from multiple images.

        Parameters
        ----------
        images : list of ndarray (H, W) or (H, W, 3)
            Images captured under different lighting. Must be same size.
            If color, they are converted to grayscale.
        light_dirs : list of LightDirection
            Corresponding light directions. Must have len >= 3.
        mask : ndarray (H, W), optional
            Binary mask of pixels to process.
        """
        if len(images) < 3:
            raise ValueError(f"Need at least 3 images, got {len(images)}")
        if len(images) != len(light_dirs):
            raise ValueError("Number of images must match number of light directions")

        # Convert to grayscale float64
        gray_images = []
        for img in images:
            if img.ndim == 3:
                g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                g = img.copy()
            gray_images.append(g.astype(np.float64))

        h, w = gray_images[0].shape
        n = len(gray_images)

        # Build light matrix L (n, 3)
        L = np.array([ld.to_array() for ld in light_dirs], dtype=np.float64)

        # Build intensity matrix I (n, num_pixels)
        if mask is not None:
            pixel_mask = mask.astype(bool).ravel()
        else:
            pixel_mask = np.ones(h * w, dtype=bool)

        I = np.zeros((n, int(pixel_mask.sum())), dtype=np.float64)
        for i, g in enumerate(gray_images):
            I[i] = g.ravel()[pixel_mask]

        # Solve for normals
        if self._method == "least_squares":
            normals_flat = self._solve_least_squares(L, I)
        elif self._method == "robust":
            normals_flat = self._solve_robust(L, I)
        else:
            normals_flat = self._solve_rpca(L, I)

        # Extract albedo and normalize normals
        albedo_flat = np.linalg.norm(normals_flat, axis=0)
        albedo_flat = np.where(albedo_flat > 1e-10, albedo_flat, 1e-10)
        normals_flat = normals_flat / albedo_flat[np.newaxis, :]

        # Reconstruct full-size maps
        normal_map = np.zeros((h * w, 3), dtype=np.float64)
        normal_map[pixel_mask] = normals_flat.T
        normal_map = normal_map.reshape(h, w, 3)

        albedo_map = np.zeros(h * w, dtype=np.float64)
        albedo_map[pixel_mask] = albedo_flat
        albedo_map = albedo_map.reshape(h, w)

        # Normalize albedo to [0, 1]
        amax = albedo_map.max()
        if amax > 0:
            albedo_map /= amax

        # Compute gradients (p = -nx/nz, q = -ny/nz)
        nz = normal_map[:, :, 2]
        nz_safe = np.where(np.abs(nz) > 1e-10, nz, np.sign(nz) * 1e-10)
        # Handle exact-zero case (sign returns 0 for 0)
        nz_safe = np.where(nz_safe == 0, 1e-10, nz_safe)
        gradient_x = (-normal_map[:, :, 0] / nz_safe).astype(np.float32)
        gradient_y = (-normal_map[:, :, 1] / nz_safe).astype(np.float32)

        result = PhotometricResult(
            normal_map=normal_map.astype(np.float32),
            albedo_map=albedo_map.astype(np.float32),
            gradient_x=gradient_x,
            gradient_y=gradient_y,
        )

        if self._integrate_depth:
            result.depth_map = self._integrate_gradients(gradient_x, gradient_y)

        result.mean_curvature = result.curvature_map()
        logger.info("Photometric stereo: %s method, %d images, (%d x %d)", self._method, n, w, h)
        return result

    def _solve_least_squares(self, L: np.ndarray, I: np.ndarray) -> np.ndarray:
        """Classic least-squares: N = (L^T L)^-1 L^T I"""
        return np.linalg.lstsq(L, I, rcond=None)[0]  # (3, num_pixels)

    def _solve_robust(self, L: np.ndarray, I: np.ndarray, iterations: int = 5) -> np.ndarray:
        """Iteratively Reweighted Least Squares with Huber loss."""
        N = self._solve_least_squares(L, I)
        delta = 1.345  # Huber threshold

        for _ in range(iterations):
            residuals = I - L @ N
            abs_res = np.abs(residuals)
            weights = np.where(abs_res <= delta, 1.0, delta / (abs_res + 1e-10))

            # Weighted least squares (element-wise multiply avoids O(n^2) diag matrix)
            for j in range(N.shape[1]):
                W_j = weights[:, j]              # (n,)
                LW = L * W_j[:, np.newaxis]      # (n, 3) -- weighted L
                LtWL = LW.T @ L                  # (3, 3)
                LtWI = LW.T @ I[:, j]            # (3,)
                try:
                    N[:, j] = np.linalg.solve(LtWL, LtWI)
                except np.linalg.LinAlgError:
                    pass  # Keep previous estimate
        return N

    def _solve_rpca(self, L: np.ndarray, I: np.ndarray) -> np.ndarray:
        """Robust PCA to separate diffuse and specular components."""
        # Simplified RPCA via iterative thresholding
        S = np.zeros_like(I)  # Sparse (specular) component
        lam = 1.0 / np.sqrt(max(I.shape))

        for _ in range(10):
            D = I - S
            N = np.linalg.lstsq(L, D, rcond=None)[0]
            D_approx = L @ N
            residual = I - D_approx
            S = np.sign(residual) * np.maximum(np.abs(residual) - lam, 0)

        return N

    @staticmethod
    def _integrate_gradients(p: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Integrate gradient field to depth via Poisson solver (FFT-based)."""
        h, w = p.shape
        # Compute divergence
        dpx = np.zeros_like(p)
        dqy = np.zeros_like(q)
        dpx[:, 1:] = p[:, 1:] - p[:, :-1]
        dqy[1:, :] = q[1:, :] - q[:-1, :]
        div = dpx + dqy

        # Solve Poisson equation in frequency domain
        div_fft = np.fft.fft2(div)
        u = np.arange(h).reshape(-1, 1)
        v = np.arange(w).reshape(1, -1)
        denom = (2 * np.cos(2 * np.pi * u / h) - 2) + (2 * np.cos(2 * np.pi * v / w) - 2)
        denom[0, 0] = 1.0  # Avoid division by zero
        depth_fft = div_fft / denom
        depth_fft[0, 0] = 0  # Set mean to zero
        depth = np.real(np.fft.ifft2(depth_fft)).astype(np.float32)
        return depth

    @staticmethod
    def create_standard_4_light(elevation_deg: float = 45.0) -> List[LightDirection]:
        """Create standard 4-directional light setup (N, E, S, W)."""
        return [
            LightDirection.from_angle(0, elevation_deg),
            LightDirection.from_angle(90, elevation_deg),
            LightDirection.from_angle(180, elevation_deg),
            LightDirection.from_angle(270, elevation_deg),
        ]

    @staticmethod
    def create_standard_8_light(elevation_deg: float = 45.0) -> List[LightDirection]:
        """Create standard 8-directional light setup."""
        return [LightDirection.from_angle(i * 45, elevation_deg) for i in range(8)]
