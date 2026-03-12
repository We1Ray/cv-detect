"""Inference pipeline for single-image and batch anomaly detection.

Loads a trained checkpoint, runs images through the autoencoder, computes
error maps, applies morphological post-processing, and returns structured
inspection results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import cv2
import numpy as np
import torch

from dl_anomaly.config import Config
from dl_anomaly.core.anomaly_scorer import AnomalyScorer
from dl_anomaly.core.autoencoder import AnomalyAutoencoder
from dl_anomaly.core.preprocessor import ImagePreprocessor
from dl_anomaly.pipeline.trainer import TrainingPipeline

logger = logging.getLogger(__name__)


# ======================================================================
# Result data class
# ======================================================================

@dataclass
class InspectionResult:
    """Container for the output of a single-image inspection."""

    original: np.ndarray          # uint8 (H, W, C) or (H, W)
    reconstruction: np.ndarray    # uint8, same shape as original
    error_map: np.ndarray         # float32 (H, W) in [0, 1]
    defect_mask: np.ndarray       # uint8 (H, W) binary mask (0 or 255)
    anomaly_score: float          # scalar image-level score
    is_defective: bool            # True when score > threshold
    defect_regions: List[Dict[str, Any]] = field(default_factory=list)


# ======================================================================
# Inference pipeline
# ======================================================================

class InferencePipeline:
    """Load a trained autoencoder and inspect images for anomalies.

    Parameters
    ----------
    checkpoint_path:
        Path to a ``.pt`` checkpoint saved by :class:`TrainingPipeline`.
    device:
        ``'cuda'`` or ``'cpu'``.  Defaults to the value stored in the
        checkpoint's config.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        device: Optional[str] = None,
    ) -> None:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        if device is None:
            # 自動偵測最佳裝置
            if torch.cuda.is_available():
                dev = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                dev = "mps"
            else:
                dev = "cpu"
        else:
            dev = device
        self.model, self.config, state = TrainingPipeline.load_checkpoint(checkpoint_path, dev)
        self.device = torch.device(dev)
        self.model.to(self.device)
        self.model.eval()

        self.preprocessor = ImagePreprocessor(self.config.image_size, self.config.grayscale)
        self.scorer = AnomalyScorer(device=str(self.device))

        # Restore threshold from checkpoint if available
        if "threshold" in state:
            self.scorer.threshold = state["threshold"]
            logger.info("Loaded anomaly threshold: %.6f", self.scorer.threshold)
        else:
            logger.warning("No threshold found in checkpoint -- classify() will fail until fit_threshold() is called.")

    # ------------------------------------------------------------------
    # Single image
    # ------------------------------------------------------------------

    @torch.no_grad()
    def inspect_single(self, image_path: Union[str, Path]) -> InspectionResult:
        """Run the full inspection pipeline on a single image."""
        image_path = Path(image_path)

        # Preprocess
        tensor = self.preprocessor.load_and_preprocess(image_path)
        batch = tensor.unsqueeze(0).to(self.device)

        # Forward
        recon_batch = self.model(batch)

        # Back to numpy (uint8)
        original_np = self.preprocessor.inverse_normalize(tensor)
        recon_np = self.preprocessor.inverse_normalize(recon_batch.squeeze(0))

        # Error map
        error_map = self.scorer.compute_combined_error(
            original_np, recon_np, self.config.ssim_weight
        )
        smoothed = self.scorer.create_anomaly_map(error_map, gaussian_sigma=4.0)
        score = self.scorer.compute_image_score(error_map)

        # Classification
        is_defective = False
        if self.scorer.threshold is not None:
            is_defective = self.scorer.classify(score)

        # Binary mask via Otsu on the smoothed map
        mask = self._create_defect_mask(smoothed)

        # Connected-component analysis
        regions = self._extract_defect_regions(mask)

        return InspectionResult(
            original=original_np,
            reconstruction=recon_np,
            error_map=smoothed,
            defect_mask=mask,
            anomaly_score=score,
            is_defective=is_defective,
            defect_regions=regions,
        )

    # ------------------------------------------------------------------
    # Batch
    # ------------------------------------------------------------------

    def inspect_batch(
        self,
        image_dir: Union[str, Path],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[InspectionResult]:
        """Inspect every supported image in *image_dir*.

        Parameters
        ----------
        progress_callback:
            Called with ``(current_index, total_count)`` after each image.
        """
        image_dir = Path(image_dir)
        extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        paths = sorted(
            p for p in image_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in extensions
        )

        if not paths:
            logger.warning("No images found in %s", image_dir)
            return []

        results: List[InspectionResult] = []
        for idx, p in enumerate(paths):
            try:
                result = self.inspect_single(p)
                results.append(result)
            except Exception:
                logger.exception("Failed to inspect %s", p)

            if progress_callback is not None:
                progress_callback(idx + 1, len(paths))

        return results

    # ------------------------------------------------------------------
    # Post-processing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _create_defect_mask(anomaly_map: np.ndarray) -> np.ndarray:
        """Threshold + morphological cleanup -> binary mask (uint8, 0/255)."""
        # Convert to uint8 for OpenCV
        map_u8 = (anomaly_map * 255).astype(np.uint8)

        # Otsu threshold
        _, mask = cv2.threshold(map_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological cleanup: close small gaps, then remove tiny specks
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        return mask

    @staticmethod
    def _extract_defect_regions(mask: np.ndarray) -> List[Dict[str, Any]]:
        """Run connected-component analysis and return region descriptors."""
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        regions: List[Dict[str, Any]] = []
        for i in range(1, num_labels):  # skip background (label 0)
            x, y, w, h, area = stats[i]
            cx, cy = centroids[i]

            # Filter tiny noise regions
            if area < 20:
                continue

            regions.append(
                {
                    "id": len(regions) + 1,
                    "bbox": (int(x), int(y), int(w), int(h)),
                    "area": int(area),
                    "centroid": (float(cx), float(cy)),
                }
            )

        return regions
