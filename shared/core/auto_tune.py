"""Auto-tune -- automatic threshold calibration for anomaly detection methods.

Given OK (good) and NG (defective) sample image sets, finds the optimal
detection threshold that maximises classification performance.

The core :class:`AutoTuner` is method-agnostic: it accepts any callable that
maps a BGR uint8 image to a scalar anomaly score.  Convenience factory
functions are provided for PatchCore, Autoencoder, and ImageDifferencer
models so callers can tune thresholds in a single call.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


# ====================================================================== #
#  TuneResult                                                             #
# ====================================================================== #

@dataclass
class TuneResult:
    """Container for threshold-tuning results."""

    optimal_threshold: float
    best_score: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    ok_scores: List[float] = field(repr=False)
    ng_scores: List[float] = field(repr=False)
    thresholds: List[float] = field(repr=False)
    metric_values: List[float] = field(repr=False)


# ====================================================================== #
#  AutoTuner                                                              #
# ====================================================================== #

class AutoTuner:
    """Automatic threshold tuner for any detection method.

    Parameters
    ----------
    scorer:
        A callable that takes a BGR uint8 ``np.ndarray`` and returns a
        scalar anomaly score (higher = more anomalous).
    """

    def __init__(self, scorer: Callable[[np.ndarray], float]) -> None:
        self._scorer = scorer

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def tune(
        self,
        ok_dir: Union[str, Path],
        ng_dir: Union[str, Path],
        n_thresholds: int = 200,
        metric: str = "f1",
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> TuneResult:
        """Find the optimal threshold separating OK from NG images.

        Parameters
        ----------
        ok_dir:
            Directory of defect-free images (label 0).
        ng_dir:
            Directory of defective images (label 1).
        n_thresholds:
            Number of evenly-spaced thresholds to evaluate between the
            minimum and maximum observed anomaly score.
        metric:
            Optimisation target -- one of ``"f1"``, ``"precision"``,
            ``"recall"``, ``"balanced_accuracy"``, ``"youden_j"``.
        progress_callback:
            Optional ``(current, total) -> None`` callable invoked after
            each image is scored.  Useful for GUI progress bars.

        Returns
        -------
        TuneResult
            Dataclass containing the optimal threshold and detailed metrics.
        """
        ok_dir, ng_dir = Path(ok_dir), Path(ng_dir)
        ok_paths = self._list_images(ok_dir)
        ng_paths = self._list_images(ng_dir)
        total = len(ok_paths) + len(ng_paths)

        if not ok_paths:
            raise FileNotFoundError(f"No images found in OK directory: {ok_dir}")
        if not ng_paths:
            raise FileNotFoundError(f"No images found in NG directory: {ng_dir}")

        logger.info("Scoring %d OK + %d NG images", len(ok_paths), len(ng_paths))

        # --- Score all images ---
        ok_scores = self._score_images(
            self._scorer, ok_paths, progress_callback, start_idx=0, total=total,
        )
        ng_scores = self._score_images(
            self._scorer, ng_paths, progress_callback,
            start_idx=len(ok_paths), total=total,
        )

        # --- Build label / score arrays ---
        labels = np.array([0] * len(ok_scores) + [1] * len(ng_scores))
        scores = np.array(ok_scores + ng_scores)

        # --- Threshold sweep ---
        lo, hi = float(scores.min()), float(scores.max())
        if lo == hi:
            logger.warning("All anomaly scores are identical (%.4f); "
                           "cannot discriminate OK from NG", lo)
            hi = lo + 1.0
        thresholds = np.linspace(lo, hi, n_thresholds).tolist()

        best_val = -1.0
        best_thr = thresholds[0]
        metric_values: List[float] = []
        best_metrics: Dict[str, float] = {}

        for thr in thresholds:
            m = self._compute_metrics(labels, scores, thr)
            val = m[metric]
            metric_values.append(val)
            if val > best_val:
                best_val = val
                best_thr = thr
                best_metrics = m

        roc_auc = self._compute_roc_auc(labels, scores)

        logger.info(
            "Optimal threshold=%.4f  %s=%.4f  "
            "P=%.3f  R=%.3f  F1=%.3f  AUC=%.3f",
            best_thr, metric, best_val,
            best_metrics["precision"], best_metrics["recall"],
            best_metrics["f1"], roc_auc,
        )

        return TuneResult(
            optimal_threshold=best_thr,
            best_score=best_val,
            precision=best_metrics["precision"],
            recall=best_metrics["recall"],
            f1=best_metrics["f1"],
            roc_auc=roc_auc,
            ok_scores=ok_scores,
            ng_scores=ng_scores,
            thresholds=thresholds,
            metric_values=metric_values,
        )

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _list_images(directory: Path) -> List[Path]:
        """Return sorted list of image paths in *directory*."""
        return sorted(
            p for p in directory.iterdir()
            if p.suffix.lower() in _IMAGE_EXTENSIONS
        )

    @staticmethod
    def _score_images(
        scorer: Callable[[np.ndarray], float],
        paths: List[Path],
        progress_callback: Optional[Callable[[int, int], None]] = None,
        start_idx: int = 0,
        total: int = 0,
    ) -> List[float]:
        """Load and score a list of image files."""
        results: List[float] = []
        for i, path in enumerate(paths):
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                logger.warning("Failed to read image, skipping: %s", path)
                continue
            score = float(scorer(img))
            results.append(score)
            if progress_callback is not None:
                progress_callback(start_idx + i + 1, total)
        return results

    @staticmethod
    def _compute_metrics(
        labels: np.ndarray,
        scores: np.ndarray,
        threshold: float,
    ) -> Dict[str, float]:
        """Compute classification metrics at *threshold*.

        Prediction rule: ``score >= threshold`` -> predicted NG (positive).
        """
        preds = (scores >= threshold).astype(int)
        tp = int(np.sum((preds == 1) & (labels == 1)))
        fp = int(np.sum((preds == 1) & (labels == 0)))
        fn = int(np.sum((preds == 0) & (labels == 1)))
        tn = int(np.sum((preds == 0) & (labels == 0)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        tpr = recall
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        balanced_accuracy = (tpr + tnr) / 2.0
        youden_j = tpr + tnr - 1.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "balanced_accuracy": balanced_accuracy,
            "youden_j": youden_j,
        }

    @staticmethod
    def _compute_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
        """Compute ROC AUC via the trapezoidal rule (no sklearn needed)."""
        # Sort by descending score
        order = np.argsort(-scores)
        sorted_labels = labels[order]

        n_pos = int(np.sum(labels == 1))
        n_neg = int(np.sum(labels == 0))
        if n_pos == 0 or n_neg == 0:
            return 0.0

        tpr_list: List[float] = [0.0]
        fpr_list: List[float] = [0.0]
        tp = 0
        fp = 0

        for lab in sorted_labels:
            if lab == 1:
                tp += 1
            else:
                fp += 1
            tpr_list.append(tp / n_pos)
            fpr_list.append(fp / n_neg)

        # Trapezoidal integration
        auc = 0.0
        for i in range(1, len(fpr_list)):
            auc += (fpr_list[i] - fpr_list[i - 1]) * (tpr_list[i] + tpr_list[i - 1]) / 2.0
        return float(auc)


# ====================================================================== #
#  Convenience factory functions                                          #
# ====================================================================== #

def tune_from_patchcore(
    model_path: Union[str, Path],
    ok_dir: Union[str, Path],
    ng_dir: Union[str, Path],
    device: str = "auto",
    **kwargs,
) -> TuneResult:
    """Create a scorer from a PatchCore model file and tune the threshold.

    Parameters
    ----------
    model_path:
        Path to a ``.npz`` PatchCore model file.
    ok_dir / ng_dir:
        Directories of OK and NG sample images.
    device:
        ``"auto"``, ``"cuda"``, or ``"cpu"``.
    **kwargs:
        Forwarded to :meth:`AutoTuner.tune`.
    """
    import torch
    from shared.core.patchcore import PatchCoreInference, PatchCoreModel

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = PatchCoreModel.load(model_path)
    engine = PatchCoreInference(model, device=device)

    from dl_anomaly.core.preprocessor import ImagePreprocessor
    preprocessor = ImagePreprocessor(image_size=model.image_size)

    def scorer(image: np.ndarray) -> float:
        tensor = preprocessor.preprocess(image)
        score, _ = engine.score_image(tensor)
        return float(score)

    tuner = AutoTuner(scorer)
    return tuner.tune(ok_dir, ng_dir, **kwargs)


def tune_from_autoencoder(
    checkpoint_path: Union[str, Path],
    ok_dir: Union[str, Path],
    ng_dir: Union[str, Path],
    device: str = "auto",
    **kwargs,
) -> TuneResult:
    """Create a scorer from an Autoencoder checkpoint and tune the threshold.

    Parameters
    ----------
    checkpoint_path:
        Path to a ``.pth`` Autoencoder checkpoint.
    ok_dir / ng_dir:
        Directories of OK and NG sample images.
    device:
        ``"auto"``, ``"cuda"``, or ``"cpu"``.
    **kwargs:
        Forwarded to :meth:`AutoTuner.tune`.
    """
    import torch
    from dl_anomaly.core.autoencoder import AnomalyAutoencoder
    from dl_anomaly.core.preprocessor import ImagePreprocessor

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    ae = AnomalyAutoencoder(**ckpt.get("model_config", {}))
    ae.load_state_dict(ckpt["model_state_dict"])
    ae.to(device).eval()

    image_size = ckpt.get("model_config", {}).get("image_size", 256)
    preprocessor = ImagePreprocessor(image_size=image_size)

    def scorer(image: np.ndarray) -> float:
        tensor = preprocessor.preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            recon = ae(tensor)
        error = torch.mean((tensor - recon) ** 2).item()
        return float(error)

    tuner = AutoTuner(scorer)
    return tuner.tune(ok_dir, ng_dir, **kwargs)


def tune_from_difference(
    reference_image: Union[str, Path, np.ndarray],
    ok_dir: Union[str, Path],
    ng_dir: Union[str, Path],
    **kwargs,
) -> TuneResult:
    """Create a scorer from an :class:`ImageDifferencer` and tune the threshold.

    Parameters
    ----------
    reference_image:
        File path or BGR uint8 array of the golden reference image.
    ok_dir / ng_dir:
        Directories of OK and NG sample images.
    **kwargs:
        Forwarded to :meth:`AutoTuner.tune`.
    """
    from shared.core.image_difference import ImageDifferencer

    differ = ImageDifferencer()

    if isinstance(reference_image, (str, Path)):
        ref = cv2.imread(str(reference_image), cv2.IMREAD_COLOR)
        if ref is None:
            raise FileNotFoundError(
                f"Cannot read reference image: {reference_image}"
            )
    else:
        ref = reference_image

    differ.set_reference(ref)

    def scorer(image: np.ndarray) -> float:
        result = differ.detect(image)
        return float(result["anomaly_score"])

    tuner = AutoTuner(scorer)
    return tuner.tune(ok_dir, ng_dir, **kwargs)
