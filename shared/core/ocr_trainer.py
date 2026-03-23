"""
core/ocr_trainer.py - Custom OCR font training for industrial inspection.

Provides tools to train character recognition models on domain-specific fonts
(e.g. dot-matrix date codes, laser-etched lot numbers) where general-purpose
OCR engines underperform.  Supports both feature-based (HOG + KNN) and
DL-based (simple CNN) backends, plus Tesseract training-data export.

Categories:
    1. Data Classes (OCRTrainingConfig, OCRSample)
    2. Character Extraction
    3. Auto-Labelling via Template Matching
    4. Font Trainer (training, evaluation, persistence)
    5. Font Inference
    6. Tesseract Export Helpers
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ====================================================================== #
#  Named constants                                                        #
# ====================================================================== #

DEFAULT_IMAGE_SIZE: Tuple[int, int] = (32, 32)
DEFAULT_HOG_WIN_SIZE: Tuple[int, int] = (32, 32)
DEFAULT_HOG_BLOCK_SIZE: Tuple[int, int] = (16, 16)
DEFAULT_HOG_BLOCK_STRIDE: Tuple[int, int] = (8, 8)
DEFAULT_HOG_CELL_SIZE: Tuple[int, int] = (8, 8)
DEFAULT_HOG_NBINS: int = 9
DEFAULT_KNN_K: int = 3
MIN_CONTOUR_AREA: int = 20
MATCH_THRESHOLD: float = 0.25


class BackendType(Enum):
    """Recognition backend selection."""
    HOG_KNN = "hog_knn"
    CNN = "cnn"


# ====================================================================== #
#  Data Classes                                                           #
# ====================================================================== #


@dataclass
class OCRTrainingConfig:
    """Configuration for OCR font training."""
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE
    num_classes: int = 62  # 0-9 + A-Z + a-z
    font_name: str = "custom"
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3
    backend: BackendType = BackendType.HOG_KNN
    knn_k: int = DEFAULT_KNN_K
    augment: bool = True


@dataclass
class OCRSample:
    """A single training / inference sample."""
    image: np.ndarray                           # grayscale character image
    label: str = ""                             # character label
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # (x, y, w, h) in source


@dataclass
class OCRResult:
    """Recognition result for a single character."""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)


# ====================================================================== #
#  Character Extraction                                                   #
# ====================================================================== #


class CharacterExtractor:
    """Segment individual characters from an image region.

    Parameters
    ----------
    min_area:
        Minimum contour area to keep (filters noise).
    padding:
        Extra pixels around each bounding box.
    """

    def __init__(self, min_area: int = MIN_CONTOUR_AREA, padding: int = 2) -> None:
        self.min_area = min_area
        self.padding = padding

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def extract_characters(
        self,
        image: np.ndarray,
        method: str = "connected_component",
    ) -> List[OCRSample]:
        """Segment individual characters from *image*.

        Parameters
        ----------
        image:
            Grayscale or BGR image containing a text region.
        method:
            ``"connected_component"`` (default) or ``"contour"``.

        Returns
        -------
        List[OCRSample]
            Sorted left-to-right by bounding-box x position.
        """
        gray = self._to_gray(image)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 4,
        )

        if method == "connected_component":
            samples = self._extract_cc(gray, binary)
        elif method == "contour":
            samples = self._extract_contour(gray, binary)
        else:
            raise ValueError(f"Unknown extraction method: {method!r}")

        # Sort left-to-right
        samples.sort(key=lambda s: s.bbox[0])
        logger.debug("Extracted %d characters via %s", len(samples), method)
        return samples

    def auto_label(
        self,
        characters: List[OCRSample],
        reference_font: Dict[str, np.ndarray],
        threshold: float = MATCH_THRESHOLD,
    ) -> List[OCRSample]:
        """Assign labels using template matching against *reference_font*.

        Parameters
        ----------
        characters:
            Unlabelled samples (from :meth:`extract_characters`).
        reference_font:
            Mapping ``{label: template_image}`` of canonical character images.
        threshold:
            Minimum normalised match score (0-1) to accept a label.

        Returns
        -------
        List[OCRSample]
            Same samples with ``.label`` populated where a match was found.
        """
        labelled: List[OCRSample] = []
        for sample in characters:
            best_label = ""
            best_score = -1.0
            resized = cv2.resize(sample.image, DEFAULT_IMAGE_SIZE)
            for label, template in reference_font.items():
                tmpl = cv2.resize(template, DEFAULT_IMAGE_SIZE)
                result = cv2.matchTemplate(resized, tmpl, cv2.TM_CCOEFF_NORMED)
                score = float(result.max())
                if score > best_score:
                    best_score = score
                    best_label = label
            if best_score >= threshold:
                sample.label = best_label
                logger.debug("Auto-labelled char as %r (score=%.3f)", best_label, best_score)
            else:
                logger.debug("No match above threshold for char at bbox=%s", sample.bbox)
            labelled.append(sample)
        return labelled

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def _extract_cc(self, gray: np.ndarray, binary: np.ndarray) -> List[OCRSample]:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        samples: List[OCRSample] = []
        h_img, w_img = gray.shape[:2]
        for i in range(1, num_labels):  # skip background
            x, y, w, h, area = stats[i]
            if area < self.min_area:
                continue
            x0 = max(x - self.padding, 0)
            y0 = max(y - self.padding, 0)
            x1 = min(x + w + self.padding, w_img)
            y1 = min(y + h + self.padding, h_img)
            crop = gray[y0:y1, x0:x1]
            samples.append(OCRSample(image=crop, bbox=(x0, y0, x1 - x0, y1 - y0)))
        return samples

    def _extract_contour(self, gray: np.ndarray, binary: np.ndarray) -> List[OCRSample]:
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        samples: List[OCRSample] = []
        h_img, w_img = gray.shape[:2]
        for cnt in contours:
            if cv2.contourArea(cnt) < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            x0 = max(x - self.padding, 0)
            y0 = max(y - self.padding, 0)
            x1 = min(x + w + self.padding, w_img)
            y1 = min(y + h + self.padding, h_img)
            crop = gray[y0:y1, x0:x1]
            samples.append(OCRSample(image=crop, bbox=(x0, y0, x1 - x0, y1 - y0)))
        return samples


# ====================================================================== #
#  HOG Feature Computation                                                #
# ====================================================================== #


def _compute_hog_features(images: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    """Return HOG descriptor vectors for a batch of images."""
    hog = cv2.HOGDescriptor(
        _winSize=image_size,
        _blockSize=DEFAULT_HOG_BLOCK_SIZE,
        _blockStride=DEFAULT_HOG_BLOCK_STRIDE,
        _cellSize=DEFAULT_HOG_CELL_SIZE,
        _nbins=DEFAULT_HOG_NBINS,
    )
    features: List[np.ndarray] = []
    for img in images:
        resized = cv2.resize(img, image_size)
        if resized.ndim == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        feat = hog.compute(resized)
        features.append(feat.ravel())
    return np.array(features, dtype=np.float32)


# ====================================================================== #
#  OCR Font Trainer                                                       #
# ====================================================================== #


class OCRFontTrainer:
    """Train a character recognition model on domain-specific fonts.

    Parameters
    ----------
    config:
        Training configuration.
    """

    def __init__(self, config: Optional[OCRTrainingConfig] = None) -> None:
        self.config = config or OCRTrainingConfig()
        self._images: List[np.ndarray] = []
        self._labels: List[str] = []
        self._label_map: Dict[str, int] = {}
        self._inv_label_map: Dict[int, str] = {}
        self._model: Any = None

    # ------------------------------------------------------------------ #
    #  Data ingestion                                                     #
    # ------------------------------------------------------------------ #

    def add_samples(self, images: Sequence[np.ndarray], labels: Sequence[str]) -> None:
        """Add training images with corresponding character labels."""
        if len(images) != len(labels):
            raise ValueError("images and labels must have the same length")
        self._images.extend(images)
        self._labels.extend(labels)
        logger.info("Added %d samples (total: %d)", len(images), len(self._images))

    def add_sample_directory(self, path: Union[str, Path], label_file: Union[str, Path]) -> int:
        """Bulk-load character images from a directory.

        Parameters
        ----------
        path:
            Directory containing character image files (PNG/JPG).
        label_file:
            JSON file mapping filename -> label, e.g.
            ``{"img_001.png": "A", "img_002.png": "B"}``.

        Returns
        -------
        int
            Number of samples loaded.
        """
        path = Path(path)
        label_file = Path(label_file)
        if not path.is_dir():
            raise FileNotFoundError(f"Sample directory not found: {path}")
        with open(label_file, "r", encoding="utf-8") as fh:
            mapping: Dict[str, str] = json.load(fh)

        count = 0
        for fname, label in mapping.items():
            img_path = path / fname
            if not img_path.exists():
                logger.warning("Image not found, skipping: %s", img_path)
                continue
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.warning("Failed to read image: %s", img_path)
                continue
            self._images.append(img)
            self._labels.append(label)
            count += 1
        logger.info("Loaded %d samples from %s", count, path)
        return count

    # ------------------------------------------------------------------ #
    #  Training                                                           #
    # ------------------------------------------------------------------ #

    def _build_label_maps(self) -> None:
        unique = sorted(set(self._labels))
        self._label_map = {ch: i for i, ch in enumerate(unique)}
        self._inv_label_map = {i: ch for ch, i in self._label_map.items()}
        self.config.num_classes = len(unique)

    def train(self) -> Dict[str, Any]:
        """Train the recognition model on accumulated samples.

        Returns
        -------
        dict
            Training summary with keys ``num_samples``, ``num_classes``,
            ``backend``.
        """
        if len(self._images) == 0:
            raise RuntimeError("No training samples added")

        self._build_label_maps()
        int_labels = np.array(
            [self._label_map[lb] for lb in self._labels], dtype=np.int32,
        )

        if self.config.backend == BackendType.HOG_KNN:
            self._train_hog_knn(int_labels)
        elif self.config.backend == BackendType.CNN:
            self._train_mlp(int_labels)
        else:
            raise ValueError(f"Unsupported backend: {self.config.backend}")

        summary = {
            "num_samples": len(self._images),
            "num_classes": self.config.num_classes,
            "backend": self.config.backend.value,
        }
        logger.info("Training complete: %s", summary)
        return summary

    def _train_hog_knn(self, int_labels: np.ndarray) -> None:
        features = _compute_hog_features(self._images, self.config.image_size)
        knn = cv2.ml.KNearest_create()
        knn.setDefaultK(self.config.knn_k)
        knn.train(features, cv2.ml.ROW_SAMPLE, int_labels)
        self._model = knn

    def _train_mlp(self, int_labels: np.ndarray) -> None:
        """Train an MLP classifier on HOG features (labeled as 'CNN' backend for historical reasons).

        Uses OpenCV's ANN_MLP as a lightweight stand-in to avoid a hard
        dependency on deep-learning frameworks while still providing a
        multi-layer, non-linear classifier trained via back-propagation.
        """
        features = _compute_hog_features(self._images, self.config.image_size)
        n_features = features.shape[1]

        mlp = cv2.ml.ANN_MLP_create()
        mlp.setLayerSizes(np.array([n_features, 128, 64, self.config.num_classes]))
        mlp.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 1.0, 1.0)
        mlp.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)
        mlp.setBackpropWeightScale(0.1)
        mlp.setBackpropMomentumScale(0.1)
        mlp.setTermCriteria(
            (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, self.config.epochs * 1000, 1e-5),
        )

        # One-hot encode labels for MLP
        one_hot = np.zeros((len(int_labels), self.config.num_classes), dtype=np.float32)
        for i, lbl in enumerate(int_labels):
            one_hot[i, lbl] = 1.0

        mlp.train(features, cv2.ml.ROW_SAMPLE, one_hot)
        self._model = mlp

    # ------------------------------------------------------------------ #
    #  Evaluation                                                         #
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        test_images: Sequence[np.ndarray],
        test_labels: Sequence[str],
    ) -> Dict[str, Any]:
        """Evaluate the trained model on a test set.

        Returns
        -------
        dict
            ``accuracy``, ``total``, ``correct``, ``per_class`` accuracy.
        """
        if self._model is None:
            raise RuntimeError("Model not trained yet")

        features = _compute_hog_features(list(test_images), self.config.image_size)
        int_labels = np.array(
            [self._label_map.get(lb, -1) for lb in test_labels], dtype=np.int32,
        )

        predictions = self._predict_features(features)

        correct = int(np.sum(predictions == int_labels))
        total = len(int_labels)
        accuracy = correct / total if total > 0 else 0.0

        # Per-class accuracy
        per_class: Dict[str, float] = {}
        for label_str, label_int in self._label_map.items():
            mask = int_labels == label_int
            if mask.sum() == 0:
                continue
            per_class[label_str] = float(np.sum(predictions[mask] == label_int) / mask.sum())

        result = {"accuracy": accuracy, "total": total, "correct": correct, "per_class": per_class}
        logger.info("Evaluation: %.2f%% accuracy (%d/%d)", accuracy * 100, correct, total)
        return result

    def _predict_features(self, features: np.ndarray) -> np.ndarray:
        if isinstance(self._model, cv2.ml.KNearest):
            _, results, _, _ = self._model.findNearest(features, self.config.knn_k)
            return results.ravel().astype(np.int32)
        else:
            # ANN_MLP
            _, outputs = self._model.predict(features)
            return np.argmax(outputs, axis=1).astype(np.int32)

    # ------------------------------------------------------------------ #
    #  Persistence                                                        #
    # ------------------------------------------------------------------ #

    def save(self, path: Union[str, Path]) -> None:
        """Save the trained model and label maps to *path* (directory)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self._model is None:
            raise RuntimeError("No model to save")

        model_file = path / "model.xml"
        self._model.save(str(model_file))

        meta = {
            "label_map": self._label_map,
            "inv_label_map": {str(k): v for k, v in self._inv_label_map.items()},
            "config": {
                "image_size": list(self.config.image_size),
                "num_classes": self.config.num_classes,
                "font_name": self.config.font_name,
                "backend": self.config.backend.value,
                "knn_k": self.config.knn_k,
            },
        }
        with open(path / "meta.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        logger.info("Model saved to %s", path)

    def load(self, path: Union[str, Path]) -> None:
        """Load a previously saved model from *path* (directory)."""
        path = Path(path)
        with open(path / "meta.json", "r", encoding="utf-8") as fh:
            meta = json.load(fh)

        cfg = meta["config"]
        self.config.image_size = tuple(cfg["image_size"])
        self.config.num_classes = cfg["num_classes"]
        self.config.font_name = cfg["font_name"]
        self.config.backend = BackendType(cfg["backend"])
        self.config.knn_k = cfg.get("knn_k", DEFAULT_KNN_K)

        self._label_map = meta["label_map"]
        self._inv_label_map = {int(k): v for k, v in meta["inv_label_map"].items()}

        model_file = path / "model.xml"
        if self.config.backend == BackendType.HOG_KNN:
            self._model = cv2.ml.KNearest_load(str(model_file))
        else:
            self._model = cv2.ml.ANN_MLP_load(str(model_file))
        logger.info("Model loaded from %s", path)

    # ------------------------------------------------------------------ #
    #  Tesseract export                                                   #
    # ------------------------------------------------------------------ #

    def export_tesseract_traindata(self, output_dir: Union[str, Path]) -> Path:
        """Export training samples in Tesseract box/tif format.

        Creates a ``.tif`` image strip and a corresponding ``.box`` file that
        can be consumed by ``tesseract`` training tools.

        Returns
        -------
        Path
            Path to the generated ``.box`` file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if len(self._images) == 0:
            raise RuntimeError("No samples to export")

        cell_h, cell_w = self.config.image_size
        n = len(self._images)
        cols = min(n, 50)
        rows = (n + cols - 1) // cols
        strip = np.ones((rows * cell_h, cols * cell_w), dtype=np.uint8) * 255

        box_lines: List[str] = []
        page_h = rows * cell_h

        for idx, (img, label) in enumerate(zip(self._images, self._labels)):
            r, c = divmod(idx, cols)
            resized = cv2.resize(img, (cell_w, cell_h))
            if resized.ndim == 3:
                resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            y0 = r * cell_h
            x0 = c * cell_w
            strip[y0:y0 + cell_h, x0:x0 + cell_w] = resized
            # Tesseract box format: <char> <left> <bottom> <right> <top> <page>
            left = x0
            bottom = page_h - (y0 + cell_h)
            right = x0 + cell_w
            top = page_h - y0
            box_lines.append(f"{label} {left} {bottom} {right} {top} 0")

        font = self.config.font_name
        tif_path = output_dir / f"{font}.font.exp0.tif"
        box_path = output_dir / f"{font}.font.exp0.box"

        cv2.imwrite(str(tif_path), strip)
        with open(box_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(box_lines) + "\n")

        logger.info(
            "Exported Tesseract traindata: %d chars -> %s", n, box_path,
        )
        return box_path


# ====================================================================== #
#  OCR Font Inference                                                     #
# ====================================================================== #


class OCRFontInference:
    """Recognise characters using a custom-trained font model.

    Parameters
    ----------
    model_path:
        Path to saved model directory (produced by :class:`OCRFontTrainer`).
    """

    def __init__(self, model_path: Union[str, Path]) -> None:
        self._trainer = OCRFontTrainer()
        self._trainer.load(model_path)
        self._extractor = CharacterExtractor()

    def recognize(
        self,
        image: np.ndarray,
        extraction_method: str = "connected_component",
    ) -> List[OCRResult]:
        """Recognise characters in *image*.

        Parameters
        ----------
        image:
            Grayscale or BGR image containing text.
        extraction_method:
            Passed to :meth:`CharacterExtractor.extract_characters`.

        Returns
        -------
        List[OCRResult]
            One result per detected character, sorted left-to-right.
        """
        samples = self._extractor.extract_characters(image, method=extraction_method)
        if not samples:
            return []

        imgs = [s.image for s in samples]
        features = _compute_hog_features(imgs, self._trainer.config.image_size)
        predictions = self._trainer._predict_features(features)

        results: List[OCRResult] = []

        # Compute KNN distance-based confidence when using HOG_KNN backend
        if self._trainer.config.backend == BackendType.HOG_KNN and self._trainer._model is not None:
            k = self._trainer.config.knn_k
            model = self._trainer._model
            ret, knn_results, neighbors, dist = model.findNearest(features, k=k)
            for i, (sample, pred_int) in enumerate(zip(samples, predictions)):
                label = self._trainer._inv_label_map.get(int(pred_int), "?")
                # Distance-based confidence (inverse of mean distance, normalized)
                mean_dist = float(dist[i].mean()) if dist is not None else 0.0
                confidence = max(0.0, 1.0 - mean_dist / 100.0)  # normalize roughly to [0,1]
                results.append(OCRResult(text=label, confidence=confidence, bbox=sample.bbox))
        else:
            for sample, pred_int in zip(samples, predictions):
                label = self._trainer._inv_label_map.get(int(pred_int), "?")
                confidence = 1.0  # placeholder; refined below for MLP backend
                results.append(OCRResult(text=label, confidence=confidence, bbox=sample.bbox))

        # Refine confidence for MLP backend
        if self._trainer.config.backend == BackendType.CNN and self._trainer._model is not None:
            _, outputs = self._trainer._model.predict(features)
            for i, row in enumerate(outputs):
                softmax = np.exp(row - row.max())
                softmax /= softmax.sum()
                results[i].confidence = float(softmax.max())

        logger.info("Recognised %d characters", len(results))
        return results
