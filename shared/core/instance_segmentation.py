"""Multi-class instance and semantic segmentation.

Extends the existing binary U-Net segmentation with full multi-class,
per-instance capabilities using industry-standard backends.

Supported backends
------------------
- **Mask R-CNN** via torchvision (``detectron2`` compatible weights).
- **YOLOv8-seg** via ultralytics.
- **SAM** (Segment Anything Model) via the ``segment_anything`` package.

Key components
--------------
- ``SegmentationResult`` -- per-instance masks, class IDs, scores, and overlay.
- ``InstanceSegmentor`` -- multi-backend instance segmentation engine.
- ``SemanticSegmentor`` -- pixel-wise multi-class label map segmentation.
- Utility functions for mask/label conversion and IoU computation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from shared.validation import validate_image
from shared.op_logger import log_operation

logger = logging.getLogger(__name__)

# ====================================================================== #
#  Constants                                                              #
# ====================================================================== #

_DEFAULT_CONFIDENCE: float = 0.5
_DEFAULT_IOU_THRESHOLD: float = 0.5
_DEFAULT_OVERLAY_ALPHA: float = 0.45

# Distinct colours for up to 20 classes; wraps for larger counts.
_CLASS_PALETTE: List[Tuple[int, int, int]] = [
    (0, 114, 189), (217, 83, 25), (237, 177, 32), (126, 47, 142),
    (119, 172, 48), (77, 190, 238), (162, 20, 47), (0, 128, 128),
    (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
    (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34),
    (23, 190, 207), (31, 119, 180), (255, 187, 120), (174, 199, 232),
]


# ====================================================================== #
#  Device helper                                                          #
# ====================================================================== #

def _resolve_device(device: str) -> str:
    """Resolve ``'auto'`` to the best available device string."""
    if device == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"
    return device


# ====================================================================== #
#  Data classes                                                           #
# ====================================================================== #

@dataclass
class SegmentationResult:
    """Collection of per-instance segmentation results for a single image.

    Attributes:
        masks:          List of binary masks (uint8 ``H x W``, 255 = object).
        class_ids:      Integer class ID for each instance.
        class_names:    Human-readable class name for each instance.
        scores:         Confidence score in ``[0, 1]`` for each instance.
        colored_overlay: BGR overlay image with colour-coded masks
                         (``None`` until :meth:`build_overlay` is called or
                         the segmentor generates it automatically).
        areas:          Pixel area of each instance mask.
        centroids:      ``(cx, cy)`` centroid of each instance mask.
        processing_time_ms: Wall-clock inference time in milliseconds.
    """

    masks: List[np.ndarray] = field(default_factory=list)
    class_ids: List[int] = field(default_factory=list)
    class_names: List[str] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    colored_overlay: Optional[np.ndarray] = None
    areas: List[int] = field(default_factory=list)
    centroids: List[Tuple[float, float]] = field(default_factory=list)
    processing_time_ms: float = 0.0

    # ------------------------------------------------------------------ #
    #  Convenience properties                                             #
    # ------------------------------------------------------------------ #

    @property
    def count(self) -> int:
        """Number of detected instances."""
        return len(self.masks)

    def filter_by_class(self, class_name: str) -> "SegmentationResult":
        """Return a new result containing only instances of *class_name*."""
        indices = [i for i, n in enumerate(self.class_names) if n == class_name]
        return self._subset(indices)

    def filter_by_score(self, min_score: float) -> "SegmentationResult":
        """Return a new result containing only instances above *min_score*."""
        indices = [i for i, s in enumerate(self.scores) if s >= min_score]
        return self._subset(indices)

    def _subset(self, indices: List[int]) -> "SegmentationResult":
        return SegmentationResult(
            masks=[self.masks[i] for i in indices],
            class_ids=[self.class_ids[i] for i in indices],
            class_names=[self.class_names[i] for i in indices],
            scores=[self.scores[i] for i in indices],
            areas=[self.areas[i] for i in indices] if self.areas else [],
            centroids=[self.centroids[i] for i in indices] if self.centroids else [],
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise (without heavy mask arrays) for JSON transport."""
        return {
            "count": self.count,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "instances": [
                {
                    "class_id": cid,
                    "class_name": cn,
                    "score": round(sc, 4),
                    "area": ar,
                    "centroid": list(ct),
                }
                for cid, cn, sc, ar, ct in zip(
                    self.class_ids,
                    self.class_names,
                    self.scores,
                    self.areas or [0] * self.count,
                    self.centroids or [(0.0, 0.0)] * self.count,
                )
            ],
        }


# ====================================================================== #
#  Instance segmentor                                                     #
# ====================================================================== #

class InstanceSegmentor:
    """Multi-backend instance segmentation engine.

    Supported backends:
      - ``"yolov8-seg"`` -- YOLOv8 segmentation model via *ultralytics*.
      - ``"mask-rcnn"``  -- Mask R-CNN via *torchvision*.
      - ``"sam"``        -- Segment Anything Model (prompted or automatic).

    Parameters
    ----------
    model_path : str or Path
        Path to model weights.
    backend : str
        One of ``"yolov8-seg"``, ``"mask-rcnn"``, ``"sam"``.
    confidence : float
        Minimum confidence threshold.
    device : str
        ``"auto"``, ``"cpu"``, ``"cuda"``, or ``"mps"``.
    class_names : list of str, optional
        Override the model's class name mapping.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        backend: str = "yolov8-seg",
        confidence: float = _DEFAULT_CONFIDENCE,
        device: str = "auto",
        class_names: Optional[List[str]] = None,
    ) -> None:
        self._model_path = Path(model_path)
        self._backend = backend.lower()
        self._confidence = confidence
        self._device = _resolve_device(device)
        self._class_names = class_names
        self._model: Any = None

        if self._backend not in ("yolov8-seg", "mask-rcnn", "sam"):
            raise ValueError(
                f"Unsupported backend '{backend}'. "
                "Choose from: yolov8-seg, mask-rcnn, sam."
            )
        self._load_model()

    # ------------------------------------------------------------------ #
    #  Model loading                                                      #
    # ------------------------------------------------------------------ #

    def _load_model(self) -> None:
        if self._backend == "yolov8-seg":
            self._load_yolov8_seg()
        elif self._backend == "mask-rcnn":
            self._load_mask_rcnn()
        elif self._backend == "sam":
            self._load_sam()
        logger.info(
            "InstanceSegmentor: loaded %s from %s (device=%s)",
            self._backend, self._model_path, self._device,
        )

    def _load_yolov8_seg(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics is required for YOLOv8-seg. "
                "Install with: pip install ultralytics"
            )
        self._model = YOLO(str(self._model_path))

    def _load_mask_rcnn(self) -> None:
        try:
            import torch
            import torchvision
        except ImportError:
            raise ImportError(
                "torch and torchvision are required for Mask R-CNN."
            )
        from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
        if self._model_path.exists():
            model = maskrcnn_resnet50_fpn(weights=None)
            state = torch.load(
                str(self._model_path),
                map_location=self._device,
                weights_only=True,
            )
            model.load_state_dict(state)
        else:
            model = maskrcnn_resnet50_fpn(
                weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT,
            )
        model.to(self._device)
        model.eval()
        self._model = model

    def _load_sam(self) -> None:
        try:
            from segment_anything import sam_model_registry, SamPredictor
        except ImportError:
            raise ImportError(
                "segment_anything is required for SAM. "
                "Install from: https://github.com/facebookresearch/segment-anything"
            )
        # Infer model type from filename convention (sam_vit_h, sam_vit_l, sam_vit_b).
        name = self._model_path.stem.lower()
        if "vit_h" in name:
            model_type = "vit_h"
        elif "vit_l" in name:
            model_type = "vit_l"
        else:
            model_type = "vit_b"

        sam = sam_model_registry[model_type](checkpoint=str(self._model_path))
        sam.to(self._device)
        self._model = SamPredictor(sam)

    # ------------------------------------------------------------------ #
    #  Inference                                                          #
    # ------------------------------------------------------------------ #

    @log_operation(logger)
    def segment(self, image: np.ndarray) -> SegmentationResult:
        """Run instance segmentation on a single image.

        Parameters
        ----------
        image : ndarray
            BGR or RGB image as ``(H, W, C)`` uint8.

        Returns
        -------
        SegmentationResult
            Per-instance masks, class IDs, scores, overlay, and geometry.
        """
        validate_image(image, "image")
        t0 = time.perf_counter()

        if self._backend == "yolov8-seg":
            result = self._segment_yolov8(image)
        elif self._backend == "mask-rcnn":
            result = self._segment_mask_rcnn(image)
        elif self._backend == "sam":
            result = self._segment_sam_auto(image)
        else:
            result = SegmentationResult()

        result.processing_time_ms = (time.perf_counter() - t0) * 1000.0
        _fill_geometry(result)
        result.colored_overlay = build_overlay(image, result)
        return result

    @log_operation(logger)
    def segment_with_prompts(
        self,
        image: np.ndarray,
        points: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        boxes: Optional[np.ndarray] = None,
    ) -> SegmentationResult:
        """SAM-style prompted segmentation.

        Parameters
        ----------
        image : ndarray
            BGR or RGB image ``(H, W, C)`` uint8.
        points : ndarray, optional
            ``(N, 2)`` array of ``(x, y)`` prompt points.
        point_labels : ndarray, optional
            ``(N,)`` labels for each point (1 = foreground, 0 = background).
        boxes : ndarray, optional
            ``(M, 4)`` array of ``[x1, y1, x2, y2]`` bounding boxes.

        Returns
        -------
        SegmentationResult
        """
        validate_image(image, "image")
        if self._backend != "sam":
            raise RuntimeError(
                "Prompted segmentation is only supported with the SAM backend."
            )

        t0 = time.perf_counter()
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self._model.set_image(rgb)

        all_masks: List[np.ndarray] = []
        all_scores: List[float] = []

        if points is not None:
            if point_labels is None:
                point_labels = np.ones(len(points), dtype=np.int32)
            masks, scores, _ = self._model.predict(
                point_coords=points,
                point_labels=point_labels,
                multimask_output=True,
            )
            best = int(np.argmax(scores))
            mask_uint8 = (masks[best] * 255).astype(np.uint8)
            all_masks.append(mask_uint8)
            all_scores.append(float(scores[best]))

        if boxes is not None:
            for box in boxes:
                masks, scores, _ = self._model.predict(
                    box=box,
                    multimask_output=True,
                )
                best = int(np.argmax(scores))
                mask_uint8 = (masks[best] * 255).astype(np.uint8)
                all_masks.append(mask_uint8)
                all_scores.append(float(scores[best]))

        result = SegmentationResult(
            masks=all_masks,
            class_ids=[0] * len(all_masks),
            class_names=["object"] * len(all_masks),
            scores=all_scores,
        )
        result.processing_time_ms = (time.perf_counter() - t0) * 1000.0
        _fill_geometry(result)
        result.colored_overlay = build_overlay(image, result)
        return result

    # ------------------------------------------------------------------ #
    #  Backend-specific implementations                                   #
    # ------------------------------------------------------------------ #

    def _segment_yolov8(self, image: np.ndarray) -> SegmentationResult:
        results = self._model(
            image,
            conf=self._confidence,
            device=self._device,
            verbose=False,
        )
        masks_out: List[np.ndarray] = []
        class_ids: List[int] = []
        class_names: List[str] = []
        scores: List[float] = []

        h, w = image.shape[:2]
        for r in results:
            if r.boxes is None:
                continue
            for i in range(len(r.boxes)):
                cls_id = int(r.boxes.cls[i].item())
                conf = float(r.boxes.conf[i].item())
                name = (
                    self._class_names[cls_id]
                    if self._class_names and cls_id < len(self._class_names)
                    else r.names.get(cls_id, str(cls_id))
                )
                mask = np.zeros((h, w), dtype=np.uint8)
                if r.masks is not None and i < len(r.masks.data):
                    raw = r.masks.data[i].cpu().numpy()
                    mask = cv2.resize(
                        raw, (w, h), interpolation=cv2.INTER_LINEAR,
                    )
                    mask = (mask > 0.5).astype(np.uint8) * 255

                masks_out.append(mask)
                class_ids.append(cls_id)
                class_names.append(name)
                scores.append(conf)

        return SegmentationResult(
            masks=masks_out,
            class_ids=class_ids,
            class_names=class_names,
            scores=scores,
        )

    def _segment_mask_rcnn(self, image: np.ndarray) -> SegmentationResult:
        import torch

        from torchvision.transforms.functional import normalize
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
        tensor = normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        tensor = tensor.to(self._device)

        with torch.no_grad():
            preds = self._model([tensor])[0]

        masks_out: List[np.ndarray] = []
        class_ids: List[int] = []
        class_names: List[str] = []
        scores_out: List[float] = []

        pred_masks = preds["masks"].cpu().numpy()
        pred_labels = preds["labels"].cpu().numpy()
        pred_scores = preds["scores"].cpu().numpy()

        for i in range(len(pred_scores)):
            if pred_scores[i] < self._confidence:
                continue
            mask = (pred_masks[i, 0] > 0.5).astype(np.uint8) * 255
            cls_id = int(pred_labels[i])
            name = (
                self._class_names[cls_id]
                if self._class_names and cls_id < len(self._class_names)
                else str(cls_id)
            )
            masks_out.append(mask)
            class_ids.append(cls_id)
            class_names.append(name)
            scores_out.append(float(pred_scores[i]))

        return SegmentationResult(
            masks=masks_out,
            class_ids=class_ids,
            class_names=class_names,
            scores=scores_out,
        )

    def _segment_sam_auto(self, image: np.ndarray) -> SegmentationResult:
        """Automatic mask generation with SAM (grid prompts)."""
        try:
            from segment_anything import SamAutomaticMaskGenerator
        except ImportError:
            raise ImportError("segment_anything is required for SAM.")

        # The predictor's underlying model is accessible for the generator.
        generator = SamAutomaticMaskGenerator(self._model.model)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sam_masks = generator.generate(rgb)

        masks_out: List[np.ndarray] = []
        scores_out: List[float] = []
        for m in sam_masks:
            mask_uint8 = (m["segmentation"].astype(np.uint8)) * 255
            masks_out.append(mask_uint8)
            scores_out.append(float(m["stability_score"]))

        return SegmentationResult(
            masks=masks_out,
            class_ids=[0] * len(masks_out),
            class_names=["object"] * len(masks_out),
            scores=scores_out,
        )


# ====================================================================== #
#  Semantic segmentor                                                     #
# ====================================================================== #

class SemanticSegmentor:
    """Multi-class pixel-wise semantic segmentation.

    Uses an ONNX or TorchScript model that outputs a ``(H, W)`` label map
    or a ``(C, H, W)`` class-probability volume.

    Parameters
    ----------
    model_path : str or Path
        Path to model weights (``.onnx`` or ``.pt``).
    num_classes : int
        Number of semantic classes (including background at index 0).
    input_size : tuple of int
        ``(height, width)`` expected by the network.
    class_names : list of str, optional
        Human-readable class names.
    device : str
        ``"auto"``, ``"cpu"``, ``"cuda"``, or ``"mps"``.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        num_classes: int,
        input_size: Tuple[int, int] = (512, 512),
        class_names: Optional[List[str]] = None,
        device: str = "auto",
    ) -> None:
        self._model_path = Path(model_path)
        self._num_classes = num_classes
        self._input_size = input_size
        self._class_names = class_names or [str(i) for i in range(num_classes)]
        self._device = _resolve_device(device)
        self._model: Any = None
        self._load_model()

    def _load_model(self) -> None:
        suffix = self._model_path.suffix.lower()
        if suffix == ".onnx":
            self._load_onnx()
        else:
            self._load_torch()
        logger.info(
            "SemanticSegmentor: loaded from %s (%d classes, device=%s)",
            self._model_path, self._num_classes, self._device,
        )

    def _load_onnx(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime is required for ONNX models.")
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self._device == "cuda"
            else ["CPUExecutionProvider"]
        )
        self._model = ort.InferenceSession(
            str(self._model_path), providers=providers,
        )

    def _load_torch(self) -> None:
        import torch
        self._model = torch.jit.load(
            str(self._model_path), map_location=self._device,
        )
        self._model.eval()

    @log_operation(logger)
    def segment(self, image: np.ndarray) -> np.ndarray:
        """Produce a class label map for *image*.

        Parameters
        ----------
        image : ndarray
            BGR input image ``(H, W, C)`` uint8.

        Returns
        -------
        ndarray
            ``(H, W)`` int32 label map where each pixel is a class index.
        """
        validate_image(image, "image")
        orig_h, orig_w = image.shape[:2]
        blob = self._preprocess(image)

        if hasattr(self._model, "run"):
            # ONNX runtime
            inp_name = self._model.get_inputs()[0].name
            out = self._model.run(None, {inp_name: blob})[0]
        else:
            import torch
            tensor = torch.from_numpy(blob).to(self._device)
            with torch.no_grad():
                out = self._model(tensor).cpu().numpy()

        # out shape: (1, C, H, W) or (1, H, W)
        if out.ndim == 4:
            labels = np.argmax(out[0], axis=0).astype(np.int32)
        else:
            labels = out[0].astype(np.int32)

        # Resize back to original resolution.
        if labels.shape != (orig_h, orig_w):
            labels = cv2.resize(
                labels, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST,
            )
        return labels

    def overlay(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        class_colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
        alpha: float = _DEFAULT_OVERLAY_ALPHA,
    ) -> np.ndarray:
        """Blend a colour-coded label map onto *image*.

        Parameters
        ----------
        image : ndarray
            BGR image ``(H, W, C)`` uint8.
        labels : ndarray
            ``(H, W)`` int32 label map.
        class_colors : dict, optional
            Mapping ``{class_id: (B, G, R)}``. Falls back to the palette.
        alpha : float
            Overlay transparency in ``[0, 1]``.

        Returns
        -------
        ndarray
            Blended BGR image.
        """
        vis = image.copy()
        if vis.ndim == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

        colour_map = np.zeros_like(vis)
        for cid in range(self._num_classes):
            if cid == 0:
                continue  # skip background
            color = (
                class_colors[cid]
                if class_colors and cid in class_colors
                else _CLASS_PALETTE[cid % len(_CLASS_PALETTE)]
            )
            colour_map[labels == cid] = color

        mask = labels > 0
        vis[mask] = cv2.addWeighted(
            vis, 1.0 - alpha, colour_map, alpha, 0.0,
        )[mask]
        return vis

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Resize and normalise to ``(1, 3, H, W)`` float32 blob."""
        h, w = self._input_size
        resized = cv2.resize(image, (w, h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        blob = rgb.astype(np.float32).transpose(2, 0, 1)[np.newaxis] / 255.0
        return blob


# ====================================================================== #
#  Geometry helpers (private)                                             #
# ====================================================================== #

def _fill_geometry(result: SegmentationResult) -> None:
    """Compute area and centroid for every mask in *result* (in place)."""
    areas: List[int] = []
    centroids: List[Tuple[float, float]] = []
    for mask in result.masks:
        binary = mask > 127
        area = int(binary.sum())
        areas.append(area)
        if area > 0:
            ys, xs = np.where(binary)
            centroids.append((float(xs.mean()), float(ys.mean())))
        else:
            centroids.append((0.0, 0.0))
    result.areas = areas
    result.centroids = centroids


# ====================================================================== #
#  Overlay builder                                                        #
# ====================================================================== #

def build_overlay(
    image: np.ndarray,
    result: SegmentationResult,
    alpha: float = _DEFAULT_OVERLAY_ALPHA,
) -> np.ndarray:
    """Create a colour-coded overlay from instance masks.

    Each instance receives a unique colour derived from its class ID.

    Parameters
    ----------
    image : ndarray
        BGR image ``(H, W, C)`` uint8.
    result : SegmentationResult
        Segmentation result with masks and class IDs.
    alpha : float
        Overlay transparency.

    Returns
    -------
    ndarray
        Blended BGR image with coloured instance masks.
    """
    vis = image.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    colour_layer = np.zeros_like(vis)
    for mask, cid in zip(result.masks, result.class_ids):
        color = _CLASS_PALETTE[cid % len(_CLASS_PALETTE)]
        binary = mask > 127
        colour_layer[binary] = color

    combined_mask = np.zeros(vis.shape[:2], dtype=bool)
    for mask in result.masks:
        combined_mask |= mask > 127

    blended = cv2.addWeighted(vis, 1.0 - alpha, colour_layer, alpha, 0.0)
    vis[combined_mask] = blended[combined_mask]
    return vis


# ====================================================================== #
#  Utility functions                                                      #
# ====================================================================== #

def masks_to_labels(masks: List[np.ndarray]) -> np.ndarray:
    """Convert a list of binary instance masks to a single label map.

    Masks are applied in order; later masks overwrite earlier ones where
    they overlap.  Background is labelled ``0``; the first mask becomes ``1``.

    Parameters
    ----------
    masks : list of ndarray
        Each mask is ``(H, W)`` uint8 with 255 = foreground.

    Returns
    -------
    ndarray
        ``(H, W)`` int32 label map.
    """
    if not masks:
        raise ValueError("masks list must not be empty.")
    h, w = masks[0].shape[:2]
    labels = np.zeros((h, w), dtype=np.int32)
    for idx, mask in enumerate(masks, start=1):
        labels[mask > 127] = idx
    return labels


def labels_to_masks(labels: np.ndarray) -> List[np.ndarray]:
    """Convert a label map to a list of binary instance masks.

    Background (label ``0``) is skipped.

    Parameters
    ----------
    labels : ndarray
        ``(H, W)`` int32 label map.

    Returns
    -------
    list of ndarray
        Binary masks in ascending label order, each ``(H, W)`` uint8.
    """
    unique_ids = np.unique(labels)
    masks: List[np.ndarray] = []
    for uid in unique_ids:
        if uid == 0:
            continue
        mask = (labels == uid).astype(np.uint8) * 255
        masks.append(mask)
    return masks


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute Intersection over Union between two binary masks.

    Parameters
    ----------
    mask1 : ndarray
        ``(H, W)`` binary mask (non-zero = foreground).
    mask2 : ndarray
        ``(H, W)`` binary mask (non-zero = foreground).

    Returns
    -------
    float
        IoU value in ``[0, 1]``.  Returns ``0.0`` when both masks are empty.
    """
    b1 = mask1 > 0
    b2 = mask2 > 0
    intersection = np.logical_and(b1, b2).sum()
    union = np.logical_or(b1, b2).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)
