"""Object detection using YOLO and other detection models.

Provides a unified interface for object detection in industrial inspection,
supporting YOLOv8/YOLOv5 models and custom-trained detectors.

Key components
--------------
- ``DetectionResult`` -- single detection bounding box with class and score.
- ``DetectionOutput`` -- collection of detections for one image.
- ``ObjectDetector`` -- model loading and inference.
- ``DetectorTrainer`` -- fine-tuning on custom datasets.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Single detected object."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    mask: Optional[np.ndarray] = None  # instance segmentation mask if available

    @property
    def center(self) -> Tuple[float, float]:
        x, y, w, h = self.bbox
        return (x + w / 2.0, y + h / 2.0)

    @property
    def area(self) -> int:
        return self.bbox[2] * self.bbox[3]


@dataclass
class DetectionOutput:
    """All detections for a single image."""
    detections: List[DetectionResult] = field(default_factory=list)
    image_shape: Tuple[int, ...] = (0, 0)
    processing_time_ms: float = 0.0

    @property
    def count(self) -> int:
        return len(self.detections)

    def filter_by_class(self, class_name: str) -> List[DetectionResult]:
        return [d for d in self.detections if d.class_name == class_name]

    def filter_by_confidence(self, min_conf: float) -> List[DetectionResult]:
        return [d for d in self.detections if d.confidence >= min_conf]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "processing_time_ms": self.processing_time_ms,
            "detections": [
                {
                    "class_id": d.class_id,
                    "class_name": d.class_name,
                    "confidence": round(d.confidence, 4),
                    "bbox": list(d.bbox),
                    "center": list(d.center),
                    "area": d.area,
                }
                for d in self.detections
            ],
        }


class ObjectDetector:
    """Unified object detection interface.

    Supports:
    - YOLOv8 (via ultralytics)
    - YOLOv5 (via torch hub)
    - ONNX models (via onnxruntime)
    - Custom PyTorch models

    Parameters
    ----------
    model_path : Path or str
        Path to model file (.pt, .onnx, or directory).
    model_type : str
        One of "yolov8", "yolov5", "onnx", "torchscript".
    confidence_threshold : float
        Minimum confidence for detections (default 0.25).
    iou_threshold : float
        NMS IoU threshold (default 0.45).
    device : str
        "auto", "cpu", "cuda", or "mps".
    class_names : list of str or None
        Override class name mapping. If None, uses model's built-in names.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        model_type: str = "yolov8",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "auto",
        class_names: Optional[List[str]] = None,
    ) -> None:
        self._model_path = Path(model_path)
        self._model_type = model_type
        self._conf_thresh = confidence_threshold
        self._iou_thresh = iou_threshold
        self._device = self._resolve_device(device)
        self._class_names = class_names
        self._model: Any = None
        self._load_model()

    @staticmethod
    def _resolve_device(device: str) -> str:
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

    def _load_model(self) -> None:
        if not self._model_path.exists():
            raise FileNotFoundError(f"Model not found: {self._model_path}")

        if self._model_type == "yolov8":
            self._load_yolov8()
        elif self._model_type == "yolov5":
            self._load_yolov5()
        elif self._model_type == "onnx":
            self._load_onnx()
        elif self._model_type == "torchscript":
            self._load_torchscript()
        else:
            raise ValueError(f"Unsupported model type: {self._model_type}")
        logger.info(
            "Loaded %s model from %s (device=%s)",
            self._model_type, self._model_path, self._device,
        )

    def _load_yolov8(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics is required for YOLOv8. "
                "Install with: pip install ultralytics"
            )
        self._model = YOLO(str(self._model_path))

    def _load_yolov5(self) -> None:
        import torch
        self._model = torch.hub.load(
            "ultralytics/yolov5", "custom", path=str(self._model_path),
        )
        self._model.conf = self._conf_thresh
        self._model.iou = self._iou_thresh
        self._model.to(self._device)

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

    def _load_torchscript(self) -> None:
        import torch
        self._model = torch.jit.load(
            str(self._model_path), map_location=self._device,
        )
        self._model.eval()

    def detect(self, image: np.ndarray) -> DetectionOutput:
        """Run detection on a single image (BGR or RGB numpy array)."""
        if image is None or image.size == 0:
            raise ValueError("Input image must be a non-empty numpy array")
        t0 = time.perf_counter()

        if self._model_type == "yolov8":
            output = self._detect_yolov8(image)
        elif self._model_type == "yolov5":
            output = self._detect_yolov5(image)
        elif self._model_type == "onnx":
            output = self._detect_onnx(image)
        elif self._model_type == "torchscript":
            output = self._detect_torchscript(image)
        else:
            output = DetectionOutput()

        output.image_shape = image.shape[:2]
        output.processing_time_ms = (time.perf_counter() - t0) * 1000
        return output

    def _detect_yolov8(self, image: np.ndarray) -> DetectionOutput:
        results = self._model(
            image,
            conf=self._conf_thresh,
            iou=self._iou_thresh,
            device=self._device,
            verbose=False,
        )
        detections: List[DetectionResult] = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                cls_name = (
                    self._class_names[cls_id]
                    if self._class_names and cls_id < len(self._class_names)
                    else r.names.get(cls_id, str(cls_id))
                )
                mask = None
                if r.masks is not None and i < len(r.masks.data):
                    mask = r.masks.data[i].cpu().numpy().astype(np.uint8)
                detections.append(DetectionResult(
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf,
                    bbox=(int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                    mask=mask,
                ))
        return DetectionOutput(detections=detections)

    def _detect_yolov5(self, image: np.ndarray) -> DetectionOutput:
        results = self._model(image)
        df = results.pandas().xyxy[0]
        detections: List[DetectionResult] = []
        for _, row in df.iterrows():
            x1 = int(row["xmin"])
            y1 = int(row["ymin"])
            x2 = int(row["xmax"])
            y2 = int(row["ymax"])
            cls_id = int(row["class"])
            cls_name = (
                self._class_names[cls_id]
                if self._class_names and cls_id < len(self._class_names)
                else row["name"]
            )
            detections.append(DetectionResult(
                class_id=int(row["class"]),
                class_name=cls_name,
                confidence=float(row["confidence"]),
                bbox=(x1, y1, x2 - x1, y2 - y1),
            ))
        return DetectionOutput(detections=detections)

    def _detect_onnx(self, image: np.ndarray) -> DetectionOutput:
        inp = self._model.get_inputs()[0]
        _, c, h, w = inp.shape
        resized = cv2.resize(image, (w, h))
        blob = resized.astype(np.float32).transpose(2, 0, 1)[np.newaxis] / 255.0
        outputs = self._model.run(None, {inp.name: blob})
        # Parse YOLO ONNX output format
        detections = self._parse_onnx_output(
            outputs[0], image.shape[:2], (h, w),
        )
        return DetectionOutput(detections=detections)

    def _detect_torchscript(self, image: np.ndarray) -> DetectionOutput:
        """Run detection using a TorchScript model.

        Raises
        ------
        NotImplementedError
            TorchScript detection requires model-specific output parsing.
            Subclass ObjectDetector to implement.
        """
        raise NotImplementedError(
            "TorchScript detection requires model-specific output parsing. "
            "Subclass ObjectDetector to implement."
        )

    def _parse_onnx_output(
        self,
        output: np.ndarray,
        orig_shape: Tuple[int, int],
        input_shape: Tuple[int, int],
    ) -> List[DetectionResult]:
        """Parse YOLO ONNX output (1, N, 5+C) or (1, 5+C, N) format."""
        detections: List[DetectionResult] = []
        if output.ndim == 3:
            output = output[0]
        if output.shape[0] < output.shape[1]:
            output = output.T

        oh, ow = orig_shape
        ih, iw = input_shape
        sx, sy = ow / iw, oh / ih

        for row in output:
            if len(row) < 6:
                continue
            cx, cy, w, h = row[:4]
            scores = row[4:]
            cls_id = int(np.argmax(scores))
            conf = float(scores[cls_id])
            if conf < self._conf_thresh:
                continue
            x1 = int((cx - w / 2) * sx)
            y1 = int((cy - h / 2) * sy)
            bw = int(w * sx)
            bh = int(h * sy)
            cls_name = (
                self._class_names[cls_id]
                if self._class_names and cls_id < len(self._class_names)
                else str(cls_id)
            )
            detections.append(DetectionResult(
                class_id=cls_id,
                class_name=cls_name,
                confidence=conf,
                bbox=(x1, y1, bw, bh),
            ))
        return self._nms(detections)

    def _nms(self, detections: List[DetectionResult]) -> List[DetectionResult]:
        if not detections:
            return []
        boxes = np.array(
            [
                [d.bbox[0], d.bbox[1], d.bbox[0] + d.bbox[2], d.bbox[1] + d.bbox[3]]
                for d in detections
            ],
            dtype=np.float32,
        )
        scores = np.array(
            [d.confidence for d in detections], dtype=np.float32,
        )
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), scores.tolist(), self._conf_thresh, self._iou_thresh,
        )
        if isinstance(indices, np.ndarray):
            indices = indices.flatten().tolist()
        return [detections[i] for i in indices]

    def detect_batch(self, images: List[np.ndarray]) -> List[DetectionOutput]:
        """Run detection on a batch of images."""
        return [self.detect(img) for img in images]

    @staticmethod
    def draw_detections(
        image: np.ndarray,
        output: DetectionOutput,
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw bounding boxes and labels on image."""
        vis = image.copy()
        colors: Dict[str, Tuple[int, ...]] = {}
        for det in output.detections:
            if det.class_name not in colors:
                h = hash(det.class_name) % 180
                colors[det.class_name] = tuple(
                    int(c)
                    for c in cv2.cvtColor(
                        np.uint8([[[h, 200, 200]]]), cv2.COLOR_HSV2BGR,
                    )[0, 0]
                )
            color = colors[det.class_name]
            x, y, w, h = det.bbox
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, thickness)
            label = f"{det.class_name} {det.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1,
            )
            cv2.rectangle(vis, (x, y - th - 6), (x + tw + 4, y), color, -1)
            cv2.putText(
                vis, label, (x + 2, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )
        return vis


class DetectorTrainer:
    """Fine-tune a YOLO model on custom datasets.

    Parameters
    ----------
    base_model : str
        Base model name or path (e.g., "yolov8n.pt", "yolov8s.pt").
    data_yaml : str or Path
        Path to YOLO-format data.yaml file.
    epochs : int
        Number of training epochs.
    image_size : int
        Training image size.
    device : str
        Training device.
    """

    def __init__(
        self,
        base_model: str = "yolov8n.pt",
        data_yaml: Union[str, Path] = "data.yaml",
        epochs: int = 100,
        image_size: int = 640,
        device: str = "auto",
    ) -> None:
        self._base_model = base_model
        self._data_yaml = str(data_yaml)
        self._epochs = epochs
        self._image_size = image_size
        self._device = ObjectDetector._resolve_device(device)

    def train(
        self,
        project: str = "runs/detect",
        name: str = "train",
        batch_size: int = 16,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
    ) -> Path:
        """Start training. Returns path to best weights.

        Parameters
        ----------
        progress_callback : callable(epoch, total_epochs, loss)
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics is required for training. "
                "Install with: pip install ultralytics"
            )

        model = YOLO(self._base_model)
        results = model.train(
            data=self._data_yaml,
            epochs=self._epochs,
            imgsz=self._image_size,
            batch=batch_size,
            device=self._device,
            project=project,
            name=name,
            verbose=True,
        )
        best_path = Path(project) / name / "weights" / "best.pt"
        logger.info("Training complete. Best weights: %s", best_path)
        return best_path

    @staticmethod
    def export_onnx(
        model_path: Union[str, Path], image_size: int = 640,
    ) -> Path:
        """Export a trained model to ONNX format."""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ultralytics required for export.")
        model = YOLO(str(model_path))
        export_path = model.export(format="onnx", imgsz=image_size)
        logger.info("Exported ONNX model to %s", export_path)
        return Path(export_path)
