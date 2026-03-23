"""Teacher-Student Feature Pyramid Matching (STPM-style) anomaly detection.

Teacher = frozen pretrained CNN.  Student = same architecture, trained to mimic
teacher features on OK images.  At inference, feature discrepancy = anomaly.

The training objective is a multi-scale MSE loss: for each hooked layer the
student must reproduce the teacher's activation map.  At test time the
per-layer squared-error maps are upsampled to input resolution and summed to
form a dense anomaly map; the image-level score is the map maximum.

Key components
--------------
- ``TeacherStudentModel`` -- serialisable dataclass (student weights + meta).
- ``TeacherStudentTrainer`` -- trains the student network.
- ``TeacherStudentInference`` -- scores images via teacher-student discrepancy.
- ``save_model`` / ``load_model`` -- persistence helpers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

logger = logging.getLogger(__name__)

# ======================================================================
# Device selection (mirrors dl_anomaly.config._select_device)
# ======================================================================

def _select_device(requested: str = "auto") -> str:
    """Pick the best available device.

    Priority: user request (if available) > CUDA > MPS > CPU.
    """
    req = requested.strip().lower()

    if req == "cuda" and torch.cuda.is_available():
        return "cuda"
    if req == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if req == "cpu":
        return "cpu"

    # auto
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ======================================================================
# Backbone registry
# ======================================================================

_BACKBONE_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}


def _populate_registry() -> None:
    """Lazily populate the backbone registry from torchvision."""
    if _BACKBONE_REGISTRY:
        return
    _BACKBONE_REGISTRY["resnet18"] = lambda: models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    _BACKBONE_REGISTRY["resnet50"] = lambda: models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    _BACKBONE_REGISTRY["wide_resnet50_2"] = lambda: models.wide_resnet50_2(
        weights=models.Wide_ResNet50_2_Weights.DEFAULT,
    )
    _BACKBONE_REGISTRY["efficientnet_b0"] = lambda: models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.DEFAULT,
    )


def list_available_backbones() -> List[str]:
    """Return list of supported backbone names."""
    return ["resnet18", "resnet50", "wide_resnet50_2", "efficientnet_b0"]


# ======================================================================
# Hook-based multi-layer feature extractor
# ======================================================================

class _FeatureHooks:
    """Attach forward hooks to named layers of a network.

    Captured features are stored in :attr:`features` keyed by layer name.
    This mirrors the hook pattern used in ``patchcore.FeatureExtractor``.
    """

    def __init__(self, network: nn.Module, layers: Tuple[str, ...]) -> None:
        self.features: Dict[str, torch.Tensor] = {}
        self._hooks: List[torch.utils.hooks.RemovableHook] = []

        modules_dict = dict(network.named_modules())
        for name in layers:
            module = modules_dict.get(name)
            if module is None:
                raise ValueError(
                    f"Layer '{name}' not found in network.  "
                    f"Available: {list(modules_dict.keys())[:20]} ..."
                )
            hook = module.register_forward_hook(self._make_hook(name))
            self._hooks.append(hook)

    def _make_hook(self, layer_name: str) -> Callable:
        def hook_fn(_module: nn.Module, _input: Any, output: torch.Tensor) -> None:
            self.features[layer_name] = output
        return hook_fn

    def clear(self) -> None:
        self.features.clear()

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ======================================================================
# Simple image-folder dataset (ImageNet normalisation)
# ======================================================================

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class _ImageFolderDataset(Dataset):
    """Flat image folder -- all images are loaded as RGB and normalised."""

    def __init__(self, image_dir: Union[str, Path], image_size: int) -> None:
        self.image_dir = Path(image_dir)
        self.paths = sorted(
            p for p in self.image_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
        )
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = cv2.imread(str(self.paths[idx]))
        if img is None:
            raise IOError(f"Cannot read image: {self.paths[idx]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.transform(img)


# ======================================================================
# Serialisable model dataclass
# ======================================================================

@dataclass
class TeacherStudentModel:
    """Container for a trained Teacher-Student anomaly detector.

    Attributes
    ----------
    student_state_dict:
        Trained student network weights.
    backbone_name:
        Backbone architecture name (e.g. ``"resnet18"``).
    layers:
        Tuple of layer names used for feature matching.
    image_size:
        Spatial resolution of input images.
    threshold:
        Anomaly score threshold fitted from training data.
    config:
        Extra parameters for persistence / reproducibility.
    """

    student_state_dict: dict
    backbone_name: str
    layers: Tuple[str, ...]
    image_size: int
    threshold: float
    config: Dict[str, Any] = field(default_factory=dict)


# ======================================================================
# Trainer
# ======================================================================

class TeacherStudentTrainer:
    """Train a student network to mimic a frozen teacher on OK images.

    Parameters
    ----------
    backbone_name:
        Pre-trained backbone (see :func:`list_available_backbones`).
    layers:
        Backbone layers whose feature maps are matched.
    image_size:
        Input image resolution (square).
    device:
        ``"auto"``, ``"cuda"``, ``"mps"``, or ``"cpu"``.
    """

    def __init__(
        self,
        backbone_name: str = "resnet18",
        layers: Tuple[str, ...] = ("layer1", "layer2", "layer3"),
        image_size: int = 256,
        device: str = "auto",
    ) -> None:
        _populate_registry()
        if backbone_name not in _BACKBONE_REGISTRY:
            raise ValueError(
                f"Unknown backbone '{backbone_name}'. "
                f"Choose from {list_available_backbones()}."
            )

        self.backbone_name = backbone_name
        self.layers = layers
        self.image_size = image_size
        self.device = torch.device(_select_device(device))

        # Teacher -- frozen pretrained
        self.teacher: nn.Module = _BACKBONE_REGISTRY[backbone_name]()
        self.teacher.to(self.device).eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self._teacher_hooks = _FeatureHooks(self.teacher, layers)

        # Student -- trainable copy (randomly re-initialised)
        self.student: nn.Module = _BACKBONE_REGISTRY[backbone_name]()
        # Re-initialise student weights so it must *learn* to match the teacher
        self._reinit_weights(self.student)
        self.student.to(self.device).train()
        self._student_hooks = _FeatureHooks(self.student, layers)

    # ------------------------------------------------------------------

    @staticmethod
    def _reinit_weights(model: nn.Module) -> None:
        """Xavier-uniform re-initialisation for Conv2d and Linear layers."""
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Dataset helper
    # ------------------------------------------------------------------

    def _build_dataset(
        self, image_dir: Union[str, Path],
    ) -> _ImageFolderDataset:
        """Load images with ImageNet normalisation."""
        ds = _ImageFolderDataset(image_dir, self.image_size)
        if len(ds) == 0:
            raise RuntimeError(f"No images found in {image_dir}")
        return ds

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        image_dir: Union[str, Path],
        epochs: int = 100,
        lr: float = 0.001,
        batch_size: int = 16,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
    ) -> TeacherStudentModel:
        """Train the student to reproduce teacher features on defect-free images.

        Parameters
        ----------
        image_dir:
            Directory containing OK training images.
        epochs:
            Number of training epochs.
        lr:
            Learning rate for Adam optimiser.
        batch_size:
            Mini-batch size.
        progress_callback:
            Called with ``(current_epoch, total_epochs, epoch_loss)`` after
            each epoch.

        Returns
        -------
        TeacherStudentModel
            Serialisable model with trained student weights and metadata.
        """
        dataset = self._build_dataset(image_dir)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(self.device.type == "cuda"),
            drop_last=False,
        )

        optimiser = torch.optim.Adam(self.student.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=epochs, eta_min=lr * 0.01,
        )

        logger.info(
            "Teacher-Student training: %d images, backbone=%s, layers=%s, "
            "epochs=%d, device=%s",
            len(dataset), self.backbone_name, self.layers, epochs, self.device,
        )

        self.student.train()
        best_loss = float("inf")
        best_state: Optional[dict] = None

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            n_batches = 0

            for images in loader:
                images = images.to(self.device)

                # Teacher forward (no grad)
                self._teacher_hooks.clear()
                with torch.no_grad():
                    self.teacher(images)
                teacher_feats = dict(self._teacher_hooks.features)

                # Student forward
                self._student_hooks.clear()
                self.student(images)
                student_feats = dict(self._student_hooks.features)

                # Multi-layer MSE loss
                loss = torch.tensor(0.0, device=self.device)
                for layer_name in self.layers:
                    t_feat = teacher_feats[layer_name]
                    s_feat = student_feats[layer_name]
                    # Normalise per-layer to balance scale differences
                    loss = loss + F.mse_loss(s_feat, t_feat)

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = {k: v.cpu().clone() for k, v in self.student.state_dict().items()}

            if epoch % max(1, epochs // 10) == 0 or epoch == 1:
                logger.info(
                    "  Epoch %d/%d  loss=%.6f  best=%.6f  lr=%.2e",
                    epoch, epochs, avg_loss, best_loss,
                    scheduler.get_last_lr()[0],
                )

            if progress_callback is not None:
                progress_callback(epoch, epochs, avg_loss)

        # Use best weights
        if best_state is None:
            best_state = {k: v.cpu().clone() for k, v in self.student.state_dict().items()}

        # Fit threshold from training data
        threshold = self._fit_threshold(loader, best_state)

        model = TeacherStudentModel(
            student_state_dict=best_state,
            backbone_name=self.backbone_name,
            layers=self.layers,
            image_size=self.image_size,
            threshold=threshold,
            config={
                "epochs": epochs,
                "lr": lr,
                "batch_size": batch_size,
                "best_loss": best_loss,
            },
        )

        logger.info(
            "Teacher-Student training complete (best_loss=%.6f, threshold=%.6f)",
            best_loss, threshold,
        )
        return model

    # ------------------------------------------------------------------

    @torch.no_grad()
    def _fit_threshold(
        self,
        loader: DataLoader,
        student_state: dict,
        percentile: float = 99.5,
    ) -> float:
        """Score training images and set threshold at the given percentile."""
        # Temporarily load best weights into student
        device_state = {k: v.to(self.device) for k, v in student_state.items()}
        self.student.load_state_dict(device_state)
        self.student.eval()

        scores: List[float] = []
        for images in loader:
            images = images.to(self.device)

            self._teacher_hooks.clear()
            self.teacher(images)
            teacher_feats = dict(self._teacher_hooks.features)

            self._student_hooks.clear()
            self.student(images)
            student_feats = dict(self._student_hooks.features)

            # Per-image anomaly score = max of summed per-layer MSE maps
            batch_size = images.shape[0]
            anomaly_map = torch.zeros(
                batch_size, 1, self.image_size, self.image_size,
                device=self.device,
            )
            for layer_name in self.layers:
                diff = (teacher_feats[layer_name] - student_feats[layer_name]) ** 2
                diff = diff.mean(dim=1, keepdim=True)  # channel mean -> (B,1,H_l,W_l)
                diff = F.interpolate(
                    diff,
                    size=(self.image_size, self.image_size),
                    mode="bilinear",
                    align_corners=False,
                )
                anomaly_map = anomaly_map + diff

            for i in range(batch_size):
                scores.append(float(anomaly_map[i].max().cpu()))

        threshold = float(np.percentile(scores, percentile))
        logger.info(
            "Threshold fitted: %.6f (percentile=%.1f, n_images=%d, "
            "score_range=[%.6f, %.6f])",
            threshold, percentile, len(scores), min(scores), max(scores),
        )
        return threshold

    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Remove hooks and free GPU memory."""
        self._teacher_hooks.remove()
        self._student_hooks.remove()


# ======================================================================
# Inference
# ======================================================================

class TeacherStudentInference:
    """Score images using a trained Teacher-Student model.

    Parameters
    ----------
    model:
        A trained :class:`TeacherStudentModel`.
    device:
        Device string (``"auto"``, ``"cuda"``, ``"mps"``, ``"cpu"``).
    """

    def __init__(self, model: TeacherStudentModel, device: str = "auto") -> None:
        _populate_registry()
        self.model = model
        self.device = torch.device(_select_device(device))

        # Teacher -- frozen
        self.teacher: nn.Module = _BACKBONE_REGISTRY[model.backbone_name]()
        self.teacher.to(self.device).eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self._teacher_hooks = _FeatureHooks(self.teacher, model.layers)

        # Student -- load trained weights
        self.student: nn.Module = _BACKBONE_REGISTRY[model.backbone_name]()
        self.student.load_state_dict(model.student_state_dict)
        self.student.to(self.device).eval()
        for p in self.student.parameters():
            p.requires_grad = False
        self._student_hooks = _FeatureHooks(self.student, model.layers)

        # Preprocessing transform
        self._transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((model.image_size, model.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

        # Gaussian kernel for anomaly map smoothing
        self._smooth_sigma: float = 4.0

    # ------------------------------------------------------------------
    # Core scoring
    # ------------------------------------------------------------------

    @torch.no_grad()
    def score_image(self, image: np.ndarray) -> Tuple[float, np.ndarray]:
        """Compute anomaly score and dense anomaly map.

        Parameters
        ----------
        image:
            ``(H, W, 3)`` BGR or RGB uint8 image (BGR is auto-converted).

        Returns
        -------
        tuple
            ``(anomaly_score, anomaly_map)`` where *anomaly_map* is a
            float32 ``(image_size, image_size)`` array normalised to [0, 1].
        """
        # Preprocess
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

        tensor = self._transform(image).unsqueeze(0).to(self.device)

        # Teacher forward
        self._teacher_hooks.clear()
        self.teacher(tensor)
        teacher_feats = dict(self._teacher_hooks.features)

        # Student forward
        self._student_hooks.clear()
        self.student(tensor)
        student_feats = dict(self._student_hooks.features)

        # Build anomaly map: sum of per-layer MSE, upsampled to input size
        h, w = self.model.image_size, self.model.image_size
        anomaly_map = torch.zeros(1, 1, h, w, device=self.device)

        for layer_name in self.model.layers:
            t_feat = teacher_feats[layer_name]
            s_feat = student_feats[layer_name]
            diff = (t_feat - s_feat) ** 2
            diff = diff.mean(dim=1, keepdim=True)  # (1, 1, H_l, W_l)
            diff = F.interpolate(
                diff, size=(h, w), mode="bilinear", align_corners=False,
            )
            anomaly_map = anomaly_map + diff

        anomaly_map_np = anomaly_map.squeeze().cpu().numpy()  # (H, W)
        anomaly_score = float(anomaly_map_np.max())

        # Normalise to [0, 1] for visualisation
        amap_min = anomaly_map_np.min()
        amap_max = anomaly_map_np.max()
        if amap_max - amap_min > 1e-8:
            anomaly_map_np = (anomaly_map_np - amap_min) / (amap_max - amap_min)
        else:
            anomaly_map_np = np.zeros_like(anomaly_map_np)

        # Gaussian smoothing
        anomaly_map_np = cv2.GaussianBlur(
            anomaly_map_np.astype(np.float32),
            ksize=(0, 0),
            sigmaX=self._smooth_sigma,
        )

        return anomaly_score, anomaly_map_np

    # ------------------------------------------------------------------
    # DetectStep-compatible result
    # ------------------------------------------------------------------

    def detect(
        self,
        image: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run detection and return a result dictionary.

        Parameters
        ----------
        image:
            ``(H, W, 3)`` uint8 image.
        threshold:
            Override model threshold; uses ``model.threshold`` by default.

        Returns
        -------
        dict
            Keys: ``anomaly_score``, ``is_defective``, ``defect_mask``,
            ``defect_regions``, ``error_map``.
        """
        thresh = threshold if threshold is not None else self.model.threshold
        anomaly_score, anomaly_map = self.score_image(image)

        is_defective = anomaly_score > thresh

        # Binary defect mask via Otsu
        defect_mask = self._create_defect_mask(anomaly_map)

        # Extract connected-component regions
        defect_regions = self._extract_regions(defect_mask)

        return {
            "anomaly_score": anomaly_score,
            "is_defective": is_defective,
            "defect_mask": defect_mask,
            "defect_regions": defect_regions,
            "error_map": anomaly_map,
        }

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    @staticmethod
    def _create_defect_mask(anomaly_map: np.ndarray) -> np.ndarray:
        """Convert normalised anomaly map to a binary defect mask (uint8 0/255)."""
        map_u8 = (anomaly_map * 255).astype(np.uint8)
        _, mask = cv2.threshold(
            map_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return mask

    @staticmethod
    def _extract_regions(mask: np.ndarray) -> List[Dict[str, Any]]:
        """Find bounding boxes of connected components in the defect mask."""
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        regions: List[Dict[str, Any]] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 10:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            regions.append({
                "bbox": (x, y, w, h),
                "area": float(area),
                "centroid": (x + w // 2, y + h // 2),
            })
        return regions

    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Remove hooks and free resources."""
        self._teacher_hooks.remove()
        self._student_hooks.remove()


# ======================================================================
# Persistence
# ======================================================================

def save_model(model: TeacherStudentModel, path: Union[str, Path]) -> None:
    """Save a TeacherStudentModel as a ``.pt`` file.

    The file contains the student state dict together with all metadata
    needed to reconstruct the inference pipeline.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "student_state_dict": model.student_state_dict,
        "backbone_name": model.backbone_name,
        "layers": list(model.layers),
        "image_size": model.image_size,
        "threshold": model.threshold,
        "config": model.config,
    }
    torch.save(payload, str(path))
    logger.info("Teacher-Student model saved to %s", path)


def load_model(path: Union[str, Path], device: str = "auto") -> TeacherStudentModel:
    """Load a TeacherStudentModel from a ``.pt`` file.

    Parameters
    ----------
    path:
        Path to the saved ``.pt`` file.
    device:
        Device for ``torch.load`` map location.

    Returns
    -------
    TeacherStudentModel
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    resolved_device = _select_device(device)
    payload = torch.load(str(path), map_location=resolved_device, weights_only=True)

    model = TeacherStudentModel(
        student_state_dict=payload["student_state_dict"],
        backbone_name=payload["backbone_name"],
        layers=tuple(payload["layers"]),
        image_size=payload["image_size"],
        threshold=payload["threshold"],
        config=payload.get("config", {}),
    )
    logger.info("Teacher-Student model loaded from %s", path)
    return model
