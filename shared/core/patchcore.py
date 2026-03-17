"""PatchCore anomaly detection using pre-trained CNN features and memory bank.

Implements the PatchCore algorithm (Roth et al., 2022) for industrial anomaly
detection.  A pre-trained backbone extracts multi-layer patch features from
defect-free training images; a coreset of those features is stored as a
*memory bank*.  At inference time, each test patch is compared against the
bank and the maximum nearest-neighbour distance serves as the anomaly score.

Key components
--------------
- ``FeatureExtractor`` -- multi-layer feature extraction via forward hooks.
- ``PatchCoreModel`` -- serialisable dataclass holding the memory bank.
- ``PatchCoreTrainer`` -- builds the memory bank from a training image set.
- ``PatchCoreInference`` -- scores images against the memory bank.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader

try:
    from dl_anomaly.config import Config
    from dl_anomaly.core.anomaly_scorer import AnomalyScorer
    from dl_anomaly.core.dataset import DefectFreeDataset
    from dl_anomaly.core.preprocessor import ImagePreprocessor
    from dl_anomaly.pipeline.inference import InspectionResult
except ImportError:
    from config import Config  # type: ignore[assignment]
    from core.preprocessor import ImagePreprocessor  # type: ignore[assignment]
    from core.inspector import InspectionResult  # type: ignore[assignment]
    AnomalyScorer = None  # type: ignore[assignment,misc]
    DefectFreeDataset = None  # type: ignore[assignment,misc]
from shared.op_logger import log_operation

logger = logging.getLogger(__name__)


# ======================================================================
# Supported backbones
# ======================================================================

_BACKBONE_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}


def _populate_registry() -> None:
    """Lazily populate the backbone registry from torchvision."""
    if _BACKBONE_REGISTRY:
        return
    from torchvision import models

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
# Feature extractor
# ======================================================================

class FeatureExtractor(nn.Module):
    """Wraps a pre-trained backbone for multi-layer feature extraction.

    Forward hooks are registered on the specified *layers* so that
    intermediate feature maps can be captured without modifying the
    backbone itself.

    Parameters
    ----------
    backbone_name:
        One of :func:`list_available_backbones`.
    layers:
        Names of backbone sub-modules whose outputs will be captured
        (e.g. ``("layer2", "layer3")`` for ResNet variants).
    device:
        ``'cuda'`` or ``'cpu'``.
    """

    def __init__(
        self,
        backbone_name: str = "wide_resnet50_2",
        layers: Tuple[str, ...] = ("layer2", "layer3"),
        device: str = "cpu",
    ) -> None:
        super().__init__()
        _populate_registry()

        if backbone_name not in _BACKBONE_REGISTRY:
            raise ValueError(
                f"Unknown backbone '{backbone_name}'. "
                f"Choose from {list_available_backbones()}."
            )

        self.backbone_name = backbone_name
        self.layers = layers
        self.device = torch.device(device)

        self.backbone: nn.Module = _BACKBONE_REGISTRY[backbone_name]()
        self.backbone.to(self.device)
        self.backbone.eval()

        # Freeze all parameters -- no training needed
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Hook storage
        self._features: Dict[str, torch.Tensor] = {}
        self._hooks: List[torch.utils.hooks.RemovableHook] = []
        self._register_hooks()

        # Cache feature dimensions (populated on first forward pass)
        self._feature_dims: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def _register_hooks(self) -> None:
        """Attach forward hooks to each requested layer."""
        for layer_name in self.layers:
            module = dict(self.backbone.named_modules()).get(layer_name)
            if module is None:
                raise ValueError(
                    f"Layer '{layer_name}' not found in backbone "
                    f"'{self.backbone_name}'."
                )
            hook = module.register_forward_hook(self._make_hook(layer_name))
            self._hooks.append(hook)

    def _make_hook(self, layer_name: str) -> Callable:
        """Return a hook function that stores the output of *layer_name*."""
        def hook_fn(_module: nn.Module, _input: Any, output: torch.Tensor) -> None:
            self._features[layer_name] = output
        return hook_fn

    def remove_hooks(self) -> None:
        """Remove all registered forward hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract(self, image_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run the backbone and return intermediate features.

        Parameters
        ----------
        image_tensor:
            ``(B, C, H, W)`` normalised image batch.

        Returns
        -------
        dict
            Mapping from layer name to feature tensor ``(B, C_l, H_l, W_l)``.
        """
        self._features.clear()
        image_tensor = image_tensor.to(self.device)
        self.backbone(image_tensor)

        # Cache channel dimensions on first call
        if not self._feature_dims:
            for name, feat in self._features.items():
                self._feature_dims[name] = feat.shape[1]

        return dict(self._features)

    def get_feature_dim(self) -> int:
        """Return the total feature channels across all hooked layers.

        Must be called after at least one :meth:`extract` call so that
        channel dimensions are known.
        """
        if not self._feature_dims:
            raise RuntimeError(
                "Feature dimensions unknown -- call extract() first."
            )
        return sum(self._feature_dims.values())

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Alias for :meth:`extract` to satisfy ``nn.Module`` interface."""
        return self.extract(x)


# ======================================================================
# PatchCore model (serialisable)
# ======================================================================

@dataclass
class PatchCoreModel:
    """Serialisable container for a trained PatchCore memory bank.

    Attributes
    ----------
    memory_bank:
        ``(N, D)`` coreset of patch features stored as float16.
    backbone_name:
        Name of the backbone used during training.
    layers:
        Tuple of layer names used for feature extraction.
    image_size:
        Spatial resolution of input images.
    feature_dim:
        Dimensionality *D* of each patch feature vector.
    coreset_ratio:
        Fraction of total patches retained in the coreset.
    threshold:
        Anomaly threshold fitted from training scores.
    config:
        Serialisable configuration dictionary.
    """

    memory_bank: np.ndarray
    backbone_name: str
    layers: Tuple[str, ...]
    image_size: int
    feature_dim: int
    coreset_ratio: float
    threshold: Optional[float]
    config: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Save the model to a ``.npz`` file.

        The memory bank is stored as a numpy array.  All scalar metadata
        and the configuration dictionary are stored as a JSON string
        inside the archive.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        metadata = {
            "backbone_name": self.backbone_name,
            "layers": list(self.layers),
            "image_size": self.image_size,
            "feature_dim": self.feature_dim,
            "coreset_ratio": self.coreset_ratio,
            "threshold": self.threshold,
            "config": self.config,
        }

        np.savez_compressed(
            str(path),
            memory_bank=self.memory_bank,
            metadata=np.array([json.dumps(metadata)]),
        )
        logger.info(
            "PatchCore model saved to %s  (bank: %s, %.1f MB)",
            path,
            self.memory_bank.shape,
            self.memory_bank.nbytes / 1024 / 1024,
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "PatchCoreModel":
        """Load a PatchCore model from a ``.npz`` file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        data = np.load(str(path), allow_pickle=False)
        memory_bank = data["memory_bank"]
        metadata = json.loads(str(data["metadata"][0]))

        logger.info(
            "PatchCore model loaded from %s  (bank: %s)",
            path,
            memory_bank.shape,
        )
        return cls(
            memory_bank=memory_bank,
            backbone_name=metadata["backbone_name"],
            layers=tuple(metadata["layers"]),
            image_size=metadata["image_size"],
            feature_dim=metadata["feature_dim"],
            coreset_ratio=metadata["coreset_ratio"],
            threshold=metadata.get("threshold"),
            config=metadata.get("config", {}),
        )


# ======================================================================
# Trainer
# ======================================================================

class PatchCoreTrainer:
    """Build a PatchCore memory bank from defect-free training images.

    Parameters
    ----------
    config:
        Project-wide configuration.
    backbone_name:
        Pre-trained backbone to use for feature extraction.
    layers:
        Backbone layers whose features are concatenated.
    coreset_ratio:
        Fraction of patch features to retain via greedy coreset selection.
    device:
        ``'cuda'`` or ``'cpu'``.
    """

    def __init__(
        self,
        config: Config,
        backbone_name: str = "wide_resnet50_2",
        layers: Tuple[str, ...] = ("layer2", "layer3"),
        coreset_ratio: float = 0.01,
        device: str = "cpu",
    ) -> None:
        self.config = config
        self.backbone_name = backbone_name
        self.layers = layers
        self.coreset_ratio = coreset_ratio
        self.device = device

        self.preprocessor = ImagePreprocessor(config.image_size, grayscale=False)
        self.extractor = FeatureExtractor(backbone_name, layers, device)

    # ------------------------------------------------------------------
    # Core training entry point
    # ------------------------------------------------------------------

    @log_operation(logger)
    def train(
        self,
        image_dir: Union[str, Path],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> PatchCoreModel:
        """Build the PatchCore memory bank.

        Parameters
        ----------
        image_dir:
            Directory of defect-free training images.
        progress_callback:
            Called with a status dict after each processing stage.

        Returns
        -------
        PatchCoreModel
            The trained model containing the coreset memory bank.
        """
        image_dir = Path(image_dir)

        # 1. Dataset -------------------------------------------------------
        transform = self.preprocessor.get_transforms(augment=False)
        dataset = DefectFreeDataset(image_dir, transform=transform, grayscale=False)
        if len(dataset) == 0:
            raise RuntimeError(f"No images found in {image_dir}")

        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(str(self.device) == "cuda"),
        )
        logger.info(
            "PatchCore training: %d images, backbone=%s, layers=%s",
            len(dataset),
            self.backbone_name,
            self.layers,
        )

        if progress_callback:
            progress_callback({"stage": "feature_extraction", "progress": 0.0})

        # 2. Extract multi-layer features ----------------------------------
        all_features: List[np.ndarray] = []
        per_image_patches: List[np.ndarray] = []  # cached for threshold fitting
        n_batches = len(loader)

        for batch_idx, (images, _paths) in enumerate(loader):
            features = self.extractor.extract(images)
            merged = self._merge_features(features)  # (B, D, H, W)

            # Reshape to (B * H * W, D)
            b, d, h, w = merged.shape
            patches = merged.permute(0, 2, 3, 1).reshape(b, h * w, d)
            patches_np = patches.cpu().numpy().astype(np.float32)
            all_features.append(patches_np.reshape(-1, d))

            # Cache per-image patches for threshold fitting (avoids re-extraction)
            for i in range(b):
                per_image_patches.append(patches_np[i])

            if progress_callback:
                progress_callback({
                    "stage": "feature_extraction",
                    "progress": (batch_idx + 1) / n_batches,
                    "batch": batch_idx + 1,
                    "total_batches": n_batches,
                })

        feature_matrix = np.concatenate(all_features, axis=0)
        feature_dim = feature_matrix.shape[1]
        logger.info(
            "Extracted %d patch features (dim=%d)",
            feature_matrix.shape[0],
            feature_dim,
        )

        # 3. Coreset subsampling -------------------------------------------
        if progress_callback:
            progress_callback({"stage": "coreset_selection", "progress": 0.0})

        memory_bank = self._coreset_selection(feature_matrix, self.coreset_ratio)
        memory_bank = memory_bank.astype(np.float16)
        logger.info(
            "Coreset: %d / %d patches (ratio=%.4f)",
            memory_bank.shape[0],
            feature_matrix.shape[0],
            self.coreset_ratio,
        )

        if progress_callback:
            progress_callback({"stage": "coreset_selection", "progress": 1.0})

        # 4. Fit threshold from training scores ----------------------------
        if progress_callback:
            progress_callback({"stage": "threshold_fitting", "progress": 0.0})

        threshold = self._fit_training_threshold(
            per_image_patches, memory_bank.astype(np.float32),
        )

        if progress_callback:
            progress_callback({"stage": "threshold_fitting", "progress": 1.0})

        # 5. Assemble model ------------------------------------------------
        model = PatchCoreModel(
            memory_bank=memory_bank,
            backbone_name=self.backbone_name,
            layers=self.layers,
            image_size=self.config.image_size,
            feature_dim=feature_dim,
            coreset_ratio=self.coreset_ratio,
            threshold=threshold,
            config=self.config.to_dict(),
        )

        logger.info(
            "PatchCore training complete (bank=%s, threshold=%.6f)",
            memory_bank.shape,
            threshold,
        )
        return model

    # ------------------------------------------------------------------
    # Feature merging
    # ------------------------------------------------------------------

    def _merge_features(
        self, features: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Concatenate multi-layer features along the channel axis.

        All feature maps are bilinearly interpolated to the spatial
        resolution of the *smallest* layer before concatenation.
        """
        tensors = list(features.values())
        # Use the smallest spatial size as target
        target_h = min(t.shape[2] for t in tensors)
        target_w = min(t.shape[3] for t in tensors)

        aligned: List[torch.Tensor] = []
        for t in tensors:
            if t.shape[2] != target_h or t.shape[3] != target_w:
                t = F.interpolate(
                    t,
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                )
            aligned.append(t)

        return torch.cat(aligned, dim=1)  # (B, D_total, H, W)

    # ------------------------------------------------------------------
    # Greedy coreset selection (farthest-point sampling)
    # ------------------------------------------------------------------

    @staticmethod
    def _coreset_selection(
        features: np.ndarray, ratio: float,
    ) -> np.ndarray:
        """Select a representative coreset via greedy farthest-point sampling.

        Parameters
        ----------
        features:
            ``(N, D)`` matrix of patch features.
        ratio:
            Fraction of patches to retain.

        Returns
        -------
        np.ndarray
            ``(M, D)`` coreset where ``M = max(int(N * ratio), 1)``.
        """
        n = features.shape[0]
        target = max(int(n * ratio), 1)

        if target >= n:
            logger.info("Coreset target (%d) >= total (%d); keeping all", target, n)
            return features.copy()

        logger.info("Coreset selection: %d -> %d patches", n, target)

        rng = np.random.default_rng(seed=42)
        first_idx = int(rng.integers(0, n))
        selected: List[int] = [first_idx]

        # Initial distances from the first selected point
        diff = features - features[first_idx]
        min_distances = np.sqrt(np.einsum('ij,ij->i', diff, diff))

        for i in range(target - 1):
            idx = int(np.argmax(min_distances))
            selected.append(idx)

            diff = features - features[idx]
            new_dist = np.sqrt(np.einsum('ij,ij->i', diff, diff))
            min_distances = np.minimum(min_distances, new_dist)

            if (i + 1) % 500 == 0:
                logger.debug(
                    "Coreset progress: %d / %d selected", i + 1, target,
                )

        return features[selected]

    # ------------------------------------------------------------------
    # Training threshold
    # ------------------------------------------------------------------

    def _fit_training_threshold(
        self,
        per_image_patches: List[np.ndarray],
        memory_bank: np.ndarray,
        percentile: float = 99.5,
    ) -> float:
        """Score training images and set threshold at the given percentile.

        Uses cached per-image patch features to avoid redundant backbone
        forward passes.
        """
        nn_index = NearestNeighbors(
            n_neighbors=1, metric="euclidean", algorithm="ball_tree",
        )
        nn_index.fit(memory_bank)

        scores: List[float] = []
        for patch_np in per_image_patches:
            dists, _ = nn_index.kneighbors(patch_np)
            image_score = float(np.max(dists))
            scores.append(image_score)

        threshold = float(np.percentile(scores, percentile))
        logger.info(
            "Training threshold fitted: %.6f (percentile=%.1f, n=%d)",
            threshold,
            percentile,
            len(scores),
        )
        return threshold


# ======================================================================
# Inference
# ======================================================================

class PatchCoreInference:
    """Score images against a trained PatchCore memory bank.

    Parameters
    ----------
    model:
        A trained :class:`PatchCoreModel`.
    device:
        ``'cuda'`` or ``'cpu'``.
    """

    def __init__(
        self,
        model: PatchCoreModel,
        device: str = "cpu",
        n_neighbors: int = 9,
        smooth_sigma: float = 4.0,
        feature_resize_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        self.model = model
        self.device = device

        self.extractor = FeatureExtractor(
            model.backbone_name, model.layers, device,
        )
        self.preprocessor = ImagePreprocessor(model.image_size, grayscale=False)
        self.scorer = AnomalyScorer(device=device)

        # Upcast memory bank to float32 for distance computation
        self._bank_f32: np.ndarray = model.memory_bank.astype(np.float32)

        # Gaussian kernel for anomaly map smoothing
        self._smooth_sigma: float = smooth_sigma
        self._knn_k: int = n_neighbors

        # Optional explicit feature resize target (None = use smallest layer)
        self._feature_resize_size: Optional[Tuple[int, int]] = feature_resize_size

        # Persist tuneable parameters in the model config so they survive
        # save/load round-trips.
        model.config.setdefault("n_neighbors", n_neighbors)
        model.config.setdefault("smooth_sigma", smooth_sigma)
        if feature_resize_size is not None:
            model.config.setdefault("feature_resize_size", list(feature_resize_size))

        # Pre-fit NearestNeighbors index for fast k-NN queries
        self._nn_index = NearestNeighbors(
            n_neighbors=self._knn_k,
            metric="euclidean",
            algorithm="ball_tree",
        )
        self._nn_index.fit(self._bank_f32)

    # ------------------------------------------------------------------
    # Single-image scoring
    # ------------------------------------------------------------------

    @torch.no_grad()
    @log_operation(logger)
    def score_image(
        self, image_tensor: torch.Tensor,
    ) -> Tuple[float, np.ndarray]:
        """Compute image-level anomaly score and pixel-level anomaly map.

        Parameters
        ----------
        image_tensor:
            ``(C, H, W)`` or ``(1, C, H, W)`` normalised image tensor.

        Returns
        -------
        tuple
            ``(image_score, anomaly_map)`` where *anomaly_map* has the
            same spatial resolution as the input image.
        """
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        # 1. Extract and merge features ------------------------------------
        features = self.extractor.extract(image_tensor)
        merged = self._merge_features(features)  # (1, D, H_f, W_f)
        _, d, h_feat, w_feat = merged.shape

        patches = merged.permute(0, 2, 3, 1).reshape(-1, d)  # (H*W, D)
        patch_np = patches.cpu().numpy().astype(np.float32)

        # 2. k-NN distances against memory bank ----------------------------
        knn_dists, _ = self._nn_index.kneighbors(patch_np)
        patch_scores = knn_dists.min(axis=1)

        # 3. Image-level score: max of patch scores ------------------------
        image_score = float(np.max(patch_scores))

        # 4. Anomaly map: upsample to original image resolution ------------
        score_map = patch_scores.reshape(h_feat, w_feat)
        score_tensor = (
            torch.from_numpy(score_map)
            .unsqueeze(0)
            .unsqueeze(0)
            .float()
        )
        anomaly_map = F.interpolate(
            score_tensor,
            size=(self.model.image_size, self.model.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze().numpy()

        # 5. Gaussian smoothing --------------------------------------------
        anomaly_map = self.scorer.create_anomaly_map(
            anomaly_map, gaussian_sigma=self._smooth_sigma,
        )

        return image_score, anomaly_map

    # ------------------------------------------------------------------
    # Feature merging (mirrors trainer)
    # ------------------------------------------------------------------

    def _merge_features(
        self, features: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Concatenate multi-layer features, matching spatial resolution.

        When ``_feature_resize_size`` is set, all feature maps are resized
        to that explicit ``(H, W)`` instead of defaulting to the smallest
        layer size.
        """
        tensors = list(features.values())

        if self._feature_resize_size is not None:
            target_h, target_w = self._feature_resize_size
        else:
            # Default behaviour: use the smallest spatial size
            target_h = min(t.shape[2] for t in tensors)
            target_w = min(t.shape[3] for t in tensors)

        aligned: List[torch.Tensor] = []
        for t in tensors:
            if t.shape[2] != target_h or t.shape[3] != target_w:
                t = F.interpolate(
                    t,
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                )
            aligned.append(t)

        return torch.cat(aligned, dim=1)

    # ------------------------------------------------------------------
    # High-level inspection
    # ------------------------------------------------------------------

    def inspect_single(
        self, image_path: Union[str, Path],
    ) -> InspectionResult:
        """Run the full PatchCore inspection pipeline on a single image.

        Returns an :class:`InspectionResult` compatible with the
        existing inference pipeline.
        """
        image_path = Path(image_path)
        tensor = self.preprocessor.load_and_preprocess(image_path)
        original_np = self.preprocessor.inverse_normalize(tensor)

        image_score, anomaly_map = self.score_image(tensor)

        is_defective = False
        if self.model.threshold is not None:
            is_defective = image_score > self.model.threshold

        # Binary defect mask via thresholding
        defect_mask = self._create_defect_mask(anomaly_map)

        return InspectionResult(
            original=original_np,
            reconstruction=original_np,  # PatchCore has no reconstruction
            error_map=anomaly_map,
            defect_mask=defect_mask,
            anomaly_score=image_score,
            is_defective=is_defective,
            defect_regions=[],
        )

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
        """Convert a normalised anomaly map to a binary defect mask.

        Uses Otsu thresholding followed by morphological cleanup.
        Returns a uint8 ``(H, W)`` array with values 0 or 255.
        """
        import cv2

        map_u8 = (anomaly_map * 255).astype(np.uint8)
        _, mask = cv2.threshold(
            map_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        return mask
