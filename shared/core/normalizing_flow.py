"""Normalizing Flow anomaly detection (FastFlow-style).

Uses a pretrained backbone for feature extraction, then trains 2D normalizing
flow models on each feature layer.  Low log-likelihood under the flow = anomaly.

References: Yu et al., FastFlow, arXiv 2111.07677, 2021.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

try:
    from dl_anomaly.core.dataset import DefectFreeDataset
    from dl_anomaly.core.preprocessor import ImagePreprocessor
except ImportError:
    from core.preprocessor import ImagePreprocessor  # type: ignore[assignment]
    DefectFreeDataset = None  # type: ignore[assignment,misc]

from shared.op_logger import log_operation

logger = logging.getLogger(__name__)

# -- Device selection ------------------------------------------------------

def _select_device(device: str) -> torch.device:
    """Resolve ``'auto'`` to the best available accelerator."""
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# -- Backbone registry -----------------------------------------------------

_BACKBONE_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}

def _populate_registry() -> None:
    if _BACKBONE_REGISTRY:
        return
    from torchvision import models
    _BACKBONE_REGISTRY["resnet18"] = lambda: models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT,
    )
    _BACKBONE_REGISTRY["wide_resnet50_2"] = lambda: models.wide_resnet50_2(
        weights=models.Wide_ResNet50_2_Weights.DEFAULT,
    )

# -- Feature extractor (frozen backbone + hooks) ---------------------------

class _FeatureExtractor(nn.Module):
    """Frozen backbone with forward hooks for multi-layer feature capture."""

    def __init__(self, backbone_name: str, layers: Tuple[str, ...],
                 device: torch.device) -> None:
        super().__init__()
        _populate_registry()
        if backbone_name not in _BACKBONE_REGISTRY:
            raise ValueError(f"Unknown backbone '{backbone_name}'. "
                             f"Available: {list(_BACKBONE_REGISTRY)}")
        self.backbone: nn.Module = _BACKBONE_REGISTRY[backbone_name]()
        self.backbone.to(device).eval()
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.layers = layers
        self._features: Dict[str, Tensor] = {}
        self._hooks: list = []
        for name in layers:
            module = dict(self.backbone.named_modules()).get(name)
            if module is None:
                raise ValueError(f"Layer '{name}' not found in '{backbone_name}'.")
            self._hooks.append(
                module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str) -> Callable:
        def fn(_mod: nn.Module, _inp: Any, out: Tensor) -> None:
            self._features[name] = out
        return fn

    @torch.no_grad()
    def extract(self, x: Tensor) -> Dict[str, Tensor]:
        """Return ``{layer_name: (B, C, H, W)}`` feature maps."""
        self._features.clear()
        self.backbone(x)
        return dict(self._features)

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

# -- Affine coupling block -------------------------------------------------

class AffineCouplingBlock(nn.Module):
    """Single affine coupling layer with 2D conv subnet.

    Splits channels in half; one half is transformed by a learned affine
    function conditioned on the other (unchanged) half.
    """

    def __init__(self, channels: int, hidden_channels: int = 256) -> None:
        super().__init__()
        assert channels % 2 == 0, "channels must be even for the split"
        half = channels // 2
        self.subnet = nn.Sequential(
            nn.Conv2d(half, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, half * 2, 3, padding=1),
        )
        # Zero-init last layer so the coupling starts as identity.
        nn.init.zeros_(self.subnet[-1].weight)
        nn.init.zeros_(self.subnet[-1].bias)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns ``(output, log_det_jacobian)``."""
        x_a, x_b = x.chunk(2, dim=1)
        log_s, t = self.subnet(x_a).chunk(2, dim=1)
        log_s = torch.clamp(log_s, -2.0, 2.0)
        y_b = x_b * torch.exp(log_s) + t
        log_det = log_s.sum(dim=(1, 2, 3))  # (B,)
        return torch.cat([x_a, y_b], dim=1), log_det

    def inverse(self, y: Tensor) -> Tensor:
        """Inverse transform for sampling / debugging."""
        y_a, y_b = y.chunk(2, dim=1)
        log_s, t = self.subnet(y_a).chunk(2, dim=1)
        log_s = torch.clamp(log_s, -2.0, 2.0)
        x_b = (y_b - t) * torch.exp(-log_s)
        return torch.cat([y_a, x_b], dim=1)

# -- Normalizing flow (stack of coupling + permutation) --------------------

class NormalizingFlow2D(nn.Module):
    """Stack of affine coupling blocks with fixed channel permutations.

    After training on normal features, latent *z* approximates N(0,1).
    Anomalous features yield low probability under the learned density.
    """

    def __init__(self, channels: int, n_blocks: int = 8,
                 hidden_channels: int = 256) -> None:
        super().__init__()
        self.channels = channels
        self.n_blocks = n_blocks
        self.blocks = nn.ModuleList()
        self.permutations = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(AffineCouplingBlock(channels, hidden_channels))
            # Fixed random channel permutation via frozen 1x1 conv.
            perm = nn.Conv2d(channels, channels, 1, bias=False)
            weight = torch.zeros(channels, channels, 1, 1)
            indices = torch.randperm(channels)
            for i, j in enumerate(indices):
                weight[i, j, 0, 0] = 1.0
            perm.weight.data = weight
            perm.weight.requires_grad = False
            self.permutations.append(perm)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns ``(z, total_log_det)``. *z* should be ~N(0,1) for normal."""
        total_log_det = torch.zeros(x.shape[0], device=x.device)
        for block, perm in zip(self.blocks, self.permutations):
            x, ld = block(x)
            total_log_det = total_log_det + ld
            x = perm(x)
        return x, total_log_det

    def log_prob(self, x: Tensor) -> Tensor:
        """Per-sample log probability (more negative = more anomalous)."""
        z, log_det = self.forward(x)
        log_pz = -0.5 * (z ** 2 + math.log(2 * math.pi))
        return log_pz.sum(dim=(1, 2, 3)) + log_det  # (B,)

    def log_prob_map(self, x: Tensor) -> Tensor:
        """Spatial log-prob map ``(B, H, W)`` -- sum over channels only."""
        z, _log_det = self.forward(x)
        log_pz = -0.5 * (z ** 2 + math.log(2 * math.pi))
        return log_pz.sum(dim=1)

# -- Serialisable model container ------------------------------------------

@dataclass
class NormFlowModel:
    """Trained normalizing flow model for anomaly detection."""
    flow_state_dicts: List[dict]   # one per feature layer
    backbone_name: str
    layers: Tuple[str, ...]
    image_size: int
    threshold: float
    config: Dict[str, Any] = field(default_factory=dict)

# -- Persistence -----------------------------------------------------------

def save_model(model: NormFlowModel, path: Union[str, Path]) -> None:
    """Save a :class:`NormFlowModel` to disk as a ``.pt`` archive."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "flow_state_dicts": model.flow_state_dicts,
        "backbone_name": model.backbone_name,
        "layers": list(model.layers),
        "image_size": model.image_size,
        "threshold": model.threshold,
        "config": model.config,
    }, str(path))
    logger.info("NormFlow model saved to %s", path)

def load_model(path: Union[str, Path], device: str = "auto") -> NormFlowModel:
    """Load a :class:`NormFlowModel` from a ``.pt`` archive."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    dev = _select_device(device)
    payload = torch.load(str(path), map_location=dev, weights_only=False)
    model = NormFlowModel(
        flow_state_dicts=payload["flow_state_dicts"],
        backbone_name=payload["backbone_name"],
        layers=tuple(payload["layers"]),
        image_size=payload["image_size"],
        threshold=payload["threshold"],
        config=payload.get("config", {}),
    )
    logger.info("NormFlow model loaded from %s", path)
    return model

# -- Trainer ---------------------------------------------------------------

class NormFlowTrainer:
    """Train normalizing flows on normal-image features.

    One :class:`NormalizingFlow2D` per backbone layer, trained independently
    to model that layer's feature distribution via maximum likelihood.
    """

    def __init__(
        self,
        backbone_name: str = "wide_resnet50_2",
        layers: Tuple[str, ...] = ("layer2", "layer3"),
        image_size: int = 256,
        n_flow_blocks: int = 8,
        hidden_channels: int = 256,
        device: str = "auto",
    ) -> None:
        self.backbone_name = backbone_name
        self.layers = layers
        self.image_size = image_size
        self.n_flow_blocks = n_flow_blocks
        self.hidden_channels = hidden_channels
        self.device = _select_device(device)
        self.preprocessor = ImagePreprocessor(image_size, grayscale=False)
        self.extractor = _FeatureExtractor(backbone_name, layers, self.device)

    @log_operation(logger)
    def train(
        self,
        image_dir: Union[str, Path],
        epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 16,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> NormFlowModel:
        """Train flows on defect-free images.  Loss = negative log-likelihood."""
        image_dir = Path(image_dir)
        transform = self.preprocessor.get_transforms(augment=False)
        dataset = DefectFreeDataset(image_dir, transform=transform, grayscale=False)
        if len(dataset) == 0:
            raise RuntimeError(f"No images found in {image_dir}")

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=0,
                            pin_memory=(self.device.type == "cuda"))
        logger.info("NormFlow training: %d images, backbone=%s, layers=%s, "
                     "epochs=%d, device=%s",
                     len(dataset), self.backbone_name, self.layers,
                     epochs, self.device)

        # Probe forward pass to get per-layer channel counts
        probe_features = self.extractor.extract(
            next(iter(loader))[0].to(self.device))
        channel_counts: Dict[str, int] = {
            n: f.shape[1] for n, f in probe_features.items()}
        for name, c in channel_counts.items():
            if c % 2 != 0:
                raise ValueError(
                    f"Layer '{name}' has {c} channels (odd); needs even count.")

        # Build one flow per layer
        flows: Dict[str, NormalizingFlow2D] = {}
        optimizers: Dict[str, torch.optim.Optimizer] = {}
        for name in self.layers:
            flow = NormalizingFlow2D(
                channel_counts[name], self.n_flow_blocks,
                self.hidden_channels).to(self.device)
            flows[name] = flow
            optimizers[name] = torch.optim.AdamW(flow.parameters(), lr=lr)

        # Training loop
        n_batches = len(loader)
        for epoch in range(epochs):
            epoch_losses: Dict[str, float] = {n: 0.0 for n in self.layers}
            for images, _paths in loader:
                images = images.to(self.device)
                features = self.extractor.extract(images)
                for name in self.layers:
                    loss = -flows[name].log_prob(features[name]).mean()
                    optimizers[name].zero_grad()
                    loss.backward()
                    optimizers[name].step()
                    epoch_losses[name] += loss.item()

            avg = {n: v / n_batches for n, v in epoch_losses.items()}
            if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
                loss_str = ", ".join(f"{n}={v:.4f}" for n, v in avg.items())
                logger.info("Epoch %d/%d  NLL: %s", epoch + 1, epochs, loss_str)
            if progress_callback:
                progress_callback({"stage": "training", "epoch": epoch + 1,
                                   "total_epochs": epochs,
                                   "progress": (epoch + 1) / epochs,
                                   "losses": avg})

        # Fit threshold from training scores
        threshold = self._fit_threshold(loader, flows)

        flow_state_dicts = [flows[n].cpu().state_dict() for n in self.layers]
        model = NormFlowModel(
            flow_state_dicts=flow_state_dicts,
            backbone_name=self.backbone_name, layers=self.layers,
            image_size=self.image_size, threshold=threshold,
            config={"n_flow_blocks": self.n_flow_blocks,
                    "hidden_channels": self.hidden_channels,
                    "epochs": epochs, "lr": lr, "batch_size": batch_size,
                    "channel_counts": channel_counts})
        logger.info("NormFlow training complete (threshold=%.4f)", threshold)
        return model

    @torch.no_grad()
    def _fit_threshold(self, loader: DataLoader,
                       flows: Dict[str, NormalizingFlow2D]) -> float:
        """Set threshold as mean + 3*std of training anomaly scores."""
        for flow in flows.values():
            flow.eval()
        scores: List[float] = []
        for images, _paths in loader:
            images = images.to(self.device)
            features = self.extractor.extract(images)
            batch_score = torch.zeros(images.shape[0], device=self.device)
            for name in self.layers:
                batch_score = batch_score - flows[name].log_prob(features[name])
            scores.extend(batch_score.cpu().tolist())
        arr = np.array(scores, dtype=np.float64)
        threshold = float(arr.mean() + 3.0 * arr.std())
        logger.info("Threshold: mean=%.4f, std=%.4f, value=%.4f",
                     arr.mean(), arr.std(), threshold)
        return threshold

# -- Inference -------------------------------------------------------------

class NormFlowInference:
    """Score images against a trained normalizing flow model."""

    def __init__(self, model: NormFlowModel, device: str = "auto") -> None:
        self.model = model
        self.device = _select_device(device)
        self.extractor = _FeatureExtractor(
            model.backbone_name, model.layers, self.device)
        self.preprocessor = ImagePreprocessor(model.image_size, grayscale=False)

        # Rebuild flows from saved state dicts
        cc = model.config.get("channel_counts", {})
        n_blk = model.config.get("n_flow_blocks", 8)
        h_ch = model.config.get("hidden_channels", 256)
        self.flows: Dict[str, NormalizingFlow2D] = {}
        for idx, name in enumerate(model.layers):
            ch = cc.get(name)
            if ch is None:
                raise ValueError(
                    f"channel_counts missing for layer '{name}' in config.")
            flow = NormalizingFlow2D(ch, n_blk, h_ch)
            flow.load_state_dict(model.flow_state_dicts[idx])
            flow.to(self.device).eval()
            self.flows[name] = flow

    @torch.no_grad()
    @log_operation(logger)
    def score_image(self, image: np.ndarray) -> Tuple[float, np.ndarray]:
        """Returns ``(anomaly_score, anomaly_map)``.

        ``anomaly_map`` is a ``(H, W)`` float32 heatmap of per-pixel anomaly
        intensity (negative log-prob), upsampled to the original resolution.
        """
        h_orig, w_orig = image.shape[:2]
        tensor = self.preprocessor.preprocess(image)
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        tensor = tensor.to(self.device)
        features = self.extractor.extract(tensor)

        combined_map: Optional[Tensor] = None
        total_nll = 0.0
        for name in self.model.layers:
            feat = features[name]
            flow = self.flows[name]
            neg_log_map = -flow.log_prob_map(feat)  # (1, H_l, W_l)
            if combined_map is None:
                combined_map = neg_log_map
            else:
                target = (combined_map.shape[1], combined_map.shape[2])
                neg_log_map = F.interpolate(
                    neg_log_map.unsqueeze(1), size=target,
                    mode="bilinear", align_corners=False).squeeze(1)
                combined_map = combined_map + neg_log_map
            total_nll += (-flow.log_prob(feat)).item()

        amap = F.interpolate(
            combined_map.unsqueeze(1), size=(h_orig, w_orig),
            mode="bilinear", align_corners=False,
        ).squeeze(0).squeeze(0)
        return float(total_nll), amap.cpu().numpy().astype(np.float32)

    def detect(self, image: np.ndarray,
               threshold: Optional[float] = None) -> Dict[str, Any]:
        """DetectStep-compatible interface."""
        thr = threshold if threshold is not None else self.model.threshold
        score, amap = self.score_image(image)
        return {"is_anomalous": score > thr, "anomaly_score": score,
                "threshold": thr, "anomaly_map": amap}
