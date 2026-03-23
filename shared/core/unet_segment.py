"""U-Net anomaly segmentation for pixel-level defect detection.

Supports two modes:

1. **Supervised** -- Train with defect mask annotations (OK images + NG images
   with binary masks).  Uses a combination of BCE and Dice loss.
2. **Reconstruction** -- Train on OK images only.  The U-Net acts as an
   autoencoder (``in_channels == num_classes``); anomaly is measured via
   per-pixel reconstruction error.

Key components
--------------
- ``ConvBlock`` -- double-convolution building block (Conv-BN-ReLU x2).
- ``UNet`` -- standard encoder-decoder with skip connections.
- ``UNetModel`` -- serialisable dataclass holding trained weights and metadata.
- ``UNetTrainer`` -- training loop for both supervised and reconstruction modes.
- ``UNetInference`` -- scoring and segmentation at inference time.
"""

from __future__ import annotations

import json
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

from shared.op_logger import log_operation

logger = logging.getLogger(__name__)

# ImageNet normalisation constants
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ======================================================================
# Device helpers
# ======================================================================

def _resolve_device(device: str) -> torch.device:
    """Resolve ``'auto'`` to the best available device."""
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


# ======================================================================
# Model dataclass
# ======================================================================

@dataclass
class UNetModel:
    """Serialisable container for a trained U-Net and its metadata."""

    state_dict: Dict[str, Any]
    mode: str                    # "supervised" or "reconstruction"
    in_channels: int
    num_classes: int
    base_channels: int
    depth: int
    image_size: int
    threshold: float
    config: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Save the model to a ``.pth`` file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "state_dict": self.state_dict,
            "metadata": {
                "mode": self.mode,
                "in_channels": self.in_channels,
                "num_classes": self.num_classes,
                "base_channels": self.base_channels,
                "depth": self.depth,
                "image_size": self.image_size,
                "threshold": self.threshold,
                "config": self.config,
            },
        }
        torch.save(payload, str(path))
        logger.info("UNet model saved to %s", path)

    @classmethod
    def load(cls, path: Union[str, Path], device: str = "auto") -> "UNetModel":
        """Load a UNet model from a ``.pth`` file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        dev = _resolve_device(device)
        payload = torch.load(str(path), map_location=dev, weights_only=True)
        meta = payload["metadata"]

        logger.info("UNet model loaded from %s  (mode=%s)", path, meta["mode"])
        return cls(
            state_dict=payload["state_dict"],
            mode=meta["mode"],
            in_channels=meta["in_channels"],
            num_classes=meta["num_classes"],
            base_channels=meta["base_channels"],
            depth=meta["depth"],
            image_size=meta["image_size"],
            threshold=meta["threshold"],
            config=meta.get("config", {}),
        )


# Convenience aliases matching the design spec
save_model = lambda model, path: model.save(path)
load_model = UNetModel.load


# ======================================================================
# Architecture
# ======================================================================

class ConvBlock(nn.Module):
    """Double convolution: Conv2d -> BN -> ReLU -> Conv2d -> BN -> ReLU."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """Standard U-Net with configurable depth and base channel width.

    Parameters
    ----------
    in_channels:
        Number of input channels (e.g. 3 for RGB).
    num_classes:
        Number of output channels.  For binary segmentation use 1.
        For reconstruction mode set equal to *in_channels*.
    base_channels:
        Channel count in the first encoder level.  Doubled at each
        subsequent level.
    depth:
        Number of encoder/decoder stages (excluding the bottleneck).
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_channels: int = 32,
        depth: int = 4,
    ) -> None:
        super().__init__()
        self.depth = depth

        # Encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch_in = in_channels
        for i in range(depth):
            ch_out = base_channels * (2 ** i)
            self.encoders.append(ConvBlock(ch_in, ch_out))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            ch_in = ch_out

        # Bottleneck
        bottleneck_ch = base_channels * (2 ** depth)
        self.bottleneck = ConvBlock(ch_in, bottleneck_ch)

        # Decoder
        self.upsamplers = nn.ModuleList()
        self.decoders = nn.ModuleList()
        ch_in = bottleneck_ch
        for i in reversed(range(depth)):
            ch_skip = base_channels * (2 ** i)
            self.upsamplers.append(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            )
            self.decoders.append(ConvBlock(ch_in + ch_skip, ch_skip))
            ch_in = ch_skip

        # Output head
        self.head = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns segmentation map of shape ``(B, num_classes, H, W)``
        with values in ``[0, 1]`` (sigmoid-activated).
        """
        # Encoder path -- store skip connections
        skips: List[torch.Tensor] = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        # Decoder path -- concat skip connections
        for up, dec, skip in zip(self.upsamplers, self.decoders, reversed(skips)):
            x = up(x)
            # Handle spatial size mismatch from non-power-of-two inputs
            if x.shape != skip.shape:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=False,
                )
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        return torch.sigmoid(self.head(x))


# ======================================================================
# Internal datasets
# ======================================================================

class _SupervisedDataset(Dataset):
    """Loads (image, mask) pairs for supervised segmentation training."""

    def __init__(
        self,
        ok_dir: Path,
        ng_dir: Path,
        mask_dir: Path,
        image_size: int,
    ) -> None:
        self.image_size = image_size
        self.samples: List[Tuple[Path, Optional[Path]]] = []

        # OK images -- ground truth mask is all zeros (no defect)
        for p in sorted(ok_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS:
                self.samples.append((p, None))

        # NG images -- paired with a mask from mask_dir (same filename)
        for p in sorted(ng_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS:
                mask_path = mask_dir / p.name
                if not mask_path.exists():
                    # Try common alternative extensions
                    for ext in _IMAGE_EXTENSIONS:
                        alt = mask_dir / (p.stem + ext)
                        if alt.exists():
                            mask_path = alt
                            break
                if mask_path.exists():
                    self.samples.append((p, mask_path))
                else:
                    logger.warning(
                        "No mask found for NG image %s, skipping", p.name,
                    )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.samples[idx]

        # Load and preprocess image
        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Cannot read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size))
        img_t = self._normalise(img)

        # Load or create mask
        if mask_path is not None:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise RuntimeError(f"Cannot read mask: {mask_path}")
            mask = cv2.resize(
                mask, (self.image_size, self.image_size),
                interpolation=cv2.INTER_NEAREST,
            )
            mask_t = torch.from_numpy(mask).float().unsqueeze(0) / 255.0
        else:
            mask_t = torch.zeros(1, self.image_size, self.image_size)

        return img_t, mask_t

    @staticmethod
    def _normalise(img: np.ndarray) -> torch.Tensor:
        """HWC uint8 -> CHW float32 with ImageNet normalisation."""
        t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        mean = torch.tensor(_IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(_IMAGENET_STD).view(3, 1, 1)
        return (t - mean) / std


class _ReconstructionDataset(Dataset):
    """Loads OK images for reconstruction (autoencoder) training."""

    def __init__(self, ok_dir: Path, image_size: int) -> None:
        self.image_size = image_size
        self.paths: List[Path] = sorted(
            p for p in ok_dir.iterdir()
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = cv2.imread(str(self.paths[idx]))
        if img is None:
            raise RuntimeError(f"Cannot read image: {self.paths[idx]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size))
        return _SupervisedDataset._normalise(img)


# ======================================================================
# Losses
# ======================================================================

def _dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Soft Dice loss for binary segmentation."""
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def _bce_dice_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Combined BCE + Dice loss (equal weighting)."""
    bce = F.binary_cross_entropy(pred, target, reduction="mean")
    dice = _dice_loss(pred, target)
    return bce + dice


# ======================================================================
# Trainer
# ======================================================================

class UNetTrainer:
    """Train a U-Net for anomaly segmentation.

    Parameters
    ----------
    mode:
        ``'supervised'`` for mask-annotated training, ``'reconstruction'``
        for autoencoder-style OK-only training.
    in_channels:
        Number of input image channels (3 for RGB).
    num_classes:
        Number of output classes.  Ignored in reconstruction mode (set
        to *in_channels* automatically).
    base_channels:
        Channel count in the first encoder level.
    depth:
        Number of encoder/decoder stages.
    image_size:
        Images are resized to ``(image_size, image_size)``.
    device:
        ``'auto'``, ``'cuda'``, ``'mps'``, or ``'cpu'``.
    """

    def __init__(
        self,
        mode: str = "supervised",
        in_channels: int = 3,
        num_classes: int = 1,
        base_channels: int = 32,
        depth: int = 4,
        image_size: int = 256,
        device: str = "auto",
    ) -> None:
        if mode not in ("supervised", "reconstruction"):
            raise ValueError(f"Unknown mode '{mode}'. Choose 'supervised' or 'reconstruction'.")

        self.mode = mode
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.depth = depth
        self.image_size = image_size
        self.device = _resolve_device(device)

        # In reconstruction mode the output must reconstruct the input
        self.num_classes = in_channels if mode == "reconstruction" else num_classes

    # ------------------------------------------------------------------
    # Core training entry point
    # ------------------------------------------------------------------

    @log_operation(logger)
    def train(
        self,
        ok_dir: Union[str, Path],
        ng_dir: Optional[Union[str, Path]] = None,
        mask_dir: Optional[Union[str, Path]] = None,
        epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 8,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> UNetModel:
        """Train the U-Net and return a serialisable :class:`UNetModel`.

        Parameters
        ----------
        ok_dir:
            Directory of defect-free (OK) images.
        ng_dir:
            Directory of defective (NG) images.  Required for supervised mode.
        mask_dir:
            Directory of binary mask annotations corresponding to *ng_dir*.
            Required for supervised mode.
        epochs:
            Number of training epochs.
        lr:
            Learning rate for Adam optimiser.
        batch_size:
            Mini-batch size.
        progress_callback:
            Called with a status dict after each epoch.

        Returns
        -------
        UNetModel
            The trained model ready for inference or persistence.
        """
        ok_dir = Path(ok_dir)

        # Build dataset and loader
        if self.mode == "supervised":
            if ng_dir is None or mask_dir is None:
                raise ValueError(
                    "Supervised mode requires both ng_dir and mask_dir."
                )
            dataset = _SupervisedDataset(
                ok_dir, Path(ng_dir), Path(mask_dir), self.image_size,
            )
        else:
            dataset = _ReconstructionDataset(ok_dir, self.image_size)

        if len(dataset) == 0:
            raise RuntimeError(f"No images found in {ok_dir}")

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(self.device.type == "cuda"),
        )

        logger.info(
            "UNet training: mode=%s, %d images, depth=%d, base_ch=%d, epochs=%d",
            self.mode, len(dataset), self.depth, self.base_channels, epochs,
        )

        # Build network
        net = UNet(
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            base_channels=self.base_channels,
            depth=self.depth,
        ).to(self.device)

        optimiser = torch.optim.Adam(net.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

        # Training loop
        net.train()
        best_loss = float("inf")
        best_state: Dict[str, Any] = {}

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch in loader:
                if self.mode == "supervised":
                    images, masks = batch
                    images = images.to(self.device)
                    masks = masks.to(self.device)
                    pred = net(images)
                    loss = _bce_dice_loss(pred, masks)
                else:
                    images = batch.to(self.device)
                    pred = net(images)
                    # Target: reverse ImageNet normalisation to get [0,1] pixels
                    target = self._to_pixel_space(images)
                    loss = F.mse_loss(pred, target)

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}

            if progress_callback:
                progress_callback({
                    "stage": "training",
                    "epoch": epoch + 1,
                    "total_epochs": epochs,
                    "loss": avg_loss,
                    "best_loss": best_loss,
                    "progress": (epoch + 1) / epochs,
                })

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    "  epoch %d/%d  loss=%.6f  best=%.6f",
                    epoch + 1, epochs, avg_loss, best_loss,
                )

        # Fit threshold on training set
        threshold = self._fit_threshold(net, loader)

        return UNetModel(
            state_dict=best_state,
            mode=self.mode,
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            base_channels=self.base_channels,
            depth=self.depth,
            image_size=self.image_size,
            threshold=threshold,
            config={
                "epochs": epochs,
                "lr": lr,
                "batch_size": batch_size,
                "best_loss": best_loss,
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_pixel_space(images: torch.Tensor) -> torch.Tensor:
        """Reverse ImageNet normalisation so pixel values are in [0, 1].

        The U-Net sigmoid output naturally produces [0,1] values, so
        the reconstruction target must also be in that range.
        """
        mean = torch.tensor(_IMAGENET_MEAN, device=images.device).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, device=images.device).view(1, 3, 1, 1)
        return (images * std + mean).clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # Threshold fitting
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _fit_threshold(
        self,
        net: UNet,
        loader: DataLoader,
        percentile: float = 99.5,
    ) -> float:
        """Compute a threshold on training data scores.

        For supervised mode the threshold is computed on max predicted
        probability per image.  For reconstruction mode it is based on
        mean reconstruction error.
        """
        net.eval()
        scores: List[float] = []

        for batch in loader:
            if self.mode == "supervised":
                images, _masks = batch
                images = images.to(self.device)
                pred = net(images)
                for i in range(pred.size(0)):
                    scores.append(float(pred[i].max().cpu()))
            else:
                images = batch.to(self.device)
                pred = net(images)
                target = self._to_pixel_space(images)
                error = (pred - target).pow(2).mean(dim=(1, 2, 3))
                scores.extend(error.cpu().tolist())

        threshold = float(np.percentile(scores, percentile))
        logger.info(
            "Training threshold fitted: %.6f (percentile=%.1f, n=%d)",
            threshold, percentile, len(scores),
        )
        return threshold


# ======================================================================
# Inference
# ======================================================================

class UNetInference:
    """Score and segment images using a trained U-Net.

    Parameters
    ----------
    model:
        A trained :class:`UNetModel`.
    device:
        ``'auto'``, ``'cuda'``, ``'mps'``, or ``'cpu'``.
    """

    def __init__(self, model: UNetModel, device: str = "auto") -> None:
        self.model = model
        self.device = _resolve_device(device)

        self.net = UNet(
            in_channels=model.in_channels,
            num_classes=model.num_classes,
            base_channels=model.base_channels,
            depth=model.depth,
        )
        self.net.load_state_dict(model.state_dict)
        self.net.to(self.device)
        self.net.eval()

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Convert a BGR or RGB ``(H, W, C)`` uint8 image to a normalised
        ``(1, C, H, W)`` tensor on the target device.
        """
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (self.model.image_size, self.model.image_size))
        t = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        mean = torch.tensor(_IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(_IMAGENET_STD).view(3, 1, 1)
        t = (t - mean) / std
        return t.unsqueeze(0).to(self.device)

    # ------------------------------------------------------------------
    # Core segmentation
    # ------------------------------------------------------------------

    @torch.no_grad()
    @log_operation(logger)
    def segment(self, image: np.ndarray) -> Tuple[float, np.ndarray]:
        """Compute anomaly score and segmentation mask.

        Parameters
        ----------
        image:
            Input image as ``(H, W, C)`` uint8 numpy array (BGR or RGB).

        Returns
        -------
        tuple
            ``(anomaly_score, segmentation_mask)`` where
            *segmentation_mask* is a float32 ``(H, W)`` array in ``[0, 1]``.

            - Supervised: score = max probability in predicted mask.
            - Reconstruction: score = mean reconstruction error.
        """
        orig_h, orig_w = image.shape[:2]
        tensor = self._preprocess(image)
        pred = self.net(tensor)  # (1, C, H, W), values in [0, 1]

        if self.model.mode == "supervised":
            seg_map = pred[0, 0].cpu().numpy()  # (H, W) probability map
            score = float(seg_map.max())
        else:
            # Reconstruction error map
            mean = torch.tensor(_IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
            std = torch.tensor(_IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
            target = (tensor * std + mean).clamp(0.0, 1.0)
            error = (pred - target).pow(2).mean(dim=1)  # (1, H, W)
            seg_map = error[0].cpu().numpy()
            score = float(seg_map.mean())
            # Normalise to [0, 1] for visualisation
            seg_max = seg_map.max()
            if seg_max > 0:
                seg_map = seg_map / seg_max

        # Resize segmentation map back to original resolution
        seg_map = cv2.resize(
            seg_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR,
        )

        return score, seg_map

    # ------------------------------------------------------------------
    # DetectStep-compatible interface
    # ------------------------------------------------------------------

    def detect(
        self,
        image: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """High-level detection returning a result dictionary.

        Parameters
        ----------
        image:
            Input image as ``(H, W, C)`` uint8 numpy array.
        threshold:
            Override the model's fitted threshold.

        Returns
        -------
        dict
            ``{score, is_defective, mask, threshold}``
        """
        thr = threshold if threshold is not None else self.model.threshold
        score, seg_map = self.segment(image)

        # Binary mask via threshold
        binary_mask = (seg_map > 0.5).astype(np.uint8) * 255
        if self.model.mode == "supervised":
            binary_mask = (seg_map > 0.5).astype(np.uint8) * 255
        else:
            binary_mask = (seg_map > 0.3).astype(np.uint8) * 255

        return {
            "score": score,
            "is_defective": score > thr,
            "mask": binary_mask,
            "heatmap": seg_map,
            "threshold": thr,
        }
