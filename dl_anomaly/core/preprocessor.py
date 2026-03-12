"""Image preprocessing utilities for the anomaly detection pipeline.

Provides torchvision-compatible transforms for training (with augmentation)
and inference, plus an inverse-normalisation helper for visualisation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T


class ImagePreprocessor:
    """Build and apply image transforms that are consistent across the project."""

    # ImageNet statistics -- sensible defaults for pretrained back-bones and
    # general-purpose normalisation even when we train from scratch.
    MEAN_RGB = [0.485, 0.456, 0.406]
    STD_RGB = [0.229, 0.224, 0.225]
    MEAN_GRAY = [0.449]
    STD_GRAY = [0.226]

    def __init__(self, image_size: int = 256, grayscale: bool = False) -> None:
        self.image_size = image_size
        self.grayscale = grayscale
        self.mean = self.MEAN_GRAY if grayscale else self.MEAN_RGB
        self.std = self.STD_GRAY if grayscale else self.STD_RGB

        # Cached transforms (avoid re-building per call)
        self._train_transform = None
        self._eval_transform = None

        # Cached tensors for inverse_normalize
        self._inv_mean = torch.tensor(self.mean, dtype=torch.float32).view(-1, 1, 1)
        self._inv_std = torch.tensor(self.std, dtype=torch.float32).view(-1, 1, 1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_transforms(self, augment: bool = False) -> T.Compose:
        """Return a torchvision ``Compose`` pipeline.

        Transforms are cached after the first call to avoid rebuilding
        the pipeline repeatedly.

        Parameters
        ----------
        augment:
            When *True*, lightweight augmentations (flip, rotation, jitter) are
            prepended so the pipeline is suitable for training.
        """
        if augment:
            if self._train_transform is not None:
                return self._train_transform
        else:
            if self._eval_transform is not None:
                return self._eval_transform

        ops: list = []

        if self.grayscale:
            ops.append(T.Grayscale(num_output_channels=1))

        ops.append(T.Resize((self.image_size, self.image_size), interpolation=T.InterpolationMode.BILINEAR))

        if augment:
            ops.extend(
                [
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomRotation(degrees=5),
                    T.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.02, hue=0.01),
                ]
            )

        ops.append(T.ToTensor())  # [0, 1]
        ops.append(T.Normalize(mean=self.mean, std=self.std))

        composed = T.Compose(ops)
        if augment:
            self._train_transform = composed
        else:
            self._eval_transform = composed
        return composed

    def load_and_preprocess(self, path: Union[str, Path]) -> torch.Tensor:
        """Load a single image from *path* and return a normalised tensor ``(C, H, W)``."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        img = Image.open(path).convert("L" if self.grayscale else "RGB")
        transform = self.get_transforms(augment=False)
        return transform(img)

    def inverse_normalize(self, tensor: torch.Tensor) -> np.ndarray:
        """Undo normalisation and convert a ``(C, H, W)`` tensor to a uint8 ``(H, W, C)`` ndarray.

        Works on both GPU and CPU tensors.  Returned array is always on CPU.
        """
        t = tensor.detach().cpu().float()

        t = t * self._inv_std + self._inv_mean
        t = t.clamp(0.0, 1.0)

        # (C, H, W) -> (H, W, C) -> uint8
        arr = (t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)

        # Squeeze single-channel to (H, W) for convenience
        if arr.shape[2] == 1:
            arr = arr.squeeze(axis=2)

        return arr
