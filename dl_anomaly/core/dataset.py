"""PyTorch Dataset for defect-free (good) training images.

Images are discovered at construction time but loaded lazily so that memory
consumption stays constant regardless of dataset size.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class DefectFreeDataset(Dataset):
    """Dataset of defect-free images for autoencoder training.

    Parameters
    ----------
    root_dir:
        Directory containing only *good* images (may be nested).
    transform:
        A torchvision-compatible transform applied to every PIL image.
    grayscale:
        Open images in grayscale mode when *True*.
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        transform: Optional[Callable] = None,
        grayscale: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.grayscale = grayscale
        self.image_paths: List[Path] = self._discover_images()

        if len(self.image_paths) == 0:
            logger.warning("No images found in %s", self.root_dir)
        else:
            logger.info("Found %d images in %s", len(self.image_paths), self.root_dir)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        path = self.image_paths[index]
        img = Image.open(path).convert("L" if self.grayscale else "RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, str(path)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _discover_images(self) -> List[Path]:
        """Recursively collect all image files under *root_dir*, sorted for reproducibility."""
        if not self.root_dir.exists():
            logger.error("Directory does not exist: %s", self.root_dir)
            return []

        paths: List[Path] = []
        for p in sorted(self.root_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTENSIONS:
                paths.append(p)
        return paths
