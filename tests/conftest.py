"""Shared pytest fixtures for the cv-detect test suite.

Provides synthetic images, temporary directories, and minimal Config objects
so that every test is self-contained and requires no external files.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure the project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "dl_anomaly")):
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """Return a fresh temporary directory (alias for tmp_path)."""
    return tmp_path


@pytest.fixture
def sample_image() -> np.ndarray:
    """Create a synthetic 256x256x3 BGR/RGB uint8 image.

    The image has a gradient pattern so it is not uniform, which is useful
    for testing error-map and scoring functions.
    """
    rng = np.random.RandomState(42)
    img = rng.randint(50, 200, size=(256, 256, 3), dtype=np.uint8)
    return img


@pytest.fixture
def sample_grayscale() -> np.ndarray:
    """Create a synthetic 256x256 single-channel uint8 image."""
    rng = np.random.RandomState(42)
    img = rng.randint(50, 200, size=(256, 256), dtype=np.uint8)
    return img


@pytest.fixture
def sample_config(tmp_path: Path):
    """Create a minimal Config object pointing to temporary directories.

    This avoids writing into the real project tree during tests.
    """
    from dl_anomaly.config import Config

    cfg = Config(
        train_image_dir=tmp_path / "train",
        test_image_dir=tmp_path / "test",
        checkpoint_dir=tmp_path / "checkpoints",
        results_dir=tmp_path / "results",
        image_size=64,
        grayscale=False,
        latent_dim=32,
        base_channels=16,
        num_encoder_blocks=2,
        batch_size=4,
        learning_rate=0.001,
        num_epochs=2,
        early_stopping_patience=2,
        device="cpu",
    )
    return cfg
