"""Configuration module for the DL Anomaly Detection project.

Loads settings from a .env file and exposes them through a typed dataclass.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
from dotenv import load_dotenv

# Load .env from the project root (same directory as this file)
_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(_ENV_PATH, override=False)


def _resolve_path(raw: str) -> Path:
    """Resolve a path relative to the project root when it starts with '.'."""
    p = Path(raw)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent / p
    return p


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in ("true", "1", "yes")


def _select_device(requested: str) -> str:
    """Return 'cuda' only when both requested AND available; else 'cpu'."""
    if requested.strip().lower() == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class Config:
    """Central, immutable-by-convention configuration for the entire project."""

    # --- Image paths ---
    train_image_dir: Path = field(default_factory=lambda: Path(os.getenv("TRAIN_IMAGE_DIR", ".")))
    test_image_dir: Path = field(default_factory=lambda: Path(os.getenv("TEST_IMAGE_DIR", ".")))

    # --- Model persistence ---
    checkpoint_dir: Path = field(default_factory=lambda: _resolve_path(os.getenv("CHECKPOINT_DIR", "./checkpoints")))
    results_dir: Path = field(default_factory=lambda: _resolve_path(os.getenv("RESULTS_DIR", "./results")))

    # --- Preprocessing ---
    image_size: int = field(default_factory=lambda: int(os.getenv("IMAGE_SIZE", "256")))
    grayscale: bool = field(default_factory=lambda: _parse_bool(os.getenv("GRAYSCALE", "false")))

    # --- Architecture ---
    latent_dim: int = field(default_factory=lambda: int(os.getenv("LATENT_DIM", "128")))
    base_channels: int = field(default_factory=lambda: int(os.getenv("BASE_CHANNELS", "32")))
    num_encoder_blocks: int = field(default_factory=lambda: int(os.getenv("NUM_ENCODER_BLOCKS", "4")))

    # --- Training ---
    batch_size: int = field(default_factory=lambda: int(os.getenv("BATCH_SIZE", "16")))
    learning_rate: float = field(default_factory=lambda: float(os.getenv("LEARNING_RATE", "0.001")))
    num_epochs: int = field(default_factory=lambda: int(os.getenv("NUM_EPOCHS", "100")))
    early_stopping_patience: int = field(default_factory=lambda: int(os.getenv("EARLY_STOPPING_PATIENCE", "10")))
    device: str = field(default_factory=lambda: _select_device(os.getenv("DEVICE", "cuda")))

    # --- Anomaly detection ---
    anomaly_threshold_percentile: float = field(
        default_factory=lambda: float(os.getenv("ANOMALY_THRESHOLD_PERCENTILE", "95"))
    )
    ssim_weight: float = field(default_factory=lambda: float(os.getenv("SSIM_WEIGHT", "0.5")))

    # --- Derived (computed in __post_init__) ---
    in_channels: int = field(init=False)

    def __post_init__(self) -> None:
        self.in_channels = 1 if self.grayscale else 3
        # Guarantee output dirs exist
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """Serialise to a plain dict (paths become strings)."""
        return {
            k: str(v) if isinstance(v, Path) else v
            for k, v in self.__dict__.items()
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        """Reconstruct a Config from a dict (e.g. loaded from a checkpoint)."""
        path_fields = {"train_image_dir", "test_image_dir", "checkpoint_dir", "results_dir"}
        kwargs = {}
        for k, v in d.items():
            if k == "in_channels":
                continue  # derived field
            if k in path_fields:
                kwargs[k] = Path(v)
            else:
                kwargs[k] = v
        return cls(**kwargs)
