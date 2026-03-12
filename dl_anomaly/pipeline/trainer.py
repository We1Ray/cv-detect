"""Training pipeline for the anomaly-detection autoencoder.

Orchestrates dataset creation, model building, the training loop with early
stopping, checkpointing, and post-training threshold computation.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split

from dl_anomaly.config import Config
from dl_anomaly.core.anomaly_scorer import AnomalyScorer
from dl_anomaly.core.autoencoder import AnomalyAutoencoder
from dl_anomaly.core.dataset import DefectFreeDataset
from dl_anomaly.core.preprocessor import ImagePreprocessor

logger = logging.getLogger(__name__)


# ======================================================================
# Differentiable SSIM loss (for training -- avoids skimage in the
# autograd graph).
# ======================================================================

def _gaussian_kernel_1d(size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def _gaussian_kernel_2d(size: int = 11, sigma: float = 1.5, channels: int = 3) -> torch.Tensor:
    k1d = _gaussian_kernel_1d(size, sigma)
    k2d = k1d.unsqueeze(1) @ k1d.unsqueeze(0)  # outer product
    kernel = k2d.expand(channels, 1, size, size).contiguous()
    return kernel


_ssim_kernel_cache: Dict[Tuple[int, int, str, Any], torch.Tensor] = {}


def ssim_loss(x: torch.Tensor, y: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Differentiable 1 - SSIM loss averaged over the batch.

    Parameters
    ----------
    x, y : (B, C, H, W) tensors in the same normalisation space.
    """
    C = x.size(1)
    cache_key = (window_size, C, str(x.device), x.dtype)
    if cache_key not in _ssim_kernel_cache:
        _ssim_kernel_cache[cache_key] = _gaussian_kernel_2d(window_size, 1.5, C).to(x.device, x.dtype)
    kernel = _ssim_kernel_cache[cache_key]
    pad = window_size // 2

    mu_x = F.conv2d(x, kernel, padding=pad, groups=C)
    mu_y = F.conv2d(y, kernel, padding=pad, groups=C)

    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(x * x, kernel, padding=pad, groups=C) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, kernel, padding=pad, groups=C) - mu_y_sq
    sigma_xy = F.conv2d(x * y, kernel, padding=pad, groups=C) - mu_xy

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
        (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
    )

    return 1.0 - ssim_map.mean()


# ======================================================================
# Training pipeline
# ======================================================================

class TrainingPipeline:
    """End-to-end training pipeline.

    Parameters
    ----------
    config : Config
        Project-wide configuration dataclass.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.preprocessor = ImagePreprocessor(config.image_size, config.grayscale)
        self.scorer = AnomalyScorer(device=config.device)
        self.device = torch.device(config.device)

        # Will be initialised during run()
        self.model: Optional[AnomalyAutoencoder] = None
        self.optimizer: Optional[AdamW] = None
        self.scheduler: Optional[CosineAnnealingLR] = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None

        # Stop flag for GUI integration
        self._stop_requested = False

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def request_stop(self) -> None:
        """Signal the training loop to stop after the current epoch."""
        self._stop_requested = True

    def build_model(self) -> AnomalyAutoencoder:
        model = AnomalyAutoencoder(
            in_channels=self.config.in_channels,
            latent_dim=self.config.latent_dim,
            base_channels=self.config.base_channels,
            num_blocks=self.config.num_encoder_blocks,
            image_size=self.config.image_size,
        )
        return model

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def compute_loss(self, original: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
        """Combined MSE + (1-SSIM) loss."""
        mse = F.mse_loss(reconstruction, original)
        s_loss = ssim_loss(original, reconstruction)
        weight = self.config.ssim_weight
        return (1.0 - weight) * mse + weight * s_loss

    # ------------------------------------------------------------------
    # Single-epoch helpers
    # ------------------------------------------------------------------

    def train_epoch(self, epoch: int) -> float:
        """Run one training epoch and return the average loss."""
        assert self.model is not None and self.train_loader is not None
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for images, _ in self.train_loader:
            images = images.to(self.device, non_blocking=True)
            recon = self.model(images)
            loss = self.compute_loss(images, recon)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def validate(self) -> float:
        """Compute the average loss on the validation split."""
        assert self.model is not None and self.val_loader is not None
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for images, _ in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            recon = self.model(images)
            loss = self.compute_loss(images, recon)
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    # ------------------------------------------------------------------
    # Threshold from training data
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_training_threshold(self) -> float:
        """Compute anomaly threshold from reconstruction errors of the training set.

        Uses batch GPU MSE for speed when SSIM weight < 1.0, falling back to
        per-image SSIM only when needed.
        """
        assert self.model is not None and self.train_loader is not None
        self.model.eval()
        scores: List[float] = []

        ssim_w = self.config.ssim_weight

        for images, _ in self.train_loader:
            images = images.to(self.device, non_blocking=True)
            recon = self.model(images)

            # Fast batch MSE on GPU: (B, C, H, W) -> (B,)
            mse_per_image = ((images - recon) ** 2).mean(dim=(1, 2, 3))

            if ssim_w <= 0.0:
                # Pure MSE — no per-sample numpy conversion needed
                scores.extend(mse_per_image.cpu().tolist())
            else:
                # Need per-sample SSIM (GPU-accelerated via scorer)
                for i in range(images.size(0)):
                    orig_np = self.preprocessor.inverse_normalize(images[i])
                    recon_np = self.preprocessor.inverse_normalize(recon[i])
                    ssim_map = self.scorer.compute_ssim_map(orig_np, recon_np)
                    mse_val = float(mse_per_image[i].item())
                    ssim_val = float(np.mean(ssim_map))
                    scores.append((1.0 - ssim_w) * mse_val + ssim_w * ssim_val)

        threshold = self.scorer.fit_threshold(scores, self.config.anomaly_threshold_percentile)
        return threshold

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Execute the full training procedure.

        Parameters
        ----------
        progress_callback:
            Called after every epoch with a dict containing at least
            ``epoch``, ``train_loss``, ``val_loss``, ``best_loss``,
            ``lr``.  Useful for GUI progress updates.

        Returns
        -------
        dict
            Training history and paths to saved artefacts.
        """
        self._stop_requested = False
        cfg = self.config

        # 1. Dataset ---------------------------------------------------
        train_transform = self.preprocessor.get_transforms(augment=True)
        val_transform = self.preprocessor.get_transforms(augment=False)

        full_dataset = DefectFreeDataset(cfg.train_image_dir, transform=train_transform, grayscale=cfg.grayscale)
        if len(full_dataset) == 0:
            raise RuntimeError(f"No images found in {cfg.train_image_dir}")

        val_size = max(1, int(len(full_dataset) * 0.1))
        train_size = len(full_dataset) - val_size
        train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

        # The validation split should not use augmentation.  We swap the
        # underlying dataset's transform for validation items.  Since
        # random_split just stores indices, we create a thin wrapper.
        class _TransformOverride(torch.utils.data.Dataset):
            def __init__(self, subset, transform):
                self.subset = subset
                self.transform = transform

            def __len__(self):
                return len(self.subset)

            def __getitem__(self, idx):
                # Temporarily swap transform
                orig_transform = self.subset.dataset.transform
                self.subset.dataset.transform = self.transform
                item = self.subset[idx]
                self.subset.dataset.transform = orig_transform
                return item

        val_wrapped = _TransformOverride(val_ds, val_transform)

        # Use workers for data loading when CUDA is available (significant
        # speed-up by overlapping CPU I/O with GPU compute).  On Windows the
        # workers must be spawned inside ``if __name__ == "__main__":`` guards
        # which the app already provides via main.py / START.bat.
        import sys
        num_workers = 2 if cfg.device == "cuda" and sys.platform != "darwin" else 0
        loader_kwargs = dict(
            batch_size=cfg.batch_size,
            num_workers=num_workers,
            pin_memory=(cfg.device == "cuda"),
            persistent_workers=(num_workers > 0),
        )
        self.train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
        self.val_loader = DataLoader(val_wrapped, shuffle=False, **loader_kwargs)

        logger.info("Train: %d | Val: %d", train_size, val_size)

        # 2. Model / optimiser ----------------------------------------
        self.model = self.build_model().to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=cfg.learning_rate, weight_decay=1e-5)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=cfg.num_epochs, eta_min=1e-6)

        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info("Model parameters: %s", f"{total_params:,}")

        # 3. Training loop --------------------------------------------
        best_val_loss = float("inf")
        patience_counter = 0
        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "lr": []}

        for epoch in range(1, cfg.num_epochs + 1):
            if self._stop_requested:
                logger.info("Training stopped by user at epoch %d", epoch)
                break

            t0 = time.time()
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            self.scheduler.step()
            elapsed = time.time() - t0

            current_lr = self.scheduler.get_last_lr()[0]
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["lr"].append(current_lr)

            logger.info(
                "Epoch %03d/%03d  train=%.6f  val=%.6f  lr=%.2e  (%.1fs)",
                epoch,
                cfg.num_epochs,
                train_loss,
                val_loss,
                current_lr,
                elapsed,
            )

            # Early stopping / checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_ckpt = cfg.checkpoint_dir / "best_model.pt"
                self.save_checkpoint(best_ckpt, epoch, val_loss)
            else:
                patience_counter += 1

            if progress_callback is not None:
                progress_callback(
                    {
                        "epoch": epoch,
                        "total_epochs": cfg.num_epochs,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "best_loss": best_val_loss,
                        "lr": current_lr,
                        "elapsed": elapsed,
                    }
                )

            if patience_counter >= cfg.early_stopping_patience:
                logger.info("Early stopping triggered at epoch %d", epoch)
                break

        # 4. Load best & compute threshold ----------------------------
        best_ckpt = cfg.checkpoint_dir / "best_model.pt"
        if best_ckpt.exists():
            state = torch.load(best_ckpt, map_location=self.device, weights_only=False)
            self.model.load_state_dict(state["model_state_dict"])
            logger.info("Loaded best checkpoint from epoch %d", state["epoch"])

        threshold = self.compute_training_threshold()

        # 5. Save final model (including config + threshold) ----------
        final_path = cfg.checkpoint_dir / "final_model.pt"
        self.save_checkpoint(final_path, epoch=state.get("epoch", 0) if best_ckpt.exists() else 0, loss=best_val_loss)
        # Append threshold + scorer to final checkpoint
        final_state = torch.load(final_path, map_location="cpu", weights_only=False)
        final_state["threshold"] = threshold
        torch.save(final_state, final_path)
        logger.info("Final model saved to %s (threshold=%.6f)", final_path, threshold)

        return {
            "history": history,
            "best_val_loss": best_val_loss,
            "threshold": threshold,
            "checkpoint_path": str(final_path),
        }

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: Path, epoch: int, loss: float) -> None:
        assert self.model is not None
        torch.save(
            {
                "epoch": epoch,
                "loss": loss,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
                "config": self.config.to_dict(),
            },
            path,
        )
        logger.debug("Checkpoint saved: %s", path)

    @staticmethod
    def load_checkpoint(
        path: Path,
        device: str = "cpu",
    ) -> Tuple[AnomalyAutoencoder, Config, Dict[str, Any]]:
        """Restore model and config from a saved checkpoint.

        Returns ``(model, config, full_state_dict)``.
        """
        state = torch.load(path, map_location=device, weights_only=False)
        cfg = Config.from_dict(state["config"])

        model = AnomalyAutoencoder(
            in_channels=cfg.in_channels,
            latent_dim=cfg.latent_dim,
            base_channels=cfg.base_channels,
            num_blocks=cfg.num_encoder_blocks,
            image_size=cfg.image_size,
        )
        model.load_state_dict(state["model_state_dict"])
        model.to(device)
        model.eval()

        return model, cfg, state
