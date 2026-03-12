"""Matplotlib figures for monitoring the training process.

Every public function returns a ``matplotlib.figure.Figure`` that can be
displayed in the GUI (via ``FigureCanvasTkAgg``) or saved to disk.
"""

from __future__ import annotations

from typing import List, Optional

import matplotlib
import numpy as np

matplotlib.use("Agg")  # non-interactive backend -- safe for threading
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plot_loss_curve(
    train_losses: List[float],
    val_losses: List[float],
    title: str = "Training & Validation Loss",
) -> Figure:
    """Line chart of training and validation loss per epoch."""
    fig, ax = plt.subplots(figsize=(7, 4), dpi=100)
    epochs = list(range(1, len(train_losses) + 1))

    ax.plot(epochs, train_losses, label="Train", linewidth=1.5)
    ax.plot(epochs, val_losses, label="Validation", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_reconstruction_samples(
    originals: List[np.ndarray],
    reconstructions: List[np.ndarray],
    n: int = 8,
) -> Figure:
    """Two-row grid: originals on top, reconstructions on the bottom.

    Parameters
    ----------
    originals, reconstructions:
        Lists of uint8 ``(H, W)`` or ``(H, W, 3)`` images.
    n : Maximum number of samples to show.
    """
    n = min(n, len(originals), len(reconstructions))
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4.5), dpi=100)

    if n == 1:
        axes = axes.reshape(2, 1)

    for i in range(n):
        for row, imgs, label in [
            (0, originals, "Original"),
            (1, reconstructions, "Reconstruction"),
        ]:
            ax = axes[row, i]
            img = imgs[i]
            cmap = "gray" if img.ndim == 2 else None
            ax.imshow(img, cmap=cmap)
            ax.axis("off")
            if i == 0:
                ax.set_ylabel(label, fontsize=10)

    fig.suptitle("Reconstruction Samples", fontsize=12)
    fig.tight_layout()
    return fig


def plot_error_distribution(
    errors: List[float],
    threshold: Optional[float] = None,
    title: str = "Anomaly Score Distribution",
) -> Figure:
    """Histogram of image-level anomaly scores with an optional threshold line."""
    fig, ax = plt.subplots(figsize=(7, 4), dpi=100)
    ax.hist(errors, bins=50, alpha=0.7, color="steelblue", edgecolor="white")

    if threshold is not None:
        ax.axvline(threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold = {threshold:.4f}")
        ax.legend()

    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig
