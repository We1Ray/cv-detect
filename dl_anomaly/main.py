"""Entry point for the DL Anomaly Detector application (HALCON HDevelop-style).

Sets up ``sys.path`` so that the project package is importable regardless of
the working directory, configures logging, loads configuration, and launches
the HALCON-style Tkinter GUI.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path


def _setup_path() -> None:
    """Ensure the project *parent* directory is on ``sys.path`` so that
    ``import dl_anomaly`` works even when this script is invoked directly.
    """
    project_dir = Path(__file__).resolve().parent.parent
    project_str = str(project_dir)
    if project_str not in sys.path:
        sys.path.insert(0, project_str)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-5s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    # Suppress noisy third-party loggers
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def main() -> None:
    _setup_path()
    _setup_logging()

    logger = logging.getLogger(__name__)
    logger.info("Starting DL Anomaly Detector (HALCON HDevelop Style)")

    # Set the working directory to the project root so that relative
    # paths in .env (e.g. .\checkpoints) resolve correctly.
    os.chdir(Path(__file__).resolve().parent)

    from dl_anomaly.config import Config
    from dl_anomaly.gui.halcon_app import HalconApp

    config = Config()
    logger.info("Device: %s", config.device)
    logger.info("Image size: %d | Grayscale: %s", config.image_size, config.grayscale)
    logger.info("Architecture: latent=%d, base_ch=%d, blocks=%d",
                config.latent_dim, config.base_channels, config.num_encoder_blocks)

    app = HalconApp(config)
    app.mainloop()


if __name__ == "__main__":
    main()
