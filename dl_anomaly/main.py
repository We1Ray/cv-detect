"""Entry point for the DL Anomaly Detector application (Industrial Vision-style).

Sets up ``sys.path`` so that the project package is importable regardless of
the working directory, configures logging, loads configuration, and launches
the Industrial Vision-style Tkinter GUI.
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


def _show_splash() -> "tk.Tk":
    """顯示啟動畫面，讓使用者知道程式正在載入。"""
    import tkinter as tk

    splash = tk.Tk()
    splash.title("DL 異常偵測器")
    splash.overrideredirect(True)

    w, h = 420, 160
    splash.update_idletasks()
    x = (splash.winfo_screenwidth() - w) // 2
    y = (splash.winfo_screenheight() - h) // 2
    splash.geometry(f"{w}x{h}+{x}+{y}")

    splash.configure(bg="#2b2b2b")
    tk.Label(
        splash, text="DL 異常偵測器",
        font=("Helvetica", 18, "bold"), fg="#e0e0e0", bg="#2b2b2b",
    ).pack(pady=(30, 10))
    tk.Label(
        splash, text="正在載入模組，請稍候…",
        font=("Helvetica", 12), fg="#999999", bg="#2b2b2b",
    ).pack()

    splash.update()
    return splash


def main() -> None:
    _setup_path()
    _setup_logging()

    logger = logging.getLogger(__name__)
    logger.info("Starting DL Anomaly Detector (Industrial Vision Style)")

    # Set the working directory to the project root so that relative
    # paths in .env (e.g. .\checkpoints) resolve correctly.
    os.chdir(Path(__file__).resolve().parent)

    # 顯示啟動畫面（純 tkinter，無重量級 import）
    splash = _show_splash()

    from dl_anomaly.config import Config
    from dl_anomaly.gui.inspector_app import InspectorApp

    config = Config()
    logger.info("Device: %s", config.device)
    logger.info("Image size: %d | Grayscale: %s", config.image_size, config.grayscale)
    logger.info("Architecture: latent=%d, base_ch=%d, blocks=%d",
                config.latent_dim, config.base_channels, config.num_encoder_blocks)

    # 關閉啟動畫面，啟動主視窗
    splash.destroy()

    app = InspectorApp(config)
    app.mainloop()


if __name__ == "__main__":
    main()
