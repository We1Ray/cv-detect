"""Settings tab for the DL Anomaly Detector GUI.

Exposes architecture, training, device, and anomaly-detection parameters as
editable widgets.  Changes are applied to the shared ``Config`` instance and
can optionally be persisted back to the ``.env`` file.
"""

from __future__ import annotations

import logging
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any, Dict

import torch

from dl_anomaly.config import Config

logger = logging.getLogger(__name__)

import platform as _platform
_SYS = _platform.system()
if _SYS == "Darwin":
    _FONT_FAMILY = "Helvetica Neue"
    _MONO_FAMILY = "Menlo"
elif _SYS == "Linux":
    _FONT_FAMILY = "DejaVu Sans"
    _MONO_FAMILY = "DejaVu Sans Mono"
else:
    _FONT_FAMILY = _FONT_FAMILY
    _MONO_FAMILY = "Consolas"


class SettingsTab(ttk.Frame):
    """Settings panel with Apply / Reset / Save controls."""

    def __init__(self, master: tk.Misc, config: Config, **kwargs) -> None:
        super().__init__(master, **kwargs)
        self.config = config

        # Tk variables bound to widgets (populated in _build_ui)
        self._vars: Dict[str, tk.Variable] = {}

        self._build_ui()
        self._load_from_config()

    # ==================================================================
    # UI
    # ==================================================================

    def _build_ui(self) -> None:
        container = ttk.Frame(self, padding=8)
        container.pack(fill=tk.BOTH, expand=True)

        row = 0

        # --- Architecture ---------------------------------------------
        row = self._section_header(container, "模型架構", row)
        row = self._add_entry(container, "latent_dim", "潛在維度：", row, width=10)
        row = self._add_entry(container, "base_channels", "基礎通道數：", row, width=10)
        row = self._add_entry(container, "num_encoder_blocks", "編碼器層數：", row, width=10)
        row = self._add_entry(container, "image_size", "影像尺寸：", row, width=10)

        # --- Training -------------------------------------------------
        row = self._section_header(container, "訓練參數", row)
        row = self._add_entry(container, "learning_rate", "學習率：", row, width=12)
        row = self._add_entry(container, "batch_size", "批次大小：", row, width=10)
        row = self._add_entry(container, "num_epochs", "訓練輪次：", row, width=10)
        row = self._add_entry(container, "early_stopping_patience", "早停耐心值：", row, width=10)

        # --- Device ---------------------------------------------------
        row = self._section_header(container, "運算裝置", row)
        self._vars["device"] = tk.StringVar()
        ttk.Label(container, text="裝置：").grid(row=row, column=0, sticky=tk.W, pady=2)
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            devices.append("mps")
        combo = ttk.Combobox(container, textvariable=self._vars["device"], values=devices, state="readonly", width=10)
        combo.grid(row=row, column=1, sticky=tk.W, pady=2)
        row += 1

        # --- Grayscale toggle -----------------------------------------
        self._vars["grayscale"] = tk.BooleanVar()
        ttk.Checkbutton(container, text="灰階輸入", variable=self._vars["grayscale"]).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, pady=2
        )
        row += 1

        # --- Anomaly --------------------------------------------------
        row = self._section_header(container, "異常檢測", row)

        # Threshold percentile slider
        self._vars["anomaly_threshold_percentile"] = tk.DoubleVar()
        ttk.Label(container, text="閾值百分位：").grid(row=row, column=0, sticky=tk.W, pady=2)
        pct_frame = ttk.Frame(container)
        pct_frame.grid(row=row, column=1, sticky=tk.W)
        self._pct_slider = tk.Scale(
            pct_frame, from_=80.0, to=99.9, resolution=0.1, orient=tk.HORIZONTAL,
            variable=self._vars["anomaly_threshold_percentile"], length=200,
        )
        self._pct_slider.pack(side=tk.LEFT)
        row += 1

        # SSIM weight slider
        self._vars["ssim_weight"] = tk.DoubleVar()
        ttk.Label(container, text="SSIM 權重：").grid(row=row, column=0, sticky=tk.W, pady=2)
        sw_frame = ttk.Frame(container)
        sw_frame.grid(row=row, column=1, sticky=tk.W)
        self._ssim_slider = tk.Scale(
            sw_frame, from_=0.0, to=1.0, resolution=0.05, orient=tk.HORIZONTAL,
            variable=self._vars["ssim_weight"], length=200,
        )
        self._ssim_slider.pack(side=tk.LEFT)
        row += 1

        # --- Buttons --------------------------------------------------
        btn_frame = ttk.Frame(container)
        btn_frame.grid(row=row, column=0, columnspan=2, pady=12)
        ttk.Button(btn_frame, text="套用", command=self._apply).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="重設", command=self._load_from_config).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="儲存至 .env", command=self._save_env).pack(side=tk.LEFT, padx=4)

    # ------------------------------------------------------------------
    # Widget helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _section_header(parent: tk.Widget, text: str, row: int) -> int:
        ttk.Separator(parent, orient=tk.HORIZONTAL).grid(row=row, column=0, columnspan=2, sticky="ew", pady=(10, 2))
        row += 1
        ttk.Label(parent, text=text, font=(_FONT_FAMILY, 10, "bold")).grid(row=row, column=0, columnspan=2, sticky=tk.W)
        row += 1
        return row

    def _add_entry(self, parent: tk.Widget, key: str, label: str, row: int, width: int = 20) -> int:
        self._vars[key] = tk.StringVar()
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(parent, textvariable=self._vars[key], width=width).grid(row=row, column=1, sticky=tk.W, pady=2)
        return row + 1

    # ==================================================================
    # Config <-> widgets
    # ==================================================================

    def _load_from_config(self) -> None:
        """Populate widgets from the current Config."""
        c = self.config
        self._vars["latent_dim"].set(str(c.latent_dim))
        self._vars["base_channels"].set(str(c.base_channels))
        self._vars["num_encoder_blocks"].set(str(c.num_encoder_blocks))
        self._vars["image_size"].set(str(c.image_size))
        self._vars["learning_rate"].set(str(c.learning_rate))
        self._vars["batch_size"].set(str(c.batch_size))
        self._vars["num_epochs"].set(str(c.num_epochs))
        self._vars["early_stopping_patience"].set(str(c.early_stopping_patience))
        self._vars["device"].set(c.device)
        self._vars["grayscale"].set(c.grayscale)
        self._vars["anomaly_threshold_percentile"].set(c.anomaly_threshold_percentile)
        self._vars["ssim_weight"].set(c.ssim_weight)

    def _apply(self) -> None:
        """Write widget values back into the Config object."""
        try:
            self.config.latent_dim = int(self._vars["latent_dim"].get())
            self.config.base_channels = int(self._vars["base_channels"].get())
            self.config.num_encoder_blocks = int(self._vars["num_encoder_blocks"].get())
            self.config.image_size = int(self._vars["image_size"].get())
            self.config.learning_rate = float(self._vars["learning_rate"].get())
            self.config.batch_size = int(self._vars["batch_size"].get())
            self.config.num_epochs = int(self._vars["num_epochs"].get())
            self.config.early_stopping_patience = int(self._vars["early_stopping_patience"].get())
            self.config.device = self._vars["device"].get()
            self.config.grayscale = self._vars["grayscale"].get()
            self.config.in_channels = 1 if self.config.grayscale else 3
            self.config.anomaly_threshold_percentile = float(self._vars["anomaly_threshold_percentile"].get())
            self.config.ssim_weight = float(self._vars["ssim_weight"].get())
            messagebox.showinfo("設定", "設定已成功套用。")
        except (ValueError, TypeError) as exc:
            messagebox.showerror("驗證錯誤", str(exc))

    # ==================================================================
    # .env persistence
    # ==================================================================

    def _save_env(self) -> None:
        """Overwrite the project .env file with the current settings."""
        self._apply()  # ensure config is up-to-date

        env_path = Path(__file__).resolve().parent.parent / ".env"
        lines = [
            "# === Image Paths ===",
            f"TRAIN_IMAGE_DIR={self.config.train_image_dir}",
            f"TEST_IMAGE_DIR={self.config.test_image_dir}",
            "",
            "# === Model Persistence ===",
            f"CHECKPOINT_DIR={self.config.checkpoint_dir}",
            f"RESULTS_DIR={self.config.results_dir}",
            "",
            "# === Preprocessing ===",
            f"IMAGE_SIZE={self.config.image_size}",
            f"GRAYSCALE={'true' if self.config.grayscale else 'false'}",
            "",
            "# === Architecture ===",
            f"LATENT_DIM={self.config.latent_dim}",
            f"BASE_CHANNELS={self.config.base_channels}",
            f"NUM_ENCODER_BLOCKS={self.config.num_encoder_blocks}",
            "",
            "# === Training ===",
            f"BATCH_SIZE={self.config.batch_size}",
            f"LEARNING_RATE={self.config.learning_rate}",
            f"NUM_EPOCHS={self.config.num_epochs}",
            f"EARLY_STOPPING_PATIENCE={self.config.early_stopping_patience}",
            f"DEVICE={self.config.device}",
            "",
            "# === Anomaly Detection ===",
            f"ANOMALY_THRESHOLD_PERCENTILE={self.config.anomaly_threshold_percentile}",
            f"SSIM_WEIGHT={self.config.ssim_weight}",
        ]
        try:
            env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            messagebox.showinfo("已儲存", f"設定已寫入：\n{env_path}")
        except OSError as exc:
            messagebox.showerror("儲存錯誤", str(exc))
