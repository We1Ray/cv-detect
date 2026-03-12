"""
gui/properties_panel.py - 影像屬性面板（右側上方）

顯示目前影像的屬性資訊：
- 影像名稱、路徑
- 尺寸 (W x H)
- 資料型別 (uint8 / float32 / float64)
- 通道數
- 最小值 / 最大值 / 平均值
- 小型直方圖預覽
"""

from __future__ import annotations

import logging
import tkinter as tk
from tkinter import ttk
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

logger = logging.getLogger(__name__)


class PropertiesPanel(ttk.LabelFrame):
    """影像屬性顯示面板。"""

    def __init__(self, master: tk.Widget, **kwargs) -> None:
        kwargs.setdefault("text", "  影像屬性")
        super().__init__(master, **kwargs)

        self._build_ui()

    def _build_ui(self) -> None:
        """建構 UI。"""
        # 使用 grid 排列
        pad_opts = {"padx": 6, "pady": 1, "sticky": tk.W}

        row = 0
        ttk.Label(self, text="名稱:", font=("", 8, "bold")).grid(row=row, column=0, **pad_opts)
        self._name_var = tk.StringVar(value="--")
        ttk.Label(self, textvariable=self._name_var, font=("", 8), wraplength=200).grid(
            row=row, column=1, **pad_opts,
        )

        row += 1
        ttk.Label(self, text="尺寸:", font=("", 8, "bold")).grid(row=row, column=0, **pad_opts)
        self._size_var = tk.StringVar(value="--")
        ttk.Label(self, textvariable=self._size_var, font=("", 8)).grid(
            row=row, column=1, **pad_opts,
        )

        row += 1
        ttk.Label(self, text="型別:", font=("", 8, "bold")).grid(row=row, column=0, **pad_opts)
        self._dtype_var = tk.StringVar(value="--")
        ttk.Label(self, textvariable=self._dtype_var, font=("", 8)).grid(
            row=row, column=1, **pad_opts,
        )

        row += 1
        ttk.Label(self, text="通道:", font=("", 8, "bold")).grid(row=row, column=0, **pad_opts)
        self._channels_var = tk.StringVar(value="--")
        ttk.Label(self, textvariable=self._channels_var, font=("", 8)).grid(
            row=row, column=1, **pad_opts,
        )

        row += 1
        ttk.Label(self, text="最小值:", font=("", 8, "bold")).grid(row=row, column=0, **pad_opts)
        self._min_var = tk.StringVar(value="--")
        ttk.Label(self, textvariable=self._min_var, font=("", 8)).grid(
            row=row, column=1, **pad_opts,
        )

        row += 1
        ttk.Label(self, text="最大值:", font=("", 8, "bold")).grid(row=row, column=0, **pad_opts)
        self._max_var = tk.StringVar(value="--")
        ttk.Label(self, textvariable=self._max_var, font=("", 8)).grid(
            row=row, column=1, **pad_opts,
        )

        row += 1
        ttk.Label(self, text="平均值:", font=("", 8, "bold")).grid(row=row, column=0, **pad_opts)
        self._mean_var = tk.StringVar(value="--")
        ttk.Label(self, textvariable=self._mean_var, font=("", 8)).grid(
            row=row, column=1, **pad_opts,
        )

        # 區域資訊
        row += 1
        ttk.Label(self, text="區域:", font=("", 8, "bold")).grid(row=row, column=0, **pad_opts)
        self._region_var = tk.StringVar(value="--")
        ttk.Label(self, textvariable=self._region_var, font=("", 8)).grid(
            row=row, column=1, **pad_opts,
        )

        # 直方圖預覽
        row += 1
        ttk.Label(self, text="直方圖:", font=("", 8, "bold")).grid(
            row=row, column=0, columnspan=2, **pad_opts,
        )

        row += 1
        self._hist_canvas = tk.Canvas(
            self, width=220, height=60, bg="#1e1e1e", highlightthickness=0,
        )
        self._hist_canvas.grid(row=row, column=0, columnspan=2, padx=6, pady=2, sticky=tk.EW)

        self.columnconfigure(1, weight=1)

    # ================================================================== #
    #  公開 API                                                           #
    # ================================================================== #

    def update_properties(
        self,
        image: np.ndarray,
        name: str = "",
        region=None,
    ) -> None:
        """更新所有屬性顯示。

        Args:
            image: 影像陣列。
            name: 影像名稱。
            region: 可選的 Region 物件。
        """
        self._name_var.set(name if name else "--")

        h, w = image.shape[:2]
        channels = image.shape[2] if image.ndim == 3 else 1

        self._size_var.set(f"{w} x {h}")
        self._dtype_var.set(str(image.dtype))
        self._channels_var.set(str(channels))
        self._min_var.set(f"{image.min():.2f}" if image.dtype in (np.float32, np.float64) else str(image.min()))
        self._max_var.set(f"{image.max():.2f}" if image.dtype in (np.float32, np.float64) else str(image.max()))
        self._mean_var.set(f"{image.mean():.2f}")

        # 區域資訊
        if region is not None and hasattr(region, "num_regions"):
            total_area = sum(p.area for p in region.properties) if region.properties else 0
            self._region_var.set(f"{region.num_regions} 個 (面積={total_area})")
        else:
            self._region_var.set("--")

        self._draw_histogram(image)

    def clear_properties(self) -> None:
        """清除所有屬性。"""
        self._name_var.set("--")
        self._size_var.set("--")
        self._dtype_var.set("--")
        self._channels_var.set("--")
        self._min_var.set("--")
        self._max_var.set("--")
        self._mean_var.set("--")
        self._region_var.set("--")
        self._hist_canvas.delete("all")

    # ================================================================== #
    #  內部方法                                                            #
    # ================================================================== #

    def _draw_histogram(self, image: np.ndarray) -> None:
        """繪製小型直方圖。"""
        self._hist_canvas.delete("all")

        canvas_w = 220
        canvas_h = 60

        # 轉為灰階計算直方圖
        if image.ndim == 3:
            gray = cv2.cvtColor(
                image.astype(np.uint8) if image.dtype == np.uint8 else
                ((image - image.min()) / max(image.max() - image.min(), 1) * 255).astype(np.uint8),
                cv2.COLOR_BGR2GRAY if image.shape[2] == 3 else cv2.COLOR_BGRA2GRAY,
            )
        elif image.dtype != np.uint8:
            min_v = image.min()
            max_v = image.max()
            if max_v - min_v > 0:
                gray = ((image - min_v) / (max_v - min_v) * 255).astype(np.uint8)
            else:
                gray = np.zeros_like(image, dtype=np.uint8)
        else:
            gray = image

        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        hist = hist.flatten()

        max_val = hist.max()
        if max_val == 0:
            return

        bin_w = canvas_w / len(hist)

        for i, count in enumerate(hist):
            bar_h = int((count / max_val) * (canvas_h - 4))
            x1 = int(i * bin_w)
            x2 = int((i + 1) * bin_w)
            y1 = canvas_h - 2 - bar_h
            y2 = canvas_h - 2

            # 色彩梯度
            intensity = int((i / len(hist)) * 200 + 55)
            color = f"#{intensity:02x}{intensity:02x}{intensity:02x}"

            self._hist_canvas.create_rectangle(
                x1, y1, x2, y2, fill=color, outline="",
            )
