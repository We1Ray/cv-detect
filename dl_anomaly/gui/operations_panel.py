"""Right panel (bottom): operation controls.

Provides sliders and buttons for anomaly-detection parameters and
basic image-processing operations. Applying an operation adds a new
step to the pipeline.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


class OperationsPanel(ttk.LabelFrame):
    """Operation controls panel.

    Parameters
    ----------
    on_apply : callable(operation_name: str, result_array: np.ndarray)
        Called when the user clicks "Apply" or one of the quick-action buttons.
        The caller should add the result to the pipeline.
    get_current_image : callable() -> np.ndarray | None
        Returns the numpy array of the currently displayed image.
    """

    def __init__(
        self,
        master: tk.Misc,
        on_apply: Optional[Callable] = None,
        get_current_image: Optional[Callable] = None,
        **kwargs,
    ) -> None:
        kwargs.setdefault("text", "\u64cd\u4f5c\u63a7\u5236")  # "Operation Controls"
        super().__init__(master, **kwargs)
        self._on_apply = on_apply
        self._get_current_image = get_current_image

        self._build_ui()

    def _build_ui(self) -> None:
        # ---- Anomaly Detection Parameters ----
        param_label = ttk.Label(
            self,
            text="\u7570\u5e38\u5075\u6e2c\u53c3\u6578",  # "Anomaly Detection Parameters"
            font=("Segoe UI", 9, "bold"),
        )
        param_label.pack(fill=tk.X, padx=6, pady=(4, 2))

        # Threshold percentile
        self._threshold_var = tk.DoubleVar(value=95.0)
        self._add_slider(
            "\u95be\u503c\u767e\u5206\u4f4d:",  # "Threshold %:"
            self._threshold_var, 80.0, 99.9, 0.1,
        )

        # SSIM Weight
        self._ssim_var = tk.DoubleVar(value=0.5)
        self._add_slider(
            "SSIM \u6b0a\u91cd:",  # "SSIM Weight:"
            self._ssim_var, 0.0, 1.0, 0.05,
        )

        # Gaussian sigma
        self._sigma_var = tk.DoubleVar(value=4.0)
        self._add_slider(
            "\u9ad8\u65af\u6a21\u7cca (Sigma):",  # "Gaussian Sigma:"
            self._sigma_var, 0.5, 10.0, 0.5,
        )

        # Min defect area
        area_frame = ttk.Frame(self)
        area_frame.pack(fill=tk.X, padx=6, pady=2)
        ttk.Label(area_frame, text="\u6700\u5c0f\u7f3a\u9677\u9762\u7a4d:", font=("Segoe UI", 8)).pack(
            side=tk.LEFT
        )  # "Min Defect Area:"
        self._min_area_var = tk.StringVar(value="50")
        ttk.Entry(area_frame, textvariable=self._min_area_var, width=8).pack(
            side=tk.LEFT, padx=4
        )

        # ---- Apply button ----
        apply_frame = ttk.Frame(self)
        apply_frame.pack(fill=tk.X, padx=6, pady=(6, 2))
        self._apply_btn = ttk.Button(
            apply_frame,
            text="\u5957\u7528\u53c3\u6578",  # "Apply Parameters"
            command=self._on_apply_params,
        )
        self._apply_btn.pack(fill=tk.X)

        # ---- Separator ----
        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=6, pady=6)

        # ---- Image Processing ----
        ip_label = ttk.Label(
            self,
            text="\u5f71\u50cf\u8655\u7406",  # "Image Processing"
            font=("Segoe UI", 9, "bold"),
        )
        ip_label.pack(fill=tk.X, padx=6, pady=(0, 2))

        ops = [
            ("\u7070\u968e\u8f49\u63db", self._op_grayscale),     # "Grayscale"
            ("\u9ad8\u65af\u6a21\u7cca", self._op_blur),           # "Gaussian Blur"
            ("\u908a\u7de3\u5075\u6e2c", self._op_edge),           # "Edge Detection"
            ("\u76f4\u65b9\u5716\u5747\u8861", self._op_histeq),   # "Histogram EQ"
            ("\u53cd\u8272", self._op_invert),                      # "Invert"
        ]
        for label_text, cmd in ops:
            btn = ttk.Button(self, text=label_text, command=cmd)
            btn.pack(fill=tk.X, padx=6, pady=1)

    # ------------------------------------------------------------------
    # Slider helper
    # ------------------------------------------------------------------

    def _add_slider(
        self,
        label: str,
        var: tk.DoubleVar,
        from_: float,
        to_: float,
        resolution: float,
    ) -> None:
        frame = ttk.Frame(self)
        frame.pack(fill=tk.X, padx=6, pady=1)
        ttk.Label(frame, text=label, font=("Segoe UI", 8)).pack(side=tk.LEFT)
        val_label = ttk.Label(frame, text=f"{var.get():.1f}", font=("Segoe UI", 8), width=6)
        val_label.pack(side=tk.RIGHT)

        slider = tk.Scale(
            self,
            from_=from_,
            to=to_,
            resolution=resolution,
            orient=tk.HORIZONTAL,
            variable=var,
            showvalue=False,
            bg="#2b2b2b",
            fg="#e0e0e0",
            troughcolor="#1e1e1e",
            highlightthickness=0,
            length=200,
            command=lambda v, lbl=val_label, fmt=resolution: lbl.configure(
                text=f"{float(v):.1f}" if fmt >= 0.1 else f"{float(v):.2f}"
            ),
        )
        slider.pack(fill=tk.X, padx=6, pady=0)

    # ------------------------------------------------------------------
    # Public getters
    # ------------------------------------------------------------------

    def get_threshold_percentile(self) -> float:
        return self._threshold_var.get()

    def get_ssim_weight(self) -> float:
        return self._ssim_var.get()

    def get_sigma(self) -> float:
        return self._sigma_var.get()

    def get_min_area(self) -> int:
        try:
            return int(self._min_area_var.get())
        except ValueError:
            return 50

    def set_params(
        self,
        threshold: Optional[float] = None,
        ssim_weight: Optional[float] = None,
        sigma: Optional[float] = None,
        min_area: Optional[int] = None,
    ) -> None:
        if threshold is not None:
            self._threshold_var.set(threshold)
        if ssim_weight is not None:
            self._ssim_var.set(ssim_weight)
        if sigma is not None:
            self._sigma_var.set(sigma)
        if min_area is not None:
            self._min_area_var.set(str(min_area))

    # ------------------------------------------------------------------
    # Apply params callback (notifies parent)
    # ------------------------------------------------------------------

    def _on_apply_params(self) -> None:
        """Signal the parent that parameters have changed and should be re-applied."""
        if self._on_apply:
            self._on_apply("apply_params", None)

    # ------------------------------------------------------------------
    # Image processing operations
    # ------------------------------------------------------------------

    def _get_img(self) -> Optional[np.ndarray]:
        if self._get_current_image:
            return self._get_current_image()
        return None

    def _emit(self, name: str, result: np.ndarray) -> None:
        if self._on_apply:
            self._on_apply(name, result)

    def _op_grayscale(self) -> None:
        img = self._get_img()
        if img is None:
            return
        if img.ndim == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img if img.ndim == 2 else img[:, :, 0]
        self._emit("\u7070\u968e", gray)

    def _op_blur(self) -> None:
        img = self._get_img()
        if img is None:
            return
        sigma = self._sigma_var.get()
        if img.ndim == 2:
            blurred = gaussian_filter(img.astype(np.float64), sigma=sigma)
        else:
            blurred = np.stack(
                [gaussian_filter(img[:, :, c].astype(np.float64), sigma=sigma) for c in range(img.shape[2])],
                axis=2,
            )
        blurred = np.clip(blurred, 0, 255).astype(np.uint8)
        self._emit("\u6a21\u7cca", blurred)

    def _op_edge(self) -> None:
        img = self._get_img()
        if img is None:
            return
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        edges = cv2.Canny(gray, 50, 150)
        self._emit("\u908a\u7de3", edges)

    def _op_histeq(self) -> None:
        img = self._get_img()
        if img is None:
            return
        if img.ndim == 2:
            result = cv2.equalizeHist(img)
        else:
            # Convert to YCrCb and equalise Y channel
            ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        self._emit("\u76f4\u65b9\u5716\u5747\u8861", result)

    def _op_invert(self) -> None:
        img = self._get_img()
        if img is None:
            return
        result = 255 - img
        self._emit("\u53cd\u8272", result)
