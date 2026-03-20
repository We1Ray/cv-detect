"""Right panel: operation controls with full parameter freedom.

Provides sliders, entries and buttons for both DL and VM anomaly-detection
parameters plus configurable image-processing operations.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


class OperationsPanel(ttk.LabelFrame):
    """Operation controls panel with DL + VM parameter sections.

    Parameters
    ----------
    on_apply : callable(operation_name: str, result_array: np.ndarray | None)
        Called when the user clicks "Apply" or a quick-action button.
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
        kwargs.setdefault("text", "\u64cd\u4f5c\u63a7\u5236")
        super().__init__(master, **kwargs)
        self._on_apply = on_apply
        self._get_current_image = get_current_image
        self._build_ui()

    def _build_ui(self) -> None:
        # Use a canvas + scrollbar for scrollable content
        canvas = tk.Canvas(self, bg="#2b2b2b", highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=canvas.yview)
        self._scroll_frame = ttk.Frame(canvas)

        self._scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=self._scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Enable mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(-1 * (event.delta // 120), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel, add="+")

        parent = self._scroll_frame

        # ==============================================================
        # DL Anomaly Detection Parameters
        # ==============================================================
        self._add_section_header(parent, "DL \u7570\u5e38\u5075\u6e2c\u53c3\u6578")

        self._threshold_var = tk.DoubleVar(value=95.0)
        self._add_slider(parent, "\u95be\u503c\u767e\u5206\u4f4d:", self._threshold_var, 80.0, 99.9, 0.1)

        self._ssim_var = tk.DoubleVar(value=0.5)
        self._add_slider(parent, "SSIM \u6b0a\u91cd:", self._ssim_var, 0.0, 1.0, 0.05)

        self._sigma_var = tk.DoubleVar(value=4.0)
        self._add_slider(parent, "\u5e73\u6ed1 Sigma:", self._sigma_var, 0.5, 10.0, 0.5)

        self._min_area_var = tk.StringVar(value="50")
        self._add_entry(parent, "\u6700\u5c0f\u7f3a\u9677\u9762\u7a4d (px\u00b2):", self._min_area_var)

        # Apply DL params button
        apply_dl = ttk.Button(parent, text="\u5957\u7528 DL \u53c3\u6578", command=self._on_apply_params)
        apply_dl.pack(fill=tk.X, padx=6, pady=(4, 2))

        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=6, pady=6)

        # ==============================================================
        # VM (Variation Model) Parameters
        # ==============================================================
        self._add_section_header(parent, "VM \u7d71\u8a08\u6a21\u578b\u53c3\u6578")

        self._vm_abs_thresh_var = tk.DoubleVar(value=10.0)
        self._add_slider(parent, "\u7d55\u5c0d\u95be\u503c:", self._vm_abs_thresh_var, 0.0, 100.0, 1.0)

        self._vm_var_thresh_var = tk.DoubleVar(value=3.0)
        self._add_slider(parent, "\u8b8a\u7570\u500d\u6578 (\u03c3):", self._vm_var_thresh_var, 0.5, 10.0, 0.1)

        self._vm_blur_kernel_var = tk.IntVar(value=3)
        self._add_slider_int(parent, "\u6a21\u7cca\u6838 (k):", self._vm_blur_kernel_var, 1, 15, 2)

        self._vm_morph_kernel_var = tk.IntVar(value=3)
        self._add_slider_int(parent, "\u5f62\u614b\u6838 (k):", self._vm_morph_kernel_var, 1, 15, 2)

        self._vm_min_area_var = tk.StringVar(value="50")
        self._add_entry(parent, "VM \u6700\u5c0f\u7f3a\u9677\u9762\u7a4d:", self._vm_min_area_var)

        self._vm_multiscale_var = tk.BooleanVar(value=True)
        cb = tk.Checkbutton(
            parent, text="\u555f\u7528\u591a\u5c3a\u5ea6\u5075\u6e2c",
            variable=self._vm_multiscale_var,
            bg="#2b2b2b", fg="#e0e0e0", selectcolor="#3c3c3c",
            activebackground="#2b2b2b", activeforeground="#e0e0e0",
            font=("Segoe UI", 8),
        )
        cb.pack(fill=tk.X, padx=6, pady=1)

        self._vm_scale_levels_var = tk.IntVar(value=3)
        self._add_slider_int(parent, "\u91d1\u5b57\u5854\u5c64\u6578:", self._vm_scale_levels_var, 1, 5, 1)

        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=6, pady=6)

        # ==============================================================
        # Image Processing (with configurable parameters)
        # ==============================================================
        self._add_section_header(parent, "\u5f71\u50cf\u8655\u7406")

        # Edge detection parameters
        edge_frame = ttk.LabelFrame(parent, text="\u908a\u7de3\u5075\u6e2c\u53c3\u6578")
        edge_frame.pack(fill=tk.X, padx=6, pady=2)

        self._canny_low_var = tk.DoubleVar(value=50.0)
        self._add_slider(edge_frame, "Canny Low:", self._canny_low_var, 0, 255, 5)

        self._canny_high_var = tk.DoubleVar(value=150.0)
        self._add_slider(edge_frame, "Canny High:", self._canny_high_var, 0, 255, 5)

        # Quick operation buttons
        ops = [
            ("\u7070\u968e\u8f49\u63db", self._op_grayscale),
            ("\u9ad8\u65af\u6a21\u7cca (\u03c3={sigma})", self._op_blur),
            ("\u908a\u7de3\u5075\u6e2c (Canny)", self._op_edge),
            ("\u76f4\u65b9\u5716\u5747\u8861", self._op_histeq),
            ("\u53cd\u8272", self._op_invert),
        ]
        for label_text, cmd in ops:
            btn = ttk.Button(parent, text=label_text, command=cmd)
            btn.pack(fill=tk.X, padx=6, pady=1)

    # ------------------------------------------------------------------
    # Section header
    # ------------------------------------------------------------------

    @staticmethod
    def _add_section_header(parent, text: str) -> None:
        ttk.Label(
            parent, text=text,
            font=("Segoe UI", 9, "bold"),
        ).pack(fill=tk.X, padx=6, pady=(6, 2))

    # ------------------------------------------------------------------
    # Slider helpers
    # ------------------------------------------------------------------

    def _add_slider(
        self, parent, label: str, var: tk.DoubleVar,
        from_: float, to_: float, resolution: float,
    ) -> None:
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, padx=6, pady=1)
        ttk.Label(frame, text=label, font=("Segoe UI", 8)).pack(side=tk.LEFT)
        fmt = ".1f" if resolution >= 0.1 else ".2f"
        val_label = ttk.Label(frame, text=f"{var.get():{fmt}}", font=("Segoe UI", 8), width=6)
        val_label.pack(side=tk.RIGHT)

        slider = tk.Scale(
            parent, from_=from_, to=to_, resolution=resolution,
            orient=tk.HORIZONTAL, variable=var, showvalue=False,
            bg="#2b2b2b", fg="#e0e0e0", troughcolor="#1e1e1e",
            highlightthickness=0, length=200,
            command=lambda v, lbl=val_label, f=fmt: lbl.configure(text=f"{float(v):{f}}"),
        )
        slider.pack(fill=tk.X, padx=6, pady=0)

    def _add_slider_int(
        self, parent, label: str, var: tk.IntVar,
        from_: int, to_: int, resolution: int,
    ) -> None:
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, padx=6, pady=1)
        ttk.Label(frame, text=label, font=("Segoe UI", 8)).pack(side=tk.LEFT)
        val_label = ttk.Label(frame, text=str(var.get()), font=("Segoe UI", 8), width=6)
        val_label.pack(side=tk.RIGHT)

        slider = tk.Scale(
            parent, from_=from_, to=to_, resolution=resolution,
            orient=tk.HORIZONTAL, variable=var, showvalue=False,
            bg="#2b2b2b", fg="#e0e0e0", troughcolor="#1e1e1e",
            highlightthickness=0, length=200,
            command=lambda v, lbl=val_label: lbl.configure(text=str(int(float(v)))),
        )
        slider.pack(fill=tk.X, padx=6, pady=0)

    @staticmethod
    def _add_entry(parent, label: str, var: tk.StringVar) -> None:
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, padx=6, pady=2)
        ttk.Label(frame, text=label, font=("Segoe UI", 8)).pack(side=tk.LEFT)
        ttk.Entry(frame, textvariable=var, width=8).pack(side=tk.LEFT, padx=4)

    # ------------------------------------------------------------------
    # Public getters — DL
    # ------------------------------------------------------------------

    def get_threshold_percentile(self) -> float:
        return self._threshold_var.get()

    def get_ssim_weight(self) -> float:
        return self._ssim_var.get()

    def get_sigma(self) -> float:
        return self._sigma_var.get()

    def get_min_area(self) -> int:
        try:
            return max(0, int(self._min_area_var.get()))
        except ValueError:
            return 50

    # ------------------------------------------------------------------
    # Public getters — VM
    # ------------------------------------------------------------------

    def get_vm_abs_threshold(self) -> int:
        return int(self._vm_abs_thresh_var.get())

    def get_vm_var_threshold(self) -> float:
        return self._vm_var_thresh_var.get()

    def get_vm_blur_kernel(self) -> int:
        k = self._vm_blur_kernel_var.get()
        return k if k % 2 == 1 else k + 1

    def get_vm_morph_kernel(self) -> int:
        k = self._vm_morph_kernel_var.get()
        return k if k % 2 == 1 else k + 1

    def get_vm_min_area(self) -> int:
        try:
            return max(0, int(self._vm_min_area_var.get()))
        except ValueError:
            return 50

    def get_vm_multiscale(self) -> bool:
        return self._vm_multiscale_var.get()

    def get_vm_scale_levels(self) -> int:
        return self._vm_scale_levels_var.get()

    # ------------------------------------------------------------------
    # Public getters — Edge detection
    # ------------------------------------------------------------------

    def get_canny_low(self) -> float:
        return self._canny_low_var.get()

    def get_canny_high(self) -> float:
        return self._canny_high_var.get()

    # ------------------------------------------------------------------
    # Public setter (used by InspectorApp.__init__)
    # ------------------------------------------------------------------

    def set_params(
        self,
        threshold: Optional[float] = None,
        ssim_weight: Optional[float] = None,
        sigma: Optional[float] = None,
        min_area: Optional[int] = None,
        # VM params
        vm_abs_threshold: Optional[int] = None,
        vm_var_threshold: Optional[float] = None,
        vm_blur_kernel: Optional[int] = None,
        vm_morph_kernel: Optional[int] = None,
        vm_min_area: Optional[int] = None,
    ) -> None:
        if threshold is not None:
            self._threshold_var.set(threshold)
        if ssim_weight is not None:
            self._ssim_var.set(ssim_weight)
        if sigma is not None:
            self._sigma_var.set(sigma)
        if min_area is not None:
            self._min_area_var.set(str(min_area))
        if vm_abs_threshold is not None:
            self._vm_abs_thresh_var.set(float(vm_abs_threshold))
        if vm_var_threshold is not None:
            self._vm_var_thresh_var.set(vm_var_threshold)
        if vm_blur_kernel is not None:
            self._vm_blur_kernel_var.set(vm_blur_kernel)
        if vm_morph_kernel is not None:
            self._vm_morph_kernel_var.set(vm_morph_kernel)
        if vm_min_area is not None:
            self._vm_min_area_var.set(str(vm_min_area))

    # ------------------------------------------------------------------
    # Apply params callback
    # ------------------------------------------------------------------

    def _on_apply_params(self) -> None:
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
                [gaussian_filter(img[:, :, c].astype(np.float64), sigma=sigma)
                 for c in range(img.shape[2])],
                axis=2,
            )
        blurred = np.clip(blurred, 0, 255).astype(np.uint8)
        self._emit(f"\u6a21\u7cca (\u03c3={sigma:.1f})", blurred)

    def _op_edge(self) -> None:
        img = self._get_img()
        if img is None:
            return
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
        low = self._canny_low_var.get()
        high = self._canny_high_var.get()
        edges = cv2.Canny(gray, low, high)
        self._emit(f"\u908a\u7de3 (Canny {int(low)}/{int(high)})", edges)

    def _op_histeq(self) -> None:
        img = self._get_img()
        if img is None:
            return
        if img.ndim == 2:
            result = cv2.equalizeHist(img)
        else:
            ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        self._emit("\u76f4\u65b9\u5716\u5747\u8861", result)

    def _op_invert(self) -> None:
        img = self._get_img()
        if img is None:
            return
        self._emit("\u53cd\u8272", 255 - img)
