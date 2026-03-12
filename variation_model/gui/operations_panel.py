"""
gui/operations_panel.py - 操作控制面板（右側下方）

提供影像處理操作的參數控制：
- 前處理操作：灰階、模糊、縮放、旋轉
- 偵測參數：絕對閾值、變異閾值
- 形態學：核心尺寸、最小瑕疵面積
- 套用按鈕
"""

from __future__ import annotations

import logging
import tkinter as tk
from tkinter import ttk
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)


class OperationsPanel(ttk.LabelFrame):
    """操作控制面板。"""

    def __init__(
        self,
        master: tk.Widget,
        on_apply: Optional[Callable[[str, Dict], None]] = None,
        on_param_change: Optional[Callable[[str, float], None]] = None,
        **kwargs,
    ) -> None:
        kwargs.setdefault("text", "  操作參數")
        super().__init__(master, **kwargs)

        self._on_apply = on_apply
        self._on_param_change = on_param_change
        self._debounce_ids: Dict[str, str] = {}

        self._build_ui()

    def _build_ui(self) -> None:
        """建構操作面板 UI。"""
        # 可捲動容器
        canvas = tk.Canvas(self, highlightthickness=0, bg="#2b2b2b")
        scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=canvas.yview)
        scrollable = ttk.Frame(canvas)

        scrollable.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=scrollable, anchor=tk.NW)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        canvas.bind("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

        # ── 前處理操作 ──
        preproc = ttk.LabelFrame(scrollable, text="前處理", padding=6)
        preproc.pack(fill=tk.X, padx=4, pady=3)

        # 高斯模糊核心
        ttk.Label(preproc, text="高斯模糊核心:", font=("", 8)).pack(anchor=tk.W)
        blur_frame = ttk.Frame(preproc)
        blur_frame.pack(fill=tk.X, pady=2)
        self._blur_var = tk.IntVar(value=3)
        self._blur_label = ttk.Label(blur_frame, text="3", width=4, font=("", 8))
        self._blur_label.pack(side=tk.RIGHT)
        self._blur_scale = ttk.Scale(
            blur_frame, from_=1, to=15, orient=tk.HORIZONTAL,
            variable=self._blur_var,
            command=lambda v: self._on_slider_change("blur_kernel", v, self._blur_label, is_int=True, odd_only=True),
        )
        self._blur_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # 套用按鈕
        btn_frame_pre = ttk.Frame(preproc)
        btn_frame_pre.pack(fill=tk.X, pady=2)
        ttk.Button(
            btn_frame_pre, text="灰階轉換",
            command=lambda: self._apply_op("grayscale"),
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(
            btn_frame_pre, text="高斯模糊",
            command=lambda: self._apply_op("gaussian_blur"),
        ).pack(side=tk.LEFT, padx=2)

        btn_frame_pre2 = ttk.Frame(preproc)
        btn_frame_pre2.pack(fill=tk.X, pady=2)
        ttk.Button(
            btn_frame_pre2, text="中值濾波",
            command=lambda: self._apply_op("median_blur"),
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(
            btn_frame_pre2, text="直方圖均衡",
            command=lambda: self._apply_op("histogram_eq"),
        ).pack(side=tk.LEFT, padx=2)

        # ── 偵測參數 ──
        detect = ttk.LabelFrame(scrollable, text="偵測閾值", padding=6)
        detect.pack(fill=tk.X, padx=4, pady=3)

        # 絕對閾值
        ttk.Label(detect, text="絕對閾值:", font=("", 8)).pack(anchor=tk.W)
        abs_frame = ttk.Frame(detect)
        abs_frame.pack(fill=tk.X, pady=2)
        self._abs_thresh_var = tk.IntVar(value=10)
        self._abs_thresh_label = ttk.Label(abs_frame, text="10", width=4, font=("", 8))
        self._abs_thresh_label.pack(side=tk.RIGHT)
        self._abs_thresh_scale = ttk.Scale(
            abs_frame, from_=0, to=50, orient=tk.HORIZONTAL,
            variable=self._abs_thresh_var,
            command=lambda v: self._on_slider_change("abs_threshold", v, self._abs_thresh_label, is_int=True),
        )
        self._abs_thresh_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # 變異閾值
        ttk.Label(detect, text="變異閾值:", font=("", 8)).pack(anchor=tk.W)
        var_frame = ttk.Frame(detect)
        var_frame.pack(fill=tk.X, pady=2)
        self._var_thresh_var = tk.DoubleVar(value=3.0)
        self._var_thresh_label = ttk.Label(var_frame, text="3.0", width=4, font=("", 8))
        self._var_thresh_label.pack(side=tk.RIGHT)
        self._var_thresh_scale = ttk.Scale(
            var_frame, from_=0.5, to=10.0, orient=tk.HORIZONTAL,
            variable=self._var_thresh_var,
            command=lambda v: self._on_slider_change("var_threshold", v, self._var_thresh_label, is_float=True),
        )
        self._var_thresh_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # ── 形態學 ──
        morph = ttk.LabelFrame(scrollable, text="形態學處理", padding=6)
        morph.pack(fill=tk.X, padx=4, pady=3)

        # 核心尺寸
        ttk.Label(morph, text="核心尺寸:", font=("", 8)).pack(anchor=tk.W)
        morph_k_frame = ttk.Frame(morph)
        morph_k_frame.pack(fill=tk.X, pady=2)
        self._morph_kernel_var = tk.IntVar(value=3)
        self._morph_kernel_label = ttk.Label(morph_k_frame, text="3", width=4, font=("", 8))
        self._morph_kernel_label.pack(side=tk.RIGHT)
        ttk.Scale(
            morph_k_frame, from_=1, to=15, orient=tk.HORIZONTAL,
            variable=self._morph_kernel_var,
            command=lambda v: self._on_slider_change("morph_kernel", v, self._morph_kernel_label, is_int=True, odd_only=True),
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # 最小瑕疵面積
        ttk.Label(morph, text="最小瑕疵面積:", font=("", 8)).pack(anchor=tk.W)
        area_frame = ttk.Frame(morph)
        area_frame.pack(fill=tk.X, pady=2)
        self._min_area_var = tk.IntVar(value=50)
        self._min_area_label = ttk.Label(area_frame, text="50", width=5, font=("", 8))
        self._min_area_label.pack(side=tk.RIGHT)
        ttk.Scale(
            area_frame, from_=1, to=500, orient=tk.HORIZONTAL,
            variable=self._min_area_var,
            command=lambda v: self._on_slider_change("min_area", v, self._min_area_label, is_int=True),
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # 形態學操作按鈕
        morph_btn = ttk.Frame(morph)
        morph_btn.pack(fill=tk.X, pady=2)
        ttk.Button(
            morph_btn, text="侵蝕", command=lambda: self._apply_op("erode"),
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(
            morph_btn, text="膨脹", command=lambda: self._apply_op("dilate"),
        ).pack(side=tk.LEFT, padx=2)

        morph_btn2 = ttk.Frame(morph)
        morph_btn2.pack(fill=tk.X, pady=2)
        ttk.Button(
            morph_btn2, text="開運算", command=lambda: self._apply_op("morph_open"),
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(
            morph_btn2, text="閉運算", command=lambda: self._apply_op("morph_close"),
        ).pack(side=tk.LEFT, padx=2)

        # ── 邊緣偵測 ──
        edge = ttk.LabelFrame(scrollable, text="邊緣偵測", padding=6)
        edge.pack(fill=tk.X, padx=4, pady=3)

        edge_btn = ttk.Frame(edge)
        edge_btn.pack(fill=tk.X, pady=2)
        ttk.Button(
            edge_btn, text="Canny", command=lambda: self._apply_op("canny"),
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(
            edge_btn, text="Sobel", command=lambda: self._apply_op("sobel"),
        ).pack(side=tk.LEFT, padx=2)

        # ── 二值化 ──
        thresh = ttk.LabelFrame(scrollable, text="二值化", padding=6)
        thresh.pack(fill=tk.X, padx=4, pady=3)

        thresh_btn = ttk.Frame(thresh)
        thresh_btn.pack(fill=tk.X, pady=2)
        ttk.Button(
            thresh_btn, text="Otsu", command=lambda: self._apply_op("threshold_otsu"),
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(
            thresh_btn, text="自適應", command=lambda: self._apply_op("threshold_adaptive"),
        ).pack(side=tk.LEFT, padx=2)

        # ── 主要套用按鈕 ──
        sep = ttk.Separator(scrollable, orient=tk.HORIZONTAL)
        sep.pack(fill=tk.X, padx=4, pady=6)

        main_btn = ttk.Frame(scrollable)
        main_btn.pack(fill=tk.X, padx=4, pady=3)
        self._apply_inspect_btn = ttk.Button(
            main_btn, text="執行瑕疵檢測",
            command=lambda: self._apply_op("run_inspection"),
        )
        self._apply_inspect_btn.pack(fill=tk.X, pady=2)

    # ================================================================== #
    #  公開 API                                                           #
    # ================================================================== #

    def get_params(self) -> Dict:
        """取得目前所有參數值。"""
        blur_k = int(self._blur_var.get())
        if blur_k % 2 == 0:
            blur_k += 1

        morph_k = int(self._morph_kernel_var.get())
        if morph_k % 2 == 0:
            morph_k += 1

        return {
            "blur_kernel": blur_k,
            "abs_threshold": int(self._abs_thresh_var.get()),
            "var_threshold": round(float(self._var_thresh_var.get()), 1),
            "morph_kernel": morph_k,
            "min_area": int(self._min_area_var.get()),
        }

    def set_params(self, params: Dict) -> None:
        """設定參數值。"""
        if "blur_kernel" in params:
            self._blur_var.set(params["blur_kernel"])
            self._blur_label.configure(text=str(params["blur_kernel"]))
        if "abs_threshold" in params:
            self._abs_thresh_var.set(params["abs_threshold"])
            self._abs_thresh_label.configure(text=str(params["abs_threshold"]))
        if "var_threshold" in params:
            self._var_thresh_var.set(params["var_threshold"])
            self._var_thresh_label.configure(text=f"{params['var_threshold']:.1f}")
        if "morph_kernel" in params:
            self._morph_kernel_var.set(params["morph_kernel"])
            self._morph_kernel_label.configure(text=str(params["morph_kernel"]))
        if "min_area" in params:
            self._min_area_var.set(params["min_area"])
            self._min_area_label.configure(text=str(params["min_area"]))

    # ================================================================== #
    #  內部方法                                                            #
    # ================================================================== #

    def _on_slider_change(
        self,
        param_name: str,
        value: str,
        label: ttk.Label,
        is_int: bool = False,
        is_float: bool = False,
        odd_only: bool = False,
    ) -> None:
        """滑桿數值變更（含防抖）。"""
        val = float(value)
        if is_int:
            val = int(val)
            if odd_only and val % 2 == 0:
                val += 1
            label.configure(text=str(val))
        elif is_float:
            val = round(val, 1)
            label.configure(text=f"{val:.1f}")
        else:
            label.configure(text=f"{val:.2f}")

        # 防抖
        key = f"debounce_{param_name}"
        if key in self._debounce_ids:
            self.after_cancel(self._debounce_ids[key])

        self._debounce_ids[key] = self.after(
            200, lambda: self._fire_param_change(param_name, val),
        )

    def _fire_param_change(self, param_name: str, value: float) -> None:
        """觸發參數變更回呼。"""
        if self._on_param_change:
            self._on_param_change(param_name, value)

    def _apply_op(self, operation: str) -> None:
        """套用操作。"""
        if self._on_apply:
            self._on_apply(operation, self.get_params())
