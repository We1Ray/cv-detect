"""
gui/inspection_tools_dialog.py - Phase 3 inspection tools dialog.

Provides a tabbed dialog combining:
1. FFT frequency domain processing (filtering, spectrum display)
2. Color inspection (sampling, delta-E, uniformity, palette)
3. OCR text recognition (Tesseract / PaddleOCR)
4. Barcode / QR Code reading (OpenCV / pyzbar)

All heavy imports are lazy-loaded inside handlers with try/except
so the dialog opens instantly even when optional dependencies are
missing.
"""
from __future__ import annotations

import logging
import re
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

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
    _FONT_FAMILY = "Segoe UI"
    _MONO_FAMILY = "Consolas"

# --------------------------------------------------------------------------- #
# Theme constants
# --------------------------------------------------------------------------- #
_BG = "#2b2b2b"
_BG_MEDIUM = "#3c3c3c"
_FG = "#e0e0e0"
_FG_DIM = "#cccccc"
_ACCENT = "#0078d4"
_ACTIVE_BG = "#3a3a5c"
_CANVAS_BG = "#1e1e1e"
_PASS_FG = "#88cc88"
_FAIL_FG = "#cc6666"

# Shared button kwargs
_BTN_KW: Dict[str, Any] = dict(
    bg=_BG_MEDIUM,
    fg=_FG,
    activebackground=_ACTIVE_BG,
    activeforeground="#ffffff",
    relief=tk.FLAT,
    padx=10,
    pady=3,
    font=("", 9),
)
_BTN_ACCENT_KW: Dict[str, Any] = dict(
    bg=_ACCENT,
    fg="#ffffff",
    activebackground="#005a9e",
    activeforeground="#ffffff",
    relief=tk.FLAT,
    padx=12,
    pady=4,
    font=("", 10, "bold"),
)
_LABEL_KW: Dict[str, Any] = dict(bg=_BG, fg=_FG, font=("", 9))
_ENTRY_KW: Dict[str, Any] = dict(
    bg=_BG_MEDIUM, fg=_FG, insertbackground=_FG, relief=tk.FLAT,
)
_SPINBOX_KW: Dict[str, Any] = dict(
    bg=_BG_MEDIUM, fg=_FG, buttonbackground=_BG_MEDIUM,
    insertbackground=_FG, relief=tk.FLAT,
)
_SCALE_KW: Dict[str, Any] = dict(
    bg=_BG, fg=_FG, troughcolor=_BG_MEDIUM,
    highlightthickness=0, sliderrelief=tk.FLAT,
)


# =========================================================================== #
#  InspectionToolsDialog                                                       #
# =========================================================================== #


class InspectionToolsDialog(tk.Toplevel):
    """Combined Phase 3 inspection tools dialog.

    Parameters
    ----------
    master : tk.Widget
        Parent widget.
    get_current_image : callable
        ``() -> np.ndarray | None`` returning the current working image.
    add_pipeline_step : callable
        ``(name: str, image: np.ndarray) -> None`` to push a result
        into the parent pipeline.
    set_status : callable
        ``(msg: str) -> None`` to update the parent status bar.
    """

    def __init__(
        self,
        master: tk.Widget,
        get_current_image: Callable[[], Optional[np.ndarray]],
        add_pipeline_step: Callable[[str, np.ndarray], None],
        set_status: Callable[[str], None],
    ) -> None:
        super().__init__(master)
        self.title("檢測工具 - 頻域 / 色彩 / OCR / 條碼")
        self.geometry("900x650")
        self.resizable(True, True)
        self.configure(bg=_BG)

        self.transient(master)
        self.grab_set()

        self._get_current_image = get_current_image
        self._add_pipeline_step = add_pipeline_step
        self._set_status = set_status

        # PhotoImage references (prevent GC)
        self._photo_refs: List[Any] = []

        # ---- State ----------------------------------------------------------
        self._fft_result: Optional[np.ndarray] = None
        self._fft_magnitude: Optional[np.ndarray] = None
        self._color_samples: List[Any] = []
        self._ocr_results: List[Any] = []
        self._barcode_results: List[Any] = []

        # ---- Tkinter variables (Frequency) ----------------------------------
        self._filter_type_var = tk.StringVar(value="低通 (Lowpass)")
        self._filter_method_var = tk.StringVar(value="高斯 (Gaussian)")
        self._cutoff_var = tk.IntVar(value=30)
        self._cutoff_low_var = tk.IntVar(value=10)
        self._cutoff_high_var = tk.IntVar(value=60)
        self._bw_order_var = tk.IntVar(value=2)
        self._notch_auto_var = tk.BooleanVar(value=True)
        self._notch_centers_var = tk.StringVar(value="")

        # ---- Tkinter variables (Color) --------------------------------------
        self._sample_mode_var = tk.StringVar(value="全圖平均")
        self._roi_x_var = tk.StringVar(value="0")
        self._roi_y_var = tk.StringVar(value="0")
        self._roi_w_var = tk.StringVar(value="100")
        self._roi_h_var = tk.StringVar(value="100")
        self._grid_rows_var = tk.IntVar(value=3)
        self._grid_cols_var = tk.IntVar(value=3)
        self._ref_l_var = tk.StringVar(value="50.0")
        self._ref_a_var = tk.StringVar(value="0.0")
        self._ref_b_var = tk.StringVar(value="0.0")
        self._de_method_var = tk.StringVar(value="CIEDE2000")
        self._de_tolerance_var = tk.StringVar(value="3.0")
        self._palette_n_var = tk.IntVar(value=5)

        # ---- Tkinter variables (OCR) ----------------------------------------
        self._ocr_engine_var = tk.StringVar(value="Tesseract")
        self._ocr_lang_var = tk.StringVar(value="eng")
        self._ocr_psm_var = tk.StringVar(value="自動 (3)")
        self._ocr_preprocess_var = tk.StringVar(value="自適應")
        self._ocr_deskew_var = tk.BooleanVar(value=False)
        self._ocr_expected_var = tk.StringVar(value="")

        # ---- Tkinter variables (Barcode) ------------------------------------
        self._bc_decoder_var = tk.StringVar(value="pyzbar")
        self._bc_type_all_var = tk.BooleanVar(value=True)
        self._bc_type_ean13_var = tk.BooleanVar(value=False)
        self._bc_type_code128_var = tk.BooleanVar(value=False)
        self._bc_type_qr_var = tk.BooleanVar(value=False)
        self._bc_type_dm_var = tk.BooleanVar(value=False)
        self._bc_quality_var = tk.BooleanVar(value=False)
        self._bc_expected_var = tk.StringVar(value="")

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._close)

    # =================================================================== #
    #  UI construction                                                      #
    # =================================================================== #

    def _build_ui(self) -> None:
        """Build the four-tab notebook layout."""
        style = ttk.Style(self)
        style.configure("Inspect.TNotebook", background=_BG)
        style.configure(
            "Inspect.TNotebook.Tab",
            background=_BG_MEDIUM, foreground=_FG, padding=[12, 4],
        )
        style.map(
            "Inspect.TNotebook.Tab",
            background=[("selected", _ACTIVE_BG)],
            foreground=[("selected", "#ffffff")],
        )

        self._notebook = ttk.Notebook(self, style="Inspect.TNotebook")
        self._notebook.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # Tab 1 - Frequency Domain
        tab1 = tk.Frame(self._notebook, bg=_BG)
        self._notebook.add(tab1, text=" 頻域處理 ")
        self._build_frequency_tab(tab1)

        # Tab 2 - Color Inspection
        tab2 = tk.Frame(self._notebook, bg=_BG)
        self._notebook.add(tab2, text=" 色彩檢測 ")
        self._build_color_tab(tab2)

        # Tab 3 - OCR
        tab3 = tk.Frame(self._notebook, bg=_BG)
        self._notebook.add(tab3, text=" OCR 文字辨識 ")
        self._build_ocr_tab(tab3)

        # Tab 4 - Barcode / QR
        tab4 = tk.Frame(self._notebook, bg=_BG)
        self._notebook.add(tab4, text=" 條碼/QR Code ")
        self._build_barcode_tab(tab4)

        # Bottom close button
        btn_frame = tk.Frame(self, bg=_BG)
        btn_frame.pack(fill=tk.X, padx=10, pady=(0, 8))
        tk.Button(
            btn_frame, text="關閉", **_BTN_KW, command=self._close,
        ).pack(side=tk.RIGHT)

    # =================================================================== #
    #  Tab 1: Frequency Domain                                              #
    # =================================================================== #

    def _build_frequency_tab(self, parent: tk.Frame) -> None:
        paned = tk.PanedWindow(
            parent, orient=tk.HORIZONTAL, bg=_BG,
            sashwidth=4, sashrelief=tk.FLAT,
        )
        paned.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        left = tk.Frame(paned, bg=_BG)
        right = tk.Frame(paned, bg=_BG)
        paned.add(left, minsize=340)
        paned.add(right, minsize=300)

        # ---- Left column: parameters ------------------------------------
        param_lf = tk.LabelFrame(
            left, text=" 濾波設定 ", bg=_BG, fg=_FG,
            font=("", 10, "bold"), padx=8, pady=6,
        )
        param_lf.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Filter type
        row = tk.Frame(param_lf, bg=_BG)
        row.pack(fill=tk.X, pady=2)
        tk.Label(row, text="濾波類型:", **_LABEL_KW).pack(side=tk.LEFT, padx=(0, 4))
        filter_cb = ttk.Combobox(
            row, textvariable=self._filter_type_var, state="readonly", width=20,
            values=[
                "低通 (Lowpass)", "高通 (Highpass)",
                "帶通 (Bandpass)", "帶阻 (Bandstop)",
                "陷波 (Notch)",
            ],
        )
        filter_cb.pack(side=tk.LEFT)
        filter_cb.bind("<<ComboboxSelected>>", self._on_filter_type_changed)

        # Filter method
        row = tk.Frame(param_lf, bg=_BG)
        row.pack(fill=tk.X, pady=2)
        tk.Label(row, text="濾波器:", **_LABEL_KW).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Combobox(
            row, textvariable=self._filter_method_var, state="readonly", width=20,
            values=["高斯 (Gaussian)", "Butterworth"],
        ).pack(side=tk.LEFT)

        # Dynamic parameters container
        self._freq_params_frame = tk.Frame(param_lf, bg=_BG)
        self._freq_params_frame.pack(fill=tk.X, pady=4)
        self._build_freq_dynamic_params()

        # Butterworth order
        row = tk.Frame(param_lf, bg=_BG)
        row.pack(fill=tk.X, pady=2)
        tk.Label(row, text="Butterworth 階數:", **_LABEL_KW).pack(
            side=tk.LEFT, padx=(0, 4),
        )
        tk.Spinbox(
            row, textvariable=self._bw_order_var,
            from_=1, to=10, increment=1, width=5, **_SPINBOX_KW,
        ).pack(side=tk.LEFT)

        # Action buttons
        btn_frame = tk.Frame(param_lf, bg=_BG)
        btn_frame.pack(fill=tk.X, pady=(8, 2))

        tk.Button(
            btn_frame, text="套用濾波", **_BTN_ACCENT_KW,
            command=self._apply_fft_filter,
        ).pack(side=tk.LEFT, padx=(0, 6))

        tk.Button(
            btn_frame, text="自動去除週期紋理", **_BTN_KW,
            command=self._remove_periodic_pattern,
        ).pack(side=tk.LEFT)

        btn_frame2 = tk.Frame(param_lf, bg=_BG)
        btn_frame2.pack(fill=tk.X, pady=(4, 2))

        tk.Button(
            btn_frame2, text="顯示頻譜", **_BTN_KW,
            command=self._show_spectrum,
        ).pack(side=tk.LEFT, padx=(0, 6))

        tk.Button(
            btn_frame2, text="重建影像", **_BTN_KW,
            command=self._reconstruct_image,
        ).pack(side=tk.LEFT)

        # ---- Right column: preview canvas --------------------------------
        preview_lf = tk.LabelFrame(
            right, text=" 頻譜預覽 ", bg=_BG, fg=_FG,
            font=("", 10, "bold"), padx=4, pady=4,
        )
        preview_lf.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self._freq_canvas = tk.Canvas(
            preview_lf, width=300, height=300, bg=_CANVAS_BG,
            highlightthickness=0,
        )
        self._freq_canvas.pack(expand=True)

    def _build_freq_dynamic_params(self) -> None:
        """Rebuild dynamic parameter widgets based on current filter type."""
        for w in self._freq_params_frame.winfo_children():
            w.destroy()

        ftype = self._filter_type_var.get()

        if ftype in ("低通 (Lowpass)", "高通 (Highpass)"):
            row = tk.Frame(self._freq_params_frame, bg=_BG)
            row.pack(fill=tk.X, pady=2)
            tk.Label(row, text="截止頻率:", **_LABEL_KW).pack(
                side=tk.LEFT, padx=(0, 4),
            )
            tk.Scale(
                row, from_=1, to=200, orient=tk.HORIZONTAL,
                variable=self._cutoff_var, length=180, **_SCALE_KW,
            ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        elif ftype in ("帶通 (Bandpass)", "帶阻 (Bandstop)"):
            row1 = tk.Frame(self._freq_params_frame, bg=_BG)
            row1.pack(fill=tk.X, pady=2)
            tk.Label(row1, text="低截止:", **_LABEL_KW).pack(
                side=tk.LEFT, padx=(0, 4),
            )
            tk.Scale(
                row1, from_=1, to=200, orient=tk.HORIZONTAL,
                variable=self._cutoff_low_var, length=180, **_SCALE_KW,
            ).pack(side=tk.LEFT, fill=tk.X, expand=True)

            row2 = tk.Frame(self._freq_params_frame, bg=_BG)
            row2.pack(fill=tk.X, pady=2)
            tk.Label(row2, text="高截止:", **_LABEL_KW).pack(
                side=tk.LEFT, padx=(0, 4),
            )
            tk.Scale(
                row2, from_=1, to=200, orient=tk.HORIZONTAL,
                variable=self._cutoff_high_var, length=180, **_SCALE_KW,
            ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        elif ftype == "陷波 (Notch)":
            row = tk.Frame(self._freq_params_frame, bg=_BG)
            row.pack(fill=tk.X, pady=2)
            tk.Checkbutton(
                row, text="自動偵測", variable=self._notch_auto_var,
                bg=_BG, fg=_FG, selectcolor=_BG_MEDIUM,
                activebackground=_BG, activeforeground=_FG,
                command=self._toggle_notch_manual,
            ).pack(side=tk.LEFT, padx=(0, 8))

            self._notch_entry_frame = tk.Frame(self._freq_params_frame, bg=_BG)
            self._notch_entry_frame.pack(fill=tk.X, pady=2)
            tk.Label(
                self._notch_entry_frame, text="頻率中心 (u1,v1;u2,v2;...):",
                **_LABEL_KW,
            ).pack(side=tk.LEFT, padx=(0, 4))
            self._notch_entry = tk.Entry(
                self._notch_entry_frame, textvariable=self._notch_centers_var,
                width=24, **_ENTRY_KW,
            )
            self._notch_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self._toggle_notch_manual()

    def _on_filter_type_changed(self, _event: Any = None) -> None:
        self._build_freq_dynamic_params()

    def _toggle_notch_manual(self) -> None:
        if hasattr(self, "_notch_entry_frame"):
            state = tk.NORMAL if not self._notch_auto_var.get() else tk.DISABLED
            for child in self._notch_entry_frame.winfo_children():
                try:
                    child.configure(state=state)
                except tk.TclError:
                    pass

    # =================================================================== #
    #  Tab 2: Color Inspection                                              #
    # =================================================================== #

    def _build_color_tab(self, parent: tk.Frame) -> None:
        paned = tk.PanedWindow(
            parent, orient=tk.HORIZONTAL, bg=_BG,
            sashwidth=4, sashrelief=tk.FLAT,
        )
        paned.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        left = tk.Frame(paned, bg=_BG)
        right = tk.Frame(paned, bg=_BG)
        paned.add(left, minsize=360)
        paned.add(right, minsize=300)

        # ---- Left: sampling + delta-E ------------------------------------
        # Sampling section
        sample_lf = tk.LabelFrame(
            left, text=" 取樣設定 ", bg=_BG, fg=_FG,
            font=("", 10, "bold"), padx=8, pady=6,
        )
        sample_lf.pack(fill=tk.X, padx=4, pady=(4, 2))

        row = tk.Frame(sample_lf, bg=_BG)
        row.pack(fill=tk.X, pady=2)
        tk.Label(row, text="取樣模式:", **_LABEL_KW).pack(side=tk.LEFT, padx=(0, 4))
        mode_cb = ttk.Combobox(
            row, textvariable=self._sample_mode_var, state="readonly", width=14,
            values=["全圖平均", "ROI 取樣", "網格取樣"],
        )
        mode_cb.pack(side=tk.LEFT)
        mode_cb.bind("<<ComboboxSelected>>", self._on_sample_mode_changed)

        # ROI params container
        self._roi_frame = tk.Frame(sample_lf, bg=_BG)
        self._roi_frame.pack(fill=tk.X, pady=2)
        for label_text, var in [
            ("X:", self._roi_x_var), ("Y:", self._roi_y_var),
            ("W:", self._roi_w_var), ("H:", self._roi_h_var),
        ]:
            tk.Label(self._roi_frame, text=label_text, **_LABEL_KW).pack(
                side=tk.LEFT, padx=(4, 1),
            )
            tk.Entry(
                self._roi_frame, textvariable=var, width=5, **_ENTRY_KW,
            ).pack(side=tk.LEFT, padx=(0, 2))

        # Grid params container
        self._grid_frame = tk.Frame(sample_lf, bg=_BG)
        self._grid_frame.pack(fill=tk.X, pady=2)
        tk.Label(self._grid_frame, text="列:", **_LABEL_KW).pack(
            side=tk.LEFT, padx=(4, 1),
        )
        tk.Spinbox(
            self._grid_frame, textvariable=self._grid_rows_var,
            from_=1, to=20, increment=1, width=4, **_SPINBOX_KW,
        ).pack(side=tk.LEFT, padx=(0, 8))
        tk.Label(self._grid_frame, text="行:", **_LABEL_KW).pack(
            side=tk.LEFT, padx=(0, 1),
        )
        tk.Spinbox(
            self._grid_frame, textvariable=self._grid_cols_var,
            from_=1, to=20, increment=1, width=4, **_SPINBOX_KW,
        ).pack(side=tk.LEFT)

        self._on_sample_mode_changed()  # hide/show appropriate frames

        tk.Button(
            sample_lf, text="取樣", **_BTN_ACCENT_KW,
            command=self._color_sample,
        ).pack(anchor=tk.W, pady=(4, 2))

        # Delta-E section
        de_lf = tk.LabelFrame(
            left, text=" 色差計算 ", bg=_BG, fg=_FG,
            font=("", 10, "bold"), padx=8, pady=6,
        )
        de_lf.pack(fill=tk.X, padx=4, pady=2)

        ref_row = tk.Frame(de_lf, bg=_BG)
        ref_row.pack(fill=tk.X, pady=2)
        tk.Label(ref_row, text="參考色", **_LABEL_KW).pack(side=tk.LEFT, padx=(0, 4))
        for label_text, var in [
            ("L*:", self._ref_l_var), ("a*:", self._ref_a_var),
            ("b*:", self._ref_b_var),
        ]:
            tk.Label(ref_row, text=label_text, **_LABEL_KW).pack(
                side=tk.LEFT, padx=(4, 1),
            )
            tk.Entry(
                ref_row, textvariable=var, width=6, **_ENTRY_KW,
            ).pack(side=tk.LEFT, padx=(0, 2))

        tk.Button(
            ref_row, text="從圖片取樣", **_BTN_KW, font=("", 8),
            command=self._sample_reference_color,
        ).pack(side=tk.LEFT, padx=(6, 0))

        method_row = tk.Frame(de_lf, bg=_BG)
        method_row.pack(fill=tk.X, pady=2)
        tk.Label(method_row, text="方法:", **_LABEL_KW).pack(
            side=tk.LEFT, padx=(0, 4),
        )
        ttk.Combobox(
            method_row, textvariable=self._de_method_var, state="readonly",
            width=12, values=["CIE76", "CIEDE2000"],
        ).pack(side=tk.LEFT, padx=(0, 12))
        tk.Label(method_row, text="容差 (Delta-E):", **_LABEL_KW).pack(
            side=tk.LEFT, padx=(0, 4),
        )
        tk.Entry(
            method_row, textvariable=self._de_tolerance_var, width=6, **_ENTRY_KW,
        ).pack(side=tk.LEFT)

        de_btn_row = tk.Frame(de_lf, bg=_BG)
        de_btn_row.pack(fill=tk.X, pady=(4, 2))
        tk.Button(
            de_btn_row, text="計算色差", **_BTN_ACCENT_KW,
            command=self._compute_delta_e,
        ).pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(
            de_btn_row, text="色差熱力圖", **_BTN_KW,
            command=self._delta_e_heatmap,
        ).pack(side=tk.LEFT)

        # Delta-E result display
        self._de_result_var = tk.StringVar(value="--")
        self._de_verdict_var = tk.StringVar(value="")
        res_row = tk.Frame(de_lf, bg=_BG)
        res_row.pack(fill=tk.X, pady=2)
        tk.Label(res_row, text="結果:", **_LABEL_KW).pack(side=tk.LEFT, padx=(0, 4))
        tk.Label(
            res_row, textvariable=self._de_result_var,
            bg=_BG, fg="#88cc88", font=(_MONO_FAMILY, 9),
        ).pack(side=tk.LEFT, padx=(0, 8))
        self._de_verdict_label = tk.Label(
            res_row, textvariable=self._de_verdict_var,
            bg=_BG, fg=_PASS_FG, font=("", 9, "bold"),
        )
        self._de_verdict_label.pack(side=tk.LEFT)

        # ---- Right: classification, uniformity, palette ------------------
        tools_lf = tk.LabelFrame(
            right, text=" 色彩分析工具 ", bg=_BG, fg=_FG,
            font=("", 10, "bold"), padx=8, pady=6,
        )
        tools_lf.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        tk.Button(
            tools_lf, text="色彩分類", **_BTN_KW,
            command=self._classify_color,
        ).pack(fill=tk.X, pady=2)

        tk.Button(
            tools_lf, text="色彩一致性", **_BTN_KW,
            command=self._check_uniformity,
        ).pack(fill=tk.X, pady=2)

        palette_row = tk.Frame(tools_lf, bg=_BG)
        palette_row.pack(fill=tk.X, pady=2)
        tk.Button(
            palette_row, text="提取調色板", **_BTN_KW,
            command=self._extract_palette,
        ).pack(side=tk.LEFT, padx=(0, 6))
        tk.Label(palette_row, text="色數:", **_LABEL_KW).pack(
            side=tk.LEFT, padx=(0, 2),
        )
        tk.Spinbox(
            palette_row, textvariable=self._palette_n_var,
            from_=2, to=16, increment=1, width=4, **_SPINBOX_KW,
        ).pack(side=tk.LEFT)

        # Results display (scrolled text)
        self._color_result_text = tk.Text(
            tools_lf, height=12, width=36,
            bg=_CANVAS_BG, fg=_FG, font=(_MONO_FAMILY, 9),
            insertbackground=_FG, relief=tk.FLAT, state=tk.DISABLED,
        )
        self._color_result_text.pack(fill=tk.BOTH, expand=True, pady=(6, 0))

    def _on_sample_mode_changed(self, _event: Any = None) -> None:
        mode = self._sample_mode_var.get()
        if mode == "ROI 取樣":
            self._roi_frame.pack(fill=tk.X, pady=2)
            self._grid_frame.pack_forget()
        elif mode == "網格取樣":
            self._roi_frame.pack_forget()
            self._grid_frame.pack(fill=tk.X, pady=2)
        else:
            self._roi_frame.pack_forget()
            self._grid_frame.pack_forget()

    # =================================================================== #
    #  Tab 3: OCR                                                           #
    # =================================================================== #

    def _build_ocr_tab(self, parent: tk.Frame) -> None:
        # Settings
        settings_lf = tk.LabelFrame(
            parent, text=" OCR 設定 ", bg=_BG, fg=_FG,
            font=("", 10, "bold"), padx=8, pady=6,
        )
        settings_lf.pack(fill=tk.X, padx=8, pady=(8, 4))

        # Row 1: Engine + Language
        row1 = tk.Frame(settings_lf, bg=_BG)
        row1.pack(fill=tk.X, pady=2)

        tk.Label(row1, text="引擎:", **_LABEL_KW).pack(side=tk.LEFT, padx=(0, 4))
        engine_values = self._detect_ocr_engines()
        engine_cb = ttk.Combobox(
            row1, textvariable=self._ocr_engine_var, state="readonly",
            width=12, values=engine_values,
        )
        engine_cb.pack(side=tk.LEFT, padx=(0, 12))
        engine_cb.bind("<<ComboboxSelected>>", self._on_ocr_engine_changed)

        tk.Label(row1, text="語言:", **_LABEL_KW).pack(side=tk.LEFT, padx=(0, 4))
        self._ocr_lang_cb = ttk.Combobox(
            row1, textvariable=self._ocr_lang_var, state="readonly", width=10,
            values=["eng", "chi_tra", "chi_sim", "jpn"],
        )
        self._ocr_lang_cb.pack(side=tk.LEFT)

        # Row 2: PSM + preprocess
        row2 = tk.Frame(settings_lf, bg=_BG)
        row2.pack(fill=tk.X, pady=2)

        tk.Label(row2, text="PSM 模式:", **_LABEL_KW).pack(
            side=tk.LEFT, padx=(0, 4),
        )
        ttk.Combobox(
            row2, textvariable=self._ocr_psm_var, state="readonly", width=10,
            values=["自動 (3)", "區塊 (6)", "單行 (7)", "單字 (8)"],
        ).pack(side=tk.LEFT, padx=(0, 12))

        tk.Label(row2, text="前處理:", **_LABEL_KW).pack(
            side=tk.LEFT, padx=(0, 4),
        )
        ttk.Combobox(
            row2, textvariable=self._ocr_preprocess_var, state="readonly", width=8,
            values=["自適應", "Otsu", "無"],
        ).pack(side=tk.LEFT, padx=(0, 12))

        tk.Checkbutton(
            row2, text="校正歪斜", variable=self._ocr_deskew_var,
            bg=_BG, fg=_FG, selectcolor=_BG_MEDIUM,
            activebackground=_BG, activeforeground=_FG,
        ).pack(side=tk.LEFT)

        # Buttons
        btn_row = tk.Frame(settings_lf, bg=_BG)
        btn_row.pack(fill=tk.X, pady=(6, 2))
        tk.Button(
            btn_row, text="辨識", **_BTN_ACCENT_KW,
            command=self._run_ocr,
        ).pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(
            btn_row, text="繪製結果", **_BTN_KW,
            command=self._draw_ocr_results,
        ).pack(side=tk.LEFT)

        # Verification
        verify_row = tk.Frame(settings_lf, bg=_BG)
        verify_row.pack(fill=tk.X, pady=2)
        tk.Label(verify_row, text="預期 (regex):", **_LABEL_KW).pack(
            side=tk.LEFT, padx=(0, 4),
        )
        tk.Entry(
            verify_row, textvariable=self._ocr_expected_var, width=20, **_ENTRY_KW,
        ).pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(
            verify_row, text="驗證", **_BTN_KW,
            command=self._verify_ocr,
        ).pack(side=tk.LEFT)

        # Results Treeview
        results_lf = tk.LabelFrame(
            parent, text=" 辨識結果 ", bg=_BG, fg=_FG,
            font=("", 10, "bold"), padx=8, pady=6,
        )
        results_lf.pack(fill=tk.BOTH, expand=True, padx=8, pady=(4, 8))

        cols = ("#", "文字", "信心度", "位置")
        self._ocr_tree = ttk.Treeview(
            results_lf, columns=cols, show="headings", height=10,
        )
        for col in cols:
            anchor = tk.CENTER if col in ("#", "信心度") else tk.W
            width = 40 if col == "#" else (200 if col == "文字" else 80)
            self._ocr_tree.heading(col, text=col)
            self._ocr_tree.column(col, width=width, anchor=anchor)

        ocr_scroll = ttk.Scrollbar(
            results_lf, orient=tk.VERTICAL, command=self._ocr_tree.yview,
        )
        self._ocr_tree.configure(yscrollcommand=ocr_scroll.set)
        self._ocr_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ocr_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    # =================================================================== #
    #  Tab 4: Barcode / QR Code                                             #
    # =================================================================== #

    def _build_barcode_tab(self, parent: tk.Frame) -> None:
        # Settings
        settings_lf = tk.LabelFrame(
            parent, text=" 條碼設定 ", bg=_BG, fg=_FG,
            font=("", 10, "bold"), padx=8, pady=6,
        )
        settings_lf.pack(fill=tk.X, padx=8, pady=(8, 4))

        # Row 1: Decoder
        row1 = tk.Frame(settings_lf, bg=_BG)
        row1.pack(fill=tk.X, pady=2)
        tk.Label(row1, text="解碼器:", **_LABEL_KW).pack(side=tk.LEFT, padx=(0, 4))
        decoder_values = self._detect_barcode_decoders()
        ttk.Combobox(
            row1, textvariable=self._bc_decoder_var, state="readonly",
            width=12, values=decoder_values,
        ).pack(side=tk.LEFT)

        # Row 2: Type checkboxes
        row2 = tk.Frame(settings_lf, bg=_BG)
        row2.pack(fill=tk.X, pady=2)
        tk.Label(row2, text="類型:", **_LABEL_KW).pack(side=tk.LEFT, padx=(0, 4))

        cb_kw = dict(
            bg=_BG, fg=_FG, selectcolor=_BG_MEDIUM,
            activebackground=_BG, activeforeground=_FG,
        )
        tk.Checkbutton(
            row2, text="全部", variable=self._bc_type_all_var, **cb_kw,
            command=self._on_bc_type_all_toggled,
        ).pack(side=tk.LEFT, padx=(0, 4))
        tk.Checkbutton(
            row2, text="EAN-13", variable=self._bc_type_ean13_var, **cb_kw,
        ).pack(side=tk.LEFT, padx=(0, 4))
        tk.Checkbutton(
            row2, text="Code128", variable=self._bc_type_code128_var, **cb_kw,
        ).pack(side=tk.LEFT, padx=(0, 4))
        tk.Checkbutton(
            row2, text="QR", variable=self._bc_type_qr_var, **cb_kw,
        ).pack(side=tk.LEFT, padx=(0, 4))
        tk.Checkbutton(
            row2, text="DataMatrix", variable=self._bc_type_dm_var, **cb_kw,
        ).pack(side=tk.LEFT)

        # Row 3: Quality + buttons
        row3 = tk.Frame(settings_lf, bg=_BG)
        row3.pack(fill=tk.X, pady=(4, 2))

        tk.Checkbutton(
            row3, text="品質評級", variable=self._bc_quality_var, **cb_kw,
        ).pack(side=tk.LEFT, padx=(0, 12))

        tk.Button(
            row3, text="掃描", **_BTN_ACCENT_KW,
            command=self._scan_barcode,
        ).pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(
            row3, text="繪製結果", **_BTN_KW,
            command=self._draw_barcode_results,
        ).pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(
            row3, text="品質剖面", **_BTN_KW,
            command=self._barcode_quality_profile,
        ).pack(side=tk.LEFT)

        # Verification
        verify_row = tk.Frame(settings_lf, bg=_BG)
        verify_row.pack(fill=tk.X, pady=2)
        tk.Label(verify_row, text="預期資料:", **_LABEL_KW).pack(
            side=tk.LEFT, padx=(0, 4),
        )
        tk.Entry(
            verify_row, textvariable=self._bc_expected_var, width=24, **_ENTRY_KW,
        ).pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(
            verify_row, text="驗證", **_BTN_KW,
            command=self._verify_barcode,
        ).pack(side=tk.LEFT)

        # Results Treeview
        results_lf = tk.LabelFrame(
            parent, text=" 掃描結果 ", bg=_BG, fg=_FG,
            font=("", 10, "bold"), padx=8, pady=6,
        )
        results_lf.pack(fill=tk.BOTH, expand=True, padx=8, pady=(4, 8))

        cols = ("#", "資料", "類型", "品質等級", "信心度")
        self._bc_tree = ttk.Treeview(
            results_lf, columns=cols, show="headings", height=8,
        )
        for col in cols:
            anchor = tk.CENTER if col in ("#", "品質等級", "信心度") else tk.W
            width = 40 if col == "#" else (220 if col == "資料" else 90)
            self._bc_tree.heading(col, text=col)
            self._bc_tree.column(col, width=width, anchor=anchor)

        bc_scroll = ttk.Scrollbar(
            results_lf, orient=tk.VERTICAL, command=self._bc_tree.yview,
        )
        self._bc_tree.configure(yscrollcommand=bc_scroll.set)
        self._bc_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        bc_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    # =================================================================== #
    #  Helpers                                                              #
    # =================================================================== #

    def _get_image_or_warn(self) -> Optional[np.ndarray]:
        """Return current image or show a warning."""
        img = self._get_current_image()
        if img is None:
            messagebox.showwarning("無影像", "請先載入影像。", parent=self)
            return None
        return img

    def _set_color_result(self, text: str) -> None:
        """Write text into the color results Text widget."""
        self._color_result_text.configure(state=tk.NORMAL)
        self._color_result_text.delete("1.0", tk.END)
        self._color_result_text.insert(tk.END, text)
        self._color_result_text.configure(state=tk.DISABLED)

    @staticmethod
    def _detect_ocr_engines() -> List[str]:
        """Return list of available OCR engine names."""
        engines: List[str] = []
        try:
            import pytesseract  # noqa: F401
            engines.append("Tesseract")
        except ImportError:
            pass
        try:
            from paddleocr import PaddleOCR  # noqa: F401
            engines.append("PaddleOCR")
        except ImportError:
            pass
        return engines if engines else ["Tesseract"]

    @staticmethod
    def _detect_barcode_decoders() -> List[str]:
        """Return list of available barcode decoder names."""
        decoders: List[str] = []
        try:
            import cv2
            if hasattr(cv2, "barcode"):
                decoders.append("OpenCV")
        except ImportError:
            pass
        try:
            import pyzbar  # noqa: F401
            decoders.append("pyzbar")
        except ImportError:
            pass
        return decoders if decoders else ["pyzbar"]

    def _on_ocr_engine_changed(self, _event: Any = None) -> None:
        """Update language options when OCR engine changes."""
        engine = self._ocr_engine_var.get()
        if engine == "PaddleOCR":
            self._ocr_lang_cb.configure(values=["en", "ch", "japan"])
            self._ocr_lang_var.set("en")
        else:
            self._ocr_lang_cb.configure(
                values=["eng", "chi_tra", "chi_sim", "jpn"],
            )
            self._ocr_lang_var.set("eng")

    def _on_bc_type_all_toggled(self) -> None:
        """When 'all' is checked, uncheck individual types."""
        if self._bc_type_all_var.get():
            self._bc_type_ean13_var.set(False)
            self._bc_type_code128_var.set(False)
            self._bc_type_qr_var.set(False)
            self._bc_type_dm_var.set(False)

    def _update_preview(
        self, canvas: tk.Canvas, image: np.ndarray, max_size: int = 300,
    ) -> None:
        """Render a numpy image onto a tk.Canvas, scaled to fit."""
        try:
            from PIL import Image, ImageTk
        except ImportError:
            logger.warning("Pillow not installed; cannot preview.")
            return

        if image.ndim == 2:
            pil = Image.fromarray(image, mode="L")
        else:
            pil = Image.fromarray(image)

        w, h = pil.size
        scale = min(max_size / w, max_size / h, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)
        pil = pil.resize((new_w, new_h), Image.LANCZOS)

        photo = ImageTk.PhotoImage(pil)
        self._photo_refs.append(photo)

        canvas.delete("all")
        canvas.create_image(
            max_size // 2, max_size // 2, image=photo, anchor=tk.CENTER,
        )

    # =================================================================== #
    #  Frequency handlers                                                   #
    # =================================================================== #

    def _apply_fft_filter(self) -> None:
        img = self._get_image_or_warn()
        if img is None:
            return
        try:
            from dl_anomaly.core.frequency import (
                compute_fft,
                apply_frequency_filter,
                create_gaussian_filter,
                create_butterworth_filter,
                create_bandpass_filter,
                create_bandstop_filter,
                create_notch_filter,
                remove_periodic_pattern,
            )
        except ImportError:
            messagebox.showerror(
                "模組缺失", "無法載入 frequency 模組。", parent=self,
            )
            return

        ftype = self._filter_type_var.get()
        method = self._filter_method_var.get()
        is_gaussian = "Gaussian" in method or "高斯" in method
        order = self._bw_order_var.get()
        cutoff = self._cutoff_var.get()

        try:
            # Compute FFT to determine padded shape for filter creation
            fft_r = compute_fft(img)
            fft_shape = fft_r.magnitude.shape

            if "Lowpass" in ftype or "低通" in ftype:
                if is_gaussian:
                    fmask = create_gaussian_filter(fft_shape, sigma=cutoff, filter_type="lowpass")
                else:
                    fmask = create_butterworth_filter(fft_shape, cutoff=cutoff, order=order, filter_type="lowpass")
            elif "Highpass" in ftype or "高通" in ftype:
                if is_gaussian:
                    fmask = create_gaussian_filter(fft_shape, sigma=cutoff, filter_type="highpass")
                else:
                    fmask = create_butterworth_filter(fft_shape, cutoff=cutoff, order=order, filter_type="highpass")
            elif "Bandpass" in ftype or "帶通" in ftype:
                fmask = create_bandpass_filter(
                    fft_shape,
                    low_cutoff=self._cutoff_low_var.get(),
                    high_cutoff=self._cutoff_high_var.get(),
                    order=order,
                )
            elif "Bandstop" in ftype or "帶阻" in ftype:
                fmask = create_bandstop_filter(
                    fft_shape,
                    low_cutoff=self._cutoff_low_var.get(),
                    high_cutoff=self._cutoff_high_var.get(),
                    order=order,
                )
            elif "Notch" in ftype or "陷波" in ftype:
                if self._notch_auto_var.get():
                    # Use remove_periodic_pattern for auto notch
                    result = remove_periodic_pattern(img)
                    self._fft_result = result
                    self._add_pipeline_step(f"FFT {ftype} (自動)", result)
                    self._set_status(f"已套用 FFT 自動陷波濾波")
                    return
                else:
                    centers_str = self._notch_centers_var.get()
                    centers = []
                    for pair in centers_str.split(";"):
                        parts = pair.strip().split(",")
                        if len(parts) == 2:
                            centers.append(
                                (int(parts[0].strip()), int(parts[1].strip())),
                            )
                    if not centers:
                        messagebox.showwarning("提示", "請輸入陷波中心座標 (格式: u,v;u,v)", parent=self)
                        return
                    fmask = create_notch_filter(fft_shape, centers=centers)
            else:
                messagebox.showwarning("未知濾波類型", ftype, parent=self)
                return

            result = apply_frequency_filter(img, fmask)
            self._fft_result = result
            self._add_pipeline_step(f"FFT {ftype}", result)
            self._set_status(f"已套用 FFT 濾波: {ftype}")
        except Exception as exc:
            logger.exception("FFT filter error")
            messagebox.showerror("FFT 錯誤", str(exc), parent=self)

    def _remove_periodic_pattern(self) -> None:
        img = self._get_image_or_warn()
        if img is None:
            return
        try:
            from dl_anomaly.core.frequency import remove_periodic_pattern
        except ImportError:
            messagebox.showerror(
                "模組缺失", "無法載入 frequency 模組。", parent=self,
            )
            return
        try:
            result = remove_periodic_pattern(img)
            self._add_pipeline_step("去除週期紋理", result)
            self._set_status("已自動去除週期紋理")
        except Exception as exc:
            logger.exception("Remove periodic pattern error")
            messagebox.showerror("錯誤", str(exc), parent=self)

    def _show_spectrum(self) -> None:
        img = self._get_image_or_warn()
        if img is None:
            return
        try:
            from dl_anomaly.core.frequency import compute_fft, draw_spectrum
        except ImportError:
            messagebox.showerror(
                "模組缺失", "無法載入 frequency 模組。", parent=self,
            )
            return
        try:
            fft_r = compute_fft(img)
            mag = draw_spectrum(fft_r.magnitude)
            self._fft_magnitude = mag
            self._update_preview(self._freq_canvas, mag)
            self._set_status("已顯示 FFT 頻譜")
        except Exception as exc:
            logger.exception("FFT spectrum error")
            messagebox.showerror("錯誤", str(exc), parent=self)

    def _reconstruct_image(self) -> None:
        if self._fft_result is None:
            messagebox.showinfo("提示", "請先套用濾波。", parent=self)
            return
        self._add_pipeline_step("FFT 重建", self._fft_result)
        self._set_status("已將 FFT 結果加入管線")

    # =================================================================== #
    #  Color handlers                                                       #
    # =================================================================== #

    def _color_sample(self) -> None:
        img = self._get_image_or_warn()
        if img is None:
            return
        try:
            from dl_anomaly.core.color_inspect import (
                sample_color,
                sample_colors_grid,
            )
        except ImportError:
            messagebox.showerror(
                "模組缺失", "無法載入 color_inspect 模組。", parent=self,
            )
            return

        mode = self._sample_mode_var.get()
        try:
            if mode == "全圖平均":
                samples = sample_color(img)
            elif mode == "ROI 取樣":
                x = int(self._roi_x_var.get())
                y = int(self._roi_y_var.get())
                w = int(self._roi_w_var.get())
                h = int(self._roi_h_var.get())
                samples = sample_color(img, roi=(x, y, w, h))
            elif mode == "網格取樣":
                rows = self._grid_rows_var.get()
                cols = self._grid_cols_var.get()
                samples = sample_colors_grid(img, grid_rows=rows, grid_cols=cols)
            else:
                return

            self._color_samples = samples if isinstance(samples, list) else [samples]
            result_lines = [f"取樣模式: {mode}", ""]
            if isinstance(samples, list):
                for i, s in enumerate(samples):
                    result_lines.append(
                        f"#{i + 1}: L={s.lab[0]:.1f} a={s.lab[1]:.1f} b={s.lab[2]:.1f}  "
                        f"RGB={s.rgb}"
                    )
            else:
                result_lines.append(
                    f"L={samples.lab[0]:.1f} a={samples.lab[1]:.1f} b={samples.lab[2]:.1f}  "
                    f"RGB={samples.rgb}  HSV={samples.hsv}"
                )
            self._set_color_result("\n".join(result_lines))
            self._set_status(f"色彩取樣完成 ({mode})")
        except Exception as exc:
            logger.exception("Color sample error")
            messagebox.showerror("色彩取樣錯誤", str(exc), parent=self)

    def _sample_reference_color(self) -> None:
        img = self._get_image_or_warn()
        if img is None:
            return
        try:
            from dl_anomaly.core.color_inspect import sample_color
        except ImportError:
            messagebox.showerror(
                "模組缺失", "無法載入 color_inspect 模組。", parent=self,
            )
            return
        try:
            sample = sample_color(img)
            L, a, b = sample.lab
            self._ref_l_var.set(f"{L:.2f}")
            self._ref_a_var.set(f"{a:.2f}")
            self._ref_b_var.set(f"{b:.2f}")
            self._set_status(f"已從圖片取樣參考色: L={L:.1f} a={a:.1f} b={b:.1f}")
        except Exception as exc:
            logger.exception("Sample reference error")
            messagebox.showerror("取樣錯誤", str(exc), parent=self)

    def _compute_delta_e(self) -> None:
        img = self._get_image_or_warn()
        if img is None:
            return
        try:
            from dl_anomaly.core.color_inspect import (
                sample_color, compute_delta_e, ColorSample,
            )
        except ImportError:
            messagebox.showerror(
                "模組缺失", "無法載入 color_inspect 模組。", parent=self,
            )
            return
        try:
            ref_L = float(self._ref_l_var.get())
            ref_a = float(self._ref_a_var.get())
            ref_b = float(self._ref_b_var.get())
            method = self._de_method_var.get()
            tolerance = float(self._de_tolerance_var.get())

            # Sample current image
            img_sample = sample_color(img)
            # Build reference sample
            ref_sample = ColorSample(
                lab=(ref_L, ref_a, ref_b), rgb=(0, 0, 0), hsv=(0, 0, 0),
                std=(0.0, 0.0, 0.0), area=0,
            )

            result = compute_delta_e(
                img_sample, ref_sample, method=method, tolerance=tolerance,
            )

            self._de_result_var.set(
                f"Delta-E={result.delta_e:.2f}  "
                f"DL={result.delta_l:.2f}  "
                f"Da={result.delta_a:.2f}  "
                f"Db={result.delta_b:.2f}"
            )

            verdict = "PASS" if result.pass_fail else "FAIL"
            self._de_verdict_var.set(verdict)
            self._de_verdict_label.configure(fg=_PASS_FG if result.pass_fail else _FAIL_FG)
            self._set_status(f"色差計算完成: Delta-E={result.delta_e:.2f} [{verdict}]")
        except Exception as exc:
            logger.exception("Delta-E error")
            messagebox.showerror("色差計算錯誤", str(exc), parent=self)

    def _delta_e_heatmap(self) -> None:
        img = self._get_image_or_warn()
        if img is None:
            return
        try:
            from dl_anomaly.core.color_inspect import (
                compute_delta_e_map, draw_delta_e_map,
            )
        except ImportError:
            messagebox.showerror(
                "模組缺失", "無法載入 color_inspect 模組。", parent=self,
            )
            return
        try:
            ref_L = float(self._ref_l_var.get())
            ref_a = float(self._ref_a_var.get())
            ref_b = float(self._ref_b_var.get())
            de_map = compute_delta_e_map(
                img, reference_color=(ref_L, ref_a, ref_b),
            )
            heatmap = draw_delta_e_map(de_map)
            self._add_pipeline_step("色差熱力圖", heatmap)
            self._set_status("已產生色差熱力圖")
        except Exception as exc:
            logger.exception("Delta-E heatmap error")
            messagebox.showerror("錯誤", str(exc), parent=self)

    def _classify_color(self) -> None:
        img = self._get_image_or_warn()
        if img is None:
            return
        try:
            from dl_anomaly.core.color_inspect import classify_color, sample_color
        except ImportError:
            messagebox.showerror(
                "模組缺失", "無法載入 color_inspect 模組。", parent=self,
            )
            return
        try:
            sample = sample_color(img)
            result = classify_color(sample)
            lines = [
                f"色彩分類結果:",
                f"  名稱: {result.class_name}",
                f"  信心度: {result.confidence:.2%}",
                f"  Lab: ({result.lab[0]:.1f}, {result.lab[1]:.1f}, {result.lab[2]:.1f})",
                f"  Delta-E: {result.delta_e:.2f}",
            ]
            self._set_color_result("\n".join(lines))
            self._set_status(f"色彩分類完成: {result.class_name}")
        except Exception as exc:
            logger.exception("Color classify error")
            messagebox.showerror("錯誤", str(exc), parent=self)

    def _check_uniformity(self) -> None:
        img = self._get_image_or_warn()
        if img is None:
            return
        try:
            from dl_anomaly.core.color_inspect import check_color_uniformity
        except ImportError:
            messagebox.showerror(
                "模組缺失", "無法載入 color_inspect 模組。", parent=self,
            )
            return
        try:
            result = check_color_uniformity(img)
            verdict = "PASS (均勻)" if result["uniform"] else "FAIL (不均勻)"
            lines = [
                f"色彩一致性: {verdict}",
                f"  L* 標準差: {result['std_l']:.2f}",
                f"  a* 標準差: {result['std_a']:.2f}",
                f"  b* 標準差: {result['std_b']:.2f}",
                f"  平均 Lab: ({result['mean_lab'][0]:.1f}, {result['mean_lab'][1]:.1f}, {result['mean_lab'][2]:.1f})",
            ]
            self._set_color_result("\n".join(lines))
            self._set_status(f"色彩一致性: {verdict}")
        except Exception as exc:
            logger.exception("Uniformity check error")
            messagebox.showerror("錯誤", str(exc), parent=self)

    def _extract_palette(self) -> None:
        img = self._get_image_or_warn()
        if img is None:
            return
        try:
            from dl_anomaly.core.color_inspect import build_color_palette
        except ImportError:
            messagebox.showerror(
                "模組缺失", "無法載入 color_inspect 模組。", parent=self,
            )
            return
        try:
            n = self._palette_n_var.get()
            palette = build_color_palette(img, n_colors=n)
            lines = [f"提取調色板 (n={n}):", ""]
            for i, s in enumerate(palette):
                lines.append(
                    f"  #{i + 1}: RGB={s.rgb}  "
                    f"L={s.lab[0]:.1f} a={s.lab[1]:.1f} b={s.lab[2]:.1f}  "
                    f"面積={s.area}"
                )
            self._set_color_result("\n".join(lines))
            self._set_status(f"已提取 {n} 色調色板")
        except Exception as exc:
            logger.exception("Palette extraction error")
            messagebox.showerror("錯誤", str(exc), parent=self)

    # =================================================================== #
    #  OCR handlers                                                         #
    # =================================================================== #

    def _run_ocr(self) -> None:
        img = self._get_image_or_warn()
        if img is None:
            return
        try:
            from dl_anomaly.core.ocr_engine import (
                ocr_tesseract, ocr_paddle, preprocess_for_ocr, deskew_image,
            )
        except ImportError:
            messagebox.showerror(
                "模組缺失",
                "無法載入 ocr_engine 模組。\n請確認已安裝 pytesseract 或 paddleocr。",
                parent=self,
            )
            return

        engine = self._ocr_engine_var.get()
        lang = self._ocr_lang_var.get()
        psm_text = self._ocr_psm_var.get()
        psm_match = re.search(r"\((\d+)\)", psm_text)
        psm = int(psm_match.group(1)) if psm_match else 3
        preprocess = self._ocr_preprocess_var.get()
        do_deskew = self._ocr_deskew_var.get()

        try:
            processed = img.copy()
            if do_deskew:
                processed = deskew_image(processed)
            if preprocess:
                processed = preprocess_for_ocr(processed, method="adaptive")

            if "Tesseract" in engine:
                results = ocr_tesseract(
                    processed, lang=lang, config=f"--psm {psm}",
                )
            elif "PaddleOCR" in engine or "Paddle" in engine:
                results = ocr_paddle(processed, lang=lang)
            else:
                # Default to tesseract
                results = ocr_tesseract(
                    processed, lang=lang, config=f"--psm {psm}",
                )

            self._ocr_results = results if isinstance(results, list) else [results]

            # Populate treeview
            for item in self._ocr_tree.get_children():
                self._ocr_tree.delete(item)

            for i, r in enumerate(self._ocr_results):
                self._ocr_tree.insert(
                    "", tk.END,
                    values=(
                        i + 1, r.text, f"{r.confidence * 100:.1f}%",
                        str(r.bbox),
                    ),
                )

            self._set_status(f"OCR 完成: 辨識到 {len(self._ocr_results)} 個結果")
        except Exception as exc:
            logger.exception("OCR error")
            messagebox.showerror("OCR 錯誤", str(exc), parent=self)

    def _draw_ocr_results(self) -> None:
        if not self._ocr_results:
            messagebox.showinfo("提示", "請先執行 OCR 辨識。", parent=self)
            return
        img = self._get_image_or_warn()
        if img is None:
            return
        try:
            from dl_anomaly.core.ocr_engine import draw_ocr_results
        except ImportError:
            messagebox.showerror("模組缺失", "無法載入 ocr_engine 模組。", parent=self)
            return
        try:
            result_img = draw_ocr_results(img, self._ocr_results)
            self._add_pipeline_step("OCR 結果", result_img)
            self._set_status("已繪製 OCR 結果")
        except Exception as exc:
            logger.exception("Draw OCR error")
            messagebox.showerror("錯誤", str(exc), parent=self)

    def _verify_ocr(self) -> None:
        if not self._ocr_results:
            messagebox.showinfo("提示", "請先執行 OCR 辨識。", parent=self)
            return
        pattern = self._ocr_expected_var.get().strip()
        if not pattern:
            messagebox.showwarning("提示", "請輸入預期的正規表達式。", parent=self)
            return
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            messagebox.showerror("正規表達式錯誤", str(exc), parent=self)
            return

        matches = 0
        total = len(self._ocr_results)
        for r in self._ocr_results:
            text = getattr(r, "text", str(r)) if not isinstance(r, dict) else r.get("text", "")
            if regex.search(text):
                matches += 1

        verdict = "PASS" if matches > 0 else "FAIL"
        messagebox.showinfo(
            "驗證結果",
            f"匹配: {matches}/{total}\n結果: {verdict}",
            parent=self,
        )
        self._set_status(f"OCR 驗證: {matches}/{total} [{verdict}]")

    # =================================================================== #
    #  Barcode handlers                                                     #
    # =================================================================== #

    def _scan_barcode(self) -> None:
        img = self._get_image_or_warn()
        if img is None:
            return
        try:
            from dl_anomaly.core.barcode_engine import (
                decode_barcodes, grade_barcode_quality,
            )
        except ImportError:
            messagebox.showerror(
                "模組缺失",
                "無法載入 barcode_engine 模組。\n請確認已安裝 pyzbar 或 opencv-contrib。",
                parent=self,
            )
            return

        decoder = self._bc_decoder_var.get()
        quality = self._bc_quality_var.get()

        # Collect selected types
        if self._bc_type_all_var.get():
            bc_types = None  # all types
        else:
            bc_types = []
            if self._bc_type_ean13_var.get():
                bc_types.append("EAN13")
            if self._bc_type_code128_var.get():
                bc_types.append("CODE128")
            if self._bc_type_qr_var.get():
                bc_types.append("QRCODE")
            if self._bc_type_dm_var.get():
                bc_types.append("DATAMATRIX")

        try:
            results = decode_barcodes(
                img,
                decoder=decoder.lower() if decoder else "auto",
                types=bc_types,
            )

            # Optionally grade quality
            if quality:
                for r in results:
                    try:
                        grade_barcode_quality(img, r)  # updates r in-place
                    except Exception:
                        pass

            self._barcode_results = results

            # Populate treeview
            for item in self._bc_tree.get_children():
                self._bc_tree.delete(item)

            for i, r in enumerate(self._barcode_results):
                self._bc_tree.insert(
                    "", tk.END,
                    values=(
                        i + 1, r.data, r.type,
                        getattr(r, "quality_grade", "--"),
                        f"{r.confidence * 100:.1f}%",
                    ),
                )

            self._set_status(f"條碼掃描完成: 偵測到 {len(self._barcode_results)} 個")
        except Exception as exc:
            logger.exception("Barcode scan error")
            messagebox.showerror("條碼掃描錯誤", str(exc), parent=self)

    def _draw_barcode_results(self) -> None:
        if not self._barcode_results:
            messagebox.showinfo("提示", "請先執行條碼掃描。", parent=self)
            return
        img = self._get_image_or_warn()
        if img is None:
            return
        try:
            from dl_anomaly.core.barcode_engine import draw_barcode_results
        except ImportError:
            messagebox.showerror(
                "模組缺失", "無法載入 barcode_engine 模組。", parent=self,
            )
            return
        try:
            result_img = draw_barcode_results(img, self._barcode_results)
            self._add_pipeline_step("條碼結果", result_img)
            self._set_status("已繪製條碼結果")
        except Exception as exc:
            logger.exception("Draw barcode error")
            messagebox.showerror("錯誤", str(exc), parent=self)

    def _barcode_quality_profile(self) -> None:
        if not self._barcode_results:
            messagebox.showinfo("提示", "請先執行條碼掃描。", parent=self)
            return
        # Use the selected item in treeview, or first result
        selection = self._bc_tree.selection()
        idx = 0
        if selection:
            values = self._bc_tree.item(selection[0], "values")
            idx = int(values[0]) - 1 if values else 0

        if idx >= len(self._barcode_results):
            return

        img = self._get_image_or_warn()
        if img is None:
            return
        try:
            from dl_anomaly.core.barcode_engine import (
                compute_scan_profile, draw_scan_profile,
            )
        except ImportError:
            messagebox.showerror(
                "模組缺失", "無法載入 barcode_engine 模組。", parent=self,
            )
            return
        try:
            bc_result = self._barcode_results[idx]
            profile = compute_scan_profile(img, bc_result.bbox)
            profile_img = draw_scan_profile(profile)
            self._add_pipeline_step("條碼品質剖面", profile_img)
            self._set_status("已產生條碼品質剖面")
        except Exception as exc:
            logger.exception("Barcode profile error")
            messagebox.showerror("錯誤", str(exc), parent=self)

    def _verify_barcode(self) -> None:
        if not self._barcode_results:
            messagebox.showinfo("提示", "請先執行條碼掃描。", parent=self)
            return
        expected = self._bc_expected_var.get().strip()
        if not expected:
            messagebox.showwarning("提示", "請輸入預期資料。", parent=self)
            return

        matches = 0
        total = len(self._barcode_results)
        for r in self._barcode_results:
            data = r.get("data", "") if isinstance(r, dict) else getattr(r, "data", str(r))
            if data == expected:
                matches += 1

        verdict = "PASS" if matches > 0 else "FAIL"
        messagebox.showinfo(
            "驗證結果",
            f"匹配: {matches}/{total}\n結果: {verdict}",
            parent=self,
        )
        self._set_status(f"條碼驗證: {matches}/{total} [{verdict}]")

    # =================================================================== #
    #  Close                                                                #
    # =================================================================== #

    def _close(self) -> None:
        self._photo_refs.clear()
        self.grab_release()
        self.destroy()
