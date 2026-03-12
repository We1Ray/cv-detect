"""
gui/engineering_tools_dialog.py - Phase 4 engineering tools dialog.

Provides a tabbed dialog combining:
1. Camera calibration & world-coordinate mapping
2. Multi-threaded processing pipeline
3. SQLite results database & SPC control charts
4. Image stitching / panorama

All heavy imports are lazy-loaded inside handlers with try/except
so the dialog opens instantly even when optional dependencies are
missing.
"""
from __future__ import annotations

import logging
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

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
#  EngineeringToolsDialog                                                      #
# =========================================================================== #


class EngineeringToolsDialog(tk.Toplevel):
    """Combined Phase 4 engineering tools dialog.

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
        self.title("工程工具 - 標定 / 管線 / SPC / 拼接")
        self.geometry("950x700")
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
        self._calib_result: Any = None          # CalibrationResult
        self._world_mapping: Any = None         # WorldMapping
        self._calib_images: List[np.ndarray] = []
        self._measure_points: List[Tuple[float, float]] = []

        self._pipeline: Any = None              # ParallelPipeline
        self._pipeline_running = False

        self._results_db: Any = None            # ResultsDatabase
        self._spc_metrics: Any = None           # SPCMetrics

        self._stitch_images_list: List[np.ndarray] = []
        self._stitch_result: Any = None         # StitchResult

        # ---- Tkinter variables (Calibration) --------------------------------
        self._pattern_type_var = tk.StringVar(value="chessboard")
        self._pattern_rows_var = tk.StringVar(value="6")
        self._pattern_cols_var = tk.StringVar(value="9")
        self._square_size_var = tk.StringVar(value="25.0")
        self._known_px_dist_var = tk.StringVar(value="100.0")
        self._known_mm_dist_var = tk.StringVar(value="10.0")
        self._grid_spacing_var = tk.StringVar(value="10.0")

        # ---- Tkinter variables (Pipeline) -----------------------------------
        self._pipe_preprocess_workers_var = tk.StringVar(value="2")
        self._pipe_postprocess_workers_var = tk.StringVar(value="2")
        self._pipe_batch_workers_var = tk.StringVar(value="4")
        self._pipe_timeout_var = tk.StringVar(value="30")
        self._pipe_input_dir_var = tk.StringVar(value="")

        # ---- Tkinter variables (SPC) ----------------------------------------
        self._spc_db_path_var = tk.StringVar(value="inspection_results.db")
        self._spc_start_date_var = tk.StringVar(value="")
        self._spc_end_date_var = tk.StringVar(value="")
        self._spc_model_type_var = tk.StringVar(value="")
        self._spc_batch_id_var = tk.StringVar(value="")
        self._spc_field_var = tk.StringVar(value="anomaly_score")
        self._spc_usl_var = tk.StringVar(value="")
        self._spc_lsl_var = tk.StringVar(value="")

        # ---- Tkinter variables (Stitching) ----------------------------------
        self._stitch_mode_var = tk.StringVar(value="panorama")
        self._stitch_feature_var = tk.StringVar(value="ORB")
        self._stitch_blend_var = tk.StringVar(value="multiband")
        self._stitch_direction_var = tk.StringVar(value="horizontal")
        self._stitch_overlap_var = tk.StringVar(value="0.3")
        self._stitch_grid_rows_var = tk.StringVar(value="2")
        self._stitch_grid_cols_var = tk.StringVar(value="2")

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._close)

    # =================================================================== #
    #  UI construction                                                      #
    # =================================================================== #

    def _build_ui(self) -> None:
        """Build the four-tab notebook layout."""
        style = ttk.Style(self)
        style.configure("Eng.TNotebook", background=_BG)
        style.configure(
            "Eng.TNotebook.Tab",
            background=_BG_MEDIUM, foreground=_FG, padding=[12, 4],
        )
        style.map(
            "Eng.TNotebook.Tab",
            background=[("selected", _ACTIVE_BG)],
            foreground=[("selected", "#ffffff")],
        )

        self._notebook = ttk.Notebook(self, style="Eng.TNotebook")
        self._notebook.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # Tab 1 - Calibration
        tab1 = tk.Frame(self._notebook, bg=_BG)
        self._notebook.add(tab1, text=" 座標標定 ")
        self._build_calibration_tab(tab1)

        # Tab 2 - Pipeline
        tab2 = tk.Frame(self._notebook, bg=_BG)
        self._notebook.add(tab2, text=" 並行管線 ")
        self._build_pipeline_tab(tab2)

        # Tab 3 - SPC
        tab3 = tk.Frame(self._notebook, bg=_BG)
        self._notebook.add(tab3, text=" SPC 統計 ")
        self._build_spc_tab(tab3)

        # Tab 4 - Stitching
        tab4 = tk.Frame(self._notebook, bg=_BG)
        self._notebook.add(tab4, text=" 影像拼接 ")
        self._build_stitching_tab(tab4)

        # Bottom close button
        btn_frame = tk.Frame(self, bg=_BG)
        btn_frame.pack(fill=tk.X, padx=10, pady=(0, 8))
        tk.Button(
            btn_frame, text="關閉", **_BTN_KW, command=self._close,
        ).pack(side=tk.RIGHT)

    # =================================================================== #
    #  Common helpers                                                       #
    # =================================================================== #

    def _get_image_or_warn(self) -> Optional[np.ndarray]:
        """Return current image or show a warning dialog."""
        img = self._get_current_image()
        if img is None:
            messagebox.showwarning("無影像", "請先載入影像。", parent=self)
        return img

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

    def _make_canvas(
        self, parent: tk.Widget, size: int = 300,
    ) -> tk.Canvas:
        """Create a dark-themed square canvas."""
        canvas = tk.Canvas(
            parent, width=size, height=size, bg=_CANVAS_BG,
            highlightthickness=0,
        )
        return canvas

    # =================================================================== #
    #  Tab 1: Calibration (座標標定)                                        #
    # =================================================================== #

    def _build_calibration_tab(self, parent: tk.Frame) -> None:
        paned = tk.PanedWindow(
            parent, orient=tk.HORIZONTAL, bg=_BG,
            sashwidth=4, sashrelief=tk.FLAT,
        )
        paned.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        left = tk.Frame(paned, bg=_BG)
        right = tk.Frame(paned, bg=_BG)
        paned.add(left, minsize=360)
        paned.add(right, minsize=300)

        # ---- Left column: parameters ------------------------------------
        # Pattern settings
        param_lf = tk.LabelFrame(
            left, text=" 標定板設定 ", bg=_BG, fg=_FG,
            font=("", 10, "bold"), padx=8, pady=6,
        )
        param_lf.pack(fill=tk.X, padx=4, pady=4)

        row = tk.Frame(param_lf, bg=_BG)
        row.pack(fill=tk.X, pady=2)
        tk.Label(row, text="圖案類型:", **_LABEL_KW).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Combobox(
            row, textvariable=self._pattern_type_var, state="readonly", width=16,
            values=["chessboard", "circle_grid"],
        ).pack(side=tk.LEFT)

        row = tk.Frame(param_lf, bg=_BG)
        row.pack(fill=tk.X, pady=2)
        tk.Label(row, text="列數:", **_LABEL_KW).pack(side=tk.LEFT, padx=(0, 4))
        tk.Spinbox(
            row, textvariable=self._pattern_rows_var,
            from_=2, to=20, increment=1, width=5, **_SPINBOX_KW,
        ).pack(side=tk.LEFT, padx=(0, 10))
        tk.Label(row, text="行數:", **_LABEL_KW).pack(side=tk.LEFT, padx=(0, 4))
        tk.Spinbox(
            row, textvariable=self._pattern_cols_var,
            from_=2, to=20, increment=1, width=5, **_SPINBOX_KW,
        ).pack(side=tk.LEFT)

        row = tk.Frame(param_lf, bg=_BG)
        row.pack(fill=tk.X, pady=2)
        tk.Label(row, text="方格大小 (mm):", **_LABEL_KW).pack(side=tk.LEFT, padx=(0, 4))
        tk.Entry(row, textvariable=self._square_size_var, width=8, **_ENTRY_KW).pack(side=tk.LEFT)

        # Calibration images
        btn_frame = tk.Frame(param_lf, bg=_BG)
        btn_frame.pack(fill=tk.X, pady=(8, 2))
        tk.Button(
            btn_frame, text="載入標定影像", **_BTN_KW,
            command=self._load_calib_images,
        ).pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(
            btn_frame, text="執行標定", **_BTN_ACCENT_KW,
            command=self._run_calibration,
        ).pack(side=tk.LEFT)

        self._calib_images_label = tk.Label(
            param_lf, text="已載入: 0 張影像", **_LABEL_KW,
        )
        self._calib_images_label.pack(fill=tk.X, pady=2)

        # Quick calibration section
        quick_lf = tk.LabelFrame(
            left, text=" 快速標定 ", bg=_BG, fg=_FG,
            font=("", 10, "bold"), padx=8, pady=6,
        )
        quick_lf.pack(fill=tk.X, padx=4, pady=4)

        row = tk.Frame(quick_lf, bg=_BG)
        row.pack(fill=tk.X, pady=2)
        tk.Label(row, text="像素距離:", **_LABEL_KW).pack(side=tk.LEFT, padx=(0, 4))
        tk.Entry(row, textvariable=self._known_px_dist_var, width=8, **_ENTRY_KW).pack(side=tk.LEFT, padx=(0, 8))
        tk.Label(row, text="實際 (mm):", **_LABEL_KW).pack(side=tk.LEFT, padx=(0, 4))
        tk.Entry(row, textvariable=self._known_mm_dist_var, width=8, **_ENTRY_KW).pack(side=tk.LEFT)

        btn_frame = tk.Frame(quick_lf, bg=_BG)
        btn_frame.pack(fill=tk.X, pady=(4, 2))
        tk.Button(
            btn_frame, text="已知距離標定", **_BTN_KW,
            command=self._calibrate_known_distance,
        ).pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(
            btn_frame, text="棋盤格快速標定", **_BTN_KW,
            command=self._calibrate_from_chessboard,
        ).pack(side=tk.LEFT)

        # Persistence
        persist_lf = tk.LabelFrame(
            left, text=" 存取標定 ", bg=_BG, fg=_FG,
            font=("", 10, "bold"), padx=8, pady=6,
        )
        persist_lf.pack(fill=tk.X, padx=4, pady=4)

        btn_frame = tk.Frame(persist_lf, bg=_BG)
        btn_frame.pack(fill=tk.X, pady=2)
        tk.Button(
            btn_frame, text="儲存標定", **_BTN_KW,
            command=self._save_calibration,
        ).pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(
            btn_frame, text="載入標定", **_BTN_KW,
            command=self._load_calibration,
        ).pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(
            btn_frame, text="去畸變", **_BTN_ACCENT_KW,
            command=self._undistort_image,
        ).pack(side=tk.LEFT)

        # Measurement section
        meas_lf = tk.LabelFrame(
            left, text=" 量測 ", bg=_BG, fg=_FG,
            font=("", 10, "bold"), padx=8, pady=6,
        )
        meas_lf.pack(fill=tk.X, padx=4, pady=4)

        btn_frame = tk.Frame(meas_lf, bg=_BG)
        btn_frame.pack(fill=tk.X, pady=2)
        tk.Button(
            btn_frame, text="量測距離 (點擊兩點)", **_BTN_KW,
            command=self._start_measure,
        ).pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(
            btn_frame, text="顯示世界網格", **_BTN_KW,
            command=self._show_world_grid,
        ).pack(side=tk.LEFT)

        row = tk.Frame(meas_lf, bg=_BG)
        row.pack(fill=tk.X, pady=2)
        tk.Label(row, text="網格間距 (mm):", **_LABEL_KW).pack(side=tk.LEFT, padx=(0, 4))
        tk.Entry(row, textvariable=self._grid_spacing_var, width=8, **_ENTRY_KW).pack(side=tk.LEFT)

        # ---- Right column: canvas & info ---------------------------------
        self._calib_canvas = self._make_canvas(right, 320)
        self._calib_canvas.pack(padx=4, pady=4)
        self._calib_canvas.bind("<Button-1>", self._on_calib_canvas_click)

        self._calib_info_text = tk.Text(
            right, height=10, bg=_BG_MEDIUM, fg=_FG,
            font=("Courier", 9), relief=tk.FLAT, state=tk.DISABLED,
        )
        self._calib_info_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    # =================================================================== #
    #  Tab 2: Pipeline (並行管線)                                           #
    # =================================================================== #

    def _build_pipeline_tab(self, parent: tk.Frame) -> None:
        paned = tk.PanedWindow(
            parent, orient=tk.HORIZONTAL, bg=_BG,
            sashwidth=4, sashrelief=tk.FLAT,
        )
        paned.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        left = tk.Frame(paned, bg=_BG)
        right = tk.Frame(paned, bg=_BG)
        paned.add(left, minsize=360)
        paned.add(right, minsize=300)

        # ---- Left: stage config ------------------------------------------
        config_lf = tk.LabelFrame(
            left, text=" 管線設定 ", bg=_BG, fg=_FG,
            font=("", 10, "bold"), padx=8, pady=6,
        )
        config_lf.pack(fill=tk.X, padx=4, pady=4)

        row = tk.Frame(config_lf, bg=_BG)
        row.pack(fill=tk.X, pady=2)
        tk.Label(row, text="前處理 Workers:", **_LABEL_KW).pack(side=tk.LEFT, padx=(0, 4))
        tk.Spinbox(
            row, textvariable=self._pipe_preprocess_workers_var,
            from_=1, to=16, increment=1, width=5, **_SPINBOX_KW,
        ).pack(side=tk.LEFT)

        row = tk.Frame(config_lf, bg=_BG)
        row.pack(fill=tk.X, pady=2)
        tk.Label(row, text="後處理 Workers:", **_LABEL_KW).pack(side=tk.LEFT, padx=(0, 4))
        tk.Spinbox(
            row, textvariable=self._pipe_postprocess_workers_var,
            from_=1, to=16, increment=1, width=5, **_SPINBOX_KW,
        ).pack(side=tk.LEFT)

        row = tk.Frame(config_lf, bg=_BG)
        row.pack(fill=tk.X, pady=2)
        tk.Label(row, text="批次 Workers:", **_LABEL_KW).pack(side=tk.LEFT, padx=(0, 4))
        tk.Spinbox(
            row, textvariable=self._pipe_batch_workers_var,
            from_=1, to=16, increment=1, width=5, **_SPINBOX_KW,
        ).pack(side=tk.LEFT)

        row = tk.Frame(config_lf, bg=_BG)
        row.pack(fill=tk.X, pady=2)
        tk.Label(row, text="逾時 (秒):", **_LABEL_KW).pack(side=tk.LEFT, padx=(0, 4))
        tk.Entry(row, textvariable=self._pipe_timeout_var, width=8, **_ENTRY_KW).pack(side=tk.LEFT)

        # Preset buttons
        preset_lf = tk.LabelFrame(
            left, text=" 預設管線 ", bg=_BG, fg=_FG,
            font=("", 10, "bold"), padx=8, pady=6,
        )
        preset_lf.pack(fill=tk.X, padx=4, pady=4)

        btn_frame = tk.Frame(preset_lf, bg=_BG)
        btn_frame.pack(fill=tk.X, pady=2)
        tk.Button(
            btn_frame, text="檢測管線", **_BTN_ACCENT_KW,
            command=self._create_inspection_pipeline,
        ).pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(
            btn_frame, text="批次處理", **_BTN_KW,
            command=self._create_batch_processor,
        ).pack(side=tk.LEFT)

        # Input folder
        input_lf = tk.LabelFrame(
            left, text=" 輸入資料 ", bg=_BG, fg=_FG,
            font=("", 10, "bold"), padx=8, pady=6,
        )
        input_lf.pack(fill=tk.X, padx=4, pady=4)

        row = tk.Frame(input_lf, bg=_BG)
        row.pack(fill=tk.X, pady=2)
        tk.Entry(
            row, textvariable=self._pipe_input_dir_var, **_ENTRY_KW,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        tk.Button(
            row, text="瀏覽...", **_BTN_KW,
            command=self._browse_pipeline_input,
        ).pack(side=tk.LEFT)

        # Action buttons
        action_lf = tk.LabelFrame(
            left, text=" 執行 ", bg=_BG, fg=_FG,
            font=("", 10, "bold"), padx=8, pady=6,
        )
        action_lf.pack(fill=tk.X, padx=4, pady=4)

        btn_frame = tk.Frame(action_lf, bg=_BG)
        btn_frame.pack(fill=tk.X, pady=2)
        tk.Button(
            btn_frame, text="開始處理", **_BTN_ACCENT_KW,
            command=self._start_pipeline,
        ).pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(
            btn_frame, text="效能測試", **_BTN_KW,
            command=self._benchmark_pipeline,
        ).pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(
            btn_frame, text="停止", **_BTN_KW,
            command=self._stop_pipeline,
        ).pack(side=tk.LEFT)

        # Progress bar
        self._pipe_progress_var = tk.DoubleVar(value=0.0)
        self._pipe_progress = ttk.Progressbar(
            action_lf, variable=self._pipe_progress_var,
            maximum=100, mode="determinate",
        )
        self._pipe_progress.pack(fill=tk.X, pady=(4, 2))

        # ---- Right: results display --------------------------------------
        self._pipe_results_text = tk.Text(
            right, bg=_BG_MEDIUM, fg=_FG,
            font=("Courier", 9), relief=tk.FLAT, state=tk.DISABLED,
        )
        self._pipe_results_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    # =================================================================== #
    #  Tab 3: SPC Analytics (SPC 統計)                                     #
    # =================================================================== #

    def _build_spc_tab(self, parent: tk.Frame) -> None:
        paned = tk.PanedWindow(
            parent, orient=tk.HORIZONTAL, bg=_BG,
            sashwidth=4, sashrelief=tk.FLAT,
        )
        paned.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        left = tk.Frame(paned, bg=_BG)
        right = tk.Frame(paned, bg=_BG)
        paned.add(left, minsize=360)
        paned.add(right, minsize=300)

        # ---- Left: database & query --------------------------------------
        db_lf = tk.LabelFrame(
            left, text=" 資料庫 ", bg=_BG, fg=_FG,
            font=("", 10, "bold"), padx=8, pady=6,
        )
        db_lf.pack(fill=tk.X, padx=4, pady=4)

        row = tk.Frame(db_lf, bg=_BG)
        row.pack(fill=tk.X, pady=2)
        tk.Entry(
            row, textvariable=self._spc_db_path_var, **_ENTRY_KW,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        tk.Button(
            row, text="瀏覽...", **_BTN_KW,
            command=self._browse_db_path,
        ).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(
            row, text="開啟資料庫", **_BTN_ACCENT_KW,
            command=self._open_database,
        ).pack(side=tk.LEFT)

        # Query filters
        query_lf = tk.LabelFrame(
            left, text=" 查詢條件 ", bg=_BG, fg=_FG,
            font=("", 10, "bold"), padx=8, pady=6,
        )
        query_lf.pack(fill=tk.X, padx=4, pady=4)

        row = tk.Frame(query_lf, bg=_BG)
        row.pack(fill=tk.X, pady=2)
        tk.Label(row, text="起始日期:", **_LABEL_KW).pack(side=tk.LEFT, padx=(0, 4))
        tk.Entry(row, textvariable=self._spc_start_date_var, width=14, **_ENTRY_KW).pack(side=tk.LEFT, padx=(0, 8))
        tk.Label(row, text="結束日期:", **_LABEL_KW).pack(side=tk.LEFT, padx=(0, 4))
        tk.Entry(row, textvariable=self._spc_end_date_var, width=14, **_ENTRY_KW).pack(side=tk.LEFT)

        row = tk.Frame(query_lf, bg=_BG)
        row.pack(fill=tk.X, pady=2)
        tk.Label(row, text="模型類型:", **_LABEL_KW).pack(side=tk.LEFT, padx=(0, 4))
        tk.Entry(row, textvariable=self._spc_model_type_var, width=14, **_ENTRY_KW).pack(side=tk.LEFT, padx=(0, 8))
        tk.Label(row, text="批次 ID:", **_LABEL_KW).pack(side=tk.LEFT, padx=(0, 4))
        tk.Entry(row, textvariable=self._spc_batch_id_var, width=14, **_ENTRY_KW).pack(side=tk.LEFT)

        tk.Button(
            query_lf, text="查詢記錄", **_BTN_KW,
            command=self._query_records,
        ).pack(anchor=tk.W, pady=(4, 2))

        # Record list Treeview
        tree_frame = tk.Frame(left, bg=_BG)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        cols = ("id", "timestamp", "model", "score", "defective")
        self._record_tree = ttk.Treeview(
            tree_frame, columns=cols, show="headings", height=6,
        )
        self._record_tree.heading("id", text="ID")
        self._record_tree.heading("timestamp", text="時間")
        self._record_tree.heading("model", text="模型")
        self._record_tree.heading("score", text="分數")
        self._record_tree.heading("defective", text="異常")
        self._record_tree.column("id", width=40)
        self._record_tree.column("timestamp", width=120)
        self._record_tree.column("model", width=80)
        self._record_tree.column("score", width=60)
        self._record_tree.column("defective", width=50)

        tree_scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self._record_tree.yview)
        self._record_tree.configure(yscrollcommand=tree_scroll.set)
        self._record_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # SPC controls
        spc_lf = tk.LabelFrame(
            left, text=" SPC 分析 ", bg=_BG, fg=_FG,
            font=("", 10, "bold"), padx=8, pady=6,
        )
        spc_lf.pack(fill=tk.X, padx=4, pady=4)

        row = tk.Frame(spc_lf, bg=_BG)
        row.pack(fill=tk.X, pady=2)
        tk.Label(row, text="分析欄位:", **_LABEL_KW).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Combobox(
            row, textvariable=self._spc_field_var, state="readonly", width=18,
            values=["anomaly_score", "defect_count", "total_defect_area", "max_defect_area"],
        ).pack(side=tk.LEFT)

        row = tk.Frame(spc_lf, bg=_BG)
        row.pack(fill=tk.X, pady=2)
        tk.Label(row, text="USL:", **_LABEL_KW).pack(side=tk.LEFT, padx=(0, 4))
        tk.Entry(row, textvariable=self._spc_usl_var, width=8, **_ENTRY_KW).pack(side=tk.LEFT, padx=(0, 8))
        tk.Label(row, text="LSL:", **_LABEL_KW).pack(side=tk.LEFT, padx=(0, 4))
        tk.Entry(row, textvariable=self._spc_lsl_var, width=8, **_ENTRY_KW).pack(side=tk.LEFT)

        btn_frame = tk.Frame(spc_lf, bg=_BG)
        btn_frame.pack(fill=tk.X, pady=(4, 2))
        tk.Button(
            btn_frame, text="計算 SPC", **_BTN_ACCENT_KW,
            command=self._compute_spc,
        ).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(
            btn_frame, text="偵測異常", **_BTN_KW,
            command=self._detect_out_of_control,
        ).pack(side=tk.LEFT)

        btn_frame2 = tk.Frame(spc_lf, bg=_BG)
        btn_frame2.pack(fill=tk.X, pady=(2, 2))
        tk.Button(
            btn_frame2, text="管制圖", **_BTN_KW,
            command=self._plot_control_chart,
        ).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(
            btn_frame2, text="直方圖", **_BTN_KW,
            command=self._plot_histogram,
        ).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(
            btn_frame2, text="趨勢圖", **_BTN_KW,
            command=self._plot_trend,
        ).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(
            btn_frame2, text="Pareto", **_BTN_KW,
            command=self._plot_pareto,
        ).pack(side=tk.LEFT)

        # Export
        btn_frame3 = tk.Frame(spc_lf, bg=_BG)
        btn_frame3.pack(fill=tk.X, pady=(2, 2))
        tk.Button(
            btn_frame3, text="匯出 CSV", **_BTN_KW,
            command=self._export_csv,
        ).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(
            btn_frame3, text="匯出 JSON", **_BTN_KW,
            command=self._export_json,
        ).pack(side=tk.LEFT)

        # ---- Right: canvas & SPC info ------------------------------------
        self._spc_canvas = self._make_canvas(right, 400)
        self._spc_canvas.pack(padx=4, pady=4)

        self._spc_info_text = tk.Text(
            right, height=10, bg=_BG_MEDIUM, fg=_FG,
            font=("Courier", 9), relief=tk.FLAT, state=tk.DISABLED,
        )
        self._spc_info_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    # =================================================================== #
    #  Tab 4: Stitching (影像拼接)                                         #
    # =================================================================== #

    def _build_stitching_tab(self, parent: tk.Frame) -> None:
        paned = tk.PanedWindow(
            parent, orient=tk.HORIZONTAL, bg=_BG,
            sashwidth=4, sashrelief=tk.FLAT,
        )
        paned.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        left = tk.Frame(paned, bg=_BG)
        right = tk.Frame(paned, bg=_BG)
        paned.add(left, minsize=340)
        paned.add(right, minsize=300)

        # ---- Left: settings ----------------------------------------------
        mode_lf = tk.LabelFrame(
            left, text=" 拼接模式 ", bg=_BG, fg=_FG,
            font=("", 10, "bold"), padx=8, pady=6,
        )
        mode_lf.pack(fill=tk.X, padx=4, pady=4)

        row = tk.Frame(mode_lf, bg=_BG)
        row.pack(fill=tk.X, pady=2)
        tk.Label(row, text="模式:", **_LABEL_KW).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Combobox(
            row, textvariable=self._stitch_mode_var, state="readonly", width=12,
            values=["panorama", "scan", "grid"],
        ).pack(side=tk.LEFT)

        row = tk.Frame(mode_lf, bg=_BG)
        row.pack(fill=tk.X, pady=2)
        tk.Label(row, text="特徵方法:", **_LABEL_KW).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Combobox(
            row, textvariable=self._stitch_feature_var, state="readonly", width=10,
            values=["ORB", "SIFT", "AKAZE"],
        ).pack(side=tk.LEFT)

        row = tk.Frame(mode_lf, bg=_BG)
        row.pack(fill=tk.X, pady=2)
        tk.Label(row, text="混合模式:", **_LABEL_KW).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Combobox(
            row, textvariable=self._stitch_blend_var, state="readonly", width=12,
            values=["multiband", "feather", "none"],
        ).pack(side=tk.LEFT)

        # Strip mode settings
        strip_lf = tk.LabelFrame(
            left, text=" 條帶模式設定 ", bg=_BG, fg=_FG,
            font=("", 10, "bold"), padx=8, pady=6,
        )
        strip_lf.pack(fill=tk.X, padx=4, pady=4)

        row = tk.Frame(strip_lf, bg=_BG)
        row.pack(fill=tk.X, pady=2)
        tk.Label(row, text="方向:", **_LABEL_KW).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Combobox(
            row, textvariable=self._stitch_direction_var, state="readonly", width=12,
            values=["horizontal", "vertical"],
        ).pack(side=tk.LEFT)

        row = tk.Frame(strip_lf, bg=_BG)
        row.pack(fill=tk.X, pady=2)
        tk.Label(row, text="重疊比例:", **_LABEL_KW).pack(side=tk.LEFT, padx=(0, 4))
        tk.Entry(row, textvariable=self._stitch_overlap_var, width=8, **_ENTRY_KW).pack(side=tk.LEFT)

        # Grid mode settings
        grid_lf = tk.LabelFrame(
            left, text=" 網格模式設定 ", bg=_BG, fg=_FG,
            font=("", 10, "bold"), padx=8, pady=6,
        )
        grid_lf.pack(fill=tk.X, padx=4, pady=4)

        row = tk.Frame(grid_lf, bg=_BG)
        row.pack(fill=tk.X, pady=2)
        tk.Label(row, text="列:", **_LABEL_KW).pack(side=tk.LEFT, padx=(0, 4))
        tk.Spinbox(
            row, textvariable=self._stitch_grid_rows_var,
            from_=1, to=10, increment=1, width=4, **_SPINBOX_KW,
        ).pack(side=tk.LEFT, padx=(0, 10))
        tk.Label(row, text="行:", **_LABEL_KW).pack(side=tk.LEFT, padx=(0, 4))
        tk.Spinbox(
            row, textvariable=self._stitch_grid_cols_var,
            from_=1, to=10, increment=1, width=4, **_SPINBOX_KW,
        ).pack(side=tk.LEFT)

        # Action buttons
        action_lf = tk.LabelFrame(
            left, text=" 操作 ", bg=_BG, fg=_FG,
            font=("", 10, "bold"), padx=8, pady=6,
        )
        action_lf.pack(fill=tk.X, padx=4, pady=4)

        btn_frame = tk.Frame(action_lf, bg=_BG)
        btn_frame.pack(fill=tk.X, pady=2)
        tk.Button(
            btn_frame, text="載入影像", **_BTN_KW,
            command=self._load_stitch_images,
        ).pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(
            btn_frame, text="從資料夾", **_BTN_KW,
            command=self._stitch_from_dir,
        ).pack(side=tk.LEFT)

        self._stitch_images_label = tk.Label(
            action_lf, text="已載入: 0 張影像", **_LABEL_KW,
        )
        self._stitch_images_label.pack(fill=tk.X, pady=2)

        btn_frame2 = tk.Frame(action_lf, bg=_BG)
        btn_frame2.pack(fill=tk.X, pady=(4, 2))
        tk.Button(
            btn_frame2, text="拼接", **_BTN_ACCENT_KW,
            command=self._run_stitch,
        ).pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(
            btn_frame2, text="裁剪黑邊", **_BTN_KW,
            command=self._crop_borders,
        ).pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(
            btn_frame2, text="特徵匹配可視化", **_BTN_KW,
            command=self._show_feature_matches,
        ).pack(side=tk.LEFT)

        btn_frame3 = tk.Frame(action_lf, bg=_BG)
        btn_frame3.pack(fill=tk.X, pady=(2, 2))
        tk.Button(
            btn_frame3, text="套用結果", **_BTN_ACCENT_KW,
            command=self._apply_stitch_result,
        ).pack(side=tk.LEFT)

        # ---- Right: canvas & result info ---------------------------------
        self._stitch_canvas = self._make_canvas(right, 400)
        self._stitch_canvas.pack(padx=4, pady=4)

        self._stitch_info_label = tk.Label(
            right, text="", **_LABEL_KW, wraplength=380, justify=tk.LEFT,
        )
        self._stitch_info_label.pack(fill=tk.X, padx=4, pady=4)

    # =================================================================== #
    #  Calibration handlers                                                 #
    # =================================================================== #

    def _load_calib_images(self) -> None:
        paths = filedialog.askopenfilenames(
            title="選擇標定影像", parent=self,
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")],
        )
        if not paths:
            return
        try:
            import cv2
        except ImportError:
            messagebox.showerror("錯誤", "需要 OpenCV (cv2)。", parent=self)
            return

        self._calib_images.clear()
        for p in paths:
            img = cv2.imread(p)
            if img is not None:
                self._calib_images.append(img)
            else:
                logger.warning("Failed to read calibration image: %s", p)
        self._calib_images_label.config(
            text=f"已載入: {len(self._calib_images)} 張影像",
        )
        self._set_status(f"已載入 {len(self._calib_images)} 張標定影像")

    def _run_calibration(self) -> None:
        if not self._calib_images:
            messagebox.showwarning("無影像", "請先載入標定影像。", parent=self)
            return
        try:
            from dl_anomaly.core.calibration import calibrate_camera
        except ImportError:
            messagebox.showerror("錯誤", "無法載入 calibration 模組。", parent=self)
            return

        try:
            rows = int(self._pattern_rows_var.get())
            cols = int(self._pattern_cols_var.get())
            square_size = float(self._square_size_var.get())

            result = calibrate_camera(
                self._calib_images,
                pattern_size=(cols, rows),
                square_size=square_size,
            )
            self._calib_result = result
            # Derive world mapping from calibration
            ppm = result.camera_matrix[0, 0] / square_size if square_size > 0 else 1.0
            self._set_status(f"標定完成 - RMS 誤差: {result.rms_error:.4f} px")
            self._update_calib_info()

            # Show corners on first image
            self._show_calib_corners()

        except Exception as exc:
            logger.exception("Calibration failed")
            messagebox.showerror("標定失敗", str(exc), parent=self)

    def _show_calib_corners(self) -> None:
        """Display calibration corners on the canvas."""
        if not self._calib_images or self._calib_result is None:
            return
        try:
            from dl_anomaly.core.calibration import (
                draw_calibration_corners,
                find_chessboard_corners,
            )
            import cv2
        except ImportError:
            return

        img = self._calib_images[0]
        rows = int(self._pattern_rows_var.get())
        cols = int(self._pattern_cols_var.get())
        pattern_size = (cols, rows)

        corners = find_chessboard_corners(img, pattern_size)
        if corners is not None:
            vis = draw_calibration_corners(img, corners, pattern_size)
            vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            self._update_preview(self._calib_canvas, vis_rgb, 320)

    def _calibrate_known_distance(self) -> None:
        try:
            from dl_anomaly.core.calibration import calibrate_from_known_distance
        except ImportError:
            messagebox.showerror("錯誤", "無法載入 calibration 模組。", parent=self)
            return

        try:
            px_dist = float(self._known_px_dist_var.get())
            mm_dist = float(self._known_mm_dist_var.get())
            mapping = calibrate_from_known_distance(px_dist, mm_dist)
            self._world_mapping = mapping
            self._set_status(
                f"快速標定完成 - {mapping.pixels_per_mm:.4f} px/mm"
            )
            self._update_calib_info()
        except Exception as exc:
            logger.exception("Known distance calibration failed")
            messagebox.showerror("標定失敗", str(exc), parent=self)

    def _calibrate_from_chessboard(self) -> None:
        img = self._get_image_or_warn()
        if img is None:
            return
        try:
            from dl_anomaly.core.calibration import calibrate_from_chessboard
        except ImportError:
            messagebox.showerror("錯誤", "無法載入 calibration 模組。", parent=self)
            return

        try:
            rows = int(self._pattern_rows_var.get())
            cols = int(self._pattern_cols_var.get())
            square_size = float(self._square_size_var.get())

            mapping = calibrate_from_chessboard(
                img, pattern_size=(cols, rows), square_size_mm=square_size,
            )
            self._world_mapping = mapping
            self._set_status(
                f"棋盤格標定完成 - {mapping.pixels_per_mm:.4f} px/mm"
            )
            self._update_calib_info()
        except Exception as exc:
            logger.exception("Chessboard calibration failed")
            messagebox.showerror("標定失敗", str(exc), parent=self)

    def _save_calibration(self) -> None:
        if self._calib_result is None:
            messagebox.showwarning("無標定", "尚未執行標定。", parent=self)
            return
        try:
            from dl_anomaly.core.calibration import save_calibration
        except ImportError:
            messagebox.showerror("錯誤", "無法載入 calibration 模組。", parent=self)
            return

        path = filedialog.asksaveasfilename(
            title="儲存標定結果", parent=self,
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return
        try:
            save_calibration(self._calib_result, path)
            self._set_status(f"標定已儲存至 {path}")
        except Exception as exc:
            logger.exception("Save calibration failed")
            messagebox.showerror("儲存失敗", str(exc), parent=self)

    def _load_calibration(self) -> None:
        try:
            from dl_anomaly.core.calibration import load_calibration
        except ImportError:
            messagebox.showerror("錯誤", "無法載入 calibration 模組。", parent=self)
            return

        path = filedialog.askopenfilename(
            title="載入標定結果", parent=self,
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return
        try:
            result = load_calibration(path)
            self._calib_result = result
            self._set_status(f"已載入標定 (RMS={result.rms_error:.4f})")
            self._update_calib_info()
        except Exception as exc:
            logger.exception("Load calibration failed")
            messagebox.showerror("載入失敗", str(exc), parent=self)

    def _undistort_image(self) -> None:
        img = self._get_image_or_warn()
        if img is None:
            return
        if self._calib_result is None:
            messagebox.showwarning("無標定", "請先執行或載入標定。", parent=self)
            return
        try:
            from dl_anomaly.core.calibration import undistort_image
            import cv2
        except ImportError:
            messagebox.showerror("錯誤", "無法載入 calibration 模組。", parent=self)
            return

        try:
            undistorted = undistort_image(img, self._calib_result)
            self._add_pipeline_step("去畸變", undistorted)
            vis_rgb = cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB) if undistorted.ndim == 3 else undistorted
            self._update_preview(self._calib_canvas, vis_rgb, 320)
            self._set_status("去畸變完成")
        except Exception as exc:
            logger.exception("Undistort failed")
            messagebox.showerror("去畸變失敗", str(exc), parent=self)

    def _start_measure(self) -> None:
        if self._world_mapping is None and self._calib_result is None:
            messagebox.showwarning(
                "無標定", "請先執行標定或快速標定以建立座標映射。", parent=self,
            )
            return
        self._measure_points.clear()
        self._set_status("點擊標定畫布上的兩個點以量測距離")

    def _on_calib_canvas_click(self, event: tk.Event) -> None:
        """Handle click on calibration canvas for measurement."""
        if self._world_mapping is None and self._calib_result is None:
            return

        self._measure_points.append((float(event.x), float(event.y)))

        # Draw marker on canvas
        r = 3
        self._calib_canvas.create_oval(
            event.x - r, event.y - r, event.x + r, event.y + r,
            fill="#00ffff", outline="#00ffff",
        )

        if len(self._measure_points) >= 2:
            pt1 = self._measure_points[-2]
            pt2 = self._measure_points[-1]

            # Draw line
            self._calib_canvas.create_line(
                pt1[0], pt1[1], pt2[0], pt2[1],
                fill="#00ffff", width=1,
            )

            try:
                from dl_anomaly.core.calibration import (
                    measure_distance_mm,
                    calibrate_from_known_distance,
                )
            except ImportError:
                return

            mapping = self._world_mapping
            if mapping is None:
                # Derive a simple mapping from calibration result
                if self._calib_result is not None:
                    sq = float(self._square_size_var.get()) or 1.0
                    fx = self._calib_result.camera_matrix[0, 0]
                    mapping = calibrate_from_known_distance(fx, fx / (fx / sq))
                else:
                    return

            try:
                dist = measure_distance_mm(
                    pt1[0], pt1[1], pt2[0], pt2[1], mapping,
                )
                # Show label on canvas
                mid_x = (pt1[0] + pt2[0]) / 2
                mid_y = (pt1[1] + pt2[1]) / 2
                self._calib_canvas.create_text(
                    mid_x, mid_y - 10,
                    text=f"{dist:.2f} mm",
                    fill="#00ffff", font=("", 9, "bold"),
                )
                self._set_status(f"量測距離: {dist:.2f} mm")
            except Exception as exc:
                logger.warning("Measurement failed: %s", exc)

            self._measure_points.clear()

    def _show_world_grid(self) -> None:
        img = self._get_image_or_warn()
        if img is None:
            return
        if self._world_mapping is None:
            messagebox.showwarning("無標定", "請先建立座標映射。", parent=self)
            return
        try:
            from dl_anomaly.core.calibration import draw_world_grid
            import cv2
        except ImportError:
            messagebox.showerror("錯誤", "無法載入 calibration 模組。", parent=self)
            return

        try:
            spacing = float(self._grid_spacing_var.get())
            vis = draw_world_grid(img, self._world_mapping, grid_spacing_mm=spacing)
            vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            self._update_preview(self._calib_canvas, vis_rgb, 320)
            self._set_status("世界網格已顯示")
        except Exception as exc:
            logger.exception("Draw world grid failed")
            messagebox.showerror("錯誤", str(exc), parent=self)

    def _update_calib_info(self) -> None:
        """Update the calibration info text widget."""
        lines: List[str] = []
        if self._calib_result is not None:
            r = self._calib_result
            lines.append(f"RMS 誤差: {r.rms_error:.4f} px")
            lines.append(f"影像數量: {r.num_images}")
            lines.append(f"圖案大小: {r.pattern_size}")
            lines.append(f"方格大小: {r.square_size} mm")
            lines.append(f"影像尺寸: {r.image_size}")
            fx = r.camera_matrix[0, 0]
            fy = r.camera_matrix[1, 1]
            lines.append(f"焦距: fx={fx:.2f}, fy={fy:.2f}")

        if self._world_mapping is not None:
            m = self._world_mapping
            lines.append("")
            lines.append(f"px/mm: {m.pixels_per_mm:.4f}")
            lines.append(f"px/mm (X): {m.pixels_per_mm_x:.4f}")
            lines.append(f"px/mm (Y): {m.pixels_per_mm_y:.4f}")
            lines.append(f"mm/px: {1.0/m.pixels_per_mm:.4f}")
            lines.append(f"旋轉角度: {m.rotation_deg:.2f} deg")
            lines.append(f"方法: {m.method}")

        self._calib_info_text.config(state=tk.NORMAL)
        self._calib_info_text.delete("1.0", tk.END)
        self._calib_info_text.insert("1.0", "\n".join(lines))
        self._calib_info_text.config(state=tk.DISABLED)

    # =================================================================== #
    #  Pipeline handlers                                                    #
    # =================================================================== #

    def _browse_pipeline_input(self) -> None:
        d = filedialog.askdirectory(title="選擇輸入資料夾", parent=self)
        if d:
            self._pipe_input_dir_var.set(d)

    def _create_inspection_pipeline(self) -> None:
        try:
            from dl_anomaly.core.parallel_pipeline import create_inspection_pipeline
        except ImportError:
            messagebox.showerror(
                "錯誤", "無法載入 parallel_pipeline 模組。", parent=self,
            )
            return

        try:
            n_pre = int(self._pipe_preprocess_workers_var.get())
            n_post = int(self._pipe_postprocess_workers_var.get())
            self._pipeline = create_inspection_pipeline(
                num_preprocess_workers=n_pre,
                num_postprocess_workers=n_post,
            )
            self._set_status("檢測管線已建立")
            self._append_pipe_text("檢測管線已建立 "
                                   f"(前處理={n_pre}, 後處理={n_post})\n")
        except Exception as exc:
            logger.exception("Create inspection pipeline failed")
            messagebox.showerror("錯誤", str(exc), parent=self)

    def _create_batch_processor(self) -> None:
        try:
            from dl_anomaly.core.parallel_pipeline import create_batch_processor
        except ImportError:
            messagebox.showerror(
                "錯誤", "無法載入 parallel_pipeline 模組。", parent=self,
            )
            return

        try:
            import cv2

            n_workers = int(self._pipe_batch_workers_var.get())
            timeout = float(self._pipe_timeout_var.get())

            def _process_func(path: str) -> Dict[str, Any]:
                img = cv2.imread(path)
                if img is None:
                    raise FileNotFoundError(f"Cannot read: {path}")
                return {"path": path, "shape": img.shape}

            self._pipeline = create_batch_processor(
                process_func=_process_func,
                num_workers=n_workers,
                timeout=timeout,
            )
            self._set_status("批次處理管線已建立")
            self._append_pipe_text(
                f"批次處理管線已建立 (workers={n_workers}, timeout={timeout}s)\n"
            )
        except Exception as exc:
            logger.exception("Create batch processor failed")
            messagebox.showerror("錯誤", str(exc), parent=self)

    def _start_pipeline(self) -> None:
        if self._pipeline is None:
            messagebox.showwarning("無管線", "請先建立管線。", parent=self)
            return

        input_dir = self._pipe_input_dir_var.get().strip()
        if not input_dir or not os.path.isdir(input_dir):
            messagebox.showwarning(
                "無輸入", "請選擇有效的輸入資料夾。", parent=self,
            )
            return

        # Gather image files
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        image_files = sorted(
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if os.path.splitext(f)[1].lower() in exts
        )
        if not image_files:
            messagebox.showwarning("無影像", "資料夾中沒有影像檔案。", parent=self)
            return

        self._pipeline_running = True
        self._pipe_progress_var.set(0)
        total = len(image_files)
        self._append_pipe_text(f"\n開始處理 {total} 張影像...\n")

        def _progress(current: int, total_: int) -> None:
            pct = current / total_ * 100 if total_ > 0 else 0
            self.after(0, lambda: self._pipe_progress_var.set(pct))

        def _run() -> None:
            try:
                results = self._pipeline.process_batch(
                    image_files, progress_callback=_progress,
                )
                stats = self._pipeline.get_stats()
                self.after(0, lambda: self._show_pipeline_results(results, stats))
            except Exception as exc:
                logger.exception("Pipeline processing failed")
                self.after(
                    0, lambda: messagebox.showerror("錯誤", str(exc), parent=self),
                )
            finally:
                self._pipeline_running = False

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

    def _show_pipeline_results(
        self, results: List[Any], stats: Any,
    ) -> None:
        """Display pipeline results in the text widget."""
        lines: List[str] = []
        lines.append("=" * 40)
        lines.append("處理結果")
        lines.append(f"  總計處理: {stats.total_processed}")
        lines.append(f"  失敗數量: {stats.total_failed}")
        lines.append(f"  平均時間: {stats.avg_time:.3f} s")
        lines.append(f"  吞吐量:   {stats.throughput:.2f} items/s")
        lines.append(f"  總耗時:   {stats.elapsed:.2f} s")
        lines.append("")
        lines.append("各階段平均時間:")
        for name, t in stats.stage_timings.items():
            lines.append(f"  {name}: {t:.4f} s")
        lines.append("=" * 40)

        self._append_pipe_text("\n".join(lines) + "\n")
        self._pipe_progress_var.set(100)
        self._set_status(
            f"管線處理完成: {stats.total_processed} 項, "
            f"{stats.throughput:.2f} items/s"
        )

    def _benchmark_pipeline(self) -> None:
        if self._pipeline is None:
            messagebox.showwarning("無管線", "請先建立管線。", parent=self)
            return

        input_dir = self._pipe_input_dir_var.get().strip()
        if not input_dir or not os.path.isdir(input_dir):
            messagebox.showwarning(
                "無輸入", "請選擇有效的輸入資料夾。", parent=self,
            )
            return

        try:
            from dl_anomaly.core.parallel_pipeline import benchmark_pipeline
        except ImportError:
            messagebox.showerror(
                "錯誤", "無法載入 parallel_pipeline 模組。", parent=self,
            )
            return

        exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        image_files = sorted(
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if os.path.splitext(f)[1].lower() in exts
        )
        if not image_files:
            messagebox.showwarning("無影像", "資料夾中沒有影像檔案。", parent=self)
            return

        self._append_pipe_text("\n效能測試中...\n")

        def _run() -> None:
            try:
                summary = benchmark_pipeline(self._pipeline, image_files)
                lines = [
                    "效能測試結果:",
                    f"  平均時間: {summary['avg_time']:.3f} s",
                    f"  吞吐量:   {summary['throughput']:.2f} items/s",
                    f"  總耗時:   {summary['elapsed']:.2f} s",
                    f"  成功:     {summary['successful_items']}/{summary['total_items']}",
                ]
                for name, t in summary.get("stage_timings", {}).items():
                    lines.append(f"  {name}: {t:.4f} s")
                self.after(
                    0, lambda: self._append_pipe_text("\n".join(lines) + "\n"),
                )
            except Exception as exc:
                logger.exception("Benchmark failed")
                self.after(
                    0, lambda: messagebox.showerror("錯誤", str(exc), parent=self),
                )

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

    def _stop_pipeline(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline_running = False
            self._set_status("管線已停止")
            self._append_pipe_text("管線已停止\n")

    def _append_pipe_text(self, text: str) -> None:
        """Append text to the pipeline results widget."""
        self._pipe_results_text.config(state=tk.NORMAL)
        self._pipe_results_text.insert(tk.END, text)
        self._pipe_results_text.see(tk.END)
        self._pipe_results_text.config(state=tk.DISABLED)

    # =================================================================== #
    #  SPC handlers                                                         #
    # =================================================================== #

    def _browse_db_path(self) -> None:
        path = filedialog.askopenfilename(
            title="選擇資料庫", parent=self,
            filetypes=[("SQLite", "*.db *.sqlite"), ("All", "*.*")],
        )
        if path:
            self._spc_db_path_var.set(path)

    def _open_database(self) -> None:
        try:
            from dl_anomaly.core.results_db import ResultsDatabase
        except ImportError:
            messagebox.showerror(
                "錯誤", "無法載入 results_db 模組。", parent=self,
            )
            return

        db_path = self._spc_db_path_var.get().strip()
        if not db_path:
            messagebox.showwarning("無路徑", "請輸入資料庫路徑。", parent=self)
            return

        try:
            self._results_db = ResultsDatabase(db_path)
            self._set_status(f"已開啟資料庫: {db_path}")
            self._query_records()
        except Exception as exc:
            logger.exception("Open database failed")
            messagebox.showerror("開啟失敗", str(exc), parent=self)

    def _get_spc_filters(self) -> Dict[str, Any]:
        """Collect SPC query filter values."""
        filters: Dict[str, Any] = {}
        v = self._spc_start_date_var.get().strip()
        if v:
            filters["start_date"] = v
        v = self._spc_end_date_var.get().strip()
        if v:
            filters["end_date"] = v
        v = self._spc_model_type_var.get().strip()
        if v:
            filters["model_type"] = v
        v = self._spc_batch_id_var.get().strip()
        if v:
            filters["batch_id"] = v
        return filters

    def _query_records(self) -> None:
        if self._results_db is None:
            messagebox.showwarning("無資料庫", "請先開啟資料庫。", parent=self)
            return

        filters = self._get_spc_filters()
        try:
            records = self._results_db.query_records(**filters, limit=200)
        except Exception as exc:
            logger.exception("Query records failed")
            messagebox.showerror("查詢失敗", str(exc), parent=self)
            return

        # Clear and populate treeview
        for item in self._record_tree.get_children():
            self._record_tree.delete(item)

        for rec in records:
            self._record_tree.insert("", tk.END, values=(
                rec.id,
                rec.timestamp,
                rec.model_type,
                f"{rec.anomaly_score:.4f}",
                "Yes" if rec.is_defective else "No",
            ))

        self._set_status(f"查詢到 {len(records)} 筆記錄")

    def _get_usl_lsl(self) -> Tuple[Optional[float], Optional[float]]:
        """Parse USL/LSL from entry widgets."""
        usl: Optional[float] = None
        lsl: Optional[float] = None
        v = self._spc_usl_var.get().strip()
        if v:
            try:
                usl = float(v)
            except ValueError:
                pass
        v = self._spc_lsl_var.get().strip()
        if v:
            try:
                lsl = float(v)
            except ValueError:
                pass
        return usl, lsl

    def _compute_spc(self) -> None:
        if self._results_db is None:
            messagebox.showwarning("無資料庫", "請先開啟資料庫。", parent=self)
            return

        field = self._spc_field_var.get()
        usl, lsl = self._get_usl_lsl()
        filters = self._get_spc_filters()

        try:
            metrics = self._results_db.compute_spc_metrics(
                field=field, usl=usl, lsl=lsl,
                start_date=filters.get("start_date"),
                end_date=filters.get("end_date"),
            )
            self._spc_metrics = metrics
        except Exception as exc:
            logger.exception("Compute SPC failed")
            messagebox.showerror("SPC 計算失敗", str(exc), parent=self)
            return

        lines: List[str] = [
            f"SPC 分析 - {field}",
            "=" * 30,
            f"樣本數: {metrics.n_samples}",
            f"平均值: {metrics.mean:.4f}",
            f"標準差: {metrics.std:.4f}",
            f"UCL:    {metrics.ucl:.4f}",
            f"LCL:    {metrics.lcl:.4f}",
            f"管制外點: {metrics.n_out_of_control}",
        ]
        if metrics.cp is not None:
            lines.append(f"Cp:     {metrics.cp:.4f}")
        if metrics.cpk is not None:
            lines.append(f"Cpk:    {metrics.cpk:.4f}")
        if metrics.pp is not None:
            lines.append(f"Pp:     {metrics.pp:.4f}")
        if metrics.ppk is not None:
            lines.append(f"Ppk:    {metrics.ppk:.4f}")

        self._spc_info_text.config(state=tk.NORMAL)
        self._spc_info_text.delete("1.0", tk.END)
        self._spc_info_text.insert("1.0", "\n".join(lines))
        self._spc_info_text.config(state=tk.DISABLED)

        self._set_status(f"SPC 計算完成 - Cpk={metrics.cpk}" if metrics.cpk else "SPC 計算完成")

    def _detect_out_of_control(self) -> None:
        if self._results_db is None:
            messagebox.showwarning("無資料庫", "請先開啟資料庫。", parent=self)
            return

        field = self._spc_field_var.get()
        filters = self._get_spc_filters()

        try:
            violations = self._results_db.detect_out_of_control(
                field=field,
                start_date=filters.get("start_date"),
                end_date=filters.get("end_date"),
            )
        except Exception as exc:
            logger.exception("Out-of-control detection failed")
            messagebox.showerror("偵測失敗", str(exc), parent=self)
            return

        lines: List[str] = [
            f"管制外偵測 - {field}",
            "=" * 30,
            f"違規數量: {len(violations)}",
            "",
        ]
        for v in violations[:50]:
            lines.append(
                f"  Rule {v['rule']}: idx={v['index']}, "
                f"value={v['value']:.4f}, time={v['timestamp']}"
            )
        if len(violations) > 50:
            lines.append(f"  ... 及 {len(violations) - 50} 筆更多")

        self._spc_info_text.config(state=tk.NORMAL)
        self._spc_info_text.delete("1.0", tk.END)
        self._spc_info_text.insert("1.0", "\n".join(lines))
        self._spc_info_text.config(state=tk.DISABLED)

        self._set_status(f"偵測到 {len(violations)} 個管制外點")

    def _plot_spc_chart(self, chart_method: str) -> None:
        """Generic SPC chart plotter."""
        if self._results_db is None:
            messagebox.showwarning("無資料庫", "請先開啟資料庫。", parent=self)
            return

        field = self._spc_field_var.get()
        usl, lsl = self._get_usl_lsl()
        filters = self._get_spc_filters()

        try:
            method = getattr(self._results_db, chart_method)
            kwargs: Dict[str, Any] = {
                "field": field,
                "start_date": filters.get("start_date"),
                "end_date": filters.get("end_date"),
            }
            if chart_method == "plot_histogram":
                kwargs["usl"] = usl
                kwargs["lsl"] = lsl
            if chart_method != "plot_pareto":
                kwargs["field"] = field

            chart_img = method(**kwargs)
            self._update_preview(self._spc_canvas, chart_img, 400)
            self._set_status(f"{chart_method} 已產生")
        except Exception as exc:
            logger.exception("SPC chart generation failed")
            messagebox.showerror("圖表失敗", str(exc), parent=self)

    def _plot_control_chart(self) -> None:
        self._plot_spc_chart("plot_control_chart")

    def _plot_histogram(self) -> None:
        self._plot_spc_chart("plot_histogram")

    def _plot_trend(self) -> None:
        self._plot_spc_chart("plot_trend")

    def _plot_pareto(self) -> None:
        if self._results_db is None:
            messagebox.showwarning("無資料庫", "請先開啟資料庫。", parent=self)
            return

        filters = self._get_spc_filters()
        try:
            chart_img = self._results_db.plot_pareto(
                start_date=filters.get("start_date"),
                end_date=filters.get("end_date"),
            )
            self._update_preview(self._spc_canvas, chart_img, 400)
            self._set_status("Pareto 圖已產生")
        except Exception as exc:
            logger.exception("Pareto chart failed")
            messagebox.showerror("圖表失敗", str(exc), parent=self)

    def _export_csv(self) -> None:
        if self._results_db is None:
            messagebox.showwarning("無資料庫", "請先開啟資料庫。", parent=self)
            return

        path = filedialog.asksaveasfilename(
            title="匯出 CSV", parent=self,
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
        )
        if not path:
            return

        filters = self._get_spc_filters()
        try:
            self._results_db.export_to_csv(
                path,
                start_date=filters.get("start_date"),
                end_date=filters.get("end_date"),
            )
            self._set_status(f"已匯出至 {path}")
        except Exception as exc:
            logger.exception("CSV export failed")
            messagebox.showerror("匯出失敗", str(exc), parent=self)

    def _export_json(self) -> None:
        if self._results_db is None:
            messagebox.showwarning("無資料庫", "請先開啟資料庫。", parent=self)
            return

        path = filedialog.asksaveasfilename(
            title="匯出 JSON", parent=self,
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return

        filters = self._get_spc_filters()
        try:
            self._results_db.export_to_json(
                path,
                start_date=filters.get("start_date"),
                end_date=filters.get("end_date"),
            )
            self._set_status(f"已匯出至 {path}")
        except Exception as exc:
            logger.exception("JSON export failed")
            messagebox.showerror("匯出失敗", str(exc), parent=self)

    # =================================================================== #
    #  Stitching handlers                                                   #
    # =================================================================== #

    def _load_stitch_images(self) -> None:
        paths = filedialog.askopenfilenames(
            title="選擇拼接影像", parent=self,
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")],
        )
        if not paths:
            return
        try:
            import cv2
        except ImportError:
            messagebox.showerror("錯誤", "需要 OpenCV (cv2)。", parent=self)
            return

        self._stitch_images_list.clear()
        for p in paths:
            img = cv2.imread(p)
            if img is not None:
                self._stitch_images_list.append(img)
            else:
                logger.warning("Failed to read stitch image: %s", p)

        self._stitch_images_label.config(
            text=f"已載入: {len(self._stitch_images_list)} 張影像",
        )
        self._set_status(f"已載入 {len(self._stitch_images_list)} 張拼接影像")

    def _run_stitch(self) -> None:
        if not self._stitch_images_list:
            messagebox.showwarning("無影像", "請先載入拼接影像。", parent=self)
            return

        try:
            from dl_anomaly.core.stitching import (
                stitch_images,
                stitch_strip,
                stitch_grid,
            )
            import cv2
        except ImportError:
            messagebox.showerror(
                "錯誤", "無法載入 stitching 模組。", parent=self,
            )
            return

        mode = self._stitch_mode_var.get()
        blend = self._stitch_blend_var.get()

        try:
            if mode == "scan":
                direction = self._stitch_direction_var.get()
                overlap = float(self._stitch_overlap_var.get())
                result = stitch_strip(
                    self._stitch_images_list,
                    overlap_ratio=overlap,
                    direction=direction,
                )
            elif mode == "grid":
                grid_rows = int(self._stitch_grid_rows_var.get())
                grid_cols = int(self._stitch_grid_cols_var.get())
                overlap = float(self._stitch_overlap_var.get())
                result = stitch_grid(
                    self._stitch_images_list,
                    grid_shape=(grid_rows, grid_cols),
                    overlap_ratio=overlap,
                )
            else:
                result = stitch_images(
                    self._stitch_images_list, mode=mode, blend=blend,
                )

            self._stitch_result = result

            if result.status != "failed" and result.panorama.size > 0:
                vis_rgb = cv2.cvtColor(result.panorama, cv2.COLOR_BGR2RGB)
                self._update_preview(self._stitch_canvas, vis_rgb, 400)

            info = (
                f"狀態: {result.status} | "
                f"影像數: {result.num_images} | "
                f"信心度: {result.confidence:.2f}\n"
                f"{result.message}"
            )
            self._stitch_info_label.config(text=info)
            self._set_status(f"拼接完成 - {result.status}")

        except Exception as exc:
            logger.exception("Stitching failed")
            messagebox.showerror("拼接失敗", str(exc), parent=self)

    def _stitch_from_dir(self) -> None:
        d = filedialog.askdirectory(title="選擇影像資料夾", parent=self)
        if not d:
            return

        try:
            from dl_anomaly.core.stitching import stitch_from_directory
            import cv2
        except ImportError:
            messagebox.showerror(
                "錯誤", "無法載入 stitching 模組。", parent=self,
            )
            return

        mode = self._stitch_mode_var.get()

        try:
            result = stitch_from_directory(d, mode=mode)
            self._stitch_result = result

            if result.status != "failed" and result.panorama.size > 0:
                vis_rgb = cv2.cvtColor(result.panorama, cv2.COLOR_BGR2RGB)
                self._update_preview(self._stitch_canvas, vis_rgb, 400)

            info = (
                f"狀態: {result.status} | "
                f"影像數: {result.num_images} | "
                f"信心度: {result.confidence:.2f}\n"
                f"{result.message}"
            )
            self._stitch_info_label.config(text=info)
            self._set_status(f"資料夾拼接完成 - {result.status}")

        except Exception as exc:
            logger.exception("Stitch from directory failed")
            messagebox.showerror("拼接失敗", str(exc), parent=self)

    def _crop_borders(self) -> None:
        if self._stitch_result is None or self._stitch_result.panorama.size == 0:
            messagebox.showwarning("無結果", "請先執行拼接。", parent=self)
            return

        try:
            from dl_anomaly.core.stitching import crop_black_borders
            import cv2
        except ImportError:
            messagebox.showerror(
                "錯誤", "無法載入 stitching 模組。", parent=self,
            )
            return

        try:
            cropped = crop_black_borders(self._stitch_result.panorama)
            self._stitch_result.panorama = cropped
            vis_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            self._update_preview(self._stitch_canvas, vis_rgb, 400)
            self._set_status(
                f"黑邊已裁剪 - {cropped.shape[1]}x{cropped.shape[0]}"
            )
        except Exception as exc:
            logger.exception("Crop borders failed")
            messagebox.showerror("裁剪失敗", str(exc), parent=self)

    def _show_feature_matches(self) -> None:
        if len(self._stitch_images_list) < 2:
            messagebox.showwarning(
                "影像不足", "需要至少 2 張影像以顯示特徵匹配。", parent=self,
            )
            return

        try:
            from dl_anomaly.core.stitching import (
                detect_and_match_features,
                draw_matches,
            )
            import cv2
        except ImportError:
            messagebox.showerror(
                "錯誤", "無法載入 stitching 模組。", parent=self,
            )
            return

        method = self._stitch_feature_var.get().lower()

        try:
            img1 = self._stitch_images_list[0]
            img2 = self._stitch_images_list[1]
            kp1, kp2, matches = detect_and_match_features(
                img1, img2, method=method,
            )
            vis = draw_matches(img1, img2, kp1, kp2, matches)
            vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            self._update_preview(self._stitch_canvas, vis_rgb, 400)
            self._set_status(f"特徵匹配: {len(matches)} 個匹配點")
        except Exception as exc:
            logger.exception("Feature match visualization failed")
            messagebox.showerror("錯誤", str(exc), parent=self)

    def _apply_stitch_result(self) -> None:
        if self._stitch_result is None or self._stitch_result.panorama.size == 0:
            messagebox.showwarning("無結果", "請先執行拼接。", parent=self)
            return

        self._add_pipeline_step("拼接結果", self._stitch_result.panorama)
        self._set_status("拼接結果已套用至管線")

    # =================================================================== #
    #  Close                                                                #
    # =================================================================== #

    def _close(self) -> None:
        self._photo_refs.clear()
        self.grab_release()
        self.destroy()
