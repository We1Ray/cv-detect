"""Main HALCON HDevelop-style application window.

Assembles the full layout:
- Menu bar
- Toolbar
- Three-panel PanedWindow: Pipeline (left) | Image Viewer (centre) | Properties + Operations (right)
- Status bar with position, pixel value, zoom level

Orchestrates all interactions between panels, dialogs, and the core
inference / training pipelines.
"""

from __future__ import annotations

import logging
import os
import sys as _sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, List, Optional

_sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from shared.app_state import AppState
from shared.progress_manager import ProgressManager
from shared.error_dialog import show_error
from shared.history_panel import HistoryPanel

import cv2
import numpy as np
from PIL import Image

from dl_anomaly.config import Config
from dl_anomaly.core.anomaly_scorer import AnomalyScorer
from dl_anomaly.core.preprocessor import ImagePreprocessor
from dl_anomaly.gui.dialogs import (
    BatchInspectDialog,
    HistogramDialog,
    ModelInfoDialog,
    ReconstructionDialog,
    SettingsDialog,
    TrainingDialog,
)
from dl_anomaly.gui.image_viewer import ImageViewer
from dl_anomaly.gui.operations_panel import OperationsPanel
from dl_anomaly.gui.pipeline_panel import PipelinePanel
from dl_anomaly.gui.properties_panel import PropertiesPanel
from dl_anomaly.gui.compare_dialog import CompareDialog
from dl_anomaly.gui.recipe_apply_dialog import RecipeApplyDialog
from dl_anomaly.gui.batch_compare_dialog import BatchCompareDialog
from dl_anomaly.gui.toolbar import Toolbar
from dl_anomaly.visualization.heatmap import (
    create_defect_overlay,
    create_error_heatmap,
)

logger = logging.getLogger(__name__)


class HalconApp(tk.Tk):
    """Top-level HALCON HDevelop-style application window."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config

        self.title("DL \u7570\u5e38\u5075\u6e2c\u5668 - HALCON Style")
        self.geometry("1400x900")
        self.minsize(1000, 650)

        # Configure dark theme
        self._setup_theme()

        # Persistent application state
        self._app_state = AppState("dl_anomaly")

        # Core state
        self._inference_pipeline = None  # InferencePipeline instance
        self._model = None
        self._model_state: Optional[Dict] = None
        self._current_image_path: Optional[str] = None
        self._recent_files: List[str] = []
        self._undo_stack: List[int] = []  # pipeline step indices
        self._redo_stack: List[int] = []
        self._initial_loaded: bool = False  # True after first image load

        # HALCON feature state
        self._pixel_inspector = None
        self._script_editor = None
        self._script_editor_visible = False
        self._current_region = None

        # Build UI
        self._build_menu()
        self._build_toolbar()
        self._build_main_layout()
        self._build_statusbar()
        self._bind_shortcuts()

        # Restore persisted window geometry and sash positions
        self._app_state.restore_geometry(self)
        self.after(200, lambda: self._app_state.restore_sash_positions(self._paned))

        # Load persisted recent files
        self._recent_files = self._app_state.get_recent_files()
        self._update_recent_menu()

        # Graceful close
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ==================================================================
    # Theme
    # ==================================================================

    def _setup_theme(self) -> None:
        style = ttk.Style()
        available = style.theme_names()
        # Prefer 'clam' for better dark-theme customisation
        if "clam" in available:
            style.theme_use("clam")

        # Dark colours
        bg = "#2b2b2b"
        fg = "#e0e0e0"
        sel_bg = "#3a3a5c"
        trough = "#1e1e1e"

        style.configure(".", background=bg, foreground=fg, fieldbackground=bg)
        style.configure("TFrame", background=bg)
        style.configure("TLabel", background=bg, foreground=fg)
        style.configure("TLabelframe", background=bg, foreground=fg)
        style.configure("TLabelframe.Label", background=bg, foreground=fg)
        style.configure("TButton", background="#3c3c3c", foreground=fg)
        style.configure("TEntry", fieldbackground="#3c3c3c", foreground=fg)
        style.configure("TCombobox", fieldbackground="#3c3c3c", foreground=fg)
        style.configure("TNotebook", background=bg)
        style.configure("TNotebook.Tab", background="#3c3c3c", foreground=fg)
        style.configure("Treeview", background="#1e1e1e", foreground=fg, fieldbackground="#1e1e1e")
        style.configure("Treeview.Heading", background="#3c3c3c", foreground=fg)
        style.map("Treeview", background=[("selected", sel_bg)])
        style.configure("TProgressbar", troughcolor=trough, background="#4fc3f7")
        style.configure("TSeparator", background="#555555")
        style.configure("Toolbar.TButton", background="#3c3c3c", foreground=fg, font=("Segoe UI", 11), padding=(4, 2))

        style.configure("Success.Status.TLabel", background="#2e7d32", foreground="#ffffff")

        self.configure(bg=bg)

    # ==================================================================
    # Menu bar
    # ==================================================================

    def _build_menu(self) -> None:
        menubar = tk.Menu(self, bg="#2b2b2b", fg="#e0e0e0", activebackground="#3a3a5c", activeforeground="#ffffff")

        # -- File --
        file_menu = tk.Menu(menubar, tearoff=0, bg="#2b2b2b", fg="#e0e0e0",
                            activebackground="#3a3a5c", activeforeground="#ffffff")
        file_menu.add_command(label="\u958b\u555f\u5716\u7247...", command=self._cmd_open_image, accelerator="Ctrl+O")
        file_menu.add_command(label="\u958b\u555f\u8cc7\u6599\u593e...", command=self._cmd_open_dir)
        file_menu.add_separator()
        file_menu.add_command(label="\u5132\u5b58\u7576\u524d\u5716\u7247...", command=self._cmd_save_image, accelerator="Ctrl+S")
        file_menu.add_command(label="\u5132\u5b58\u6240\u6709\u6b65\u9a5f...", command=self._cmd_save_all)
        file_menu.add_separator()
        file_menu.add_command(label="\u5132\u5b58\u6d41\u7a0b...", command=self._cmd_save_recipe)
        file_menu.add_command(label="\u8f09\u5165\u4e26\u5957\u7528\u6d41\u7a0b...", command=self._cmd_load_and_apply_recipe)
        file_menu.add_command(label="\u6279\u6b21\u5957\u7528\u6d41\u7a0b...", command=self._cmd_batch_apply_recipe)
        file_menu.add_separator()
        file_menu.add_command(label="\u74b0\u5883\u8a2d\u5b9a...", command=self._cmd_settings)
        file_menu.add_separator()
        self._recent_menu = tk.Menu(file_menu, tearoff=0, bg="#2b2b2b", fg="#e0e0e0",
                                    activebackground="#3a3a5c", activeforeground="#ffffff")
        file_menu.add_cascade(label="\u6700\u8fd1\u958b\u555f", menu=self._recent_menu)
        file_menu.add_separator()
        file_menu.add_command(label="\u7d50\u675f", command=self._on_close)
        menubar.add_cascade(label="\u6a94\u6848", menu=file_menu)

        # -- Operations --
        ops_menu = tk.Menu(menubar, tearoff=0, bg="#2b2b2b", fg="#e0e0e0",
                           activebackground="#3a3a5c", activeforeground="#ffffff")
        ops_menu.add_command(label="\u7070\u968e\u8f49\u63db", command=lambda: self._ops_panel._op_grayscale())
        ops_menu.add_command(label="\u9ad8\u65af\u6a21\u7cca", command=lambda: self._ops_panel._op_blur())
        ops_menu.add_command(label="\u908a\u7de3\u5075\u6e2c", command=lambda: self._ops_panel._op_edge())
        ops_menu.add_command(label="\u76f4\u65b9\u5716\u5747\u8861", command=lambda: self._ops_panel._op_histeq())
        ops_menu.add_command(label="\u53cd\u8272", command=lambda: self._ops_panel._op_invert())
        ops_menu.add_separator()
        ops_menu.add_command(label="\u57f7\u884c\u81ea\u52d5\u7de8\u78bc\u5668", command=self._cmd_run_autoencoder)
        ops_menu.add_command(label="\u8a08\u7b97\u8aa4\u5dee\u5716", command=self._cmd_compute_error_map)
        ops_menu.add_command(label="\u5957\u7528 SSIM", command=self._cmd_apply_ssim)
        ops_menu.add_command(label="\u5957\u7528\u95be\u503c\u906e\u7f69", command=self._cmd_apply_threshold_mask)
        menubar.add_cascade(label="\u64cd\u4f5c", menu=ops_menu)

        # -- Region --
        region_menu = tk.Menu(menubar, tearoff=0, bg="#2b2b2b", fg="#e0e0e0",
                              activebackground="#3a3a5c", activeforeground="#ffffff")
        region_menu.add_command(label="\u50cf\u7d20\u503c\u6aa2\u67e5\u5668...", command=self._toggle_pixel_inspector, accelerator="Ctrl+I")
        region_menu.add_command(
            label="\u50cf\u7d20\u6aa2\u67e5\u5de5\u5177",
            command=lambda: self._cmd_tool_pixel_inspect(
                not self._toolbar.get_toggle_state("tool_pixel_inspect")),
            accelerator="Ctrl+Shift+I",
        )
        region_menu.add_command(
            label="\u5340\u57df\u9078\u53d6\u5de5\u5177",
            command=lambda: self._cmd_tool_region_select(
                not self._toolbar.get_toggle_state("tool_region_select")),
            accelerator="Ctrl+Shift+R",
        )
        region_menu.add_separator()
        region_menu.add_command(label="二值化...", command=self._open_binarize_dialog)
        region_menu.add_separator()
        region_menu.add_command(label="\u95be\u503c\u5206\u5272...", command=self._open_threshold_dialog, accelerator="Ctrl+T")
        region_menu.add_command(label="\u81ea\u52d5\u95be\u503c (Otsu)", command=self._auto_threshold_otsu)
        region_menu.add_command(label="自適應閾值...", command=self._open_adaptive_threshold_dialog)
        region_menu.add_command(label="可變閾值...", command=self._dlg_var_threshold)
        region_menu.add_command(label="局部閾值...", command=self._dlg_local_threshold)
        region_menu.add_separator()
        region_menu.add_command(label="\u6253\u6563 (Connection)", command=self._region_connection)
        region_menu.add_command(label="\u586b\u5145 (Fill Up)...", command=self._region_fill_up)

        shape_trans_menu = tk.Menu(region_menu, tearoff=0, bg="#2b2b2b", fg="#e0e0e0",
                                   activebackground="#3a3a5c", activeforeground="#ffffff")
        for st in ["convex", "rectangle", "circle", "ellipse"]:
            shape_trans_menu.add_command(label=st, command=lambda s=st: self._region_shape_trans(s))
        region_menu.add_cascade(label="\u5f62\u72c0\u8b8a\u63db", menu=shape_trans_menu)

        region_menu.add_separator()
        region_menu.add_command(label="\u5340\u57df\u4fb5\u8755...", command=lambda: self._region_morphology("erosion"))
        region_menu.add_command(label="\u5340\u57df\u81a8\u8139...", command=lambda: self._region_morphology("dilation"))
        region_menu.add_command(label="\u5340\u57df\u958b\u904b\u7b97...", command=lambda: self._region_morphology("opening"))
        region_menu.add_command(label="\u5340\u57df\u9589\u904b\u7b97...", command=lambda: self._region_morphology("closing"))
        region_menu.add_separator()
        region_menu.add_command(label="\u7be9\u9078\u5340\u57df...", command=self._open_region_filter)
        region_menu.add_command(label="\u4f9d\u7070\u5ea6\u7be9\u9078...", command=self._region_select_gray)
        region_menu.add_command(label="\u6392\u5e8f\u5340\u57df...", command=self._region_sort)
        region_menu.add_separator()
        region_menu.add_command(label="\u5340\u57df\u806f\u96c6", command=lambda: self._region_set_op("union"))
        region_menu.add_command(label="\u5340\u57df\u4ea4\u96c6", command=lambda: self._region_set_op("intersection"))
        region_menu.add_command(label="\u5340\u57df\u5dee\u96c6", command=lambda: self._region_set_op("difference"))
        region_menu.add_command(label="\u5340\u57df\u88dc\u96c6", command=lambda: self._region_set_op("complement"))
        region_menu.add_separator()
        region_menu.add_command(label="縮減域 (Reduce Domain)", command=self._cmd_reduce_domain)
        region_menu.add_command(label="裁切域 (Crop Domain)", command=self._cmd_crop_domain)
        region_menu.add_separator()
        region_menu.add_command(label="Blob \u5206\u6790...", command=self._open_blob_analysis)
        region_menu.add_command(label="輪廓檢測...", command=self._open_contour_detection_dialog)
        menubar.add_cascade(label="\u5340\u57df", menu=region_menu)

        # -- HALCON --
        halcon_menu = tk.Menu(menubar, tearoff=0, bg="#2b2b2b", fg="#e0e0e0",
                              activebackground="#3a3a5c", activeforeground="#ffffff")

        filter_menu = tk.Menu(halcon_menu, tearoff=0, bg="#2b2b2b", fg="#e0e0e0",
                              activebackground="#3a3a5c", activeforeground="#ffffff")
        filter_menu.add_command(label="均值濾波...", command=self._dlg_mean_image)
        filter_menu.add_command(label="中值濾波...", command=self._dlg_median_image)
        filter_menu.add_command(label="高斯模糊...", command=self._dlg_gauss_blur)
        filter_menu.add_command(label="雙邊濾波...", command=self._dlg_bilateral_filter)
        filter_menu.add_command(label="銳化...", command=self._dlg_sharpen)
        filter_menu.add_command(label="強調...", command=self._dlg_emphasize)
        filter_menu.add_command(label="Laplacian", command=lambda: self._apply_halcon_op("laplace_filter"))
        halcon_menu.add_cascade(label="\u6ffe\u6ce2", menu=filter_menu)

        edge_menu2 = tk.Menu(halcon_menu, tearoff=0, bg="#2b2b2b", fg="#e0e0e0",
                             activebackground="#3a3a5c", activeforeground="#ffffff")
        edge_menu2.add_command(label="Canny 邊緣...", command=self._dlg_canny)
        edge_menu2.add_command(label="Sobel", command=lambda: self._apply_halcon_op("sobel_filter"))
        edge_menu2.add_command(label="Prewitt", command=lambda: self._apply_halcon_op("prewitt_filter"))
        edge_menu2.add_command(label="零交叉", command=lambda: self._apply_halcon_op("zero_crossing"))
        edge_menu2.add_command(label="高斯導數...", command=self._dlg_derivative_gauss)
        halcon_menu.add_cascade(label="\u908a\u7de3", menu=edge_menu2)

        morph_menu2 = tk.Menu(halcon_menu, tearoff=0, bg="#2b2b2b", fg="#e0e0e0",
                              activebackground="#3a3a5c", activeforeground="#ffffff")
        morph_menu2.add_command(label="灰度侵蝕...", command=self._dlg_gray_erosion)
        morph_menu2.add_command(label="灰度膨脹...", command=self._dlg_gray_dilation)
        morph_menu2.add_command(label="灰度開運算...", command=self._dlg_gray_opening)
        morph_menu2.add_command(label="灰度閉運算...", command=self._dlg_gray_closing)
        morph_menu2.add_command(label="Top-hat...", command=self._dlg_top_hat)
        morph_menu2.add_command(label="Bottom-hat...", command=self._dlg_bottom_hat)
        morph_menu2.add_separator()
        morph_menu2.add_command(label="\u52d5\u614b\u95be\u503c\u5206\u5272...", command=self._open_dyn_threshold_dialog)
        halcon_menu.add_cascade(label="\u5f62\u614b\u5b78", menu=morph_menu2)

        geom_menu = tk.Menu(halcon_menu, tearoff=0, bg="#2b2b2b", fg="#e0e0e0",
                            activebackground="#3a3a5c", activeforeground="#ffffff")
        for label, op in [("\u65cb\u8f49 90\u00b0", "rotate_90"), ("\u65cb\u8f49 180\u00b0", "rotate_180"),
                          ("\u65cb\u8f49 270\u00b0", "rotate_270"),
                          ("\u6c34\u5e73\u93e1\u50cf", "mirror_h"), ("\u5782\u76f4\u93e1\u50cf", "mirror_v"),
                          ("\u7e2e\u653e 50%", "zoom_50"), ("\u7e2e\u653e 200%", "zoom_200")]:
            geom_menu.add_command(label=label, command=lambda o=op: self._apply_halcon_op(o))
        halcon_menu.add_cascade(label="\u5e7e\u4f55", menu=geom_menu)

        color_menu = tk.Menu(halcon_menu, tearoff=0, bg="#2b2b2b", fg="#e0e0e0",
                             activebackground="#3a3a5c", activeforeground="#ffffff")
        for label, op in [("\u8f49\u7070\u968e", "rgb_to_gray"), ("\u8f49 HSV", "rgb_to_hsv"),
                          ("轉 HLS", "rgb_to_hls"),
                          ("\u76f4\u65b9\u5716\u5747\u8861", "histogram_eq_halcon"),
                          ("\u53cd\u8272", "invert_image"), ("\u5149\u7167\u6821\u6b63", "illuminate")]:
            color_menu.add_command(label=label, command=lambda o=op: self._apply_halcon_op(o))
        color_menu.add_separator()
        color_menu.add_command(label="CLAHE...", command=self._dlg_clahe)
        halcon_menu.add_cascade(label="\u8272\u5f69", menu=color_menu)

        gray_trans_menu = tk.Menu(halcon_menu, tearoff=0, bg="#2b2b2b", fg="#e0e0e0",
                                  activebackground="#3a3a5c", activeforeground="#ffffff")
        gray_trans_menu.add_command(label="亮度/對比度調整...", command=self._dlg_scale_image)
        gray_trans_menu.add_command(label="\u7d55\u5c0d\u503c", command=lambda: self._apply_halcon_op("abs_image"))
        gray_trans_menu.add_command(label="\u53cd\u8272", command=lambda: self._apply_halcon_op("invert_image"))
        gray_trans_menu.add_command(label="對數變換...", command=self._dlg_log_image)
        gray_trans_menu.add_command(label="指數變換...", command=self._dlg_exp_image)
        gray_trans_menu.add_command(label="Gamma 校正...", command=self._dlg_gamma_image)
        halcon_menu.add_cascade(label="灰度變換", menu=gray_trans_menu)

        img_op_menu = tk.Menu(halcon_menu, tearoff=0, bg="#2b2b2b", fg="#e0e0e0",
                              activebackground="#3a3a5c", activeforeground="#ffffff")
        img_op_menu.add_command(label="圖像相減...", command=self._open_subtract_dialog)
        img_op_menu.add_command(label="絕對差分", command=lambda: self._apply_halcon_op("abs_diff_image"))
        halcon_menu.add_cascade(label="圖像運算", menu=img_op_menu)

        freq_menu = tk.Menu(halcon_menu, tearoff=0, bg="#2b2b2b", fg="#e0e0e0",
                            activebackground="#3a3a5c", activeforeground="#ffffff")
        freq_menu.add_command(label="FFT 頻譜...", command=self._dlg_fft)
        freq_menu.add_command(label="低通濾波...", command=lambda: self._dlg_freq_filter("lowpass"))
        freq_menu.add_command(label="高通濾波...", command=lambda: self._dlg_freq_filter("highpass"))
        halcon_menu.add_cascade(label="頻域處理", menu=freq_menu)

        texture_menu = tk.Menu(halcon_menu, tearoff=0, bg="#2b2b2b", fg="#e0e0e0",
                               activebackground="#3a3a5c", activeforeground="#ffffff")
        for label, op in [("\u71b5\u5f71\u50cf", "entropy_image"), ("\u6a19\u6e96\u5dee\u5f71\u50cf", "deviation_image"),
                          ("\u5c40\u90e8\u6700\u5c0f", "local_min"), ("\u5c40\u90e8\u6700\u5927", "local_max")]:
            texture_menu.add_command(label=label, command=lambda o=op: self._apply_halcon_op(o))
        halcon_menu.add_cascade(label="\u7d0b\u7406", menu=texture_menu)

        barcode_menu = tk.Menu(halcon_menu, tearoff=0, bg="#2b2b2b", fg="#e0e0e0",
                               activebackground="#3a3a5c", activeforeground="#ffffff")
        for label, op in [("\u689d\u78bc\u5075\u6e2c", "find_barcode"), ("QR Code", "find_qrcode"),
                          ("DataMatrix", "find_datamatrix")]:
            barcode_menu.add_command(label=label, command=lambda o=op: self._apply_halcon_op(o))
        halcon_menu.add_cascade(label="\u689d\u78bc", menu=barcode_menu)

        # 分割
        seg_menu = tk.Menu(halcon_menu, tearoff=0, bg="#2b2b2b", fg="#e0e0e0",
                           activebackground="#3a3a5c", activeforeground="#ffffff")
        seg_menu.add_command(label="分水嶺...", command=self._dlg_watersheds)
        seg_menu.add_command(label="距離變換...", command=self._dlg_distance_transform)
        seg_menu.add_command(label="骨架化", command=lambda: self._apply_halcon_op("skeleton"))
        halcon_menu.add_cascade(label="分割", menu=seg_menu)

        # 特徵點
        feat_menu = tk.Menu(halcon_menu, tearoff=0, bg="#2b2b2b", fg="#e0e0e0",
                            activebackground="#3a3a5c", activeforeground="#ffffff")
        feat_menu.add_command(label="Harris 角點...", command=self._dlg_points_harris)
        feat_menu.add_command(label="Shi-Tomasi 特徵點...", command=self._dlg_points_shi_tomasi)
        halcon_menu.add_cascade(label="特徵點", menu=feat_menu)

        # 直線/圓偵測
        hough_menu = tk.Menu(halcon_menu, tearoff=0, bg="#2b2b2b", fg="#e0e0e0",
                             activebackground="#3a3a5c", activeforeground="#ffffff")
        hough_menu.add_command(label="Hough 直線...", command=self._dlg_hough_lines)
        hough_menu.add_command(label="Hough 圓...", command=self._dlg_hough_circles)
        halcon_menu.add_cascade(label="直線/圓偵測", menu=hough_menu)

        # 相機
        camera_menu = tk.Menu(halcon_menu, tearoff=0, bg="#2b2b2b", fg="#e0e0e0",
                              activebackground="#3a3a5c", activeforeground="#ffffff")
        camera_menu.add_command(label="擷取影像", command=lambda: self._apply_halcon_op("grab_image"))
        halcon_menu.add_cascade(label="相機", menu=camera_menu)

        halcon_menu.add_separator()
        halcon_menu.add_command(label="\u8173\u672c\u7de8\u8f2f\u5668", command=self._toggle_script_editor, accelerator="F8")
        menubar.add_cascade(label="HALCON", menu=halcon_menu)

        # -- Model --
        model_menu = tk.Menu(menubar, tearoff=0, bg="#2b2b2b", fg="#e0e0e0",
                             activebackground="#3a3a5c", activeforeground="#ffffff")
        model_menu.add_command(label="\u8a13\u7df4\u65b0\u6a21\u578b...", command=self._cmd_train, accelerator="F6")
        model_menu.add_command(label="\u8f09\u5165 Checkpoint...", command=self._cmd_load_model)
        model_menu.add_command(label="\u5132\u5b58 Checkpoint...", command=self._cmd_save_checkpoint)
        model_menu.add_separator()
        model_menu.add_command(label="\u6a21\u578b\u8cc7\u8a0a...", command=self._cmd_model_info)
        model_menu.add_command(label="\u8a08\u7b97\u95be\u503c", command=self._cmd_compute_threshold)
        menubar.add_cascade(label="\u6a21\u578b", menu=model_menu)

        # -- View --
        view_menu = tk.Menu(menubar, tearoff=0, bg="#2b2b2b", fg="#e0e0e0",
                            activebackground="#3a3a5c", activeforeground="#ffffff")
        view_menu.add_command(label="\u7e2e\u653e\u81f3\u7a97\u53e3", command=self._cmd_fit, accelerator="Space")
        view_menu.add_command(label="\u653e\u5927", command=self._cmd_zoom_in, accelerator="+")
        view_menu.add_command(label="\u7e2e\u5c0f", command=self._cmd_zoom_out, accelerator="-")
        view_menu.add_command(label="1:1 \u539f\u59cb\u5927\u5c0f", command=self._cmd_actual_size)
        view_menu.add_separator()
        view_menu.add_command(label="\u76f4\u65b9\u5716...", command=self._cmd_histogram)
        view_menu.add_command(label="\u640d\u5931\u66f2\u7dda", command=self._cmd_toggle_loss_curve)
        view_menu.add_command(label="\u91cd\u5efa\u5c0d\u6bd4...", command=self._cmd_reconstruction_compare)
        view_menu.add_command(label="\u5716\u7247\u6bd4\u5c0d...", command=self._cmd_compare_steps)
        view_menu.add_command(label="批次圖片比對...", command=self._cmd_batch_compare_steps)
        menubar.add_cascade(label="\u6aa2\u8996", menu=view_menu)

        # -- Help --
        help_menu = tk.Menu(menubar, tearoff=0, bg="#2b2b2b", fg="#e0e0e0",
                            activebackground="#3a3a5c", activeforeground="#ffffff")
        help_menu.add_command(label="\u5feb\u6377\u9375...", command=self._cmd_shortcuts)
        help_menu.add_command(label="\u95dc\u65bc...", command=self._cmd_about)
        menubar.add_cascade(label="\u5e6b\u52a9", menu=help_menu)

        self.configure(menu=menubar)

    # ==================================================================
    # Toolbar
    # ==================================================================

    def _build_toolbar(self) -> None:
        callbacks = {
            "open": self._cmd_open_image,
            "save": self._cmd_save_image,
            "undo": self._cmd_undo,
            "redo": self._cmd_redo,
            "fit": self._cmd_fit,
            "zoom_in": self._cmd_zoom_in,
            "zoom_out": self._cmd_zoom_out,
            "actual_size": self._cmd_actual_size,
            "train": self._cmd_train,
            "load_model": self._cmd_load_model,
            "inspect": self._cmd_inspect_single,
            "batch": self._cmd_batch_inspect,
            "toggle_pixel_inspector": self._toggle_pixel_inspector,
            "threshold": self._open_threshold_dialog,
            "blob_analysis": self._open_blob_analysis,
            "toggle_script_editor": self._toggle_script_editor,
            "tool_pixel_inspect": self._cmd_tool_pixel_inspect,
            "tool_region_select": self._cmd_tool_region_select,
            "compare": self._cmd_compare_steps,
            "grid": self._cmd_toggle_grid,
            "crosshair": self._cmd_toggle_crosshair,
        }
        self._toolbar = Toolbar(self, callbacks=callbacks)
        self._toolbar.pack(fill=tk.X, padx=2, pady=(2, 0))

    # ==================================================================
    # Main layout: three-panel PanedWindow
    # ==================================================================

    def _build_main_layout(self) -> None:
        self._paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        self._paned.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # -- Left: Pipeline panel --
        self._pipeline_panel = PipelinePanel(
            self._paned,
            on_step_selected=self._on_pipeline_step_selected,
            on_step_delete=self._on_pipeline_step_delete,
            on_step_export=self._on_pipeline_step_export,
        )
        self._paned.add(self._pipeline_panel, weight=0)

        # -- Centre: Image viewer --
        self._viewer = ImageViewer(
            self._paned,
            coord_callback=self._on_coord_update,
            zoom_callback=self._on_zoom_update,
            click_callback=self._on_pixel_click,
            region_callback=self._on_region_selected,
            show_loss_panel=True,
        )
        self._paned.add(self._viewer, weight=1)

        # -- Right: Properties + Operations --
        right_frame = ttk.Frame(self._paned)
        self._paned.add(right_frame, weight=0)

        self._props_panel = PropertiesPanel(right_frame)
        self._props_panel.set_region_highlight_callback(self._on_region_highlight)
        self._props_panel.set_region_remove_callback(self._on_region_remove)
        self._props_panel.pack(fill=tk.X, padx=2, pady=(2, 4))

        self._ops_panel = OperationsPanel(
            right_frame,
            on_apply=self._on_operation_applied,
            get_current_image=self._get_current_image,
        )
        self._ops_panel.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        self._history_panel = HistoryPanel(right_frame)
        self._history_panel.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # Initialise operations panel with config values
        self._ops_panel.set_params(
            threshold=self.config.anomaly_threshold_percentile,
            ssim_weight=self.config.ssim_weight,
            sigma=4.0,
            min_area=50,
        )

    # ==================================================================
    # Status bar
    # ==================================================================

    def _build_statusbar(self) -> None:
        self._statusbar = sb = ttk.Frame(self, relief=tk.SUNKEN)
        sb.pack(fill=tk.X, side=tk.BOTTOM, padx=2, pady=(0, 2))

        self._progress = ProgressManager(self, sb)

        self._status_var = tk.StringVar(value="\u5c31\u7dd2")
        self._status_label = ttk.Label(sb, textvariable=self._status_var, anchor=tk.W, width=40)
        self._status_label.pack(side=tk.LEFT, padx=4)

        ttk.Separator(sb, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=4)

        self._pos_var = tk.StringVar(value="\u4f4d\u7f6e: --, --")
        ttk.Label(sb, textvariable=self._pos_var, anchor=tk.W, width=20).pack(side=tk.LEFT, padx=4)

        ttk.Separator(sb, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=4)

        self._pixel_var = tk.StringVar(value="\u50cf\u7d20\u503c: --")
        ttk.Label(sb, textvariable=self._pixel_var, anchor=tk.W, width=30).pack(side=tk.LEFT, padx=4)

        ttk.Separator(sb, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=4)

        self._zoom_var = tk.StringVar(value="100%")
        ttk.Label(sb, textvariable=self._zoom_var, anchor=tk.E, width=8).pack(side=tk.RIGHT, padx=4)

        self._step_var = tk.StringVar(value="0 / 0")
        ttk.Label(sb, textvariable=self._step_var, anchor=tk.E, width=10).pack(side=tk.RIGHT, padx=4)

    def set_status(self, text: str) -> None:
        self._status_var.set(text)

    def set_status_success(self, text: str) -> None:
        """Set status text with a brief green flash to indicate success."""
        self._status_var.set(text)
        self._status_label.configure(style="Success.Status.TLabel")
        self.after(1500, lambda: self._status_label.configure(style="TLabel"))

    def _update_recent_menu(self) -> None:
        """Rebuild the recent-files cascade menu."""
        self._recent_menu.delete(0, tk.END)
        for path in self._recent_files:
            name = Path(path).name
            self._recent_menu.add_command(
                label=name,
                command=lambda p=path: self._load_image_to_pipeline(p),
            )

    def _show_error(self, context: str, exc: Exception) -> None:
        """Show a user-friendly error dialog via shared.error_dialog."""
        show_error(self, context, exc)

    def _show_shortcuts_dialog(self) -> None:
        """Show a Toplevel window listing all keyboard shortcuts."""
        dlg = tk.Toplevel(self)
        dlg.title("\u5feb\u6377\u9375\u4e00\u89bd")
        dlg.transient(self)
        dlg.grab_set()
        dlg.resizable(False, False)
        dlg.configure(bg="#2b2b2b")

        shortcuts = [
            ("Ctrl+O", "\u958b\u555f\u5716\u7247"),
            ("Ctrl+S", "\u5132\u5b58\u5716\u7247"),
            ("Ctrl+Z", "\u5fa9\u539f"),
            ("Ctrl+Y", "\u91cd\u505a"),
            ("Ctrl+I", "\u50cf\u7d20\u503c\u6aa2\u67e5\u5668\u8996\u7a97"),
            ("Ctrl+Shift+I", "\u50cf\u7d20\u6aa2\u67e5\u5de5\u5177"),
            ("Ctrl+Shift+R", "\u5340\u57df\u9078\u53d6\u5de5\u5177"),
            ("Ctrl+T", "\u95be\u503c\u5206\u5272"),
            ("Escape", "\u8fd4\u56de\u5e73\u79fb\u6a21\u5f0f"),
            ("Space", "\u7e2e\u653e\u81f3\u7a97\u53e3"),
            ("+/-", "\u653e\u5927/\u7e2e\u5c0f"),
            ("F1", "\u5feb\u6377\u9375\u4e00\u89bd"),
            ("F5", "\u6aa2\u6e2c\u5716\u7247"),
            ("F6", "\u8a13\u7df4\u6a21\u578b"),
            ("F8", "\u8173\u672c\u7de8\u8f2f\u5668"),
            ("F9", "\u57f7\u884c\u8173\u672c"),
            ("Delete", "\u522a\u9664\u6b65\u9a5f"),
        ]

        frame = ttk.Frame(dlg)
        frame.pack(padx=16, pady=12)
        for i, (key, desc) in enumerate(shortcuts):
            ttk.Label(frame, text=key, font=("Consolas", 10, "bold"), width=18, anchor=tk.W).grid(row=i, column=0, sticky=tk.W, pady=1)
            ttk.Label(frame, text=desc).grid(row=i, column=1, sticky=tk.W, padx=(8, 0), pady=1)

        ttk.Button(frame, text="\u95dc\u9589", command=dlg.destroy).grid(row=len(shortcuts), column=0, columnspan=2, pady=(12, 0))

        dlg.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() - dlg.winfo_width()) // 2
        y = self.winfo_y() + (self.winfo_height() - dlg.winfo_height()) // 2
        dlg.geometry(f"+{x}+{y}")

    # ==================================================================
    # Background task helper (prevents UI freeze)
    # ==================================================================

    def _run_in_bg(
        self,
        func,
        on_done=None,
        on_error=None,
        status_msg: str = "",
    ) -> None:
        """Run *func* in a background thread. Call *on_done(result)* or
        *on_error(exc)* back on the main thread when finished.

        This prevents heavy computations from freezing the Tkinter event loop.
        """
        if status_msg:
            self.set_status(status_msg)
            self.update_idletasks()

        self._progress.start_indeterminate()

        def _worker():
            try:
                result = func()
                def _finish(r=result):
                    self._progress.stop()
                    if on_done is not None:
                        on_done(r)
                self.after(0, _finish)
            except Exception as exc:
                def _fail(e=exc):
                    self._progress.stop()
                    if on_error is not None:
                        on_error(e)
                    else:
                        self._show_error("\u80cc\u666f\u4efb\u52d9\u5931\u6557", e)
                self.after(0, _fail)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    # ==================================================================
    # Keyboard shortcuts
    # ==================================================================

    def _bind_shortcuts(self) -> None:
        self.bind_all("<Control-o>", lambda e: self._cmd_open_image())
        self.bind_all("<Control-O>", lambda e: self._cmd_open_image())
        self.bind_all("<Control-s>", lambda e: self._cmd_save_image())
        self.bind_all("<Control-S>", lambda e: self._cmd_save_image())
        self.bind_all("<Control-z>", lambda e: self._cmd_undo())
        self.bind_all("<Control-Z>", lambda e: self._cmd_undo())
        self.bind_all("<Control-y>", lambda e: self._cmd_redo())
        self.bind_all("<Control-Y>", lambda e: self._cmd_redo())
        self.bind_all("<space>", lambda e: self._cmd_fit())
        self.bind_all("<plus>", lambda e: self._cmd_zoom_in())
        self.bind_all("<equal>", lambda e: self._cmd_zoom_in())
        self.bind_all("<minus>", lambda e: self._cmd_zoom_out())
        self.bind_all("<F5>", lambda e: self._cmd_inspect_single())
        self.bind_all("<F6>", lambda e: self._cmd_train())
        self.bind_all("<F8>", lambda e: self._toggle_script_editor())
        self.bind_all("<F9>", lambda e: self._run_script())
        self.bind_all("<Control-i>", lambda e: self._toggle_pixel_inspector())
        self.bind_all("<Control-I>", lambda e: self._toggle_pixel_inspector())
        self.bind_all("<Control-t>", lambda e: self._open_threshold_dialog())
        self.bind_all("<Control-T>", lambda e: self._open_threshold_dialog())
        self.bind_all("<Delete>", lambda e: self._cmd_delete_step())
        self.bind_all("<Control-Shift-i>", lambda e: self._cmd_tool_pixel_inspect(
            not self._toolbar.get_toggle_state("tool_pixel_inspect")))
        self.bind_all("<Control-Shift-I>", lambda e: self._cmd_tool_pixel_inspect(
            not self._toolbar.get_toggle_state("tool_pixel_inspect")))
        self.bind_all("<Control-Shift-r>", lambda e: self._cmd_tool_region_select(
            not self._toolbar.get_toggle_state("tool_region_select")))
        self.bind_all("<Control-Shift-R>", lambda e: self._cmd_tool_region_select(
            not self._toolbar.get_toggle_state("tool_region_select")))
        self.bind_all("<Escape>", lambda e: self._set_active_tool("pan"))
        self.bind_all("<F1>", lambda e: self._show_shortcuts_dialog())

    # ==================================================================
    # Callbacks from panels
    # ==================================================================

    def _on_coord_update(self, x: int, y: int, pixel_value: tuple) -> None:
        self._pos_var.set(f"\u4f4d\u7f6e: ({x}, {y})")
        if len(pixel_value) == 1:
            self._pixel_var.set(f"\u50cf\u7d20\u503c: [{pixel_value[0]}]")
        elif len(pixel_value) == 3:
            self._pixel_var.set(f"\u50cf\u7d20\u503c: [{pixel_value[0]}, {pixel_value[1]}, {pixel_value[2]}]")
        else:
            self._pixel_var.set("\u50cf\u7d20\u503c: --")

        # Live-update PixelInspector on hover
        if self._pixel_inspector is not None and self._pixel_inspector.winfo_exists():
            img = self._get_current_image()
            if img is not None:
                self._pixel_inspector.update_values(img, x, y)

    def _on_zoom_update(self, zoom_pct: float) -> None:
        self._zoom_var.set(f"{zoom_pct:.0f}%")

    # ------------------------------------------------------------------
    # Tool mode switching
    # ------------------------------------------------------------------

    _TOOL_ACTIONS = ["tool_pixel_inspect", "tool_region_select"]

    def _cmd_tool_pixel_inspect(self, state: bool = False) -> None:
        if state:
            self._set_active_tool("pixel_inspect")
        else:
            self._set_active_tool("pan")

    def _cmd_tool_region_select(self, state: bool = False) -> None:
        if state:
            self._set_active_tool("region_select")
        else:
            self._set_active_tool("pan")

    def _set_active_tool(self, tool_name: str) -> None:
        from dl_anomaly.gui.image_viewer import ActiveTool
        tool_map = {
            "pan": ActiveTool.PAN,
            "pixel_inspect": ActiveTool.PIXEL_INSPECT,
            "region_select": ActiveTool.REGION_SELECT,
        }
        tool = tool_map.get(tool_name, ActiveTool.PAN)
        self._viewer.set_active_tool(tool)

        # Radio-style: only one tool toggle active at a time
        active_action = {
            "pixel_inspect": "tool_pixel_inspect",
            "region_select": "tool_region_select",
        }.get(tool_name, "")
        self._toolbar.set_tool_exclusive(active_action, self._TOOL_ACTIONS)

        status_map = {
            "pan": "\u5c31\u7dd2",
            "pixel_inspect": "\u6a21\u5f0f: \u50cf\u7d20\u6aa2\u67e5 \u2014 \u9ede\u64ca\u5716\u7247\u4e0a\u7684\u50cf\u7d20\u67e5\u770b RGB \u503c",
            "region_select": "\u6a21\u5f0f: \u5340\u57df\u9078\u53d6 \u2014 \u62d6\u66f3\u9078\u53d6\u77e9\u5f62\u5340\u57df",
        }
        self.set_status(status_map.get(tool_name, "\u5c31\u7dd2"))

    # ------------------------------------------------------------------
    # Pixel click callback
    # ------------------------------------------------------------------

    def _on_pixel_click(self, x: int, y: int, pixel_value: tuple) -> None:
        """Called when user clicks a pixel in PIXEL_INSPECT tool mode."""
        if len(pixel_value) == 1:
            val_str = f"Gray={pixel_value[0]}"
        elif len(pixel_value) == 3:
            val_str = f"R={pixel_value[0]} G={pixel_value[1]} B={pixel_value[2]}"
        else:
            val_str = "--"
        self._pos_var.set(f"\u9ede\u64ca: ({x}, {y})")
        self._pixel_var.set(val_str)
        self.set_status(f"Pixel ({x}, {y}): {val_str}")

        # Update PixelInspector if open
        if self._pixel_inspector is not None and self._pixel_inspector.winfo_exists():
            img = self._get_current_image()
            if img is not None:
                self._pixel_inspector.update_values(img, x, y)

    # ------------------------------------------------------------------
    # Region selection callback
    # ------------------------------------------------------------------

    def _on_region_selected(self, x: int, y: int, w: int, h: int) -> None:
        """Called when user completes a region selection drag."""
        area = w * h
        self._pos_var.set(f"\u5340\u57df: ({x}, {y})")
        self._pixel_var.set(f"\u5c3a\u5bf8: {w} \u00d7 {h}  \u9762\u7a4d: {area}")
        self.set_status(
            f"\u5340\u57df\u9078\u53d6: \u539f\u9ede=({x},{y}) \u5c3a\u5bf8={w}\u00d7{h} \u9762\u7a4d={area}px"
        )

        # Update properties panel
        img = self._get_current_image()
        if img is not None:
            roi = img[y:y + h, x:x + w]
            roi_stats = {"mean": float(roi.mean())} if roi.size > 0 else None
            self._props_panel.update_measurement(x, y, w, h, roi_stats)

        # Build a Region object from the rectangular selection so that
        # downstream region operations (erosion, dilation, etc.) can use it.
        step = self._pipeline_panel.get_current_step()
        if step is not None:
            arr = step.array
            ih, iw = arr.shape[:2]
            from dl_anomaly.core.region import Region
            from dl_anomaly.core.region_ops import compute_region_properties
            mask = np.zeros((ih, iw), dtype=np.uint8)
            # Clamp to image bounds
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(iw, x + w)
            y2 = min(ih, y + h)
            mask[y1:y2, x1:x2] = 255
            num, labels = cv2.connectedComponents(mask, connectivity=8)
            labels = labels.astype(np.int32)
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if arr.ndim == 3 else arr
            props = compute_region_properties(labels, gray)
            self._current_region = Region(
                labels=labels,
                num_regions=num - 1,
                properties=props,
                source_image=gray,
                source_shape=(ih, iw),
            )

    def _on_pipeline_step_selected(self, index: int) -> None:
        step = self._pipeline_panel.get_step(index)
        if step is None:
            return
        self._viewer.set_image(step.array)
        self._current_region = step.region
        self._props_panel.update_properties(step.name, step.array, region=step.region)
        total = self._pipeline_panel.get_step_count()
        self._step_var.set(f"{index + 1} / {total}")

    def _on_pipeline_step_delete(self, index: int) -> None:
        self._pipeline_panel.delete_step(index)
        step = self._pipeline_panel.get_current_step()
        if step:
            self._viewer.set_image(step.array)
            self._props_panel.update_properties(step.name, step.array)
        else:
            self._viewer.clear()
            self._props_panel.clear()
        total = self._pipeline_panel.get_step_count()
        ci = self._pipeline_panel.get_current_index()
        self._step_var.set(f"{ci + 1} / {total}" if total > 0 else "0 / 0")

    def _on_pipeline_step_export(self, index: int) -> None:
        step = self._pipeline_panel.get_step(index)
        if step is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp")],
            initialfile=f"{step.name}.png",
        )
        if path:
            if step.array.ndim == 2:
                img = Image.fromarray(step.array, mode="L")
            else:
                img = Image.fromarray(step.array, mode="RGB")
            img.save(path)
            self.set_status(f"\u5df2\u5132\u5b58: {path}")

    def _on_operation_applied(self, op_name: str, result: Optional[np.ndarray]) -> None:
        """Called by OperationsPanel when an operation produces a result."""
        if op_name == "apply_params":
            # Re-run the full inspection pipeline with new parameters
            self._rerun_inspection_with_params()
            return

        if result is not None:
            _quick_op_map = {
                "\u7070\u968e": "grayscale",
                "\u6a21\u7cca": "blur",
                "\u908a\u7de3": "edge",
                "\u76f4\u65b9\u5716\u5747\u8861": "histeq",
                "\u53cd\u8272": "invert",
            }
            op_key = _quick_op_map.get(op_name)
            if op_key is not None:
                meta = {"category": "quick_op", "op": op_key, "params": {}}
                if op_key == "blur":
                    meta["params"]["sigma"] = self._ops_panel.get_sigma()
            else:
                meta = None
            self._add_pipeline_step(op_name, result, op_meta=meta)

    def _get_current_image(self) -> Optional[np.ndarray]:
        step = self._pipeline_panel.get_current_step()
        if step:
            return step.array.copy()
        return None

    # ==================================================================
    # Pipeline helpers
    # ==================================================================

    def _add_pipeline_step(self, name: str, array: np.ndarray, op_meta=None) -> None:
        """Add a new step to the pipeline and display it."""
        # Record undo point
        ci = self._pipeline_panel.get_current_index()
        if ci >= 0:
            self._undo_stack.append(ci)
        self._redo_stack.clear()

        self._pipeline_panel.add_step(name, array, select=True, op_meta=op_meta)

    def _load_image_to_pipeline(self, path: str) -> None:
        """Load an image file and add it as the first pipeline step."""
        from PIL import Image as PILImage

        img = PILImage.open(path)
        # Proportionally resize so the longest side is at most 1080 px
        MAX_SIDE = 1080
        w, h = img.size
        if max(w, h) > MAX_SIDE:
            scale = MAX_SIDE / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), PILImage.LANCZOS)
        if img.mode == "L":
            array = np.array(img)
        elif img.mode in ("RGB", "RGBA"):
            array = np.array(img.convert("RGB"))
        else:
            array = np.array(img.convert("RGB"))

        self._current_image_path = path
        name = Path(path).stem
        op_meta = {"category": "source"}
        if not self._initial_loaded:
            op_meta["initial"] = True
        self._pipeline_panel.add_step(
            f"\u539f\u5716: {name}", array, select=True,
            op_meta=op_meta,
        )
        if not self._initial_loaded:
            self._initial_loaded = True
        self.set_status(f"\u5df2\u8f09\u5165: {Path(path).name}")

        # Update recent files
        if path not in self._recent_files:
            self._recent_files.insert(0, path)
            self._recent_files = self._recent_files[:10]
        self._app_state.add_recent_file(path)
        self._update_recent_menu()

    # ==================================================================
    # File commands
    # ==================================================================

    def _cmd_open_image(self) -> None:
        path = filedialog.askopenfilename(
            title="\u958b\u555f\u5716\u7247",
            filetypes=[
                ("\u5716\u7247\u6a94\u6848", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("\u6240\u6709\u6a94\u6848", "*.*"),
            ],
        )
        if path:
            try:
                self._load_image_to_pipeline(path)
            except Exception as exc:
                self._show_error("\u7121\u6cd5\u8f09\u5165\u5716\u7247", exc)

    def _cmd_open_dir(self) -> None:
        d = filedialog.askdirectory(title="\u958b\u555f\u5716\u7247\u8cc7\u6599\u593e")
        if not d:
            return
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        paths = sorted(str(p) for p in Path(d).rglob("*") if p.is_file() and p.suffix.lower() in exts)
        if not paths:
            messagebox.showinfo("\u8cc7\u8a0a", "\u8cc7\u6599\u593e\u4e2d\u6c92\u6709\u627e\u5230\u5716\u7247\u6a94\u6848")
            return
        # Load ALL images sequentially into the pipeline
        self._load_all_images_to_pipeline(paths)

    def _load_all_images_to_pipeline(self, paths: list) -> None:
        """Load all images from *paths* as sequential pipeline steps."""
        from PIL import Image as PILImage

        self._current_image_path = paths[0]
        mark_initial = not self._initial_loaded

        MAX_SIDE = 1080
        for i, path in enumerate(paths):
            try:
                img = PILImage.open(path)
                w, h = img.size
                if max(w, h) > MAX_SIDE:
                    scale = MAX_SIDE / max(w, h)
                    new_w, new_h = int(w * scale), int(h * scale)
                    img = img.resize((new_w, new_h), PILImage.LANCZOS)
                if img.mode == "L":
                    array = np.array(img)
                elif img.mode in ("RGB", "RGBA"):
                    array = np.array(img.convert("RGB"))
                else:
                    array = np.array(img.convert("RGB"))
                name = Path(path).stem
                op_meta = {"category": "source"}
                if mark_initial:
                    op_meta["initial"] = True
                self._pipeline_panel.add_step(
                    f"\u539f\u5716: {name}", array, select=(i == 0),
                    op_meta=op_meta,
                )
            except Exception as exc:
                logger.warning("Failed to load %s: %s", path, exc)

        if mark_initial:
            self._initial_loaded = True
        self.set_status(f"\u5df2\u8f09\u5165 {len(paths)} \u5f35\u5716\u7247")

    def _cmd_save_image(self) -> None:
        step = self._pipeline_panel.get_current_step()
        if step is None:
            messagebox.showinfo("\u8cc7\u8a0a", "\u6c92\u6709\u53ef\u5132\u5b58\u7684\u5716\u7247")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp")],
            initialfile=f"{step.name}.png",
        )
        if path:
            if step.array.ndim == 2:
                img = Image.fromarray(step.array, mode="L")
            else:
                img = Image.fromarray(step.array, mode="RGB")
            img.save(path)
            self.set_status(f"\u5df2\u5132\u5b58: {path}")

    def _cmd_save_all(self) -> None:
        steps = self._pipeline_panel.get_all_steps()
        if not steps:
            messagebox.showinfo("\u8cc7\u8a0a", "\u6c92\u6709\u53ef\u5132\u5b58\u7684\u6b65\u9a5f")
            return
        d = filedialog.askdirectory(title="\u9078\u64c7\u8f38\u51fa\u8cc7\u6599\u593e")
        if not d:
            return
        for i, step in enumerate(steps):
            fname = f"{i + 1:02d}_{step.name}.png"
            # Sanitise filename
            fname = "".join(c if c.isalnum() or c in "._- " else "_" for c in fname)
            out_path = Path(d) / fname
            if step.array.ndim == 2:
                img = Image.fromarray(step.array, mode="L")
            else:
                img = Image.fromarray(step.array, mode="RGB")
            img.save(str(out_path))
        self.set_status(f"\u5df2\u5132\u5b58 {len(steps)} \u500b\u6b65\u9a5f\u81f3 {d}")

    # ==================================================================
    # Undo / Redo
    # ==================================================================

    def _cmd_undo(self) -> None:
        if not self._undo_stack:
            return
        current = self._pipeline_panel.get_current_index()
        self._redo_stack.append(current)
        target = self._undo_stack.pop()
        self._pipeline_panel.select_step(target)

    def _cmd_redo(self) -> None:
        if not self._redo_stack:
            return
        current = self._pipeline_panel.get_current_index()
        self._undo_stack.append(current)
        target = self._redo_stack.pop()
        self._pipeline_panel.select_step(target)

    # ==================================================================
    # View commands
    # ==================================================================

    def _cmd_fit(self) -> None:
        self._viewer.fit_to_window()

    def _cmd_zoom_in(self) -> None:
        self._viewer.zoom_in()

    def _cmd_zoom_out(self) -> None:
        self._viewer.zoom_out()

    def _cmd_actual_size(self) -> None:
        self._viewer.zoom_to_actual()

    def _cmd_toggle_grid(self, state: bool = False) -> None:
        self._viewer.set_grid(state)

    def _cmd_toggle_crosshair(self, state: bool = False) -> None:
        self._viewer.set_crosshair(state)

    def _cmd_histogram(self) -> None:
        step = self._pipeline_panel.get_current_step()
        if step is None:
            messagebox.showinfo("\u8cc7\u8a0a", "\u8acb\u5148\u8f09\u5165\u5716\u7247")
            return
        HistogramDialog(self, step.array, title_text=f"\u76f4\u65b9\u5716 - {step.name}")

    def _cmd_toggle_loss_curve(self) -> None:
        if self._viewer._loss_panel_visible:
            self._viewer.hide_loss_panel()
        else:
            self._viewer.show_loss_panel()

    def _cmd_reconstruction_compare(self) -> None:
        # Find "Original" and "Reconstruction" steps
        steps = self._pipeline_panel.get_all_steps()
        original = None
        reconstruction = None
        for s in steps:
            if "\u539f\u5716" in s.name or "Original" in s.name:
                original = s.array
            if "\u91cd\u5efa" in s.name or "Reconstruct" in s.name:
                reconstruction = s.array
        if original is None or reconstruction is None:
            messagebox.showinfo("\u8cc7\u8a0a", "\u9700\u8981\u539f\u5716\u548c\u91cd\u5efa\u5716\u624d\u80fd\u5c0d\u6bd4")
            return
        ReconstructionDialog(self, original, reconstruction)

    def _cmd_compare_steps(self) -> None:
        """Open a subtraction-based image comparison dialog with rules."""
        items = self._collect_compare_items()
        if len(items) < 2:
            messagebox.showinfo("資訊", "需要至少兩張可比對的圖片")
            return
        CompareDialog(self, items, fetch_steps_cb=self._collect_compare_items)

    def _collect_compare_items(self):
        """Return all pipeline steps as (index, name, array) list."""
        steps = self._pipeline_panel.get_all_steps()
        return [(i, s.name, s.array) for i, s in enumerate(steps)]

    def _cmd_batch_compare_steps(self) -> None:
        """Open the batch 1-to-N image comparison dialog."""
        items = self._collect_compare_items()
        if len(items) < 2:
            messagebox.showinfo("資訊", "需要至少兩張可比對的圖片")
            return
        BatchCompareDialog(self, items,
                           fetch_steps_cb=self._collect_compare_items)

    def _cmd_delete_step(self) -> None:
        idx = self._pipeline_panel.get_current_index()
        if idx >= 0:
            self._on_pipeline_step_delete(idx)

    # ==================================================================
    # Model commands
    # ==================================================================

    def _cmd_train(self) -> None:
        def on_complete(result: Dict[str, Any]) -> None:
            self.set_status(
                f"\u8a13\u7df4\u5b8c\u6210  best_val={result['best_val_loss']:.6f}  "
                f"threshold={result['threshold']:.6f}"
            )
            # Update loss curve in main viewer
            pipeline = dlg.get_pipeline()
            if pipeline:
                history = result.get("history", {})
                tl = history.get("train_loss", [])
                vl = history.get("val_loss", [])
                self._viewer.update_loss_plot(tl, vl, title="\u8a13\u7df4\u640d\u5931\u66f2\u7dda")
                self._viewer.show_loss_panel()
            # Auto-load the trained model for inspection
            ckpt = result.get("checkpoint_path")
            if ckpt and Path(ckpt).exists():
                self._load_inference_pipeline(ckpt)

        dlg = TrainingDialog(self, self.config, on_complete=on_complete)

    def _cmd_load_model(self) -> None:
        path = filedialog.askopenfilename(
            title="\u8f09\u5165 Checkpoint",
            filetypes=[("PyTorch Checkpoint", "*.pt")],
            initialdir=str(self.config.checkpoint_dir),
        )
        if path:
            self._load_inference_pipeline(path)

    def _load_inference_pipeline(self, path: str) -> None:
        try:
            from dl_anomaly.pipeline.inference import InferencePipeline
            self._inference_pipeline = InferencePipeline(path, device=self.config.device)

            # Also store model + state for info dialog
            from dl_anomaly.pipeline.trainer import TrainingPipeline
            self._model, _, self._model_state = TrainingPipeline.load_checkpoint(
                Path(path), self.config.device
            )

            self.set_status(f"\u6a21\u578b\u5df2\u8f09\u5165: {Path(path).name}")
        except Exception as exc:
            self._show_error("\u6a21\u578b\u8f09\u5165\u5931\u6557", exc)

    def _cmd_save_checkpoint(self) -> None:
        if self._model is None:
            messagebox.showinfo("\u8cc7\u8a0a", "\u6c92\u6709\u53ef\u5132\u5b58\u7684\u6a21\u578b")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".pt",
            filetypes=[("PyTorch Checkpoint", "*.pt")],
            initialdir=str(self.config.checkpoint_dir),
        )
        if not path:
            return
        import torch
        state = {
            "model_state_dict": self._model.state_dict(),
            "config": self.config.to_dict(),
            "epoch": self._model_state.get("epoch", 0) if self._model_state else 0,
            "loss": self._model_state.get("loss", 0) if self._model_state else 0,
        }
        if self._model_state and "threshold" in self._model_state:
            state["threshold"] = self._model_state["threshold"]
        torch.save(state, path)
        self.set_status(f"Checkpoint \u5df2\u5132\u5b58: {path}")

    def _cmd_model_info(self) -> None:
        if self._model is None:
            messagebox.showinfo("\u8cc7\u8a0a", "\u8acb\u5148\u8f09\u5165\u6a21\u578b")
            return
        ModelInfoDialog(self, self._model, self.config, self._model_state)

    def _cmd_compute_threshold(self) -> None:
        if self._inference_pipeline is None:
            messagebox.showinfo("\u8cc7\u8a0a", "\u8acb\u5148\u8f09\u5165\u6a21\u578b")
            return
        # Need training data — file dialog must stay on main thread
        d = filedialog.askdirectory(title="\u9078\u64c7\u8a13\u7df4\u5716\u7247\u8cc7\u6599\u593e (\u7528\u65bc\u8a08\u7b97\u95be\u503c)")
        if not d:
            return

        # Capture state for background thread
        scorer = self._inference_pipeline.scorer
        preprocessor = self._inference_pipeline.preprocessor
        model = self._inference_pipeline.model
        device = self._inference_pipeline.device
        ssim_w = self.config.ssim_weight
        pct = self._ops_panel.get_threshold_percentile()

        def _compute():
            import torch
            exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
            paths = sorted(p for p in Path(d).rglob("*") if p.is_file() and p.suffix.lower() in exts)
            scores = []
            for p in paths:
                tensor = preprocessor.load_and_preprocess(p)
                batch = tensor.unsqueeze(0).to(device)
                with torch.no_grad():
                    recon = model(batch)
                orig_np = preprocessor.inverse_normalize(tensor)
                recon_np = preprocessor.inverse_normalize(recon.squeeze(0))
                err = scorer.compute_combined_error(orig_np, recon_np, ssim_w)
                scores.append(scorer.compute_image_score(err))
            threshold = scorer.fit_threshold(scores, pct)
            return threshold, pct, len(scores)

        def _done(result):
            threshold, pct, n = result
            self.set_status(f"\u95be\u503c\u5df2\u8a2d\u5b9a: {threshold:.6f} (percentile={pct})")
            messagebox.showinfo("\u95be\u503c", f"\u7570\u5e38\u95be\u503c: {threshold:.6f}\n(percentile={pct}, n={n})")

        self._run_in_bg(_compute, on_done=_done, status_msg="\u8a08\u7b97\u95be\u503c\u4e2d...")

    # ==================================================================
    # Inspection commands
    # ==================================================================

    def _cmd_inspect_single(self) -> None:
        """Run inspection on a single image (or the currently loaded one)."""
        if self._inference_pipeline is None:
            messagebox.showinfo("\u8cc7\u8a0a", "\u8acb\u5148\u8f09\u5165\u6a21\u578b (Checkpoint)")
            return

        # If we have a current image path, use it; otherwise ask for one
        path = self._current_image_path
        if path is None or not Path(path).exists():
            path = filedialog.askopenfilename(
                title="\u9078\u64c7\u8981\u6aa2\u6e2c\u7684\u5716\u7247",
                filetypes=[("\u5716\u7247", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")],
            )
            if not path:
                return

        pipeline = self._inference_pipeline
        inspect_path = path

        def _compute():
            return pipeline.inspect_single(inspect_path)

        def _done(result):
            self._display_inspection_result(result, inspect_path)
            self.set_status(f"\u6aa2\u6e2c\u5b8c\u6210: {Path(inspect_path).name}")

        def _error(exc):
            self._show_error("\u6aa2\u6e2c\u5931\u6557", exc)
            self.set_status("\u5c31\u7dd2")

        self._run_in_bg(_compute, on_done=_done, on_error=_error,
                        status_msg=f"\u6aa2\u6e2c\u4e2d: {Path(path).name}")

    def _display_inspection_result(self, result, path: str) -> None:
        """Populate the pipeline with all inspection steps."""
        from dl_anomaly.pipeline.inference import InspectionResult

        name = Path(path).stem
        self._pipeline_panel.clear_all()
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._current_image_path = path

        # Ensure RGB for display
        orig = result.original
        if orig.ndim == 2:
            orig_rgb = cv2.cvtColor(orig, cv2.COLOR_GRAY2RGB)
        else:
            orig_rgb = orig

        recon = result.reconstruction
        if recon.ndim == 2:
            recon_rgb = cv2.cvtColor(recon, cv2.COLOR_GRAY2RGB)
        else:
            recon_rgb = recon

        # 1. Original
        self._pipeline_panel.add_step(f"\u539f\u5716: {name}", orig_rgb, select=False)

        # 2. Preprocessed (same as original for display)
        # Show the normalised version as a visual indicator
        self._pipeline_panel.add_step("\u524d\u8655\u7406", orig_rgb, select=False)

        # 3. Reconstruction
        self._pipeline_panel.add_step("\u91cd\u5efa", recon_rgb, select=False)

        # 4. Error Map (heatmap)
        heatmap = create_error_heatmap(result.error_map)
        self._pipeline_panel.add_step("\u8aa4\u5dee\u5716", heatmap, select=False)

        # 5. Smoothed error (re-compute with current sigma)
        sigma = self._ops_panel.get_sigma()
        scorer = AnomalyScorer(device=self.config.device)
        smoothed = scorer.create_anomaly_map(result.error_map, gaussian_sigma=sigma)
        smoothed_heatmap = create_error_heatmap(smoothed)
        self._pipeline_panel.add_step("\u5e73\u6ed1\u8aa4\u5dee", smoothed_heatmap, select=False)

        # 6. Defect Mask
        mask_rgb = cv2.cvtColor(result.defect_mask, cv2.COLOR_GRAY2RGB)
        self._pipeline_panel.add_step("\u7f3a\u9677\u906e\u7f69", mask_rgb, select=False)

        # 7. Overlay
        overlay = create_defect_overlay(orig_rgb, smoothed, threshold=0.5, alpha=0.4)
        self._pipeline_panel.add_step("\u7d50\u679c\u758a\u52a0", overlay, select=True)

        # Set overlay rects for defect regions
        rects = []
        for reg in result.defect_regions:
            rects.append({"bbox": reg["bbox"], "color": "#ff3333"})
        self._viewer.set_overlay_rects(rects)

        # Status
        label = "\u7f3a\u9677" if result.is_defective else "\u901a\u904e"
        self.set_status(
            f"\u6aa2\u6e2c\u7d50\u679c: {label}  |  \u5206\u6578: {result.anomaly_score:.6f}  |  "
            f"\u7f3a\u9677\u5340\u57df: {len(result.defect_regions)}"
        )

    def _cmd_batch_inspect(self) -> None:
        if self._inference_pipeline is None:
            messagebox.showinfo("\u8cc7\u8a0a", "\u8acb\u5148\u8f09\u5165\u6a21\u578b")
            return

        def on_result_selected(idx: int, result, path: str) -> None:
            self._display_inspection_result(result, path)

        BatchInspectDialog(
            self,
            self._inference_pipeline,
            on_result_selected=on_result_selected,
        )

    # ==================================================================
    # DL-specific operations (from menu)
    # ==================================================================

    def _cmd_run_autoencoder(self) -> None:
        """Feed current image through the autoencoder and add reconstruction as a step."""
        if self._inference_pipeline is None:
            messagebox.showinfo("\u8cc7\u8a0a", "\u8acb\u5148\u8f09\u5165\u6a21\u578b")
            return
        step = self._pipeline_panel.get_current_step()
        if step is None:
            messagebox.showinfo("\u8cc7\u8a0a", "\u8acb\u5148\u8f09\u5165\u5716\u7247")
            return

        # Capture state for background thread
        arr = step.array.copy()
        preprocessor = self._inference_pipeline.preprocessor
        model = self._inference_pipeline.model
        device = self._inference_pipeline.device

        def _compute():
            import torch
            from PIL import Image as PILImage
            if arr.ndim == 2:
                pil_img = PILImage.fromarray(arr, mode="L")
            else:
                pil_img = PILImage.fromarray(arr, mode="RGB")
            transform = preprocessor.get_transforms(augment=False)
            tensor = transform(pil_img).unsqueeze(0).to(device)
            with torch.no_grad():
                recon_tensor = model(tensor)
            recon_np = preprocessor.inverse_normalize(recon_tensor.squeeze(0))
            if recon_np.ndim == 2:
                recon_np = cv2.cvtColor(recon_np, cv2.COLOR_GRAY2RGB)
            return recon_np

        def _done(recon_np):
            self._add_pipeline_step("\u91cd\u5efa", recon_np)
            self.set_status("\u81ea\u52d5\u7de8\u78bc\u5668\u57f7\u884c\u5b8c\u6210")

        self._run_in_bg(_compute, on_done=_done, status_msg="\u57f7\u884c\u81ea\u52d5\u7de8\u78bc\u5668...")

    def _cmd_compute_error_map(self) -> None:
        """Compute pixel-wise error between 'Original' and 'Reconstruction' steps."""
        steps = self._pipeline_panel.get_all_steps()
        original = None
        reconstruction = None
        for s in steps:
            if "\u539f\u5716" in s.name or "Original" in s.name:
                original = s.array
            if "\u91cd\u5efa" in s.name or "Reconstruct" in s.name:
                reconstruction = s.array

        if original is None or reconstruction is None:
            messagebox.showinfo("\u8cc7\u8a0a", "\u9700\u8981\u539f\u5716\u548c\u91cd\u5efa\u5716\u624d\u80fd\u8a08\u7b97\u8aa4\u5dee\u5716")
            return

        # Capture values for background thread
        orig_copy = original.copy()
        recon_copy = reconstruction.copy()
        ssim_w = self._ops_panel.get_ssim_weight()
        dev = self.config.device

        def _compute():
            scorer = AnomalyScorer(device=dev)
            error_map = scorer.compute_combined_error(orig_copy, recon_copy, ssim_w)
            return create_error_heatmap(error_map)

        def _done(heatmap):
            self._add_pipeline_step("\u8aa4\u5dee\u5716", heatmap)
            self.set_status("\u8aa4\u5dee\u5716\u8a08\u7b97\u5b8c\u6210")

        self._run_in_bg(_compute, on_done=_done, status_msg="\u8a08\u7b97\u8aa4\u5dee\u5716...")

    def _cmd_apply_ssim(self) -> None:
        """Compute SSIM map between 'Original' and 'Reconstruction'."""
        steps = self._pipeline_panel.get_all_steps()
        original = None
        reconstruction = None
        for s in steps:
            if "\u539f\u5716" in s.name:
                original = s.array
            if "\u91cd\u5efa" in s.name:
                reconstruction = s.array

        if original is None or reconstruction is None:
            messagebox.showinfo("\u8cc7\u8a0a", "\u9700\u8981\u539f\u5716\u548c\u91cd\u5efa\u5716")
            return

        orig_copy = original.copy()
        recon_copy = reconstruction.copy()
        dev = self.config.device

        def _compute():
            scorer = AnomalyScorer(device=dev)
            ssim_map = scorer.compute_ssim_map(orig_copy, recon_copy)
            return create_error_heatmap(ssim_map)

        def _done(heatmap):
            self._add_pipeline_step("SSIM \u5716", heatmap)
            self.set_status("SSIM \u5716\u8a08\u7b97\u5b8c\u6210")

        self._run_in_bg(_compute, on_done=_done, status_msg="\u8a08\u7b97 SSIM \u5716...")

    def _cmd_apply_threshold_mask(self) -> None:
        """Threshold the error/smoothed map to produce a binary mask."""
        step = self._pipeline_panel.get_current_step()
        if step is None:
            return

        # Try to interpret current image as a single-channel map
        arr = step.array
        if arr.ndim == 3:
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        else:
            gray = arr

        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        self._add_pipeline_step("\u95be\u503c\u906e\u7f69", mask_rgb)

    def _rerun_inspection_with_params(self) -> None:
        """Re-run inspection on the current image with updated parameters."""
        if self._inference_pipeline is None or self._current_image_path is None:
            return
        path = self._current_image_path
        if not Path(path).exists():
            return

        # Update config from operations panel
        self.config.anomaly_threshold_percentile = self._ops_panel.get_threshold_percentile()
        self.config.ssim_weight = self._ops_panel.get_ssim_weight()

        # Capture all parameters for background thread
        preprocessor = self._inference_pipeline.preprocessor
        model = self._inference_pipeline.model
        device = self._inference_pipeline.device
        dev_str = self.config.device
        ssim_w = self._ops_panel.get_ssim_weight()
        sigma = self._ops_panel.get_sigma()
        min_area = self._ops_panel.get_min_area()
        threshold = self._inference_pipeline.scorer.threshold
        rerun_path = path

        def _compute():
            import torch
            scorer = AnomalyScorer(device=dev_str)

            tensor = preprocessor.load_and_preprocess(rerun_path)
            batch = tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                recon_batch = model(batch)

            orig_np = preprocessor.inverse_normalize(tensor)
            recon_np = preprocessor.inverse_normalize(recon_batch.squeeze(0))

            error_map = scorer.compute_combined_error(orig_np, recon_np, ssim_w)
            smoothed = scorer.create_anomaly_map(error_map, gaussian_sigma=sigma)
            score = scorer.compute_image_score(error_map)

            is_defective = False
            if threshold is not None:
                is_defective = score > threshold

            # Create mask
            map_u8 = (smoothed * 255).astype(np.uint8)
            _, mask = cv2.threshold(map_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

            # Filter by min area
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            filtered_mask = np.zeros_like(mask)
            regions = []
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area >= min_area:
                    filtered_mask[labels == i] = 255
                    x, y, w, h = stats[i, :4]
                    regions.append({"bbox": (int(x), int(y), int(w), int(h)), "area": int(area)})

            from dl_anomaly.pipeline.inference import InspectionResult
            return InspectionResult(
                original=orig_np,
                reconstruction=recon_np,
                error_map=smoothed,
                defect_mask=filtered_mask,
                anomaly_score=score,
                is_defective=is_defective,
                defect_regions=regions,
            )

        def _done(result):
            self._display_inspection_result(result, rerun_path)
            self.set_status(f"\u91cd\u65b0\u6aa2\u6e2c\u5b8c\u6210: {Path(rerun_path).name}")

        self._run_in_bg(_compute, on_done=_done, status_msg="\u91cd\u65b0\u6aa2\u6e2c\u4e2d...")

    # ==================================================================
    # Help
    # ==================================================================

    def _cmd_shortcuts(self) -> None:
        shortcuts = (
            "Ctrl+O         \u958b\u555f\u5716\u7247\n"
            "Ctrl+S         \u5132\u5b58\u5716\u7247\n"
            "Ctrl+Z         \u5fa9\u539f\n"
            "Ctrl+Y         \u91cd\u505a\n"
            "Ctrl+I         \u50cf\u7d20\u503c\u6aa2\u67e5\u5668\u8996\u7a97\n"
            "Ctrl+Shift+I   \u50cf\u7d20\u6aa2\u67e5\u5de5\u5177\n"
            "Ctrl+Shift+R   \u5340\u57df\u9078\u53d6\u5de5\u5177\n"
            "Escape         \u8fd4\u56de\u5e73\u79fb\u6a21\u5f0f\n"
            "Ctrl+T         \u95be\u503c\u5206\u5272\n"
            "Space          \u7e2e\u653e\u81f3\u7a97\u53e3\n"
            "+/-            \u653e\u5927/\u7e2e\u5c0f\n"
            "F5             \u6aa2\u6e2c\u5716\u7247\n"
            "F6             \u8a13\u7df4\u6a21\u578b\n"
            "F8             \u8173\u672c\u7de8\u8f2f\u5668\n"
            "F9             \u57f7\u884c\u8173\u672c\n"
            "Delete         \u522a\u9664\u6b65\u9a5f\n"
            "\n"
            "\u6ed1\u9f20\u5de6\u9375\u62d6\u66f3    \u5e73\u79fb\u5716\u7247 (\u5e73\u79fb\u6a21\u5f0f)\n"
            "\u6ed1\u9f20\u6efe\u8f2a        \u7e2e\u653e\n"
            "\u96d9\u64ca\u5de6\u9375        \u7e2e\u653e\u81f3\u7a97\u53e3\n"
            "\u53f3\u9375\u62d6\u66f3        \u7e2e\u653e\u81f3\u9078\u5340"
        )
        messagebox.showinfo("\u5feb\u6377\u9375", shortcuts)

    def _cmd_about(self) -> None:
        messagebox.showinfo(
            "\u95dc\u65bc",
            "DL \u7570\u5e38\u5075\u6e2c\u5668 - HALCON HDevelop Style\n\n"
            "\u6df1\u5ea6\u5b78\u7fd2\u7570\u5e38\u5075\u6e2c\u7cfb\u7d71\n"
            "\u4f7f\u7528 PyTorch \u5377\u7a4d\u81ea\u52d5\u7de8\u78bc\u5668\n\n"
            "\u67b6\u69cb: Encoder-Decoder + \u6b98\u5dee\u5340\u584a\n"
            "\u640d\u5931: MSE + SSIM \u7d44\u5408\u640d\u5931\n"
            "\u5f8c\u8655\u7406: \u9ad8\u65af\u5e73\u6ed1 + Otsu \u95be\u503c\n"
            "\u81ea\u9069\u61c9\u95be\u503c\u5206\u985e",
        )

    # ==================================================================
    # Recipe commands
    # ==================================================================

    def _cmd_save_recipe(self) -> None:
        """Export the current pipeline as a recipe JSON file."""
        from dl_anomaly.core.recipe import Recipe

        recipe = Recipe.from_pipeline(self._pipeline_panel)
        if not recipe.steps:
            messagebox.showinfo("\u8cc7\u8a0a", "\u76ee\u524d\u6c92\u6709\u53ef\u5132\u5b58\u7684\u64cd\u4f5c\u6b65\u9a5f\u3002")
            return
        path = filedialog.asksaveasfilename(
            title="\u5132\u5b58\u6d41\u7a0b",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("\u6240\u6709\u6a94\u6848", "*.*")],
        )
        if path:
            recipe.save(path)
            self.set_status(f"\u6d41\u7a0b\u5df2\u5132\u5b58: {Path(path).name} ({len(recipe.steps)} \u6b65)")

    def _cmd_load_and_apply_recipe(self) -> None:
        """Open the recipe-apply dialog for windowed multi-image recipe application."""
        def _add_step(name, arr, region=None):
            self._pipeline_panel.add_step(name, arr, select=False, region=region)
            total = self._pipeline_panel.get_step_count()
            self._pipeline_panel.select_step(total - 1)

        RecipeApplyDialog(self, add_step_cb=_add_step,
                          set_status_cb=self.set_status)

    def _cmd_batch_apply_recipe(self) -> None:
        """Batch-apply a recipe to a folder of images."""
        path = filedialog.askopenfilename(
            title="\u9078\u64c7\u6d41\u7a0b\u6a94\u6848",
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return

        from dl_anomaly.core.recipe import Recipe, replay_recipe

        try:
            recipe = Recipe.load(path)
        except Exception as exc:
            self._show_error("\u7121\u6cd5\u8f09\u5165\u6d41\u7a0b", exc)
            return

        input_dir = filedialog.askdirectory(title="\u9078\u64c7\u8f38\u5165\u5716\u7247\u8cc7\u6599\u593e")
        if not input_dir:
            return

        output_dir = filedialog.askdirectory(title="\u9078\u64c7\u8f38\u51fa\u8cc7\u6599\u593e")
        if not output_dir:
            return

        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        image_paths = sorted(
            str(p) for p in Path(input_dir).rglob("*")
            if p.is_file() and p.suffix.lower() in exts
        )
        if not image_paths:
            messagebox.showinfo("\u8cc7\u8a0a", "\u8cc7\u6599\u593e\u4e2d\u6c92\u6709\u5716\u7247\u6a94\u6848\u3002")
            return

        batch_recipe = recipe
        batch_in_dir = input_dir
        batch_out_dir = output_dir
        batch_paths = image_paths

        def _compute():
            from PIL import Image as PILImage
            MAX_SIDE = 1080
            processed = 0
            for img_path in batch_paths:
                try:
                    pil = PILImage.open(img_path)
                    w, h = pil.size
                    if max(w, h) > MAX_SIDE:
                        scale = MAX_SIDE / max(w, h)
                        pil = pil.resize(
                            (int(w * scale), int(h * scale)), PILImage.LANCZOS)
                    if pil.mode == "L":
                        arr = np.array(pil)
                    else:
                        arr = np.array(pil.convert("RGB"))

                    results = replay_recipe(batch_recipe, arr)
                    if results:
                        final_name, final_arr, _ = results[-1]
                        if final_arr.ndim == 2:
                            out_img = PILImage.fromarray(final_arr, mode="L")
                        else:
                            out_img = PILImage.fromarray(final_arr, mode="RGB")
                        stem = Path(img_path).stem
                        out_path = Path(batch_out_dir) / f"{stem}_result.png"
                        out_img.save(str(out_path))
                        processed += 1
                except Exception:
                    logger.exception("Batch: failed on %s", img_path)
            return processed

        def _done(count):
            self.set_status(f"\u6279\u6b21\u5957\u7528\u5b8c\u6210: {count}/{len(batch_paths)} \u5f35")
            messagebox.showinfo(
                "\u5b8c\u6210",
                f"\u6279\u6b21\u5957\u7528\u6d41\u7a0b\u5b8c\u6210\u3002\n"
                f"\u8655\u7406: {count}/{len(batch_paths)} \u5f35\n"
                f"\u8f38\u51fa: {batch_out_dir}",
            )

        self._run_in_bg(
            _compute, on_done=_done,
            status_msg=f"\u6279\u6b21\u5957\u7528\u6d41\u7a0b ({len(batch_paths)} \u5f35)...")

    # ==================================================================
    # Settings
    # ==================================================================

    def _cmd_settings(self) -> None:
        SettingsDialog(self, self.config)

    # ==================================================================
    # HALCON Region Operations
    # ==================================================================

    def _toggle_pixel_inspector(self, state: bool = None) -> None:
        """Toggle the pixel inspector window."""
        if self._pixel_inspector is not None and self._pixel_inspector.winfo_exists():
            self._pixel_inspector.destroy()
            self._pixel_inspector = None
        else:
            from dl_anomaly.gui.pixel_inspector import PixelInspector
            self._pixel_inspector = PixelInspector(self)

    def _open_threshold_dialog(self) -> None:
        """Open threshold segmentation dialog."""
        img = self._get_current_image()
        if img is None:
            messagebox.showwarning("\u8b66\u544a", "\u8acb\u5148\u8f09\u5165\u5716\u7247\u3002")
            return

        from dl_anomaly.gui.threshold_dialog import ThresholdDialog

        def on_accept(region, display_image, name):
            self._current_region = region
            # Try to parse min/max from name like "閾值 [100, 200]"
            import re
            m = re.search(r"\[(\d+),\s*(\d+)\]", name)
            if m:
                min_val, max_val = int(m.group(1)), int(m.group(2))
                op_meta = {"category": "threshold", "op": "manual",
                           "params": {"min_val": min_val, "max_val": max_val}}
            else:
                op_meta = {"category": "threshold", "op": "manual", "params": {}}
            self._pipeline_panel.add_step(name, display_image, region=region, op_meta=op_meta)
            self.set_status(f"\u95be\u503c\u5206\u5272\u5b8c\u6210: {region.num_regions} \u500b\u5340\u57df")

        ThresholdDialog(self, img, on_accept=on_accept)

    def _ensure_region(self) -> bool:
        """Ensure ``_current_region`` exists.

        If no region has been created yet, automatically run Otsu threshold
        on the current image to produce a meaningful initial region
        (matching HALCON behaviour where the default domain is the full image).
        """
        if self._current_region is not None:
            return True
        img = self._get_current_image()
        if img is None:
            messagebox.showwarning("\u8b66\u544a", "\u8acb\u5148\u8f09\u5165\u5716\u7247\u3002")
            return False

        from dl_anomaly.core.region_ops import binary_threshold
        self._current_region = binary_threshold(img, method="otsu")
        self._pipeline_panel.add_step(
            "Otsu \u95be\u503c (\u81ea\u52d5)",
            self._current_region.to_binary_mask(),
            region=self._current_region,
            op_meta={"category": "threshold", "op": "otsu", "params": {}},
        )
        return True

    def _auto_threshold_otsu(self) -> None:
        """Auto Otsu threshold segmentation."""
        img = self._get_current_image()
        if img is None:
            messagebox.showwarning("\u8b66\u544a", "\u8acb\u5148\u8f09\u5165\u5716\u7247\u3002")
            return

        def _compute():
            from dl_anomaly.core.region_ops import binary_threshold
            region = binary_threshold(img, method="otsu")
            return region

        def _done(region):
            self._current_region = region
            op_meta = {"category": "threshold", "op": "otsu", "params": {}}
            self._pipeline_panel.add_step(
                "Otsu \u95be\u503c", region.to_binary_mask(), region=region, op_meta=op_meta)
            self.set_status(f"Otsu \u95be\u503c: {region.num_regions} \u500b\u5340\u57df")

        self._run_in_bg(_compute, on_done=_done, status_msg="Otsu \u95be\u503c\u5206\u5272\u4e2d...")

    def _auto_threshold_adaptive(self) -> None:
        """Auto adaptive threshold segmentation."""
        img = self._get_current_image()
        if img is None:
            messagebox.showwarning("\u8b66\u544a", "\u8acb\u5148\u8f09\u5165\u5716\u7247\u3002")
            return

        def _compute():
            from dl_anomaly.core.region_ops import binary_threshold
            region = binary_threshold(img, method="adaptive")
            return region

        def _done(region):
            self._current_region = region
            op_meta = {"category": "threshold", "op": "adaptive", "params": {}}
            self._pipeline_panel.add_step(
                "\u81ea\u9069\u61c9\u95be\u503c", region.to_binary_mask(), region=region, op_meta=op_meta)
            self.set_status(f"\u81ea\u9069\u61c9\u95be\u503c: {region.num_regions} \u500b\u5340\u57df")

        self._run_in_bg(_compute, on_done=_done, status_msg="\u81ea\u9069\u61c9\u95be\u503c\u5206\u5272\u4e2d...")

    def _region_connection(self) -> None:
        """Connection operation."""
        if not self._ensure_region():
            return

        cur_region = self._current_region
        img = self._get_current_image()

        def _compute():
            from dl_anomaly.core.region_ops import connection
            return connection(cur_region)

        def _done(region):
            self._current_region = region
            # Show each component with a distinct gray level so user can see them
            n = max(region.num_regions, 1)
            vis = np.zeros(region.labels.shape, dtype=np.uint8)
            for i in range(1, n + 1):
                gray_val = int(55 + 200 * i / n)  # range 55-255, avoid pure black
                vis[region.labels == i] = gray_val
            op_meta = {"category": "region", "op": "connection", "params": {}}
            self._pipeline_panel.add_step(
                f"Connection ({region.num_regions})", vis, region=region, op_meta=op_meta)
            self.set_status(f"\u6253\u6563: {region.num_regions} \u500b\u7368\u7acb\u5340\u57df")

        self._run_in_bg(_compute, on_done=_done, status_msg="Connection...")

    def _region_fill_up(self) -> None:
        """Fill region holes with parameter dialog."""
        if not self._ensure_region():
            return

        dlg = tk.Toplevel(self)
        dlg.title("\u586b\u5145 (Fill Up)")
        dlg.configure(bg="#2b2b2b")
        dlg.resizable(False, False)
        dlg.grab_set()

        tk.Label(dlg, text="\u586b\u5145\u5340\u57df\u5167\u90e8\u7a7a\u6d1e",
                 bg="#2b2b2b", fg="#e0e0e0", font=("", 10, "bold")).grid(
            row=0, column=0, columnspan=2, pady=(10, 6), padx=10)

        tk.Label(dlg, text="\u6700\u5c0f\u7a7a\u6d1e\u9762\u7a4d:", bg="#2b2b2b", fg="#e0e0e0").grid(
            row=1, column=0, sticky="e", padx=(10, 4), pady=4)
        min_area_var = tk.StringVar(value="0")
        tk.Entry(dlg, textvariable=min_area_var, width=10,
                 bg="#3c3c3c", fg="#e0e0e0", insertbackground="#e0e0e0").grid(
            row=1, column=1, padx=(0, 10), pady=4)
        tk.Label(dlg, text="(0 = \u586b\u5145\u6240\u6709\u7a7a\u6d1e)", bg="#2b2b2b", fg="#999999",
                 font=("", 8)).grid(row=2, column=0, columnspan=2, padx=10)

        def _apply():
            try:
                min_area = int(min_area_var.get())
            except ValueError:
                messagebox.showwarning("\u8b66\u544a", "\u8acb\u8f38\u5165\u6709\u6548\u7684\u6578\u503c\u3002", parent=dlg)
                return
            dlg.destroy()
            cur_region = self._current_region

            def _compute():
                from dl_anomaly.core.region_ops import compute_region_properties
                from dl_anomaly.core.region import Region

                labels_in = cur_region.labels
                filled_labels = np.zeros_like(labels_in)
                new_id = 1

                for lbl in range(1, cur_region.num_regions + 1):
                    comp = ((labels_in == lbl) * 255).astype(np.uint8)
                    if not np.any(comp):
                        continue

                    # Flood-fill from borders to find enclosed holes
                    h, w = comp.shape
                    border = np.full((h + 2, w + 2), 255, dtype=np.uint8)
                    border[1:-1, 1:-1] = cv2.bitwise_not(comp)
                    # Seed (0,0)=255 → flood fills all 255s connected to
                    # border with 0, leaving only enclosed holes as 255.
                    cv2.floodFill(border, None, (0, 0), 0)
                    holes = border[1:-1, 1:-1]

                    if min_area > 0:
                        # Only fill holes whose area >= min_area
                        hole_n, hole_labels = cv2.connectedComponents(holes, connectivity=8)
                        for hi in range(1, hole_n):
                            hole_mask = (hole_labels == hi)
                            if hole_mask.sum() >= min_area:
                                comp[hole_mask] = 255
                    else:
                        # Fill all enclosed holes
                        comp = comp | holes

                    filled_labels[comp > 0] = new_id
                    new_id += 1

                num, labels_out = cv2.connectedComponents(
                    (filled_labels > 0).astype(np.uint8) * 255, connectivity=8)
                labels_out = labels_out.astype(np.int32)
                props = compute_region_properties(labels_out, cur_region.source_image)
                return Region(labels=labels_out, num_regions=num - 1, properties=props,
                              source_image=cur_region.source_image,
                              source_shape=cur_region.source_shape)

            def _done(region):
                self._current_region = region
                name = f"Fill Up (min={min_area})" if min_area > 0 else "Fill Up"
                op_meta = {"category": "region", "op": "fill_up",
                           "params": {"min_area": min_area}}
                self._pipeline_panel.add_step(
                    name, region.to_binary_mask(), region=region, op_meta=op_meta)
                self.set_status(f"\u586b\u5145\u5b8c\u6210: {region.num_regions} \u500b\u5340\u57df")

            self._run_in_bg(_compute, on_done=_done, status_msg="Fill Up...")

        btn_frame = tk.Frame(dlg, bg="#2b2b2b")
        btn_frame.grid(row=3, column=0, columnspan=2, pady=(6, 10))
        tk.Button(btn_frame, text="\u57f7\u884c", command=_apply, width=8,
                  bg="#3a3a5c", fg="#e0e0e0", activebackground="#4a4a6c",
                  activeforeground="#ffffff").pack(side="left", padx=4)
        tk.Button(btn_frame, text="\u53d6\u6d88", command=dlg.destroy, width=8,
                  bg="#3a3a5c", fg="#e0e0e0", activebackground="#4a4a6c",
                  activeforeground="#ffffff").pack(side="left", padx=4)

        dlg.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() - dlg.winfo_width()) // 2
        y = self.winfo_y() + (self.winfo_height() - dlg.winfo_height()) // 2
        dlg.geometry(f"+{x}+{y}")

    def _region_shape_trans(self, shape_type: str) -> None:
        """Shape transformation."""
        if not self._ensure_region():
            return

        cur_region = self._current_region
        img = self._get_current_image()

        def _compute():
            from dl_anomaly.core.region_ops import compute_region_properties
            from dl_anomaly.core.region import Region
            mask = cur_region.to_binary_mask()
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            result = np.zeros_like(mask)
            for cnt in contours:
                if shape_type == "convex":
                    hull = cv2.convexHull(cnt)
                    cv2.drawContours(result, [hull], -1, 255, -1)
                elif shape_type == "rectangle":
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(result, (x, y), (x + w, y + h), 255, -1)
                elif shape_type == "circle":
                    (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                    cv2.circle(result, (int(cx), int(cy)), int(radius), 255, -1)
                elif shape_type == "ellipse":
                    if len(cnt) >= 5:
                        ellipse = cv2.fitEllipse(cnt)
                        cv2.ellipse(result, ellipse, 255, -1)
            num, labels = cv2.connectedComponents(result, connectivity=8)
            labels = labels.astype(np.int32)
            props = compute_region_properties(labels, cur_region.source_image)
            return Region(labels=labels, num_regions=num - 1, properties=props,
                          source_image=cur_region.source_image,
                          source_shape=cur_region.source_shape)

        def _done(region):
            self._current_region = region
            self._pipeline_panel.add_step(f"Shape Trans ({shape_type})", region.to_binary_mask(), region=region)

        self._run_in_bg(_compute, on_done=_done, status_msg=f"Shape Trans ({shape_type})...")

    def _region_morphology(self, op: str) -> None:
        """Region morphology with parameter dialog."""
        if not self._ensure_region():
            return

        op_names = {
            "erosion": "\u5340\u57df\u4fb5\u8755",
            "dilation": "\u5340\u57df\u81a8\u8139",
            "opening": "\u5340\u57df\u958b\u904b\u7b97",
            "closing": "\u5340\u57df\u9589\u904b\u7b97",
        }
        title = op_names.get(op, op)

        dlg = tk.Toplevel(self)
        dlg.title(title)
        dlg.configure(bg="#2b2b2b")
        dlg.resizable(False, False)
        dlg.grab_set()

        tk.Label(dlg, text=title, bg="#2b2b2b", fg="#e0e0e0",
                 font=("", 10, "bold")).grid(
            row=0, column=0, columnspan=2, pady=(10, 6), padx=10)

        tk.Label(dlg, text="\u6838\u5927\u5c0f (ksize):", bg="#2b2b2b", fg="#e0e0e0").grid(
            row=1, column=0, sticky="e", padx=(10, 4), pady=4)
        ksize_var = tk.StringVar(value="5")
        tk.Entry(dlg, textvariable=ksize_var, width=8,
                 bg="#3c3c3c", fg="#e0e0e0", insertbackground="#e0e0e0").grid(
            row=1, column=1, padx=(0, 10), pady=4)

        tk.Label(dlg, text="\u6838\u5f62\u72c0:", bg="#2b2b2b", fg="#e0e0e0").grid(
            row=2, column=0, sticky="e", padx=(10, 4), pady=4)
        shape_var = tk.StringVar(value="ellipse")
        ttk.Combobox(dlg, textvariable=shape_var, width=10,
                     values=["ellipse", "rectangle", "cross"],
                     state="readonly").grid(row=2, column=1, padx=(0, 10), pady=4)

        tk.Label(dlg, text="\u8fed\u4ee3\u6b21\u6578:", bg="#2b2b2b", fg="#e0e0e0").grid(
            row=3, column=0, sticky="e", padx=(10, 4), pady=4)
        iter_var = tk.StringVar(value="1")
        tk.Entry(dlg, textvariable=iter_var, width=8,
                 bg="#3c3c3c", fg="#e0e0e0", insertbackground="#e0e0e0").grid(
            row=3, column=1, padx=(0, 10), pady=4)

        def _apply():
            try:
                ks = int(ksize_var.get())
                iters = int(iter_var.get())
            except ValueError:
                messagebox.showwarning("\u8b66\u544a", "\u8acb\u8f38\u5165\u6709\u6548\u7684\u6578\u503c\u3002", parent=dlg)
                return
            if ks < 1:
                ks = 1
            if ks % 2 == 0:
                ks += 1
            if iters < 1:
                iters = 1
            sh = shape_var.get()
            dlg.destroy()

            cur_region = self._current_region
            shape_map = {"ellipse": cv2.MORPH_ELLIPSE,
                         "rectangle": cv2.MORPH_RECT,
                         "cross": cv2.MORPH_CROSS}

            def _compute():
                from dl_anomaly.core.region_ops import compute_region_properties
                from dl_anomaly.core.region import Region
                mask = cur_region.to_binary_mask()
                morph = shape_map.get(sh, cv2.MORPH_ELLIPSE)
                kernel = cv2.getStructuringElement(morph, (ks, ks))
                if op == "erosion":
                    result = cv2.erode(mask, kernel, iterations=iters)
                elif op == "dilation":
                    result = cv2.dilate(mask, kernel, iterations=iters)
                elif op == "opening":
                    result = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iters)
                elif op == "closing":
                    result = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iters)
                else:
                    return None
                num, labels = cv2.connectedComponents(result, connectivity=8)
                labels = labels.astype(np.int32)
                props = compute_region_properties(labels, cur_region.source_image)
                region = Region(labels=labels, num_regions=num - 1, properties=props,
                              source_image=cur_region.source_image,
                              source_shape=cur_region.source_shape)
                name = f"{title} k={ks} {sh} x{iters}"
                return region, name

            def _done(result):
                if result is None:
                    return
                region, name = result
                self._current_region = region
                self._pipeline_panel.add_step(name, region.to_binary_mask(), region=region)
                self.set_status(f"{name}: {region.num_regions} \u500b\u5340\u57df")

            self._run_in_bg(_compute, on_done=_done, status_msg=f"{title}...")

        btn_frame = tk.Frame(dlg, bg="#2b2b2b")
        btn_frame.grid(row=4, column=0, columnspan=2, pady=(6, 10))
        tk.Button(btn_frame, text="\u57f7\u884c", command=_apply, width=8,
                  bg="#3a3a5c", fg="#e0e0e0", activebackground="#4a4a6c",
                  activeforeground="#ffffff").pack(side="left", padx=4)
        tk.Button(btn_frame, text="\u53d6\u6d88", command=dlg.destroy, width=8,
                  bg="#3a3a5c", fg="#e0e0e0", activebackground="#4a4a6c",
                  activeforeground="#ffffff").pack(side="left", padx=4)

        dlg.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() - dlg.winfo_width()) // 2
        y = self.winfo_y() + (self.winfo_height() - dlg.winfo_height()) // 2
        dlg.geometry(f"+{x}+{y}")

    def _open_region_filter(self) -> None:
        """Open region filter dialog."""
        if not self._ensure_region():
            return
        img = self._get_current_image()
        if img is None:
            return
        try:
            from dl_anomaly.gui.region_filter_dialog import RegionFilterDialog

            def on_accept(filtered_region, display_image, name, conditions_list=None):
                self._current_region = filtered_region
                op_meta = {"category": "region", "op": "select_shape",
                           "params": {"conditions": conditions_list or []}}
                self._pipeline_panel.add_step(
                    name, display_image, region=filtered_region, op_meta=op_meta)
                self.set_status(f"\u7be9\u9078\u5b8c\u6210: {filtered_region.num_regions} \u500b\u5340\u57df")

            RegionFilterDialog(self, self._current_region, img, on_accept=on_accept)
        except Exception as exc:
            self._show_error("\u958b\u555f\u7be9\u9078\u5c0d\u8a71\u6846\u5931\u6557", exc)

    def _region_select_gray(self) -> None:
        """Filter regions by gray value — opens a dialog for min/max input."""
        if not self._ensure_region():
            return
        img = self._get_current_image()
        if img is None:
            return

        # --- build dialog ---
        dlg = tk.Toplevel(self)
        dlg.title("\u4f9d\u7070\u5ea6\u7be9\u9078")
        dlg.configure(bg="#2b2b2b")
        dlg.resizable(False, False)
        dlg.grab_set()

        tk.Label(dlg, text="\u7be9\u9078\u5340\u57df\u7684\u5e73\u5747\u7070\u5ea6\u7bc4\u570d",
                 bg="#2b2b2b", fg="#e0e0e0", font=("", 10, "bold")).grid(
            row=0, column=0, columnspan=2, pady=(10, 6), padx=10)

        tk.Label(dlg, text="\u6700\u5c0f\u7070\u5ea6:", bg="#2b2b2b", fg="#e0e0e0").grid(
            row=1, column=0, sticky="e", padx=(10, 4), pady=4)
        min_var = tk.StringVar(value="0")
        min_entry = tk.Entry(dlg, textvariable=min_var, width=8,
                             bg="#3c3c3c", fg="#e0e0e0", insertbackground="#e0e0e0")
        min_entry.grid(row=1, column=1, padx=(0, 10), pady=4)

        tk.Label(dlg, text="\u6700\u5927\u7070\u5ea6:", bg="#2b2b2b", fg="#e0e0e0").grid(
            row=2, column=0, sticky="e", padx=(10, 4), pady=4)
        max_var = tk.StringVar(value="128")
        max_entry = tk.Entry(dlg, textvariable=max_var, width=8,
                             bg="#3c3c3c", fg="#e0e0e0", insertbackground="#e0e0e0")
        max_entry.grid(row=2, column=1, padx=(0, 10), pady=4)

        def _apply():
            try:
                mn = float(min_var.get())
                mx = float(max_var.get())
            except ValueError:
                messagebox.showwarning("\u8b66\u544a", "\u8acb\u8f38\u5165\u6709\u6548\u7684\u6578\u503c\u3002", parent=dlg)
                return
            dlg.destroy()

            cur_region = self._current_region
            src_img = img

            def _compute():
                from dl_anomaly.core.region_ops import select_shape
                return select_shape(cur_region, "mean_value", mn, mx)

            def _done(filtered):
                self._current_region = filtered
                self._pipeline_panel.add_step(
                    f"\u7070\u5ea6\u7be9\u9078 [{mn:.0f}-{mx:.0f}] ({filtered.num_regions})",
                    filtered.to_binary_mask(), region=filtered)
                self.set_status(f"\u7070\u5ea6\u7be9\u9078: {filtered.num_regions} \u500b\u5340\u57df")

            self._run_in_bg(_compute, on_done=_done, status_msg="\u7070\u5ea6\u7be9\u9078\u4e2d...")

        btn_frame = tk.Frame(dlg, bg="#2b2b2b")
        btn_frame.grid(row=3, column=0, columnspan=2, pady=(6, 10))
        tk.Button(btn_frame, text="\u78ba\u5b9a", command=_apply, width=8,
                  bg="#3a3a5c", fg="#e0e0e0", activebackground="#4a4a6c",
                  activeforeground="#ffffff").pack(side="left", padx=4)
        tk.Button(btn_frame, text="\u53d6\u6d88", command=dlg.destroy, width=8,
                  bg="#3a3a5c", fg="#e0e0e0", activebackground="#4a4a6c",
                  activeforeground="#ffffff").pack(side="left", padx=4)

        dlg.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() - dlg.winfo_width()) // 2
        y = self.winfo_y() + (self.winfo_height() - dlg.winfo_height()) // 2
        dlg.geometry(f"+{x}+{y}")
        min_entry.focus_set()

    def _region_sort(self) -> None:
        """Sort regions by area."""
        if not self._ensure_region():
            return
        try:
            import copy
            from dl_anomaly.core.region import Region
            sorted_props = sorted(self._current_region.properties, key=lambda p: p.area, reverse=True)
            new_labels = np.zeros_like(self._current_region.labels)
            new_props = []
            for new_idx, p in enumerate(sorted_props, 1):
                new_labels[self._current_region.labels == p.index] = new_idx
                p_copy = copy.copy(p)
                p_copy.index = new_idx
                new_props.append(p_copy)
            region = Region(labels=new_labels, num_regions=len(new_props), properties=new_props,
                          source_image=self._current_region.source_image,
                          source_shape=self._current_region.source_shape)
            self._current_region = region
            self._pipeline_panel.add_step("\u6392\u5e8f\u5340\u57df (\u9762\u7a4d)", region.to_binary_mask(), region=region)
        except Exception as exc:
            self._show_error("\u6392\u5e8f\u5931\u6557", exc)

    def _region_set_op(self, op: str) -> None:
        """Region set operations."""
        if not self._ensure_region():
            return
        try:
            if op == "complement":
                region = self._current_region.complement()
            else:
                messagebox.showinfo("\u63d0\u793a", f"\u5340\u57df{op}\u9700\u8981\u5169\u500b\u5340\u57df\uff0c\u76ee\u524d\u50c5\u6709\u4e00\u500b\u3002\n\u8acb\u5148\u5206\u5225\u7522\u751f\u5169\u500b\u5340\u57df\u6b65\u9a5f\u3002")
                return
            self._current_region = region
            self._pipeline_panel.add_step(f"\u5340\u57df\u88dc\u96c6 ({region.num_regions})", region.to_binary_mask(), region=region)
        except Exception as exc:
            self._show_error("\u96c6\u5408\u64cd\u4f5c\u5931\u6557", exc)

    def _open_blob_analysis(self) -> None:
        """Open blob analysis dialog."""
        img = self._get_current_image()
        if img is None:
            messagebox.showwarning("\u8b66\u544a", "\u8acb\u5148\u8f09\u5165\u5716\u7247\u3002")
            return
        try:
            from dl_anomaly.gui.blob_analysis import BlobAnalysisDialog

            def on_accept(steps):
                for step_name, step_img, step_region in steps:
                    self._pipeline_panel.add_step(step_name, step_img, region=step_region)
                if steps:
                    last_region = steps[-1][2]
                    if last_region is not None:
                        self._current_region = last_region
                self.set_status("Blob \u5206\u6790\u5b8c\u6210")

            BlobAnalysisDialog(self, img, on_accept=on_accept)
        except Exception as exc:
            self._show_error("Blob \u5206\u6790\u958b\u555f\u5931\u6557", exc)

    # ==================================================================
    # HALCON Operators
    # ==================================================================

    # ==================================================================
    # Reusable parameter dialog helper
    # ==================================================================

    def _open_param_dialog(self, title, params, on_apply):
        """Build a dark-themed parameter dialog from a specification list.

        Parameters
        ----------
        title : str
            Dialog window title.
        params : list of dict
            Each dict has keys: label, key, type ("int"|"float"|"combo"),
            default, and optionally values (for combo).
        on_apply : callable(dict)
            Called with {key: value, ...} when user clicks Apply.
        """
        dlg = tk.Toplevel(self)
        dlg.title(title)
        dlg.configure(bg="#2b2b2b")
        dlg.resizable(False, False)
        dlg.grab_set()

        tk.Label(dlg, text=title, bg="#2b2b2b", fg="#e0e0e0",
                 font=("", 10, "bold")).grid(
            row=0, column=0, columnspan=2, pady=(10, 6), padx=10)

        vars_map = {}
        for i, p in enumerate(params, start=1):
            tk.Label(dlg, text=p["label"], bg="#2b2b2b", fg="#e0e0e0").grid(
                row=i, column=0, sticky="e", padx=(10, 4), pady=4)
            var = tk.StringVar(value=str(p["default"]))
            vars_map[p["key"]] = (var, p["type"])
            if p["type"] == "combo":
                combo = ttk.Combobox(dlg, textvariable=var, width=14,
                                     values=p.get("values", []), state="readonly")
                combo.grid(row=i, column=1, padx=(0, 10), pady=4)
            else:
                tk.Entry(dlg, textvariable=var, width=10,
                         bg="#3c3c3c", fg="#e0e0e0",
                         insertbackground="#e0e0e0").grid(
                    row=i, column=1, padx=(0, 10), pady=4)

        def _do_apply():
            result = {}
            for key, (var, vtype) in vars_map.items():
                raw = var.get()
                try:
                    if vtype == "int":
                        result[key] = int(raw)
                    elif vtype == "float":
                        result[key] = float(raw)
                    else:
                        result[key] = raw
                except ValueError:
                    messagebox.showwarning("警告", f"參數 '{key}' 的值無效: {raw}", parent=dlg)
                    return
            dlg.destroy()
            on_apply(result)

        btn_frame = tk.Frame(dlg, bg="#2b2b2b")
        btn_frame.grid(row=len(params) + 1, column=0, columnspan=2, pady=(6, 10))
        tk.Button(btn_frame, text="執行", command=_do_apply, width=8,
                  bg="#3a3a5c", fg="#e0e0e0", activebackground="#4a4a6c",
                  activeforeground="#ffffff").pack(side="left", padx=4)
        tk.Button(btn_frame, text="取消", command=dlg.destroy, width=8,
                  bg="#3a3a5c", fg="#e0e0e0", activebackground="#4a4a6c",
                  activeforeground="#ffffff").pack(side="left", padx=4)

        dlg.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() - dlg.winfo_width()) // 2
        y = self.winfo_y() + (self.winfo_height() - dlg.winfo_height()) // 2
        dlg.geometry(f"+{x}+{y}")

    # ==================================================================
    # Parameterized HALCON filter dialogs
    # ==================================================================

    def _open_filter_dialog(self, op_label, params, apply_func):
        """Open a parameter dialog for a filter operation.

        Parameters
        ----------
        op_label : str
            Display name for the operation (used in pipeline step name).
        params : list of dict
            Parameter spec for _open_param_dialog.
        apply_func : callable(img, param_dict) -> ndarray
            Function that applies the filter and returns result image.
        """
        img = self._get_current_image()
        if img is None:
            messagebox.showwarning("警告", "請先載入圖片。")
            return

        def _on_apply(p):
            def _compute():
                return apply_func(img, p)

            def _done(result):
                param_str = " ".join(f"{k}={v}" for k, v in p.items())
                name = f"{op_label} {param_str}"
                op_meta = {"category": "dialog_op", "op": op_label, "params": dict(p)}
                self._add_pipeline_step(name, result, op_meta=op_meta)
                self.set_status(f"完成: {name}")

            self._run_in_bg(_compute, on_done=_done, status_msg=f"執行 {op_label}...")

        self._open_param_dialog(op_label, params, _on_apply)

    # ------------------------------------------------------------------
    # Individual filter dialog launchers
    # ------------------------------------------------------------------

    def _dlg_mean_image(self):
        from dl_anomaly.core import halcon_ops as hops
        self._open_filter_dialog("均值濾波", [
            {"label": "核大小 (ksize):", "key": "ksize", "type": "int", "default": 5},
        ], lambda img, p: hops.mean_image(img, p["ksize"]))

    def _dlg_median_image(self):
        from dl_anomaly.core import halcon_ops as hops
        self._open_filter_dialog("中值濾波", [
            {"label": "核大小 (ksize):", "key": "ksize", "type": "int", "default": 5},
        ], lambda img, p: hops.median_image(img, p["ksize"]))

    def _dlg_gauss_blur(self):
        from dl_anomaly.core import halcon_ops as hops
        self._open_filter_dialog("高斯模糊", [
            {"label": "核大小 (ksize):", "key": "ksize", "type": "int", "default": 5},
            {"label": "Sigma:", "key": "sigma", "type": "float", "default": 0},
        ], lambda img, p: (
            hops.gauss_blur(img, p["ksize"]) if p["sigma"] == 0
            else hops.gauss_filter(img, p["sigma"])
        ))

    def _dlg_bilateral_filter(self):
        from dl_anomaly.core import halcon_ops as hops
        self._open_filter_dialog("雙邊濾波", [
            {"label": "d:", "key": "d", "type": "int", "default": 9},
            {"label": "Sigma Color:", "key": "sigma_color", "type": "float", "default": 75},
            {"label": "Sigma Space:", "key": "sigma_space", "type": "float", "default": 75},
        ], lambda img, p: hops.bilateral_filter(img, p["d"], p["sigma_color"], p["sigma_space"]))

    def _dlg_sharpen(self):
        from dl_anomaly.core import halcon_ops as hops
        self._open_filter_dialog("銳化", [
            {"label": "強度 (amount):", "key": "amount", "type": "float", "default": 0.5},
        ], lambda img, p: hops.sharpen_image(img, p["amount"]))

    def _dlg_canny(self):
        from dl_anomaly.core import halcon_ops as hops
        self._open_filter_dialog("Canny 邊緣", [
            {"label": "低閾值 (low):", "key": "low", "type": "float", "default": 50},
            {"label": "高閾值 (high):", "key": "high", "type": "float", "default": 150},
            {"label": "Sigma:", "key": "sigma", "type": "float", "default": 1.0},
        ], lambda img, p: hops.edges_canny(img, p["low"], p["high"], p["sigma"]))

    def _dlg_gray_erosion(self):
        from dl_anomaly.core import halcon_ops as hops
        self._open_filter_dialog("灰度侵蝕", [
            {"label": "核大小 (ksize):", "key": "ksize", "type": "int", "default": 5},
        ], lambda img, p: hops.gray_erosion(img, p["ksize"]))

    def _dlg_gray_dilation(self):
        from dl_anomaly.core import halcon_ops as hops
        self._open_filter_dialog("灰度膨脹", [
            {"label": "核大小 (ksize):", "key": "ksize", "type": "int", "default": 5},
        ], lambda img, p: hops.gray_dilation(img, p["ksize"]))

    def _dlg_gray_opening(self):
        from dl_anomaly.core import halcon_ops as hops
        self._open_filter_dialog("灰度開運算", [
            {"label": "核大小 (ksize):", "key": "ksize", "type": "int", "default": 5},
        ], lambda img, p: hops.gray_opening(img, p["ksize"]))

    def _dlg_gray_closing(self):
        from dl_anomaly.core import halcon_ops as hops
        self._open_filter_dialog("灰度閉運算", [
            {"label": "核大小 (ksize):", "key": "ksize", "type": "int", "default": 5},
        ], lambda img, p: hops.gray_closing(img, p["ksize"]))

    def _dlg_top_hat(self):
        from dl_anomaly.core import halcon_ops as hops
        self._open_filter_dialog("Top-hat", [
            {"label": "核大小 (ksize):", "key": "ksize", "type": "int", "default": 9},
        ], lambda img, p: hops.top_hat(img, p["ksize"]))

    def _dlg_bottom_hat(self):
        from dl_anomaly.core import halcon_ops as hops
        self._open_filter_dialog("Bottom-hat", [
            {"label": "核大小 (ksize):", "key": "ksize", "type": "int", "default": 9},
        ], lambda img, p: hops.bottom_hat(img, p["ksize"]))

    def _dlg_emphasize(self):
        from dl_anomaly.core import halcon_ops as hops
        self._open_filter_dialog("強調", [
            {"label": "核大小 (ksize):", "key": "ksize", "type": "int", "default": 7},
            {"label": "強度 (factor):", "key": "factor", "type": "float", "default": 1.5},
        ], lambda img, p: hops.emphasize(img, p["ksize"], p["factor"]))

    def _dlg_scale_image(self):
        from dl_anomaly.core import halcon_ops as hops
        self._open_filter_dialog("亮度/對比度調整", [
            {"label": "乘數 (mult):", "key": "mult", "type": "float", "default": 1.0},
            {"label": "偏移 (add):", "key": "add", "type": "float", "default": 0},
        ], lambda img, p: hops.scale_image(img, p["mult"], p["add"]))

    def _dlg_log_image(self):
        from dl_anomaly.core import halcon_ops as hops
        self._open_filter_dialog("對數變換", [
            {"label": "底數 (base):", "key": "base", "type": "combo",
             "default": "e", "values": ["e", "2", "10"]},
        ], lambda img, p: hops.log_image(img, p["base"]))

    def _dlg_exp_image(self):
        from dl_anomaly.core import halcon_ops as hops
        self._open_filter_dialog("指數變換", [
            {"label": "底數 (base):", "key": "base", "type": "combo",
             "default": "e", "values": ["e", "2", "10"]},
        ], lambda img, p: hops.exp_image(img, p["base"]))

    def _dlg_gamma_image(self):
        from dl_anomaly.core import halcon_ops as hops
        self._open_filter_dialog("Gamma 校正", [
            {"label": "Gamma:", "key": "gamma", "type": "float", "default": 1.0},
        ], lambda img, p: hops.gamma_image(img, p["gamma"]))

    def _dlg_var_threshold(self):
        from dl_anomaly.core import halcon_ops as hops
        img = self._get_current_image()
        if img is None:
            messagebox.showwarning("警告", "請先載入圖片。")
            return

        def _on_apply(p):
            def _compute():
                return hops.var_threshold(
                    img, p["width"], p["height"], p["std_mult"],
                    p["abs_thresh"], p["light_dark"])

            def _done(result):
                from dl_anomaly.core.halcon_ops import Region
                name = f"可變閾值 w={p['width']} h={p['height']} s={p['std_mult']}"
                region = Region(mask=result)
                self._pipeline_panel.add_step(name, img, region=region)
                self._current_region = region
                overlay = img.copy()
                if overlay.ndim == 2:
                    overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
                overlay[result > 0] = (0, 0, 255)
                self._viewer.display_array(overlay)
                self.set_status(f"完成: {name}")

            self._run_in_bg(_compute, on_done=_done, status_msg="執行可變閾值...")

        self._open_param_dialog("可變閾值", [
            {"label": "寬度 (width):", "key": "width", "type": "int", "default": 15},
            {"label": "高度 (height):", "key": "height", "type": "int", "default": 15},
            {"label": "標準差倍數:", "key": "std_mult", "type": "float", "default": 0.2},
            {"label": "絕對閾值:", "key": "abs_thresh", "type": "float", "default": 2},
            {"label": "明暗模式:", "key": "light_dark", "type": "combo",
             "default": "dark", "values": ["dark", "light", "equal", "not_equal"]},
        ], _on_apply)

    def _dlg_local_threshold(self):
        from dl_anomaly.core import halcon_ops as hops
        img = self._get_current_image()
        if img is None:
            messagebox.showwarning("警告", "請先載入圖片。")
            return

        def _on_apply(p):
            def _compute():
                return hops.local_threshold(
                    img, p["method"], p["light_dark"], p["ksize"], p["scale"])

            def _done(result):
                from dl_anomaly.core.halcon_ops import Region
                name = f"局部閾值 k={p['ksize']} s={p['scale']}"
                region = Region(mask=result)
                self._pipeline_panel.add_step(name, img, region=region)
                self._current_region = region
                overlay = img.copy()
                if overlay.ndim == 2:
                    overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
                overlay[result > 0] = (0, 0, 255)
                self._viewer.display_array(overlay)
                self.set_status(f"完成: {name}")

            self._run_in_bg(_compute, on_done=_done, status_msg="執行局部閾值...")

        self._open_param_dialog("局部閾值", [
            {"label": "方法:", "key": "method", "type": "combo",
             "default": "adapted_std_deviation",
             "values": ["adapted_std_deviation", "mean"]},
            {"label": "明暗模式:", "key": "light_dark", "type": "combo",
             "default": "dark", "values": ["dark", "light", "not_equal"]},
            {"label": "核大小 (ksize):", "key": "ksize", "type": "int", "default": 15},
            {"label": "比例 (scale):", "key": "scale", "type": "float", "default": 0.2},
        ], _on_apply)

    def _dlg_fft(self):
        from dl_anomaly.core import halcon_ops as hops
        self._open_filter_dialog("FFT 頻譜", [],
                                 lambda img, p: hops.fft_image(img))

    def _dlg_freq_filter(self, filter_type="lowpass"):
        from dl_anomaly.core import halcon_ops as hops
        label = "低通濾波" if filter_type == "lowpass" else "高通濾波"
        self._open_filter_dialog(label, [
            {"label": "截止頻率 (sigma):", "key": "cutoff", "type": "float", "default": 30},
        ], lambda img, p: hops.freq_filter(img, filter_type, p["cutoff"]))

    def _dlg_derivative_gauss(self):
        from dl_anomaly.core import halcon_ops as hops
        self._open_filter_dialog("高斯導數", [
            {"label": "Sigma:", "key": "sigma", "type": "float", "default": 1.0},
            {"label": "方向:", "key": "component", "type": "combo",
             "default": "x", "values": ["x", "y"]},
        ], lambda img, p: hops.derivative_gauss(img, p["sigma"], p["component"]))

    def _dlg_watersheds(self):
        from dl_anomaly.core import halcon_ops as hops
        self._open_filter_dialog("分水嶺", [
            {"label": "標記閾值:", "key": "marker_thresh", "type": "float", "default": 0.5},
        ], lambda img, p: hops.watersheds(img, p["marker_thresh"]))

    def _dlg_distance_transform(self):
        from dl_anomaly.core import halcon_ops as hops
        self._open_filter_dialog("距離變換", [
            {"label": "方法:", "key": "method", "type": "combo",
             "default": "L2", "values": ["L1", "L2", "C"]},
        ], lambda img, p: hops.distance_transform(img, p["method"]))

    def _dlg_points_harris(self):
        from dl_anomaly.core import halcon_ops as hops
        self._open_filter_dialog("Harris 角點", [
            {"label": "區塊大小:", "key": "block_size", "type": "int", "default": 2},
            {"label": "ksize:", "key": "ksize", "type": "int", "default": 3},
            {"label": "k:", "key": "k", "type": "float", "default": 0.04},
            {"label": "閾值:", "key": "threshold", "type": "float", "default": 0.01},
        ], lambda img, p: hops.points_harris(img, p["block_size"], p["ksize"], p["k"], p["threshold"]))

    def _dlg_points_shi_tomasi(self):
        from dl_anomaly.core import halcon_ops as hops
        self._open_filter_dialog("Shi-Tomasi 特徵點", [
            {"label": "最大角點數:", "key": "max_corners", "type": "int", "default": 100},
            {"label": "品質:", "key": "quality", "type": "float", "default": 0.01},
            {"label": "最小距離:", "key": "min_distance", "type": "float", "default": 10},
        ], lambda img, p: hops.points_shi_tomasi(img, p["max_corners"], p["quality"], p["min_distance"]))

    def _dlg_hough_lines(self):
        from dl_anomaly.core import halcon_ops as hops
        self._open_filter_dialog("Hough 直線", [
            {"label": "rho (像素):", "key": "rho", "type": "float", "default": 1.0},
            {"label": "theta (度):", "key": "theta_deg", "type": "float", "default": 1.0},
            {"label": "閾值:", "key": "threshold", "type": "int", "default": 100},
        ], lambda img, p: hops.hough_lines(img, p["rho"], p["theta_deg"], p["threshold"]))

    def _dlg_hough_circles(self):
        from dl_anomaly.core import halcon_ops as hops
        self._open_filter_dialog("Hough 圓", [
            {"label": "dp:", "key": "dp", "type": "float", "default": 1.2},
            {"label": "最小距離:", "key": "min_dist", "type": "float", "default": 30},
            {"label": "param1:", "key": "param1", "type": "float", "default": 50},
            {"label": "param2:", "key": "param2", "type": "float", "default": 30},
        ], lambda img, p: hops.hough_circles(img, p["dp"], p["min_dist"], p["param1"], p["param2"]))

    def _dlg_clahe(self):
        from dl_anomaly.core import halcon_ops as hops
        self._open_filter_dialog("CLAHE", [
            {"label": "對比度限制:", "key": "clip_limit", "type": "float", "default": 2.0},
            {"label": "格子大小:", "key": "tile_size", "type": "int", "default": 8},
        ], lambda img, p: hops.clahe(img, p["clip_limit"], p["tile_size"]))

    def _dlg_estimate_noise(self):
        from dl_anomaly.core import halcon_ops as hops
        img = self._get_current_image()
        if img is None:
            messagebox.showwarning("警告", "請先載入圖片。")
            return
        sigma = hops.estimate_noise(img)
        messagebox.showinfo("噪聲估計", f"估計噪聲標準差 \u03c3 = {sigma:.4f}")

    def _dlg_gen_gauss_pyramid(self):
        from dl_anomaly.core import halcon_ops as hops
        self._open_filter_dialog("高斯金字塔", [
            {"label": "層數:", "key": "levels", "type": "int", "default": 4},
        ], lambda img, p: hops.gen_gauss_pyramid(img, p["levels"])[-1])

    # ==================================================================
    # Domain operations (reduce / crop)
    # ==================================================================

    def _cmd_reduce_domain(self):
        """Restrict current image to the domain of the current region."""
        if self._current_region is None:
            messagebox.showwarning("警告", "請先建立區域 (閾值分割、Blob 分析等)。")
            return
        img = self._get_current_image()
        if img is None:
            messagebox.showwarning("警告", "請先載入圖片。")
            return

        from dl_anomaly.core.halcon_ops import reduce_domain
        result = reduce_domain(img, self._current_region)
        self._add_pipeline_step("縮減域", result)
        self.set_status("完成: 縮減域 (Reduce Domain)")

    def _cmd_crop_domain(self):
        """Crop current image to the bounding box of the current region."""
        if self._current_region is None:
            messagebox.showwarning("警告", "請先建立區域 (閾值分割、Blob 分析等)。")
            return
        img = self._get_current_image()
        if img is None:
            messagebox.showwarning("警告", "請先載入圖片。")
            return

        from dl_anomaly.core.halcon_ops import crop_domain
        result = crop_domain(img, self._current_region)
        self._add_pipeline_step("裁切域", result)
        self.set_status(f"完成: 裁切域 ({result.shape[1]}x{result.shape[0]})")

    # ==================================================================
    # Region highlight (from properties table click)
    # ==================================================================

    def _on_region_highlight(self, region_index: int) -> None:
        """Re-render viewer with specific region highlighted in yellow."""
        if self._current_region is None:
            return
        step = self._pipeline_panel.get_current_step()
        if step is None:
            return
        try:
            import cv2
            from dl_anomaly.core.region_ops import region_to_display_image
            display = region_to_display_image(
                self._current_region,
                step.array,
                highlight_indices=[region_index],
                highlight_color=(255, 255, 0),
            )
            # region_to_display_image returns BGR; viewer expects RGB
            if display.ndim == 3 and display.shape[2] == 3:
                display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            self._viewer.set_image(display)
        except Exception:
            pass

    def _on_region_remove(self, region_index: int) -> None:
        """Remove a single region by its 1-based index and add a new pipeline step."""
        if self._current_region is None:
            return
        step = self._pipeline_panel.get_current_step()
        if step is None:
            return
        try:
            remaining = [
                p.index for p in self._current_region.properties
                if p.index != region_index
            ]
            if not remaining:
                return
            new_region = self._current_region._keep_indices(remaining)
            self._current_region = new_region

            # Record undo point, then add pipeline step with the updated region
            ci = self._pipeline_panel.get_current_index()
            if ci >= 0:
                self._undo_stack.append(ci)
            self._redo_stack.clear()

            from dl_anomaly.core.region_ops import region_to_display_image
            import cv2
            display = region_to_display_image(new_region, step.array)
            if display.ndim == 3 and display.shape[2] == 3:
                display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)

            self._pipeline_panel.add_step(
                f"\u79fb\u9664\u5340\u57df #{region_index}",
                display,
                region=new_region,
                op_meta={"category": "region", "op": "remove", "params": {"index": region_index}},
            )
            self.set_status(f"\u5df2\u79fb\u9664\u5340\u57df #{region_index}\uff0c\u5269\u9918 {new_region.num_regions} \u500b")
        except Exception as exc:
            self._show_error("\u79fb\u9664\u5340\u57df\u5931\u6557", exc)

    # ==================================================================
    # New image operations: Binarize, Adaptive Threshold, Subtract, Contour
    # ==================================================================

    def _open_binarize_dialog(self):
        """Open binarization dialog — produces a binary IMAGE (not Region)."""
        img = self._get_current_image()
        if img is None:
            messagebox.showwarning("警告", "請先載入圖片。")
            return

        methods = ["THRESH_BINARY", "THRESH_BINARY_INV", "THRESH_OTSU", "THRESH_TRIANGLE"]
        self._open_param_dialog("二值化", [
            {"label": "閾值 (0-255):", "key": "thresh", "type": "int", "default": 128},
            {"label": "方法:", "key": "method", "type": "combo", "default": "THRESH_BINARY", "values": methods},
        ], lambda p: self._apply_binarize(img, p))

    def _apply_binarize(self, img, p):
        method_map = {
            "THRESH_BINARY": cv2.THRESH_BINARY,
            "THRESH_BINARY_INV": cv2.THRESH_BINARY_INV,
            "THRESH_OTSU": cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            "THRESH_TRIANGLE": cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE,
        }
        thresh_val = p["thresh"]
        cv_method = method_map.get(p["method"], cv2.THRESH_BINARY)

        def _compute():
            from dl_anomaly.core.halcon_ops import _ensure_gray
            gray = _ensure_gray(img)
            _, binary = cv2.threshold(gray, thresh_val, 255, cv_method)
            return binary

        def _done(binary):
            name = f"二值化 t={thresh_val} {p['method']}"
            self._add_pipeline_step(name, binary)
            self.set_status(f"完成: {name}")

        self._run_in_bg(_compute, on_done=_done, status_msg="二值化處理中...")

    def _open_adaptive_threshold_dialog(self):
        """Open adaptive threshold dialog with parameter control."""
        img = self._get_current_image()
        if img is None:
            messagebox.showwarning("警告", "請先載入圖片。")
            return

        self._open_param_dialog("自適應閾值", [
            {"label": "區塊大小 (block_size):", "key": "block_size", "type": "int", "default": 15},
            {"label": "常數 C:", "key": "c_value", "type": "int", "default": 5},
            {"label": "方法:", "key": "method", "type": "combo",
             "default": "GAUSSIAN_C", "values": ["MEAN_C", "GAUSSIAN_C"]},
        ], lambda p: self._apply_adaptive_threshold(img, p))

    def _apply_adaptive_threshold(self, img, p):
        block_size = p["block_size"]
        c_value = p["c_value"]
        adapt_method = p["method"]

        def _compute():
            from dl_anomaly.core.region_ops import binary_threshold
            cv_method = (cv2.ADAPTIVE_THRESH_GAUSSIAN_C if adapt_method == "GAUSSIAN_C"
                         else cv2.ADAPTIVE_THRESH_MEAN_C)
            # Ensure block_size is odd and >= 3
            bs = block_size
            if bs % 2 == 0:
                bs += 1
            if bs < 3:
                bs = 3
            region = binary_threshold(img, method="adaptive", block_size=bs, c_value=c_value)
            return region, bs

        def _done(result):
            region, bs = result
            self._current_region = region
            name = f"自適應閾值 bs={bs} C={c_value} {adapt_method}"
            self._pipeline_panel.add_step(name, region.to_binary_mask(), region=region)
            self.set_status(f"{name}: {region.num_regions} 個區域")

        self._run_in_bg(_compute, on_done=_done, status_msg="自適應閾值分割中...")

    def _open_subtract_dialog(self):
        """Open image subtraction dialog — select two pipeline steps."""
        steps = self._pipeline_panel.get_all_steps()
        if len(steps) < 2:
            messagebox.showwarning("警告", "需要至少兩個管線步驟才能進行圖像相減。")
            return

        step_names = [f"[{i}] {s.name}" for i, s in enumerate(steps)]

        dlg = tk.Toplevel(self)
        dlg.title("圖像相減")
        dlg.configure(bg="#2b2b2b")
        dlg.resizable(False, False)
        dlg.grab_set()

        tk.Label(dlg, text="圖像相減 (A - B) * mult + add",
                 bg="#2b2b2b", fg="#e0e0e0", font=("", 10, "bold")).grid(
            row=0, column=0, columnspan=2, pady=(10, 6), padx=10)

        tk.Label(dlg, text="圖像 A:", bg="#2b2b2b", fg="#e0e0e0").grid(
            row=1, column=0, sticky="e", padx=(10, 4), pady=4)
        a_var = tk.StringVar(value=step_names[0] if step_names else "")
        ttk.Combobox(dlg, textvariable=a_var, width=30,
                     values=step_names, state="readonly").grid(
            row=1, column=1, padx=(0, 10), pady=4)

        tk.Label(dlg, text="圖像 B:", bg="#2b2b2b", fg="#e0e0e0").grid(
            row=2, column=0, sticky="e", padx=(10, 4), pady=4)
        b_var = tk.StringVar(value=step_names[-1] if len(step_names) > 1 else step_names[0])
        ttk.Combobox(dlg, textvariable=b_var, width=30,
                     values=step_names, state="readonly").grid(
            row=2, column=1, padx=(0, 10), pady=4)

        tk.Label(dlg, text="乘數 (mult):", bg="#2b2b2b", fg="#e0e0e0").grid(
            row=3, column=0, sticky="e", padx=(10, 4), pady=4)
        mult_var = tk.StringVar(value="1.0")
        tk.Entry(dlg, textvariable=mult_var, width=10,
                 bg="#3c3c3c", fg="#e0e0e0", insertbackground="#e0e0e0").grid(
            row=3, column=1, sticky="w", padx=(0, 10), pady=4)

        tk.Label(dlg, text="偏移 (add):", bg="#2b2b2b", fg="#e0e0e0").grid(
            row=4, column=0, sticky="e", padx=(10, 4), pady=4)
        add_var = tk.StringVar(value="0.0")
        tk.Entry(dlg, textvariable=add_var, width=10,
                 bg="#3c3c3c", fg="#e0e0e0", insertbackground="#e0e0e0").grid(
            row=4, column=1, sticky="w", padx=(0, 10), pady=4)

        def _apply():
            try:
                mult = float(mult_var.get())
                add = float(add_var.get())
            except ValueError:
                messagebox.showwarning("警告", "請輸入有效的數值。", parent=dlg)
                return
            a_sel = a_var.get()
            b_sel = b_var.get()
            a_idx = step_names.index(a_sel) if a_sel in step_names else 0
            b_idx = step_names.index(b_sel) if b_sel in step_names else 0
            img_a = steps[a_idx].array.copy()
            img_b = steps[b_idx].array.copy()
            dlg.destroy()

            def _compute():
                from dl_anomaly.core import halcon_ops as hops
                return hops.sub_image(img_a, img_b, mult, add)

            def _done(result):
                name = f"圖像相減 [{a_idx}]-[{b_idx}] m={mult} a={add}"
                self._add_pipeline_step(name, result)
                self.set_status(f"完成: {name}")

            self._run_in_bg(_compute, on_done=_done, status_msg="圖像相減中...")

        btn_frame = tk.Frame(dlg, bg="#2b2b2b")
        btn_frame.grid(row=5, column=0, columnspan=2, pady=(6, 10))
        tk.Button(btn_frame, text="執行", command=_apply, width=8,
                  bg="#3a3a5c", fg="#e0e0e0", activebackground="#4a4a6c",
                  activeforeground="#ffffff").pack(side="left", padx=4)
        tk.Button(btn_frame, text="取消", command=dlg.destroy, width=8,
                  bg="#3a3a5c", fg="#e0e0e0", activebackground="#4a4a6c",
                  activeforeground="#ffffff").pack(side="left", padx=4)

        dlg.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() - dlg.winfo_width()) // 2
        y = self.winfo_y() + (self.winfo_height() - dlg.winfo_height()) // 2
        dlg.geometry(f"+{x}+{y}")

    def _open_contour_detection_dialog(self):
        """Open contour detection dialog with area filter."""
        img = self._get_current_image()
        if img is None:
            messagebox.showwarning("警告", "請先載入圖片。")
            return

        modes = ["RETR_LIST", "RETR_EXTERNAL"]
        self._open_param_dialog("輪廓檢測", [
            {"label": "最小面積:", "key": "min_area", "type": "int", "default": 0},
            {"label": "最大面積:", "key": "max_area", "type": "int", "default": 10000},
            {"label": "模式:", "key": "mode", "type": "combo", "default": "RETR_LIST", "values": modes},
        ], lambda p: self._apply_contour_detection(img, p))

    def _apply_contour_detection(self, img, p):
        min_area = p["min_area"]
        max_area = p["max_area"]
        mode_str = p["mode"]

        def _compute():
            from dl_anomaly.core.halcon_ops import _ensure_gray
            mode_map = {"RETR_LIST": cv2.RETR_LIST, "RETR_EXTERNAL": cv2.RETR_EXTERNAL}
            cv_mode = mode_map.get(mode_str, cv2.RETR_LIST)

            gray = _ensure_gray(img)
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv_mode, cv2.CHAIN_APPROX_SIMPLE)

            # Area filter
            filtered = [c for c in contours if min_area <= cv2.contourArea(c) <= max_area]

            # Draw on image
            vis = img.copy()
            if vis.ndim == 2:
                vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(vis, filtered, -1, (0, 255, 0), 2)
            return vis, len(filtered)

        def _done(result):
            vis, count = result
            name = f"輪廓檢測 area=[{min_area},{max_area}] ({count})"
            self._add_pipeline_step(name, vis)
            self.set_status(f"輪廓檢測完成: 找到 {count} 個輪廓")

        self._run_in_bg(_compute, on_done=_done, status_msg="輪廓檢測中...")

    def _open_dyn_threshold_dialog(self) -> None:
        """Open dialog for dynamic threshold segmentation (HALCON style)."""
        img = self._get_current_image()
        if img is None:
            messagebox.showwarning("\u8b66\u544a", "\u8acb\u5148\u8f09\u5165\u5716\u7247\u3002")
            return

        dlg = tk.Toplevel(self)
        dlg.title("\u52d5\u614b\u95be\u503c\u5206\u5272")
        dlg.configure(bg="#2b2b2b")
        dlg.resizable(False, False)
        dlg.grab_set()

        tk.Label(dlg, text="\u52d5\u614b\u95be\u503c\u5206\u5272 (dyn_threshold)",
                 bg="#2b2b2b", fg="#e0e0e0", font=("", 10, "bold")).grid(
            row=0, column=0, columnspan=2, pady=(10, 6), padx=10)

        # Kernel size
        tk.Label(dlg, text="\u6838\u5927\u5c0f (ksize):", bg="#2b2b2b", fg="#e0e0e0").grid(
            row=1, column=0, sticky="e", padx=(10, 4), pady=4)
        ksize_var = tk.StringVar(value="7")
        tk.Entry(dlg, textvariable=ksize_var, width=8,
                 bg="#3c3c3c", fg="#e0e0e0", insertbackground="#e0e0e0").grid(
            row=1, column=1, padx=(0, 10), pady=4)

        # Kernel shape
        tk.Label(dlg, text="\u6838\u5f62\u72c0:", bg="#2b2b2b", fg="#e0e0e0").grid(
            row=2, column=0, sticky="e", padx=(10, 4), pady=4)
        shape_var = tk.StringVar(value="octagon")
        shape_combo = ttk.Combobox(dlg, textvariable=shape_var, width=10,
                                   values=["octagon", "rectangle", "ellipse", "cross"],
                                   state="readonly")
        shape_combo.grid(row=2, column=1, padx=(0, 10), pady=4)

        # Offset
        tk.Label(dlg, text="\u504f\u79fb\u91cf (offset):", bg="#2b2b2b", fg="#e0e0e0").grid(
            row=3, column=0, sticky="e", padx=(10, 4), pady=4)
        offset_var = tk.StringVar(value="75")
        tk.Entry(dlg, textvariable=offset_var, width=8,
                 bg="#3c3c3c", fg="#e0e0e0", insertbackground="#e0e0e0").grid(
            row=3, column=1, padx=(0, 10), pady=4)

        # Mode
        tk.Label(dlg, text="\u6a21\u5f0f:", bg="#2b2b2b", fg="#e0e0e0").grid(
            row=4, column=0, sticky="e", padx=(10, 4), pady=4)
        mode_var = tk.StringVar(value="not_equal")
        mode_combo = ttk.Combobox(dlg, textvariable=mode_var, width=10,
                                  values=["not_equal", "light", "dark", "equal"],
                                  state="readonly")
        mode_combo.grid(row=4, column=1, padx=(0, 10), pady=4)

        def _apply():
            try:
                ks = int(ksize_var.get())
                ofs = float(offset_var.get())
            except ValueError:
                messagebox.showwarning("\u8b66\u544a", "\u8acb\u8f38\u5165\u6709\u6548\u7684\u6578\u503c\u3002", parent=dlg)
                return
            sh = shape_var.get()
            md = mode_var.get()
            dlg.destroy()

            src = img

            def _compute():
                from dl_anomaly.core import halcon_ops as hops
                from dl_anomaly.core.region import Region
                from dl_anomaly.core.region_ops import compute_region_properties

                gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY) if src.ndim == 3 else src
                img_opening = hops.gray_opening_shape(gray, ks, ks, sh)
                img_closing = hops.gray_closing_shape(gray, ks, ks, sh)
                mask = hops.dyn_threshold(img_opening, img_closing, ofs, md)

                num, labels = cv2.connectedComponents(mask, connectivity=8)
                labels = labels.astype(np.int32)
                props = compute_region_properties(labels, gray)
                return Region(
                    labels=labels, num_regions=num - 1, properties=props,
                    source_image=gray, source_shape=gray.shape[:2],
                )

            def _done(region):
                self._current_region = region
                name = f"\u52d5\u614b\u95be\u503c k={ks} {sh} ofs={ofs} {md}"
                self._pipeline_panel.add_step(name, region.to_binary_mask(), region=region)
                self.set_status(f"{name}: {region.num_regions} \u500b\u5340\u57df")

            self._run_in_bg(_compute, on_done=_done, status_msg="\u52d5\u614b\u95be\u503c\u5206\u5272\u4e2d...")

        btn_frame = tk.Frame(dlg, bg="#2b2b2b")
        btn_frame.grid(row=5, column=0, columnspan=2, pady=(6, 10))
        tk.Button(btn_frame, text="\u57f7\u884c", command=_apply, width=8,
                  bg="#3a3a5c", fg="#e0e0e0", activebackground="#4a4a6c",
                  activeforeground="#ffffff").pack(side="left", padx=4)
        tk.Button(btn_frame, text="\u53d6\u6d88", command=dlg.destroy, width=8,
                  bg="#3a3a5c", fg="#e0e0e0", activebackground="#4a4a6c",
                  activeforeground="#ffffff").pack(side="left", padx=4)

        dlg.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() - dlg.winfo_width()) // 2
        y = self.winfo_y() + (self.winfo_height() - dlg.winfo_height()) // 2
        dlg.geometry(f"+{x}+{y}")

    def _apply_halcon_op(self, op: str) -> None:
        """Apply a HALCON operator in a background thread."""
        img = self._get_current_image()
        if img is None:
            messagebox.showwarning("\u8b66\u544a", "\u8acb\u5148\u8f09\u5165\u5716\u7247\u3002")
            return

        def _compute():
            return self._halcon_op_compute(op, img)

        def _done(pair):
            name, result = pair
            if result is not None:
                op_meta = {"category": "halcon", "op": op, "params": {}}
                self._add_pipeline_step(name, result, op_meta=op_meta)
                self.set_status_success(f"\u5b8c\u6210: {name}")
                self._history_panel.add_entry(name, f"op={op}")

        self._run_in_bg(_compute, on_done=_done, status_msg=f"\u57f7\u884c {op}...")

    def _halcon_op_compute(self, op: str, img: np.ndarray):
        """Pure computation for a HALCON op (runs in background thread).
        Returns (name, result_array) or (name, None).
        """
        from dl_anomaly.core import halcon_ops as hops

        result = None
        name = ""

        if op == "mean_image":
            result = hops.mean_image(img, 5)
            name = "\u5747\u503c\u6ffe\u6ce2 k=5"
        elif op == "median_image":
            result = hops.median_image(img, 5)
            name = "\u4e2d\u503c\u6ffe\u6ce2 k=5"
        elif op == "gauss_filter":
            result = hops.gauss_filter(img, 1.5)
            name = "\u9ad8\u65af\u6ffe\u6ce2 \u03c3=1.5"
        elif op == "bilateral_filter":
            result = hops.bilateral_filter(img, 9, 75, 75)
            name = "\u96d9\u908a\u6ffe\u6ce2"
        elif op == "sharpen_image":
            result = hops.sharpen_image(img, 0.5)
            name = "\u92b3\u5316 0.5"
        elif op == "emphasize":
            result = hops.emphasize(img, 7, 1.5)
            name = "\u5f37\u8abf\u6ffe\u6ce2"
        elif op == "laplace_filter":
            result = hops.laplace_filter(img)
            name = "Laplacian"
        elif op == "edges_canny":
            result = hops.edges_canny(img, 50, 150, 1.0)
            name = "Canny \u908a\u7de3"
        elif op == "sobel_filter":
            result = hops.sobel_filter(img, "both")
            name = "Sobel \u908a\u7de3"
        elif op == "prewitt_filter":
            result = hops.prewitt_filter(img)
            name = "Prewitt \u908a\u7de3"
        elif op == "zero_crossing":
            result = hops.zero_crossing(img)
            name = "零交叉"
        elif op == "gray_erosion":
            result = hops.gray_erosion(img, 5)
            name = "\u7070\u5ea6\u4fb5\u8755 k=5"
        elif op == "gray_dilation":
            result = hops.gray_dilation(img, 5)
            name = "\u7070\u5ea6\u81a8\u8139 k=5"
        elif op == "gray_opening":
            result = hops.gray_opening(img, 5)
            name = "\u7070\u5ea6\u958b\u904b\u7b97 k=5"
        elif op == "gray_closing":
            result = hops.gray_closing(img, 5)
            name = "\u7070\u5ea6\u9589\u904b\u7b97 k=5"
        elif op == "top_hat":
            result = hops.top_hat(img, 9)
            name = "Top-hat k=9"
        elif op == "bottom_hat":
            result = hops.bottom_hat(img, 9)
            name = "Bottom-hat k=9"
        elif op == "rotate_90":
            result = hops.rotate_image(img, 90, "constant")
            name = "\u65cb\u8f49 90\u00b0"
        elif op == "rotate_180":
            result = hops.rotate_image(img, 180, "constant")
            name = "\u65cb\u8f49 180\u00b0"
        elif op == "rotate_270":
            result = hops.rotate_image(img, 270, "constant")
            name = "\u65cb\u8f49 270\u00b0"
        elif op == "mirror_h":
            result = hops.mirror_image(img, "horizontal")
            name = "\u6c34\u5e73\u93e1\u50cf"
        elif op == "mirror_v":
            result = hops.mirror_image(img, "vertical")
            name = "\u5782\u76f4\u93e1\u50cf"
        elif op == "zoom_50":
            result = hops.zoom_image(img, 0.5, 0.5)
            name = "\u7e2e\u653e 50%"
        elif op == "zoom_200":
            result = hops.zoom_image(img, 2.0, 2.0)
            name = "\u7e2e\u653e 200%"
        elif op == "rgb_to_gray":
            result = hops.rgb_to_gray(img)
            name = "\u8f49\u7070\u968e"
        elif op == "rgb_to_hsv":
            result = hops.rgb_to_hsv(img)
            name = "\u8f49 HSV"
        elif op == "rgb_to_hls":
            result = hops.rgb_to_hls(img)
            name = "轉 HLS"
        elif op == "histogram_eq_halcon":
            result = hops.histogram_eq(img)
            name = "\u76f4\u65b9\u5716\u5747\u8861"
        elif op == "invert_image":
            result = hops.invert_image(img)
            name = "\u53cd\u8272"
        elif op == "illuminate":
            result = hops.illuminate(img, 41, 1.0)
            name = "\u5149\u7167\u6821\u6b63"
        elif op == "abs_image":
            result = hops.abs_image(img)
            name = "\u7d55\u5c0d\u503c"
        elif op == "bright_up":
            result = hops.scale_image(img, 1.0, 30)
            name = "\u4eae\u5ea6 +30"
        elif op == "bright_down":
            result = hops.scale_image(img, 1.0, -30)
            name = "\u4eae\u5ea6 -30"
        elif op == "contrast_up":
            result = hops.scale_image(img, 1.3, 0)
            name = "\u5c0d\u6bd4\u5ea6\u589e\u5f37"
        elif op == "entropy_image":
            result = hops.entropy_image(img, 5)
            name = "\u71b5\u5f71\u50cf k=5"
        elif op == "deviation_image":
            result = hops.deviation_image(img, 5)
            name = "\u6a19\u6e96\u5dee\u5f71\u50cf k=5"
        elif op == "local_min":
            result = hops.local_min(img, 5)
            name = "\u5c40\u90e8\u6700\u5c0f k=5"
        elif op == "local_max":
            result = hops.local_max(img, 5)
            name = "\u5c40\u90e8\u6700\u5927 k=5"
        elif op == "find_barcode":
            results = hops.find_barcode(img)
            if results:
                result = img.copy()
                if result.ndim == 2:
                    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
                for r in results:
                    pts = r.get("points")
                    data = r.get("data", "")
                    if pts is not None:
                        pts_arr = np.array(pts, dtype=np.int32)
                        cv2.polylines(result, [pts_arr], True, (0, 255, 0), 2)
                        cv2.putText(result, data, (pts_arr[0][0], pts_arr[0][1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                name = f"\u689d\u78bc ({len(results)})"
        elif op == "find_qrcode":
            results = hops.find_qrcode(img)
            if results:
                result = img.copy()
                if result.ndim == 2:
                    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
                for r in results:
                    pts = r.get("points")
                    data = r.get("data", "")
                    if pts is not None:
                        pts_arr = np.array(pts, dtype=np.int32)
                        cv2.polylines(result, [pts_arr], True, (0, 0, 255), 2)
                        cv2.putText(result, data[:30], (pts_arr[0][0], pts_arr[0][1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                name = f"QR Code ({len(results)})"
        elif op == "find_datamatrix":
            results = hops.find_datamatrix(img)
            if results:
                result = img.copy()
                if result.ndim == 2:
                    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
                for r in results:
                    pts = r.get("points")
                    if pts is not None:
                        pts_arr = np.array(pts, dtype=np.int32)
                        cv2.polylines(result, [pts_arr], True, (255, 0, 0), 2)
                name = f"DataMatrix ({len(results)})"
        elif op == "skeleton":
            result = hops.skeleton(img)
            name = "骨架化"
        elif op == "grab_image":
            result = hops.grab_image(0)
            if result is not None:
                name = "相機擷取"
            else:
                return ("", None)
        elif op == "abs_diff_image":
            ci = self._pipeline_panel.get_current_index()
            if ci is not None and ci > 0:
                prev_step = self._pipeline_panel.get_step(ci - 1)
                if prev_step is not None:
                    prev = prev_step.array.copy()
                    result = hops.abs_diff_image(img, prev)
                    name = "絕對差分"

        return (name, result)

    # ==================================================================
    # Script Editor
    # ==================================================================

    def _toggle_script_editor(self, state: bool = None) -> None:
        """Toggle script editor panel."""
        if self._script_editor_visible:
            if self._script_editor is not None:
                self._script_editor.pack_forget()
            self._script_editor_visible = False
            self.set_status("\u8173\u672c\u7de8\u8f2f\u5668\u5df2\u95dc\u9589")
        else:
            try:
                from dl_anomaly.gui.script_editor import ScriptEditor
                if self._script_editor is None:
                    self._script_editor = ScriptEditor(self, app=self)
                self._script_editor.pack(fill=tk.BOTH, expand=False, side=tk.BOTTOM)
                self._script_editor.pack_configure(fill=tk.BOTH, expand=False)
            except Exception as exc:
                self._show_error("\u8173\u672c\u7de8\u8f2f\u5668\u8f09\u5165\u5931\u6557", exc)
                return
            self._script_editor_visible = True
            self.set_status("\u8173\u672c\u7de8\u8f2f\u5668\u5df2\u958b\u555f (F9 \u57f7\u884c)")

    def _run_script(self) -> None:
        """Run script from the script editor."""
        if self._script_editor is not None and self._script_editor_visible:
            self._script_editor.run_script()

    # ==================================================================
    # Close
    # ==================================================================

    def _on_close(self) -> None:
        self._app_state.save_geometry(self)
        self._app_state.save_sash_positions(self._paned)
        self.destroy()
