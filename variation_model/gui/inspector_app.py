"""
gui/inspector_app.py - Industrial Vision 風格主應用程式

三欄式介面：
- 左側：管線步驟面板（步驟清單 + 縮圖）
- 中央：可縮放影像檢視器
- 右側：影像屬性 + 操作參數面板
- 頂部：選單列 + 工具列
- 底部：狀態列
"""

from __future__ import annotations

import logging
import os as _os
import platform
import sys as _sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..', '..'))
from shared.app_state import AppState
from shared.error_dialog import show_error
from shared.history_panel import HistoryPanel
from shared.progress_manager import ProgressManager
from shared.judgment_indicator import JudgmentIndicator
from shared.live_inspection_panel import LiveInspectionPanel
from shared.dashboard_panel import DashboardPanel

from config import Config
from gui.image_viewer import ImageViewer
from gui.operations_panel import OperationsPanel
from gui.pipeline_panel import PipelinePanel
from gui.properties_panel import PropertiesPanel
from gui.toolbar import Toolbar

# 延遲載入的模組（用到時才 import，加速啟動）
# - core.variation_model.VariationModel
# - gui.dialogs: BatchInspectDialog, HistogramDialog, ModelInfoDialog, SettingsDialog, TrainingDialog

logger = logging.getLogger(__name__)

# 最近開啟的檔案上限
MAX_RECENT_FILES = 10

# Platform-aware accelerator modifier for menu labels
_ACCEL_MOD = "Cmd" if platform.system() == "Darwin" else "Ctrl"


class InspectorApp(tk.Tk):
    """Industrial Vision 風格的 Variation Model Inspector 主應用程式。"""

    def __init__(self, config: Config) -> None:
        super().__init__()

        self.config = config
        self.model = None  # Optional[VariationModel], lazy-imported

        # 目前載入的影像資訊
        self._current_image_path: Optional[Path] = None
        self._current_raw_image: Optional[np.ndarray] = None

        # 最近開啟的檔案
        self._recent_files: List[str] = []

        # 影像處理功能狀態
        self._pixel_inspector = None
        self._script_editor = None
        self._script_editor_visible = False
        self._current_region = None  # 目前的 Region 物件

        # ── 持久化狀態 ──
        self._closing = False
        self._app_state = AppState("variation_model")

        # ── 視窗設定 ──
        self.title("Variation Model Inspector - Industrial Vision")
        self.geometry("1400x900")
        self.minsize(1000, 700)

        # ── 設定暗色主題樣式 ──
        self._setup_styles()

        # ── 建立 UI ──
        self._create_menu()
        self._create_toolbar()
        self._create_main_layout()
        self._create_status_bar()

        # ── 鍵盤快捷鍵 ──
        self._bind_shortcuts()

        # ── 關閉事件 ──
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # ── 恢復持久化狀態 ──
        self._app_state.restore_geometry(self)
        self.after(200, lambda: self._app_state.restore_sash_positions(self._main_paned))
        self._recent_files = self._app_state.get_recent_files()
        self._update_recent_menu()

        logger.info("Vision 風格 GUI 初始化完成")

    # ================================================================== #
    #  暗色主題樣式                                                        #
    # ================================================================== #

    def _setup_styles(self) -> None:
        """設定 ttk 暗色主題樣式。"""
        style = ttk.Style()

        # 嘗試使用 clam 主題作為基底
        available = style.theme_names()
        if "clam" in available:
            style.theme_use("clam")

        # 基本色彩
        bg_dark = "#2b2b2b"
        bg_medium = "#3c3c3c"
        fg_light = "#cccccc"
        fg_white = "#e0e0e0"
        accent = "#0078d4"
        select_bg = "#264f78"

        style.configure(".", background=bg_dark, foreground=fg_light, fieldbackground=bg_medium)
        style.configure("TFrame", background=bg_dark)
        style.configure("TLabel", background=bg_dark, foreground=fg_light)
        style.configure("TButton", background=bg_medium, foreground=fg_white, padding=4)
        style.map("TButton",
                  background=[("active", accent), ("pressed", "#005a9e")],
                  foreground=[("active", "#ffffff")])
        style.configure("TLabelframe", background=bg_dark, foreground=fg_light)
        style.configure("TLabelframe.Label", background=bg_dark, foreground=fg_light)
        style.configure("TEntry", fieldbackground=bg_medium, foreground=fg_white)
        style.configure("TScale", background=bg_dark, troughcolor=bg_medium)
        style.configure("TCheckbutton", background=bg_dark, foreground=fg_light)
        style.configure("TRadiobutton", background=bg_dark, foreground=fg_light)
        style.configure("TNotebook", background=bg_dark)
        style.configure("TNotebook.Tab", background=bg_medium, foreground=fg_light, padding=[8, 4])
        style.configure("TSeparator", background="#555555")

        style.configure("Treeview",
                        background=bg_medium, foreground=fg_light,
                        fieldbackground=bg_medium, rowheight=25)
        style.map("Treeview", background=[("selected", select_bg)])
        style.configure("Treeview.Heading",
                        background="#444444", foreground=fg_white,
                        relief=tk.FLAT)

        style.configure("TProgressbar", background=accent, troughcolor=bg_medium)

        style.configure("Horizontal.TScrollbar",
                        background=bg_medium, troughcolor=bg_dark)
        style.configure("Vertical.TScrollbar",
                        background=bg_medium, troughcolor=bg_dark)

        # 管線面板樣式
        style.configure("Pipeline.TFrame", background=bg_dark)
        style.configure("Pipeline.TLabel", background=bg_dark, foreground=fg_light)
        style.configure("Selected.TFrame", background=select_bg)
        style.configure("Selected.TLabel", background=select_bg, foreground="#ffffff")

        # Toolbutton 樣式
        style.configure("Toolbutton", background=bg_medium, foreground=fg_light, padding=3)
        style.map("Toolbutton",
                  background=[("selected", accent), ("active", "#444444")],
                  foreground=[("selected", "#ffffff")])

        # 狀態列
        style.configure("Status.TLabel", background="#1e1e1e", foreground="#aaaaaa", padding=(6, 2))
        style.configure("Status.TFrame", background="#1e1e1e")
        style.configure("Success.Status.TLabel", background="#2e7d32", foreground="#ffffff")

        self.configure(bg=bg_dark)

    # ================================================================== #
    #  選單列                                                              #
    # ================================================================== #

    def _create_menu(self) -> None:
        """建立選單列。"""
        menubar = tk.Menu(self, bg="#2b2b2b", fg="#cccccc", activebackground="#0078d4",
                          activeforeground="#ffffff")

        # ── 檔案 ──
        file_menu = tk.Menu(menubar, tearoff=0, bg="#3c3c3c", fg="#cccccc",
                           activebackground="#0078d4", activeforeground="#ffffff")
        file_menu.add_command(label="開啟影像...", command=self._open_image, accelerator=f"{_ACCEL_MOD}+O")
        file_menu.add_command(label="開啟目錄...", command=self._open_directory)
        file_menu.add_separator()
        file_menu.add_command(label="儲存目前影像...", command=self._save_current_image, accelerator=f"{_ACCEL_MOD}+S")
        file_menu.add_command(label="儲存所有結果...", command=self._save_all_results)
        file_menu.add_separator()

        # 最近開啟的檔案子選單
        self._recent_menu = tk.Menu(file_menu, tearoff=0, bg="#3c3c3c", fg="#cccccc",
                                    activebackground="#0078d4", activeforeground="#ffffff")
        file_menu.add_cascade(label="最近開啟", menu=self._recent_menu)
        file_menu.add_separator()
        file_menu.add_command(label="環境設定...", command=self._open_settings)
        file_menu.add_separator()
        file_menu.add_command(label="結束", command=self._on_close)
        menubar.add_cascade(label="檔案", menu=file_menu)

        # ── 操作 ──
        ops_menu = tk.Menu(menubar, tearoff=0, bg="#3c3c3c", fg="#cccccc",
                          activebackground="#0078d4", activeforeground="#ffffff")
        ops_menu.add_command(label="灰階轉換", command=lambda: self._apply_operation("grayscale"))
        ops_menu.add_separator()

        blur_menu = tk.Menu(ops_menu, tearoff=0, bg="#3c3c3c", fg="#cccccc",
                           activebackground="#0078d4", activeforeground="#ffffff")
        blur_menu.add_command(label="高斯模糊", command=lambda: self._apply_operation("gaussian_blur"))
        blur_menu.add_command(label="中值濾波", command=lambda: self._apply_operation("median_blur"))
        ops_menu.add_cascade(label="平滑", menu=blur_menu)

        thresh_menu = tk.Menu(ops_menu, tearoff=0, bg="#3c3c3c", fg="#cccccc",
                             activebackground="#0078d4", activeforeground="#ffffff")
        thresh_menu.add_command(label="Otsu 二值化", command=lambda: self._apply_operation("threshold_otsu"))
        thresh_menu.add_command(label="自適應二值化", command=lambda: self._apply_operation("threshold_adaptive"))
        ops_menu.add_cascade(label="二值化", menu=thresh_menu)

        edge_menu = tk.Menu(ops_menu, tearoff=0, bg="#3c3c3c", fg="#cccccc",
                           activebackground="#0078d4", activeforeground="#ffffff")
        edge_menu.add_command(label="Canny 邊緣偵測", command=lambda: self._apply_operation("canny"))
        edge_menu.add_command(label="Sobel 邊緣偵測", command=lambda: self._apply_operation("sobel"))
        ops_menu.add_cascade(label="邊緣偵測", menu=edge_menu)

        morph_menu = tk.Menu(ops_menu, tearoff=0, bg="#3c3c3c", fg="#cccccc",
                            activebackground="#0078d4", activeforeground="#ffffff")
        morph_menu.add_command(label="侵蝕", command=lambda: self._apply_operation("erode"))
        morph_menu.add_command(label="膨脹", command=lambda: self._apply_operation("dilate"))
        morph_menu.add_command(label="開運算", command=lambda: self._apply_operation("morph_open"))
        morph_menu.add_command(label="閉運算", command=lambda: self._apply_operation("morph_close"))
        ops_menu.add_cascade(label="形態學", menu=morph_menu)

        ops_menu.add_separator()
        ops_menu.add_command(label="直方圖均衡化", command=lambda: self._apply_operation("histogram_eq"))
        menubar.add_cascade(label="操作", menu=ops_menu)

        # ── 區域 ──
        region_menu = tk.Menu(menubar, tearoff=0, bg="#3c3c3c", fg="#cccccc",
                              activebackground="#0078d4", activeforeground="#ffffff")
        region_menu.add_command(label="像素值檢查器...", command=self._toggle_pixel_inspector, accelerator=f"{_ACCEL_MOD}+I")
        region_menu.add_separator()
        region_menu.add_command(label="閾值分割...", command=self._open_threshold_dialog, accelerator=f"{_ACCEL_MOD}+T")
        region_menu.add_command(label="自動閾值 (Otsu)", command=self._auto_threshold_otsu)
        region_menu.add_command(label="自動閾值 (自適應)", command=self._auto_threshold_adaptive)
        region_menu.add_separator()
        region_menu.add_command(label="打散 (Connection)", command=self._region_connection)
        region_menu.add_command(label="填充 (Fill Up)", command=self._region_fill_up)

        shape_trans_menu = tk.Menu(region_menu, tearoff=0, bg="#3c3c3c", fg="#cccccc",
                                   activebackground="#0078d4", activeforeground="#ffffff")
        for st in ["convex", "rectangle", "circle", "ellipse"]:
            shape_trans_menu.add_command(label=st, command=lambda s=st: self._region_shape_trans(s))
        region_menu.add_cascade(label="形狀變換", menu=shape_trans_menu)

        region_menu.add_separator()
        region_menu.add_command(label="區域侵蝕", command=lambda: self._region_morphology("erosion"))
        region_menu.add_command(label="區域膨脹", command=lambda: self._region_morphology("dilation"))
        region_menu.add_command(label="區域開運算", command=lambda: self._region_morphology("opening"))
        region_menu.add_command(label="區域閉運算", command=lambda: self._region_morphology("closing"))
        region_menu.add_separator()
        region_menu.add_command(label="篩選區域...", command=self._open_region_filter)
        region_menu.add_command(label="依灰度篩選...", command=self._region_select_gray)
        region_menu.add_command(label="排序區域...", command=self._region_sort)
        region_menu.add_separator()
        region_menu.add_command(label="區域聯集", command=lambda: self._region_set_op("union"))
        region_menu.add_command(label="區域交集", command=lambda: self._region_set_op("intersection"))
        region_menu.add_command(label="區域差集", command=lambda: self._region_set_op("difference"))
        region_menu.add_command(label="區域補集", command=lambda: self._region_set_op("complement"))
        region_menu.add_separator()
        region_menu.add_command(label="Blob 分析...", command=self._open_blob_analysis)
        menubar.add_cascade(label="區域", menu=region_menu)

        # ── 影像處理 ──
        vision_menu = tk.Menu(menubar, tearoff=0, bg="#3c3c3c", fg="#cccccc",
                              activebackground="#0078d4", activeforeground="#ffffff")

        # 濾波子選單
        filter_menu = tk.Menu(vision_menu, tearoff=0, bg="#3c3c3c", fg="#cccccc",
                              activebackground="#0078d4", activeforeground="#ffffff")
        for label, op in [("均值濾波", "mean_image"), ("中值濾波", "median_image"),
                          ("高斯濾波", "gauss_filter"), ("高斯模糊", "gauss_blur"),
                          ("雙邊濾波", "bilateral_filter"),
                          ("銳化", "sharpen_image"), ("強調", "emphasize"),
                          ("Laplacian", "laplace_filter")]:
            filter_menu.add_command(label=label, command=lambda o=op: self._apply_vision_op(o))
        vision_menu.add_cascade(label="濾波", menu=filter_menu)

        # 邊緣子選單
        edge_menu2 = tk.Menu(vision_menu, tearoff=0, bg="#3c3c3c", fg="#cccccc",
                             activebackground="#0078d4", activeforeground="#ffffff")
        for label, op in [("Canny", "edges_canny"), ("Sobel", "sobel_filter"), ("Prewitt", "prewitt_filter")]:
            edge_menu2.add_command(label=label, command=lambda o=op: self._apply_vision_op(o))
        vision_menu.add_cascade(label="邊緣", menu=edge_menu2)

        # 形態學子選單
        morph_menu2 = tk.Menu(vision_menu, tearoff=0, bg="#3c3c3c", fg="#cccccc",
                              activebackground="#0078d4", activeforeground="#ffffff")
        for label, op in [("灰度侵蝕", "gray_erosion"), ("灰度膨脹", "gray_dilation"),
                          ("灰度開運算", "gray_opening"), ("灰度閉運算", "gray_closing"),
                          ("灰度開運算 (形狀)", "gray_opening_shape"),
                          ("灰度閉運算 (形狀)", "gray_closing_shape"),
                          ("Top-hat", "top_hat"), ("Bottom-hat", "bottom_hat")]:
            morph_menu2.add_command(label=label, command=lambda o=op: self._apply_vision_op(o))
        morph_menu2.add_separator()
        morph_menu2.add_command(label="動態閾值分割", command=lambda: self._apply_vision_op("dyn_threshold"))
        morph_menu2.add_command(label="可變閾值", command=lambda: self._apply_vision_op("var_threshold"))
        morph_menu2.add_command(label="局部閾值", command=lambda: self._apply_vision_op("local_threshold"))
        vision_menu.add_cascade(label="形態學", menu=morph_menu2)

        # 幾何子選單
        geom_menu = tk.Menu(vision_menu, tearoff=0, bg="#3c3c3c", fg="#cccccc",
                            activebackground="#0078d4", activeforeground="#ffffff")
        for label, op in [("旋轉 90°", "rotate_90"), ("旋轉 180°", "rotate_180"),
                          ("旋轉 270°", "rotate_270"),
                          ("水平鏡像", "mirror_h"), ("垂直鏡像", "mirror_v"),
                          ("縮放 50%", "zoom_50"), ("縮放 200%", "zoom_200")]:
            geom_menu.add_command(label=label, command=lambda o=op: self._apply_vision_op(o))
        vision_menu.add_cascade(label="幾何", menu=geom_menu)

        # 色彩子選單
        color_menu = tk.Menu(vision_menu, tearoff=0, bg="#3c3c3c", fg="#cccccc",
                             activebackground="#0078d4", activeforeground="#ffffff")
        for label, op in [("轉灰階", "rgb_to_gray"), ("轉 HSV", "rgb_to_hsv"),
                          ("轉 HLS", "rgb_to_hls"),
                          ("直方圖均衡", "histogram_eq_vision"),
                          ("反色", "invert_image"), ("光照校正", "illuminate"),
                          ("通道分離", "decompose3")]:
            color_menu.add_command(label=label, command=lambda o=op: self._apply_vision_op(o))
        color_menu.add_separator()
        color_menu.add_command(label="CLAHE", command=lambda: self._apply_vision_op("clahe"))
        vision_menu.add_cascade(label="色彩", menu=color_menu)

        # 算術子選單
        arith_menu = tk.Menu(vision_menu, tearoff=0, bg="#3c3c3c", fg="#cccccc",
                             activebackground="#0078d4", activeforeground="#ffffff")
        for label, op in [("絕對值", "abs_image"), ("反色", "invert_image"),
                          ("亮度 +30", "bright_up"), ("亮度 -30", "bright_down"),
                          ("對比度增強", "contrast_up"), ("絕對差分", "abs_diff_image")]:
            arith_menu.add_command(label=label, command=lambda o=op: self._apply_vision_op(o))
        vision_menu.add_cascade(label="算術", menu=arith_menu)

        # 紋理子選單
        texture_menu = tk.Menu(vision_menu, tearoff=0, bg="#3c3c3c", fg="#cccccc",
                               activebackground="#0078d4", activeforeground="#ffffff")
        for label, op in [("熵影像", "entropy_image"), ("標準差影像", "deviation_image"),
                          ("局部最小", "local_min"), ("局部最大", "local_max"),
                          ("Laws 紋理", "texture_laws"), ("平均曲率", "mean_curvature")]:
            texture_menu.add_command(label=label, command=lambda o=op: self._apply_vision_op(o))
        vision_menu.add_cascade(label="紋理", menu=texture_menu)

        # 條碼子選單
        barcode_menu = tk.Menu(vision_menu, tearoff=0, bg="#3c3c3c", fg="#cccccc",
                               activebackground="#0078d4", activeforeground="#ffffff")
        for label, op in [("條碼偵測", "find_barcode"), ("QR Code", "find_qrcode"),
                          ("DataMatrix", "find_datamatrix")]:
            barcode_menu.add_command(label=label, command=lambda o=op: self._apply_vision_op(o))
        vision_menu.add_cascade(label="條碼", menu=barcode_menu)

        # 灰度變換
        gray_trans_menu = tk.Menu(vision_menu, tearoff=0, bg="#3c3c3c", fg="#cccccc",
                                  activebackground="#0078d4", activeforeground="#ffffff")
        for label, op in [("對數變換", "log_image"), ("指數變換", "exp_image"),
                          ("Gamma 校正", "gamma_image")]:
            gray_trans_menu.add_command(label=label, command=lambda o=op: self._apply_vision_op(o))
        vision_menu.add_cascade(label="灰度變換", menu=gray_trans_menu)

        # 頻域處理
        freq_menu = tk.Menu(vision_menu, tearoff=0, bg="#3c3c3c", fg="#cccccc",
                            activebackground="#0078d4", activeforeground="#ffffff")
        for label, op in [("FFT 頻譜", "fft_image"),
                          ("低通濾波", "freq_filter_lowpass"),
                          ("高通濾波", "freq_filter_highpass")]:
            freq_menu.add_command(label=label, command=lambda o=op: self._apply_vision_op(o))
        vision_menu.add_cascade(label="頻域處理", menu=freq_menu)

        # 分割
        seg_menu = tk.Menu(vision_menu, tearoff=0, bg="#3c3c3c", fg="#cccccc",
                           activebackground="#0078d4", activeforeground="#ffffff")
        for label, op in [("分水嶺", "watersheds"), ("距離變換", "distance_transform"),
                          ("骨架化", "skeleton")]:
            seg_menu.add_command(label=label, command=lambda o=op: self._apply_vision_op(o))
        vision_menu.add_cascade(label="分割", menu=seg_menu)

        # 特徵點
        feat_menu = tk.Menu(vision_menu, tearoff=0, bg="#3c3c3c", fg="#cccccc",
                            activebackground="#0078d4", activeforeground="#ffffff")
        for label, op in [("Harris 角點", "points_harris"), ("Shi-Tomasi 特徵點", "points_shi_tomasi")]:
            feat_menu.add_command(label=label, command=lambda o=op: self._apply_vision_op(o))
        vision_menu.add_cascade(label="特徵點", menu=feat_menu)

        # 直線/圓偵測
        hough_menu = tk.Menu(vision_menu, tearoff=0, bg="#3c3c3c", fg="#cccccc",
                             activebackground="#0078d4", activeforeground="#ffffff")
        for label, op in [("Hough 直線", "hough_lines"), ("Hough 圓", "hough_circles")]:
            hough_menu.add_command(label=label, command=lambda o=op: self._apply_vision_op(o))
        vision_menu.add_cascade(label="直線/圓偵測", menu=hough_menu)

        # 相機
        camera_menu = tk.Menu(vision_menu, tearoff=0, bg="#3c3c3c", fg="#cccccc",
                              activebackground="#0078d4", activeforeground="#ffffff")
        camera_menu.add_command(label="擷取影像", command=lambda: self._apply_vision_op("grab_image"))
        vision_menu.add_cascade(label="相機", menu=camera_menu)

        vision_menu.add_separator()
        vision_menu.add_command(label="腳本編輯器", command=self._toggle_script_editor, accelerator="F8")
        menubar.add_cascade(label="影像處理", menu=vision_menu)

        # ── 模型 ──
        model_menu = tk.Menu(menubar, tearoff=0, bg="#3c3c3c", fg="#cccccc",
                            activebackground="#0078d4", activeforeground="#ffffff")
        model_menu.add_command(label="訓練新模型...", command=self._train_model)
        model_menu.add_command(label="載入模型...", command=self._load_model)
        model_menu.add_command(label="儲存模型...", command=self._save_model)
        model_menu.add_separator()
        model_menu.add_command(label="模型資訊...", command=self._show_model_info)
        model_menu.add_command(label="重新計算閾值", command=self._reprepare_thresholds)
        menubar.add_cascade(label="模型", menu=model_menu)

        # ── 檢視 ──
        view_menu = tk.Menu(menubar, tearoff=0, bg="#3c3c3c", fg="#cccccc",
                           activebackground="#0078d4", activeforeground="#ffffff")
        view_menu.add_command(label="適配視窗", command=self._fit_window, accelerator="Space")
        view_menu.add_command(label="放大", command=self._zoom_in, accelerator="+")
        view_menu.add_command(label="縮小", command=self._zoom_out, accelerator="-")
        view_menu.add_command(label="100% 檢視", command=self._zoom_100)
        view_menu.add_separator()
        view_menu.add_command(label="切換格線", command=self._toggle_grid)
        view_menu.add_command(label="切換十字游標", command=self._toggle_crosshair)
        view_menu.add_separator()
        view_menu.add_command(label="顯示直方圖...", command=self._show_histogram)
        menubar.add_cascade(label="檢視", menu=view_menu)

        # ── 工具 ──
        tools_menu = tk.Menu(menubar, tearoff=0, bg="#3c3c3c", fg="#cccccc",
                             activebackground="#0078d4", activeforeground="#ffffff")
        tools_menu.add_command(label="形狀匹配...", command=self._open_shape_matching)
        tools_menu.add_command(label="量測工具...", command=self._open_metrology)
        tools_menu.add_separator()
        tools_menu.add_command(label="ROI 管理...", command=self._open_roi_manager)
        tools_menu.add_separator()
        tools_menu.add_command(label="PatchCore / ONNX 模型...", command=self._open_advanced_models)
        tools_menu.add_separator()
        tools_menu.add_command(label="檢測工具 (FFT/色彩/OCR/條碼)...", command=self._open_inspection_tools)
        tools_menu.add_separator()
        tools_menu.add_command(label="工程工具 (標定/管線/SPC/拼接)...", command=self._open_engineering_tools)
        tools_menu.add_separator()
        tools_menu.add_command(label="MVP 工具 (相機/流程/報表)...", command=self._open_mvp_tools)
        menubar.add_cascade(label="工具", menu=tools_menu)

        # ── 說明 ──
        help_menu = tk.Menu(menubar, tearoff=0, bg="#3c3c3c", fg="#cccccc",
                           activebackground="#0078d4", activeforeground="#ffffff")
        help_menu.add_command(label="鍵盤快捷鍵...", command=self._show_shortcuts)
        help_menu.add_command(label="關於...", command=self._show_about)
        menubar.add_cascade(label="說明", menu=help_menu)

        self.configure(menu=menubar)

    # ================================================================== #
    #  工具列                                                              #
    # ================================================================== #

    def _create_toolbar(self) -> None:
        """建立工具列。"""
        toolbar_frame = ttk.Frame(self)
        toolbar_frame.pack(fill=tk.X, padx=2, pady=2)

        self.toolbar = Toolbar(toolbar_frame, commands={
            "open_image": self._open_image,
            "save_image": self._save_current_image,
            "undo": self._undo,
            "redo": self._redo,
            "fit_window": self._fit_window,
            "zoom_in": self._zoom_in,
            "zoom_out": self._zoom_out,
            "zoom_100": self._zoom_100,
            "train_model": self._train_model,
            "load_model": self._load_model,
            "inspect_single": self._inspect_single,
            "inspect_batch": self._inspect_batch,
            "toggle_grid": self._toggle_grid,
            "toggle_crosshair": self._toggle_crosshair,
            "toggle_pixel_inspector": self._toggle_pixel_inspector,
            "threshold": self._open_threshold_dialog,
            "blob_analysis": self._open_blob_analysis,
            "toggle_script_editor": self._toggle_script_editor,
        })
        self.toolbar.pack(fill=tk.X)

    # ================================================================== #
    #  主區域佈局                                                          #
    # ================================================================== #

    def _create_main_layout(self) -> None:
        """建立三欄式主佈局。"""
        # 使用 PanedWindow 實現可調整大小的面板
        self._main_paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        self._main_paned.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # ── 左側：管線面板 ──
        self.pipeline_panel = PipelinePanel(
            self._main_paned,
            on_step_selected=self._on_pipeline_step_selected,
        )
        self._main_paned.add(self.pipeline_panel, weight=0)

        # ── 中央：影像檢視器 ──
        center_frame = ttk.Frame(self._main_paned)
        self._main_paned.add(center_frame, weight=1)

        self.image_viewer = ImageViewer(center_frame)
        self.image_viewer.pack(fill=tk.BOTH, expand=True)
        self.image_viewer.set_pixel_info_callback(self._on_pixel_info)
        self.image_viewer.set_zoom_change_callback(self._on_zoom_change)

        # ── 右側：屬性 + 操作面板 ──
        right_frame = ttk.Frame(self._main_paned)
        self._main_paned.add(right_frame, weight=0)

        # OK/NG 判定指示器
        self._judgment_indicator = JudgmentIndicator(right_frame, height=100)
        self._judgment_indicator.pack(fill=tk.X, padx=2, pady=(2, 4))

        # 右側使用 PanedWindow 上下分割
        right_paned = ttk.PanedWindow(right_frame, orient=tk.VERTICAL)
        right_paned.pack(fill=tk.BOTH, expand=True)

        # 屬性面板
        self.properties_panel = PropertiesPanel(right_paned)
        right_paned.add(self.properties_panel, weight=0)

        # 操作面板
        self.operations_panel = OperationsPanel(
            right_paned,
            on_apply=self._on_operation_apply,
            on_param_change=self._on_param_change,
        )
        right_paned.add(self.operations_panel, weight=1)

        # 即時檢測面板
        self._live_panel = LiveInspectionPanel(
            right_paned,
            on_start=self._on_live_start,
            on_stop=self._on_live_stop,
            on_inspect_single=lambda: self._inspect_single(),
        )
        right_paned.add(self._live_panel, weight=0)

        # SPC Dashboard 面板
        self._dashboard_panel = DashboardPanel(right_paned)
        right_paned.add(self._dashboard_panel, weight=0)

        # 操作歷史面板
        self._history_panel = HistoryPanel(right_paned)
        right_paned.add(self._history_panel, weight=0)

        # 設定操作面板的初始參數
        self.operations_panel.set_params({
            "blur_kernel": self.config.gaussian_blur_kernel,
            "abs_threshold": self.config.abs_threshold,
            "var_threshold": self.config.var_threshold,
            "morph_kernel": self.config.morph_kernel_size,
            "min_area": self.config.min_defect_area,
        })

    # ================================================================== #
    #  狀態列                                                              #
    # ================================================================== #

    def _create_status_bar(self) -> None:
        """建立底部狀態列。"""
        status_frame = ttk.Frame(self, style="Status.TFrame")
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        self._statusbar = status_frame

        # 狀態文字
        self._status_var = tk.StringVar(value="就緒")
        self._status_label = ttk.Label(
            status_frame, textvariable=self._status_var,
            style="Status.TLabel", width=30, anchor=tk.W,
        )
        self._status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # 進度條管理器
        self._progress = ProgressManager(self, status_frame)

        # 分隔
        ttk.Separator(status_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=2)

        # 像素座標
        self._pos_var = tk.StringVar(value="座標: --")
        ttk.Label(
            status_frame, textvariable=self._pos_var,
            style="Status.TLabel", width=18, anchor=tk.CENTER,
        ).pack(side=tk.LEFT)

        ttk.Separator(status_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=2)

        # 像素值
        self._pixel_val_var = tk.StringVar(value="像素值: --")
        ttk.Label(
            status_frame, textvariable=self._pixel_val_var,
            style="Status.TLabel", width=15, anchor=tk.CENTER,
        ).pack(side=tk.LEFT)

        ttk.Separator(status_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=2)

        # 縮放
        self._zoom_var = tk.StringVar(value="縮放: 100%")
        ttk.Label(
            status_frame, textvariable=self._zoom_var,
            style="Status.TLabel", width=12, anchor=tk.CENTER,
        ).pack(side=tk.LEFT)

        ttk.Separator(status_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=2)

        # 步驟
        self._step_var = tk.StringVar(value="步驟: 0/0")
        ttk.Label(
            status_frame, textvariable=self._step_var,
            style="Status.TLabel", width=10, anchor=tk.CENTER,
        ).pack(side=tk.LEFT)

    # ================================================================== #
    #  鍵盤快捷鍵                                                          #
    # ================================================================== #

    def _should_handle_key(self, event=None) -> bool:
        """Return False if focus is on a text-input widget."""
        w = self.focus_get()
        if w is None:
            return True
        widget_class = w.winfo_class()
        return widget_class not in ("Entry", "Text", "TCombobox", "Spinbox", "TEntry", "TSpinbox")

    def _bind_shortcuts(self) -> None:
        """綁定鍵盤快捷鍵。"""
        import platform as _platform
        _MOD = "Command" if _platform.system() == "Darwin" else "Control"

        # --- Mod+key shortcuts (Cmd on macOS, Ctrl on Win/Linux) ---
        self.bind_all(f"<{_MOD}-o>", lambda e: self._open_image())
        self.bind_all(f"<{_MOD}-O>", lambda e: self._open_image())
        self.bind_all(f"<{_MOD}-s>", lambda e: self._save_current_image())
        self.bind_all(f"<{_MOD}-S>", lambda e: self._save_current_image())
        self.bind_all(f"<{_MOD}-z>", lambda e: self._undo())
        self.bind_all(f"<{_MOD}-Z>", lambda e: self._undo())
        self.bind_all(f"<{_MOD}-y>", lambda e: self._redo())
        self.bind_all(f"<{_MOD}-Y>", lambda e: self._redo())
        self.bind_all(f"<{_MOD}-i>", lambda e: self._toggle_pixel_inspector())
        self.bind_all(f"<{_MOD}-I>", lambda e: self._toggle_pixel_inspector())
        self.bind_all(f"<{_MOD}-t>", lambda e: self._open_threshold_dialog())
        self.bind_all(f"<{_MOD}-T>", lambda e: self._open_threshold_dialog())

        # --- Plain keys (guarded: skip when focus is on text input) ---
        self.bind_all("<space>", lambda e: self._fit_window() if self._should_handle_key(e) else None)
        self.bind_all("<plus>", lambda e: self._zoom_in() if self._should_handle_key(e) else None)
        self.bind_all("<minus>", lambda e: self._zoom_out() if self._should_handle_key(e) else None)
        self.bind_all("<equal>", lambda e: self._zoom_in() if self._should_handle_key(e) else None)
        self.bind_all("<Delete>", lambda e: self._delete_current_step() if self._should_handle_key(e) else None)

        # --- Function keys (safe, no conflict with text input) ---
        self.bind_all("<F5>", lambda e: self._inspect_single())
        self.bind_all("<F8>", lambda e: self._toggle_script_editor())
        self.bind_all("<F9>", lambda e: self._run_script())
        self.bind_all("<F1>", lambda e: self._show_shortcuts_dialog())
        self.bind_all("<Escape>", lambda e: self._cancel_current_tool())

    # ================================================================== #
    #  回呼處理                                                            #
    # ================================================================== #

    def _on_pixel_info(self, x: int, y: int, value: str) -> None:
        """影像檢視器的像素資訊回呼。"""
        self._pos_var.set(f"座標: ({x}, {y})")
        self._pixel_val_var.set(f"像素值: {value}")

        # 更新像素檢查器
        if self._pixel_inspector is not None and self._pixel_inspector.winfo_exists():
            img = self.image_viewer.get_image()
            if img is not None:
                try:
                    self._pixel_inspector.update_values(img, x, y)
                except Exception:
                    pass

    def _on_zoom_change(self, zoom_percent: float) -> None:
        """縮放變更回呼。"""
        self._zoom_var.set(f"縮放: {zoom_percent:.0f}%")

    def _on_pipeline_step_selected(self, index: int) -> None:
        """管線步驟被選取。"""
        step = self.pipeline_panel.get_step(index)
        if step is None:
            return

        self.image_viewer.set_image(step.image)
        self._current_region = step.region
        if step.region is not None:
            self.image_viewer.set_region_overlay(step.region)
        self.properties_panel.update_properties(step.image, name=step.name, region=step.region)
        self._update_step_label()

    def _on_operation_apply(self, operation: str, params: Dict) -> None:
        """操作面板套用回呼。"""
        self._apply_operation(operation, params)

    def _on_param_change(self, param_name: str, value: float) -> None:
        """操作面板參數即時變更回呼（目前僅記錄，不自動套用）。"""
        pass

    # ================================================================== #
    #  檔案操作                                                            #
    # ================================================================== #

    def _open_image(self) -> None:
        """開啟影像檔案。"""
        path = filedialog.askopenfilename(
            title="開啟影像",
            filetypes=[
                ("影像檔案", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("所有檔案", "*.*"),
            ],
        )
        if not path:
            return

        self._load_image_file(Path(path))

    def _open_directory(self) -> None:
        """開啟目錄中的第一張影像。"""
        d = filedialog.askdirectory(title="開啟影像目錄")
        if not d:
            return

        exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        images = sorted(
            f for f in Path(d).iterdir()
            if f.suffix.lower() in exts
        )
        if not images:
            messagebox.showwarning("警告", "目錄中未找到影像檔案。")
            return

        self._load_image_file(images[0])
        self._status_var.set(f"目錄中共 {len(images)} 張影像")

    def _load_image_file(self, path: Path) -> None:
        """載入影像檔案並加入管線。"""
        try:
            from core.preprocessor import ImagePreprocessor

            preprocessor = ImagePreprocessor(self.config)
            image = preprocessor.load_image(path)

            self._current_image_path = path
            self._current_raw_image = image

            # 清除舊管線
            self.pipeline_panel.clear_all()

            # 加入原始影像步驟
            self.pipeline_panel.add_step("原始影像", image, "original")

            # 更新最近檔案
            self._add_recent_file(str(path))

            self._status_var.set(f"已載入: {path.name}")
            logger.info("影像已載入: %s", path)
        except Exception as exc:
            logger.exception("載入影像失敗")
            self._show_error("載入影像", exc)

    def _save_current_image(self) -> None:
        """儲存目前顯示的影像。"""
        step = self.pipeline_panel.get_current_step()
        if step is None:
            messagebox.showwarning("警告", "沒有可儲存的影像。")
            return

        path = filedialog.asksaveasfilename(
            title="儲存影像",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp"), ("所有檔案", "*.*")],
            initialfile=f"{step.name}.png",
        )
        if not path:
            return

        try:
            img = step.image
            if img.dtype != np.uint8:
                min_v = img.min()
                max_v = img.max()
                if max_v - min_v > 0:
                    img = ((img - min_v) / (max_v - min_v) * 255).astype(np.uint8)
                else:
                    img = np.zeros_like(img, dtype=np.uint8)
            cv2.imwrite(path, img)
            self._status_var.set(f"已儲存: {Path(path).name}")
        except Exception as exc:
            self._show_error("儲存影像", exc)

    def _save_all_results(self) -> None:
        """儲存所有管線步驟的影像。"""
        count = self.pipeline_panel.get_step_count()
        if count == 0:
            messagebox.showwarning("警告", "沒有可儲存的結果。")
            return

        out_dir = filedialog.askdirectory(title="選擇輸出目錄")
        if not out_dir:
            return

        try:
            for i in range(count):
                step = self.pipeline_panel.get_step(i)
                if step is None:
                    continue
                img = step.image
                if img.dtype != np.uint8:
                    min_v = img.min()
                    max_v = img.max()
                    if max_v - min_v > 0:
                        img = ((img - min_v) / (max_v - min_v) * 255).astype(np.uint8)
                    else:
                        img = np.zeros_like(img, dtype=np.uint8)
                filename = f"{i + 1:02d}_{step.name}.png"
                cv2.imwrite(str(Path(out_dir) / filename), img)

            self._status_var.set(f"已儲存 {count} 張影像至 {out_dir}")
            messagebox.showinfo("完成", f"已儲存 {count} 張影像至:\n{out_dir}")
        except Exception as exc:
            self._show_error("儲存所有結果", exc)

    def _add_recent_file(self, path: str) -> None:
        """加入最近開啟檔案清單。"""
        if path in self._recent_files:
            self._recent_files.remove(path)
        self._recent_files.insert(0, path)
        self._recent_files = self._recent_files[:MAX_RECENT_FILES]
        self._update_recent_menu()
        self._app_state.add_recent_file(path)

    def _update_recent_menu(self) -> None:
        """更新最近開啟檔案選單。"""
        self._recent_menu.delete(0, tk.END)
        for path in self._recent_files:
            name = Path(path).name
            self._recent_menu.add_command(
                label=name,
                command=lambda p=path: self._load_image_file(Path(p)),
            )

    # ================================================================== #
    #  影像處理操作                                                        #
    # ================================================================== #

    def _apply_operation(self, operation: str, params: Optional[Dict] = None) -> None:
        """套用影像處理操作。"""
        step = self.pipeline_panel.get_current_step()
        if step is None:
            messagebox.showwarning("警告", "請先載入影像。")
            return

        if params is None:
            params = self.operations_panel.get_params()

        src = step.image
        result = None
        name = ""

        try:
            if operation == "grayscale":
                if src.ndim == 2:
                    messagebox.showinfo("提示", "影像已經是灰階。")
                    return
                result = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
                name = "灰階"

            elif operation == "gaussian_blur":
                k = params.get("blur_kernel", 3)
                if k % 2 == 0:
                    k += 1
                result = cv2.GaussianBlur(src, (k, k), 0)
                name = f"高斯模糊 k={k}"

            elif operation == "median_blur":
                k = params.get("blur_kernel", 3)
                if k % 2 == 0:
                    k += 1
                result = cv2.medianBlur(src, k)
                name = f"中值濾波 k={k}"

            elif operation == "histogram_eq":
                if src.ndim == 3:
                    # 轉為 YCrCb 進行均衡化
                    ycrcb = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)
                    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
                    result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
                else:
                    result = cv2.equalizeHist(src if src.dtype == np.uint8 else src.astype(np.uint8))
                name = "直方圖均衡"

            elif operation == "canny":
                gray = src if src.ndim == 2 else cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
                if gray.dtype != np.uint8:
                    gray = ((gray - gray.min()) / max(gray.max() - gray.min(), 1) * 255).astype(np.uint8)
                result = cv2.Canny(gray, 50, 150)
                name = "Canny 邊緣"

            elif operation == "sobel":
                gray = src if src.ndim == 2 else cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
                if gray.dtype != np.uint8:
                    gray = ((gray - gray.min()) / max(gray.max() - gray.min(), 1) * 255).astype(np.uint8)
                sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                result = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
                result = np.clip(result, 0, 255).astype(np.uint8)
                name = "Sobel 邊緣"

            elif operation == "threshold_otsu":
                gray = src if src.ndim == 2 else cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
                if gray.dtype != np.uint8:
                    gray = ((gray - gray.min()) / max(gray.max() - gray.min(), 1) * 255).astype(np.uint8)
                _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                name = "Otsu 二值化"

            elif operation == "threshold_adaptive":
                gray = src if src.ndim == 2 else cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
                if gray.dtype != np.uint8:
                    gray = ((gray - gray.min()) / max(gray.max() - gray.min(), 1) * 255).astype(np.uint8)
                result = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 11, 2,
                )
                name = "自適應二值化"

            elif operation == "erode":
                k = params.get("morph_kernel", 3)
                if k % 2 == 0:
                    k += 1
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
                result = cv2.erode(src, kernel, iterations=1)
                name = f"侵蝕 k={k}"

            elif operation == "dilate":
                k = params.get("morph_kernel", 3)
                if k % 2 == 0:
                    k += 1
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
                result = cv2.dilate(src, kernel, iterations=1)
                name = f"膨脹 k={k}"

            elif operation == "morph_open":
                k = params.get("morph_kernel", 3)
                if k % 2 == 0:
                    k += 1
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
                result = cv2.morphologyEx(src, cv2.MORPH_OPEN, kernel)
                name = f"開運算 k={k}"

            elif operation == "morph_close":
                k = params.get("morph_kernel", 3)
                if k % 2 == 0:
                    k += 1
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
                result = cv2.morphologyEx(src, cv2.MORPH_CLOSE, kernel)
                name = f"閉運算 k={k}"

            elif operation == "run_inspection":
                self._inspect_single()
                return

            else:
                messagebox.showwarning("警告", f"未知的操作: {operation}")
                return

            if result is not None:
                self.pipeline_panel.add_step(name, result, "process")
                self.set_status_success(f"操作完成: {name}")
                self._history_panel.add_entry("操作", name)

        except Exception as exc:
            logger.exception("操作執行失敗: %s", operation)
            self._show_error(f"操作 [{operation}]", exc)

    # ================================================================== #
    #  模型操作                                                            #
    # ================================================================== #

    def _train_model(self) -> None:
        """開啟訓練對話框。"""
        def on_complete(model):
            self.model = model
            self._status_var.set(f"訓練完成: {model.count} 張影像")

            # 將模型影像加入管線（如果有影像正在顯示）
            imgs = model.get_model_images()
            if imgs["mean"] is not None:
                self.pipeline_panel.add_step("模型均值", imgs["mean"], "model")
            if imgs["std"] is not None:
                self.pipeline_panel.add_step("模型標準差", imgs["std"], "model")

        from gui.dialogs import TrainingDialog
        TrainingDialog(self, self.config, on_complete=on_complete)

    def _load_model(self) -> None:
        """載入模型。"""
        path = filedialog.askopenfilename(
            title="載入模型",
            filetypes=[("NumPy Archive", "*.npz"), ("所有檔案", "*.*")],
            initialdir=str(self.config.model_save_dir),
        )
        if not path:
            return

        try:
            from core.variation_model import VariationModel
            model = VariationModel.load(path)
            self.model = model
            self._status_var.set(f"模型已載入: {Path(path).name} ({model.count} 張影像)")

            # 確保模型有閾值
            if model.get_model_images()["upper"] is None:
                model.prepare(
                    abs_threshold=self.config.abs_threshold,
                    var_threshold=self.config.var_threshold,
                )

            messagebox.showinfo("載入成功", f"模型已載入。\n訓練影像數: {model.count}")
        except Exception as exc:
            logger.exception("載入模型失敗")
            self._show_error("載入模型", exc)

    def _save_model(self) -> None:
        """儲存模型。"""
        if self.model is None or not self.model.is_trained:
            messagebox.showwarning("警告", "沒有已訓練的模型可以儲存。")
            return

        path = filedialog.asksaveasfilename(
            title="儲存模型",
            defaultextension=".npz",
            filetypes=[("NumPy Archive", "*.npz")],
            initialdir=str(self.config.model_save_dir),
        )
        if not path:
            return

        try:
            self.model.save(path)
            self._status_var.set(f"模型已儲存: {Path(path).name}")
            messagebox.showinfo("儲存成功", f"模型已儲存至:\n{path}")
        except Exception as exc:
            self._show_error("儲存模型", exc)

    def _show_model_info(self) -> None:
        """顯示模型資訊。"""
        if self.model is None:
            messagebox.showwarning("警告", "尚未載入模型。")
            return
        from gui.dialogs import ModelInfoDialog
        ModelInfoDialog(self, self.model)

    def _reprepare_thresholds(self) -> None:
        """以目前參數重新計算閾值。"""
        if self.model is None or not self.model.is_trained:
            messagebox.showwarning("警告", "沒有已訓練的模型。")
            return

        params = self.operations_panel.get_params()
        try:
            self.model.prepare(
                abs_threshold=params["abs_threshold"],
                var_threshold=params["var_threshold"],
            )
            self._status_var.set("閾值已重新計算")
            messagebox.showinfo("完成", "閾值已使用新參數重新計算。")
        except Exception as exc:
            self._show_error("閾值計算", exc)

    # ================================================================== #
    #  檢測操作                                                            #
    # ================================================================== #

    def _inspect_single(self) -> None:
        """單張影像檢測。"""
        if self.model is None or not self.model.is_trained:
            messagebox.showwarning("警告", "請先載入或訓練模型。")
            return

        # 確保模型有閾值
        imgs = self.model.get_model_images()
        if imgs["upper"] is None:
            params = self.operations_panel.get_params()
            self.model.prepare(
                abs_threshold=params["abs_threshold"],
                var_threshold=params["var_threshold"],
            )

        # 如果已有影像，使用目前影像
        if self._current_image_path and self._current_image_path.exists():
            self._run_inspection(self._current_image_path)
        else:
            # 開啟檔案選擇
            path = filedialog.askopenfilename(
                title="選擇測試影像",
                filetypes=[
                    ("影像檔案", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                    ("所有檔案", "*.*"),
                ],
            )
            if path:
                self._load_image_file(Path(path))
                self._run_inspection(Path(path))

    def _run_inspection(self, image_path: Path) -> None:
        """執行單張檢測（背景執行緒）。"""
        def _do_inspect():
            from pipeline.inference import InferencePipeline
            from visualization.heatmap import (
                create_defect_overlay,
                create_difference_heatmap,
            )

            params = self.operations_panel.get_params()
            updated_config = self.config.update(
                abs_threshold=params["abs_threshold"],
                var_threshold=params["var_threshold"],
                morph_kernel_size=params["morph_kernel"],
                min_defect_area=params["min_area"],
                gaussian_blur_kernel=params["blur_kernel"],
            )

            # 確保模型閾值使用最新參數
            self.model.prepare(
                abs_threshold=params["abs_threshold"],
                var_threshold=params["var_threshold"],
            )

            pipeline = InferencePipeline(self.model, updated_config)
            result, processed = pipeline.inspect_single(image_path)

            # 產生視覺化
            overlay = create_defect_overlay(
                processed, result.defect_mask,
                result.too_bright_mask, result.too_dark_mask, alpha=0.5,
            )
            heatmap = create_difference_heatmap(result.difference_image)
            return result, processed, overlay, heatmap

        def _on_done(payload):
            result, processed, overlay, heatmap = payload
            self._on_inspection_done(result, processed, overlay, heatmap, image_path)

        self._run_in_bg(
            _do_inspect,
            on_done=_on_done,
            status_msg=f"檢測中: {image_path.name}...",
        )

    def _on_inspection_done(self, result, processed, overlay, heatmap, image_path):
        """檢測完成回呼。"""
        from core.inspector import InspectionResult

        # 加入管線步驟
        self.pipeline_panel.add_step("前處理影像", processed, "preprocess")
        self.pipeline_panel.add_step("差異熱力圖", heatmap, "heatmap")
        self.pipeline_panel.add_step("瑕疵遮罩", result.defect_mask, "mask")
        self.pipeline_panel.add_step("瑕疵疊加", overlay, "overlay")

        # 顯示疊加結果上的瑕疵區域
        if result.defect_regions:
            self.image_viewer.add_overlay_regions(
                result.defect_regions,
                color="#FF4444",
            )

        status = "NG - 瑕疵" if result.is_defective else "PASS - 合格"
        detail = (
            f"{status}  |  分數: {result.score:.4f}%  |  "
            f"瑕疵數: {result.num_defects}"
        )
        self.set_status_success(f"檢測完成: {image_path.name} - {detail}")
        self._history_panel.add_entry("檢測", f"{image_path.name} - {status}")

        # 更新 OK/NG 判定指示器
        self._judgment_indicator.set_result(
            is_pass=not result.is_defective,
            score=result.score,
            message=f"瑕疵數: {result.num_defects}",
        )
        self._dashboard_panel.update_from_result(
            not result.is_defective, result.score,
        )
        # 更新即時檢測面板
        if self._live_panel.is_running:
            self._live_panel.update_result(
                not result.is_defective, result.score, str(image_path),
            )

        if result.is_defective:
            messagebox.showwarning(
                "檢測結果 - NG",
                f"檔案: {image_path.name}\n"
                f"結果: {status}\n"
                f"分數: {result.score:.4f}%\n"
                f"瑕疵數: {result.num_defects}",
            )

    def _inspect_batch(self) -> None:
        """批次檢測。"""
        if self.model is None or not self.model.is_trained:
            messagebox.showwarning("警告", "請先載入或訓練模型。")
            return

        # 確保模型有閾值
        imgs = self.model.get_model_images()
        if imgs["upper"] is None:
            params = self.operations_panel.get_params()
            self.model.prepare(
                abs_threshold=params["abs_threshold"],
                var_threshold=params["var_threshold"],
            )

        def on_view_result(result_tuple):
            """從批次結果雙擊跳轉到主視窗。"""
            r, path, processed = result_tuple
            from visualization.heatmap import create_defect_overlay, create_difference_heatmap

            self._load_image_file(path)
            self.pipeline_panel.add_step("前處理影像", processed, "preprocess")
            heatmap = create_difference_heatmap(r.difference_image)
            overlay = create_defect_overlay(
                processed, r.defect_mask,
                r.too_bright_mask, r.too_dark_mask, alpha=0.5,
            )
            self.pipeline_panel.add_step("差異熱力圖", heatmap, "heatmap")
            self.pipeline_panel.add_step("瑕疵遮罩", r.defect_mask, "mask")
            self.pipeline_panel.add_step("瑕疵疊加", overlay, "overlay")

        params = self.operations_panel.get_params()
        updated_config = self.config.update(
            abs_threshold=params["abs_threshold"],
            var_threshold=params["var_threshold"],
            morph_kernel_size=params["morph_kernel"],
            min_defect_area=params["min_area"],
            gaussian_blur_kernel=params["blur_kernel"],
        )

        from gui.dialogs import BatchInspectDialog
        BatchInspectDialog(
            self, self.model, updated_config,
            on_view_result=on_view_result,
        )

    # ================================================================== #
    #  檢視操作                                                            #
    # ================================================================== #

    def _fit_window(self) -> None:
        self.image_viewer.fit_to_window()

    def _zoom_in(self) -> None:
        self.image_viewer.zoom_in()

    def _zoom_out(self) -> None:
        self.image_viewer.zoom_out()

    def _zoom_100(self) -> None:
        self.image_viewer.zoom_100()

    def _toggle_grid(self) -> None:
        current = self.toolbar.get_toggle_state("grid")
        self.image_viewer.set_grid(current)

    def _toggle_crosshair(self) -> None:
        current = self.toolbar.get_toggle_state("crosshair")
        self.image_viewer.set_crosshair(current)

    def _show_histogram(self) -> None:
        step = self.pipeline_panel.get_current_step()
        if step is None:
            messagebox.showwarning("警告", "沒有可顯示的影像。")
            return
        from gui.dialogs import HistogramDialog
        HistogramDialog(self, step.image, f"直方圖 - {step.name}")

    # ================================================================== #
    #  復原 / 重做                                                         #
    # ================================================================== #

    def _undo(self) -> None:
        """復原（回到上一步驟）。"""
        self.pipeline_panel.go_previous()

    def _redo(self) -> None:
        """重做（前進到下一步驟）。"""
        self.pipeline_panel.go_next()

    def _delete_current_step(self) -> None:
        """刪除目前管線步驟。"""
        idx = self.pipeline_panel.get_current_index()
        if idx >= 0:
            self.pipeline_panel.delete_step(idx)

    # ================================================================== #
    #  說明                                                                #
    # ================================================================== #

    def _show_shortcuts(self) -> None:
        """顯示鍵盤快捷鍵。"""
        shortcuts = (
            "鍵盤快捷鍵:\n\n"
            "Ctrl+O    開啟影像\n"
            "Ctrl+S    儲存目前影像\n"
            "Ctrl+Z    復原 (上一步驟)\n"
            "Ctrl+Y    重做 (下一步驟)\n"
            "Ctrl+I    像素值檢查器\n"
            "Ctrl+T    閾值分割\n"
            "Space     適配視窗\n"
            "+/-       放大/縮小\n"
            "F5        執行檢測\n"
            "F8        腳本編輯器\n"
            "F9        執行腳本\n"
            "Delete    刪除目前步驟\n"
            "\n滑鼠操作:\n"
            "滾輪      縮放\n"
            "左鍵拖曳   平移\n"
            "雙擊      適配視窗\n"
        )
        messagebox.showinfo("鍵盤快捷鍵", shortcuts)

    def _show_about(self) -> None:
        """顯示關於對話框。"""
        messagebox.showinfo(
            "關於",
            "Variation Model Inspector\n"
            "Industrial Vision Style\n\n"
            "基於統計變異模型的瑕疵檢測系統\n"
            "使用 Welford 線上演算法計算影像統計量\n\n"
            "介面風格：Industrial Vision\n"
            "Framework: tkinter + OpenCV",
        )

    # ================================================================== #
    #  輔助方法                                                            #
    # ================================================================== #

    def _update_step_label(self) -> None:
        """更新狀態列的步驟標籤。"""
        idx = self.pipeline_panel.get_current_index()
        total = self.pipeline_panel.get_step_count()
        if total > 0 and idx >= 0:
            self._step_var.set(f"步驟: {idx + 1}/{total}")
        else:
            self._step_var.set("步驟: 0/0")

    def set_status(self, message: str) -> None:
        """公開 API：更新狀態列。"""
        self._status_var.set(message)

    def _open_settings(self) -> None:
        """開啟環境設定對話框。"""
        def on_settings_applied(new_config):
            self.config = new_config
            # 同步更新操作面板參數
            self.operations_panel.set_params({
                "blur_kernel": new_config.gaussian_blur_kernel,
                "abs_threshold": new_config.abs_threshold,
                "var_threshold": new_config.var_threshold,
                "morph_kernel": new_config.morph_kernel_size,
                "min_area": new_config.min_defect_area,
            })
            self._status_var.set("設定已更新")

        from gui.dialogs import SettingsDialog
        SettingsDialog(self, self.config, on_apply=on_settings_applied)

    # ================================================================== #
    #  影像處理區域操作                                                     #
    # ================================================================== #

    def _get_current_image(self) -> Optional[np.ndarray]:
        """取得目前管線步驟的影像。"""
        step = self.pipeline_panel.get_current_step()
        return step.image if step else None

    def _toggle_pixel_inspector(self) -> None:
        """切換像素值檢查器顯示。"""
        if self._pixel_inspector is not None and self._pixel_inspector.winfo_exists():
            self._pixel_inspector.destroy()
            self._pixel_inspector = None
        else:
            from gui.pixel_inspector import PixelInspector
            self._pixel_inspector = PixelInspector(self)

    def _open_threshold_dialog(self) -> None:
        """開啟閾值分割對話框。"""
        img = self._get_current_image()
        if img is None:
            messagebox.showwarning("警告", "請先載入影像。")
            return

        from gui.threshold_dialog import ThresholdDialog

        def on_accept(region, display_image, name):
            self._current_region = region
            self.pipeline_panel.add_step(name, display_image, "region", region=region)
            self._status_var.set(f"閾值分割完成: {region.num_regions} 個區域")

        ThresholdDialog(self, img, on_accept=on_accept)

    def _auto_threshold_otsu(self) -> None:
        """自動 Otsu 閾值分割。"""
        img = self._get_current_image()
        if img is None:
            messagebox.showwarning("警告", "請先載入影像。")
            return
        try:
            from core.region_ops import binary_threshold, region_to_display_image
            region = binary_threshold(img, method="otsu")
            display = region_to_display_image(region, img)
            self._current_region = region
            self.pipeline_panel.add_step("Otsu 閾值", display, "region", region=region)
            self._status_var.set(f"Otsu 閾值: {region.num_regions} 個區域")
        except Exception as exc:
            self._show_error("Otsu 閾值", exc)

    def _auto_threshold_adaptive(self) -> None:
        """自適應閾值分割。"""
        img = self._get_current_image()
        if img is None:
            messagebox.showwarning("警告", "請先載入影像。")
            return
        try:
            from core.region_ops import binary_threshold, region_to_display_image
            region = binary_threshold(img, method="adaptive")
            display = region_to_display_image(region, img)
            self._current_region = region
            self.pipeline_panel.add_step("自適應閾值", display, "region", region=region)
            self._status_var.set(f"自適應閾值: {region.num_regions} 個區域")
        except Exception as exc:
            self._show_error("自適應閾值", exc)

    def _region_connection(self) -> None:
        """打散 (Connection) 操作。"""
        if self._current_region is None:
            messagebox.showwarning("警告", "請先執行閾值分割產生區域。")
            return
        try:
            from core.region_ops import connection, region_to_display_image
            region = connection(self._current_region)
            img = self._get_current_image()
            display = region_to_display_image(region, img)
            self._current_region = region
            self.pipeline_panel.add_step(f"Connection ({region.num_regions})", display, "region", region=region)
            self._status_var.set(f"打散: {region.num_regions} 個獨立區域")
        except Exception as exc:
            self._show_error("Connection", exc)

    def _region_fill_up(self) -> None:
        """填充區域孔洞。"""
        if self._current_region is None:
            messagebox.showwarning("警告", "請先產生區域。")
            return
        try:
            from core.region_ops import region_to_display_image
            mask = self._current_region.to_binary_mask()
            # Fill holes using findContours + drawContours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filled = np.zeros_like(mask)
            cv2.drawContours(filled, contours, -1, 255, -1)
            # Rebuild region
            num, labels = cv2.connectedComponents(filled, connectivity=8)
            labels = labels.astype(np.int32)
            from core.region_ops import compute_region_properties
            from core.region import Region
            props = compute_region_properties(labels, self._current_region.source_image)
            region = Region(labels=labels, num_regions=num - 1, properties=props,
                          source_image=self._current_region.source_image,
                          source_shape=self._current_region.source_shape)
            img = self._get_current_image()
            display = region_to_display_image(region, img)
            self._current_region = region
            self.pipeline_panel.add_step("Fill Up", display, "region", region=region)
            self._status_var.set("區域已填充")
        except Exception as exc:
            self._show_error("Fill Up", exc)

    def _region_shape_trans(self, shape_type: str) -> None:
        """形狀變換。"""
        if self._current_region is None:
            messagebox.showwarning("警告", "請先產生區域。")
            return
        try:
            from core.region_ops import region_to_display_image
            mask = self._current_region.to_binary_mask()
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
            from core.region_ops import compute_region_properties
            from core.region import Region
            props = compute_region_properties(labels, self._current_region.source_image)
            region = Region(labels=labels, num_regions=num - 1, properties=props,
                          source_image=self._current_region.source_image,
                          source_shape=self._current_region.source_shape)
            img = self._get_current_image()
            display = region_to_display_image(region, img)
            self._current_region = region
            self.pipeline_panel.add_step(f"Shape Trans ({shape_type})", display, "region", region=region)
        except Exception as exc:
            self._show_error("Shape Trans", exc)

    def _region_morphology(self, op: str) -> None:
        """區域形態學操作。"""
        if self._current_region is None:
            messagebox.showwarning("警告", "請先產生區域。")
            return
        try:
            from core.region_ops import region_to_display_image, compute_region_properties
            from core.region import Region
            mask = self._current_region.to_binary_mask()
            ksize = 5
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
            if op == "erosion":
                result = cv2.erode(mask, kernel, iterations=1)
                name = f"區域侵蝕 k={ksize}"
            elif op == "dilation":
                result = cv2.dilate(mask, kernel, iterations=1)
                name = f"區域膨脹 k={ksize}"
            elif op == "opening":
                result = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                name = f"區域開運算 k={ksize}"
            elif op == "closing":
                result = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                name = f"區域閉運算 k={ksize}"
            else:
                return
            num, labels = cv2.connectedComponents(result, connectivity=8)
            labels = labels.astype(np.int32)
            props = compute_region_properties(labels, self._current_region.source_image)
            region = Region(labels=labels, num_regions=num - 1, properties=props,
                          source_image=self._current_region.source_image,
                          source_shape=self._current_region.source_shape)
            img = self._get_current_image()
            display = region_to_display_image(region, img)
            self._current_region = region
            self.pipeline_panel.add_step(name, display, "region", region=region)
            self._status_var.set(f"{name}: {region.num_regions} 個區域")
        except Exception as exc:
            self._show_error("形態學操作", exc)

    def _open_region_filter(self) -> None:
        """開啟篩選區域對話框。"""
        if self._current_region is None:
            messagebox.showwarning("警告", "請先產生區域。")
            return
        img = self._get_current_image()
        if img is None:
            return
        try:
            from gui.region_filter_dialog import RegionFilterDialog

            def on_accept(filtered_region, display_image, name):
                self._current_region = filtered_region
                self.pipeline_panel.add_step(name, display_image, "region", region=filtered_region)
                self._status_var.set(f"篩選完成: {filtered_region.num_regions} 個區域")

            RegionFilterDialog(self, self._current_region, img, on_accept=on_accept)
        except Exception as exc:
            self._show_error("開啟篩選對話框", exc)

    def _region_select_gray(self) -> None:
        """依灰度值篩選。"""
        if self._current_region is None:
            messagebox.showwarning("警告", "請先產生區域。")
            return
        try:
            from core.region_ops import select_shape, region_to_display_image
            # 簡易灰度篩選：以均值 > 128 為條件
            region = select_shape(self._current_region, "mean_value", 0, 128)
            img = self._get_current_image()
            display = region_to_display_image(region, img)
            self._current_region = region
            self.pipeline_panel.add_step(f"灰度篩選 ({region.num_regions})", display, "region", region=region)
        except Exception as exc:
            self._show_error("灰度篩選", exc)

    def _region_sort(self) -> None:
        """依面積排序區域。"""
        if self._current_region is None:
            messagebox.showwarning("警告", "請先產生區域。")
            return
        try:
            from core.region_ops import region_to_display_image
            # 依面積降序排列
            sorted_props = sorted(self._current_region.properties, key=lambda p: p.area, reverse=True)
            from core.region import Region
            import copy
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
            img = self._get_current_image()
            display = region_to_display_image(region, img)
            self._current_region = region
            self.pipeline_panel.add_step("排序區域 (面積)", display, "region", region=region)
        except Exception as exc:
            self._show_error("排序區域", exc)

    def _region_set_op(self, op: str) -> None:
        """區域集合操作。"""
        if self._current_region is None:
            messagebox.showwarning("警告", "請先產生區域。")
            return
        try:
            from core.region_ops import region_to_display_image
            if op == "complement":
                region = self._current_region.complement()
            else:
                messagebox.showinfo("提示", f"區域{op}需要兩個區域，目前僅有一個。\n請先分別產生兩個區域步驟。")
                return
            img = self._get_current_image()
            display = region_to_display_image(region, img)
            self._current_region = region
            self.pipeline_panel.add_step(f"區域補集 ({region.num_regions})", display, "region", region=region)
        except Exception as exc:
            self._show_error("集合操作", exc)

    def _open_blob_analysis(self) -> None:
        """開啟 Blob 分析對話框。"""
        img = self._get_current_image()
        if img is None:
            messagebox.showwarning("警告", "請先載入影像。")
            return
        try:
            from gui.blob_analysis import BlobAnalysisDialog

            def on_accept(steps):
                for step_name, step_img, step_region in steps:
                    self.pipeline_panel.add_step(step_name, step_img, "region", region=step_region)
                if steps:
                    last_region = steps[-1][2]
                    if last_region is not None:
                        self._current_region = last_region
                self._status_var.set("Blob 分析完成")

            BlobAnalysisDialog(self, img, on_accept=on_accept)
        except Exception as exc:
            self._show_error("Blob 分析", exc)

    # ================================================================== #
    #  影像處理運算子操作                                                   #
    # ================================================================== #

    def _apply_vision_op(self, op: str) -> None:
        """套用影像處理運算子。"""
        img = self._get_current_image()
        if img is None:
            messagebox.showwarning("警告", "請先載入影像。")
            return

        result = None
        name = ""

        try:
            from core import vision_ops as hops

            if op == "mean_image":
                result = hops.mean_image(img, 5)
                name = "均值濾波 k=5"
            elif op == "median_image":
                result = hops.median_image(img, 5)
                name = "中值濾波 k=5"
            elif op == "gauss_filter":
                result = hops.gauss_filter(img, 1.5)
                name = "高斯濾波 σ=1.5"
            elif op == "bilateral_filter":
                result = hops.bilateral_filter(img, 9, 75, 75)
                name = "雙邊濾波"
            elif op == "sharpen_image":
                result = hops.sharpen_image(img, 0.5)
                name = "銳化 0.5"
            elif op == "emphasize":
                result = hops.emphasize(img, 7, 1.5)
                name = "強調濾波"
            elif op == "laplace_filter":
                result = hops.laplace_filter(img)
                name = "Laplacian"
            elif op == "edges_canny":
                result = hops.edges_canny(img, 50, 150, 1.0)
                name = "Canny 邊緣"
            elif op == "sobel_filter":
                result = hops.sobel_filter(img, "both")
                name = "Sobel 邊緣"
            elif op == "prewitt_filter":
                result = hops.prewitt_filter(img)
                name = "Prewitt 邊緣"
            elif op == "gray_erosion":
                result = hops.gray_erosion(img, 5)
                name = "灰度侵蝕 k=5"
            elif op == "gray_dilation":
                result = hops.gray_dilation(img, 5)
                name = "灰度膨脹 k=5"
            elif op == "gray_opening":
                result = hops.gray_opening(img, 5)
                name = "灰度開運算 k=5"
            elif op == "gray_closing":
                result = hops.gray_closing(img, 5)
                name = "灰度閉運算 k=5"
            elif op == "top_hat":
                result = hops.top_hat(img, 9)
                name = "Top-hat k=9"
            elif op == "bottom_hat":
                result = hops.bottom_hat(img, 9)
                name = "Bottom-hat k=9"
            elif op == "rotate_90":
                result = hops.rotate_image(img, 90, "constant")
                name = "旋轉 90°"
            elif op == "rotate_180":
                result = hops.rotate_image(img, 180, "constant")
                name = "旋轉 180°"
            elif op == "rotate_270":
                result = hops.rotate_image(img, 270, "constant")
                name = "旋轉 270°"
            elif op == "mirror_h":
                result = hops.mirror_image(img, "horizontal")
                name = "水平鏡像"
            elif op == "mirror_v":
                result = hops.mirror_image(img, "vertical")
                name = "垂直鏡像"
            elif op == "zoom_50":
                result = hops.zoom_image(img, 0.5, 0.5)
                name = "縮放 50%"
            elif op == "zoom_200":
                result = hops.zoom_image(img, 2.0, 2.0)
                name = "縮放 200%"
            elif op == "rgb_to_gray":
                result = hops.rgb_to_gray(img)
                name = "轉灰階"
            elif op == "rgb_to_hsv":
                result = hops.rgb_to_hsv(img)
                name = "轉 HSV"
            elif op == "histogram_eq_vision":
                result = hops.histogram_eq(img)
                name = "直方圖均衡"
            elif op == "invert_image":
                result = hops.invert_image(img)
                name = "反色"
            elif op == "illuminate":
                result = hops.illuminate(img, 41, 1.0)
                name = "光照校正"
            elif op == "abs_image":
                result = hops.abs_image(img)
                name = "絕對值"
            elif op == "bright_up":
                result = hops.scale_image(img, 1.0, 30)
                name = "亮度 +30"
            elif op == "bright_down":
                result = hops.scale_image(img, 1.0, -30)
                name = "亮度 -30"
            elif op == "contrast_up":
                result = hops.scale_image(img, 1.3, 0)
                name = "對比度增強"
            elif op == "entropy_image":
                result = hops.entropy_image(img, 5)
                name = "熵影像 k=5"
            elif op == "deviation_image":
                result = hops.deviation_image(img, 5)
                name = "標準差影像 k=5"
            elif op == "local_min":
                result = hops.local_min(img, 5)
                name = "局部最小 k=5"
            elif op == "local_max":
                result = hops.local_max(img, 5)
                name = "局部最大 k=5"
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
                    name = f"條碼 ({len(results)})"
                else:
                    messagebox.showinfo("結果", "未偵測到條碼。")
                    return
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
                else:
                    messagebox.showinfo("結果", "未偵測到 QR Code。")
                    return
            elif op == "find_datamatrix":
                results = hops.find_datamatrix(img)
                if results:
                    result = img.copy()
                    if result.ndim == 2:
                        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
                    for r in results:
                        data = r.get("data", "")
                        pts = r.get("points")
                        if pts is not None:
                            pts_arr = np.array(pts, dtype=np.int32)
                            cv2.polylines(result, [pts_arr], True, (255, 0, 0), 2)
                    name = f"DataMatrix ({len(results)})"
                else:
                    messagebox.showinfo("結果", "未偵測到 DataMatrix。")
                    return
            # -- Synced from dl_anomaly --
            elif op == "rgb_to_hls":
                result = hops.rgb_to_hls(img)
                name = "轉 HLS"
            elif op == "log_image":
                result = hops.log_image(img)
                name = "對數變換"
            elif op == "exp_image":
                result = hops.exp_image(img)
                name = "指數變換"
            elif op == "gamma_image":
                result = hops.gamma_image(img, 0.5)
                name = "Gamma 校正"
            elif op == "fft_image":
                result = hops.fft_image(img)
                name = "FFT 頻譜"
            elif op == "freq_filter_lowpass":
                result = hops.freq_filter(img, "lowpass", 30)
                name = "低通濾波"
            elif op == "freq_filter_highpass":
                result = hops.freq_filter(img, "highpass", 30)
                name = "高通濾波"
            elif op == "clahe":
                result = hops.clahe(img)
                name = "CLAHE"
            # -- Segmentation --
            elif op == "watersheds":
                result = hops.watersheds(img)
                name = "分水嶺"
            elif op == "distance_transform":
                result = hops.distance_transform(img)
                name = "距離變換"
            elif op == "skeleton":
                result = hops.skeleton(img)
                name = "骨架化"
            # -- Feature Points --
            elif op == "points_harris":
                result = hops.points_harris(img)
                name = "Harris 角點"
            elif op == "points_shi_tomasi":
                result = hops.points_shi_tomasi(img)
                name = "Shi-Tomasi 特徵點"
            # -- Hough --
            elif op == "hough_lines":
                result = hops.hough_lines(img)
                name = "Hough 直線"
            elif op == "hough_circles":
                result = hops.hough_circles(img)
                name = "Hough 圓"
            # -- Misc --
            elif op == "grab_image":
                result = hops.grab_image(0)
                if result is None:
                    messagebox.showwarning("警告", "無法開啟相機。")
                    return
                name = "相機擷取"
            elif op == "abs_diff_image":
                # Needs previous image - use a simple abs diff with a blurred version
                blurred = cv2.GaussianBlur(img, (5, 5), 0)
                result = hops.abs_diff_image(img, blurred)
                name = "絕對差分 (vs 模糊)"
            # -- Synced ops (additional) --
            elif op == "gauss_blur":
                result = hops.gauss_blur(img, 5)
                name = "高斯模糊 k=5"
            elif op == "gray_opening_shape":
                result = hops.gray_opening_shape(img)
                name = "灰度開運算 (形狀)"
            elif op == "gray_closing_shape":
                result = hops.gray_closing_shape(img)
                name = "灰度閉運算 (形狀)"
            elif op == "dyn_threshold":
                smoothed = hops.mean_image(img, 15)
                result = hops.dyn_threshold(img, smoothed, 5, "not_equal")
                name = "動態閾值分割"
            elif op == "var_threshold":
                result = hops.var_threshold(img, 15, 15, 0.2, 2.0, "dark")
                name = "可變閾值"
            elif op == "local_threshold":
                result = hops.local_threshold(img, "adapted_std_deviation", "dark", 15, 0.2)
                name = "局部閾值"
            elif op == "texture_laws":
                result = hops.texture_laws(img, "L5E5")
                name = "Laws 紋理 L5E5"
            elif op == "mean_curvature":
                result = hops.mean_curvature(img)
                name = "平均曲率"
            elif op == "decompose3":
                ch0, ch1, ch2 = hops.decompose3(img)
                result = ch0  # Show first channel
                name = "通道分離 (Ch0)"
            else:
                messagebox.showwarning("警告", f"未知的影像處理運算子: {op}")
                return

            if result is not None:
                self.pipeline_panel.add_step(name, result, "vision")
                self.set_status_success(f"影像處理: {name}")
                self._history_panel.add_entry("影像處理", name)

        except Exception as exc:
            logger.exception("影像處理運算子失敗: %s", op)
            self._show_error(f"影像處理運算子 [{op}]", exc)

    # ================================================================== #
    #  腳本編輯器                                                          #
    # ================================================================== #

    def _toggle_script_editor(self) -> None:
        """切換腳本編輯器面板顯示。"""
        if self._script_editor_visible:
            if self._script_editor is not None:
                self._script_editor.pack_forget()
            self._script_editor_visible = False
            self._status_var.set("腳本編輯器已關閉")
        else:
            try:
                from gui.script_editor import ScriptEditor
                if self._script_editor is None:
                    self._script_editor = ScriptEditor(self, app=self)
                self._script_editor.pack(fill=tk.BOTH, expand=False, side=tk.BOTTOM, before=self._main_paned.master if hasattr(self._main_paned, 'master') else None)
                # Place it at the bottom
                self._script_editor.pack_configure(fill=tk.BOTH, expand=False)
            except Exception as exc:
                self._show_error("腳本編輯器載入", exc)
                return
            self._script_editor_visible = True
            self._status_var.set("腳本編輯器已開啟 (F9 執行)")

    def _run_script(self) -> None:
        """執行腳本編輯器中的腳本。"""
        if self._script_editor is not None and self._script_editor_visible:
            self._script_editor.run_script()

    # ================================================================== #
    #  共用工具方法                                                        #
    # ================================================================== #

    def _show_error(self, context: str, exc: Exception) -> None:
        """以結構化錯誤對話框顯示錯誤。"""
        show_error(self, context, exc)

    def set_status_success(self, message: str) -> None:
        """在狀態列顯示綠色成功訊息，2 秒後恢復。"""
        self._status_var.set(message)
        self._status_label.configure(style="Success.Status.TLabel")

        def _revert():
            self._status_label.configure(style="Status.TLabel")

        self.after(2000, _revert)

    def _run_in_bg(self, func, on_done=None, on_error=None, status_msg=""):
        """在背景執行緒執行 func，完成後回到主執行緒。"""
        if status_msg:
            self.set_status(status_msg)
            self.update_idletasks()
        self._progress.start_indeterminate()

        def _worker():
            try:
                result = func()

                def _finish(r=result):
                    if self._closing:
                        return
                    self._progress.stop()
                    if on_done is not None:
                        on_done(r)

                self.after(0, _finish)
            except Exception as exc:
                def _fail(e=exc):
                    if self._closing:
                        return
                    self._progress.stop()
                    if on_error is not None:
                        on_error(e)
                    else:
                        self._show_error("背景操作", e)

                self.after(0, _fail)

        threading.Thread(target=_worker, daemon=True).start()

    def _show_shortcuts_dialog(self) -> None:
        """以對話框顯示鍵盤快捷鍵清單 (F1)。"""
        shortcuts = (
            "鍵盤快捷鍵:\n\n"
            "Ctrl+O    開啟影像\n"
            "Ctrl+S    儲存目前影像\n"
            "Ctrl+Z    復原 (上一步驟)\n"
            "Ctrl+Y    重做 (下一步驟)\n"
            "Ctrl+I    像素值檢查器\n"
            "Ctrl+T    閾值分割\n"
            "Space     適配視窗\n"
            "+/-       放大/縮小\n"
            "F1        快捷鍵說明\n"
            "F5        執行檢測\n"
            "F8        腳本編輯器\n"
            "F9        執行腳本\n"
            "Delete    刪除目前步驟\n"
            "Escape    取消目前工具\n"
            "\n滑鼠操作:\n"
            "滾輪      縮放\n"
            "左鍵拖曳   平移\n"
            "雙擊      適配視窗\n"
        )
        messagebox.showinfo("鍵盤快捷鍵", shortcuts)

    def _cancel_current_tool(self) -> None:
        """取消目前工具 / 關閉非模態面板 (Escape)。"""
        if self._pixel_inspector is not None and self._pixel_inspector.winfo_exists():
            self._pixel_inspector.destroy()
            self._pixel_inspector = None
            self.set_status("像素檢查器已關閉")
            return
        if self._script_editor_visible:
            self._toggle_script_editor()
            return
        self.set_status("就緒")

    # ================================================================== #
    #  關閉                                                                #
    # ================================================================== #

    # ================================================================== #
    #  Phase 1 Tools: Shape Matching, Metrology, ROI Manager              #
    # ================================================================== #

    def _vm_add_pipeline_step(self, name: str, array, op_meta=None) -> None:
        """Adapter for dialog callbacks - wraps pipeline_panel.add_step."""
        self.pipeline_panel.add_step(name, array, "process")

    def _open_shape_matching(self) -> None:
        from gui.shape_matching_dialog import ShapeMatchingDialog
        ShapeMatchingDialog(
            self,
            get_current_image=self._get_current_image,
            add_pipeline_step=self._vm_add_pipeline_step,
            set_status=self.set_status,
        )

    def _open_metrology(self) -> None:
        from gui.metrology_dialog import MetrologyDialog
        MetrologyDialog(
            self,
            get_current_image=self._get_current_image,
            add_pipeline_step=self._vm_add_pipeline_step,
            set_status=self.set_status,
        )

    def _open_roi_manager(self) -> None:
        from gui.roi_dialog import ROIManagerDialog
        ROIManagerDialog(
            self,
            get_current_image=self._get_current_image,
            add_pipeline_step=self._vm_add_pipeline_step,
            set_status=self.set_status,
            viewer=self.image_viewer,
        )

    def _open_advanced_models(self) -> None:
        from gui.advanced_models_dialog import AdvancedModelsDialog
        from dl_anomaly.config import Config as DLConfig
        dl_config = DLConfig()
        AdvancedModelsDialog(
            self,
            config=dl_config,
            get_current_image=self._get_current_image,
            add_pipeline_step=self._vm_add_pipeline_step,
            set_status=self.set_status,
        )

    def _open_inspection_tools(self) -> None:
        from gui.inspection_tools_dialog import InspectionToolsDialog
        InspectionToolsDialog(
            self,
            get_current_image=self._get_current_image,
            add_pipeline_step=self._vm_add_pipeline_step,
            set_status=self.set_status,
        )

    def _open_engineering_tools(self) -> None:
        from gui.engineering_tools_dialog import EngineeringToolsDialog
        EngineeringToolsDialog(
            self,
            get_current_image=self._get_current_image,
            add_pipeline_step=self._vm_add_pipeline_step,
            set_status=self.set_status,
        )

    def _open_mvp_tools(self) -> None:
        try:
            from gui.mvp_tools_dialog import MVPToolsDialog
        except ImportError as exc:
            from tkinter import messagebox
            messagebox.showerror("匯入錯誤", f"無法載入 MVP 工具模組：\n{exc}")
            return
        MVPToolsDialog(
            self,
            get_current_image=self._get_current_image,
            add_pipeline_step=self._vm_add_pipeline_step,
            set_status=self.set_status,
        )

    # ================================================================== #
    #  即時檢測回調                                                        #
    # ================================================================== #

    def _on_live_start(self, source: str, interval: int) -> None:
        """啟動即時檢測。"""
        if self.model is None or not self.model.is_trained:
            messagebox.showwarning("警告", "請先載入或訓練模型再啟動即時檢測。")
            self._live_panel.stop_inspection()
            return
        self.set_status(f"即時檢測啟動中... 來源: {source}, 間隔: {interval}ms")

    def _on_live_stop(self) -> None:
        """停止即時檢測。"""
        self.set_status("即時檢測已停止")

    def _on_close(self) -> None:
        """關閉應用程式。"""
        self._closing = True
        if self._live_panel.is_running:
            self._live_panel.stop_inspection()
        self._app_state.save_geometry(self)
        self._app_state.save_sash_positions(self._main_paned)
        self.destroy()
