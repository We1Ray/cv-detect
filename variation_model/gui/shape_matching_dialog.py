"""
gui/shape_matching_dialog.py - Shape-Based Matching dialog.

Provides a two-tab dialog (ttk.Notebook) for creating shape models and
searching for matches in the current image.  Uses the core.shape_matching
module for all heavy computation.

Tab 1 -- Create Model: crop template from the current image or load from
         file, set pyramid / angle / scale parameters, build model.
Tab 2 -- Find Matches: set score / greediness / overlap parameters, search,
         display results in a treeview, apply annotated result to the pipeline.
"""

from __future__ import annotations

import logging
import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk

from core.shape_matching import (
    MatchResult,
    ShapeModel,
    create_shape_model,
    draw_shape_matches,
    find_shape_model,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Theme constants                                                             #
# --------------------------------------------------------------------------- #
_BG = "#2b2b2b"
_BG_MEDIUM = "#3c3c3c"
_FG = "#e0e0e0"
_FG_DIM = "#cccccc"
_ACCENT = "#0078d4"
_ACTIVE_BG = "#3a3a5c"
_CANVAS_BG = "#1e1e1e"

_TEMPLATE_PREVIEW_SIZE = 200


class ShapeMatchingDialog(tk.Toplevel):
    """Modal dialog for Vision-style shape-based matching.

    Parameters
    ----------
    master : tk.Widget
        Parent widget (typically the main application window).
    get_current_image : callable
        ``() -> Optional[np.ndarray]`` -- returns the currently active image
        in the viewer, or ``None`` if no image is loaded.
    add_pipeline_step : callable
        ``(name, array, op_meta=None)`` -- adds an annotated image to the
        pipeline history.
    set_status : callable
        ``(text)`` -- updates the status bar text.
    """

    def __init__(
        self,
        master: tk.Widget,
        get_current_image: Callable[[], Optional[np.ndarray]],
        add_pipeline_step: Callable,
        set_status: Callable[[str], None],
    ) -> None:
        super().__init__(master)
        self.title("Shape-Based Matching")
        self.configure(bg=_BG)
        self.resizable(False, False)

        # Modal behaviour
        self.transient(master)
        self.grab_set()

        # Callbacks
        self._get_current_image = get_current_image
        self._add_pipeline_step = add_pipeline_step
        self._set_status = set_status

        # State
        self._shape_model: Optional[ShapeModel] = None
        self._template_image: Optional[np.ndarray] = None
        self._match_results: List[MatchResult] = []

        # PhotoImage references (prevent GC)
        self._template_photo: Optional[ImageTk.PhotoImage] = None

        self._build_styles()
        self._build_ui()

        # Centre on parent
        self.update_idletasks()
        x = master.winfo_x() + (master.winfo_width() - self.winfo_width()) // 2
        y = master.winfo_y() + (master.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{x}+{y}")

        self.protocol("WM_DELETE_WINDOW", self._close)

    # ------------------------------------------------------------------ #
    #  ttk styles                                                         #
    # ------------------------------------------------------------------ #

    def _build_styles(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")

        style.configure("Dark.TFrame", background=_BG)
        style.configure("Dark.TLabel", background=_BG, foreground=_FG)
        style.configure(
            "Dark.TButton",
            background=_BG_MEDIUM,
            foreground=_FG,
            padding=(10, 4),
        )
        style.map(
            "Dark.TButton",
            background=[("active", _ACTIVE_BG)],
            foreground=[("active", "#ffffff")],
        )
        style.configure(
            "Accent.TButton",
            background=_ACCENT,
            foreground="#ffffff",
            padding=(14, 5),
        )
        style.map(
            "Accent.TButton",
            background=[("active", "#005a9e")],
        )
        style.configure(
            "Dark.TNotebook",
            background=_BG,
            borderwidth=0,
        )
        style.configure(
            "Dark.TNotebook.Tab",
            background=_BG_MEDIUM,
            foreground=_FG,
            padding=(12, 4),
        )
        style.map(
            "Dark.TNotebook.Tab",
            background=[("selected", _ACCENT)],
            foreground=[("selected", "#ffffff")],
        )
        style.configure(
            "Dark.TSpinbox",
            fieldbackground=_BG_MEDIUM,
            background=_BG_MEDIUM,
            foreground=_FG,
            insertcolor=_FG,
        )
        style.configure(
            "Dark.Treeview",
            background=_BG_MEDIUM,
            foreground=_FG_DIM,
            fieldbackground=_BG_MEDIUM,
            rowheight=22,
        )
        style.configure(
            "Dark.Treeview.Heading",
            background=_BG,
            foreground=_FG,
            font=("", 9, "bold"),
        )
        style.map(
            "Dark.Treeview",
            background=[("selected", _ACCENT)],
            foreground=[("selected", "#ffffff")],
        )

    # ------------------------------------------------------------------ #
    #  UI construction                                                     #
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        self._notebook = ttk.Notebook(self, style="Dark.TNotebook")
        self._notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Tab 1 - Create Model
        tab_create = tk.Frame(self._notebook, bg=_BG)
        self._notebook.add(tab_create, text=" 建立模型 ")
        self._build_create_tab(tab_create)

        # Tab 2 - Find Matches
        tab_find = tk.Frame(self._notebook, bg=_BG)
        self._notebook.add(tab_find, text=" 搜尋匹配 ")
        self._build_find_tab(tab_find)

        # Bottom close button
        btn_frame = tk.Frame(self, bg=_BG)
        btn_frame.pack(fill=tk.X, padx=8, pady=(0, 8))

        tk.Button(
            btn_frame,
            text="關閉",
            bg=_BG_MEDIUM,
            fg=_FG,
            activebackground=_ACTIVE_BG,
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=20,
            pady=4,
            font=("", 10),
            command=self._close,
        ).pack(side=tk.RIGHT)

    # ------------------------------------------------------------------ #
    #  Tab 1: Create Model                                                 #
    # ------------------------------------------------------------------ #

    def _build_create_tab(self, parent: tk.Frame) -> None:
        # -- Template source buttons --
        src_frame = tk.LabelFrame(
            parent,
            text=" 模板來源 ",
            bg=_BG,
            fg=_FG,
            font=("", 10, "bold"),
            padx=8,
            pady=6,
        )
        src_frame.pack(fill=tk.X, padx=8, pady=(8, 4))

        btn_row = tk.Frame(src_frame, bg=_BG)
        btn_row.pack(fill=tk.X)

        tk.Button(
            btn_row,
            text="從目前圖片裁切",
            bg=_BG_MEDIUM,
            fg=_FG,
            activebackground=_ACTIVE_BG,
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=10,
            pady=3,
            command=self._crop_from_current,
        ).pack(side=tk.LEFT, padx=(0, 6))

        tk.Button(
            btn_row,
            text="從檔案載入",
            bg=_BG_MEDIUM,
            fg=_FG,
            activebackground=_ACTIVE_BG,
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=10,
            pady=3,
            command=self._load_from_file,
        ).pack(side=tk.LEFT)

        # -- Template preview --
        preview_frame = tk.LabelFrame(
            parent,
            text=" 模板預覽 ",
            bg=_BG,
            fg=_FG,
            font=("", 9, "bold"),
            padx=4,
            pady=4,
        )
        preview_frame.pack(fill=tk.X, padx=8, pady=4)

        self._template_label = tk.Label(
            preview_frame,
            text="(尚未載入模板)",
            bg=_CANVAS_BG,
            fg="#888888",
            width=30,
            height=12,
            anchor=tk.CENTER,
        )
        self._template_label.pack(padx=4, pady=4)

        # -- Parameters --
        param_frame = tk.LabelFrame(
            parent,
            text=" 模型參數 ",
            bg=_BG,
            fg=_FG,
            font=("", 10, "bold"),
            padx=8,
            pady=6,
        )
        param_frame.pack(fill=tk.X, padx=8, pady=4)

        # Pyramid Levels
        row_pyr = tk.Frame(param_frame, bg=_BG)
        row_pyr.pack(fill=tk.X, pady=2)
        tk.Label(
            row_pyr, text="金字塔層數:", bg=_BG, fg=_FG, font=("", 9),
            width=14, anchor=tk.E,
        ).pack(side=tk.LEFT)
        self._pyramid_var = tk.IntVar(value=4)
        ttk.Spinbox(
            row_pyr,
            from_=1,
            to=6,
            textvariable=self._pyramid_var,
            width=6,
            style="Dark.TSpinbox",
        ).pack(side=tk.LEFT, padx=4)

        # Min Contrast
        row_contrast = tk.Frame(param_frame, bg=_BG)
        row_contrast.pack(fill=tk.X, pady=2)
        tk.Label(
            row_contrast, text="最小對比度:", bg=_BG, fg=_FG, font=("", 9),
            width=14, anchor=tk.E,
        ).pack(side=tk.LEFT)
        self._contrast_var = tk.IntVar(value=30)
        ttk.Spinbox(
            row_contrast,
            from_=10,
            to=255,
            textvariable=self._contrast_var,
            width=6,
            style="Dark.TSpinbox",
        ).pack(side=tk.LEFT, padx=4)

        # Angle Range
        row_angle = tk.Frame(param_frame, bg=_BG)
        row_angle.pack(fill=tk.X, pady=2)
        tk.Label(
            row_angle, text="角度範圍 (°):", bg=_BG, fg=_FG, font=("", 9),
            width=14, anchor=tk.E,
        ).pack(side=tk.LEFT)
        self._angle_start_var = tk.StringVar(value="0")
        tk.Entry(
            row_angle,
            textvariable=self._angle_start_var,
            bg=_BG_MEDIUM,
            fg=_FG,
            insertbackground=_FG,
            font=("Consolas", 10),
            width=6,
            relief=tk.FLAT,
            highlightthickness=1,
            highlightcolor=_ACCENT,
            highlightbackground=_BG_MEDIUM,
        ).pack(side=tk.LEFT, padx=4)
        tk.Label(
            row_angle, text="~", bg=_BG, fg=_FG, font=("", 9),
        ).pack(side=tk.LEFT)
        self._angle_extent_var = tk.StringVar(value="360")
        tk.Entry(
            row_angle,
            textvariable=self._angle_extent_var,
            bg=_BG_MEDIUM,
            fg=_FG,
            insertbackground=_FG,
            font=("Consolas", 10),
            width=6,
            relief=tk.FLAT,
            highlightthickness=1,
            highlightcolor=_ACCENT,
            highlightbackground=_BG_MEDIUM,
        ).pack(side=tk.LEFT, padx=4)

        # Scale Range
        row_scale = tk.Frame(param_frame, bg=_BG)
        row_scale.pack(fill=tk.X, pady=2)
        tk.Label(
            row_scale, text="縮放範圍:", bg=_BG, fg=_FG, font=("", 9),
            width=14, anchor=tk.E,
        ).pack(side=tk.LEFT)
        self._scale_min_var = tk.StringVar(value="0.8")
        tk.Entry(
            row_scale,
            textvariable=self._scale_min_var,
            bg=_BG_MEDIUM,
            fg=_FG,
            insertbackground=_FG,
            font=("Consolas", 10),
            width=6,
            relief=tk.FLAT,
            highlightthickness=1,
            highlightcolor=_ACCENT,
            highlightbackground=_BG_MEDIUM,
        ).pack(side=tk.LEFT, padx=4)
        tk.Label(
            row_scale, text="~", bg=_BG, fg=_FG, font=("", 9),
        ).pack(side=tk.LEFT)
        self._scale_max_var = tk.StringVar(value="1.2")
        tk.Entry(
            row_scale,
            textvariable=self._scale_max_var,
            bg=_BG_MEDIUM,
            fg=_FG,
            insertbackground=_FG,
            font=("Consolas", 10),
            width=6,
            relief=tk.FLAT,
            highlightthickness=1,
            highlightcolor=_ACCENT,
            highlightbackground=_BG_MEDIUM,
        ).pack(side=tk.LEFT, padx=4)

        # -- Create Model button --
        tk.Button(
            parent,
            text="建立模型",
            bg=_ACCENT,
            fg="#ffffff",
            activebackground="#005a9e",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=20,
            pady=4,
            font=("", 10, "bold"),
            command=self._create_model,
        ).pack(padx=8, pady=(8, 4))

        # -- Model status label --
        self._model_status_var = tk.StringVar(value="")
        self._model_status_label = tk.Label(
            parent,
            textvariable=self._model_status_var,
            bg=_BG,
            fg="#88cc88",
            font=("Consolas", 9),
            anchor=tk.W,
        )
        self._model_status_label.pack(fill=tk.X, padx=12, pady=(0, 8))

    # ------------------------------------------------------------------ #
    #  Tab 2: Find Matches                                                 #
    # ------------------------------------------------------------------ #

    def _build_find_tab(self, parent: tk.Frame) -> None:
        # -- Search Parameters --
        param_frame = tk.LabelFrame(
            parent,
            text=" 搜尋參數 ",
            bg=_BG,
            fg=_FG,
            font=("", 10, "bold"),
            padx=8,
            pady=6,
        )
        param_frame.pack(fill=tk.X, padx=8, pady=(8, 4))

        # Min Score
        row_score = tk.Frame(param_frame, bg=_BG)
        row_score.pack(fill=tk.X, pady=2)
        tk.Label(
            row_score, text="最小分數:", bg=_BG, fg=_FG, font=("", 9),
            width=12, anchor=tk.E,
        ).pack(side=tk.LEFT)
        self._min_score_var = tk.DoubleVar(value=0.5)
        self._min_score_scale = tk.Scale(
            row_score,
            from_=0.1,
            to=1.0,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            variable=self._min_score_var,
            bg=_BG,
            fg=_FG,
            troughcolor=_BG_MEDIUM,
            highlightthickness=0,
            sliderrelief=tk.FLAT,
            length=200,
        )
        self._min_score_scale.pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)

        # Max Matches
        row_max = tk.Frame(param_frame, bg=_BG)
        row_max.pack(fill=tk.X, pady=2)
        tk.Label(
            row_max, text="最大數量:", bg=_BG, fg=_FG, font=("", 9),
            width=12, anchor=tk.E,
        ).pack(side=tk.LEFT)
        self._max_matches_var = tk.IntVar(value=1)
        ttk.Spinbox(
            row_max,
            from_=1,
            to=50,
            textvariable=self._max_matches_var,
            width=6,
            style="Dark.TSpinbox",
        ).pack(side=tk.LEFT, padx=4)

        # Greediness
        row_greed = tk.Frame(param_frame, bg=_BG)
        row_greed.pack(fill=tk.X, pady=2)
        tk.Label(
            row_greed, text="貪婪度:", bg=_BG, fg=_FG, font=("", 9),
            width=12, anchor=tk.E,
        ).pack(side=tk.LEFT)
        self._greediness_var = tk.DoubleVar(value=0.9)
        tk.Scale(
            row_greed,
            from_=0.0,
            to=1.0,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            variable=self._greediness_var,
            bg=_BG,
            fg=_FG,
            troughcolor=_BG_MEDIUM,
            highlightthickness=0,
            sliderrelief=tk.FLAT,
            length=200,
        ).pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)

        # Max Overlap
        row_overlap = tk.Frame(param_frame, bg=_BG)
        row_overlap.pack(fill=tk.X, pady=2)
        tk.Label(
            row_overlap, text="最大重疊:", bg=_BG, fg=_FG, font=("", 9),
            width=12, anchor=tk.E,
        ).pack(side=tk.LEFT)
        self._max_overlap_var = tk.DoubleVar(value=0.5)
        tk.Scale(
            row_overlap,
            from_=0.0,
            to=1.0,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            variable=self._max_overlap_var,
            bg=_BG,
            fg=_FG,
            troughcolor=_BG_MEDIUM,
            highlightthickness=0,
            sliderrelief=tk.FLAT,
            length=200,
        ).pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)

        # -- Search button --
        tk.Button(
            parent,
            text="搜尋",
            bg=_ACCENT,
            fg="#ffffff",
            activebackground="#005a9e",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=20,
            pady=4,
            font=("", 10, "bold"),
            command=self._search,
        ).pack(padx=8, pady=(8, 4))

        # -- Results Treeview --
        result_frame = tk.LabelFrame(
            parent,
            text=" 匹配結果 ",
            bg=_BG,
            fg=_FG,
            font=("", 10, "bold"),
            padx=4,
            pady=4,
        )
        result_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        columns = ("#", "Row", "Col", "Angle(\u00b0)", "Scale", "Score")
        self._result_tree = ttk.Treeview(
            result_frame,
            columns=columns,
            show="headings",
            height=8,
            style="Dark.Treeview",
        )

        col_widths = {"#": 40, "Row": 80, "Col": 80, "Angle(\u00b0)": 80, "Scale": 70, "Score": 70}
        for col in columns:
            self._result_tree.heading(col, text=col)
            self._result_tree.column(
                col,
                width=col_widths.get(col, 70),
                anchor=tk.CENTER,
                minwidth=40,
            )

        tree_scroll = ttk.Scrollbar(
            result_frame, orient=tk.VERTICAL, command=self._result_tree.yview,
        )
        self._result_tree.configure(yscrollcommand=tree_scroll.set)
        self._result_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # -- Apply button --
        tk.Button(
            parent,
            text="套用結果",
            bg=_BG_MEDIUM,
            fg=_FG,
            activebackground=_ACTIVE_BG,
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=20,
            pady=4,
            font=("", 10),
            command=self._apply_result,
        ).pack(padx=8, pady=(4, 8))

    # ------------------------------------------------------------------ #
    #  Template acquisition                                                #
    # ------------------------------------------------------------------ #

    def _crop_from_current(self) -> None:
        """Crop the template from the viewer's active region selection."""
        image = self._get_current_image()
        if image is None:
            messagebox.showwarning(
                "無圖片",
                "目前沒有載入任何圖片。",
                parent=self,
            )
            return

        # Try to get region selection from the viewer widget.
        # The viewer is expected to be accessible via master._viewer and to
        # expose _current_region_sel as (x, y, w, h) in image coordinates.
        region_sel: Optional[Tuple[int, int, int, int]] = None
        viewer = getattr(self.master, "_viewer", None)
        if viewer is not None:
            region_sel = getattr(viewer, "_current_region_sel", None)

        if region_sel is None:
            messagebox.showinfo(
                "請先選取區域",
                "請先在圖片上拖曳選取一個矩形區域作為模板。\n"
                "（使用「區域選取」工具畫出矩形範圍）",
                parent=self,
            )
            return

        x, y, w, h = region_sel
        if w <= 0 or h <= 0:
            messagebox.showwarning(
                "無效區域",
                "選取的區域尺寸無效，請重新選取。",
                parent=self,
            )
            return

        # Crop template from image
        self._template_image = image[y : y + h, x : x + w].copy()
        self._update_template_preview()
        self._set_status(f"已裁切模板: {w} x {h} px")

    def _load_from_file(self) -> None:
        """Load a template image from file."""
        path = filedialog.askopenfilename(
            title="選擇模板圖片",
            filetypes=[
                ("圖片檔案", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif"),
                ("所有檔案", "*.*"),
            ],
            parent=self,
        )
        if not path:
            return

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            messagebox.showerror(
                "讀取失敗",
                f"無法讀取圖片:\n{path}",
                parent=self,
            )
            return

        self._template_image = img
        self._update_template_preview()
        h, w = img.shape[:2]
        self._set_status(f"已載入模板: {w} x {h} px")

    def _update_template_preview(self) -> None:
        """Update the template preview label with the current template image."""
        if self._template_image is None:
            self._template_label.configure(
                image="", text="(尚未載入模板)",
            )
            self._template_photo = None
            return

        img = self._template_image
        # Convert to RGB for PIL
        if img.ndim == 2:
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3 and img.shape[2] == 4:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        elif img.ndim == 3:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            rgb = img

        if rgb.dtype != np.uint8:
            mn, mx = rgb.min(), rgb.max()
            if mx - mn > 0:
                rgb = ((rgb - mn) / (mx - mn) * 255).astype(np.uint8)
            else:
                rgb = np.zeros_like(rgb, dtype=np.uint8)

        pil_img = Image.fromarray(rgb)

        # Scale to fit preview area while preserving aspect ratio
        img_w, img_h = pil_img.size
        scale = min(
            _TEMPLATE_PREVIEW_SIZE / max(img_w, 1),
            _TEMPLATE_PREVIEW_SIZE / max(img_h, 1),
        )
        if scale < 1.0:
            new_w = max(1, int(img_w * scale))
            new_h = max(1, int(img_h * scale))
            pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)

        self._template_photo = ImageTk.PhotoImage(pil_img)
        self._template_label.configure(
            image=self._template_photo, text="",
        )

    # ------------------------------------------------------------------ #
    #  Model creation                                                      #
    # ------------------------------------------------------------------ #

    def _create_model(self) -> None:
        """Build a shape model from the loaded template."""
        if self._template_image is None:
            messagebox.showwarning(
                "無模板",
                "請先載入或裁切一個模板圖片。",
                parent=self,
            )
            return

        try:
            num_levels = self._pyramid_var.get()
            min_contrast = self._contrast_var.get()
            angle_start_deg = float(self._angle_start_var.get())
            angle_extent_deg = float(self._angle_extent_var.get())
            scale_min = float(self._scale_min_var.get())
            scale_max = float(self._scale_max_var.get())
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror(
                "參數錯誤",
                f"請輸入有效的數值參數。\n{exc}",
                parent=self,
            )
            return

        angle_start_rad = math.radians(angle_start_deg)
        angle_extent_rad = math.radians(angle_extent_deg)

        self._set_status("正在建立 Shape Model ...")
        self.update_idletasks()

        try:
            model = create_shape_model(
                template=self._template_image,
                num_levels=num_levels,
                angle_start=angle_start_rad,
                angle_extent=angle_extent_rad,
                scale_min=scale_min,
                scale_max=scale_max,
                min_contrast=min_contrast,
            )
        except Exception as exc:
            logger.exception("create_shape_model failed")
            messagebox.showerror(
                "建立失敗",
                f"建立 Shape Model 時發生錯誤:\n{exc}",
                parent=self,
            )
            self._set_status("Shape Model 建立失敗")
            return

        self._shape_model = model

        n_pts = len(model.contour_points)
        info = (
            f"模型已建立: {n_pts} 輪廓點, "
            f"{model.num_levels} 層金字塔, "
            f"角度=[{angle_start_deg:.0f}, "
            f"{angle_start_deg + angle_extent_deg:.0f}]\u00b0, "
            f"縮放=[{scale_min:.2f}, {scale_max:.2f}]"
        )
        self._model_status_var.set(info)
        self._set_status("Shape Model 建立完成")

    # ------------------------------------------------------------------ #
    #  Search                                                              #
    # ------------------------------------------------------------------ #

    def _search(self) -> None:
        """Search for shape model matches in the current image."""
        if self._shape_model is None:
            messagebox.showwarning(
                "無模型",
                "請先建立 Shape Model。",
                parent=self,
            )
            return

        image = self._get_current_image()
        if image is None:
            messagebox.showwarning(
                "無圖片",
                "目前沒有載入任何圖片。",
                parent=self,
            )
            return

        try:
            min_score = self._min_score_var.get()
            max_matches = self._max_matches_var.get()
            greediness = self._greediness_var.get()
            max_overlap = self._max_overlap_var.get()
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror(
                "參數錯誤",
                f"請輸入有效的數值參數。\n{exc}",
                parent=self,
            )
            return

        self._set_status("正在搜尋匹配 ...")
        self.update_idletasks()

        try:
            results = find_shape_model(
                image=image,
                model=self._shape_model,
                min_score=min_score,
                num_matches=max_matches,
                max_overlap=max_overlap,
                greediness=greediness,
            )
        except Exception as exc:
            logger.exception("find_shape_model failed")
            messagebox.showerror(
                "搜尋失敗",
                f"搜尋時發生錯誤:\n{exc}",
                parent=self,
            )
            self._set_status("搜尋失敗")
            return

        self._match_results = results
        self._update_result_tree()

        if results:
            self._set_status(
                f"找到 {len(results)} 個匹配, "
                f"最佳分數 = {results[0].score:.3f}"
            )
        else:
            self._set_status("未找到任何匹配")

    def _update_result_tree(self) -> None:
        """Refresh the results treeview with current match results."""
        for item in self._result_tree.get_children():
            self._result_tree.delete(item)

        for idx, m in enumerate(self._match_results, start=1):
            self._result_tree.insert(
                "",
                tk.END,
                values=(
                    idx,
                    f"{m.row:.1f}",
                    f"{m.col:.1f}",
                    f"{math.degrees(m.angle):.1f}",
                    f"{m.scale:.3f}",
                    f"{m.score:.4f}",
                ),
            )

    # ------------------------------------------------------------------ #
    #  Apply result                                                        #
    # ------------------------------------------------------------------ #

    def _apply_result(self) -> None:
        """Draw matches on the current image and add to pipeline."""
        if not self._match_results:
            messagebox.showinfo(
                "無結果",
                "沒有可套用的匹配結果。請先執行搜尋。",
                parent=self,
            )
            return

        image = self._get_current_image()
        if image is None:
            messagebox.showwarning(
                "無圖片",
                "目前沒有載入任何圖片。",
                parent=self,
            )
            return

        try:
            vis = draw_shape_matches(
                image=image,
                matches=self._match_results,
                model=self._shape_model,
                color=(0, 255, 0),
                thickness=2,
            )
        except Exception as exc:
            logger.exception("draw_shape_matches failed")
            messagebox.showerror(
                "繪製失敗",
                f"繪製匹配結果時發生錯誤:\n{exc}",
                parent=self,
            )
            return

        n = len(self._match_results)
        best_score = self._match_results[0].score
        name = f"Shape Match ({n}), score={best_score:.3f}"

        op_meta = {
            "op": "find_shape_model",
            "min_score": self._min_score_var.get(),
            "num_matches": self._max_matches_var.get(),
            "greediness": self._greediness_var.get(),
            "max_overlap": self._max_overlap_var.get(),
            "results": [
                {
                    "row": m.row,
                    "col": m.col,
                    "angle_deg": math.degrees(m.angle),
                    "scale": m.scale,
                    "score": m.score,
                }
                for m in self._match_results
            ],
        }

        self._add_pipeline_step(name, vis, op_meta=op_meta)
        self._set_status(f"已套用 {n} 個匹配結果")

    # ------------------------------------------------------------------ #
    #  Close                                                               #
    # ------------------------------------------------------------------ #

    def _close(self) -> None:
        """Release grab and destroy the dialog."""
        self.grab_release()
        self.destroy()
