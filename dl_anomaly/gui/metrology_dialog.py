"""
gui/metrology_dialog.py - Sub-pixel measurement tools dialog.

Provides a tabbed dialog for:
1. Sub-pixel edge detection (Canny-style)
2. Distance / edge-pair / angle measurements
3. Geometric fitting (line, circle, ellipse)

All operators delegate to ``dl_anomaly.core.metrology``.
"""
from __future__ import annotations

import logging
import math
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from dl_anomaly.core.metrology import (
    FitResult,
    MeasureRectangle,
    MeasurementResult,
    MeasurePair,
    SubPixelEdge,
    angle_ll,
    distance_pl,
    distance_pp,
    draw_edges,
    draw_fit_result,
    draw_measure_rect,
    draw_measurement,
    edges_sub_pix,
    fit_circle_contour_xld,
    fit_ellipse_contour_xld,
    fit_line_contour_xld,
    measure_pairs,
    measure_pos,
)

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
    _MONO_FAMILY = _MONO_FAMILY

# --------------------------------------------------------------------------- #
# Theme constants
# --------------------------------------------------------------------------- #
_BG = "#2b2b2b"
_BG_MEDIUM = "#3c3c3c"
_FG = "#cccccc"
_FG_WHITE = "#e0e0e0"
_ACCENT = "#0078d4"
_ACTIVE_BG = "#3a3a5c"


# ========================================================================== #
#  MetrologyDialog                                                            #
# ========================================================================== #


class MetrologyDialog(tk.Toplevel):
    """Sub-pixel measurement tools dialog.

    Parameters
    ----------
    master : tk.Widget
        Parent widget.
    get_current_image : callable
        ``() -> np.ndarray`` returning the current working image.
    add_pipeline_step : callable
        ``(name: str, image: np.ndarray) -> None`` to push a result
        into the parent pipeline.
    set_status : callable
        ``(msg: str) -> None`` to update the parent status bar.
    """

    def __init__(
        self,
        master: tk.Widget,
        get_current_image: Callable[[], np.ndarray],
        add_pipeline_step: Callable[[str, np.ndarray], None],
        set_status: Callable[[str], None],
    ) -> None:
        super().__init__(master)
        self.title("Metrology - Sub-Pixel Measurement Tools")
        self.geometry("700x600")
        self.resizable(True, True)
        self.configure(bg=_BG)

        self.transient(master)
        self.grab_set()

        self._get_current_image = get_current_image
        self._add_pipeline_step = add_pipeline_step
        self._set_status = set_status

        # ----- State --------------------------------------------------------
        self._edges: List[SubPixelEdge] = []
        self._fit_result: Optional[FitResult] = None
        self._measure_results: list = []

        # ----- Tkinter variables --------------------------------------------
        # Tab 1 - edge detection
        self._alpha_var = tk.DoubleVar(value=1.0)
        self._low_var = tk.IntVar(value=20)
        self._high_var = tk.IntVar(value=40)
        self._edge_count_var = tk.StringVar(value="--")
        self._irls_max_iter_var = tk.IntVar(value=50)
        self._irls_tol_var = tk.StringVar(value="1e-6")

        # Tab 2 - measurement
        self._meas_type_var = tk.StringVar(value="距離")
        self._dist_sub_var = tk.StringVar(value="點到點")
        # point-to-point entries
        self._pp_row1 = tk.StringVar(value="0")
        self._pp_col1 = tk.StringVar(value="0")
        self._pp_row2 = tk.StringVar(value="0")
        self._pp_col2 = tk.StringVar(value="0")
        # point-to-line entries
        self._pl_pr = tk.StringVar(value="0")
        self._pl_pc = tk.StringVar(value="0")
        self._pl_r1 = tk.StringVar(value="0")
        self._pl_c1 = tk.StringVar(value="0")
        self._pl_r2 = tk.StringVar(value="0")
        self._pl_c2 = tk.StringVar(value="0")
        # edge pairs
        self._ep_row = tk.StringVar(value="100")
        self._ep_col = tk.StringVar(value="100")
        self._ep_phi = tk.StringVar(value="0")
        self._ep_len1 = tk.StringVar(value="50")
        self._ep_len2 = tk.StringVar(value="10")
        self._ep_sigma = tk.StringVar(value="1.0")
        self._ep_thresh = tk.StringVar(value="30")
        # angle
        self._ang_r1l1 = tk.StringVar(value="0")
        self._ang_c1l1 = tk.StringVar(value="0")
        self._ang_r2l1 = tk.StringVar(value="0")
        self._ang_c2l1 = tk.StringVar(value="0")
        self._ang_r1l2 = tk.StringVar(value="0")
        self._ang_c1l2 = tk.StringVar(value="0")
        self._ang_r2l2 = tk.StringVar(value="0")
        self._ang_c2l2 = tk.StringVar(value="0")

        self._meas_result_var = tk.StringVar(value="--")

        # Tab 3 - fitting
        self._fit_source_var = tk.StringVar(value="邊緣結果")
        self._fit_type_var = tk.StringVar(value="直線")
        self._fit_algo_var = tk.StringVar(value="tukey")
        self._fit_clip_var = tk.StringVar(value="2.0")
        self._fit_result_var = tk.StringVar(value="--")

        # Build UI
        self._build_ui()

        self.protocol("WM_DELETE_WINDOW", self._close)

    # ------------------------------------------------------------------ #
    #  UI construction                                                     #
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        """Build the three-tab notebook layout."""
        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # Tab 1 - Edge Detection
        tab1 = tk.Frame(notebook, bg=_BG)
        notebook.add(tab1, text=" 邊緣偵測 ")
        self._build_edge_tab(tab1)

        # Tab 2 - Measurement Tools
        tab2 = tk.Frame(notebook, bg=_BG)
        notebook.add(tab2, text=" 量測工具 ")
        self._build_measure_tab(tab2)

        # Tab 3 - Geometric Fitting
        tab3 = tk.Frame(notebook, bg=_BG)
        notebook.add(tab3, text=" 幾何擬合 ")
        self._build_fit_tab(tab3)

        # Bottom buttons
        btn_frame = tk.Frame(self, bg=_BG)
        btn_frame.pack(fill=tk.X, padx=10, pady=(0, 8))

        tk.Button(
            btn_frame,
            text="關閉",
            bg=_BG_MEDIUM,
            fg=_FG_WHITE,
            activebackground="#555555",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=16,
            pady=4,
            font=("", 10),
            command=self._close,
        ).pack(side=tk.RIGHT)

    # ================================================================== #
    #  Tab 1 : Edge Detection                                             #
    # ================================================================== #

    def _build_edge_tab(self, parent: tk.Frame) -> None:
        """Build edge detection parameter controls."""
        # Parameters
        param_frame = tk.LabelFrame(
            parent, text=" 參數 ", bg=_BG, fg=_FG,
            font=("", 10, "bold"), padx=8, pady=6,
        )
        param_frame.pack(fill=tk.X, padx=8, pady=(8, 4))

        # Row 1 - Alpha / Sigma slider
        row1 = tk.Frame(param_frame, bg=_BG)
        row1.pack(fill=tk.X, pady=2)

        tk.Label(
            row1, text="平滑因子 (Alpha/Sigma):",
            bg=_BG, fg=_FG, font=("", 9),
        ).pack(side=tk.LEFT, padx=(0, 4))

        self._alpha_scale = tk.Scale(
            row1,
            from_=0.5,
            to=5.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            variable=self._alpha_var,
            bg=_BG,
            fg=_FG,
            troughcolor=_BG_MEDIUM,
            highlightthickness=0,
            sliderrelief=tk.FLAT,
            length=200,
        )
        self._alpha_scale.pack(side=tk.LEFT, padx=(0, 8))

        # Row 2 - Low / High thresholds
        row2 = tk.Frame(param_frame, bg=_BG)
        row2.pack(fill=tk.X, pady=2)

        tk.Label(
            row2, text="低閾值:", bg=_BG, fg=_FG, font=("", 9),
        ).pack(side=tk.LEFT, padx=(0, 2))
        tk.Spinbox(
            row2,
            textvariable=self._low_var,
            from_=1, to=255, increment=1, width=5,
            bg=_BG_MEDIUM, fg=_FG_WHITE, buttonbackground=_BG_MEDIUM,
            insertbackground=_FG_WHITE, relief=tk.FLAT,
        ).pack(side=tk.LEFT, padx=(0, 16))

        tk.Label(
            row2, text="高閾值:", bg=_BG, fg=_FG, font=("", 9),
        ).pack(side=tk.LEFT, padx=(0, 2))
        tk.Spinbox(
            row2,
            textvariable=self._high_var,
            from_=1, to=255, increment=1, width=5,
            bg=_BG_MEDIUM, fg=_FG_WHITE, buttonbackground=_BG_MEDIUM,
            insertbackground=_FG_WHITE, relief=tk.FLAT,
        ).pack(side=tk.LEFT)

        # Row 3 - IRLS parameters
        row3 = tk.Frame(param_frame, bg=_BG)
        row3.pack(fill=tk.X, pady=2)

        tk.Label(
            row3, text="IRLS 最大迭代:", bg=_BG, fg=_FG, font=("", 9),
        ).pack(side=tk.LEFT, padx=(0, 2))
        tk.Spinbox(
            row3,
            textvariable=self._irls_max_iter_var,
            from_=1, to=500, increment=10, width=5,
            bg=_BG_MEDIUM, fg=_FG_WHITE, buttonbackground=_BG_MEDIUM,
            insertbackground=_FG_WHITE, relief=tk.FLAT,
        ).pack(side=tk.LEFT, padx=(0, 16))

        tk.Label(
            row3, text="IRLS 容差:", bg=_BG, fg=_FG, font=("", 9),
        ).pack(side=tk.LEFT, padx=(0, 2))
        tk.Entry(
            row3,
            textvariable=self._irls_tol_var,
            width=8,
            bg=_BG_MEDIUM, fg=_FG_WHITE,
            insertbackground=_FG_WHITE, relief=tk.FLAT,
        ).pack(side=tk.LEFT)

        # Detect button
        btn_row = tk.Frame(param_frame, bg=_BG)
        btn_row.pack(fill=tk.X, pady=(8, 2))

        tk.Button(
            btn_row,
            text="偵測邊緣",
            bg=_ACCENT,
            fg="#ffffff",
            activebackground="#005a9e",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=16,
            pady=4,
            font=("", 10, "bold"),
            command=self._detect_edges,
        ).pack(side=tk.LEFT)

        # Results
        result_frame = tk.LabelFrame(
            parent, text=" 結果 ", bg=_BG, fg=_FG,
            font=("", 10, "bold"), padx=8, pady=6,
        )
        result_frame.pack(fill=tk.X, padx=8, pady=4)

        res_row = tk.Frame(result_frame, bg=_BG)
        res_row.pack(fill=tk.X, pady=2)

        tk.Label(
            res_row, text="偵測到的邊緣數:", bg=_BG, fg=_FG, font=("", 9),
        ).pack(side=tk.LEFT, padx=(0, 4))
        tk.Label(
            res_row, textvariable=self._edge_count_var,
            bg=_BG, fg="#88cc88", font=(_MONO_FAMILY, 10),
        ).pack(side=tk.LEFT)

        # Draw button
        draw_row = tk.Frame(result_frame, bg=_BG)
        draw_row.pack(fill=tk.X, pady=(6, 2))

        tk.Button(
            draw_row,
            text="繪製結果",
            bg=_BG_MEDIUM,
            fg=_FG_WHITE,
            activebackground=_ACTIVE_BG,
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=16,
            pady=4,
            font=("", 9),
            command=self._draw_edges,
        ).pack(side=tk.LEFT)

    # ================================================================== #
    #  Tab 2 : Measurement Tools                                          #
    # ================================================================== #

    def _build_measure_tab(self, parent: tk.Frame) -> None:
        """Build measurement tool controls."""
        # Type selector
        type_frame = tk.Frame(parent, bg=_BG)
        type_frame.pack(fill=tk.X, padx=8, pady=(8, 4))

        tk.Label(
            type_frame, text="量測類型:",
            bg=_BG, fg=_FG, font=("", 9),
        ).pack(side=tk.LEFT, padx=(0, 4))

        self._meas_type_combo = ttk.Combobox(
            type_frame,
            textvariable=self._meas_type_var,
            values=["距離", "邊緣對", "角度"],
            state="readonly",
            width=12,
        )
        self._meas_type_combo.pack(side=tk.LEFT)
        self._meas_type_combo.bind(
            "<<ComboboxSelected>>", self._on_meas_type_changed,
        )

        # Container for swappable sub-panels
        self._meas_container = tk.Frame(parent, bg=_BG)
        self._meas_container.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        # Build all sub-panels (only one visible at a time)
        self._distance_panel = self._build_distance_panel(self._meas_container)
        self._edge_pair_panel = self._build_edge_pair_panel(self._meas_container)
        self._angle_panel = self._build_angle_panel(self._meas_container)

        # Show default
        self._show_meas_panel("距離")

    def _show_meas_panel(self, mtype: str) -> None:
        """Show the sub-panel matching *mtype*, hide others."""
        for w in (self._distance_panel, self._edge_pair_panel, self._angle_panel):
            w.pack_forget()

        panel_map: Dict[str, tk.Widget] = {
            "距離": self._distance_panel,
            "邊緣對": self._edge_pair_panel,
            "角度": self._angle_panel,
        }
        target = panel_map.get(mtype)
        if target is not None:
            target.pack(fill=tk.BOTH, expand=True)

    def _on_meas_type_changed(self, _event: object = None) -> None:
        self._show_meas_panel(self._meas_type_var.get())

    # ---- Distance sub-panel ------------------------------------------------

    def _build_distance_panel(self, parent: tk.Frame) -> tk.Frame:
        panel = tk.Frame(parent, bg=_BG)

        # Sub-option selector
        sub_row = tk.Frame(panel, bg=_BG)
        sub_row.pack(fill=tk.X, pady=(4, 4))

        tk.Label(
            sub_row, text="模式:", bg=_BG, fg=_FG, font=("", 9),
        ).pack(side=tk.LEFT, padx=(0, 4))

        self._dist_sub_combo = ttk.Combobox(
            sub_row,
            textvariable=self._dist_sub_var,
            values=["點到點", "點到線"],
            state="readonly",
            width=10,
        )
        self._dist_sub_combo.pack(side=tk.LEFT)
        self._dist_sub_combo.bind(
            "<<ComboboxSelected>>", self._on_dist_sub_changed,
        )

        # Point-to-point frame
        self._pp_frame = tk.LabelFrame(
            panel, text=" 點到點 ", bg=_BG, fg=_FG,
            font=("", 9, "bold"), padx=6, pady=4,
        )
        pp_entries = [
            ("Row1:", self._pp_row1), ("Col1:", self._pp_col1),
            ("Row2:", self._pp_row2), ("Col2:", self._pp_col2),
        ]
        r = tk.Frame(self._pp_frame, bg=_BG)
        r.pack(fill=tk.X, pady=2)
        for label_text, var in pp_entries:
            tk.Label(
                r, text=label_text, bg=_BG, fg=_FG, font=("", 9),
            ).pack(side=tk.LEFT, padx=(0, 2))
            tk.Entry(
                r, textvariable=var, width=7,
                bg=_BG_MEDIUM, fg=_FG_WHITE,
                insertbackground=_FG_WHITE, relief=tk.FLAT,
            ).pack(side=tk.LEFT, padx=(0, 8))

        # Point-to-line frame
        self._pl_frame = tk.LabelFrame(
            panel, text=" 點到線 ", bg=_BG, fg=_FG,
            font=("", 9, "bold"), padx=6, pady=4,
        )
        pl_row1 = tk.Frame(self._pl_frame, bg=_BG)
        pl_row1.pack(fill=tk.X, pady=2)
        for label_text, var in [
            ("Point Row:", self._pl_pr), ("Point Col:", self._pl_pc),
        ]:
            tk.Label(
                pl_row1, text=label_text, bg=_BG, fg=_FG, font=("", 9),
            ).pack(side=tk.LEFT, padx=(0, 2))
            tk.Entry(
                pl_row1, textvariable=var, width=7,
                bg=_BG_MEDIUM, fg=_FG_WHITE,
                insertbackground=_FG_WHITE, relief=tk.FLAT,
            ).pack(side=tk.LEFT, padx=(0, 8))

        pl_row2 = tk.Frame(self._pl_frame, bg=_BG)
        pl_row2.pack(fill=tk.X, pady=2)
        for label_text, var in [
            ("Line R1:", self._pl_r1), ("Line C1:", self._pl_c1),
            ("Line R2:", self._pl_r2), ("Line C2:", self._pl_c2),
        ]:
            tk.Label(
                pl_row2, text=label_text, bg=_BG, fg=_FG, font=("", 9),
            ).pack(side=tk.LEFT, padx=(0, 2))
            tk.Entry(
                pl_row2, textvariable=var, width=7,
                bg=_BG_MEDIUM, fg=_FG_WHITE,
                insertbackground=_FG_WHITE, relief=tk.FLAT,
            ).pack(side=tk.LEFT, padx=(0, 6))

        # Calculate button + result
        calc_row = tk.Frame(panel, bg=_BG)
        calc_row.pack(fill=tk.X, pady=(8, 2))

        tk.Button(
            calc_row,
            text="計算",
            bg=_ACCENT,
            fg="#ffffff",
            activebackground="#005a9e",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=16,
            pady=4,
            font=("", 9, "bold"),
            command=self._calc_distance,
        ).pack(side=tk.LEFT, padx=(0, 12))

        tk.Label(
            calc_row, text="結果:", bg=_BG, fg=_FG, font=("", 9),
        ).pack(side=tk.LEFT, padx=(0, 4))
        tk.Label(
            calc_row, textvariable=self._meas_result_var,
            bg=_BG, fg="#88cc88", font=(_MONO_FAMILY, 10),
        ).pack(side=tk.LEFT)

        # Show default sub-panel
        self._pp_frame.pack(fill=tk.X, pady=4)

        return panel

    def _on_dist_sub_changed(self, _event: object = None) -> None:
        sub = self._dist_sub_var.get()
        self._pp_frame.pack_forget()
        self._pl_frame.pack_forget()
        if sub == "點到點":
            self._pp_frame.pack(fill=tk.X, pady=4)
        else:
            self._pl_frame.pack(fill=tk.X, pady=4)

    # ---- Edge Pair sub-panel -----------------------------------------------

    def _build_edge_pair_panel(self, parent: tk.Frame) -> tk.Frame:
        panel = tk.Frame(parent, bg=_BG)

        param_lf = tk.LabelFrame(
            panel, text=" 量測矩形參數 ", bg=_BG, fg=_FG,
            font=("", 9, "bold"), padx=6, pady=4,
        )
        param_lf.pack(fill=tk.X, pady=(4, 4))

        # Row / Col / Phi
        r1 = tk.Frame(param_lf, bg=_BG)
        r1.pack(fill=tk.X, pady=2)
        for label_text, var in [
            ("Row:", self._ep_row), ("Col:", self._ep_col),
            ("Phi(\u00b0):", self._ep_phi),
        ]:
            tk.Label(
                r1, text=label_text, bg=_BG, fg=_FG, font=("", 9),
            ).pack(side=tk.LEFT, padx=(0, 2))
            tk.Entry(
                r1, textvariable=var, width=7,
                bg=_BG_MEDIUM, fg=_FG_WHITE,
                insertbackground=_FG_WHITE, relief=tk.FLAT,
            ).pack(side=tk.LEFT, padx=(0, 8))

        # Length1 / Length2
        r2 = tk.Frame(param_lf, bg=_BG)
        r2.pack(fill=tk.X, pady=2)
        for label_text, var in [
            ("Length1:", self._ep_len1), ("Length2:", self._ep_len2),
        ]:
            tk.Label(
                r2, text=label_text, bg=_BG, fg=_FG, font=("", 9),
            ).pack(side=tk.LEFT, padx=(0, 2))
            tk.Entry(
                r2, textvariable=var, width=7,
                bg=_BG_MEDIUM, fg=_FG_WHITE,
                insertbackground=_FG_WHITE, relief=tk.FLAT,
            ).pack(side=tk.LEFT, padx=(0, 8))

        # Sigma / Threshold
        r3 = tk.Frame(param_lf, bg=_BG)
        r3.pack(fill=tk.X, pady=2)
        for label_text, var in [
            ("Sigma:", self._ep_sigma), ("Threshold:", self._ep_thresh),
        ]:
            tk.Label(
                r3, text=label_text, bg=_BG, fg=_FG, font=("", 9),
            ).pack(side=tk.LEFT, padx=(0, 2))
            tk.Entry(
                r3, textvariable=var, width=7,
                bg=_BG_MEDIUM, fg=_FG_WHITE,
                insertbackground=_FG_WHITE, relief=tk.FLAT,
            ).pack(side=tk.LEFT, padx=(0, 8))

        # Measure button
        btn_row = tk.Frame(panel, bg=_BG)
        btn_row.pack(fill=tk.X, pady=(4, 4))

        tk.Button(
            btn_row,
            text="量測",
            bg=_ACCENT,
            fg="#ffffff",
            activebackground="#005a9e",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=16,
            pady=4,
            font=("", 9, "bold"),
            command=self._measure_edge_pairs,
        ).pack(side=tk.LEFT, padx=(0, 8))

        tk.Button(
            btn_row,
            text="繪製",
            bg=_BG_MEDIUM,
            fg=_FG_WHITE,
            activebackground=_ACTIVE_BG,
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=16,
            pady=4,
            font=("", 9),
            command=self._draw_edge_pairs,
        ).pack(side=tk.LEFT)

        # Results treeview
        tree_frame = tk.LabelFrame(
            panel, text=" 結果 ", bg=_BG, fg=_FG,
            font=("", 9, "bold"), padx=4, pady=2,
        )
        tree_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 4))

        columns = ("index", "edge1", "edge2", "distance")
        self._ep_tree = ttk.Treeview(
            tree_frame, columns=columns, show="headings", height=6,
        )
        self._ep_tree.heading("index", text="#")
        self._ep_tree.heading("edge1", text="邊緣1")
        self._ep_tree.heading("edge2", text="邊緣2")
        self._ep_tree.heading("distance", text="距離")

        self._ep_tree.column("index", width=40, anchor=tk.CENTER)
        self._ep_tree.column("edge1", width=160, anchor=tk.CENTER)
        self._ep_tree.column("edge2", width=160, anchor=tk.CENTER)
        self._ep_tree.column("distance", width=100, anchor=tk.CENTER)

        tree_scroll = ttk.Scrollbar(
            tree_frame, orient=tk.VERTICAL, command=self._ep_tree.yview,
        )
        self._ep_tree.configure(yscrollcommand=tree_scroll.set)
        self._ep_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        return panel

    # ---- Angle sub-panel ---------------------------------------------------

    def _build_angle_panel(self, parent: tk.Frame) -> tk.Frame:
        panel = tk.Frame(parent, bg=_BG)

        line1_lf = tk.LabelFrame(
            panel, text=" 直線 1 ", bg=_BG, fg=_FG,
            font=("", 9, "bold"), padx=6, pady=4,
        )
        line1_lf.pack(fill=tk.X, pady=(4, 4))

        r1 = tk.Frame(line1_lf, bg=_BG)
        r1.pack(fill=tk.X, pady=2)
        for label_text, var in [
            ("Row1:", self._ang_r1l1), ("Col1:", self._ang_c1l1),
            ("Row2:", self._ang_r2l1), ("Col2:", self._ang_c2l1),
        ]:
            tk.Label(
                r1, text=label_text, bg=_BG, fg=_FG, font=("", 9),
            ).pack(side=tk.LEFT, padx=(0, 2))
            tk.Entry(
                r1, textvariable=var, width=7,
                bg=_BG_MEDIUM, fg=_FG_WHITE,
                insertbackground=_FG_WHITE, relief=tk.FLAT,
            ).pack(side=tk.LEFT, padx=(0, 6))

        line2_lf = tk.LabelFrame(
            panel, text=" 直線 2 ", bg=_BG, fg=_FG,
            font=("", 9, "bold"), padx=6, pady=4,
        )
        line2_lf.pack(fill=tk.X, pady=(0, 4))

        r2 = tk.Frame(line2_lf, bg=_BG)
        r2.pack(fill=tk.X, pady=2)
        for label_text, var in [
            ("Row1:", self._ang_r1l2), ("Col1:", self._ang_c1l2),
            ("Row2:", self._ang_r2l2), ("Col2:", self._ang_c2l2),
        ]:
            tk.Label(
                r2, text=label_text, bg=_BG, fg=_FG, font=("", 9),
            ).pack(side=tk.LEFT, padx=(0, 2))
            tk.Entry(
                r2, textvariable=var, width=7,
                bg=_BG_MEDIUM, fg=_FG_WHITE,
                insertbackground=_FG_WHITE, relief=tk.FLAT,
            ).pack(side=tk.LEFT, padx=(0, 6))

        # Calculate button + result
        calc_row = tk.Frame(panel, bg=_BG)
        calc_row.pack(fill=tk.X, pady=(8, 2))

        tk.Button(
            calc_row,
            text="計算",
            bg=_ACCENT,
            fg="#ffffff",
            activebackground="#005a9e",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=16,
            pady=4,
            font=("", 9, "bold"),
            command=self._calc_angle,
        ).pack(side=tk.LEFT, padx=(0, 12))

        self._angle_result_var = tk.StringVar(value="--")
        tk.Label(
            calc_row, text="結果:", bg=_BG, fg=_FG, font=("", 9),
        ).pack(side=tk.LEFT, padx=(0, 4))
        tk.Label(
            calc_row, textvariable=self._angle_result_var,
            bg=_BG, fg="#88cc88", font=(_MONO_FAMILY, 10),
        ).pack(side=tk.LEFT)

        return panel

    # ================================================================== #
    #  Tab 3 : Geometric Fitting                                          #
    # ================================================================== #

    def _build_fit_tab(self, parent: tk.Frame) -> None:
        """Build geometric fitting controls."""
        param_frame = tk.LabelFrame(
            parent, text=" 擬合參數 ", bg=_BG, fg=_FG,
            font=("", 10, "bold"), padx=8, pady=6,
        )
        param_frame.pack(fill=tk.X, padx=8, pady=(8, 4))

        # Source
        r1 = tk.Frame(param_frame, bg=_BG)
        r1.pack(fill=tk.X, pady=2)

        tk.Label(
            r1, text="點來源:", bg=_BG, fg=_FG, font=("", 9),
        ).pack(side=tk.LEFT, padx=(0, 4))
        self._fit_source_combo = ttk.Combobox(
            r1,
            textvariable=self._fit_source_var,
            values=["邊緣結果", "手動輸入"],
            state="readonly",
            width=12,
        )
        self._fit_source_combo.pack(side=tk.LEFT, padx=(0, 16))
        self._fit_source_combo.bind(
            "<<ComboboxSelected>>", self._on_fit_source_changed,
        )

        tk.Label(
            r1, text="擬合類型:", bg=_BG, fg=_FG, font=("", 9),
        ).pack(side=tk.LEFT, padx=(0, 4))
        self._fit_type_combo = ttk.Combobox(
            r1,
            textvariable=self._fit_type_var,
            values=["直線", "圓", "橢圓"],
            state="readonly",
            width=8,
        )
        self._fit_type_combo.pack(side=tk.LEFT)
        self._fit_type_combo.bind(
            "<<ComboboxSelected>>", self._on_fit_type_changed,
        )

        # Algorithm + Clipping
        r2 = tk.Frame(param_frame, bg=_BG)
        r2.pack(fill=tk.X, pady=2)

        tk.Label(
            r2, text="演算法:", bg=_BG, fg=_FG, font=("", 9),
        ).pack(side=tk.LEFT, padx=(0, 4))
        self._fit_algo_combo = ttk.Combobox(
            r2,
            textvariable=self._fit_algo_var,
            values=["regression", "tukey", "huber"],
            state="readonly",
            width=12,
        )
        self._fit_algo_combo.pack(side=tk.LEFT, padx=(0, 16))

        tk.Label(
            r2, text="Clipping Factor:", bg=_BG, fg=_FG, font=("", 9),
        ).pack(side=tk.LEFT, padx=(0, 4))
        tk.Entry(
            r2, textvariable=self._fit_clip_var, width=6,
            bg=_BG_MEDIUM, fg=_FG_WHITE,
            insertbackground=_FG_WHITE, relief=tk.FLAT,
        ).pack(side=tk.LEFT)

        # Manual points input (visible when source == "手動輸入")
        self._manual_frame = tk.LabelFrame(
            param_frame, text=" 手動輸入點 (每行: row,col) ",
            bg=_BG, fg=_FG, font=("", 9), padx=4, pady=4,
        )

        self._manual_text = tk.Text(
            self._manual_frame,
            height=4,
            width=40,
            bg=_BG_MEDIUM,
            fg=_FG_WHITE,
            insertbackground=_FG_WHITE,
            relief=tk.FLAT,
            font=(_MONO_FAMILY, 9),
        )
        self._manual_text.pack(fill=tk.X)

        # Fit button
        fit_btn_row = tk.Frame(param_frame, bg=_BG)
        fit_btn_row.pack(fill=tk.X, pady=(8, 2))

        tk.Button(
            fit_btn_row,
            text="擬合",
            bg=_ACCENT,
            fg="#ffffff",
            activebackground="#005a9e",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=16,
            pady=4,
            font=("", 10, "bold"),
            command=self._perform_fit,
        ).pack(side=tk.LEFT)

        # Results display
        result_frame = tk.LabelFrame(
            parent, text=" 擬合結果 ", bg=_BG, fg=_FG,
            font=("", 10, "bold"), padx=8, pady=6,
        )
        result_frame.pack(fill=tk.X, padx=8, pady=4)

        tk.Label(
            result_frame,
            textvariable=self._fit_result_var,
            bg=_BG,
            fg="#88cc88",
            font=(_MONO_FAMILY, 9),
            anchor=tk.W,
            justify=tk.LEFT,
        ).pack(fill=tk.X)

        # Draw button
        draw_row = tk.Frame(parent, bg=_BG)
        draw_row.pack(fill=tk.X, padx=8, pady=4)

        tk.Button(
            draw_row,
            text="繪製結果",
            bg=_BG_MEDIUM,
            fg=_FG_WHITE,
            activebackground=_ACTIVE_BG,
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=16,
            pady=4,
            font=("", 9),
            command=self._draw_fit_result,
        ).pack(side=tk.LEFT)

    def _on_fit_source_changed(self, _event: object = None) -> None:
        """Show or hide the manual points input area."""
        if self._fit_source_var.get() == "手動輸入":
            self._manual_frame.pack(fill=tk.X, pady=(4, 0))
        else:
            self._manual_frame.pack_forget()

    def _on_fit_type_changed(self, _event: object = None) -> None:
        """Update available algorithms when the fit type changes."""
        fit_type = self._fit_type_var.get()
        algo_map: Dict[str, list] = {
            "直線": ["regression", "tukey", "huber"],
            "圓": ["algebraic", "geometric"],
            "橢圓": ["fitzgibbon"],
        }
        algos = algo_map.get(fit_type, ["regression"])
        self._fit_algo_combo.configure(values=algos)
        self._fit_algo_var.set(algos[0])

    # ================================================================== #
    #  Edge detection logic                                                #
    # ================================================================== #

    def _detect_edges(self) -> None:
        """Run sub-pixel edge detection on the current image."""
        try:
            image = self._get_current_image()
            if image is None:
                messagebox.showwarning(
                    "無影像", "請先載入影像。", parent=self,
                )
                return

            alpha = self._alpha_var.get()
            low = self._low_var.get()
            high = self._high_var.get()
            irls_max_iter = self._irls_max_iter_var.get()
            irls_tol = self._safe_float(self._irls_tol_var.get(), default=1e-6)

            self._edges = edges_sub_pix(
                image, alpha=alpha, low=low, high=high,
                max_iter=irls_max_iter, tolerance=irls_tol,
            )
            count = len(self._edges)
            self._edge_count_var.set(str(count))
            self._set_status(f"偵測到 {count} 個子像素邊緣")
            logger.info(
                "Detected %d sub-pixel edges (alpha=%.1f, low=%d, high=%d, "
                "irls_max_iter=%d, irls_tol=%.1e)",
                count, alpha, low, high, irls_max_iter, irls_tol,
            )

        except Exception as exc:
            logger.exception("Edge detection failed")
            messagebox.showerror(
                "邊緣偵測錯誤",
                f"執行邊緣偵測時發生錯誤:\n{exc}",
                parent=self,
            )

    def _draw_edges(self) -> None:
        """Draw detected edges on the current image and add to pipeline."""
        if not self._edges:
            messagebox.showwarning(
                "無邊緣", "請先執行邊緣偵測。", parent=self,
            )
            return

        try:
            image = self._get_current_image()
            result = draw_edges(image, self._edges)
            self._add_pipeline_step("SubPixelEdges", result)
            self._set_status(f"已繪製 {len(self._edges)} 個邊緣至影像")
        except Exception as exc:
            logger.exception("Draw edges failed")
            messagebox.showerror(
                "繪製錯誤", f"繪製邊緣時發生錯誤:\n{exc}", parent=self,
            )

    # ================================================================== #
    #  Measurement logic                                                   #
    # ================================================================== #

    def _calc_distance(self) -> None:
        """Calculate point-to-point or point-to-line distance."""
        try:
            sub = self._dist_sub_var.get()
            if sub == "點到點":
                r1 = self._safe_float(self._pp_row1.get())
                c1 = self._safe_float(self._pp_col1.get())
                r2 = self._safe_float(self._pp_row2.get())
                c2 = self._safe_float(self._pp_col2.get())
                dist = distance_pp(r1, c1, r2, c2)
                self._meas_result_var.set(f"{dist:.4f} px")

                result = MeasurementResult(
                    value=dist, unit="px", type="distance",
                    point1=(r1, c1), point2=(r2, c2),
                )
                self._measure_results = [result]
                self._set_status(f"點到點距離: {dist:.4f} px")

            else:  # 點到線
                pr = self._safe_float(self._pl_pr.get())
                pc = self._safe_float(self._pl_pc.get())
                r1 = self._safe_float(self._pl_r1.get())
                c1 = self._safe_float(self._pl_c1.get())
                r2 = self._safe_float(self._pl_r2.get())
                c2 = self._safe_float(self._pl_c2.get())
                dist = distance_pl(pr, pc, r1, c1, r2, c2)
                self._meas_result_var.set(f"{dist:.4f} px")

                result = MeasurementResult(
                    value=dist, unit="px", type="distance",
                    point1=(pr, pc), point2=(r1, c1),
                )
                self._measure_results = [result]
                self._set_status(f"點到線距離: {dist:.4f} px")

        except Exception as exc:
            logger.exception("Distance calculation failed")
            messagebox.showerror(
                "計算錯誤", f"計算距離時發生錯誤:\n{exc}", parent=self,
            )

    def _measure_edge_pairs(self) -> None:
        """Measure edge pairs along a measurement rectangle."""
        try:
            image = self._get_current_image()
            if image is None:
                messagebox.showwarning(
                    "無影像", "請先載入影像。", parent=self,
                )
                return

            row = self._safe_float(self._ep_row.get())
            col = self._safe_float(self._ep_col.get())
            phi_deg = self._safe_float(self._ep_phi.get())
            phi_rad = math.radians(phi_deg)
            length1 = self._safe_float(self._ep_len1.get())
            length2 = self._safe_float(self._ep_len2.get())
            sigma = self._safe_float(self._ep_sigma.get())
            threshold = self._safe_float(self._ep_thresh.get())

            rect = MeasureRectangle(
                row=row, col=col, phi=phi_rad,
                length1=length1, length2=length2,
            )

            pairs = measure_pairs(
                image, rect, sigma=sigma, threshold=threshold,
            )
            self._measure_results = pairs

            # Populate treeview
            for item in self._ep_tree.get_children():
                self._ep_tree.delete(item)

            for i, pair in enumerate(pairs, start=1):
                e1_str = f"({pair.edge1.row:.2f}, {pair.edge1.col:.2f})"
                e2_str = f"({pair.edge2.row:.2f}, {pair.edge2.col:.2f})"
                self._ep_tree.insert("", tk.END, values=(
                    i, e1_str, e2_str, f"{pair.distance:.4f}",
                ))

            self._set_status(f"偵測到 {len(pairs)} 個邊緣對")
            logger.info("Measured %d edge pairs", len(pairs))

        except Exception as exc:
            logger.exception("Edge pair measurement failed")
            messagebox.showerror(
                "量測錯誤", f"量測邊緣對時發生錯誤:\n{exc}", parent=self,
            )

    def _draw_edge_pairs(self) -> None:
        """Draw edge pair measurement results on the image."""
        if not self._measure_results:
            messagebox.showwarning(
                "無結果", "請先執行量測。", parent=self,
            )
            return

        try:
            image = self._get_current_image()

            # Draw the measurement rectangle
            row = self._safe_float(self._ep_row.get())
            col = self._safe_float(self._ep_col.get())
            phi_deg = self._safe_float(self._ep_phi.get())
            phi_rad = math.radians(phi_deg)
            length1 = self._safe_float(self._ep_len1.get())
            length2 = self._safe_float(self._ep_len2.get())

            rect = MeasureRectangle(
                row=row, col=col, phi=phi_rad,
                length1=length1, length2=length2,
            )
            result_img = draw_measure_rect(image, rect)

            # Draw each pair as a measurement annotation
            for pair in self._measure_results:
                if isinstance(pair, MeasurePair):
                    meas = MeasurementResult(
                        value=pair.distance,
                        unit="px",
                        type="distance",
                        point1=(pair.edge1.row, pair.edge1.col),
                        point2=(pair.edge2.row, pair.edge2.col),
                    )
                    result_img = draw_measurement(result_img, meas)

            self._add_pipeline_step("MeasureEdgePairs", result_img)
            self._set_status("已繪製邊緣對量測結果至影像")

        except Exception as exc:
            logger.exception("Draw edge pairs failed")
            messagebox.showerror(
                "繪製錯誤", f"繪製邊緣對時發生錯誤:\n{exc}", parent=self,
            )

    def _calc_angle(self) -> None:
        """Calculate the angle between two lines."""
        try:
            r1l1 = self._safe_float(self._ang_r1l1.get())
            c1l1 = self._safe_float(self._ang_c1l1.get())
            r2l1 = self._safe_float(self._ang_r2l1.get())
            c2l1 = self._safe_float(self._ang_c2l1.get())
            r1l2 = self._safe_float(self._ang_r1l2.get())
            c1l2 = self._safe_float(self._ang_c1l2.get())
            r2l2 = self._safe_float(self._ang_r2l2.get())
            c2l2 = self._safe_float(self._ang_c2l2.get())

            angle_rad = angle_ll(r1l1, c1l1, r2l1, c2l1,
                                 r1l2, c1l2, r2l2, c2l2)
            angle_deg = math.degrees(angle_rad)

            self._angle_result_var.set(
                f"{angle_rad:.6f} rad  ({angle_deg:.4f}\u00b0)"
            )
            self._set_status(f"兩線夾角: {angle_deg:.4f}\u00b0")

        except Exception as exc:
            logger.exception("Angle calculation failed")
            messagebox.showerror(
                "計算錯誤", f"計算角度時發生錯誤:\n{exc}", parent=self,
            )

    # ================================================================== #
    #  Geometric fitting logic                                             #
    # ================================================================== #

    def _get_fit_points(self) -> list:
        """Return the list of (row, col) points for fitting."""
        source = self._fit_source_var.get()

        if source == "邊緣結果":
            if not self._edges:
                raise ValueError("尚未偵測到邊緣，請先執行邊緣偵測。")
            return [(e.row, e.col) for e in self._edges]

        # Manual input
        text = self._manual_text.get("1.0", tk.END).strip()
        if not text:
            raise ValueError("請輸入手動點座標 (每行: row,col)。")

        points = []
        for line_num, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.replace(";", ",").split(",")
            if len(parts) < 2:
                raise ValueError(
                    f"第 {line_num} 行格式錯誤: 需要 row,col 格式"
                )
            points.append((float(parts[0].strip()), float(parts[1].strip())))
        return points

    def _perform_fit(self) -> None:
        """Execute geometric fitting on the selected points."""
        try:
            points = self._get_fit_points()
            fit_type = self._fit_type_var.get()
            algorithm = self._fit_algo_var.get()
            clip = self._safe_float(self._fit_clip_var.get())

            irls_max_iter = self._irls_max_iter_var.get()
            irls_tol = self._safe_float(self._irls_tol_var.get(), default=1e-6)

            if fit_type == "直線":
                result = fit_line_contour_xld(
                    points, algorithm=algorithm,
                    clipping_factor=clip,
                    max_iter=irls_max_iter, tolerance=irls_tol,
                )
            elif fit_type == "圓":
                result = fit_circle_contour_xld(
                    points, algorithm=algorithm,
                    clipping_factor=clip,
                    max_iter=irls_max_iter, tolerance=irls_tol,
                )
            elif fit_type == "橢圓":
                result = fit_ellipse_contour_xld(
                    points, algorithm=algorithm,
                    clipping_factor=clip,
                    max_iter=irls_max_iter, tolerance=irls_tol,
                )
            else:
                raise ValueError(f"未知擬合類型: {fit_type}")

            self._fit_result = result

            # Format result display
            lines = [f"類型: {result.type}"]
            for k, v in result.params.items():
                lines.append(f"  {k}: {v:.4f}")
            lines.append(f"RMS 誤差: {result.error:.6f}")
            lines.append(f"使用點數: {result.points_used}")
            self._fit_result_var.set("\n".join(lines))

            self._set_status(
                f"擬合完成: {result.type} (RMS={result.error:.4f}, "
                f"使用 {result.points_used} 點)"
            )
            logger.info(
                "Fit %s: error=%.6f, points_used=%d",
                result.type, result.error, result.points_used,
            )

        except Exception as exc:
            logger.exception("Geometric fitting failed")
            messagebox.showerror(
                "擬合錯誤", f"執行擬合時發生錯誤:\n{exc}", parent=self,
            )

    def _draw_fit_result(self) -> None:
        """Draw the fitted geometric primitive on the image."""
        if self._fit_result is None:
            messagebox.showwarning(
                "無結果", "請先執行擬合。", parent=self,
            )
            return

        try:
            image = self._get_current_image()
            result_img = draw_fit_result(image, self._fit_result)
            self._add_pipeline_step(
                f"Fit_{self._fit_result.type}", result_img,
            )
            self._set_status(f"已繪製 {self._fit_result.type} 擬合結果至影像")

        except Exception as exc:
            logger.exception("Draw fit result failed")
            messagebox.showerror(
                "繪製錯誤", f"繪製擬合結果時發生錯誤:\n{exc}", parent=self,
            )

    # ================================================================== #
    #  Close                                                               #
    # ================================================================== #

    def _close(self) -> None:
        """Close the dialog."""
        self.grab_release()
        self.destroy()

    # ================================================================== #
    #  Helpers                                                             #
    # ================================================================== #

    @staticmethod
    def _safe_float(value: str, default: float = 0.0) -> float:
        """Parse *value* as a float, returning *default* on failure."""
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
