"""
gui/region_filter_dialog.py - Interactive Vision-style select_shape dialog.

Filters Region objects by geometric / photometric properties (area,
circularity, rectangularity, aspect_ratio, mean_value, etc.) with a live
preview.  Multiple filter conditions can be combined with AND logic.

Green overlay = passing regions, Red overlay = failing regions.
"""
from __future__ import annotations

import copy
import logging
import tkinter as tk
from tkinter import ttk
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk

from core.region import Region, RegionProperties
from core.region_ops import region_to_display_image, select_shape

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

# ---------------------------------------------------------------------------
#  Theme constants
# ---------------------------------------------------------------------------
_BG_DARK = "#2b2b2b"
_BG_MEDIUM = "#3c3c3c"
_FG = "#cccccc"
_FG_WHITE = "#e0e0e0"
_ACCENT = "#0078d4"
_CANVAS_BG = "#1e1e1e"

# ---------------------------------------------------------------------------
#  Available features: (attribute_name, display_label, decimal_places)
# ---------------------------------------------------------------------------
FEATURES: List[Tuple[str, str, int]] = [
    ("area", "面積", 0),
    ("width", "寬度", 0),
    ("height", "高度", 0),
    ("circularity", "圓度", 4),
    ("rectangularity", "矩形度", 4),
    ("aspect_ratio", "長寬比", 4),
    ("compactness", "緊湊度", 4),
    ("convexity", "凸度", 4),
    ("perimeter", "周長", 2),
    ("orientation", "方向角", 2),
    ("mean_value", "平均灰度", 2),
]

# Lookup helpers
_FEATURE_ATTRS: List[str] = [f[0] for f in FEATURES]
_FEATURE_LABELS: Dict[str, str] = {f[0]: f[1] for f in FEATURES}
_FEATURE_DECIMALS: Dict[str, int] = {f[0]: f[2] for f in FEATURES}

# Slider resolution: integer features use 1, float features use 10 000 steps
_SLIDER_STEPS = 10_000


# ====================================================================== #
#  FilterCondition  (one criterion stored in the conditions listbox)       #
# ====================================================================== #

class _FilterCondition:
    """Value object that represents a single filter condition."""

    __slots__ = ("feature", "min_val", "max_val")

    def __init__(self, feature: str, min_val: float, max_val: float) -> None:
        self.feature = feature
        self.min_val = min_val
        self.max_val = max_val

    def display_text(self) -> str:
        dec = _FEATURE_DECIMALS.get(self.feature, 2)
        label = _FEATURE_LABELS.get(self.feature, self.feature)
        if dec == 0:
            return f"{label} ({self.feature}): {int(self.min_val)} ~ {int(self.max_val)}"
        fmt = f"{{:.{dec}f}}"
        return (
            f"{label} ({self.feature}): "
            f"{fmt.format(self.min_val)} ~ {fmt.format(self.max_val)}"
        )

    def matches(self, props: RegionProperties) -> bool:
        val = getattr(props, self.feature, None)
        if val is None:
            return False
        return self.min_val <= float(val) <= self.max_val


# ====================================================================== #
#  RegionFilterDialog                                                      #
# ====================================================================== #

class RegionFilterDialog(tk.Toplevel):
    """Modal dialog for filtering a Region by geometric properties.

    Parameters
    ----------
    master : tk.Widget
        Parent widget.
    region : Region
        The region set to filter.
    source_image : np.ndarray
        Background image for the preview (BGR or grayscale).
    on_accept : callable, optional
        ``on_accept(filtered_region, display_image, name)`` called when
        the user presses *Accept*.
    """

    def __init__(
        self,
        master: tk.Widget,
        region: Region,
        source_image: np.ndarray,
        on_accept: Optional[Callable[..., Any]] = None,
    ) -> None:
        super().__init__(master)

        self.title("Region Filter - select_shape")
        self.geometry("960x780")
        self.resizable(True, True)
        self.configure(bg=_BG_DARK)

        # Modal behaviour
        self.transient(master)
        self.grab_set()

        self._region = region
        self._source_image = source_image
        self._on_accept = on_accept

        # Ensure properties are computed --------------------------------
        if not self._region.properties and self._region.num_regions > 0:
            from core.region_ops import compute_region_properties

            props = compute_region_properties(
                self._region.labels, self._region.source_image,
            )
            self._region = Region(
                labels=self._region.labels,
                num_regions=self._region.num_regions,
                properties=props,
                source_image=self._region.source_image,
                source_shape=self._region.source_shape,
            )

        # Precompute per-feature (min, max) across all regions ----------
        self._feature_ranges: Dict[str, Tuple[float, float]] = {}
        self._compute_feature_ranges()

        # Condition list (AND logic) ------------------------------------
        self._conditions: List[_FilterCondition] = []

        # Debounce timer id
        self._debounce_id: Optional[str] = None

        # PhotoImage reference (prevent GC)
        self._preview_photo: Optional[ImageTk.PhotoImage] = None

        # Cached results
        self._filtered_region: Optional[Region] = None
        self._display_image: Optional[np.ndarray] = None

        # Build UI
        self._build_styles()
        self._build_ui()

        # Initial preview (after window mapped)
        self.after(80, self._update_preview)

        self.protocol("WM_DELETE_WINDOW", self._cancel)

    # ------------------------------------------------------------------ #
    #  Feature range computation                                          #
    # ------------------------------------------------------------------ #

    def _compute_feature_ranges(self) -> None:
        """Scan all region properties and record (min, max) per feature."""
        for attr, _, _ in FEATURES:
            values: List[float] = []
            for p in self._region.properties:
                v = getattr(p, attr, None)
                if v is not None:
                    values.append(float(v))
            if values:
                lo = min(values)
                hi = max(values)
                # Avoid degenerate zero-width ranges for sliders
                if lo == hi:
                    margin = max(abs(lo) * 0.1, 1.0)
                    lo -= margin
                    hi += margin
                    # Clamp non-negative features
                    if attr in (
                        "area", "width", "height", "circularity",
                        "rectangularity", "compactness", "convexity",
                        "perimeter", "mean_value",
                    ):
                        lo = max(lo, 0.0)
                self._feature_ranges[attr] = (lo, hi)
            else:
                self._feature_ranges[attr] = (0.0, 1.0)

    # ------------------------------------------------------------------ #
    #  ttk styles (dark theme)                                            #
    # ------------------------------------------------------------------ #

    def _build_styles(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")

        style.configure("Dark.TFrame", background=_BG_DARK)
        style.configure("Medium.TFrame", background=_BG_MEDIUM)
        style.configure(
            "Dark.TLabel",
            background=_BG_DARK,
            foreground=_FG,
        )
        style.configure(
            "Dark.TButton",
            background=_BG_MEDIUM,
            foreground=_FG_WHITE,
            padding=(10, 4),
        )
        style.map(
            "Dark.TButton",
            background=[("active", _ACCENT)],
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
            "Dark.TCombobox",
            fieldbackground=_BG_MEDIUM,
            background=_BG_MEDIUM,
            foreground=_FG_WHITE,
            selectbackground=_ACCENT,
        )
        style.configure(
            "Dark.Treeview",
            background=_BG_MEDIUM,
            foreground=_FG,
            fieldbackground=_BG_MEDIUM,
            rowheight=22,
        )
        style.configure(
            "Dark.Treeview.Heading",
            background=_BG_DARK,
            foreground=_FG_WHITE,
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
        """Construct the complete dialog layout."""

        # ============================================================== #
        #  TOP: feature selector + min/max sliders + add button           #
        # ============================================================== #
        selector_frame = tk.LabelFrame(
            self,
            text=" 新增篩選條件 ",
            bg=_BG_DARK,
            fg=_FG,
            font=("", 10, "bold"),
            padx=8,
            pady=6,
        )
        selector_frame.pack(fill=tk.X, padx=10, pady=(10, 4))

        # --- Row 1: feature dropdown ---
        row1 = tk.Frame(selector_frame, bg=_BG_DARK)
        row1.pack(fill=tk.X, pady=(0, 4))

        tk.Label(
            row1, text="特徵:", bg=_BG_DARK, fg=_FG, font=("", 9),
            width=8, anchor=tk.E,
        ).pack(side=tk.LEFT)

        display_names = [
            f"{attr}  ({label})" for attr, label, _ in FEATURES
        ]
        self._feature_var = tk.StringVar(value=display_names[0])
        self._feature_combo = ttk.Combobox(
            row1,
            textvariable=self._feature_var,
            values=display_names,
            state="readonly",
            width=28,
            style="Dark.TCombobox",
        )
        self._feature_combo.pack(side=tk.LEFT, padx=(4, 0))
        self._feature_combo.bind(
            "<<ComboboxSelected>>", lambda _e: self._on_feature_selection_changed(),
        )

        # --- Row 2: min slider ---
        row2 = tk.Frame(selector_frame, bg=_BG_DARK)
        row2.pack(fill=tk.X, pady=2)

        tk.Label(
            row2, text="最小值:", bg=_BG_DARK, fg=_FG, font=("", 9),
            width=8, anchor=tk.E,
        ).pack(side=tk.LEFT)

        self._min_slider = tk.Scale(
            row2,
            from_=0,
            to=_SLIDER_STEPS,
            orient=tk.HORIZONTAL,
            bg=_BG_DARK,
            fg=_FG,
            troughcolor=_BG_MEDIUM,
            highlightthickness=0,
            sliderrelief=tk.FLAT,
            showvalue=False,
            command=self._on_min_slider,
        )
        self._min_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        self._min_val_label = tk.Label(
            row2, text="0", bg=_BG_DARK, fg=_FG_WHITE,
            font=(_MONO_FAMILY, 10), width=12, anchor=tk.W,
        )
        self._min_val_label.pack(side=tk.LEFT, padx=(0, 4))

        # --- Row 3: max slider ---
        row3 = tk.Frame(selector_frame, bg=_BG_DARK)
        row3.pack(fill=tk.X, pady=2)

        tk.Label(
            row3, text="最大值:", bg=_BG_DARK, fg=_FG, font=("", 9),
            width=8, anchor=tk.E,
        ).pack(side=tk.LEFT)

        self._max_slider = tk.Scale(
            row3,
            from_=0,
            to=_SLIDER_STEPS,
            orient=tk.HORIZONTAL,
            bg=_BG_DARK,
            fg=_FG,
            troughcolor=_BG_MEDIUM,
            highlightthickness=0,
            sliderrelief=tk.FLAT,
            showvalue=False,
            command=self._on_max_slider,
        )
        self._max_slider.set(_SLIDER_STEPS)
        self._max_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        self._max_val_label = tk.Label(
            row3, text="0", bg=_BG_DARK, fg=_FG_WHITE,
            font=(_MONO_FAMILY, 10), width=12, anchor=tk.W,
        )
        self._max_val_label.pack(side=tk.LEFT, padx=(0, 4))

        # --- Row 4: Add / Remove buttons ---
        row4 = tk.Frame(selector_frame, bg=_BG_DARK)
        row4.pack(fill=tk.X, pady=(4, 0))

        ttk.Button(
            row4, text="新增條件", style="Accent.TButton",
            command=self._add_condition,
        ).pack(side=tk.LEFT, padx=(0, 6))

        ttk.Button(
            row4, text="移除選取", style="Dark.TButton",
            command=self._remove_selected_condition,
        ).pack(side=tk.LEFT, padx=(0, 6))

        ttk.Button(
            row4, text="清除全部", style="Dark.TButton",
            command=self._clear_conditions,
        ).pack(side=tk.LEFT)

        # Populate slider labels for default feature
        self._on_feature_selection_changed()

        # ============================================================== #
        #  CONDITIONS LISTBOX                                              #
        # ============================================================== #
        cond_frame = tk.LabelFrame(
            self,
            text=" 篩選條件 (AND 邏輯) ",
            bg=_BG_DARK,
            fg=_FG,
            font=("", 10, "bold"),
            padx=6,
            pady=4,
        )
        cond_frame.pack(fill=tk.X, padx=10, pady=(2, 4))

        cond_inner = tk.Frame(cond_frame, bg=_BG_DARK)
        cond_inner.pack(fill=tk.X)

        self._cond_listbox = tk.Listbox(
            cond_inner,
            bg=_BG_MEDIUM,
            fg=_FG_WHITE,
            selectbackground=_ACCENT,
            selectforeground="#ffffff",
            font=(_MONO_FAMILY, 9),
            height=4,
            relief=tk.FLAT,
            highlightthickness=0,
        )
        cond_scroll = ttk.Scrollbar(
            cond_inner, orient=tk.VERTICAL, command=self._cond_listbox.yview,
        )
        self._cond_listbox.configure(yscrollcommand=cond_scroll.set)
        self._cond_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
        cond_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # ============================================================== #
        #  MIDDLE: Preview canvas                                          #
        # ============================================================== #
        preview_frame = tk.LabelFrame(
            self,
            text=" 預覽 (綠色=通過, 紅色=不通過) ",
            bg=_BG_DARK,
            fg=_FG,
            font=("", 10, "bold"),
            padx=4,
            pady=4,
        )
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

        self._preview_canvas = tk.Canvas(
            preview_frame, bg=_CANVAS_BG, highlightthickness=0,
        )
        self._preview_canvas.pack(fill=tk.BOTH, expand=True)
        self._preview_canvas.bind("<Configure>", lambda _e: self._schedule_update())

        # ============================================================== #
        #  STATS                                                           #
        # ============================================================== #
        stats_frame = tk.Frame(self, bg=_BG_DARK)
        stats_frame.pack(fill=tk.X, padx=10, pady=(0, 2))

        self._stats_var = tk.StringVar(
            value=f"通過: {self._region.num_regions} / {self._region.num_regions} 區域",
        )
        tk.Label(
            stats_frame,
            textvariable=self._stats_var,
            bg=_BG_DARK,
            fg="#88cc88",
            font=("", 11, "bold"),
        ).pack(side=tk.LEFT)

        # ============================================================== #
        #  RESULTS TABLE                                                   #
        # ============================================================== #
        table_frame = tk.LabelFrame(
            self,
            text=" 區域結果 ",
            bg=_BG_DARK,
            fg=_FG,
            font=("", 9, "bold"),
            padx=4,
            pady=2,
        )
        table_frame.pack(fill=tk.X, padx=10, pady=(0, 4))

        columns = (
            "index", "area", "width", "height",
            "circularity", "rectangularity", "aspect_ratio",
            "perimeter", "mean_value", "result",
        )
        self._tree = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings",
            height=6,
            style="Dark.Treeview",
        )

        headings: Dict[str, str] = {
            "index": "#",
            "area": "面積",
            "width": "寬度",
            "height": "高度",
            "circularity": "圓度",
            "rectangularity": "矩形度",
            "aspect_ratio": "長寬比",
            "perimeter": "周長",
            "mean_value": "灰度均值",
            "result": "結果",
        }
        col_widths: Dict[str, int] = {
            "index": 40,
            "area": 70,
            "width": 55,
            "height": 55,
            "circularity": 65,
            "rectangularity": 70,
            "aspect_ratio": 65,
            "perimeter": 70,
            "mean_value": 70,
            "result": 60,
        }
        for col in columns:
            self._tree.heading(col, text=headings[col])
            self._tree.column(
                col,
                width=col_widths.get(col, 60),
                anchor=tk.CENTER,
                minwidth=40,
            )

        tree_scroll_y = ttk.Scrollbar(
            table_frame, orient=tk.VERTICAL, command=self._tree.yview,
        )
        tree_scroll_x = ttk.Scrollbar(
            table_frame, orient=tk.HORIZONTAL, command=self._tree.xview,
        )
        self._tree.configure(
            yscrollcommand=tree_scroll_y.set,
            xscrollcommand=tree_scroll_x.set,
        )
        self._tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        tree_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)

        # Tag-based colouring
        self._tree.tag_configure("pass", foreground="#88cc88")
        self._tree.tag_configure("fail", foreground="#cc6666")

        # ============================================================== #
        #  BOTTOM: Accept / Cancel                                         #
        # ============================================================== #
        btn_frame = tk.Frame(self, bg=_BG_DARK)
        btn_frame.pack(fill=tk.X, padx=10, pady=(4, 10))

        ttk.Button(
            btn_frame, text="確定", style="Accent.TButton",
            command=self._accept,
        ).pack(side=tk.RIGHT, padx=(6, 0))

        ttk.Button(
            btn_frame, text="取消", style="Dark.TButton",
            command=self._cancel,
        ).pack(side=tk.RIGHT)

    # ------------------------------------------------------------------ #
    #  Slider <-> real-value mapping                                      #
    # ------------------------------------------------------------------ #

    def _current_feature_attr(self) -> str:
        """Return the attribute name of the currently selected feature."""
        text = self._feature_var.get()
        for attr, label, _ in FEATURES:
            if text.startswith(attr):
                return attr
        return FEATURES[0][0]

    def _slider_to_value(self, slider_int: int, feature: str) -> float:
        """Map an integer slider position [0, _SLIDER_STEPS] to the real
        value range of *feature*."""
        lo, hi = self._feature_ranges.get(feature, (0.0, 1.0))
        t = slider_int / _SLIDER_STEPS
        return lo + t * (hi - lo)

    def _value_to_slider(self, value: float, feature: str) -> int:
        """Map a real feature value back to a slider integer."""
        lo, hi = self._feature_ranges.get(feature, (0.0, 1.0))
        span = hi - lo
        if span <= 0:
            return 0
        t = (value - lo) / span
        t = max(0.0, min(1.0, t))
        return int(round(t * _SLIDER_STEPS))

    def _format_feature_value(self, value: float, feature: str) -> str:
        """Pretty-format a value according to the feature's precision."""
        dec = _FEATURE_DECIMALS.get(feature, 2)
        if dec == 0:
            return str(int(round(value)))
        return f"{value:.{dec}f}"

    # ------------------------------------------------------------------ #
    #  Slider callbacks                                                    #
    # ------------------------------------------------------------------ #

    def _on_feature_selection_changed(self) -> None:
        """When the feature dropdown changes, reset sliders to the full
        range of that feature."""
        feat = self._current_feature_attr()
        lo, hi = self._feature_ranges.get(feat, (0.0, 1.0))

        # Reset sliders
        self._min_slider.set(0)
        self._max_slider.set(_SLIDER_STEPS)

        self._min_val_label.configure(
            text=self._format_feature_value(lo, feat),
        )
        self._max_val_label.configure(
            text=self._format_feature_value(hi, feat),
        )

    def _on_min_slider(self, _val: str) -> None:
        """Called when the min slider moves."""
        feat = self._current_feature_attr()
        pos = self._min_slider.get()
        # Clamp: min cannot exceed max
        max_pos = self._max_slider.get()
        if pos > max_pos:
            self._min_slider.set(max_pos)
            pos = max_pos
        real_val = self._slider_to_value(pos, feat)
        self._min_val_label.configure(
            text=self._format_feature_value(real_val, feat),
        )

    def _on_max_slider(self, _val: str) -> None:
        """Called when the max slider moves."""
        feat = self._current_feature_attr()
        pos = self._max_slider.get()
        # Clamp: max cannot be less than min
        min_pos = self._min_slider.get()
        if pos < min_pos:
            self._max_slider.set(min_pos)
            pos = min_pos
        real_val = self._slider_to_value(pos, feat)
        self._max_val_label.configure(
            text=self._format_feature_value(real_val, feat),
        )

    # ------------------------------------------------------------------ #
    #  Condition management                                               #
    # ------------------------------------------------------------------ #

    def _add_condition(self) -> None:
        """Read the current feature / slider values and add a new filter
        condition to the list."""
        feat = self._current_feature_attr()
        min_val = self._slider_to_value(self._min_slider.get(), feat)
        max_val = self._slider_to_value(self._max_slider.get(), feat)

        if min_val > max_val:
            min_val, max_val = max_val, min_val

        cond = _FilterCondition(feat, min_val, max_val)
        self._conditions.append(cond)
        self._cond_listbox.insert(tk.END, cond.display_text())

        self._schedule_update()

    def _remove_selected_condition(self) -> None:
        """Remove the currently selected condition from the listbox."""
        sel = self._cond_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        self._cond_listbox.delete(idx)
        if 0 <= idx < len(self._conditions):
            del self._conditions[idx]
        self._schedule_update()

    def _clear_conditions(self) -> None:
        """Remove all conditions."""
        self._conditions.clear()
        self._cond_listbox.delete(0, tk.END)
        self._schedule_update()

    # ------------------------------------------------------------------ #
    #  Preview / filtering                                                #
    # ------------------------------------------------------------------ #

    def _schedule_update(self) -> None:
        """Debounced preview update (120 ms)."""
        if self._debounce_id is not None:
            self.after_cancel(self._debounce_id)
        self._debounce_id = self.after(120, self._update_preview)

    def _update_preview(self) -> None:
        """Apply all filter conditions (AND) and refresh preview, stats,
        and results table."""
        self._debounce_id = None

        all_indices: Set[int] = {p.index for p in self._region.properties}
        passing_indices: Set[int] = set(all_indices)

        # Apply each condition (AND)
        for cond in self._conditions:
            still_pass: Set[int] = set()
            for p in self._region.properties:
                if p.index in passing_indices and cond.matches(p):
                    still_pass.add(p.index)
            passing_indices = still_pass

        failing_indices = all_indices - passing_indices

        # Build filtered region for the accept callback
        self._filtered_region = self._region._keep_indices(
            sorted(passing_indices),
        )

        # Build display image with green / red overlays
        src = (
            self._source_image
            if self._source_image is not None
            else self._region.source_image
        )
        self._display_image = self._render_pass_fail(
            src, self._region, passing_indices, failing_indices,
        )

        # Canvas
        self._show_on_canvas(self._display_image)

        # Stats
        total = len(all_indices)
        n_pass = len(passing_indices)
        self._stats_var.set(f"通過: {n_pass} / {total} 區域")

        # Table
        self._update_table(passing_indices)

    # ------------------------------------------------------------------ #
    #  Rendering helpers                                                   #
    # ------------------------------------------------------------------ #

    def _render_pass_fail(
        self,
        source: Optional[np.ndarray],
        region: Region,
        passing: Set[int],
        failing: Set[int],
    ) -> np.ndarray:
        """Draw passing regions in green and failing regions in red
        overlaid on the source image."""
        if source is not None:
            base = source.copy()
        else:
            h, w = region.labels.shape[:2]
            base = np.zeros((h, w, 3), dtype=np.uint8)

        # Normalise to uint8 BGR
        if base.dtype != np.uint8:
            mn, mx = base.min(), base.max()
            if mx - mn > 0:
                base = ((base - mn) / (mx - mn) * 255).astype(np.uint8)
            else:
                base = np.zeros_like(base, dtype=np.uint8)

        if base.ndim == 2:
            base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
        elif base.ndim == 3 and base.shape[2] == 4:
            base = cv2.cvtColor(base, cv2.COLOR_BGRA2BGR)

        overlay = base.copy()
        green = (0, 200, 0)   # BGR
        red = (0, 0, 200)     # BGR

        for i in range(1, region.num_regions + 1):
            mask_i = (region.labels == i).astype(np.uint8)
            if mask_i.sum() == 0:
                continue
            color = green if i in passing else red
            overlay[mask_i > 0] = color

        result = cv2.addWeighted(overlay, 0.45, base, 0.55, 0)

        # Annotations: bounding box, centroid cross, label number
        for p in region.properties:
            bx, by, bw, bh = p.bbox
            color = green if p.index in passing else red
            cv2.rectangle(result, (bx, by), (bx + bw, by + bh), color, 1)

            cx_i = int(p.centroid[0])
            cy_i = int(p.centroid[1])
            arm = 5
            cv2.line(result, (cx_i - arm, cy_i), (cx_i + arm, cy_i), color, 1)
            cv2.line(result, (cx_i, cy_i - arm), (cx_i, cy_i + arm), color, 1)

            cv2.putText(
                result,
                str(p.index),
                (bx, by - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return result

    def _show_on_canvas(self, image: np.ndarray) -> None:
        """Render a BGR numpy image onto the preview canvas, scaled to
        fit while preserving the aspect ratio."""
        self._preview_canvas.update_idletasks()
        cw = max(self._preview_canvas.winfo_width(), 100)
        ch = max(self._preview_canvas.winfo_height(), 100)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        scale = min(cw / w, ch / h) * 0.95
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        pil_img = Image.fromarray(rgb).resize((new_w, new_h), Image.LANCZOS)
        self._preview_photo = ImageTk.PhotoImage(pil_img)

        self._preview_canvas.delete("all")
        self._preview_canvas.create_image(
            cw // 2, ch // 2,
            image=self._preview_photo,
            anchor=tk.CENTER,
        )

    def _update_table(self, passing: Set[int]) -> None:
        """Refresh the results Treeview with all regions and their pass/fail
        status."""
        for item in self._tree.get_children():
            self._tree.delete(item)

        for p in self._region.properties:
            passed = p.index in passing
            tag = "pass" if passed else "fail"
            self._tree.insert(
                "",
                tk.END,
                values=(
                    p.index,
                    p.area,
                    p.width,
                    p.height,
                    f"{p.circularity:.4f}",
                    f"{p.rectangularity:.4f}",
                    f"{p.aspect_ratio:.4f}",
                    f"{p.perimeter:.2f}",
                    f"{p.mean_value:.2f}",
                    "通過" if passed else "不通過",
                ),
                tags=(tag,),
            )

    # ------------------------------------------------------------------ #
    #  Accept / Cancel                                                     #
    # ------------------------------------------------------------------ #

    def _accept(self) -> None:
        """Invoke the on_accept callback with filtered results and close."""
        # Ensure we have an up-to-date result
        if self._filtered_region is None:
            self._update_preview()

        if self._on_accept is not None and self._filtered_region is not None:
            # Build human-readable operation name
            if self._conditions:
                name_parts: List[str] = []
                for cond in self._conditions:
                    dec = _FEATURE_DECIMALS.get(cond.feature, 2)
                    if dec == 0:
                        part = (
                            f"{cond.feature}[{int(cond.min_val)},"
                            f"{int(cond.max_val)}]"
                        )
                    else:
                        fmt = f"{{:.{dec}f}}"
                        part = (
                            f"{cond.feature}[{fmt.format(cond.min_val)},"
                            f"{fmt.format(cond.max_val)}]"
                        )
                    name_parts.append(part)
                name = "select_shape(" + ", ".join(name_parts) + ")"
            else:
                name = "select_shape(all)"

            disp = self._display_image
            if disp is None:
                disp = np.zeros(
                    (*self._region.labels.shape[:2], 3), dtype=np.uint8,
                )

            self._on_accept(self._filtered_region, disp, name)

        self.grab_release()
        self.destroy()

    def _cancel(self) -> None:
        """Close the dialog without accepting."""
        self.grab_release()
        self.destroy()
