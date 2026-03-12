"""Main HALCON HDevelop-style application window.

Assembles the full layout:
- Menu bar
- Toolbar
- Three-panel PanedWindow: Pipeline (left) | Image Viewer (centre) | Properties + Operations (right)
- Status bar with position, pixel value, zoom level

Orchestrates all interactions between panels, dialogs, and the core
inference / training pipelines.

Method groups are split into mixin classes for maintainability:
- MenuMixin        (mixins_menu.py)      -- menu bar construction
- ImageOpsMixin    (mixins_image_ops.py)  -- file I/O, pipeline, model, inspection, DL ops, recipes, help
- DialogMixin      (mixins_dialogs.py)    -- dialog-opening methods
- RegionMixin      (mixins_region.py)     -- region / morphology operations
- HalconOpsMixin   (mixins_halcon.py)     -- HALCON operators, filter dialogs
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
from dl_anomaly.gui.image_viewer import ImageViewer
from dl_anomaly.gui.operations_panel import OperationsPanel
from dl_anomaly.gui.pipeline_panel import PipelinePanel
from dl_anomaly.gui.properties_panel import PropertiesPanel
from dl_anomaly.gui.toolbar import Toolbar

from dl_anomaly.gui.mixins_menu import MenuMixin
from dl_anomaly.gui.mixins_image_ops import ImageOpsMixin
from dl_anomaly.gui.mixins_dialogs import DialogMixin
from dl_anomaly.gui.mixins_region import RegionMixin
from dl_anomaly.gui.mixins_halcon import HalconOpsMixin

# 延遲載入的模組（用到時才 import，加速啟動）
# - dl_anomaly.core.anomaly_scorer.AnomalyScorer
# - dl_anomaly.core.preprocessor.ImagePreprocessor
# - dl_anomaly.gui.dialogs: BatchInspectDialog, HistogramDialog, ModelInfoDialog, ...
# - dl_anomaly.gui.compare_dialog, recipe_apply_dialog, batch_compare_dialog
# - dl_anomaly.gui.shape_matching_dialog, metrology_dialog, roi_dialog, ...
# - dl_anomaly.visualization.heatmap

logger = logging.getLogger(__name__)


class HalconApp(
    MenuMixin,
    ImageOpsMixin,
    DialogMixin,
    RegionMixin,
    HalconOpsMixin,
    tk.Tk,
):
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
        self.bind_all("<Control-m>", lambda e: self._open_shape_matching())
        self.bind_all("<Control-M>", lambda e: self._open_shape_matching())
        self.bind_all("<Control-Shift-m>", lambda e: self._open_metrology())
        self.bind_all("<Control-Shift-M>", lambda e: self._open_metrology())
        self.bind_all("<Control-r>", lambda e: self._open_roi_manager())
        self.bind_all("<Control-R>", lambda e: self._open_roi_manager())
        self.bind_all("<Control-Shift-p>", lambda e: self._open_advanced_models())
        self.bind_all("<Control-Shift-P>", lambda e: self._open_advanced_models())
        self.bind_all("<Control-Shift-t>", lambda e: self._open_inspection_tools())
        self.bind_all("<Control-Shift-T>", lambda e: self._open_inspection_tools())
        self.bind_all("<Control-Shift-v>", lambda e: self._open_mvp_tools())
        self.bind_all("<Control-Shift-V>", lambda e: self._open_mvp_tools())
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
    # Close
    # ==================================================================

    def _on_close(self) -> None:
        self._app_state.save_geometry(self)
        self._app_state.save_sash_positions(self._paned)
        self.destroy()
