"""Main Industrial Vision-style application window.

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
- VisionOpsMixin   (mixins_vision.py)     -- Vision operators, filter dialogs
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
from shared.judgment_indicator import JudgmentIndicator
from shared.live_inspection_panel import LiveInspectionPanel
from shared.dashboard_panel import DashboardPanel

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
from dl_anomaly.gui.mixins_vision import VisionOpsMixin
from dl_anomaly.gui.mixins_vm_ops import VMOpsMixin
from dl_anomaly.gui.platform_keys import (
    IS_MAC, bind_mod, bind_mod_shift,
    display, display_shift, DELETE_LABEL,
)

# 延遲載入的模組（用到時才 import，加速啟動）
# - dl_anomaly.core.anomaly_scorer.AnomalyScorer
# - dl_anomaly.core.preprocessor.ImagePreprocessor
# - dl_anomaly.gui.dialogs: BatchInspectDialog, HistogramDialog, ModelInfoDialog, ...
# - dl_anomaly.gui.compare_dialog, recipe_apply_dialog, batch_compare_dialog
# - dl_anomaly.gui.shape_matching_dialog, metrology_dialog, roi_dialog, ...
# - dl_anomaly.visualization.heatmap

logger = logging.getLogger(__name__)


class InspectorApp(
    MenuMixin,
    ImageOpsMixin,
    DialogMixin,
    RegionMixin,
    VisionOpsMixin,
    VMOpsMixin,
    tk.Tk,
):
    """Top-level Industrial Vision-style application window."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config

        self.title("CV \u7f3a\u9677\u5075\u6e2c\u5668 v2.0 \u2014 DL + Variation Model")
        self.geometry("1400x900")
        self.minsize(1000, 750)

        # Configure dark theme
        self._setup_theme()

        # Persistent application state
        self._app_state = AppState("dl_anomaly")

        # Core state
        self._inference_pipeline = None  # InferencePipeline instance
        self._model = None
        self._model_state: Optional[Dict] = None
        self._vm_model = None  # VariationModel instance
        self._current_image_path: Optional[str] = None
        self._recent_files: List[str] = []
        self._undo_stack: List[int] = []  # pipeline step indices
        self._redo_stack: List[int] = []
        self._initial_loaded: bool = False  # True after first image load

        # Vision feature state
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

        # Enhanced button hover/active maps
        style.map("TButton",
            background=[("pressed", "#4a4a6e"), ("active", "#454545")],
            foreground=[("pressed", "#ffffff"), ("active", "#ffffff")],
        )
        style.map("Toolbar.TButton",
            background=[("pressed", "#4a4a6e"), ("active", "#454545")],
            foreground=[("pressed", "#ffffff"), ("active", "#ffffff")],
        )

        # Tab selection effect
        style.map("TNotebook.Tab",
            background=[("selected", "#0078d4"), ("active", "#454545")],
            foreground=[("selected", "#ffffff")],
        )

        # Entry focus highlight
        style.map("TEntry",
            fieldbackground=[("focus", "#454545")],
        )

        # Combobox states
        style.map("TCombobox",
            fieldbackground=[("focus", "#454545"), ("readonly", "#3c3c3c")],
        )

        # Spinbox
        style.configure("TSpinbox", fieldbackground="#3c3c3c", foreground="#e0e0e0")
        style.map("TSpinbox",
            fieldbackground=[("focus", "#454545")],
        )

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
            "vm_train": self._cmd_vm_train,
            "vm_load": self._cmd_vm_load_model,
            "vm_inspect": self._cmd_vm_inspect_single,
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
            context_menu_callback=self._on_context_menu,
            show_loss_panel=True,
        )
        self._paned.add(self._viewer, weight=1)

        # -- Right: Properties + Operations --
        right_frame = ttk.Frame(self._paned)
        self._paned.add(right_frame, weight=0)

        # OK/NG Judgment Indicator (top of right panel, always visible)
        self._judgment_indicator = JudgmentIndicator(right_frame, height=100)
        self._judgment_indicator.pack(fill=tk.X, padx=2, pady=(2, 4))

        # Right panel uses PanedWindow for resizable sections
        right_paned = ttk.PanedWindow(right_frame, orient=tk.VERTICAL)
        right_paned.pack(fill=tk.BOTH, expand=True)

        self._props_panel = PropertiesPanel(right_paned)
        self._props_panel.set_region_highlight_callback(self._on_region_highlight)
        self._props_panel.set_region_remove_callback(self._on_region_remove)
        right_paned.add(self._props_panel, weight=0)

        self._ops_panel = OperationsPanel(
            right_paned,
            on_apply=self._on_operation_applied,
            get_current_image=self._get_current_image,
        )
        right_paned.add(self._ops_panel, weight=1)

        # Live Inspection Panel
        self._live_panel = LiveInspectionPanel(
            right_paned,
            on_start=self._on_live_start,
            on_stop=self._on_live_stop,
            on_inspect_single=self._on_live_inspect_single,
        )
        right_paned.add(self._live_panel, weight=0)

        # SPC Dashboard Panel
        self._dashboard_panel = DashboardPanel(right_paned)
        right_paned.add(self._dashboard_panel, weight=0)

        self._history_panel = HistoryPanel(right_paned)
        right_paned.add(self._history_panel, weight=0)

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

        ttk.Separator(sb, orient=tk.VERTICAL).pack(side=tk.RIGHT, fill=tk.Y, padx=4)

        # Device indicator
        device_text = self.config.device.upper()
        if device_text == "AUTO":
            try:
                import torch
                if torch.cuda.is_available():
                    device_text = "CUDA"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device_text = "MPS"
                else:
                    device_text = "CPU"
            except ImportError:
                device_text = "CPU"
        self._device_var = tk.StringVar(value=f"\U0001F5A5 {device_text}")
        ttk.Label(sb, textvariable=self._device_var, anchor=tk.E, width=10).pack(side=tk.RIGHT, padx=4)

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
        """Show a categorized keyboard shortcuts dialog."""
        dlg = tk.Toplevel(self)
        dlg.title("\u5feb\u6377\u9375\u4e00\u89bd")
        dlg.transient(self)
        dlg.grab_set()
        dlg.resizable(False, False)
        dlg.configure(bg="#2b2b2b")

        # Title
        tk.Label(
            dlg, text="\u9375\u76e4\u5feb\u6377\u9375", bg="#2b2b2b", fg="#e0e0e0",
            font=("Segoe UI", 14, "bold"), pady=8,
        ).pack(fill=tk.X, padx=16)

        # Scrollable frame
        container = ttk.Frame(dlg)
        container.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 8))

        categories = [
            ("\u6a94\u6848\u64cd\u4f5c", [
                (display("O"), "\u958b\u555f\u5716\u7247"),
                (display("S"), "\u5132\u5b58\u5716\u7247"),
                (display("Z"), "\u5fa9\u539f"),
                (display("Y"), "\u91cd\u505a"),
            ]),
            ("\u6aa2\u8996\u63a7\u5236", [
                ("Space", "\u7e2e\u653e\u81f3\u7a97\u53e3"),
                ("+  /  -", "\u653e\u5927 / \u7e2e\u5c0f"),
                ("1:1", "\u539f\u59cb\u5927\u5c0f"),
                ("Escape", "\u8fd4\u56de\u5e73\u79fb\u6a21\u5f0f"),
            ]),
            ("\u5de5\u5177", [
                (display("I"), "\u50cf\u7d20\u503c\u6aa2\u67e5\u5668\u8996\u7a97"),
                (display_shift("I"), "\u50cf\u7d20\u6aa2\u67e5\u5de5\u5177"),
                (display_shift("R"), "\u5340\u57df\u9078\u53d6\u5de5\u5177"),
                (display("T"), "\u95be\u503c\u5206\u5272"),
                (display("M"), "\u5f62\u72c0\u5339\u914d"),
                (display_shift("M"), "\u91cf\u6e2c\u5de5\u5177"),
                (display("R"), "ROI \u7ba1\u7406"),
            ]),
            ("\u6a21\u578b & \u6aa2\u6e2c", [
                ("F5", "\u57f7\u884c\u6aa2\u6e2c"),
                ("F6", "\u8a13\u7df4\u6a21\u578b"),
                (display_shift("P"), "PatchCore / ONNX"),
                (display_shift("A"), "\u81ea\u52d5\u95be\u503c\u6821\u6e96"),
            ]),
            ("\u9032\u968e\u5de5\u5177", [
                (display_shift("T"), "\u6aa2\u6e2c\u5de5\u5177 (FFT/\u8272\u5f69/OCR)"),
                (display_shift("E"), "\u5de5\u7a0b\u5de5\u5177"),
                (display_shift("V"), "MVP \u5de5\u5177"),
                ("F8", "\u8173\u672c\u7de8\u8f2f\u5668"),
                ("F9", "\u57f7\u884c\u8173\u672c"),
            ]),
            ("\u5176\u4ed6", [
                (DELETE_LABEL, "\u522a\u9664\u6b65\u9a5f"),
                ("F1", "\u5feb\u6377\u9375\u4e00\u89bd"),
            ]),
        ]

        row = 0
        for cat_name, shortcuts in categories:
            # Category header
            tk.Label(
                container, text=cat_name, bg="#2b2b2b", fg="#0078d4",
                font=("Segoe UI", 10, "bold"), anchor=tk.W,
            ).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(8, 2))
            row += 1

            # Separator
            sep = tk.Frame(container, bg="#333333", height=1)
            sep.grid(row=row, column=0, columnspan=2, sticky=tk.EW, pady=(0, 4))
            row += 1

            for key, desc in shortcuts:
                # Key badge
                key_frame = tk.Frame(container, bg="#333333", padx=6, pady=2)
                key_frame.grid(row=row, column=0, sticky=tk.E, padx=(0, 8), pady=1)
                tk.Label(
                    key_frame, text=key, bg="#333333", fg="#cccccc",
                    font=("Consolas", 9, "bold"),
                ).pack()
                # Description
                tk.Label(
                    container, text=desc, bg="#2b2b2b", fg="#b0b0b0",
                    font=("Segoe UI", 9), anchor=tk.W,
                ).grid(row=row, column=1, sticky=tk.W, pady=1)
                row += 1

        # Close button
        btn_frame = tk.Frame(dlg, bg="#2b2b2b")
        btn_frame.pack(fill=tk.X, padx=16, pady=(4, 12))
        tk.Button(
            btn_frame, text="\u95dc\u9589", bg="#3c3c3c", fg="#e0e0e0",
            activebackground="#4a4a6e", activeforeground="#ffffff",
            relief=tk.FLAT, padx=20, pady=4, font=("Segoe UI", 10),
            command=dlg.destroy,
        ).pack(anchor=tk.E)

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
        # --- Mod+key shortcuts (Cmd on macOS, Ctrl on Win/Linux) ---
        bind_mod(self, "o", lambda e: self._cmd_open_image(), bind_all=True)
        bind_mod(self, "s", lambda e: self._cmd_save_image(), bind_all=True)
        bind_mod(self, "z", lambda e: self._cmd_undo(), bind_all=True)
        bind_mod(self, "y", lambda e: self._cmd_redo(), bind_all=True)
        bind_mod(self, "i", lambda e: self._toggle_pixel_inspector(), bind_all=True)
        bind_mod(self, "t", lambda e: self._open_threshold_dialog(), bind_all=True)
        bind_mod(self, "m", lambda e: self._open_shape_matching(), bind_all=True)
        bind_mod(self, "r", lambda e: self._open_roi_manager(), bind_all=True)

        # --- Mod+Shift+key shortcuts ---
        bind_mod_shift(self, "i", lambda e: self._cmd_tool_pixel_inspect(
            not self._toolbar.get_toggle_state("tool_pixel_inspect")), bind_all=True)
        bind_mod_shift(self, "r", lambda e: self._cmd_tool_region_select(
            not self._toolbar.get_toggle_state("tool_region_select")), bind_all=True)
        bind_mod_shift(self, "m", lambda e: self._open_metrology(), bind_all=True)
        bind_mod_shift(self, "p", lambda e: self._open_advanced_models(), bind_all=True)
        bind_mod_shift(self, "t", lambda e: self._open_inspection_tools(), bind_all=True)
        bind_mod_shift(self, "v", lambda e: self._open_mvp_tools(), bind_all=True)
        bind_mod_shift(self, "e", lambda e: self._open_engineering_tools(), bind_all=True)
        bind_mod_shift(self, "a", lambda e: self._open_auto_tune(), bind_all=True)

        # --- Plain keys (platform-independent) ---
        self.bind_all("<space>", lambda e: self._cmd_fit())
        self.bind_all("<plus>", lambda e: self._cmd_zoom_in())
        self.bind_all("<equal>", lambda e: self._cmd_zoom_in())
        self.bind_all("<minus>", lambda e: self._cmd_zoom_out())
        self.bind_all("<F5>", lambda e: self._cmd_inspect_single())
        self.bind_all("<F6>", lambda e: self._cmd_train())
        self.bind_all("<F8>", lambda e: self._toggle_script_editor())
        self.bind_all("<F9>", lambda e: self._run_script())
        self.bind_all("<Delete>", lambda e: self._cmd_delete_step())
        self.bind_all("<BackSpace>", lambda e: self._cmd_delete_step())
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
        if not messagebox.askyesno("確認", "確定要刪除此步驟嗎？"):
            return
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
    # Right-click context menu
    # ==================================================================

    def _on_context_menu(self, event, img_coords, region) -> None:
        """Build and show the right-click context menu."""
        menu = tk.Menu(self, tearoff=0, bg="#2b2b2b", fg="#e0e0e0",
                       activebackground="#3a3a5c", activeforeground="#ffffff",
                       font=("Segoe UI", 10))

        has_image = self._get_current_image() is not None
        has_region = region is not None  # (x, y, w, h)

        # -- Region operations (shown when a region is selected) --
        if has_region:
            rx, ry, rw, rh = region
            menu.add_command(
                label=f"\u2702 \u88c1\u5207\u5340\u57df ({rw}\u00d7{rh})",
                command=lambda: self._ctx_crop_region(rx, ry, rw, rh),
            )
            menu.add_command(
                label="\U0001f9e9 \u5206\u5272\u5340\u57df (U-Net)",
                command=lambda: self._ctx_segment_region(rx, ry, rw, rh),
            )
            menu.add_command(
                label="\U0001f4ca \u5340\u57df\u76f4\u65b9\u5716",
                command=lambda: self._ctx_region_histogram(rx, ry, rw, rh),
            )
            menu.add_command(
                label="\U0001f50d \u5340\u57df\u95be\u503c",
                command=lambda: self._ctx_region_threshold(rx, ry, rw, rh),
            )
            menu.add_command(
                label="\U0001f4be \u5132\u5b58\u5340\u57df",
                command=lambda: self._ctx_save_region(rx, ry, rw, rh),
            )
            menu.add_separator()

        # -- Image operations --
        if has_image:
            menu.add_command(label="\u2b50 \u7e2e\u653e\u81f3\u7a97\u53e3", command=self._cmd_fit)
            menu.add_command(label="1:1 \u539f\u59cb\u5927\u5c0f", command=self._cmd_actual_size)
            menu.add_separator()
            menu.add_command(label="\U0001f4be \u5132\u5b58\u5716\u7247", command=self._cmd_save_image)
            menu.add_command(label="\U0001f4ca \u76f4\u65b9\u5716", command=self._cmd_histogram)
            menu.add_separator()

            # Quick operations
            quick_menu = tk.Menu(menu, tearoff=0, bg="#2b2b2b", fg="#e0e0e0",
                                activebackground="#3a3a5c", activeforeground="#ffffff")
            quick_menu.add_command(label="\u7070\u968e", command=lambda: self._ctx_quick_op("grayscale"))
            quick_menu.add_command(label="\u6a21\u7cca", command=lambda: self._ctx_quick_op("blur"))
            quick_menu.add_command(label="\u908a\u7de3\u5075\u6e2c", command=lambda: self._ctx_quick_op("edge"))
            quick_menu.add_command(label="\u76f4\u65b9\u5716\u5747\u8861", command=lambda: self._ctx_quick_op("histeq"))
            quick_menu.add_command(label="\u53cd\u8272", command=lambda: self._ctx_quick_op("invert"))
            menu.add_cascade(label="\u5feb\u901f\u64cd\u4f5c", menu=quick_menu)

            menu.add_separator()
            menu.add_command(label="\u95be\u503c\u5206\u5272", command=self._open_threshold_dialog)
            menu.add_command(label="\u5f62\u72c0\u5339\u914d", command=self._open_shape_matching)

            if self._inference_pipeline is not None:
                menu.add_separator()
                menu.add_command(label="\u25b6 \u57f7\u884c\u6aa2\u6e2c (F5)", command=self._cmd_inspect_single)

        # -- Tool switching --
        menu.add_separator()
        tool_menu = tk.Menu(menu, tearoff=0, bg="#2b2b2b", fg="#e0e0e0",
                            activebackground="#3a3a5c", activeforeground="#ffffff")
        tool_menu.add_command(label="\u5e73\u79fb", command=lambda: self._set_active_tool("pan"))
        tool_menu.add_command(label="\u50cf\u7d20\u6aa2\u67e5", command=lambda: self._set_active_tool("pixel_inspect"))
        tool_menu.add_command(label="\u5340\u57df\u9078\u53d6", command=lambda: self._set_active_tool("region_select"))
        menu.add_cascade(label="\u5207\u63db\u5de5\u5177", menu=tool_menu)

        menu.tk_popup(event.x_root, event.y_root)

    # ------------------------------------------------------------------
    # Context menu actions
    # ------------------------------------------------------------------

    def _ctx_crop_region(self, x: int, y: int, w: int, h: int) -> None:
        """Crop current image to the selected region."""
        img = self._get_current_image()
        if img is None:
            return
        cropped = img[y:y + h, x:x + w].copy()
        self._add_pipeline_step(f"\u88c1\u5207 ({w}\u00d7{h})", cropped)
        self.set_status(f"\u5df2\u88c1\u5207\u5340\u57df: ({x},{y}) {w}\u00d7{h}")

    def _ctx_segment_region(self, x: int, y: int, w: int, h: int) -> None:
        """Run U-Net segmentation on the selected region."""
        img = self._get_current_image()
        if img is None:
            return
        roi = img[y:y + h, x:x + w].copy()
        full_img = img.copy()
        region_bounds = (x, y, w, h)

        def _compute():
            from shared.core.unet_segment import UNetInference
            unet = UNetInference(device=self.config.device)
            result = unet.detect(roi)
            return result

        def _done(result):
            mask = result["mask"]
            # Resize mask back to region size if needed
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

            # Create overlay on full image
            overlay = full_img.copy()
            if overlay.ndim == 2:
                overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)

            # Color the segmented region
            color_mask = np.zeros_like(overlay[y:y + h, x:x + w])
            color_mask[mask > 0] = [0, 255, 0]  # Green for segmented areas
            roi_area = overlay[y:y + h, x:x + w]
            blended = cv2.addWeighted(roi_area, 0.7, color_mask, 0.3, 0)
            overlay[y:y + h, x:x + w] = blended

            # Draw region border
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 204, 0), 2)

            score = result.get("score", 0)
            label = "\u7f3a\u9677" if result.get("is_defective", False) else "\u6b63\u5e38"
            self._add_pipeline_step(
                f"\u5206\u5272 ({label}, {score:.3f})", overlay
            )
            self.set_status_success(
                f"\u5340\u57df\u5206\u5272\u5b8c\u6210: {label} | \u5206\u6578={score:.4f}"
            )

        def _error(exc):
            self._show_error("\u5340\u57df\u5206\u5272\u5931\u6557", exc)

        self._run_in_bg(_compute, on_done=_done, on_error=_error,
                        status_msg="\u5340\u57df\u5206\u5272\u4e2d...")

    def _ctx_region_histogram(self, x: int, y: int, w: int, h: int) -> None:
        """Show histogram for the selected region."""
        from dl_anomaly.gui.dialogs import HistogramDialog
        img = self._get_current_image()
        if img is None:
            return
        roi = img[y:y + h, x:x + w]
        HistogramDialog(self, roi, title_text=f"\u5340\u57df\u76f4\u65b9\u5716 ({w}\u00d7{h})")

    def _ctx_region_threshold(self, x: int, y: int, w: int, h: int) -> None:
        """Apply Otsu threshold to the selected region and overlay on full image."""
        img = self._get_current_image()
        if img is None:
            return
        roi = img[y:y + h, x:x + w].copy()
        if roi.ndim == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        else:
            gray = roi
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Create result: threshold applied only within region
        result = img.copy()
        if result.ndim == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        result[y:y + h, x:x + w] = mask_rgb
        cv2.rectangle(result, (x, y), (x + w, y + h), (255, 204, 0), 2)
        self._add_pipeline_step(f"\u5340\u57df\u95be\u503c ({w}\u00d7{h})", result)

    def _ctx_save_region(self, x: int, y: int, w: int, h: int) -> None:
        """Save the selected region as an image file."""
        img = self._get_current_image()
        if img is None:
            return
        roi = img[y:y + h, x:x + w]
        from tkinter import filedialog as fd
        path = fd.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp")],
            initialfile=f"region_{w}x{h}.png",
        )
        if path:
            if roi.ndim == 2:
                pil_img = Image.fromarray(roi, mode="L")
            else:
                pil_img = Image.fromarray(roi, mode="RGB")
            pil_img.save(path)
            self.set_status(f"\u5340\u57df\u5df2\u5132\u5b58: {path}")

    def _ctx_quick_op(self, op: str) -> None:
        """Apply a quick image operation from the context menu."""
        img = self._get_current_image()
        if img is None:
            return
        if op == "grayscale":
            if img.ndim == 3:
                result = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                result = img
            self._add_pipeline_step("\u7070\u968e", result)
        elif op == "blur":
            result = cv2.GaussianBlur(img, (0, 0), 4.0)
            self._add_pipeline_step("\u6a21\u7cca", result)
        elif op == "edge":
            if img.ndim == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img
            edges = cv2.Canny(gray, 50, 150)
            self._add_pipeline_step("\u908a\u7de3", edges)
        elif op == "histeq":
            if img.ndim == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img
            result = cv2.equalizeHist(gray)
            self._add_pipeline_step("\u76f4\u65b9\u5716\u5747\u8861", result)
        elif op == "invert":
            result = 255 - img
            self._add_pipeline_step("\u53cd\u8272", result)

    # ==================================================================
    # Live inspection callbacks
    # ==================================================================

    def _on_live_start(self, source: str, interval: int) -> None:
        """Start live inspection loop."""
        if self._inference_pipeline is None:
            messagebox.showwarning("警告", "請先載入模型再啟動即時檢測。")
            self._live_panel.stop_inspection()
            return
        self.set_status(f"即時檢測啟動中... 來源: {source}, 間隔: {interval}ms")

    def _on_live_stop(self) -> None:
        """Stop live inspection loop."""
        self.set_status("即時檢測已停止")

    def _on_live_inspect_single(self) -> None:
        """Run a single inspection from the live panel."""
        self._cmd_inspect_single()

    def update_judgment(self, is_pass: bool, score: float, message: str = "") -> None:
        """Update the OK/NG judgment indicator with inspection results.

        Called after each inspection completes to provide visual feedback.
        """
        self._judgment_indicator.set_result(is_pass, score, message)
        self._dashboard_panel.update_from_result(is_pass, score)
        # Also update live panel if running
        if self._live_panel.is_running:
            img_path = self._current_image_path or ""
            self._live_panel.update_result(is_pass, score, img_path)

    # ==================================================================
    # Close
    # ==================================================================

    def _on_close(self) -> None:
        # Stop live inspection if running
        if self._live_panel.is_running:
            self._live_panel.stop_inspection()
        self._app_state.save_geometry(self)
        self._app_state.save_sash_positions(self._paned)
        self.destroy()
