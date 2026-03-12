"""Top toolbar with HALCON HDevelop-style action buttons.

Uses Unicode symbols for icons. All buttons dispatch to callback functions
provided by the main application.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Dict, Optional


class Toolbar(ttk.Frame):
    """Horizontal toolbar with grouped action buttons.

    Parameters
    ----------
    callbacks : dict[str, callable]
        Mapping of action names to callbacks. Supported keys:

        File:   "open", "save", "undo", "redo"
        View:   "fit", "zoom_in", "zoom_out", "actual_size"
        Model:  "train", "load_model", "inspect", "batch"
        Display: "grid", "crosshair"
    """

    def __init__(
        self,
        master: tk.Misc,
        callbacks: Optional[Dict[str, Callable]] = None,
        **kwargs,
    ) -> None:
        super().__init__(master, **kwargs)
        self._callbacks = callbacks or {}
        self._toggle_states: Dict[str, tk.BooleanVar] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        style = ttk.Style()
        style.configure(
            "Toolbar.TButton",
            font=("Segoe UI", 11),
            padding=(4, 2),
        )

        # ---- File group ----
        self._add_button("\U0001F4C2", "open", "\u958b\u555f\u5716\u7247 (Ctrl+O)")  # Open
        self._add_button("\U0001F4BE", "save", "\u5132\u5b58 (Ctrl+S)")  # Save
        self._add_sep()
        self._add_button("\u21A9", "undo", "\u5fa9\u539f (Ctrl+Z)")  # Undo
        self._add_button("\u21AA", "redo", "\u91cd\u505a (Ctrl+Y)")  # Redo
        self._add_sep()

        # ---- View group ----
        self._add_button("\u229E", "fit", "\u7e2e\u653e\u81f3\u7a97\u53e3 (Space)")  # Fit
        self._add_button("\U0001F50D+", "zoom_in", "\u653e\u5927 (+)")  # Zoom+
        self._add_button("\U0001F50D-", "zoom_out", "\u7e2e\u5c0f (-)")  # Zoom-
        self._add_button("1:1", "actual_size", "\u539f\u59cb\u5927\u5c0f")  # 1:1
        self._add_sep()

        # ---- Model group ----
        self._add_button("\U0001F3AF", "train", "\u8a13\u7df4\u6a21\u578b (F6)")  # Train
        self._add_button("\U0001F4E6", "load_model", "\u8f09\u5165\u6a21\u578b")  # Load Model
        self._add_button("\U0001F50D", "inspect", "\u6aa2\u6e2c\u5716\u7247 (F5)")  # Inspect
        self._add_button("\U0001F4CB", "batch", "\u6279\u6b21\u6aa2\u6e2c")  # Batch
        self._add_sep()

        # ---- HALCON tools ----
        self._add_toggle("\u25C8", "toggle_pixel_inspector", "\u50cf\u7d20\u6aa2\u67e5\u5668 (Ctrl+I)")  # Pixel Inspector
        self._add_button("\u25A7", "threshold", "\u95be\u503c\u5206\u5272 (Ctrl+T)")  # Threshold
        self._add_button("\u25A9", "blob_analysis", "Blob \u5206\u6790")  # Blob
        self._add_toggle("\u2630", "toggle_script_editor", "\u8173\u672c\u7de8\u8f2f\u5668 (F8)")  # Script Editor
        self._add_button("\u2194", "compare", "\u5716\u7247\u6bd4\u5c0d")  # Compare
        self._add_sep()

        # ---- Measurement tools ----
        self._add_toggle("\u2316", "tool_pixel_inspect", "\u50cf\u7d20\u6aa2\u67e5\u5de5\u5177 (Ctrl+Shift+I)")
        self._add_toggle("\u25AD", "tool_region_select", "\u5340\u57df\u9078\u53d6\u5de5\u5177 (Ctrl+Shift+R)")
        self._add_sep()

        # ---- Display toggles ----
        self._add_toggle("\u25A6", "grid", "\u7db2\u683c")  # Grid
        self._add_toggle("+", "crosshair", "\u5341\u5b57\u7dda")  # Crosshair

    def _add_button(self, symbol: str, action: str, tooltip: str = "") -> None:
        cb = self._callbacks.get(action, lambda: None)
        btn = ttk.Button(
            self,
            text=symbol,
            width=4,
            style="Toolbar.TButton",
            command=cb,
        )
        btn.pack(side=tk.LEFT, padx=1, pady=2)
        if tooltip:
            self._add_tooltip(btn, tooltip)

    def _add_toggle(self, symbol: str, action: str, tooltip: str = "") -> None:
        var = tk.BooleanVar(value=False)
        self._toggle_states[action] = var

        def toggle_cmd():
            var.set(not var.get())
            cb = self._callbacks.get(action)
            if cb:
                cb(var.get())

        btn = ttk.Button(
            self,
            text=symbol,
            width=4,
            style="Toolbar.TButton",
            command=toggle_cmd,
        )
        btn.pack(side=tk.LEFT, padx=1, pady=2)
        if tooltip:
            self._add_tooltip(btn, tooltip)

    def _add_sep(self) -> None:
        sep = ttk.Separator(self, orient=tk.VERTICAL)
        sep.pack(side=tk.LEFT, fill=tk.Y, padx=4, pady=4)

    def get_toggle_state(self, action: str) -> bool:
        var = self._toggle_states.get(action)
        return var.get() if var else False

    def set_toggle_state(self, action: str, state: bool) -> None:
        """Programmatically set a toggle state without firing the callback."""
        var = self._toggle_states.get(action)
        if var is not None:
            var.set(state)

    def set_tool_exclusive(self, active_action: str, tool_actions: list) -> None:
        """Ensure only one tool toggle in the group is active."""
        for action in tool_actions:
            var = self._toggle_states.get(action)
            if var is not None:
                var.set(action == active_action)

    # ------------------------------------------------------------------
    # Tooltip (simple hover text)
    # ------------------------------------------------------------------

    @staticmethod
    def _add_tooltip(widget: tk.Widget, text: str) -> None:
        tip_window = [None]

        def show(event: tk.Event) -> None:
            if tip_window[0] is not None:
                return
            tw = tk.Toplevel(widget)
            tw.wm_overrideredirect(True)
            tw.wm_geometry(f"+{event.x_root + 10}+{event.y_root + 10}")
            label = tk.Label(
                tw,
                text=text,
                justify=tk.LEFT,
                background="#ffffe0",
                foreground="#333333",
                relief=tk.SOLID,
                borderwidth=1,
                font=("Segoe UI", 9),
                padx=4,
                pady=2,
            )
            label.pack()
            tip_window[0] = tw

        def hide(_event: tk.Event) -> None:
            if tip_window[0] is not None:
                tip_window[0].destroy()
                tip_window[0] = None

        widget.bind("<Enter>", show)
        widget.bind("<Leave>", hide)
