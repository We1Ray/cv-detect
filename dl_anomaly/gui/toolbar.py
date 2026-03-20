"""Top toolbar with Industrial Vision-style action buttons.

Uses Unicode symbols for icons. All buttons dispatch to callback functions
provided by the main application.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Dict, Optional

from dl_anomaly.gui.platform_keys import display, display_shift


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
        self._add_group_label("\u6a94\u6848")
        self._add_button("\U0001F4C2", "open", f"\u958b\u555f\u5716\u7247 ({display('O')})")  # Open
        self._add_button("\U0001F4BE", "save", f"\u5132\u5b58 ({display('S')})")  # Save
        self._add_sep()
        self._add_button("\u21A9", "undo", f"\u5fa9\u539f ({display('Z')})")  # Undo
        self._add_button("\u21AA", "redo", f"\u91cd\u505a ({display('Y')})")  # Redo
        self._add_sep()

        # ---- View group ----
        self._add_group_label("\u6aa2\u8996")
        self._add_button("\u229E", "fit", "\u7e2e\u653e\u81f3\u7a97\u53e3 (Space)")  # Fit
        self._add_button("\U0001F50D+", "zoom_in", "\u653e\u5927 (+)")  # Zoom+
        self._add_button("\U0001F50D-", "zoom_out", "\u7e2e\u5c0f (-)")  # Zoom-
        self._add_button("1:1", "actual_size", "\u539f\u59cb\u5927\u5c0f")  # 1:1
        self._add_sep()

        # ---- DL Model group ----
        self._add_group_label("DL")
        self._add_button("\U0001F3AF", "train", "DL \u8a13\u7df4 (F6)")  # Train
        self._add_button("\U0001F4E6", "load_model", "DL \u8f09\u5165 Checkpoint")  # Load
        self._add_button("\U0001F50D", "inspect", "DL \u6aa2\u6e2c (F5)")  # Inspect
        self._add_button("\U0001F4CB", "batch", "DL \u6279\u6b21\u6aa2\u6e2c")  # Batch
        self._add_sep()

        # ---- VM Model group ----
        self._add_group_label("VM")
        self._add_button("\U0001F4CA", "vm_train", "VM \u8a13\u7df4 (\u7d71\u8a08\u6a21\u578b)")
        self._add_button("\U0001F4C1", "vm_load", "VM \u8f09\u5165\u6a21\u578b (.npz)")
        self._add_button("\U0001F50E", "vm_inspect", "VM \u6aa2\u6e2c")
        self._add_sep()

        # ---- Vision tools ----
        self._add_group_label("\u5de5\u5177")
        self._add_toggle("\u25C8", "toggle_pixel_inspector", f"\u50cf\u7d20\u6aa2\u67e5\u5668 ({display('I')})")  # Pixel Inspector
        self._add_button("\u25A7", "threshold", f"\u95be\u503c\u5206\u5272 ({display('T')})")  # Threshold
        self._add_button("\u25A9", "blob_analysis", "Blob \u5206\u6790")  # Blob
        self._add_toggle("\u2630", "toggle_script_editor", "\u8173\u672c\u7de8\u8f2f\u5668 (F8)")  # Script Editor
        self._add_button("\u2194", "compare", "\u5716\u7247\u6bd4\u5c0d")  # Compare
        self._add_sep()

        # ---- Measurement tools ----
        self._add_group_label("\u91cf\u6e2c")
        self._add_toggle("\u2316", "tool_pixel_inspect", f"\u50cf\u7d20\u6aa2\u67e5\u5de5\u5177 ({display_shift('I')})")
        self._add_toggle("\u25AD", "tool_region_select", f"\u5340\u57df\u9078\u53d6\u5de5\u5177 ({display_shift('R')})")
        self._add_sep()

        # ---- Display toggles ----
        self._add_group_label("\u986f\u793a")
        self._add_toggle("\u25A6", "grid", "\u7db2\u683c")  # Grid
        self._add_toggle("+", "crosshair", "\u5341\u5b57\u7dda")  # Crosshair

    def _add_group_label(self, text: str) -> None:
        """Add a small group label above the next buttons."""
        lbl = tk.Label(
            self,
            text=text,
            bg="#2b2b2b",
            fg="#777777",
            font=("Segoe UI", 7),
            anchor=tk.S,
        )
        lbl.pack(side=tk.LEFT, padx=(4, 0), pady=(0, 0))

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
        hide_id = [None]

        def show(event: tk.Event) -> None:
            if tip_window[0] is not None:
                return
            # Delay showing tooltip by 400ms
            def _show():
                if tip_window[0] is not None:
                    return
                tw = tk.Toplevel(widget)
                tw.wm_overrideredirect(True)
                tw.wm_geometry(f"+{event.x_root + 12}+{event.y_root + 20}")
                # Prevent the tooltip from receiving focus on macOS
                tw.wm_attributes("-topmost", True)

                frame = tk.Frame(tw, bg="#1e1e1e", highlightbackground="#555555",
                               highlightthickness=1, padx=1, pady=1)
                frame.pack()

                label = tk.Label(
                    frame,
                    text=text,
                    justify=tk.LEFT,
                    background="#1e1e1e",
                    foreground="#cccccc",
                    font=("Segoe UI", 9),
                    padx=8,
                    pady=4,
                )
                label.pack()
                tip_window[0] = tw
            hide_id[0] = widget.after(400, _show)

        def hide(_event: tk.Event) -> None:
            if hide_id[0] is not None:
                widget.after_cancel(hide_id[0])
                hide_id[0] = None
            if tip_window[0] is not None:
                tip_window[0].destroy()
                tip_window[0] = None

        widget.bind("<Enter>", show)
        widget.bind("<Leave>", hide)
