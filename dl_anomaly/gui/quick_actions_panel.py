"""Quick Actions panel — one-click common industrial workflows.

Provides grouped action buttons for the most frequent operator tasks,
eliminating the need to navigate deep menu hierarchies.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Dict, Optional

import platform as _platform

_SYS = _platform.system()
if _SYS == "Darwin":
    _FONT_FAMILY = "Helvetica Neue"
    _MONO_FAMILY = "Menlo"
elif _SYS == "Linux":
    _FONT_FAMILY = "DejaVu Sans"
    _MONO_FAMILY = "DejaVu Sans Mono"
else:
    _FONT_FAMILY = _FONT_FAMILY
    _MONO_FAMILY = "Consolas"


class QuickActionsPanel(ttk.Frame):
    """Grouped quick-action buttons for industrial inspection workflows."""

    def __init__(
        self,
        master: tk.Misc,
        callbacks: Optional[Dict[str, Callable]] = None,
        **kwargs,
    ) -> None:
        super().__init__(master, **kwargs)
        self._cb = callbacks or {}
        self._build_ui()

    def _build_ui(self) -> None:
        # Scrollable container
        canvas = tk.Canvas(self, bg="#2b2b2b", highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        import platform as _platform

        def _on_wheel(event):
            if _platform.system() == "Darwin":
                canvas.yview_scroll(-event.delta, "units")
            else:
                canvas.yview_scroll(-event.delta // 120, "units")

        def _bind_scroll(event=None):
            canvas.bind_all("<MouseWheel>", _on_wheel)

        def _unbind_scroll(event=None):
            canvas.unbind_all("<MouseWheel>")

        canvas.bind("<Enter>", _bind_scroll)
        canvas.bind("<Leave>", _unbind_scroll)
        inner.bind("<Enter>", _bind_scroll)
        inner.bind("<Leave>", _unbind_scroll)

        # ── DL Workflow ──
        self._add_group(inner, "DL \u6aa2\u6e2c\u5de5\u4f5c\u6d41", [
            ("\U0001F3AF \u8a13\u7df4\u65b0\u6a21\u578b", "dl_train", "#1565c0"),
            ("\U0001F4E6 \u8f09\u5165 Checkpoint", "dl_load", "#37474f"),
            ("\U0001F50D \u55ae\u5f35\u6aa2\u6e2c (F5)", "dl_inspect", "#2e7d32"),
            ("\U0001F4CB \u6279\u6b21\u6aa2\u6e2c", "dl_batch", "#4e342e"),
            ("\U0001F4CA \u8a08\u7b97\u95be\u503c", "dl_compute_threshold", "#37474f"),
        ])

        # ── VM Workflow ──
        self._add_group(inner, "VM \u7d71\u8a08\u6a21\u578b\u5de5\u4f5c\u6d41", [
            ("\U0001F4CA \u8a13\u7df4 VM \u6a21\u578b", "vm_train", "#1565c0"),
            ("\U0001F4C1 \u8f09\u5165 VM \u6a21\u578b", "vm_load", "#37474f"),
            ("\U0001F50E VM \u55ae\u5f35\u6aa2\u6e2c", "vm_inspect", "#2e7d32"),
            ("\U0001F4C2 VM \u6279\u6b21\u6aa2\u6e2c", "vm_batch", "#4e342e"),
            ("\U0001F4C8 \u95be\u503c\u8996\u89ba\u5316", "vm_threshold_viz", "#37474f"),
        ])

        # ── Image Processing ──
        self._add_group(inner, "\u5f71\u50cf\u8655\u7406\u5feb\u6377", [
            ("\u25A7 \u95be\u503c\u5206\u5272", "threshold", "#37474f"),
            ("\u25A9 Blob \u5206\u6790", "blob", "#37474f"),
            ("\U0001F50D \u5f62\u72c0\u5339\u914d", "shape_match", "#37474f"),
            ("\U0001F4CF \u91cf\u6e2c\u5de5\u5177", "metrology", "#37474f"),
            ("\U0001F4D0 ROI \u7ba1\u7406", "roi", "#37474f"),
        ])

        # ── Advanced Tools ──
        self._add_group(inner, "\u9032\u968e\u5de5\u5177", [
            ("\U0001F9E0 PatchCore / ONNX", "patchcore", "#4a148c"),
            ("\U0001F300 FFT / \u8272\u5f69 / OCR", "inspection_tools", "#37474f"),
            ("\u2699 \u5de5\u7a0b\u5de5\u5177 (SPC/\u6a19\u5b9a)", "engineering_tools", "#37474f"),
            ("\U0001F4F7 \u76f8\u6a5f / \u6d41\u7a0b / \u5831\u8868", "mvp_tools", "#37474f"),
            ("\U0001F527 \u81ea\u52d5\u95be\u503c\u6821\u6e96", "auto_tune", "#37474f"),
        ])

        # ── Project ──
        self._add_group(inner, "\u5c08\u6848\u7ba1\u7406", [
            ("\U0001F4C1 \u958b\u555f\u5c08\u6848", "project_open", "#37474f"),
            ("\U0001F4BE \u5132\u5b58\u5c08\u6848", "project_save", "#37474f"),
            ("\U0001F4C4 \u532F\u51FA PDF \u5831\u544A", "export_report", "#37474f"),
        ])

    def _add_group(self, parent, title: str, buttons: list) -> None:
        """Add a labeled group of action buttons."""
        # Group header
        header = tk.Label(
            parent, text=title, bg="#2b2b2b", fg="#0078d4",
            font=(_FONT_FAMILY, 9, "bold"), anchor=tk.W,
        )
        header.pack(fill=tk.X, padx=8, pady=(8, 2))

        sep = tk.Frame(parent, bg="#444444", height=1)
        sep.pack(fill=tk.X, padx=8, pady=(0, 4))

        for label_text, action, color in buttons:
            cb = self._cb.get(action, lambda: None)
            btn = tk.Button(
                parent,
                text=label_text,
                command=cb,
                bg=color,
                fg="#e0e0e0",
                activebackground="#5c6bc0",
                activeforeground="#ffffff",
                font=(_FONT_FAMILY, 9),
                relief=tk.FLAT,
                anchor=tk.W,
                padx=12,
                pady=4,
                cursor="hand2",
            )
            btn.pack(fill=tk.X, padx=8, pady=1)

    def update_callbacks(self, callbacks: Dict[str, Callable]) -> None:
        """Update callbacks after initialization (for lazy binding)."""
        self._cb.update(callbacks)
