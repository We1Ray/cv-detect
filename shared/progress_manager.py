"""Progress bar and busy-cursor manager for Tkinter status bars."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class ProgressManager:
    """Manages a ttk.Progressbar embedded in a status-bar frame and the
    root window's busy cursor.

    Usage::

        pm = ProgressManager(root, status_frame)
        pm.start_indeterminate()
        # ... long operation ...
        pm.stop()
    """

    def __init__(self, root: tk.Tk, parent_frame: ttk.Frame) -> None:
        self._root = root
        self._bar = ttk.Progressbar(
            parent_frame, mode="indeterminate", length=120,
        )
        self._bar.pack(side=tk.RIGHT, padx=(4, 8))
        self._bar.pack_forget()  # hidden until needed
        self._running = False

    def start_indeterminate(self) -> None:
        """Show progress bar in indeterminate mode and set busy cursor."""
        if self._running:
            return
        self._running = True
        self._bar.pack(side=tk.RIGHT, padx=(4, 8))
        self._bar.start(15)
        self._root.configure(cursor="wait")

    def stop(self) -> None:
        """Hide progress bar and restore normal cursor."""
        if not self._running:
            return
        self._running = False
        self._bar.stop()
        self._bar.pack_forget()
        self._root.configure(cursor="")

    def set_determinate(self, value: float, maximum: float = 100.0) -> None:
        """Switch to determinate mode and set progress value."""
        if not self._running:
            self._running = True
            self._bar.pack(side=tk.RIGHT, padx=(4, 8))
            self._root.configure(cursor="wait")
        self._bar.configure(mode="determinate", maximum=maximum, value=value)
