"""Timestamped operation history panel (Treeview widget)."""

from __future__ import annotations

import tkinter as tk
from datetime import datetime
from tkinter import ttk


class HistoryPanel(ttk.LabelFrame):
    """A compact timestamped operation history displayed in a Treeview."""

    def __init__(self, parent, **kwargs) -> None:
        super().__init__(parent, text="操作歷史", **kwargs)

        # Treeview
        cols = ("time", "operation", "detail")
        self._tree = ttk.Treeview(
            self, columns=cols, show="headings", height=6,
        )
        self._tree.heading("time", text="時間")
        self._tree.heading("operation", text="操作")
        self._tree.heading("detail", text="細節")
        self._tree.column("time", width=70, minwidth=60, stretch=False)
        self._tree.column("operation", width=120, minwidth=80)
        self._tree.column("detail", width=160, minwidth=80)

        scrollbar = ttk.Scrollbar(
            self, orient=tk.VERTICAL, command=self._tree.yview,
        )
        self._tree.configure(yscrollcommand=scrollbar.set)

        self._tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._counter = 0

    def add_entry(self, op_name: str, detail: str = "") -> None:
        """Append a new timestamped entry to the history."""
        self._counter += 1
        ts = datetime.now().strftime("%H:%M:%S")
        item = self._tree.insert("", tk.END, values=(ts, op_name, detail))
        self._tree.see(item)

    def clear(self) -> None:
        """Remove all entries."""
        for item in self._tree.get_children():
            self._tree.delete(item)
        self._counter = 0
