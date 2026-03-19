"""
gui/toolbar.py - Industrial Vision 風格工具列

提供快速操作按鈕：
- 開啟 / 儲存影像
- 復原 / 重做
- 適配視窗 / 放大 / 縮小 / 100%
- 訓練模型 / 載入模型
- 單張檢測 / 批次檢測
- 格線 / 十字游標 切換
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Dict, Optional


class Toolbar(ttk.Frame):
    """Industrial Vision 風格工具列。"""

    def __init__(
        self,
        master: tk.Widget,
        commands: Optional[Dict[str, Callable]] = None,
        **kwargs,
    ) -> None:
        super().__init__(master, **kwargs)

        self._commands = commands or {}
        self._toggle_states: Dict[str, tk.BooleanVar] = {}

        self._build_ui()

    def _build_ui(self) -> None:
        """建構工具列按鈕。"""
        btn_style = {"width": 4, "padding": 2}

        # ── 檔案操作 ──
        self._add_button("open_image", "\u25a3 開啟", "開啟影像", **btn_style)
        self._add_button("save_image", "\u25a3 儲存", "儲存影像", **btn_style)

        self._add_separator()

        # ── 復原 / 重做 ──
        self._add_button("undo", "\u21a9", "復原 (Ctrl+Z)", width=3, padding=2)
        self._add_button("redo", "\u21aa", "重做 (Ctrl+Y)", width=3, padding=2)

        self._add_separator()

        # ── 縮放 ──
        self._add_button("fit_window", "\u229e", "適配視窗 (Space)", width=3, padding=2)
        self._add_button("zoom_in", "+", "放大 (+)", width=3, padding=2)
        self._add_button("zoom_out", "-", "縮小 (-)", width=3, padding=2)
        self._add_button("zoom_100", "1:1", "100% 檢視", width=3, padding=2)

        self._add_separator()

        # ── 模型操作 ──
        self._add_button("train_model", "\u25c9 訓練", "訓練模型", **btn_style)
        self._add_button("load_model", "\u25a8 載入", "載入模型", **btn_style)

        self._add_separator()

        # ── 檢測操作 ──
        self._add_button("inspect_single", "\u25ce 檢測", "單張檢測 (F5)", **btn_style)
        self._add_button("inspect_batch", "\u25a4 批次", "批次檢測", **btn_style)

        self._add_separator()

        # ── 影像處理 工具 ──
        self._add_separator()

        self._pixel_inspector_var = tk.BooleanVar(value=False)
        self._toggle_states["pixel_inspector"] = self._pixel_inspector_var
        pi_btn = ttk.Checkbutton(
            self, text="\u25c8",
            variable=self._pixel_inspector_var,
            command=lambda: self._fire("toggle_pixel_inspector"),
            style="Toolbutton",
            width=3,
        )
        pi_btn.pack(side=tk.LEFT, padx=1)
        self._create_tooltip(pi_btn, "像素檢查器 (Ctrl+I)")

        self._add_button("threshold", "\u25a7 閾值", "閾值分割 (Ctrl+T)", width=5, padding=2)
        self._add_button("blob_analysis", "\u25a9 Blob", "Blob 分析", width=5, padding=2)

        self._script_editor_var = tk.BooleanVar(value=False)
        self._toggle_states["script_editor"] = self._script_editor_var
        se_btn = ttk.Checkbutton(
            self, text="\u2630",
            variable=self._script_editor_var,
            command=lambda: self._fire("toggle_script_editor"),
            style="Toolbutton",
            width=3,
        )
        se_btn.pack(side=tk.LEFT, padx=1)
        self._create_tooltip(se_btn, "腳本編輯器 (F8)")

        self._add_separator()

        # ── 顯示切換 ──
        self._grid_var = tk.BooleanVar(value=False)
        self._toggle_states["grid"] = self._grid_var
        grid_btn = ttk.Checkbutton(
            self, text="\u25a6",
            variable=self._grid_var,
            command=lambda: self._fire("toggle_grid"),
            style="Toolbutton",
            width=3,
        )
        grid_btn.pack(side=tk.LEFT, padx=1)
        self._create_tooltip(grid_btn, "格線")

        self._crosshair_var = tk.BooleanVar(value=False)
        self._toggle_states["crosshair"] = self._crosshair_var
        cross_btn = ttk.Checkbutton(
            self, text="+",
            variable=self._crosshair_var,
            command=lambda: self._fire("toggle_crosshair"),
            style="Toolbutton",
            width=3,
        )
        cross_btn.pack(side=tk.LEFT, padx=1)
        self._create_tooltip(cross_btn, "十字游標")

    # ================================================================== #
    #  公開 API                                                           #
    # ================================================================== #

    def set_command(self, name: str, callback: Callable) -> None:
        """設定工具列按鈕的回呼函式。"""
        self._commands[name] = callback

    def get_toggle_state(self, name: str) -> bool:
        """取得切換按鈕狀態。"""
        var = self._toggle_states.get(name)
        return var.get() if var else False

    # ================================================================== #
    #  內部方法                                                            #
    # ================================================================== #

    def _add_button(self, name: str, text: str, tooltip: str = "", **kwargs) -> None:
        """新增工具列按鈕。"""
        btn = ttk.Button(
            self, text=text,
            command=lambda: self._fire(name),
            **kwargs,
        )
        btn.pack(side=tk.LEFT, padx=1)
        if tooltip:
            self._create_tooltip(btn, tooltip)

    def _add_separator(self) -> None:
        """新增分隔線。"""
        sep = ttk.Separator(self, orient=tk.VERTICAL)
        sep.pack(side=tk.LEFT, fill=tk.Y, padx=4, pady=2)

    def _fire(self, name: str) -> None:
        """觸發指定回呼。"""
        callback = self._commands.get(name)
        if callback:
            callback()

    def _create_tooltip(self, widget: tk.Widget, text: str) -> None:
        """為 widget 建立簡易 tooltip。"""
        tooltip = None

        def show(event):
            nonlocal tooltip
            tooltip = tk.Toplevel(widget)
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root + 10}+{event.y_root + 10}")
            label = tk.Label(
                tooltip, text=text,
                background="#ffffe0", foreground="#000000",
                relief=tk.SOLID, borderwidth=1,
                font=("", 8),
                padx=4, pady=2,
            )
            label.pack()

        def hide(event):
            nonlocal tooltip
            if tooltip:
                tooltip.destroy()
                tooltip = None

        widget.bind("<Enter>", show)
        widget.bind("<Leave>", hide)
