"""
gui/pipeline_panel.py - 管線步驟面板（左側）

顯示影像處理管線的步驟清單與縮圖：
- 步驟編號、名稱、小縮圖 (64x64)
- 點擊步驟切換主檢視器顯示
- 右鍵選單：匯出影像、刪除步驟、全尺寸檢視
"""

from __future__ import annotations

import logging
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk

logger = logging.getLogger(__name__)

THUMB_SIZE = 64


class PipelineStep:
    """管線中的一個處理步驟。"""

    def __init__(
        self,
        name: str,
        image: np.ndarray,
        step_type: str = "process",
        region=None,
    ) -> None:
        self.name = name
        self.image = image
        self.step_type = step_type
        self.region = region  # Optional Region object from core.region
        self.thumbnail: Optional[ImageTk.PhotoImage] = None
        self._create_thumbnail()

    def _create_thumbnail(self) -> None:
        """建立 64x64 縮圖。"""
        img = self.image
        if img.dtype != np.uint8:
            min_v = img.min()
            max_v = img.max()
            if max_v - min_v > 0:
                img = ((img - min_v) / (max_v - min_v) * 255).astype(np.uint8)
            else:
                img = np.zeros_like(img, dtype=np.uint8)

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

        h, w = img.shape[:2]
        scale = THUMB_SIZE / max(h, w)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        pil_img = Image.fromarray(img).resize((new_w, new_h), Image.LANCZOS)

        # 貼到正方形背景上
        bg = Image.new("RGB", (THUMB_SIZE, THUMB_SIZE), (30, 30, 30))
        offset_x = (THUMB_SIZE - new_w) // 2
        offset_y = (THUMB_SIZE - new_h) // 2
        bg.paste(pil_img, (offset_x, offset_y))

        self.thumbnail = ImageTk.PhotoImage(bg)


class PipelinePanel(ttk.Frame):
    """管線步驟面板。"""

    def __init__(
        self,
        master: tk.Widget,
        on_step_selected: Optional[Callable[[int], None]] = None,
        **kwargs,
    ) -> None:
        super().__init__(master, **kwargs)

        self._steps: List[PipelineStep] = []
        self._selected_index: int = -1
        self._on_step_selected = on_step_selected
        self._item_widgets: List[tk.Frame] = []

        self._build_ui()

    def _build_ui(self) -> None:
        """建構面板 UI。"""
        # 標題
        header = ttk.Frame(self)
        header.pack(fill=tk.X, padx=2, pady=(2, 0))
        ttk.Label(
            header, text="  處理管線", font=("", 10, "bold"),
        ).pack(side=tk.LEFT)

        self._count_label = ttk.Label(header, text="0 步驟")
        self._count_label.pack(side=tk.RIGHT, padx=4)

        # 可捲動區域
        container = ttk.Frame(self)
        container.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        self._canvas = tk.Canvas(
            container, bg="#2b2b2b", highlightthickness=0, width=190,
        )
        self._scrollbar = ttk.Scrollbar(
            container, orient=tk.VERTICAL, command=self._canvas.yview,
        )
        self._inner_frame = ttk.Frame(self._canvas)

        self._inner_frame.bind(
            "<Configure>",
            lambda e: self._canvas.configure(scrollregion=self._canvas.bbox("all")),
        )
        self._canvas.create_window((0, 0), window=self._inner_frame, anchor=tk.NW)
        self._canvas.configure(yscrollcommand=self._scrollbar.set)

        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 滾輪支援
        self._canvas.bind("<MouseWheel>", self._on_mousewheel)
        self._canvas.bind("<Button-4>", lambda e: self._canvas.yview_scroll(-3, "units"))
        self._canvas.bind("<Button-5>", lambda e: self._canvas.yview_scroll(3, "units"))

        # 底部按鈕
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=2, pady=2)
        ttk.Button(
            btn_frame, text="清除全部", command=self.clear_all, width=10,
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(
            btn_frame, text="匯出", command=self._export_current, width=8,
        ).pack(side=tk.RIGHT, padx=2)

    def _on_mousewheel(self, event: tk.Event) -> None:
        self._canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    # ================================================================== #
    #  公開 API                                                           #
    # ================================================================== #

    def add_step(self, name: str, image: np.ndarray, step_type: str = "process", region=None) -> int:
        """新增處理步驟。

        Args:
            name: 步驟名稱。
            image: 步驟影像。
            step_type: 步驟類型。
            region: 可選的 Region 物件。

        Returns:
            步驟索引。
        """
        step = PipelineStep(name, image, step_type, region=region)
        self._steps.append(step)
        idx = len(self._steps) - 1

        self._create_step_widget(idx, step)
        self._count_label.configure(text=f"{len(self._steps)} 步驟")

        # 自動選取最新步驟
        self.select_step(idx)

        # 自動捲動到底部
        self._canvas.update_idletasks()
        self._canvas.yview_moveto(1.0)

        return idx

    def select_step(self, index: int) -> None:
        """選取指定步驟。"""
        if index < 0 or index >= len(self._steps):
            return

        self._selected_index = index

        # 更新所有項目的視覺狀態
        for i, widget in enumerate(self._item_widgets):
            if i == index:
                widget.configure(style="Selected.TFrame")
                for child in widget.winfo_children():
                    if isinstance(child, ttk.Label):
                        child.configure(style="Selected.TLabel")
            else:
                widget.configure(style="Pipeline.TFrame")
                for child in widget.winfo_children():
                    if isinstance(child, ttk.Label):
                        child.configure(style="Pipeline.TLabel")

        if self._on_step_selected:
            self._on_step_selected(index)

    def get_step(self, index: int) -> Optional[PipelineStep]:
        """取得指定步驟。"""
        if 0 <= index < len(self._steps):
            return self._steps[index]
        return None

    def get_current_step(self) -> Optional[PipelineStep]:
        """取得目前選取的步驟。"""
        return self.get_step(self._selected_index)

    def get_current_index(self) -> int:
        """取得目前選取的步驟索引。"""
        return self._selected_index

    def get_step_count(self) -> int:
        """取得步驟總數。"""
        return len(self._steps)

    def delete_step(self, index: int) -> None:
        """刪除指定步驟。"""
        if index < 0 or index >= len(self._steps):
            return

        self._steps.pop(index)

        # 重建所有 widget
        self._rebuild_widgets()

        # 調整選取索引
        if len(self._steps) == 0:
            self._selected_index = -1
        elif index >= len(self._steps):
            self.select_step(len(self._steps) - 1)
        else:
            self.select_step(index)

        self._count_label.configure(text=f"{len(self._steps)} 步驟")

    def clear_all(self) -> None:
        """清除所有步驟。"""
        self._steps.clear()
        self._selected_index = -1
        self._rebuild_widgets()
        self._count_label.configure(text="0 步驟")

    def go_previous(self) -> bool:
        """選取上一步驟。回傳是否成功。"""
        if self._selected_index > 0:
            self.select_step(self._selected_index - 1)
            return True
        return False

    def go_next(self) -> bool:
        """選取下一步驟。回傳是否成功。"""
        if self._selected_index < len(self._steps) - 1:
            self.select_step(self._selected_index + 1)
            return True
        return False

    # ================================================================== #
    #  內部方法                                                            #
    # ================================================================== #

    def _create_step_widget(self, index: int, step: PipelineStep) -> None:
        """建立單一步驟的 widget。"""
        item_frame = ttk.Frame(self._inner_frame, style="Pipeline.TFrame")
        item_frame.pack(fill=tk.X, padx=2, pady=1)
        item_frame.bind("<Button-1>", lambda e, i=index: self.select_step(i))
        item_frame.bind("<Button-3>", lambda e, i=index: self._show_context_menu(e, i))

        # 縮圖
        if step.thumbnail is not None:
            thumb_label = ttk.Label(
                item_frame, image=step.thumbnail, style="Pipeline.TLabel",
            )
            thumb_label.image = step.thumbnail  # type: ignore[attr-defined]
            thumb_label.pack(side=tk.LEFT, padx=4, pady=2)
            thumb_label.bind("<Button-1>", lambda e, i=index: self.select_step(i))
            thumb_label.bind("<Button-3>", lambda e, i=index: self._show_context_menu(e, i))

        # 文字資訊
        info_frame = ttk.Frame(item_frame, style="Pipeline.TFrame")
        info_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        info_frame.bind("<Button-1>", lambda e, i=index: self.select_step(i))
        info_frame.bind("<Button-3>", lambda e, i=index: self._show_context_menu(e, i))

        step_label = ttk.Label(
            info_frame,
            text=f"{index + 1}. {step.name}",
            font=("", 9),
            style="Pipeline.TLabel",
        )
        step_label.pack(anchor=tk.W)
        step_label.bind("<Button-1>", lambda e, i=index: self.select_step(i))
        step_label.bind("<Button-3>", lambda e, i=index: self._show_context_menu(e, i))

        # 影像尺寸資訊
        h, w = step.image.shape[:2]
        ch = step.image.shape[2] if step.image.ndim == 3 else 1
        size_text = f"{w}x{h} ch={ch}"
        if step.region is not None:
            size_text += f"  [R:{step.region.num_regions}]"
        size_label = ttk.Label(
            info_frame,
            text=size_text,
            font=("", 7),
            foreground="#888888",
            style="Pipeline.TLabel",
        )
        size_label.pack(anchor=tk.W)
        size_label.bind("<Button-1>", lambda e, i=index: self.select_step(i))

        self._item_widgets.append(item_frame)

    def _rebuild_widgets(self) -> None:
        """重建所有步驟 widget。"""
        for w in self._item_widgets:
            w.destroy()
        self._item_widgets.clear()

        for i, step in enumerate(self._steps):
            self._create_step_widget(i, step)

    def _show_context_menu(self, event: tk.Event, index: int) -> None:
        """顯示右鍵選單。"""
        self.select_step(index)
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="匯出影像...", command=lambda: self._export_step(index))
        menu.add_command(label="刪除步驟", command=lambda: self.delete_step(index))
        menu.add_separator()
        menu.add_command(label="全尺寸檢視", command=lambda: self._view_fullsize(index))
        menu.tk_popup(event.x_root, event.y_root)

    def _export_step(self, index: int) -> None:
        """匯出指定步驟的影像。"""
        if index < 0 or index >= len(self._steps):
            return

        step = self._steps[index]
        path = filedialog.asksaveasfilename(
            title="匯出影像",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("所有檔案", "*.*")],
            initialfile=f"{step.name}.png",
        )
        if not path:
            return

        try:
            img = step.image
            if img.dtype != np.uint8:
                min_v = img.min()
                max_v = img.max()
                if max_v - min_v > 0:
                    img = ((img - min_v) / (max_v - min_v) * 255).astype(np.uint8)
                else:
                    img = np.zeros_like(img, dtype=np.uint8)
            cv2.imwrite(path, img)
            messagebox.showinfo("匯出成功", f"影像已儲存至:\n{path}")
        except Exception as exc:
            messagebox.showerror("匯出失敗", str(exc))

    def _export_current(self) -> None:
        """匯出目前步驟的影像。"""
        if self._selected_index >= 0:
            self._export_step(self._selected_index)

    def _view_fullsize(self, index: int) -> None:
        """以全尺寸視窗顯示影像。"""
        if index < 0 or index >= len(self._steps):
            return

        step = self._steps[index]
        top = tk.Toplevel(self)
        top.title(f"全尺寸檢視 - {step.name}")
        top.geometry("800x600")

        from gui.image_viewer import ImageViewer  # noqa: E402

        viewer = ImageViewer(top)
        viewer.pack(fill=tk.BOTH, expand=True)
        viewer.set_image(step.image)
