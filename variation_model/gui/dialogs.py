"""
gui/dialogs.py - 對話框集合

提供：
- TrainingDialog: 模型訓練對話框
- BatchInspectDialog: 批次檢測對話框
- ModelInfoDialog: 模型資訊對話框
- HistogramDialog: 直方圖對話框
- SettingsDialog: 組態設定對話框
"""

from __future__ import annotations

import logging
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk

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


# ====================================================================== #
#  訓練對話框                                                              #
# ====================================================================== #

class TrainingDialog(tk.Toplevel):
    """模型訓練對話框。"""

    def __init__(
        self,
        master: tk.Widget,
        config: Any,
        on_complete: Optional[Callable] = None,
    ) -> None:
        super().__init__(master)
        self.title("模型訓練")
        self.geometry("700x550")
        self.resizable(True, True)
        self.transient(master)
        self.grab_set()

        self.config = config
        self._on_complete = on_complete
        self._training_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._progress_queue: queue.Queue = queue.Queue()
        self._model = None

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        """建構 UI。"""
        # ── 目錄設定 ──
        dir_frame = ttk.LabelFrame(self, text="訓練設定", padding=8)
        dir_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(dir_frame, text="訓練影像目錄:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self._train_dir_var = tk.StringVar(value=str(self.config.train_image_dir))
        ttk.Entry(dir_frame, textvariable=self._train_dir_var, width=50).grid(
            row=0, column=1, padx=5, pady=2, sticky=tk.EW,
        )
        ttk.Button(dir_frame, text="瀏覽...", command=self._browse_dir).grid(
            row=0, column=2, padx=5, pady=2,
        )

        ttk.Label(dir_frame, text="參考影像 (選填):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self._ref_var = tk.StringVar(
            value=str(self.config.reference_image) if self.config.reference_image else "",
        )
        ttk.Entry(dir_frame, textvariable=self._ref_var, width=50).grid(
            row=1, column=1, padx=5, pady=2, sticky=tk.EW,
        )
        ttk.Button(dir_frame, text="瀏覽...", command=self._browse_ref).grid(
            row=1, column=2, padx=5, pady=2,
        )

        dir_frame.columnconfigure(1, weight=1)

        # ── 按鈕 ──
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=10, pady=3)

        self._start_btn = ttk.Button(btn_frame, text="開始訓練", command=self._start)
        self._start_btn.pack(side=tk.LEFT, padx=3)
        self._stop_btn = ttk.Button(btn_frame, text="停止", command=self._stop, state=tk.DISABLED)
        self._stop_btn.pack(side=tk.LEFT, padx=3)

        self._progress_var = tk.DoubleVar(value=0)
        self._progress_bar = ttk.Progressbar(
            btn_frame, variable=self._progress_var, maximum=100, length=300,
        )
        self._progress_bar.pack(side=tk.RIGHT, padx=5)

        self._status_var = tk.StringVar(value="就緒")
        ttk.Label(btn_frame, textvariable=self._status_var).pack(side=tk.RIGHT, padx=5)

        # ── 預覽 ──
        preview_frame = ttk.LabelFrame(self, text="模型預覽", padding=5)
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=3)

        left = ttk.LabelFrame(preview_frame, text="均值影像")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        self._mean_canvas = tk.Canvas(left, bg="#1e1e1e", highlightthickness=0)
        self._mean_canvas.pack(fill=tk.BOTH, expand=True)
        self._mean_photo = None

        right = ttk.LabelFrame(preview_frame, text="標準差影像")
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=2)
        self._std_canvas = tk.Canvas(right, bg="#1e1e1e", highlightthickness=0)
        self._std_canvas.pack(fill=tk.BOTH, expand=True)
        self._std_photo = None

        # ── 日誌 ──
        log_frame = ttk.LabelFrame(self, text="訓練日誌", padding=5)
        log_frame.pack(fill=tk.X, padx=10, pady=(3, 8))

        self._log_text = tk.Text(
            log_frame, height=6, wrap=tk.WORD, state=tk.DISABLED,
            bg="#1e1e1e", fg="#cccccc", font=(_MONO_FAMILY, 9),
        )
        scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self._log_text.yview)
        self._log_text.configure(yscrollcommand=scroll.set)
        self._log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

    # ── 目錄瀏覽 ──

    def _browse_dir(self) -> None:
        d = filedialog.askdirectory(title="選擇訓練影像目錄")
        if d:
            self._train_dir_var.set(d)

    def _browse_ref(self) -> None:
        path = filedialog.askopenfilename(
            title="選擇參考影像",
            filetypes=[("影像檔案", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")],
        )
        if path:
            self._ref_var.set(path)

    # ── 訓練控制 ──

    def _log(self, msg: str) -> None:
        self._log_text.configure(state=tk.NORMAL)
        self._log_text.insert(tk.END, msg + "\n")
        self._log_text.see(tk.END)
        self._log_text.configure(state=tk.DISABLED)

    def _start(self) -> None:
        train_dir = self._train_dir_var.get().strip()
        if not train_dir or not Path(train_dir).exists():
            messagebox.showerror("錯誤", "請選擇有效的訓練影像目錄。")
            return

        from config import Config

        # 更新組態
        updated = self.config.update(train_image_dir=train_dir)
        ref = self._ref_var.get().strip()
        if ref and Path(ref).exists():
            updated = updated.update(reference_image=ref)
        else:
            updated = updated.update(reference_image="")

        self._stop_event.clear()
        self._start_btn.configure(state=tk.DISABLED)
        self._stop_btn.configure(state=tk.NORMAL)
        self._log("訓練開始...")
        self._status_var.set("訓練中...")

        def worker():
            try:
                from pipeline.trainer import TrainingPipeline

                pipeline = TrainingPipeline(updated)

                def progress_cb(current, total, msg):
                    if self._stop_event.is_set():
                        raise InterruptedError("使用者中止訓練")
                    self._progress_queue.put(("progress", current, total, msg))

                model = pipeline.run(
                    progress_callback=progress_cb,
                    image_dir=Path(train_dir),
                )
                self._progress_queue.put(("done", model))
            except InterruptedError:
                self._progress_queue.put(("stopped",))
            except Exception as exc:
                self._progress_queue.put(("error", str(exc)))

        self._training_thread = threading.Thread(target=worker, daemon=True)
        self._training_thread.start()
        self.after(100, self._poll)

    def _stop(self) -> None:
        self._stop_event.set()
        self._log("停止請求已發送...")

    def _poll(self) -> None:
        try:
            while True:
                item = self._progress_queue.get_nowait()
                kind = item[0]

                if kind == "progress":
                    _, current, total, msg = item
                    if total > 0:
                        self._progress_var.set((current / total) * 100)
                    self._log(msg)
                    self._status_var.set(msg)

                elif kind == "done":
                    model = item[1]
                    self._model = model
                    self._log(f"訓練完成! 共 {model.count} 張影像。")
                    self._status_var.set("訓練完成")
                    self._progress_var.set(100)
                    self._start_btn.configure(state=tk.NORMAL)
                    self._stop_btn.configure(state=tk.DISABLED)
                    self._show_preview(model)
                    if self._on_complete:
                        self._on_complete(model)
                    return

                elif kind == "stopped":
                    self._log("訓練已中止。")
                    self._status_var.set("已中止")
                    self._start_btn.configure(state=tk.NORMAL)
                    self._stop_btn.configure(state=tk.DISABLED)
                    return

                elif kind == "error":
                    self._log(f"錯誤: {item[1]}")
                    self._status_var.set("訓練失敗")
                    self._start_btn.configure(state=tk.NORMAL)
                    self._stop_btn.configure(state=tk.DISABLED)
                    messagebox.showerror("訓練錯誤", item[1])
                    return
        except queue.Empty:
            pass

        if self._training_thread and self._training_thread.is_alive():
            self.after(100, self._poll)
        else:
            self._start_btn.configure(state=tk.NORMAL)
            self._stop_btn.configure(state=tk.DISABLED)

    def _show_preview(self, model) -> None:
        """顯示均值與標準差預覽。"""
        imgs = model.get_model_images()

        def show_on_canvas(canvas, img_array, attr_name):
            if img_array is None:
                return
            img = img_array.copy()
            min_v, max_v = img.min(), img.max()
            if max_v - min_v > 0:
                img = ((img - min_v) / (max_v - min_v) * 255).astype(np.uint8)
            else:
                img = np.zeros_like(img, dtype=np.uint8)

            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            canvas.update_idletasks()
            cw = max(canvas.winfo_width(), 100)
            ch = max(canvas.winfo_height(), 100)

            h, w = img.shape[:2]
            scale = min(cw / w, ch / h) * 0.9
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))

            pil = Image.fromarray(img).resize((new_w, new_h), Image.LANCZOS)
            photo = ImageTk.PhotoImage(pil)
            setattr(self, attr_name, photo)
            canvas.delete("all")
            canvas.create_image(cw // 2, ch // 2, image=photo, anchor=tk.CENTER)

        show_on_canvas(self._mean_canvas, imgs.get("mean"), "_mean_photo")
        show_on_canvas(self._std_canvas, imgs.get("std"), "_std_photo")

    def _on_close(self) -> None:
        if self._training_thread and self._training_thread.is_alive():
            self._stop_event.set()
        self.destroy()

    def get_model(self):
        return self._model


# ====================================================================== #
#  批次檢測對話框                                                          #
# ====================================================================== #

class BatchInspectDialog(tk.Toplevel):
    """批次檢測對話框。"""

    def __init__(
        self,
        master: tk.Widget,
        model: Any,
        config: Any,
        on_view_result: Optional[Callable] = None,
    ) -> None:
        super().__init__(master)
        self.title("批次瑕疵檢測")
        self.geometry("800x550")
        self.resizable(True, True)
        self.transient(master)

        self.model = model
        self.config = config
        self._on_view_result = on_view_result
        self._inspect_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._progress_queue: queue.Queue = queue.Queue()
        self._results: List = []

        self._build_ui()

    def _build_ui(self) -> None:
        # ── 頂部 ──
        top = ttk.Frame(self, padding=8)
        top.pack(fill=tk.X)

        ttk.Label(top, text="檢測目錄:").pack(side=tk.LEFT)
        self._dir_var = tk.StringVar(value=str(self.config.test_image_dir))
        ttk.Entry(top, textvariable=self._dir_var, width=50).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(top, text="瀏覽...", command=self._browse).pack(side=tk.LEFT, padx=5)

        # 按鈕列
        btn = ttk.Frame(self, padding=(8, 0, 8, 5))
        btn.pack(fill=tk.X)

        self._start_btn = ttk.Button(btn, text="開始檢測", command=self._start)
        self._start_btn.pack(side=tk.LEFT, padx=3)
        self._stop_btn = ttk.Button(btn, text="停止", command=self._stop, state=tk.DISABLED)
        self._stop_btn.pack(side=tk.LEFT, padx=3)
        ttk.Button(btn, text="匯出報告", command=self._export_report).pack(side=tk.LEFT, padx=3)

        self._progress_var = tk.DoubleVar(value=0)
        ttk.Progressbar(
            btn, variable=self._progress_var, maximum=100, length=250,
        ).pack(side=tk.RIGHT, padx=5)

        self._status_var = tk.StringVar(value="就緒")
        ttk.Label(btn, textvariable=self._status_var).pack(side=tk.RIGHT, padx=5)

        # ── 結果表格 ──
        table_frame = ttk.Frame(self, padding=8)
        table_frame.pack(fill=tk.BOTH, expand=True)

        cols = ("filename", "result", "score", "defects")
        self._tree = ttk.Treeview(table_frame, columns=cols, show="headings", height=18)
        self._tree.heading("filename", text="檔名")
        self._tree.heading("result", text="結果")
        self._tree.heading("score", text="分數 (%)")
        self._tree.heading("defects", text="瑕疵數")

        self._tree.column("filename", width=300)
        self._tree.column("result", width=80, anchor=tk.CENTER)
        self._tree.column("score", width=100, anchor=tk.CENTER)
        self._tree.column("defects", width=80, anchor=tk.CENTER)

        scroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self._tree.yview)
        self._tree.configure(yscrollcommand=scroll.set)
        self._tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self._tree.bind("<Double-1>", self._on_double_click)

        # ── 摘要 ──
        summary = ttk.Frame(self, padding=8)
        summary.pack(fill=tk.X)
        self._summary_var = tk.StringVar(value="")
        ttk.Label(summary, textvariable=self._summary_var, font=("", 9)).pack(anchor=tk.W)

    def _browse(self) -> None:
        d = filedialog.askdirectory(title="選擇測試影像目錄")
        if d:
            self._dir_var.set(d)

    def _start(self) -> None:
        test_dir = self._dir_var.get().strip()
        if not test_dir or not Path(test_dir).exists():
            messagebox.showerror("錯誤", "請選擇有效的測試影像目錄。")
            return

        self._stop_event.clear()
        self._start_btn.configure(state=tk.DISABLED)
        self._stop_btn.configure(state=tk.NORMAL)

        # 清除舊結果
        for item in self._tree.get_children():
            self._tree.delete(item)
        self._results.clear()

        def worker():
            try:
                from pipeline.inference import InferencePipeline

                pipeline = InferencePipeline(self.model, self.config)

                def progress_cb(current, total, msg):
                    if self._stop_event.is_set():
                        raise InterruptedError("使用者中止")
                    self._progress_queue.put(("progress", current, total, msg))

                results = pipeline.inspect_batch(test_dir, progress_callback=progress_cb)
                self._progress_queue.put(("done", results))
            except InterruptedError:
                self._progress_queue.put(("stopped",))
            except Exception as exc:
                self._progress_queue.put(("error", str(exc)))

        self._inspect_thread = threading.Thread(target=worker, daemon=True)
        self._inspect_thread.start()
        self.after(100, self._poll)

    def _stop(self) -> None:
        self._stop_event.set()

    def _poll(self) -> None:
        try:
            while True:
                item = self._progress_queue.get_nowait()
                kind = item[0]

                if kind == "progress":
                    _, current, total, msg = item
                    if total > 0:
                        self._progress_var.set((current / total) * 100)
                    self._status_var.set(msg)

                elif kind == "done":
                    results = item[1]
                    self._results = results
                    self._populate_results(results)
                    self._start_btn.configure(state=tk.NORMAL)
                    self._stop_btn.configure(state=tk.DISABLED)
                    self._status_var.set(f"檢測完成: {len(results)} 張影像")
                    self._progress_var.set(100)
                    return

                elif kind == "stopped":
                    self._status_var.set("已中止")
                    self._start_btn.configure(state=tk.NORMAL)
                    self._stop_btn.configure(state=tk.DISABLED)
                    return

                elif kind == "error":
                    messagebox.showerror("檢測錯誤", item[1])
                    self._start_btn.configure(state=tk.NORMAL)
                    self._stop_btn.configure(state=tk.DISABLED)
                    return
        except queue.Empty:
            pass

        if self._inspect_thread and self._inspect_thread.is_alive():
            self.after(100, self._poll)

    def _populate_results(self, results) -> None:
        for r, path, processed in results:
            status = "NG" if r.is_defective else "PASS"
            self._tree.insert("", tk.END, values=(
                path.name,
                status,
                f"{r.score:.4f}",
                r.num_defects,
            ))

        total = len(results)
        pass_count = sum(1 for r, _, _ in results if not r.is_defective)
        fail_count = total - pass_count
        self._summary_var.set(
            f"總計: {total}  |  合格: {pass_count}  |  不合格: {fail_count}  |  "
            f"合格率: {(pass_count / total * 100):.1f}%" if total > 0 else "",
        )

    def _on_double_click(self, event) -> None:
        sel = self._tree.selection()
        if not sel:
            return
        idx = self._tree.index(sel[0])
        if 0 <= idx < len(self._results) and self._on_view_result:
            self._on_view_result(self._results[idx])

    def _export_report(self) -> None:
        if not self._results:
            messagebox.showwarning("警告", "沒有可匯出的結果。")
            return

        out_dir = filedialog.askdirectory(title="選擇報告輸出目錄")
        if not out_dir:
            return

        try:
            from pipeline.inference import InferencePipeline

            pipeline = InferencePipeline(self.model, self.config)
            report_dir = pipeline.generate_report(self._results, output_dir=out_dir)
            messagebox.showinfo("匯出成功", f"報告已儲存至:\n{report_dir}")
        except Exception as exc:
            messagebox.showerror("匯出失敗", str(exc))


# ====================================================================== #
#  模型資訊對話框                                                          #
# ====================================================================== #

class ModelInfoDialog(tk.Toplevel):
    """模型資訊對話框。"""

    def __init__(self, master: tk.Widget, model: Any) -> None:
        super().__init__(master)
        self.title("模型資訊")
        self.geometry("400x300")
        self.resizable(False, False)
        self.transient(master)
        self.grab_set()

        self._build_ui(model)

    def _build_ui(self, model) -> None:
        frame = ttk.Frame(self, padding=15)
        frame.pack(fill=tk.BOTH, expand=True)

        infos = [
            ("訓練狀態:", "已訓練" if model.is_trained else "未訓練"),
            ("訓練影像數:", str(model.count)),
        ]

        imgs = model.get_model_images()
        if imgs["mean"] is not None:
            shape = imgs["mean"].shape
            infos.append(("影像尺寸:", f"{shape[1] if len(shape) > 1 else shape[0]} x {shape[0]}"))
            infos.append(("資料型別:", str(imgs["mean"].dtype)))

        if imgs["upper"] is not None:
            infos.append(("閾值狀態:", "已準備"))
            infos.append(("絕對閾值:", str(model._abs_threshold)))
            infos.append(("變異閾值:", f"{model._var_threshold:.1f}"))
        else:
            infos.append(("閾值狀態:", "未準備"))

        if model.reference_image is not None:
            infos.append(("參考影像:", "已設定"))
        else:
            infos.append(("參考影像:", "未設定"))

        for i, (label, value) in enumerate(infos):
            ttk.Label(frame, text=label, font=("", 10, "bold")).grid(
                row=i, column=0, sticky=tk.W, padx=5, pady=3,
            )
            ttk.Label(frame, text=value, font=("", 10)).grid(
                row=i, column=1, sticky=tk.W, padx=10, pady=3,
            )

        ttk.Button(frame, text="關閉", command=self.destroy).grid(
            row=len(infos), column=0, columnspan=2, pady=15,
        )


# ====================================================================== #
#  直方圖對話框                                                            #
# ====================================================================== #

class HistogramDialog(tk.Toplevel):
    """直方圖對話框。"""

    def __init__(self, master: tk.Widget, image: np.ndarray, title_text: str = "直方圖") -> None:
        super().__init__(master)
        self.title(title_text)
        self.geometry("500x350")
        self.resizable(True, True)
        self.transient(master)

        self._build_ui(image)

    def _build_ui(self, image: np.ndarray) -> None:
        self._canvas = tk.Canvas(self, bg="#1e1e1e", highlightthickness=0)
        self._canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self._canvas.bind("<Configure>", lambda e: self._draw(image))
        self.after(100, lambda: self._draw(image))

    def _draw(self, image: np.ndarray) -> None:
        self._canvas.delete("all")
        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            return

        margin = 40

        if image.ndim == 3:
            # 繪製 B, G, R 三通道
            colors_cv = [0, 1, 2]
            colors_tk = ["#4444FF", "#44FF44", "#FF4444"]
            for ci, color in zip(colors_cv, colors_tk):
                hist = cv2.calcHist([image.astype(np.uint8)], [ci], None, [256], [0, 256]).flatten()
                self._draw_hist_line(hist, cw, ch, margin, color)
        else:
            # 灰階
            if image.dtype != np.uint8:
                min_v, max_v = image.min(), image.max()
                if max_v - min_v > 0:
                    img8 = ((image - min_v) / (max_v - min_v) * 255).astype(np.uint8)
                else:
                    img8 = np.zeros_like(image, dtype=np.uint8)
            else:
                img8 = image
            hist = cv2.calcHist([img8], [0], None, [256], [0, 256]).flatten()
            self._draw_hist_line(hist, cw, ch, margin, "#AAAAAA")

        # 軸
        self._canvas.create_line(margin, ch - margin, cw - 10, ch - margin, fill="#666666")
        self._canvas.create_line(margin, 10, margin, ch - margin, fill="#666666")
        self._canvas.create_text(cw // 2, ch - 10, text="像素值", fill="#888888", font=("", 8))
        self._canvas.create_text(15, ch // 2, text="計數", fill="#888888", font=("", 8), angle=90)

    def _draw_hist_line(self, hist, cw, ch, margin, color):
        max_val = hist.max()
        if max_val == 0:
            return
        plot_w = cw - margin - 10
        plot_h = ch - margin - 10

        points = []
        for i, count in enumerate(hist):
            x = margin + (i / 255) * plot_w
            y = ch - margin - (count / max_val) * plot_h
            points.append((x, y))

        for i in range(len(points) - 1):
            self._canvas.create_line(
                points[i][0], points[i][1],
                points[i + 1][0], points[i + 1][1],
                fill=color, width=1,
            )


# ====================================================================== #
#  設定對話框                                                              #
# ====================================================================== #

class SettingsDialog(tk.Toplevel):
    """組態設定對話框。"""

    def __init__(
        self,
        master: tk.Widget,
        config: Any,
        on_apply: Optional[Callable] = None,
    ) -> None:
        super().__init__(master)
        self.title("系統設定")
        self.geometry("550x600")
        self.resizable(True, True)
        self.transient(master)
        self.grab_set()

        self.config = config
        self._on_apply = on_apply

        self._build_ui()

    def _build_ui(self) -> None:
        # 可捲動
        canvas = tk.Canvas(self, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=canvas.yview)
        scrollable = ttk.Frame(canvas)
        scrollable.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable, anchor=tk.NW)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.bind("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

        # ── 路徑 ──
        path_frame = ttk.LabelFrame(scrollable, text="路徑設定", padding=10)
        path_frame.pack(fill=tk.X, padx=10, pady=5)

        self._train_dir_var = tk.StringVar(value=str(self.config.train_image_dir))
        self._test_dir_var = tk.StringVar(value=str(self.config.test_image_dir))
        self._model_dir_var = tk.StringVar(value=str(self.config.model_save_dir))
        self._results_dir_var = tk.StringVar(value=str(self.config.results_dir))

        for row, (label, var) in enumerate([
            ("訓練影像目錄:", self._train_dir_var),
            ("測試影像目錄:", self._test_dir_var),
            ("模型儲存目錄:", self._model_dir_var),
            ("結果輸出目錄:", self._results_dir_var),
        ]):
            ttk.Label(path_frame, text=label).grid(row=row, column=0, sticky=tk.W, pady=2)
            ttk.Entry(path_frame, textvariable=var, width=40).grid(
                row=row, column=1, padx=5, pady=2, sticky=tk.EW,
            )
            ttk.Button(
                path_frame, text="...", width=3,
                command=lambda v=var: self._browse_path(v),
            ).grid(row=row, column=2, padx=2, pady=2)
        path_frame.columnconfigure(1, weight=1)

        # ── 前處理 ──
        preproc = ttk.LabelFrame(scrollable, text="前處理", padding=10)
        preproc.pack(fill=tk.X, padx=10, pady=5)

        self._width_var = tk.IntVar(value=self.config.target_width)
        self._height_var = tk.IntVar(value=self.config.target_height)
        self._grayscale_var = tk.BooleanVar(value=self.config.grayscale)
        self._blur_var = tk.IntVar(value=self.config.gaussian_blur_kernel)
        self._alignment_var = tk.BooleanVar(value=self.config.enable_alignment)
        self._align_method_var = tk.StringVar(value=self.config.alignment_method)

        r = 0
        ttk.Label(preproc, text="目標寬度:").grid(row=r, column=0, sticky=tk.W, pady=2)
        ttk.Entry(preproc, textvariable=self._width_var, width=10).grid(row=r, column=1, sticky=tk.W, padx=5)
        r += 1
        ttk.Label(preproc, text="目標高度:").grid(row=r, column=0, sticky=tk.W, pady=2)
        ttk.Entry(preproc, textvariable=self._height_var, width=10).grid(row=r, column=1, sticky=tk.W, padx=5)
        r += 1
        ttk.Checkbutton(preproc, text="灰階轉換", variable=self._grayscale_var).grid(
            row=r, column=0, columnspan=2, sticky=tk.W, pady=2,
        )
        r += 1
        ttk.Label(preproc, text="模糊核心:").grid(row=r, column=0, sticky=tk.W, pady=2)
        ttk.Entry(preproc, textvariable=self._blur_var, width=10).grid(row=r, column=1, sticky=tk.W, padx=5)
        r += 1
        ttk.Checkbutton(preproc, text="啟用對齊", variable=self._alignment_var).grid(
            row=r, column=0, columnspan=2, sticky=tk.W, pady=2,
        )
        r += 1
        ttk.Label(preproc, text="對齊方法:").grid(row=r, column=0, sticky=tk.W, pady=2)
        ttk.Combobox(
            preproc, textvariable=self._align_method_var,
            values=["ecc", "feature"], state="readonly", width=10,
        ).grid(row=r, column=1, sticky=tk.W, padx=5)

        # ── 偵測 ──
        detect = ttk.LabelFrame(scrollable, text="偵測閾值", padding=10)
        detect.pack(fill=tk.X, padx=10, pady=5)

        self._abs_var = tk.IntVar(value=self.config.abs_threshold)
        self._var_var = tk.DoubleVar(value=self.config.var_threshold)

        ttk.Label(detect, text="絕對閾值:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(detect, textvariable=self._abs_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=5)
        ttk.Label(detect, text="變異閾值:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(detect, textvariable=self._var_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=5)

        # ── 形態學 ──
        morph = ttk.LabelFrame(scrollable, text="形態學", padding=10)
        morph.pack(fill=tk.X, padx=10, pady=5)

        self._morph_var = tk.IntVar(value=self.config.morph_kernel_size)
        self._area_var = tk.IntVar(value=self.config.min_defect_area)

        ttk.Label(morph, text="核心尺寸:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(morph, textvariable=self._morph_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=5)
        ttk.Label(morph, text="最小面積:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(morph, textvariable=self._area_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=5)

        # ── 多尺度 ──
        ms = ttk.LabelFrame(scrollable, text="多尺度", padding=10)
        ms.pack(fill=tk.X, padx=10, pady=5)

        self._ms_var = tk.BooleanVar(value=self.config.enable_multiscale)
        self._levels_var = tk.IntVar(value=self.config.scale_levels)

        ttk.Checkbutton(ms, text="啟用多尺度偵測", variable=self._ms_var).grid(
            row=0, column=0, columnspan=2, sticky=tk.W, pady=2,
        )
        ttk.Label(ms, text="金字塔層數:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(ms, textvariable=self._levels_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=5)

        # ── 按鈕 ──
        btn_frame = ttk.Frame(scrollable)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(btn_frame, text="套用", command=self._apply).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="儲存至 .env", command=self._save_env).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="取消", command=self.destroy).pack(side=tk.RIGHT, padx=5)

    def _browse_path(self, var: tk.StringVar) -> None:
        d = filedialog.askdirectory()
        if d:
            var.set(d)

    def _get_updated_config(self):
        return self.config.update(
            train_image_dir=self._train_dir_var.get(),
            test_image_dir=self._test_dir_var.get(),
            model_save_dir=self._model_dir_var.get(),
            results_dir=self._results_dir_var.get(),
            target_width=self._width_var.get(),
            target_height=self._height_var.get(),
            grayscale=self._grayscale_var.get(),
            gaussian_blur_kernel=self._blur_var.get(),
            enable_alignment=self._alignment_var.get(),
            alignment_method=self._align_method_var.get(),
            abs_threshold=self._abs_var.get(),
            var_threshold=round(self._var_var.get(), 1),
            morph_kernel_size=self._morph_var.get(),
            min_defect_area=self._area_var.get(),
            enable_multiscale=self._ms_var.get(),
            scale_levels=self._levels_var.get(),
        )

    def _apply(self) -> None:
        try:
            new_config = self._get_updated_config()
            if self._on_apply:
                self._on_apply(new_config)
            messagebox.showinfo("設定", "設定已套用。")
            self.destroy()
        except Exception as exc:
            messagebox.showerror("錯誤", str(exc))

    def _save_env(self) -> None:
        try:
            new_config = self._get_updated_config()
            env_path = Path(__file__).resolve().parent.parent / ".env"
            lines = [
                "# === Image Paths ===",
                f"TRAIN_IMAGE_DIR={new_config.train_image_dir}",
                f"TEST_IMAGE_DIR={new_config.test_image_dir}",
                f"REFERENCE_IMAGE={new_config.reference_image or ''}",
                "",
                "# === Model Persistence ===",
                f"MODEL_SAVE_DIR={new_config.model_save_dir}",
                f"RESULTS_DIR={new_config.results_dir}",
                "",
                "# === Preprocessing ===",
                f"TARGET_WIDTH={new_config.target_width}",
                f"TARGET_HEIGHT={new_config.target_height}",
                f"GRAYSCALE={'true' if new_config.grayscale else 'false'}",
                f"GAUSSIAN_BLUR_KERNEL={new_config.gaussian_blur_kernel}",
                f"ENABLE_ALIGNMENT={'true' if new_config.enable_alignment else 'false'}",
                f"ALIGNMENT_METHOD={new_config.alignment_method}",
                "",
                "# === Variation Model ===",
                f"ABS_THRESHOLD={new_config.abs_threshold}",
                f"VAR_THRESHOLD={new_config.var_threshold}",
                "",
                "# === Morphological ===",
                f"MORPH_KERNEL_SIZE={new_config.morph_kernel_size}",
                f"MIN_DEFECT_AREA={new_config.min_defect_area}",
                "",
                "# === Multi-scale ===",
                f"ENABLE_MULTISCALE={'true' if new_config.enable_multiscale else 'false'}",
                f"SCALE_LEVELS={new_config.scale_levels}",
            ]
            env_path.write_text("\n".join(lines), encoding="utf-8")
            if self._on_apply:
                self._on_apply(new_config)
            messagebox.showinfo("已儲存", f"設定已儲存至:\n{env_path}")
            self.destroy()
        except Exception as exc:
            messagebox.showerror("錯誤", str(exc))
