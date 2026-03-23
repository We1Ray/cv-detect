"""Dialogs for the Industrial Vision-style DL Anomaly Detector.

Includes:
- TrainingDialog: directory selection, progress, live loss curve, sample grid
- BatchInspectDialog: directory, progress, results table
- ModelInfoDialog: architecture summary, parameter count, training config
- HistogramDialog: image histogram
- ReconstructionDialog: side-by-side comparison
- SettingsDialog: all configuration parameters, save to .env
"""

from __future__ import annotations

import logging
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable, Dict, List, Optional, Tuple

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
    _MONO_FAMILY = "Consolas"


# ======================================================================
# Training Dialog
# ======================================================================

class TrainingDialog(tk.Toplevel):
    """Modal training dialog with live loss curve and sample reconstruction grid.

    Parameters
    ----------
    config : Config
        Project configuration.
    on_complete : callable(result_dict) | None
        Called on the main thread when training finishes.
    """

    POLL_MS = 100

    def __init__(
        self,
        master: tk.Misc,
        config: "Config",
        on_complete: Optional[Callable] = None,
    ) -> None:
        super().__init__(master)
        self.title("\u6a21\u578b\u8a13\u7df4")  # "Model Training"
        self.geometry("850x680")
        self.minsize(700, 550)
        self.transient(master)
        self.grab_set()

        self.config = config
        self._on_complete = on_complete
        self._pipeline = None
        self._train_thread: Optional[threading.Thread] = None
        self._queue: queue.Queue = queue.Queue()

        self._train_losses: List[float] = []
        self._val_losses: List[float] = []

        self._build_ui()
        self._poll_queue()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        # --- Top: Directory selection ---
        dir_frame = ttk.LabelFrame(self, text="\u8a13\u7df4\u8cc7\u6599", padding=6)
        dir_frame.pack(fill=tk.X, padx=8, pady=(8, 4))

        self._dir_var = tk.StringVar(value=str(self.config.train_image_dir))
        ttk.Label(dir_frame, text="\u8cc7\u6599\u593e:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(dir_frame, textvariable=self._dir_var, width=55).grid(row=0, column=1, padx=4)
        ttk.Button(dir_frame, text="\u700f\u89bd...", command=self._browse_dir).grid(row=0, column=2)
        self._count_label = ttk.Label(dir_frame, text="\u5716\u7247: --")
        self._count_label.grid(row=0, column=3, padx=8)

        # --- Controls ---
        ctrl_frame = ttk.Frame(self, padding=4)
        ctrl_frame.pack(fill=tk.X, padx=8)

        self._start_btn = ttk.Button(ctrl_frame, text="\u958b\u59cb\u8a13\u7df4", command=self._start_training)
        self._start_btn.pack(side=tk.LEFT, padx=2)
        self._stop_btn = ttk.Button(ctrl_frame, text="\u505c\u6b62", command=self._stop_training, state=tk.DISABLED)
        self._stop_btn.pack(side=tk.LEFT, padx=2)

        # Epoch + progress
        self._epoch_var = tk.StringVar(value="Epoch: 0 / 0")
        ttk.Label(ctrl_frame, textvariable=self._epoch_var).pack(side=tk.LEFT, padx=12)
        self._progress = ttk.Progressbar(ctrl_frame, length=180, mode="determinate")
        self._progress.pack(side=tk.LEFT, padx=4)

        # LR + best val loss
        info_frame = ttk.Frame(ctrl_frame)
        info_frame.pack(side=tk.RIGHT)
        self._lr_var = tk.StringVar(value="LR: --")
        ttk.Label(info_frame, textvariable=self._lr_var, font=(_FONT_FAMILY, 8)).pack(side=tk.LEFT, padx=6)
        self._best_var = tk.StringVar(value="Best Val: --")
        ttk.Label(info_frame, textvariable=self._best_var, font=(_FONT_FAMILY, 8)).pack(side=tk.LEFT, padx=6)

        # --- Middle: loss plot + sample grid ---
        mid_frame = ttk.Frame(self)
        mid_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        # Left: loss curve (matplotlib)
        plot_frame = ttk.LabelFrame(mid_frame, text="\u640d\u5931\u66f2\u7dda", padding=4)
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        self._loss_fig = Figure(figsize=(4.5, 3.2), dpi=90)
        self._loss_fig.patch.set_facecolor("#f0f0f0")
        self._loss_ax = self._loss_fig.add_subplot(111)
        self._loss_ax.set_xlabel("Epoch")
        self._loss_ax.set_ylabel("Loss")
        self._loss_ax.grid(True, alpha=0.3)
        self._loss_fig.tight_layout()

        self._loss_canvas = FigureCanvasTkAgg(self._loss_fig, master=plot_frame)
        self._loss_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Right: sample reconstruction grid
        sample_frame = ttk.LabelFrame(mid_frame, text="\u91cd\u5efa\u6a23\u672c", padding=4)
        sample_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0))

        self._sample_fig = Figure(figsize=(3.5, 3.2), dpi=90)
        self._sample_canvas = FigureCanvasTkAgg(self._sample_fig, master=sample_frame)
        self._sample_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # --- Bottom: log ---
        log_frame = ttk.LabelFrame(self, text="\u8a13\u7df4\u65e5\u8a8c", padding=4)
        log_frame.pack(fill=tk.X, padx=8, pady=(4, 8))

        self._log_text = tk.Text(log_frame, height=6, state=tk.DISABLED, wrap=tk.WORD, font=(_MONO_FAMILY, 9))
        scrollbar = ttk.Scrollbar(log_frame, command=self._log_text.yview)
        self._log_text.configure(yscrollcommand=scrollbar.set)
        self._log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # ------------------------------------------------------------------
    # Directory
    # ------------------------------------------------------------------

    def _browse_dir(self) -> None:
        d = filedialog.askdirectory(title="\u9078\u64c7\u8a13\u7df4\u5716\u7247\u8cc7\u6599\u593e")
        if d:
            self._dir_var.set(d)
            self._update_image_count(Path(d))

    def _update_image_count(self, directory: Path) -> None:
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        count = sum(1 for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in exts)
        self._count_label.configure(text=f"\u5716\u7247: {count}")

    # ------------------------------------------------------------------
    # Training lifecycle
    # ------------------------------------------------------------------

    def _start_training(self) -> None:
        from dl_anomaly.pipeline.trainer import TrainingPipeline

        train_dir = Path(self._dir_var.get())
        if not train_dir.exists():
            messagebox.showerror("\u932f\u8aa4", f"\u8cc7\u6599\u593e\u4e0d\u5b58\u5728:\n{train_dir}")
            return

        self.config.train_image_dir = train_dir
        self._train_losses.clear()
        self._val_losses.clear()
        self._pipeline = TrainingPipeline(self.config)

        self._start_btn.configure(state=tk.DISABLED)
        self._stop_btn.configure(state=tk.NORMAL)
        self._log("\u8a13\u7df4\u958b\u59cb...")

        self._train_thread = threading.Thread(target=self._train_worker, daemon=True)
        self._train_thread.start()

    def _stop_training(self) -> None:
        if self._pipeline is not None:
            self._pipeline.request_stop()
            self._log("\u5df2\u8acb\u6c42\u505c\u6b62\uff0c\u7b49\u5f85\u7576\u524d epoch \u5b8c\u6210...")
            self._stop_btn.configure(state=tk.DISABLED)

    def _train_worker(self) -> None:
        try:
            result = self._pipeline.run(progress_callback=self._progress_cb)
            self._queue.put(("done", result))
        except Exception as exc:
            self._queue.put(("error", str(exc)))

    def _progress_cb(self, info: Dict[str, Any]) -> None:
        self._queue.put(("progress", info))

    # ------------------------------------------------------------------
    # Queue polling
    # ------------------------------------------------------------------

    def _poll_queue(self) -> None:
        try:
            while True:
                tag, data = self._queue.get_nowait()
                if tag == "progress":
                    self._handle_progress(data)
                elif tag == "done":
                    self._handle_done(data)
                elif tag == "error":
                    self._handle_error(data)
        except queue.Empty:
            pass

        if self.winfo_exists():
            self.after(self.POLL_MS, self._poll_queue)

    def _handle_progress(self, info: Dict[str, Any]) -> None:
        epoch = info["epoch"]
        total = info["total_epochs"]
        tl = info["train_loss"]
        vl = info["val_loss"]

        self._epoch_var.set(f"Epoch: {epoch} / {total}")
        self._progress["maximum"] = total
        self._progress["value"] = epoch
        self._lr_var.set(f"LR: {info['lr']:.2e}")
        self._best_var.set(f"Best Val: {info['best_loss']:.6f}")

        self._train_losses.append(tl)
        self._val_losses.append(vl)
        self._update_loss_plot()

        self._log(
            f"Epoch {epoch:3d}/{total}  train={tl:.6f}  val={vl:.6f}  "
            f"best={info['best_loss']:.6f}  lr={info['lr']:.2e}  ({info['elapsed']:.1f}s)"
        )

    def _handle_done(self, result: Dict[str, Any]) -> None:
        self._start_btn.configure(state=tk.NORMAL)
        self._stop_btn.configure(state=tk.DISABLED)
        self._log(
            f"\u8a13\u7df4\u5b8c\u6210  best_val_loss={result['best_val_loss']:.6f}  "
            f"threshold={result['threshold']:.6f}"
        )
        self._log(f"\u6a21\u578b\u5df2\u5132\u5b58: {result['checkpoint_path']}")

        if self._on_complete:
            self._on_complete(result)

    def _handle_error(self, msg: str) -> None:
        self._start_btn.configure(state=tk.NORMAL)
        self._stop_btn.configure(state=tk.DISABLED)
        self._log(f"\u932f\u8aa4: {msg}")
        messagebox.showerror("\u8a13\u7df4\u932f\u8aa4", msg)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    def _update_loss_plot(self) -> None:
        ax = self._loss_ax
        ax.clear()
        epochs = list(range(1, len(self._train_losses) + 1))
        ax.plot(epochs, self._train_losses, label="Train", linewidth=1.2)
        ax.plot(epochs, self._val_losses, label="Val", linewidth=1.2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        self._loss_fig.tight_layout()
        self._loss_canvas.draw_idle()

    # ------------------------------------------------------------------
    # Log
    # ------------------------------------------------------------------

    def _log(self, message: str) -> None:
        self._log_text.configure(state=tk.NORMAL)
        self._log_text.insert(tk.END, message + "\n")
        self._log_text.see(tk.END)
        self._log_text.configure(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------

    def _on_close(self) -> None:
        if self._train_thread and self._train_thread.is_alive():
            if not messagebox.askyesno("\u78ba\u8a8d", "\u8a13\u7df4\u4e2d\uff0c\u78ba\u5b9a\u8981\u505c\u6b62\u4e26\u95dc\u9589\uff1f"):
                return
            self._stop_training()
        self.destroy()

    def get_pipeline(self):
        return self._pipeline


# ======================================================================
# Batch Inspection Dialog
# ======================================================================

class BatchInspectDialog(tk.Toplevel):
    """Batch inspection dialog with directory selection, progress, and results table."""

    POLL_MS = 100

    def __init__(
        self,
        master: tk.Misc,
        inference_pipeline: Any,
        on_result_selected: Optional[Callable] = None,
    ) -> None:
        super().__init__(master)
        self.title("\u6279\u6b21\u6aa2\u6e2c")  # "Batch Inspection"
        self.geometry("750x550")
        self.transient(master)
        self.grab_set()

        self._pipeline = inference_pipeline
        self._on_result_selected = on_result_selected
        self._queue: queue.Queue = queue.Queue()
        self._results: List = []
        self._result_paths: List[str] = []

        self._build_ui()
        self._poll_queue()

    def _build_ui(self) -> None:
        # Directory
        dir_frame = ttk.Frame(self, padding=6)
        dir_frame.pack(fill=tk.X, padx=8, pady=(8, 4))

        self._dir_var = tk.StringVar()
        ttk.Label(dir_frame, text="\u5716\u7247\u8cc7\u6599\u593e:").pack(side=tk.LEFT)
        ttk.Entry(dir_frame, textvariable=self._dir_var, width=45).pack(side=tk.LEFT, padx=4)
        ttk.Button(dir_frame, text="\u700f\u89bd...", command=self._browse).pack(side=tk.LEFT, padx=2)
        ttk.Button(dir_frame, text="\u958b\u59cb\u6aa2\u6e2c", command=self._start).pack(side=tk.LEFT, padx=6)

        # Progress
        self._progress = ttk.Progressbar(self, length=300, mode="determinate")
        self._progress.pack(fill=tk.X, padx=8, pady=4)
        self._status_var = tk.StringVar(value="\u5c31\u7dd2")
        ttk.Label(self, textvariable=self._status_var).pack(padx=8, anchor=tk.W)

        # Results table
        table_frame = ttk.Frame(self)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        cols = ("\u6a94\u540d", "\u5206\u6578", "\u7d50\u679c", "\u7f3a\u9677\u5340\u57df")
        self._tree = ttk.Treeview(table_frame, columns=cols, show="headings", selectmode="browse")
        headers = [("\u6a94\u540d", 220), ("\u5206\u6578", 100), ("\u7d50\u679c", 80), ("\u7f3a\u9677\u5340\u57df", 80)]
        for col, (text, w) in zip(cols, headers):
            self._tree.heading(col, text=text)
            self._tree.column(col, width=w, anchor=tk.CENTER)

        tree_scroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self._tree.yview)
        self._tree.configure(yscrollcommand=tree_scroll.set)
        self._tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self._tree.bind("<Double-1>", self._on_double_click)

        # Summary
        self._summary_var = tk.StringVar(value="")
        ttk.Label(self, textvariable=self._summary_var, font=(_FONT_FAMILY, 9, "bold")).pack(
            fill=tk.X, padx=8, pady=(4, 8)
        )

    def _browse(self) -> None:
        d = filedialog.askdirectory(title="\u9078\u64c7\u5716\u7247\u8cc7\u6599\u593e")
        if d:
            self._dir_var.set(d)

    def _start(self) -> None:
        d = self._dir_var.get()
        if not d or not Path(d).exists():
            messagebox.showerror("\u932f\u8aa4", "\u8acb\u9078\u64c7\u6709\u6548\u7684\u8cc7\u6599\u593e")
            return

        self._results.clear()
        self._result_paths.clear()
        for item in self._tree.get_children():
            self._tree.delete(item)

        self._status_var.set("\u6aa2\u6e2c\u4e2d...")

        t = threading.Thread(target=self._worker, args=(d,), daemon=True)
        t.start()

    def _worker(self, directory: str) -> None:
        try:
            exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
            paths = sorted(
                str(p) for p in Path(directory).rglob("*")
                if p.is_file() and p.suffix.lower() in exts
            )

            def cb(cur: int, total: int) -> None:
                self._queue.put(("progress", (cur, total)))

            results = self._pipeline.inspect_batch(directory, progress_callback=cb)
            self._queue.put(("done", results, paths))
        except Exception as exc:
            self._queue.put(("error", str(exc)))

    def _poll_queue(self) -> None:
        try:
            while True:
                msg = self._queue.get_nowait()
                tag = msg[0]
                if tag == "progress":
                    cur, total = msg[1]
                    self._progress["maximum"] = total
                    self._progress["value"] = cur
                    self._status_var.set(f"\u6aa2\u6e2c\u4e2d... {cur}/{total}")
                elif tag == "done":
                    results, paths = msg[1], msg[2]
                    self._results = results
                    self._result_paths = paths
                    self._populate_table()
                    self._status_var.set(f"\u5b8c\u6210 - \u5171 {len(results)} \u5f35\u5716\u7247")
                elif tag == "error":
                    messagebox.showerror("\u932f\u8aa4", msg[1])
                    self._status_var.set("\u932f\u8aa4")
        except queue.Empty:
            pass

        if self.winfo_exists():
            self.after(self.POLL_MS, self._poll_queue)

    def _populate_table(self) -> None:
        for i, (result, path) in enumerate(zip(self._results, self._result_paths)):
            name = Path(path).name
            score = f"{result.anomaly_score:.6f}"
            label = "\u7f3a\u9677" if result.is_defective else "\u901a\u904e"
            regions = str(len(result.defect_regions))
            self._tree.insert("", tk.END, iid=str(i), values=(name, score, label, regions))

        # Summary
        total = len(self._results)
        defective = sum(1 for r in self._results if r.is_defective)
        rate = defective / total * 100 if total > 0 else 0
        self._summary_var.set(
            f"\u7e3d\u8a08: {total}  |  \u7f3a\u9677: {defective}  |  \u901a\u904e: {total - defective}  |  "
            f"\u7f3a\u9677\u7387: {rate:.1f}%"
        )

    def _on_double_click(self, event: tk.Event) -> None:
        sel = self._tree.selection()
        if sel and self._on_result_selected:
            idx = int(sel[0])
            self._on_result_selected(idx, self._results[idx], self._result_paths[idx])

    def get_results(self) -> Tuple[List, List[str]]:
        return self._results, self._result_paths


# ======================================================================
# Model Info Dialog
# ======================================================================

class ModelInfoDialog(tk.Toplevel):
    """Display model architecture, parameter count, and training configuration."""

    def __init__(self, master: tk.Misc, model, config, state: Optional[Dict] = None) -> None:
        super().__init__(master)
        self.title("\u6a21\u578b\u8cc7\u8a0a")  # "Model Info"
        self.geometry("500x450")
        self.transient(master)
        self.grab_set()

        text = tk.Text(self, font=(_MONO_FAMILY, 10), wrap=tk.WORD, padx=10, pady=10)
        text.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Architecture
        total_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        info_lines = [
            "=== \u6a21\u578b\u67b6\u69cb ===",
            f"\u985e\u578b: {model.__class__.__name__}",
            f"\u8f38\u5165\u901a\u9053: {model.in_channels}",
            f"\u6f5b\u5728\u7a7a\u9593\u7dad\u5ea6: {model.latent_dim}",
            f"\u57fa\u790e\u901a\u9053\u6578: {model.base_channels}",
            f"Encoder \u5340\u584a\u6578: {model.num_blocks}",
            f"\u5716\u7247\u5c3a\u5bf8: {model.image_size}x{model.image_size}",
            "",
            "=== \u53c3\u6578\u7d71\u8a08 ===",
            f"\u7e3d\u53c3\u6578\u6578: {total_params:,}",
            f"\u53ef\u8a13\u7df4\u53c3\u6578: {trainable:,}",
            "",
            "=== \u8a13\u7df4\u8a2d\u5b9a ===",
            f"\u5b78\u7fd2\u7387: {config.learning_rate}",
            f"Batch Size: {config.batch_size}",
            f"Epochs: {config.num_epochs}",
            f"Early Stopping: {config.early_stopping_patience}",
            f"SSIM Weight: {config.ssim_weight}",
            f"\u88dd\u7f6e: {config.device}",
        ]

        if state:
            info_lines.extend([
                "",
                "=== Checkpoint ===",
                f"Epoch: {state.get('epoch', '?')}",
                f"Loss: {state.get('loss', '?')}",
            ])
            if "threshold" in state:
                info_lines.append(f"\u7570\u5e38\u95be\u503c: {state['threshold']:.6f}")

        text.insert("1.0", "\n".join(info_lines))
        text.configure(state=tk.DISABLED)

        ttk.Button(self, text="\u95dc\u9589", command=self.destroy).pack(pady=8)


# ======================================================================
# Histogram Dialog
# ======================================================================

class HistogramDialog(tk.Toplevel):
    """Display histogram of the current image."""

    def __init__(self, master: tk.Misc, array: np.ndarray, title_text: str = "") -> None:
        super().__init__(master)
        self.title(title_text or "\u5716\u7247\u76f4\u65b9\u5716")  # "Image Histogram"
        self.geometry("520x380")
        self.transient(master)

        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        fig = Figure(figsize=(5, 3.5), dpi=100)
        ax = fig.add_subplot(111)

        if array.ndim == 2:
            ax.hist(array.ravel(), bins=256, range=(0, 255), color="steelblue", alpha=0.7, edgecolor="none")
        elif array.ndim == 3 and array.shape[2] == 3:
            colors = ["#e74c3c", "#2ecc71", "#3498db"]
            labels = ["R", "G", "B"]
            for c, color, label in zip(range(3), colors, labels):
                ax.hist(array[:, :, c].ravel(), bins=256, range=(0, 255),
                        color=color, alpha=0.5, edgecolor="none", label=label)
            ax.legend(fontsize=8)
        else:
            ax.hist(array.ravel(), bins=256, color="steelblue", alpha=0.7, edgecolor="none")

        ax.set_xlabel("\u50cf\u7d20\u503c")
        ax.set_ylabel("\u6b21\u6578")
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        canvas.draw()


# ======================================================================
# Reconstruction Comparison Dialog
# ======================================================================

class ReconstructionDialog(tk.Toplevel):
    """Side-by-side original vs reconstruction comparison."""

    def __init__(
        self,
        master: tk.Misc,
        original: np.ndarray,
        reconstruction: np.ndarray,
    ) -> None:
        super().__init__(master)
        self.title("\u91cd\u5efa\u5c0d\u6bd4")  # "Reconstruction Comparison"
        self.geometry("700x400")
        self.transient(master)

        self._photos = []  # prevent GC

        frame = ttk.Frame(self)
        frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Left: original
        left = ttk.LabelFrame(frame, text="\u539f\u5716", padding=4)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        self._show_image(left, original)

        # Right: reconstruction
        right = ttk.LabelFrame(frame, text="\u91cd\u5efa", padding=4)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._show_image(right, reconstruction)

    def _show_image(self, parent: tk.Widget, array: np.ndarray) -> None:
        if array.ndim == 2:
            img = Image.fromarray(array, mode="L")
        else:
            img = Image.fromarray(array, mode="RGB")
        img.thumbnail((320, 320), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self._photos.append(photo)
        lbl = ttk.Label(parent, image=photo)
        lbl.pack(fill=tk.BOTH, expand=True)


# ======================================================================
# Settings Dialog
# ======================================================================

class SettingsDialog(tk.Toplevel):
    """All configuration parameters with save-to-.env support."""

    def __init__(self, master: tk.Misc, config: "Config") -> None:
        super().__init__(master)
        self.title("\u8a2d\u5b9a")  # "Settings"
        self.geometry("480x620")
        self.transient(master)
        self.grab_set()

        self.config = config
        self._vars: Dict[str, tk.Variable] = {}

        self._build_ui()
        self._load_from_config()

    def _build_ui(self) -> None:
        # Scrollable content
        canvas = tk.Canvas(self, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=canvas.yview)
        content = ttk.Frame(canvas, padding=8)

        content.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=content, anchor=tk.NW)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        row = 0

        # --- Paths ---
        row = self._section(content, "\u8def\u5f91\u8a2d\u5b9a", row)

        self._vars["train_image_dir"] = tk.StringVar(value=str(self.config.train_image_dir))
        self._vars["test_image_dir"] = tk.StringVar(value=str(self.config.test_image_dir))
        self._vars["checkpoint_dir"] = tk.StringVar(value=str(self.config.checkpoint_dir))
        self._vars["results_dir"] = tk.StringVar(value=str(self.config.results_dir))

        for label_text, key in [
            ("\u8a13\u7df4\u5716\u7247\u76ee\u9304:", "train_image_dir"),
            ("\u6e2c\u8a66\u5716\u7247\u76ee\u9304:", "test_image_dir"),
            ("Checkpoint \u76ee\u9304:", "checkpoint_dir"),
            ("\u7d50\u679c\u8f38\u51fa\u76ee\u9304:", "results_dir"),
        ]:
            ttk.Label(content, text=label_text).grid(row=row, column=0, sticky=tk.W, pady=2)
            entry_frame = ttk.Frame(content)
            entry_frame.grid(row=row, column=1, sticky=tk.EW, pady=2)
            ttk.Entry(entry_frame, textvariable=self._vars[key], width=28).pack(side=tk.LEFT, fill=tk.X, expand=True)
            ttk.Button(
                entry_frame, text="...", width=3,
                command=lambda v=self._vars[key]: self._browse_dir(v),
            ).pack(side=tk.LEFT, padx=2)
            row += 1

        # --- Architecture ---
        row = self._section(content, "\u67b6\u69cb", row)
        row = self._entry(content, "latent_dim", "\u6f5b\u5728\u7a7a\u9593\u7dad\u5ea6:", row)
        row = self._entry(content, "base_channels", "\u57fa\u790e\u901a\u9053\u6578:", row)
        row = self._entry(content, "num_encoder_blocks", "Encoder \u5340\u584a\u6578:", row)
        row = self._entry(content, "image_size", "\u5716\u7247\u5c3a\u5bf8:", row)

        # --- Training ---
        row = self._section(content, "\u8a13\u7df4", row)
        row = self._entry(content, "learning_rate", "\u5b78\u7fd2\u7387:", row)
        row = self._entry(content, "batch_size", "Batch Size:", row)
        row = self._entry(content, "num_epochs", "Epochs:", row)
        row = self._entry(content, "early_stopping_patience", "Early Stopping:", row)

        # --- Device ---
        row = self._section(content, "\u88dd\u7f6e", row)
        import torch
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        self._vars["device"] = tk.StringVar()
        ttk.Label(content, text="\u88dd\u7f6e:").grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Combobox(content, textvariable=self._vars["device"], values=devices, state="readonly", width=10).grid(
            row=row, column=1, sticky=tk.W, pady=2
        )
        row += 1

        self._vars["grayscale"] = tk.BooleanVar()
        ttk.Checkbutton(content, text="\u7070\u968e\u8f38\u5165", variable=self._vars["grayscale"]).grid(
            row=row, column=0, columnspan=2, sticky=tk.W, pady=2
        )
        row += 1

        # --- Anomaly ---
        row = self._section(content, "\u7570\u5e38\u5075\u6e2c", row)
        self._vars["anomaly_threshold_percentile"] = tk.DoubleVar()
        ttk.Label(content, text="\u95be\u503c\u767e\u5206\u4f4d:").grid(row=row, column=0, sticky=tk.W, pady=2)
        tk.Scale(
            content, from_=80.0, to=99.9, resolution=0.1, orient=tk.HORIZONTAL,
            variable=self._vars["anomaly_threshold_percentile"], length=200,
        ).grid(row=row, column=1, sticky=tk.W, pady=2)
        row += 1

        self._vars["ssim_weight"] = tk.DoubleVar()
        ttk.Label(content, text="SSIM \u6b0a\u91cd:").grid(row=row, column=0, sticky=tk.W, pady=2)
        tk.Scale(
            content, from_=0.0, to=1.0, resolution=0.05, orient=tk.HORIZONTAL,
            variable=self._vars["ssim_weight"], length=200,
        ).grid(row=row, column=1, sticky=tk.W, pady=2)
        row += 1

        # --- Buttons ---
        btn_frame = ttk.Frame(content)
        btn_frame.grid(row=row, column=0, columnspan=2, pady=12)
        ttk.Button(btn_frame, text="\u5957\u7528", command=self._apply).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="\u91cd\u8a2d", command=self._load_from_config).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="\u5132\u5b58\u81f3 .env", command=self._save_env).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="\u95dc\u9589", command=self.destroy).pack(side=tk.LEFT, padx=4)

    def _section(self, parent: tk.Widget, text: str, row: int) -> int:
        ttk.Separator(parent, orient=tk.HORIZONTAL).grid(row=row, column=0, columnspan=2, sticky="ew", pady=(10, 2))
        row += 1
        ttk.Label(parent, text=text, font=(_FONT_FAMILY, 10, "bold")).grid(row=row, column=0, columnspan=2, sticky=tk.W)
        row += 1
        return row

    def _entry(self, parent: tk.Widget, key: str, label: str, row: int) -> int:
        self._vars[key] = tk.StringVar()
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(parent, textvariable=self._vars[key], width=14).grid(row=row, column=1, sticky=tk.W, pady=2)
        return row + 1

    def _browse_dir(self, var: tk.Variable) -> None:
        d = filedialog.askdirectory()
        if d:
            var.set(d)

    def _load_from_config(self) -> None:
        c = self.config
        # Paths
        self._vars["train_image_dir"].set(str(c.train_image_dir))
        self._vars["test_image_dir"].set(str(c.test_image_dir))
        self._vars["checkpoint_dir"].set(str(c.checkpoint_dir))
        self._vars["results_dir"].set(str(c.results_dir))
        # Architecture
        self._vars["latent_dim"].set(str(c.latent_dim))
        self._vars["base_channels"].set(str(c.base_channels))
        self._vars["num_encoder_blocks"].set(str(c.num_encoder_blocks))
        self._vars["image_size"].set(str(c.image_size))
        # Training
        self._vars["learning_rate"].set(str(c.learning_rate))
        self._vars["batch_size"].set(str(c.batch_size))
        self._vars["num_epochs"].set(str(c.num_epochs))
        self._vars["early_stopping_patience"].set(str(c.early_stopping_patience))
        # Device
        self._vars["device"].set(c.device)
        self._vars["grayscale"].set(c.grayscale)
        # Anomaly
        self._vars["anomaly_threshold_percentile"].set(c.anomaly_threshold_percentile)
        self._vars["ssim_weight"].set(c.ssim_weight)

    def _apply(self) -> None:
        try:
            self.config.train_image_dir = self._vars["train_image_dir"].get()
            self.config.test_image_dir = self._vars["test_image_dir"].get()
            self.config.checkpoint_dir = self._vars["checkpoint_dir"].get()
            self.config.results_dir = self._vars["results_dir"].get()
            self.config.latent_dim = int(self._vars["latent_dim"].get())
            self.config.base_channels = int(self._vars["base_channels"].get())
            self.config.num_encoder_blocks = int(self._vars["num_encoder_blocks"].get())
            self.config.image_size = int(self._vars["image_size"].get())
            self.config.learning_rate = float(self._vars["learning_rate"].get())
            self.config.batch_size = int(self._vars["batch_size"].get())
            self.config.num_epochs = int(self._vars["num_epochs"].get())
            self.config.early_stopping_patience = int(self._vars["early_stopping_patience"].get())
            self.config.device = self._vars["device"].get()
            self.config.grayscale = self._vars["grayscale"].get()
            self.config.in_channels = 1 if self.config.grayscale else 3
            self.config.anomaly_threshold_percentile = float(self._vars["anomaly_threshold_percentile"].get())
            self.config.ssim_weight = float(self._vars["ssim_weight"].get())
            messagebox.showinfo("\u8a2d\u5b9a", "\u8a2d\u5b9a\u5df2\u5957\u7528")
        except (ValueError, TypeError) as exc:
            messagebox.showerror("\u9a57\u8b49\u932f\u8aa4", str(exc))

    def _save_env(self) -> None:
        self._apply()
        env_path = Path(__file__).resolve().parent.parent / ".env"
        lines = [
            "# === Image Paths ===",
            f"TRAIN_IMAGE_DIR={self.config.train_image_dir}",
            f"TEST_IMAGE_DIR={self.config.test_image_dir}",
            "",
            "# === Model Persistence ===",
            f"CHECKPOINT_DIR={self.config.checkpoint_dir}",
            f"RESULTS_DIR={self.config.results_dir}",
            "",
            "# === Preprocessing ===",
            f"IMAGE_SIZE={self.config.image_size}",
            f"GRAYSCALE={'true' if self.config.grayscale else 'false'}",
            "",
            "# === Architecture ===",
            f"LATENT_DIM={self.config.latent_dim}",
            f"BASE_CHANNELS={self.config.base_channels}",
            f"NUM_ENCODER_BLOCKS={self.config.num_encoder_blocks}",
            "",
            "# === Training ===",
            f"BATCH_SIZE={self.config.batch_size}",
            f"LEARNING_RATE={self.config.learning_rate}",
            f"NUM_EPOCHS={self.config.num_epochs}",
            f"EARLY_STOPPING_PATIENCE={self.config.early_stopping_patience}",
            f"DEVICE={self.config.device}",
            "",
            "# === Anomaly Detection ===",
            f"ANOMALY_THRESHOLD_PERCENTILE={self.config.anomaly_threshold_percentile}",
            f"SSIM_WEIGHT={self.config.ssim_weight}",
        ]
        try:
            env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            messagebox.showinfo("\u5132\u5b58", f"\u8a2d\u5b9a\u5df2\u5beb\u5165:\n{env_path}")
        except OSError as exc:
            messagebox.showerror("\u5132\u5b58\u932f\u8aa4", str(exc))
