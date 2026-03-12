"""Training tab for the DL Anomaly Detector GUI.

Provides directory selection, training controls, an embedded matplotlib loss
curve, sample reconstruction previews, and a scrollable log area.  All heavy
computation runs in a background thread; UI updates are delivered via a
``queue.Queue`` polled by ``root.after(100, ...)``.
"""

from __future__ import annotations

import logging
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, List, Optional

import numpy as np

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from dl_anomaly.config import Config
from dl_anomaly.core.preprocessor import ImagePreprocessor
from dl_anomaly.pipeline.trainer import TrainingPipeline
from dl_anomaly.visualization.training_plots import plot_loss_curve

logger = logging.getLogger(__name__)


class TrainTab(ttk.Frame):
    """Training tab with controls, live loss plot, and log."""

    POLL_INTERVAL_MS = 100

    def __init__(self, master: tk.Misc, config: Config, **kwargs) -> None:
        super().__init__(master, **kwargs)
        self.config = config
        self._pipeline: Optional[TrainingPipeline] = None
        self._train_thread: Optional[threading.Thread] = None
        self._queue: queue.Queue = queue.Queue()

        # Accumulated loss history for live plotting
        self._train_losses: List[float] = []
        self._val_losses: List[float] = []

        self._build_ui()
        self._poll_queue()

    # ==================================================================
    # UI construction
    # ==================================================================

    def _build_ui(self) -> None:
        # --- Top: directory selection ---------------------------------
        dir_frame = ttk.LabelFrame(self, text="Training Data", padding=6)
        dir_frame.pack(fill=tk.X, padx=6, pady=(6, 3))

        self._dir_var = tk.StringVar(value=str(self.config.train_image_dir))
        ttk.Label(dir_frame, text="Directory:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(dir_frame, textvariable=self._dir_var, width=60).grid(row=0, column=1, padx=4)
        ttk.Button(dir_frame, text="Browse...", command=self._browse_dir).grid(row=0, column=2)
        self._count_label = ttk.Label(dir_frame, text="Images: --")
        self._count_label.grid(row=0, column=3, padx=8)

        # --- Controls -------------------------------------------------
        ctrl_frame = ttk.Frame(self, padding=6)
        ctrl_frame.pack(fill=tk.X, padx=6)

        self._start_btn = ttk.Button(ctrl_frame, text="Start Training", command=self._start_training)
        self._start_btn.pack(side=tk.LEFT, padx=2)
        self._stop_btn = ttk.Button(ctrl_frame, text="Stop", command=self._stop_training, state=tk.DISABLED)
        self._stop_btn.pack(side=tk.LEFT, padx=2)
        self._save_btn = ttk.Button(ctrl_frame, text="Save Checkpoint", command=self._save_checkpoint, state=tk.DISABLED)
        self._save_btn.pack(side=tk.LEFT, padx=2)
        self._load_btn = ttk.Button(ctrl_frame, text="Load Checkpoint", command=self._load_checkpoint)
        self._load_btn.pack(side=tk.LEFT, padx=2)

        # Progress
        self._epoch_var = tk.StringVar(value="Epoch: 0 / 0")
        ttk.Label(ctrl_frame, textvariable=self._epoch_var).pack(side=tk.LEFT, padx=12)
        self._progress = ttk.Progressbar(ctrl_frame, length=200, mode="determinate")
        self._progress.pack(side=tk.LEFT, padx=4)

        # --- Middle: loss plot + reconstruction samples ---------------
        mid_frame = ttk.Frame(self)
        mid_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=3)

        # Left: matplotlib loss curve
        plot_frame = ttk.LabelFrame(mid_frame, text="Loss Curve", padding=4)
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._loss_fig = Figure(figsize=(5, 3), dpi=90)
        self._loss_ax = self._loss_fig.add_subplot(111)
        self._loss_ax.set_xlabel("Epoch")
        self._loss_ax.set_ylabel("Loss")
        self._loss_ax.grid(True, alpha=0.3)
        self._loss_fig.tight_layout()

        self._loss_canvas = FigureCanvasTkAgg(self._loss_fig, master=plot_frame)
        self._loss_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Right: sample reconstructions (just a canvas placeholder)
        sample_frame = ttk.LabelFrame(mid_frame, text="Reconstruction Samples", padding=4)
        sample_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0))

        self._sample_fig = Figure(figsize=(4, 3), dpi=90)
        self._sample_canvas = FigureCanvasTkAgg(self._sample_fig, master=sample_frame)
        self._sample_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # --- Bottom: log area -----------------------------------------
        log_frame = ttk.LabelFrame(self, text="Training Log", padding=4)
        log_frame.pack(fill=tk.X, padx=6, pady=(3, 6))

        self._log_text = tk.Text(log_frame, height=8, state=tk.DISABLED, wrap=tk.WORD, font=("Consolas", 9))
        scrollbar = ttk.Scrollbar(log_frame, command=self._log_text.yview)
        self._log_text.configure(yscrollcommand=scrollbar.set)
        self._log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # ==================================================================
    # Directory browser
    # ==================================================================

    def _browse_dir(self) -> None:
        d = filedialog.askdirectory(title="Select Training Image Directory")
        if d:
            self._dir_var.set(d)
            self._update_image_count(Path(d))

    def _update_image_count(self, directory: Path) -> None:
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        count = sum(1 for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in exts)
        self._count_label.configure(text=f"Images: {count}")

    # ==================================================================
    # Training lifecycle
    # ==================================================================

    def _start_training(self) -> None:
        train_dir = Path(self._dir_var.get())
        if not train_dir.exists():
            messagebox.showerror("Error", f"Directory not found:\n{train_dir}")
            return

        self.config.train_image_dir = train_dir
        self._train_losses.clear()
        self._val_losses.clear()
        self._pipeline = TrainingPipeline(self.config)

        self._start_btn.configure(state=tk.DISABLED)
        self._stop_btn.configure(state=tk.NORMAL)
        self._save_btn.configure(state=tk.DISABLED)

        self._log("Training started...")

        self._train_thread = threading.Thread(target=self._train_worker, daemon=True)
        self._train_thread.start()

    def _stop_training(self) -> None:
        if self._pipeline is not None:
            self._pipeline.request_stop()
            self._log("Stop requested -- finishing current epoch...")
            self._stop_btn.configure(state=tk.DISABLED)

    def _train_worker(self) -> None:
        """Runs in a background thread."""
        try:
            result = self._pipeline.run(progress_callback=self._progress_cb)
            self._queue.put(("done", result))
        except Exception as exc:
            self._queue.put(("error", str(exc)))

    def _progress_cb(self, info: Dict[str, Any]) -> None:
        """Called from the training thread; pushes data into the queue."""
        self._queue.put(("progress", info))

    # ==================================================================
    # Queue polling (UI thread)
    # ==================================================================

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
        self.after(self.POLL_INTERVAL_MS, self._poll_queue)

    def _handle_progress(self, info: Dict[str, Any]) -> None:
        epoch = info["epoch"]
        total = info["total_epochs"]
        tl = info["train_loss"]
        vl = info["val_loss"]

        self._epoch_var.set(f"Epoch: {epoch} / {total}")
        self._progress["maximum"] = total
        self._progress["value"] = epoch

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
        self._save_btn.configure(state=tk.NORMAL)
        self._log(
            f"Training complete.  best_val_loss={result['best_val_loss']:.6f}  "
            f"threshold={result['threshold']:.6f}"
        )
        self._log(f"Checkpoint saved: {result['checkpoint_path']}")

    def _handle_error(self, msg: str) -> None:
        self._start_btn.configure(state=tk.NORMAL)
        self._stop_btn.configure(state=tk.DISABLED)
        self._log(f"ERROR: {msg}")
        messagebox.showerror("Training Error", msg)

    # ==================================================================
    # Plot update
    # ==================================================================

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

    # ==================================================================
    # Checkpoint save / load
    # ==================================================================

    def _save_checkpoint(self) -> None:
        if self._pipeline is None or self._pipeline.model is None:
            messagebox.showwarning("Warning", "No model to save.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".pt",
            filetypes=[("PyTorch checkpoint", "*.pt")],
            initialdir=str(self.config.checkpoint_dir),
        )
        if path:
            self._pipeline.save_checkpoint(Path(path), epoch=len(self._train_losses), loss=self._val_losses[-1] if self._val_losses else 0.0)
            self._log(f"Checkpoint saved: {path}")

    def _load_checkpoint(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("PyTorch checkpoint", "*.pt")],
            initialdir=str(self.config.checkpoint_dir),
        )
        if not path:
            return
        try:
            model, cfg, state = TrainingPipeline.load_checkpoint(Path(path), self.config.device)
            self._pipeline = TrainingPipeline(cfg)
            self._pipeline.model = model
            self._log(f"Checkpoint loaded: {path}  (epoch {state.get('epoch', '?')})")
            self._save_btn.configure(state=tk.NORMAL)
        except Exception as exc:
            messagebox.showerror("Load Error", str(exc))

    # ==================================================================
    # Log helper
    # ==================================================================

    def _log(self, message: str) -> None:
        self._log_text.configure(state=tk.NORMAL)
        self._log_text.insert(tk.END, message + "\n")
        self._log_text.see(tk.END)
        self._log_text.configure(state=tk.DISABLED)
