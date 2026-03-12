"""Inspection tab for the DL Anomaly Detector GUI.

Provides single-image and batch-mode inspection, a 2x2 result viewer
(original / reconstruction / heatmap / defect mask), defect statistics,
a region-details table, and navigation through batch results.

All heavy work is offloaded to a background thread with ``queue.Queue``
+ ``root.after(100)`` polling.
"""

from __future__ import annotations

import logging
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import List, Optional

import numpy as np

from dl_anomaly.config import Config
from dl_anomaly.gui.image_viewer import ImageViewer
from dl_anomaly.pipeline.inference import InferencePipeline, InspectionResult
from dl_anomaly.visualization.heatmap import create_error_heatmap, create_defect_overlay
from dl_anomaly.visualization.report import save_result_image

logger = logging.getLogger(__name__)


class InspectTab(ttk.Frame):
    """Inspection tab: load a checkpoint, inspect images, browse results."""

    POLL_INTERVAL_MS = 100

    def __init__(self, master: tk.Misc, config: Config, **kwargs) -> None:
        super().__init__(master, **kwargs)
        self.config = config
        self._pipeline: Optional[InferencePipeline] = None
        self._results: List[InspectionResult] = []
        self._result_paths: List[str] = []
        self._current_idx: int = 0
        self._queue: queue.Queue = queue.Queue()

        self._build_ui()
        self._poll_queue()

    # ==================================================================
    # UI layout
    # ==================================================================

    def _build_ui(self) -> None:
        # --- Top: checkpoint + mode -----------------------------------
        top = ttk.Frame(self, padding=4)
        top.pack(fill=tk.X, padx=6, pady=(6, 2))

        ttk.Label(top, text="Checkpoint:").pack(side=tk.LEFT)
        self._ckpt_var = tk.StringVar()
        ttk.Entry(top, textvariable=self._ckpt_var, width=50).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Load", command=self._load_checkpoint).pack(side=tk.LEFT, padx=2)

        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        self._mode_var = tk.StringVar(value="single")
        ttk.Radiobutton(top, text="Single", variable=self._mode_var, value="single").pack(side=tk.LEFT)
        ttk.Radiobutton(top, text="Batch", variable=self._mode_var, value="batch").pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Inspect", command=self._run_inspection).pack(side=tk.LEFT, padx=6)

        # --- Middle: 2x2 viewer --------------------------------------
        viewer_frame = ttk.Frame(self)
        viewer_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=2)

        self._viewers: dict[str, ImageViewer] = {}
        labels = [("Original", 0, 0), ("Reconstruction", 0, 1),
                  ("Error Heatmap", 1, 0), ("Defect Mask", 1, 1)]
        for label, r, c in labels:
            frame = ttk.LabelFrame(viewer_frame, text=label, padding=2)
            frame.grid(row=r, column=c, sticky="nsew", padx=2, pady=2)
            viewer_frame.rowconfigure(r, weight=1)
            viewer_frame.columnconfigure(c, weight=1)
            v = ImageViewer(frame, width=300, height=240)
            v.pack(fill=tk.BOTH, expand=True)
            self._viewers[label] = v

        # --- Bottom: stats + navigation -------------------------------
        bot = ttk.Frame(self, padding=4)
        bot.pack(fill=tk.X, padx=6, pady=(2, 6))

        # Result label
        self._result_var = tk.StringVar(value="Result: --")
        result_lbl = ttk.Label(bot, textvariable=self._result_var, font=("Segoe UI", 11, "bold"))
        result_lbl.pack(side=tk.LEFT, padx=4)

        self._score_var = tk.StringVar(value="Score: --")
        ttk.Label(bot, textvariable=self._score_var).pack(side=tk.LEFT, padx=8)

        self._regions_var = tk.StringVar(value="Regions: --")
        ttk.Label(bot, textvariable=self._regions_var).pack(side=tk.LEFT, padx=8)

        # Navigation
        nav = ttk.Frame(bot)
        nav.pack(side=tk.RIGHT)
        self._prev_btn = ttk.Button(nav, text="<< Prev", command=self._prev, state=tk.DISABLED)
        self._prev_btn.pack(side=tk.LEFT, padx=2)
        self._idx_var = tk.StringVar(value="0 / 0")
        ttk.Label(nav, textvariable=self._idx_var).pack(side=tk.LEFT, padx=4)
        self._next_btn = ttk.Button(nav, text="Next >>", command=self._next, state=tk.DISABLED)
        self._next_btn.pack(side=tk.LEFT, padx=2)
        ttk.Button(nav, text="Save", command=self._save_current).pack(side=tk.LEFT, padx=6)

        # Progress
        self._progress = ttk.Progressbar(bot, length=160, mode="determinate")
        self._progress.pack(side=tk.RIGHT, padx=8)

        # --- Defect details table ------------------------------------
        table_frame = ttk.LabelFrame(self, text="Defect Regions", padding=4)
        table_frame.pack(fill=tk.X, padx=6, pady=(0, 6))

        cols = ("ID", "X", "Y", "W", "H", "Area")
        self._tree = ttk.Treeview(table_frame, columns=cols, show="headings", height=4)
        for c in cols:
            self._tree.heading(c, text=c)
            self._tree.column(c, width=60, anchor=tk.CENTER)
        self._tree.pack(fill=tk.X, expand=True)

    # ==================================================================
    # Checkpoint loading
    # ==================================================================

    def _load_checkpoint(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("PyTorch checkpoint", "*.pt")],
            initialdir=str(self.config.checkpoint_dir),
        )
        if not path:
            return
        try:
            self._pipeline = InferencePipeline(path, device=self.config.device)
            self._ckpt_var.set(path)
        except Exception as exc:
            messagebox.showerror("Load Error", str(exc))

    # ==================================================================
    # Inspection dispatch
    # ==================================================================

    def _run_inspection(self) -> None:
        if self._pipeline is None:
            messagebox.showwarning("Warning", "Load a checkpoint first.")
            return

        if self._mode_var.get() == "single":
            path = filedialog.askopenfilename(
                filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")],
            )
            if not path:
                return
            self._result_paths = [path]
            t = threading.Thread(target=self._inspect_single_worker, args=(path,), daemon=True)
            t.start()
        else:
            d = filedialog.askdirectory(title="Select Image Directory")
            if not d:
                return
            self._result_paths = []  # will be populated
            t = threading.Thread(target=self._inspect_batch_worker, args=(d,), daemon=True)
            t.start()

    def _inspect_single_worker(self, path: str) -> None:
        try:
            result = self._pipeline.inspect_single(path)
            self._queue.put(("single_done", [result], [path]))
        except Exception as exc:
            self._queue.put(("error", str(exc)))

    def _inspect_batch_worker(self, directory: str) -> None:
        try:
            exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
            paths = sorted(
                str(p) for p in Path(directory).rglob("*")
                if p.is_file() and p.suffix.lower() in exts
            )

            def cb(cur: int, total: int) -> None:
                self._queue.put(("batch_progress", (cur, total)))

            results = self._pipeline.inspect_batch(directory, progress_callback=cb)
            self._queue.put(("single_done", results, paths))
        except Exception as exc:
            self._queue.put(("error", str(exc)))

    # ==================================================================
    # Queue polling
    # ==================================================================

    def _poll_queue(self) -> None:
        try:
            while True:
                tag, *data = self._queue.get_nowait()
                if tag == "single_done":
                    results, paths = data
                    self._results = results
                    self._result_paths = paths
                    self._current_idx = 0
                    self._show_result()
                    self._update_nav()
                elif tag == "batch_progress":
                    cur, total = data[0]
                    self._progress["maximum"] = total
                    self._progress["value"] = cur
                elif tag == "error":
                    messagebox.showerror("Inspection Error", data[0])
        except queue.Empty:
            pass
        self.after(self.POLL_INTERVAL_MS, self._poll_queue)

    # ==================================================================
    # Result display
    # ==================================================================

    def _show_result(self) -> None:
        if not self._results:
            return
        r = self._results[self._current_idx]

        # Prepare images
        orig = r.original if r.original.ndim == 3 else np.stack([r.original] * 3, axis=-1)
        recon = r.reconstruction if r.reconstruction.ndim == 3 else np.stack([r.reconstruction] * 3, axis=-1)
        heatmap = create_error_heatmap(r.error_map)
        mask_rgb = np.stack([r.defect_mask] * 3, axis=-1)

        self._viewers["Original"].set_image(orig)
        self._viewers["Reconstruction"].set_image(recon)
        self._viewers["Error Heatmap"].set_image(heatmap)
        self._viewers["Defect Mask"].set_image(mask_rgb)

        # Stats
        label = "DEFECTIVE" if r.is_defective else "PASS"
        self._result_var.set(f"Result: {label}")
        self._score_var.set(f"Score: {r.anomaly_score:.6f}")
        self._regions_var.set(f"Regions: {len(r.defect_regions)}")

        # Table
        for item in self._tree.get_children():
            self._tree.delete(item)
        for reg in r.defect_regions:
            x, y, w, h = reg["bbox"]
            self._tree.insert("", tk.END, values=(reg["id"], x, y, w, h, reg["area"]))

    # ==================================================================
    # Navigation
    # ==================================================================

    def _update_nav(self) -> None:
        n = len(self._results)
        self._idx_var.set(f"{self._current_idx + 1} / {n}")
        self._prev_btn.configure(state=tk.NORMAL if self._current_idx > 0 else tk.DISABLED)
        self._next_btn.configure(state=tk.NORMAL if self._current_idx < n - 1 else tk.DISABLED)

    def _prev(self) -> None:
        if self._current_idx > 0:
            self._current_idx -= 1
            self._show_result()
            self._update_nav()

    def _next(self) -> None:
        if self._current_idx < len(self._results) - 1:
            self._current_idx += 1
            self._show_result()
            self._update_nav()

    def _save_current(self) -> None:
        if not self._results:
            return
        r = self._results[self._current_idx]
        orig_path = self._result_paths[self._current_idx] if self._current_idx < len(self._result_paths) else "unknown"
        try:
            out = save_result_image(r, orig_path, self.config.results_dir)
            messagebox.showinfo("Saved", f"Result saved to:\n{out}")
        except Exception as exc:
            messagebox.showerror("Save Error", str(exc))
