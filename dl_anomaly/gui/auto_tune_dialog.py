"""gui/auto_tune_dialog.py - Automatic threshold calibration dialog."""
from __future__ import annotations

import logging
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Optional

import numpy as np

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

_BG = "#2b2b2b"
_BG_MEDIUM = "#3c3c3c"
_FG = "#e0e0e0"
_ACCENT = "#0078d4"
_ACTIVE_BG = "#3a3a5c"

_DL_METHODS = {"autoencoder", "patchcore", "teacher_student", "normalizing_flow", "unet"}


class AutoTuneDialog(tk.Toplevel):
    def __init__(self, master, set_status: Callable[[str], None]):
        super().__init__(master)
        self.title("\u81ea\u52d5\u95be\u503c\u6821\u6e96 (Auto-Tune)")
        self.configure(bg=_BG)
        self.resizable(False, False)
        self.transient(master)
        self.grab_set()
        self._set_status = set_status
        self._queue: queue.Queue = queue.Queue()
        self._build_ui()
        self._poll_queue()
        # Center on parent
        self.update_idletasks()
        x = master.winfo_x() + (master.winfo_width() - self.winfo_width()) // 2
        y = master.winfo_y() + (master.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{x}+{y}")
        self.protocol("WM_DELETE_WINDOW", self._close)

    def _build_ui(self):
        main = tk.Frame(self, bg=_BG, padx=12, pady=8)
        main.pack(fill=tk.BOTH, expand=True)

        # -- Method selection --
        row_method = tk.Frame(main, bg=_BG)
        row_method.pack(fill=tk.X, pady=4)
        tk.Label(
            row_method, text="\u6aa2\u6e2c\u65b9\u6cd5:", bg=_BG, fg=_FG,
            font=("", 10), width=12, anchor=tk.E,
        ).pack(side=tk.LEFT)
        self._method_var = tk.StringVar(value="patchcore")
        methods = [
            "autoencoder", "patchcore", "teacher_student",
            "normalizing_flow", "unet", "difference",
        ]
        self._method_combo = ttk.Combobox(
            row_method, textvariable=self._method_var,
            values=methods, state="readonly", width=20,
        )
        self._method_combo.pack(side=tk.LEFT, padx=4)
        self._method_combo.bind("<<ComboboxSelected>>", self._on_method_change)

        # -- Checkpoint path (DL methods) --
        self._cp_frame = tk.Frame(main, bg=_BG)
        self._cp_frame.pack(fill=tk.X, pady=4)
        tk.Label(
            self._cp_frame, text="\u6a21\u578b\u8def\u5f91:", bg=_BG, fg=_FG,
            font=("", 10), width=12, anchor=tk.E,
        ).pack(side=tk.LEFT)
        self._cp_var = tk.StringVar()
        tk.Entry(
            self._cp_frame, textvariable=self._cp_var,
            bg=_BG_MEDIUM, fg=_FG, insertbackground=_FG, width=35, relief=tk.FLAT,
        ).pack(side=tk.LEFT, padx=4)
        tk.Button(
            self._cp_frame, text="\u700f\u89bd", bg=_BG_MEDIUM, fg=_FG,
            relief=tk.FLAT, command=self._browse_checkpoint,
        ).pack(side=tk.LEFT)

        # -- Reference image (difference method) --
        self._ref_frame = tk.Frame(main, bg=_BG)
        self._ref_frame.pack(fill=tk.X, pady=4)
        tk.Label(
            self._ref_frame, text="\u53c3\u8003\u5f71\u50cf:", bg=_BG, fg=_FG,
            font=("", 10), width=12, anchor=tk.E,
        ).pack(side=tk.LEFT)
        self._ref_var = tk.StringVar()
        tk.Entry(
            self._ref_frame, textvariable=self._ref_var,
            bg=_BG_MEDIUM, fg=_FG, insertbackground=_FG, width=35, relief=tk.FLAT,
        ).pack(side=tk.LEFT, padx=4)
        tk.Button(
            self._ref_frame, text="\u700f\u89bd", bg=_BG_MEDIUM, fg=_FG,
            relief=tk.FLAT, command=self._browse_ref,
        ).pack(side=tk.LEFT)
        self._ref_frame.pack_forget()  # Hidden by default

        # -- OK directory --
        row_ok = tk.Frame(main, bg=_BG)
        row_ok.pack(fill=tk.X, pady=4)
        tk.Label(
            row_ok, text="OK \u6a23\u672c\u76ee\u9304:", bg=_BG, fg=_FG,
            font=("", 10), width=12, anchor=tk.E,
        ).pack(side=tk.LEFT)
        self._ok_dir_var = tk.StringVar()
        tk.Entry(
            row_ok, textvariable=self._ok_dir_var,
            bg=_BG_MEDIUM, fg=_FG, insertbackground=_FG, width=35, relief=tk.FLAT,
        ).pack(side=tk.LEFT, padx=4)
        tk.Button(
            row_ok, text="\u700f\u89bd", bg=_BG_MEDIUM, fg=_FG,
            relief=tk.FLAT, command=lambda: self._browse_dir(self._ok_dir_var),
        ).pack(side=tk.LEFT)

        # -- NG directory --
        row_ng = tk.Frame(main, bg=_BG)
        row_ng.pack(fill=tk.X, pady=4)
        tk.Label(
            row_ng, text="NG \u6a23\u672c\u76ee\u9304:", bg=_BG, fg=_FG,
            font=("", 10), width=12, anchor=tk.E,
        ).pack(side=tk.LEFT)
        self._ng_dir_var = tk.StringVar()
        tk.Entry(
            row_ng, textvariable=self._ng_dir_var,
            bg=_BG_MEDIUM, fg=_FG, insertbackground=_FG, width=35, relief=tk.FLAT,
        ).pack(side=tk.LEFT, padx=4)
        tk.Button(
            row_ng, text="\u700f\u89bd", bg=_BG_MEDIUM, fg=_FG,
            relief=tk.FLAT, command=lambda: self._browse_dir(self._ng_dir_var),
        ).pack(side=tk.LEFT)

        # -- Metric + thresholds --
        row_params = tk.Frame(main, bg=_BG)
        row_params.pack(fill=tk.X, pady=4)
        tk.Label(
            row_params, text="\u512a\u5316\u6307\u6a19:", bg=_BG, fg=_FG,
            font=("", 10), width=12, anchor=tk.E,
        ).pack(side=tk.LEFT)
        self._metric_var = tk.StringVar(value="f1")
        ttk.Combobox(
            row_params, textvariable=self._metric_var,
            values=["f1", "precision", "recall", "balanced_accuracy", "youden_j"],
            state="readonly", width=18,
        ).pack(side=tk.LEFT, padx=4)
        tk.Label(
            row_params, text="\u95be\u503c\u6578\u91cf:", bg=_BG, fg=_FG, font=("", 10),
        ).pack(side=tk.LEFT, padx=(12, 0))
        self._n_thresh_var = tk.IntVar(value=200)
        ttk.Spinbox(
            row_params, from_=10, to=1000,
            textvariable=self._n_thresh_var, width=6,
        ).pack(side=tk.LEFT, padx=4)

        # -- Start button + progress --
        btn_frame = tk.Frame(main, bg=_BG)
        btn_frame.pack(fill=tk.X, pady=(8, 4))
        self._start_btn = tk.Button(
            btn_frame, text="\u958b\u59cb\u6821\u6e96", bg=_ACCENT, fg="#ffffff",
            activebackground="#005a9e", font=("", 10, "bold"),
            relief=tk.FLAT, padx=20, pady=4, command=self._start_tune,
        )
        self._start_btn.pack(side=tk.LEFT)
        self._progress = ttk.Progressbar(btn_frame, length=250, mode="determinate")
        self._progress.pack(side=tk.LEFT, padx=12)

        # -- Results --
        result_frame = tk.LabelFrame(
            main, text=" \u6821\u6e96\u7d50\u679c ", bg=_BG, fg=_FG,
            font=("", 10, "bold"), padx=8, pady=6,
        )
        result_frame.pack(fill=tk.X, pady=(8, 4))
        self._result_var = tk.StringVar(value="(\u5c1a\u672a\u57f7\u884c)")
        tk.Label(
            result_frame, textvariable=self._result_var,
            bg=_BG, fg="#88cc88", font=(_MONO_FAMILY, 10),
            anchor=tk.W, justify=tk.LEFT,
        ).pack(fill=tk.X)

        # -- Close button --
        tk.Button(
            main, text="\u95dc\u9589", bg=_BG_MEDIUM, fg=_FG,
            activebackground=_ACTIVE_BG, relief=tk.FLAT,
            padx=20, pady=4, command=self._close,
        ).pack(anchor=tk.E, pady=(4, 0))

    # ------------------------------------------------------------------ #
    #  Method change handler                                               #
    # ------------------------------------------------------------------ #

    def _on_method_change(self, _event=None):
        method = self._method_var.get()
        if method in _DL_METHODS:
            self._cp_frame.pack(fill=tk.X, pady=4, after=self._method_combo.master)
            self._ref_frame.pack_forget()
        elif method == "difference":
            self._cp_frame.pack_forget()
            self._ref_frame.pack(fill=tk.X, pady=4, after=self._method_combo.master)
        else:
            self._cp_frame.pack_forget()
            self._ref_frame.pack_forget()

    # ------------------------------------------------------------------ #
    #  Browse helpers                                                       #
    # ------------------------------------------------------------------ #

    def _browse_checkpoint(self):
        p = filedialog.askopenfilename(
            title="\u9078\u64c7\u6a21\u578b\u6a94",
            filetypes=[("\u6a21\u578b\u6a94", "*.pt *.pth *.onnx *.npz"), ("\u6240\u6709\u6a94\u6848", "*")],
            parent=self,
        )
        if p:
            self._cp_var.set(p)

    def _browse_ref(self):
        p = filedialog.askopenfilename(
            title="\u9078\u64c7\u53c3\u8003\u5f71\u50cf",
            filetypes=[("\u5716\u7247", "*.png *.jpg *.jpeg *.bmp *.tiff"), ("\u6240\u6709\u6a94\u6848", "*")],
            parent=self,
        )
        if p:
            self._ref_var.set(p)

    def _browse_dir(self, var: tk.StringVar):
        d = filedialog.askdirectory(title="\u9078\u64c7\u76ee\u9304", parent=self)
        if d:
            var.set(d)

    # ------------------------------------------------------------------ #
    #  Tuning logic                                                        #
    # ------------------------------------------------------------------ #

    def _start_tune(self):
        method = self._method_var.get()
        ok_dir = self._ok_dir_var.get()
        ng_dir = self._ng_dir_var.get()
        if not ok_dir or not ng_dir:
            messagebox.showwarning(
                "\u8b66\u544a", "\u8acb\u9078\u64c7 OK \u548c NG \u6a23\u672c\u76ee\u9304\u3002", parent=self,
            )
            return
        if not Path(ok_dir).is_dir() or not Path(ng_dir).is_dir():
            messagebox.showerror("\u932f\u8aa4", "\u76ee\u9304\u4e0d\u5b58\u5728\u3002", parent=self)
            return

        self._start_btn.configure(state=tk.DISABLED)
        self._result_var.set("\u6821\u6e96\u4e2d...")
        self._set_status("\u81ea\u52d5\u95be\u503c\u6821\u6e96\u4e2d...")

        threading.Thread(target=self._tune_worker, daemon=True).start()

    def _tune_worker(self):
        try:
            from shared.core.auto_tune import AutoTuner

            method = self._method_var.get()
            scorer = self._build_scorer(method)
            tuner = AutoTuner(scorer)

            def progress_cb(current: int, total: int):
                self._queue.put(("progress", current, total))

            result = tuner.tune(
                self._ok_dir_var.get(),
                self._ng_dir_var.get(),
                n_thresholds=self._n_thresh_var.get(),
                metric=self._metric_var.get(),
                progress_callback=progress_cb,
            )
            self._queue.put(("done", result))
        except Exception as exc:
            logger.exception("Auto-tune failed")
            self._queue.put(("error", str(exc)))

    def _build_scorer(self, method: str):
        """Build a scorer callable for the chosen method."""
        import cv2

        if method == "patchcore":
            from shared.core.patchcore import PatchCoreModel, PatchCoreInference

            model = PatchCoreModel.load(self._cp_var.get())
            inference = PatchCoreInference(model)
            return lambda img: inference.score_image(img)[0]

        elif method == "autoencoder":
            from dl_anomaly.pipeline.inference import InferencePipeline

            pipeline = InferencePipeline(self._cp_var.get())

            def ae_scorer(img):
                import os
                import tempfile

                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                    cv2.imwrite(f.name, img)
                    tmp = f.name
                try:
                    r = pipeline.inspect_single(tmp)
                    return float(r.anomaly_score)
                finally:
                    os.unlink(tmp)

            return ae_scorer

        elif method == "teacher_student":
            from shared.core.teacher_student import TeacherStudentInference, load_model

            model = load_model(self._cp_var.get())
            inference = TeacherStudentInference(model)
            return lambda img: inference.score_image(img)[0]

        elif method == "normalizing_flow":
            from shared.core.normalizing_flow import NormFlowInference, load_model

            model = load_model(self._cp_var.get())
            inference = NormFlowInference(model)
            return lambda img: inference.score_image(img)[0]

        elif method == "unet":
            from shared.core.unet_segment import UNetInference, load_model

            model = load_model(self._cp_var.get())
            inference = UNetInference(model)
            return lambda img: inference.segment(img)[0]

        elif method == "difference":
            from shared.core.image_difference import ImageDifferencer

            ref = cv2.imread(self._ref_var.get())
            if ref is None:
                raise FileNotFoundError(f"Cannot read reference: {self._ref_var.get()}")
            differ = ImageDifferencer(registration_method="ecc")
            differ.set_reference(ref)
            return lambda img: differ.detect(img)["anomaly_score"]

        else:
            raise ValueError(f"Unknown method: {method}")

    # ------------------------------------------------------------------ #
    #  Queue polling                                                       #
    # ------------------------------------------------------------------ #

    def _poll_queue(self):
        try:
            while True:
                msg = self._queue.get_nowait()
                if msg[0] == "progress":
                    _, current, total = msg
                    self._progress["maximum"] = total
                    self._progress["value"] = current
                elif msg[0] == "done":
                    result = msg[1]
                    self._start_btn.configure(state=tk.NORMAL)
                    self._progress["value"] = self._progress["maximum"]
                    text = (
                        f"\u6700\u4f73\u95be\u503c: {result.optimal_threshold:.6f}\n"
                        f"F1: {result.f1:.4f}  Precision: {result.precision:.4f}  "
                        f"Recall: {result.recall:.4f}\n"
                        f"ROC AUC: {result.roc_auc:.4f}"
                    )
                    self._result_var.set(text)
                    self._set_status(
                        f"\u6821\u6e96\u5b8c\u6210: threshold={result.optimal_threshold:.6f}, "
                        f"F1={result.f1:.4f}"
                    )
                elif msg[0] == "error":
                    self._start_btn.configure(state=tk.NORMAL)
                    self._result_var.set(f"\u932f\u8aa4: {msg[1]}")
                    self._set_status("\u81ea\u52d5\u6821\u6e96\u5931\u6557")
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    # ------------------------------------------------------------------ #
    #  Close                                                               #
    # ------------------------------------------------------------------ #

    def _close(self):
        self.grab_release()
        self.destroy()
