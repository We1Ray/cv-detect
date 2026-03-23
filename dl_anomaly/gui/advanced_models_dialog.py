"""
gui/advanced_models_dialog.py - Advanced AI model management dialog.

Provides a tabbed dialog for:
1. PatchCore anomaly detection (training & inference)
2. ONNX model management (export, import, inference)
3. Model comparison (side-by-side evaluation)

All heavy computation runs in background threads; UI updates are
delivered via a ``queue.Queue`` polled by ``root.after(100, ...)``.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable, Dict, List, Optional, Tuple

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

# --------------------------------------------------------------------------- #
# Theme constants
# --------------------------------------------------------------------------- #
_BG = "#2b2b2b"
_BG_MEDIUM = "#3c3c3c"
_FG = "#e0e0e0"
_FG_DIM = "#cccccc"
_ACCENT = "#0078d4"
_ACTIVE_BG = "#3a3a5c"
_CANVAS_BG = "#1e1e1e"
_LOG_BG = "#1a1a1a"
_LOG_FG = "#c8c8c8"


# --------------------------------------------------------------------------- #
# Helper: detect CUDA availability
# --------------------------------------------------------------------------- #

def _detect_device() -> str:
    """Return 'cuda' if torch + CUDA available, else 'cpu'."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass  # optional dependency: torch not installed
    return "cpu"


# =========================================================================== #
#  AdvancedModelsDialog                                                        #
# =========================================================================== #


class AdvancedModelsDialog(tk.Toplevel):
    """Advanced AI model management dialog with PatchCore, ONNX, and
    model comparison tabs.

    Parameters
    ----------
    master : tk.Widget
        Parent widget.
    config : Config
        Application configuration instance.
    get_current_image : callable
        Returns the current image as ``np.ndarray`` or ``None``.
    add_pipeline_step : callable
        ``add_pipeline_step(name, array, op_meta=None)``
    set_status : callable
        ``set_status(text)`` to update status bar.
    """

    POLL_INTERVAL_MS = 100

    def __init__(
        self,
        master: tk.Widget,
        config: Any,
        get_current_image: Callable[[], Optional[np.ndarray]],
        add_pipeline_step: Callable[..., Any],
        set_status: Callable[[str], None],
    ) -> None:
        super().__init__(master)
        self.title("進階模型管理")
        self.geometry("900x700")
        self.resizable(True, True)
        self.configure(bg=_BG)
        self.transient(master)

        self._config = config
        self._get_current_image = get_current_image
        self._add_pipeline_step = add_pipeline_step
        self._set_status = set_status

        # ----- state -----
        self._patchcore_model: Any = None
        self._patchcore_trainer: Any = None
        self._onnx_engine: Any = None
        self._training_thread: Optional[threading.Thread] = None
        self._stop_training_flag = threading.Event()

        # Thread-safe queue for UI updates
        self._queue: queue.Queue = queue.Queue()

        # Loaded model registry for comparison tab
        self._loaded_models: Dict[str, Dict[str, Any]] = {}

        # PhotoImage references (prevent GC)
        self._photo_refs: List[Any] = []

        self._build_ui()

        # Start polling queue
        self.after(self.POLL_INTERVAL_MS, self._poll_queue)

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ================================================================== #
    # Queue polling                                                        #
    # ================================================================== #

    def _poll_queue(self) -> None:
        """Poll the message queue and dispatch UI updates."""
        try:
            while True:
                msg = self._queue.get_nowait()
                self._handle_queue_message(msg)
        except queue.Empty:
            pass  # no messages pending, expected during polling
        self.after(self.POLL_INTERVAL_MS, self._poll_queue)

    def _handle_queue_message(self, msg: Dict[str, Any]) -> None:
        tag = msg.get("tag", "")

        if tag == "log":
            self._append_log(msg.get("text", ""))
        elif tag == "progress":
            value = msg.get("value", 0)
            maximum = msg.get("maximum", 100)
            self._pc_progress["maximum"] = maximum
            self._pc_progress["value"] = value
        elif tag == "training_done":
            self._on_training_done(msg)
        elif tag == "training_error":
            self._on_training_error(msg.get("error", "Unknown error"))
        elif tag == "model_loaded":
            self._on_model_loaded(msg)
        elif tag == "inference_result":
            self._on_inference_result(msg)
        elif tag == "batch_progress":
            value = msg.get("value", 0)
            maximum = msg.get("maximum", 100)
            tab = msg.get("tab", "patchcore")
            if tab == "onnx":
                self._onnx_progress["maximum"] = maximum
                self._onnx_progress["value"] = value
            else:
                self._pc_progress["maximum"] = maximum
                self._pc_progress["value"] = value
        elif tag == "batch_done":
            self._append_log(msg.get("text", "批次處理完成"))
        elif tag == "onnx_export_done":
            self._onnx_status_var.set(msg.get("text", "匯出完成"))
        elif tag == "onnx_info":
            self._update_onnx_info(msg)
        elif tag == "compare_result":
            self._on_compare_result(msg)

    # ================================================================== #
    # UI construction                                                      #
    # ================================================================== #

    def _build_ui(self) -> None:
        """Build the tabbed notebook UI."""
        style = ttk.Style(self)
        style.configure("Dark.TNotebook", background=_BG)
        style.configure("Dark.TNotebook.Tab", background=_BG_MEDIUM,
                        foreground=_FG, padding=[12, 4])
        style.map("Dark.TNotebook.Tab",
                  background=[("selected", _ACTIVE_BG)],
                  foreground=[("selected", "#ffffff")])
        style.configure("Dark.TFrame", background=_BG)
        style.configure("Dark.TLabelframe", background=_BG, foreground=_FG)
        style.configure("Dark.TLabelframe.Label", background=_BG,
                        foreground=_FG)
        style.configure("Dark.TLabel", background=_BG, foreground=_FG)
        style.configure("Dark.TButton", background=_BG_MEDIUM, foreground=_FG)
        style.configure("Dark.TCheckbutton", background=_BG, foreground=_FG)

        self._notebook = ttk.Notebook(self, style="Dark.TNotebook")
        self._notebook.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        self._build_patchcore_tab()
        self._build_onnx_tab()
        self._build_compare_tab()

    # ================================================================== #
    # Tab 1: PatchCore                                                     #
    # ================================================================== #

    def _build_patchcore_tab(self) -> None:
        tab = tk.Frame(self._notebook, bg=_BG)
        self._notebook.add(tab, text="PatchCore 異常偵測")

        # Split into left (settings) and right (inference)
        paned = tk.PanedWindow(tab, orient=tk.HORIZONTAL, bg=_BG,
                               sashwidth=4, sashrelief=tk.FLAT)
        paned.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        left = tk.Frame(paned, bg=_BG)
        right = tk.Frame(paned, bg=_BG)
        paned.add(left, minsize=380)
        paned.add(right, minsize=380)

        # ---- Left column: Training settings ----
        settings_lf = tk.LabelFrame(left, text="訓練設定", bg=_BG, fg=_FG,
                                    font=("", 10, "bold"))
        settings_lf.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        row = 0
        # Backbone
        tk.Label(settings_lf, text="Backbone 選擇:", bg=_BG, fg=_FG,
                 anchor=tk.W).grid(row=row, column=0, sticky=tk.W, padx=6, pady=3)
        self._backbone_var = tk.StringVar(value="wide_resnet50_2")
        backbone_cb = ttk.Combobox(
            settings_lf, textvariable=self._backbone_var, state="readonly",
            values=["resnet18", "resnet50", "wide_resnet50_2", "efficientnet_b0"],
            width=22,
        )
        backbone_cb.grid(row=row, column=1, sticky=tk.W, padx=6, pady=3)

        # Feature layers
        row += 1
        tk.Label(settings_lf, text="特徵層:", bg=_BG, fg=_FG,
                 anchor=tk.W).grid(row=row, column=0, sticky=tk.NW, padx=6, pady=3)

        layer_frame = tk.Frame(settings_lf, bg=_BG)
        layer_frame.grid(row=row, column=1, sticky=tk.W, padx=6, pady=3)

        self._layer_vars: Dict[str, tk.BooleanVar] = {}
        for layer_name in ("layer1", "layer2", "layer3", "layer4"):
            var = tk.BooleanVar(value=layer_name in ("layer2", "layer3"))
            self._layer_vars[layer_name] = var
            cb = tk.Checkbutton(
                layer_frame, text=layer_name, variable=var,
                bg=_BG, fg=_FG, selectcolor=_BG_MEDIUM,
                activebackground=_BG, activeforeground=_FG,
            )
            cb.pack(side=tk.LEFT, padx=2)

        # Coreset ratio
        row += 1
        tk.Label(settings_lf, text="Coreset 比例:", bg=_BG, fg=_FG,
                 anchor=tk.W).grid(row=row, column=0, sticky=tk.W, padx=6, pady=3)

        coreset_frame = tk.Frame(settings_lf, bg=_BG)
        coreset_frame.grid(row=row, column=1, sticky=tk.EW, padx=6, pady=3)

        self._coreset_var = tk.DoubleVar(value=0.01)
        self._coreset_scale = tk.Scale(
            coreset_frame, from_=0.001, to=0.1, resolution=0.001,
            orient=tk.HORIZONTAL, variable=self._coreset_var,
            bg=_BG, fg=_FG, troughcolor=_BG_MEDIUM,
            highlightthickness=0, sliderrelief=tk.FLAT,
            length=180,
        )
        self._coreset_scale.pack(side=tk.LEFT)

        self._coreset_label = tk.Label(
            coreset_frame, text="0.010", bg=_BG, fg=_FG,
            font=(_MONO_FAMILY, 9), width=6,
        )
        self._coreset_label.pack(side=tk.LEFT, padx=4)
        self._coreset_var.trace_add(
            "write",
            lambda *_: self._coreset_label.configure(
                text=f"{self._coreset_var.get():.3f}"
            ),
        )

        # Image size
        row += 1
        tk.Label(settings_lf, text="影像大小:", bg=_BG, fg=_FG,
                 anchor=tk.W).grid(row=row, column=0, sticky=tk.W, padx=6, pady=3)
        self._imgsize_var = tk.IntVar(value=224)
        imgsize_spin = tk.Spinbox(
            settings_lf, from_=64, to=512, increment=32,
            textvariable=self._imgsize_var, width=8,
            bg=_BG_MEDIUM, fg=_FG, buttonbackground=_BG_MEDIUM,
            insertbackground=_FG,
        )
        imgsize_spin.grid(row=row, column=1, sticky=tk.W, padx=6, pady=3)

        # Device
        row += 1
        tk.Label(settings_lf, text="裝置:", bg=_BG, fg=_FG,
                 anchor=tk.W).grid(row=row, column=0, sticky=tk.W, padx=6, pady=3)
        detected = _detect_device()
        self._device_var = tk.StringVar(value=detected)
        device_values = ["cpu", "cuda"] if detected == "cuda" else ["cpu"]
        device_cb = ttk.Combobox(
            settings_lf, textvariable=self._device_var, state="readonly",
            values=device_values, width=10,
        )
        device_cb.grid(row=row, column=1, sticky=tk.W, padx=6, pady=3)

        # Training folder
        row += 1
        tk.Label(settings_lf, text="訓練資料夾:", bg=_BG, fg=_FG,
                 anchor=tk.W).grid(row=row, column=0, sticky=tk.W, padx=6, pady=3)

        dir_frame = tk.Frame(settings_lf, bg=_BG)
        dir_frame.grid(row=row, column=1, sticky=tk.EW, padx=6, pady=3)

        self._pc_train_dir_var = tk.StringVar()
        tk.Entry(
            dir_frame, textvariable=self._pc_train_dir_var, width=20,
            bg=_BG_MEDIUM, fg=_FG, insertbackground=_FG,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(
            dir_frame, text="瀏覽...", bg=_BG_MEDIUM, fg=_FG,
            activebackground=_ACTIVE_BG, relief=tk.FLAT,
            command=self._browse_pc_train_dir,
        ).pack(side=tk.LEFT, padx=(4, 0))

        # Training buttons
        row += 1
        btn_frame = tk.Frame(settings_lf, bg=_BG)
        btn_frame.grid(row=row, column=0, columnspan=2, sticky=tk.EW,
                       padx=6, pady=6)

        self._pc_train_btn = tk.Button(
            btn_frame, text="訓練", bg=_ACCENT, fg="#ffffff",
            activebackground="#005a9e", activeforeground="#ffffff",
            relief=tk.FLAT, padx=16, pady=4, font=("", 10),
            command=self._start_patchcore_training,
        )
        self._pc_train_btn.pack(side=tk.LEFT, padx=(0, 6))

        self._pc_stop_btn = tk.Button(
            btn_frame, text="停止", bg=_BG_MEDIUM, fg=_FG,
            activebackground=_ACTIVE_BG, relief=tk.FLAT,
            padx=16, pady=4, font=("", 10), state=tk.DISABLED,
            command=self._stop_patchcore_training,
        )
        self._pc_stop_btn.pack(side=tk.LEFT)

        # Progress bar
        row += 1
        self._pc_progress = ttk.Progressbar(
            settings_lf, length=300, mode="determinate",
        )
        self._pc_progress.grid(row=row, column=0, columnspan=2, sticky=tk.EW,
                               padx=6, pady=(0, 6))

        settings_lf.columnconfigure(1, weight=1)

        # ---- Right column: Inference ----
        infer_lf = tk.LabelFrame(right, text="推論", bg=_BG, fg=_FG,
                                 font=("", 10, "bold"))
        infer_lf.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Load model
        load_frame = tk.Frame(infer_lf, bg=_BG)
        load_frame.pack(fill=tk.X, padx=6, pady=6)

        tk.Button(
            load_frame, text="載入模型", bg=_BG_MEDIUM, fg=_FG,
            activebackground=_ACTIVE_BG, relief=tk.FLAT,
            padx=12, pady=3, command=self._load_patchcore_model,
        ).pack(side=tk.LEFT)

        # Model info
        info_lf = tk.LabelFrame(infer_lf, text="模型資訊", bg=_BG, fg=_FG)
        info_lf.pack(fill=tk.X, padx=6, pady=(0, 6))

        self._pc_info_labels: Dict[str, tk.Label] = {}
        for label_text in ("Backbone:", "特徵維度:", "Memory Bank 大小:", "閾值:"):
            frame = tk.Frame(info_lf, bg=_BG)
            frame.pack(fill=tk.X, padx=6, pady=1)
            tk.Label(frame, text=label_text, bg=_BG, fg=_FG_DIM,
                     width=16, anchor=tk.W, font=("", 9)).pack(side=tk.LEFT)
            val_lbl = tk.Label(frame, text="--", bg=_BG, fg=_FG,
                               anchor=tk.W, font=(_MONO_FAMILY, 9))
            val_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self._pc_info_labels[label_text] = val_lbl

        # Threshold entry
        thresh_frame = tk.Frame(infer_lf, bg=_BG)
        thresh_frame.pack(fill=tk.X, padx=6, pady=4)
        tk.Label(thresh_frame, text="閾值:", bg=_BG, fg=_FG,
                 anchor=tk.W).pack(side=tk.LEFT)
        self._pc_threshold_var = tk.StringVar(value="0.5")
        tk.Entry(
            thresh_frame, textvariable=self._pc_threshold_var, width=10,
            bg=_BG_MEDIUM, fg=_FG, insertbackground=_FG,
        ).pack(side=tk.LEFT, padx=4)

        # Inference buttons
        infer_btn_frame = tk.Frame(infer_lf, bg=_BG)
        infer_btn_frame.pack(fill=tk.X, padx=6, pady=4)

        tk.Button(
            infer_btn_frame, text="檢測單張", bg=_BG_MEDIUM, fg=_FG,
            activebackground=_ACTIVE_BG, relief=tk.FLAT,
            padx=12, pady=3, command=self._patchcore_infer_single,
        ).pack(side=tk.LEFT, padx=(0, 6))

        tk.Button(
            infer_btn_frame, text="批次檢測", bg=_BG_MEDIUM, fg=_FG,
            activebackground=_ACTIVE_BG, relief=tk.FLAT,
            padx=12, pady=3, command=self._patchcore_infer_batch,
        ).pack(side=tk.LEFT, padx=(0, 6))

        tk.Button(
            infer_btn_frame, text="儲存模型", bg=_BG_MEDIUM, fg=_FG,
            activebackground=_ACTIVE_BG, relief=tk.FLAT,
            padx=12, pady=3, command=self._save_patchcore_model,
        ).pack(side=tk.LEFT)

        # ---- Bottom: Log area ----
        log_frame = tk.LabelFrame(tab, text="日誌", bg=_BG, fg=_FG)
        log_frame.pack(fill=tk.X, padx=8, pady=(0, 6))

        log_inner = tk.Frame(log_frame, bg=_BG)
        log_inner.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        self._pc_log_text = tk.Text(
            log_inner, height=6, state=tk.DISABLED, wrap=tk.WORD,
            bg=_LOG_BG, fg=_LOG_FG, font=(_MONO_FAMILY, 9),
            insertbackground=_FG, selectbackground=_ACTIVE_BG,
        )
        log_scroll = tk.Scrollbar(log_inner, command=self._pc_log_text.yview)
        self._pc_log_text.configure(yscrollcommand=log_scroll.set)
        self._pc_log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    # ================================================================== #
    # Tab 2: ONNX                                                          #
    # ================================================================== #

    def _build_onnx_tab(self) -> None:
        tab = tk.Frame(self._notebook, bg=_BG)
        self._notebook.add(tab, text="ONNX 模型管理")

        paned = tk.PanedWindow(tab, orient=tk.HORIZONTAL, bg=_BG,
                               sashwidth=4, sashrelief=tk.FLAT)
        paned.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        left = tk.Frame(paned, bg=_BG)
        right = tk.Frame(paned, bg=_BG)
        paned.add(left, minsize=380)
        paned.add(right, minsize=380)

        # ---- Left: Export ----
        export_lf = tk.LabelFrame(left, text="匯出", bg=_BG, fg=_FG,
                                  font=("", 10, "bold"))
        export_lf.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        row = 0
        tk.Label(export_lf, text="匯出來源:", bg=_BG, fg=_FG,
                 anchor=tk.W).grid(row=row, column=0, sticky=tk.W, padx=6, pady=3)
        self._export_source_var = tk.StringVar(value="目前 Autoencoder")
        export_src_cb = ttk.Combobox(
            export_lf, textvariable=self._export_source_var, state="readonly",
            values=["目前 Autoencoder", "目前 PatchCore"], width=20,
        )
        export_src_cb.grid(row=row, column=1, sticky=tk.W, padx=6, pady=3)

        row += 1
        tk.Label(export_lf, text="Opset 版本:", bg=_BG, fg=_FG,
                 anchor=tk.W).grid(row=row, column=0, sticky=tk.W, padx=6, pady=3)
        self._opset_var = tk.IntVar(value=14)
        opset_spin = tk.Spinbox(
            export_lf, from_=11, to=17, textvariable=self._opset_var,
            width=6, bg=_BG_MEDIUM, fg=_FG, buttonbackground=_BG_MEDIUM,
            insertbackground=_FG,
        )
        opset_spin.grid(row=row, column=1, sticky=tk.W, padx=6, pady=3)

        row += 1
        tk.Label(export_lf, text="匯出路徑:", bg=_BG, fg=_FG,
                 anchor=tk.W).grid(row=row, column=0, sticky=tk.W, padx=6, pady=3)

        path_frame = tk.Frame(export_lf, bg=_BG)
        path_frame.grid(row=row, column=1, sticky=tk.EW, padx=6, pady=3)

        self._export_path_var = tk.StringVar()
        tk.Entry(
            path_frame, textvariable=self._export_path_var, width=20,
            bg=_BG_MEDIUM, fg=_FG, insertbackground=_FG,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(
            path_frame, text="瀏覽...", bg=_BG_MEDIUM, fg=_FG,
            activebackground=_ACTIVE_BG, relief=tk.FLAT,
            command=self._browse_export_path,
        ).pack(side=tk.LEFT, padx=(4, 0))

        row += 1
        tk.Button(
            export_lf, text="匯出", bg=_ACCENT, fg="#ffffff",
            activebackground="#005a9e", activeforeground="#ffffff",
            relief=tk.FLAT, padx=20, pady=4, font=("", 10),
            command=self._export_onnx,
        ).grid(row=row, column=0, columnspan=2, sticky=tk.W, padx=6, pady=8)

        row += 1
        self._onnx_status_var = tk.StringVar(value="")
        tk.Label(
            export_lf, textvariable=self._onnx_status_var, bg=_BG, fg=_FG_DIM,
            anchor=tk.W, font=("", 9),
        ).grid(row=row, column=0, columnspan=2, sticky=tk.EW, padx=6, pady=2)

        export_lf.columnconfigure(1, weight=1)

        # ---- Right: Import & Run ----
        import_lf = tk.LabelFrame(right, text="載入與推論", bg=_BG, fg=_FG,
                                  font=("", 10, "bold"))
        import_lf.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Load button
        tk.Button(
            import_lf, text="載入 ONNX 模型", bg=_BG_MEDIUM, fg=_FG,
            activebackground=_ACTIVE_BG, relief=tk.FLAT,
            padx=12, pady=3, command=self._load_onnx_model,
        ).pack(anchor=tk.W, padx=6, pady=6)

        # Model info
        onnx_info_lf = tk.LabelFrame(import_lf, text="模型資訊", bg=_BG, fg=_FG)
        onnx_info_lf.pack(fill=tk.X, padx=6, pady=(0, 6))

        self._onnx_info_labels: Dict[str, tk.Label] = {}
        for label_text in ("檔案路徑:", "輸入形狀:", "輸出形狀:",
                           "執行提供者:", "Opset 版本:"):
            frame = tk.Frame(onnx_info_lf, bg=_BG)
            frame.pack(fill=tk.X, padx=6, pady=1)
            tk.Label(frame, text=label_text, bg=_BG, fg=_FG_DIM,
                     width=14, anchor=tk.W, font=("", 9)).pack(side=tk.LEFT)
            val_lbl = tk.Label(frame, text="--", bg=_BG, fg=_FG,
                               anchor=tk.W, font=(_MONO_FAMILY, 9))
            val_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self._onnx_info_labels[label_text] = val_lbl

        # Threshold
        onnx_thresh_frame = tk.Frame(import_lf, bg=_BG)
        onnx_thresh_frame.pack(fill=tk.X, padx=6, pady=4)

        tk.Label(onnx_thresh_frame, text="設定閾值:", bg=_BG, fg=_FG,
                 anchor=tk.W).pack(side=tk.LEFT)
        self._onnx_threshold_var = tk.StringVar(value="0.5")
        tk.Entry(
            onnx_thresh_frame, textvariable=self._onnx_threshold_var, width=10,
            bg=_BG_MEDIUM, fg=_FG, insertbackground=_FG,
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            onnx_thresh_frame, text="從良品計算閾值", bg=_BG_MEDIUM, fg=_FG,
            activebackground=_ACTIVE_BG, relief=tk.FLAT,
            padx=8, pady=2, command=self._compute_onnx_threshold,
        ).pack(side=tk.LEFT, padx=4)

        # Inference buttons
        onnx_infer_frame = tk.Frame(import_lf, bg=_BG)
        onnx_infer_frame.pack(fill=tk.X, padx=6, pady=4)

        tk.Button(
            onnx_infer_frame, text="推論", bg=_ACCENT, fg="#ffffff",
            activebackground="#005a9e", activeforeground="#ffffff",
            relief=tk.FLAT, padx=16, pady=3, font=("", 10),
            command=self._onnx_infer_single,
        ).pack(side=tk.LEFT, padx=(0, 6))

        tk.Button(
            onnx_infer_frame, text="批次推論", bg=_BG_MEDIUM, fg=_FG,
            activebackground=_ACTIVE_BG, relief=tk.FLAT,
            padx=12, pady=3, command=self._onnx_infer_batch,
        ).pack(side=tk.LEFT)

        # Progress bar
        self._onnx_progress = ttk.Progressbar(
            import_lf, length=300, mode="determinate",
        )
        self._onnx_progress.pack(fill=tk.X, padx=6, pady=(4, 6))

    # ================================================================== #
    # Tab 3: Model Comparison                                              #
    # ================================================================== #

    def _build_compare_tab(self) -> None:
        tab = tk.Frame(self._notebook, bg=_BG)
        self._notebook.add(tab, text="模型比較")

        # Top controls
        ctrl_frame = tk.Frame(tab, bg=_BG)
        ctrl_frame.pack(fill=tk.X, padx=8, pady=8)

        tk.Label(ctrl_frame, text="模型 A:", bg=_BG, fg=_FG,
                 font=("", 10)).grid(row=0, column=0, sticky=tk.W, padx=4, pady=3)
        self._cmp_model_a_var = tk.StringVar()
        self._cmp_model_a_cb = ttk.Combobox(
            ctrl_frame, textvariable=self._cmp_model_a_var, state="readonly",
            width=30,
        )
        self._cmp_model_a_cb.grid(row=0, column=1, sticky=tk.W, padx=4, pady=3)

        tk.Label(ctrl_frame, text="模型 B:", bg=_BG, fg=_FG,
                 font=("", 10)).grid(row=1, column=0, sticky=tk.W, padx=4, pady=3)
        self._cmp_model_b_var = tk.StringVar()
        self._cmp_model_b_cb = ttk.Combobox(
            ctrl_frame, textvariable=self._cmp_model_b_var, state="readonly",
            width=30,
        )
        self._cmp_model_b_cb.grid(row=1, column=1, sticky=tk.W, padx=4, pady=3)

        tk.Label(ctrl_frame, text="比較影像:", bg=_BG, fg=_FG,
                 font=("", 10)).grid(row=2, column=0, sticky=tk.W, padx=4, pady=3)

        img_frame = tk.Frame(ctrl_frame, bg=_BG)
        img_frame.grid(row=2, column=1, sticky=tk.W, padx=4, pady=3)

        self._cmp_source_var = tk.StringVar(value="current")
        tk.Radiobutton(
            img_frame, text="目前影像", variable=self._cmp_source_var,
            value="current", bg=_BG, fg=_FG, selectcolor=_BG_MEDIUM,
            activebackground=_BG, activeforeground=_FG,
        ).pack(side=tk.LEFT, padx=(0, 8))
        tk.Radiobutton(
            img_frame, text="資料夾", variable=self._cmp_source_var,
            value="directory", bg=_BG, fg=_FG, selectcolor=_BG_MEDIUM,
            activebackground=_BG, activeforeground=_FG,
        ).pack(side=tk.LEFT, padx=(0, 4))

        self._cmp_dir_var = tk.StringVar()
        tk.Entry(
            img_frame, textvariable=self._cmp_dir_var, width=18,
            bg=_BG_MEDIUM, fg=_FG, insertbackground=_FG,
        ).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(
            img_frame, text="瀏覽...", bg=_BG_MEDIUM, fg=_FG,
            activebackground=_ACTIVE_BG, relief=tk.FLAT,
            command=self._browse_compare_dir,
        ).pack(side=tk.LEFT)

        tk.Button(
            ctrl_frame, text="執行比較", bg=_ACCENT, fg="#ffffff",
            activebackground="#005a9e", activeforeground="#ffffff",
            relief=tk.FLAT, padx=20, pady=4, font=("", 10, "bold"),
            command=self._run_comparison,
        ).grid(row=3, column=0, columnspan=2, sticky=tk.W, padx=4, pady=8)

        # Results area
        result_frame = tk.Frame(tab, bg=_BG)
        result_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        # Score comparison labels
        score_frame = tk.Frame(result_frame, bg=_BG)
        score_frame.pack(fill=tk.X, pady=(0, 4))

        self._cmp_score_a_label = tk.Label(
            score_frame, text="模型 A 分數: --", bg=_BG, fg=_FG,
            font=(_MONO_FAMILY, 10), anchor=tk.W,
        )
        self._cmp_score_a_label.pack(side=tk.LEFT, padx=(0, 20))

        self._cmp_score_b_label = tk.Label(
            score_frame, text="模型 B 分數: --", bg=_BG, fg=_FG,
            font=(_MONO_FAMILY, 10), anchor=tk.W,
        )
        self._cmp_score_b_label.pack(side=tk.LEFT)

        # Results table
        tree_frame = tk.Frame(result_frame, bg=_BG)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        columns = ("image", "score_a", "score_b", "agree")
        self._cmp_tree = ttk.Treeview(
            tree_frame, columns=columns, show="headings", height=12,
        )
        self._cmp_tree.heading("image", text="Image")
        self._cmp_tree.heading("score_a", text="Score A")
        self._cmp_tree.heading("score_b", text="Score B")
        self._cmp_tree.heading("agree", text="Agree?")

        self._cmp_tree.column("image", width=300, anchor=tk.W)
        self._cmp_tree.column("score_a", width=120, anchor=tk.CENTER)
        self._cmp_tree.column("score_b", width=120, anchor=tk.CENTER)
        self._cmp_tree.column("agree", width=80, anchor=tk.CENTER)

        tree_scroll = tk.Scrollbar(tree_frame, command=self._cmp_tree.yview)
        self._cmp_tree.configure(yscrollcommand=tree_scroll.set)
        self._cmp_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    # ================================================================== #
    # PatchCore handlers                                                   #
    # ================================================================== #

    def _browse_pc_train_dir(self) -> None:
        d = filedialog.askdirectory(title="選擇訓練資料夾", parent=self)
        if d:
            self._pc_train_dir_var.set(d)

    def _start_patchcore_training(self) -> None:
        train_dir = self._pc_train_dir_var.get().strip()
        if not train_dir or not Path(train_dir).is_dir():
            messagebox.showerror("錯誤", "請選擇有效的訓練資料夾", parent=self)
            return

        selected_layers = [
            name for name, var in self._layer_vars.items() if var.get()
        ]
        if not selected_layers:
            messagebox.showwarning("警告", "請至少選擇一個特徵層", parent=self)
            return

        self._pc_train_btn.configure(state=tk.DISABLED)
        self._pc_stop_btn.configure(state=tk.NORMAL)
        self._pc_progress["value"] = 0
        self._stop_training_flag.clear()

        self._append_log("開始 PatchCore 訓練...")

        self._training_thread = threading.Thread(
            target=self._patchcore_train_worker,
            args=(
                train_dir,
                self._backbone_var.get(),
                selected_layers,
                self._coreset_var.get(),
                self._imgsize_var.get(),
                self._device_var.get(),
            ),
            daemon=True,
        )
        self._training_thread.start()

    def _patchcore_train_worker(
        self,
        train_dir: str,
        backbone: str,
        layers: List[str],
        coreset_ratio: float,
        image_size: int,
        device: str,
    ) -> None:
        """Runs PatchCore training in a background thread."""
        try:
            from dl_anomaly.core.patchcore import (
                PatchCoreModel,
                PatchCoreTrainer,
                FeatureExtractor,
            )
        except ImportError as exc:
            self._queue.put({
                "tag": "training_error",
                "error": (
                    f"PatchCore 模組未安裝。請確認已安裝相關依賴:\n{exc}"
                ),
            })
            return

        try:
            self._queue.put({"tag": "log", "text": f"Backbone: {backbone}"})
            self._queue.put({"tag": "log", "text": f"特徵層: {layers}"})
            self._queue.put({"tag": "log", "text": f"Coreset 比例: {coreset_ratio}"})
            self._queue.put({"tag": "log", "text": f"影像大小: {image_size}"})
            self._queue.put({"tag": "log", "text": f"裝置: {device}"})

            # Collect image files
            exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
            train_path = Path(train_dir)
            image_files = sorted(
                p for p in train_path.rglob("*")
                if p.is_file() and p.suffix.lower() in exts
            )
            if not image_files:
                self._queue.put({
                    "tag": "training_error",
                    "error": "訓練資料夾中未找到影像檔案",
                })
                return

            self._queue.put({
                "tag": "log",
                "text": f"找到 {len(image_files)} 張訓練影像",
            })

            trainer = PatchCoreTrainer(
                backbone=backbone,
                layers=layers,
                image_size=image_size,
                coreset_ratio=coreset_ratio,
                device=device,
            )
            self._patchcore_trainer = trainer

            def progress_cb(current: int, total: int, msg: str = "") -> bool:
                self._queue.put({
                    "tag": "progress",
                    "value": current,
                    "maximum": total,
                })
                if msg:
                    self._queue.put({"tag": "log", "text": msg})
                return not self._stop_training_flag.is_set()

            model = trainer.train(image_files, progress_callback=progress_cb)
            self._patchcore_model = model

            self._queue.put({
                "tag": "training_done",
                "backbone": backbone,
                "feature_dim": getattr(model, "feature_dim", "N/A"),
                "memory_bank_size": getattr(model, "memory_bank_size", "N/A"),
                "threshold": getattr(model, "threshold", 0.5),
            })

        except Exception as exc:
            self._queue.put({
                "tag": "training_error",
                "error": f"訓練過程中發生錯誤:\n{exc}",
            })

    def _stop_patchcore_training(self) -> None:
        self._stop_training_flag.set()
        self._pc_stop_btn.configure(state=tk.DISABLED)
        self._append_log("正在停止訓練...")

    def _on_training_done(self, msg: Dict[str, Any]) -> None:
        self._pc_train_btn.configure(state=tk.NORMAL)
        self._pc_stop_btn.configure(state=tk.DISABLED)
        self._pc_progress["value"] = self._pc_progress["maximum"]

        self._pc_info_labels["Backbone:"].configure(
            text=str(msg.get("backbone", "--")))
        self._pc_info_labels["特徵維度:"].configure(
            text=str(msg.get("feature_dim", "--")))
        self._pc_info_labels["Memory Bank 大小:"].configure(
            text=str(msg.get("memory_bank_size", "--")))
        threshold = msg.get("threshold", 0.5)
        self._pc_info_labels["閾值:"].configure(text=f"{threshold:.4f}")
        self._pc_threshold_var.set(f"{threshold:.4f}")

        self._append_log("PatchCore 訓練完成!")
        self._set_status("PatchCore 訓練完成")

        # Register in loaded models for comparison
        self._loaded_models["PatchCore"] = {
            "type": "patchcore",
            "model": self._patchcore_model,
        }
        self._refresh_compare_models()

    def _on_training_error(self, error: str) -> None:
        self._pc_train_btn.configure(state=tk.NORMAL)
        self._pc_stop_btn.configure(state=tk.DISABLED)
        self._append_log(f"錯誤: {error}")
        messagebox.showerror("訓練錯誤", error, parent=self)

    def _load_patchcore_model(self) -> None:
        path = filedialog.askopenfilename(
            parent=self, title="載入 PatchCore 模型",
            filetypes=[("NPZ 檔案", "*.npz"), ("所有檔案", "*")],
        )
        if not path:
            return

        def _load_worker() -> None:
            try:
                from dl_anomaly.core.patchcore import PatchCoreModel
            except ImportError as exc:
                self._queue.put({
                    "tag": "training_error",
                    "error": f"PatchCore 模組未安裝:\n{exc}",
                })
                return

            try:
                model = PatchCoreModel.load(path)
                self._patchcore_model = model
                self._queue.put({
                    "tag": "model_loaded",
                    "source": "patchcore",
                    "path": path,
                    "backbone": getattr(model, "backbone", "N/A"),
                    "feature_dim": getattr(model, "feature_dim", "N/A"),
                    "memory_bank_size": getattr(model, "memory_bank_size", "N/A"),
                    "threshold": getattr(model, "threshold", 0.5),
                })
            except Exception as exc:
                self._queue.put({
                    "tag": "training_error",
                    "error": f"載入模型失敗:\n{exc}",
                })

        threading.Thread(target=_load_worker, daemon=True).start()

    def _on_model_loaded(self, msg: Dict[str, Any]) -> None:
        source = msg.get("source", "")
        if source == "patchcore":
            self._pc_info_labels["Backbone:"].configure(
                text=str(msg.get("backbone", "--")))
            self._pc_info_labels["特徵維度:"].configure(
                text=str(msg.get("feature_dim", "--")))
            self._pc_info_labels["Memory Bank 大小:"].configure(
                text=str(msg.get("memory_bank_size", "--")))
            threshold = msg.get("threshold", 0.5)
            self._pc_info_labels["閾值:"].configure(text=f"{threshold:.4f}")
            self._pc_threshold_var.set(f"{threshold:.4f}")

            self._append_log(f"PatchCore 模型已載入: {msg.get('path', '')}")

            self._loaded_models["PatchCore"] = {
                "type": "patchcore",
                "model": self._patchcore_model,
            }
            self._refresh_compare_models()

        elif source == "onnx":
            self._onnx_info_labels["檔案路徑:"].configure(
                text=str(msg.get("path", "--")))
            self._onnx_info_labels["輸入形狀:"].configure(
                text=str(msg.get("input_shape", "--")))
            self._onnx_info_labels["輸出形狀:"].configure(
                text=str(msg.get("output_shape", "--")))
            self._onnx_info_labels["執行提供者:"].configure(
                text=str(msg.get("provider", "--")))
            self._onnx_info_labels["Opset 版本:"].configure(
                text=str(msg.get("opset", "--")))

            self._append_log(f"ONNX 模型已載入: {msg.get('path', '')}")

            self._loaded_models["ONNX"] = {
                "type": "onnx",
                "engine": self._onnx_engine,
            }
            self._refresh_compare_models()

    def _save_patchcore_model(self) -> None:
        if self._patchcore_model is None:
            messagebox.showwarning("警告", "尚未載入或訓練 PatchCore 模型",
                                   parent=self)
            return

        path = filedialog.asksaveasfilename(
            parent=self, title="儲存 PatchCore 模型",
            defaultextension=".npz",
            filetypes=[("NPZ 檔案", "*.npz")],
        )
        if not path:
            return

        try:
            threshold_str = self._pc_threshold_var.get().strip()
            try:
                threshold = float(threshold_str)
                self._patchcore_model.threshold = threshold
            except ValueError as exc:
                logger.debug("Invalid threshold value '%s', keeping current: %s", threshold_str, exc)

            self._patchcore_model.save(path)
            self._append_log(f"模型已儲存: {path}")
            messagebox.showinfo("資訊", f"模型已儲存至:\n{path}", parent=self)
        except Exception as exc:
            messagebox.showerror("錯誤", f"儲存失敗:\n{exc}", parent=self)

    def _patchcore_infer_single(self) -> None:
        if self._patchcore_model is None:
            messagebox.showwarning("警告", "請先載入或訓練 PatchCore 模型",
                                   parent=self)
            return

        image = self._get_current_image()
        if image is None:
            messagebox.showwarning("警告", "目前沒有影像", parent=self)
            return

        def _infer_worker() -> None:
            try:
                from dl_anomaly.core.patchcore import PatchCoreInference
            except ImportError as exc:
                self._queue.put({
                    "tag": "training_error",
                    "error": f"PatchCore 模組未安裝:\n{exc}",
                })
                return

            try:
                inference = PatchCoreInference(self._patchcore_model)
                threshold_str = self._pc_threshold_var.get().strip()
                try:
                    threshold = float(threshold_str)
                except ValueError:
                    threshold = 0.5

                result = inference.predict(image)
                score = getattr(result, "score", 0.0)
                anomaly_map = getattr(result, "anomaly_map", None)

                self._queue.put({
                    "tag": "inference_result",
                    "source": "patchcore",
                    "score": score,
                    "threshold": threshold,
                    "anomaly_map": anomaly_map,
                    "is_anomaly": score > threshold,
                })
            except Exception as exc:
                self._queue.put({
                    "tag": "training_error",
                    "error": f"推論失敗:\n{exc}",
                })

        threading.Thread(target=_infer_worker, daemon=True).start()

    def _patchcore_infer_batch(self) -> None:
        if self._patchcore_model is None:
            messagebox.showwarning("警告", "請先載入或訓練 PatchCore 模型",
                                   parent=self)
            return

        batch_dir = filedialog.askdirectory(
            title="選擇批次檢測資料夾", parent=self,
        )
        if not batch_dir:
            return

        def _batch_worker() -> None:
            try:
                from dl_anomaly.core.patchcore import PatchCoreInference
                import cv2
            except ImportError as exc:
                self._queue.put({
                    "tag": "training_error",
                    "error": f"依賴模組未安裝:\n{exc}",
                })
                return

            try:
                inference = PatchCoreInference(self._patchcore_model)
                threshold_str = self._pc_threshold_var.get().strip()
                try:
                    threshold = float(threshold_str)
                except ValueError:
                    threshold = 0.5

                exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
                files = sorted(
                    p for p in Path(batch_dir).rglob("*")
                    if p.is_file() and p.suffix.lower() in exts
                )
                if not files:
                    self._queue.put({
                        "tag": "log",
                        "text": "資料夾中未找到影像檔案",
                    })
                    return

                total = len(files)
                anomaly_count = 0

                for i, fpath in enumerate(files):
                    if self._stop_training_flag.is_set():
                        self._queue.put({"tag": "log", "text": "批次檢測已中止"})
                        break

                    img = cv2.imread(str(fpath))
                    if img is None:
                        self._queue.put({
                            "tag": "log",
                            "text": f"無法讀取: {fpath.name}",
                        })
                        continue

                    result = inference.predict(img)
                    score = getattr(result, "score", 0.0)
                    is_anomaly = score > threshold
                    if is_anomaly:
                        anomaly_count += 1

                    status = "異常" if is_anomaly else "正常"
                    self._queue.put({
                        "tag": "log",
                        "text": f"[{i+1}/{total}] {fpath.name}: {score:.4f} ({status})",
                    })
                    self._queue.put({
                        "tag": "batch_progress",
                        "value": i + 1,
                        "maximum": total,
                        "tab": "patchcore",
                    })

                self._queue.put({
                    "tag": "batch_done",
                    "text": (
                        f"批次檢測完成: {total} 張影像, "
                        f"{anomaly_count} 張異常"
                    ),
                })
            except Exception as exc:
                self._queue.put({
                    "tag": "training_error",
                    "error": f"批次檢測失敗:\n{exc}",
                })

        self._stop_training_flag.clear()
        threading.Thread(target=_batch_worker, daemon=True).start()

    def _on_inference_result(self, msg: Dict[str, Any]) -> None:
        source = msg.get("source", "")
        score = msg.get("score", 0.0)
        threshold = msg.get("threshold", 0.5)
        is_anomaly = msg.get("is_anomaly", False)
        anomaly_map = msg.get("anomaly_map")

        status = "異常" if is_anomaly else "正常"
        self._append_log(
            f"[{source.upper()}] 分數: {score:.4f}, "
            f"閾值: {threshold:.4f}, 結果: {status}"
        )
        self._set_status(f"檢測結果: {status} (分數: {score:.4f})")

        if anomaly_map is not None:
            # Normalize anomaly map to 0-255 for display
            amap = anomaly_map.copy()
            if amap.max() > amap.min():
                amap = (amap - amap.min()) / (amap.max() - amap.min())
            amap = (amap * 255).astype(np.uint8)

            name = f"{source.upper()} Anomaly Map ({score:.4f})"
            self._add_pipeline_step(name, amap, op_meta={
                "op": f"{source}_inference",
                "score": float(score),
                "threshold": float(threshold),
                "is_anomaly": is_anomaly,
            })

    # ================================================================== #
    # ONNX handlers                                                        #
    # ================================================================== #

    def _browse_export_path(self) -> None:
        path = filedialog.asksaveasfilename(
            parent=self, title="選擇匯出路徑",
            defaultextension=".onnx",
            filetypes=[("ONNX 模型", "*.onnx")],
        )
        if path:
            self._export_path_var.set(path)

    def _export_onnx(self) -> None:
        export_path = self._export_path_var.get().strip()
        if not export_path:
            messagebox.showwarning("警告", "請指定匯出路徑", parent=self)
            return

        source = self._export_source_var.get()
        opset = self._opset_var.get()

        def _export_worker() -> None:
            try:
                from dl_anomaly.core.onnx_engine import export_to_onnx
            except ImportError as exc:
                self._queue.put({
                    "tag": "onnx_export_done",
                    "text": f"匯出失敗: ONNX 模組未安裝 ({exc})",
                })
                return

            try:
                self._queue.put({
                    "tag": "log",
                    "text": f"正在匯出 {source} 至 {export_path}...",
                })

                if source == "目前 PatchCore":
                    if self._patchcore_model is None:
                        self._queue.put({
                            "tag": "onnx_export_done",
                            "text": "匯出失敗: 尚未載入 PatchCore 模型",
                        })
                        return
                    export_to_onnx(
                        self._patchcore_model,
                        export_path,
                        opset_version=opset,
                    )
                else:
                    # Autoencoder export
                    export_to_onnx(
                        self._config,
                        export_path,
                        opset_version=opset,
                        model_type="autoencoder",
                    )

                self._queue.put({
                    "tag": "onnx_export_done",
                    "text": f"匯出成功: {export_path}",
                })
                self._queue.put({
                    "tag": "log",
                    "text": f"ONNX 匯出完成: {export_path}",
                })
            except Exception as exc:
                self._queue.put({
                    "tag": "onnx_export_done",
                    "text": f"匯出失敗: {exc}",
                })

        threading.Thread(target=_export_worker, daemon=True).start()

    def _load_onnx_model(self) -> None:
        path = filedialog.askopenfilename(
            parent=self, title="載入 ONNX 模型",
            filetypes=[("ONNX 模型", "*.onnx"), ("所有檔案", "*")],
        )
        if not path:
            return

        def _load_worker() -> None:
            try:
                from dl_anomaly.core.onnx_engine import (
                    OnnxInferenceEngine,
                    check_onnxruntime_available,
                    get_available_providers,
                )
            except ImportError as exc:
                self._queue.put({
                    "tag": "training_error",
                    "error": f"ONNX Runtime 未安裝:\n{exc}",
                })
                return

            try:
                if not check_onnxruntime_available():
                    self._queue.put({
                        "tag": "training_error",
                        "error": "ONNX Runtime 未安裝。請執行: pip install onnxruntime",
                    })
                    return

                engine = OnnxInferenceEngine(path)
                self._onnx_engine = engine

                providers = get_available_providers()

                self._queue.put({
                    "tag": "model_loaded",
                    "source": "onnx",
                    "path": path,
                    "input_shape": getattr(engine, "input_shape", "N/A"),
                    "output_shape": getattr(engine, "output_shape", "N/A"),
                    "provider": ", ".join(providers) if providers else "N/A",
                    "opset": getattr(engine, "opset_version", "N/A"),
                })
            except Exception as exc:
                self._queue.put({
                    "tag": "training_error",
                    "error": f"載入 ONNX 模型失敗:\n{exc}",
                })

        threading.Thread(target=_load_worker, daemon=True).start()

    def _update_onnx_info(self, msg: Dict[str, Any]) -> None:
        for key, value in msg.items():
            if key in self._onnx_info_labels:
                self._onnx_info_labels[key].configure(text=str(value))

    def _compute_onnx_threshold(self) -> None:
        if self._onnx_engine is None:
            messagebox.showwarning("警告", "請先載入 ONNX 模型", parent=self)
            return

        good_dir = filedialog.askdirectory(
            title="選擇良品影像資料夾", parent=self,
        )
        if not good_dir:
            return

        def _compute_worker() -> None:
            try:
                from dl_anomaly.core.onnx_engine import OnnxAnomalyDetector
                import cv2
            except ImportError as exc:
                self._queue.put({
                    "tag": "training_error",
                    "error": f"依賴模組未安裝:\n{exc}",
                })
                return

            try:
                detector = OnnxAnomalyDetector(self._onnx_engine)

                exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
                files = sorted(
                    p for p in Path(good_dir).rglob("*")
                    if p.is_file() and p.suffix.lower() in exts
                )
                if not files:
                    self._queue.put({
                        "tag": "log",
                        "text": "資料夾中未找到影像檔案",
                    })
                    return

                scores: List[float] = []
                for i, fpath in enumerate(files):
                    img = cv2.imread(str(fpath))
                    if img is None:
                        continue
                    result = detector.predict(img)
                    score = getattr(result, "score", 0.0)
                    scores.append(score)

                    self._queue.put({
                        "tag": "batch_progress",
                        "value": i + 1,
                        "maximum": len(files),
                        "tab": "onnx",
                    })

                if scores:
                    # threshold = mean + 3*std
                    mean_s = np.mean(scores)
                    std_s = np.std(scores)
                    threshold = float(mean_s + 3 * std_s)
                    self._queue.put({
                        "tag": "log",
                        "text": (
                            f"良品閾值計算: mean={mean_s:.4f}, "
                            f"std={std_s:.4f}, threshold={threshold:.4f}"
                        ),
                    })
                    # Update threshold in UI via queue
                    self._queue.put({
                        "tag": "inference_result",
                        "source": "onnx_threshold",
                        "threshold": threshold,
                    })
                    # Directly update var from main thread via queue
                    self.after(0, lambda: self._onnx_threshold_var.set(
                        f"{threshold:.4f}"))

            except Exception as exc:
                self._queue.put({
                    "tag": "training_error",
                    "error": f"閾值計算失敗:\n{exc}",
                })

        threading.Thread(target=_compute_worker, daemon=True).start()

    def _onnx_infer_single(self) -> None:
        if self._onnx_engine is None:
            messagebox.showwarning("警告", "請先載入 ONNX 模型", parent=self)
            return

        image = self._get_current_image()
        if image is None:
            messagebox.showwarning("警告", "目前沒有影像", parent=self)
            return

        def _infer_worker() -> None:
            try:
                from dl_anomaly.core.onnx_engine import OnnxAnomalyDetector
            except ImportError as exc:
                self._queue.put({
                    "tag": "training_error",
                    "error": f"ONNX 模組未安裝:\n{exc}",
                })
                return

            try:
                detector = OnnxAnomalyDetector(self._onnx_engine)
                threshold_str = self._onnx_threshold_var.get().strip()
                try:
                    threshold = float(threshold_str)
                except ValueError:
                    threshold = 0.5

                result = detector.predict(image)
                score = getattr(result, "score", 0.0)
                anomaly_map = getattr(result, "anomaly_map", None)

                self._queue.put({
                    "tag": "inference_result",
                    "source": "onnx",
                    "score": score,
                    "threshold": threshold,
                    "anomaly_map": anomaly_map,
                    "is_anomaly": score > threshold,
                })
            except Exception as exc:
                self._queue.put({
                    "tag": "training_error",
                    "error": f"ONNX 推論失敗:\n{exc}",
                })

        threading.Thread(target=_infer_worker, daemon=True).start()

    def _onnx_infer_batch(self) -> None:
        if self._onnx_engine is None:
            messagebox.showwarning("警告", "請先載入 ONNX 模型", parent=self)
            return

        batch_dir = filedialog.askdirectory(
            title="選擇批次推論資料夾", parent=self,
        )
        if not batch_dir:
            return

        def _batch_worker() -> None:
            try:
                from dl_anomaly.core.onnx_engine import OnnxAnomalyDetector
                import cv2
            except ImportError as exc:
                self._queue.put({
                    "tag": "training_error",
                    "error": f"依賴模組未安裝:\n{exc}",
                })
                return

            try:
                detector = OnnxAnomalyDetector(self._onnx_engine)
                threshold_str = self._onnx_threshold_var.get().strip()
                try:
                    threshold = float(threshold_str)
                except ValueError:
                    threshold = 0.5

                exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
                files = sorted(
                    p for p in Path(batch_dir).rglob("*")
                    if p.is_file() and p.suffix.lower() in exts
                )
                if not files:
                    self._queue.put({
                        "tag": "log",
                        "text": "資料夾中未找到影像檔案",
                    })
                    return

                total = len(files)
                anomaly_count = 0

                for i, fpath in enumerate(files):
                    img = cv2.imread(str(fpath))
                    if img is None:
                        self._queue.put({
                            "tag": "log",
                            "text": f"無法讀取: {fpath.name}",
                        })
                        continue

                    result = detector.predict(img)
                    score = getattr(result, "score", 0.0)
                    is_anomaly = score > threshold
                    if is_anomaly:
                        anomaly_count += 1

                    status = "異常" if is_anomaly else "正常"
                    self._queue.put({
                        "tag": "log",
                        "text": f"[{i+1}/{total}] {fpath.name}: {score:.4f} ({status})",
                    })
                    self._queue.put({
                        "tag": "batch_progress",
                        "value": i + 1,
                        "maximum": total,
                        "tab": "onnx",
                    })

                self._queue.put({
                    "tag": "batch_done",
                    "text": (
                        f"批次推論完成: {total} 張影像, "
                        f"{anomaly_count} 張異常"
                    ),
                })
            except Exception as exc:
                self._queue.put({
                    "tag": "training_error",
                    "error": f"批次推論失敗:\n{exc}",
                })

        self._stop_training_flag.clear()
        threading.Thread(target=_batch_worker, daemon=True).start()

    # ================================================================== #
    # Comparison handlers                                                  #
    # ================================================================== #

    def _browse_compare_dir(self) -> None:
        d = filedialog.askdirectory(title="選擇比較影像資料夾", parent=self)
        if d:
            self._cmp_dir_var.set(d)

    def _refresh_compare_models(self) -> None:
        """Refresh the model comboboxes in the comparison tab."""
        names = list(self._loaded_models.keys())
        # Also check for Autoencoder availability
        if "Autoencoder" not in names:
            if hasattr(self._config, "model") and self._config.model is not None:
                self._loaded_models["Autoencoder"] = {
                    "type": "autoencoder",
                    "config": self._config,
                }
                names = list(self._loaded_models.keys())

        self._cmp_model_a_cb["values"] = names
        self._cmp_model_b_cb["values"] = names

    def _run_comparison(self) -> None:
        model_a_name = self._cmp_model_a_var.get()
        model_b_name = self._cmp_model_b_var.get()

        if not model_a_name or not model_b_name:
            messagebox.showwarning("警告", "請選擇兩個模型進行比較", parent=self)
            return

        if model_a_name not in self._loaded_models:
            messagebox.showwarning("警告", f"模型 '{model_a_name}' 未載入",
                                   parent=self)
            return
        if model_b_name not in self._loaded_models:
            messagebox.showwarning("警告", f"模型 '{model_b_name}' 未載入",
                                   parent=self)
            return

        source = self._cmp_source_var.get()

        if source == "current":
            image = self._get_current_image()
            if image is None:
                messagebox.showwarning("警告", "目前沒有影像", parent=self)
                return
            image_list = [("current_image", image)]
        else:
            cmp_dir = self._cmp_dir_var.get().strip()
            if not cmp_dir or not Path(cmp_dir).is_dir():
                messagebox.showwarning("警告", "請選擇有效的比較影像資料夾",
                                       parent=self)
                return
            image_list = None  # Will be loaded in worker
            cmp_dir_path = cmp_dir

        def _compare_worker() -> None:
            try:
                import cv2
            except ImportError:
                self._queue.put({
                    "tag": "training_error",
                    "error": "OpenCV 未安裝",
                })
                return

            try:
                model_a_info = self._loaded_models[model_a_name]
                model_b_info = self._loaded_models[model_b_name]

                # Prepare image list
                images: List[Tuple[str, np.ndarray]] = []
                if image_list is not None:
                    images = image_list
                else:
                    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
                    files = sorted(
                        p for p in Path(cmp_dir_path).rglob("*")
                        if p.is_file() and p.suffix.lower() in exts
                    )
                    for fpath in files:
                        img = cv2.imread(str(fpath))
                        if img is not None:
                            images.append((fpath.name, img))

                if not images:
                    self._queue.put({
                        "tag": "log",
                        "text": "無可比較的影像",
                    })
                    return

                results: List[Dict[str, Any]] = []

                for img_name, img in images:
                    score_a = self._run_single_inference(model_a_info, img)
                    score_b = self._run_single_inference(model_b_info, img)

                    if score_a is not None and score_b is not None:
                        # Both above or both below threshold => agree
                        threshold = 0.5
                        agree = (score_a > threshold) == (score_b > threshold)
                        results.append({
                            "image": img_name,
                            "score_a": score_a,
                            "score_b": score_b,
                            "agree": agree,
                        })

                self._queue.put({
                    "tag": "compare_result",
                    "model_a": model_a_name,
                    "model_b": model_b_name,
                    "results": results,
                })
            except Exception as exc:
                self._queue.put({
                    "tag": "training_error",
                    "error": f"比較失敗:\n{exc}",
                })

        threading.Thread(target=_compare_worker, daemon=True).start()

    def _run_single_inference(
        self,
        model_info: Dict[str, Any],
        image: np.ndarray,
    ) -> Optional[float]:
        """Run inference with a single model. Returns score or None."""
        model_type = model_info.get("type", "")

        try:
            if model_type == "patchcore":
                from dl_anomaly.core.patchcore import PatchCoreInference
                model = model_info.get("model")
                if model is None:
                    return None
                inference = PatchCoreInference(model)
                result = inference.predict(image)
                return float(getattr(result, "score", 0.0))

            elif model_type == "onnx":
                from dl_anomaly.core.onnx_engine import OnnxAnomalyDetector
                engine = model_info.get("engine")
                if engine is None:
                    return None
                detector = OnnxAnomalyDetector(engine)
                result = detector.predict(image)
                return float(getattr(result, "score", 0.0))

            elif model_type == "autoencoder":
                # Delegate to existing autoencoder inference
                from dl_anomaly.pipeline.inference import InferencePipeline
                config = model_info.get("config")
                if config is None:
                    return None
                pipeline = InferencePipeline(config)
                result = pipeline.run_single(image)
                return float(getattr(result, "score", 0.0))

        except Exception as exc:
            logger.warning("Inference failed for %s: %s", model_type, exc)
            return None

        return None

    def _on_compare_result(self, msg: Dict[str, Any]) -> None:
        model_a = msg.get("model_a", "A")
        model_b = msg.get("model_b", "B")
        results = msg.get("results", [])

        # Clear previous results
        for item in self._cmp_tree.get_children():
            self._cmp_tree.delete(item)

        if not results:
            self._cmp_score_a_label.configure(text="模型 A 分數: --")
            self._cmp_score_b_label.configure(text="模型 B 分數: --")
            self._append_log("比較完成: 無結果")
            return

        # Update score labels for single image
        if len(results) == 1:
            r = results[0]
            self._cmp_score_a_label.configure(
                text=f"{model_a} 分數: {r['score_a']:.4f}")
            self._cmp_score_b_label.configure(
                text=f"{model_b} 分數: {r['score_b']:.4f}")
        else:
            # Show average scores
            avg_a = np.mean([r["score_a"] for r in results])
            avg_b = np.mean([r["score_b"] for r in results])
            self._cmp_score_a_label.configure(
                text=f"{model_a} 平均分數: {avg_a:.4f}")
            self._cmp_score_b_label.configure(
                text=f"{model_b} 平均分數: {avg_b:.4f}")

        # Populate table
        for r in results:
            agree_text = "Yes" if r["agree"] else "No"
            self._cmp_tree.insert("", tk.END, values=(
                r["image"],
                f"{r['score_a']:.4f}",
                f"{r['score_b']:.4f}",
                agree_text,
            ))

        agree_count = sum(1 for r in results if r["agree"])
        self._append_log(
            f"比較完成: {len(results)} 張影像, "
            f"一致: {agree_count}/{len(results)}"
        )

    # ================================================================== #
    # Log helpers                                                          #
    # ================================================================== #

    def _append_log(self, text: str) -> None:
        """Append text to the PatchCore log area (must be called from UI thread)."""
        self._pc_log_text.configure(state=tk.NORMAL)
        self._pc_log_text.insert(tk.END, text + "\n")
        self._pc_log_text.see(tk.END)
        self._pc_log_text.configure(state=tk.DISABLED)

    # ================================================================== #
    # Cleanup                                                              #
    # ================================================================== #

    def _on_close(self) -> None:
        """Handle window close: stop any running training and destroy."""
        self._stop_training_flag.set()
        self.destroy()
