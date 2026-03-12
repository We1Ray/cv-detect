"""
gui/mvp_tools_dialog.py - MVP tools dialog.

Provides a tabbed dialog combining:
1. Camera streaming (discovery, grab, live preview, parameters)
2. Inspection flow (build, execute, save/load recipe)
3. PDF report export (config, entries, one-click PDF generation)

All heavy imports are lazy-loaded inside handlers with try/except
so the dialog opens instantly even when optional dependencies are
missing.
"""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

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
_PASS_FG = "#88cc88"
_FAIL_FG = "#cc6666"

# Shared button kwargs
_BTN_KW: Dict[str, Any] = dict(
    bg=_BG_MEDIUM,
    fg=_FG,
    activebackground=_ACTIVE_BG,
    activeforeground="#ffffff",
    relief=tk.FLAT,
    padx=10,
    pady=3,
    font=("", 9),
)
_BTN_ACCENT_KW: Dict[str, Any] = dict(
    bg=_ACCENT,
    fg="#ffffff",
    activebackground="#005a9e",
    activeforeground="#ffffff",
    relief=tk.FLAT,
    padx=12,
    pady=4,
    font=("", 10, "bold"),
)
_LABEL_KW: Dict[str, Any] = dict(bg=_BG, fg=_FG, font=("", 9))
_ENTRY_KW: Dict[str, Any] = dict(
    bg=_BG_MEDIUM, fg=_FG, insertbackground=_FG, relief=tk.FLAT,
)
_SPINBOX_KW: Dict[str, Any] = dict(
    bg=_BG_MEDIUM, fg=_FG, buttonbackground=_BG_MEDIUM,
    insertbackground=_FG, relief=tk.FLAT,
)
_SCALE_KW: Dict[str, Any] = dict(
    bg=_BG, fg=_FG, troughcolor=_BG_MEDIUM,
    highlightthickness=0, sliderrelief=tk.FLAT,
)


# =========================================================================== #
#  MVPToolsDialog                                                              #
# =========================================================================== #


class MVPToolsDialog(tk.Toplevel):
    """Combined MVP tools dialog (camera / inspection flow / PDF report).

    Parameters
    ----------
    master : tk.Widget
        Parent widget.
    get_current_image : callable
        ``() -> np.ndarray | None`` returning the current working image.
    add_pipeline_step : callable
        ``(name: str, image: np.ndarray) -> None`` to push a result
        into the parent pipeline.
    set_status : callable
        ``(msg: str) -> None`` to update the parent status bar.
    """

    def __init__(
        self,
        master: tk.Widget,
        get_current_image: Callable[[], Optional[np.ndarray]],
        add_pipeline_step: Callable[[str, np.ndarray], None],
        set_status: Callable[[str], None],
    ) -> None:
        super().__init__(master)
        self.title("MVP 工具")
        self.geometry("950x720")
        self.resizable(True, True)
        self.configure(bg=_BG)

        self.transient(master)
        self.grab_set()

        self._get_current_image = get_current_image
        self._add_pipeline_step = add_pipeline_step
        self._set_status = set_status

        # PhotoImage references (prevent GC)
        self._photo_refs: List[Any] = []

        # ---- Camera state ---------------------------------------------------
        self._cam_manager: Any = None  # CameraManager (lazy)
        self._discovered_cameras: List[Any] = []
        self._streaming = False
        self._stream_queue: queue.Queue = queue.Queue(maxsize=4)
        self._stream_poll_id: Optional[str] = None

        # ---- Inspection flow state ------------------------------------------
        self._flow: Any = None  # InspectionFlow
        self._flow_steps_data: List[Dict[str, Any]] = []
        self._flow_result_queue: queue.Queue = queue.Queue(maxsize=1)
        self._flow_poll_id: Optional[str] = None

        # ---- Report state ---------------------------------------------------
        self._report_entries: List[Any] = []  # List[InspectionEntry]

        # ---- Tkinter variables (Camera) -------------------------------------
        self._exposure_var = tk.DoubleVar(value=10000.0)
        self._gain_var = tk.DoubleVar(value=0.0)
        self._trigger_mode_var = tk.StringVar(value="freerun")

        # ---- Tkinter variables (Inspection) ---------------------------------
        self._flow_name_var = tk.StringVar(value="inspection")
        self._step_type_var = tk.StringVar(value="LocateStep")
        self._loc_threshold_var = tk.StringVar(value="0.5")
        self._loc_angle_var = tk.StringVar(value="-30,30")
        self._det_model_path_var = tk.StringVar(value="")
        self._det_threshold_var = tk.StringVar(value="0.5")
        self._meas_type_var = tk.StringVar(value="edge_distance")
        self._judge_max_score_var = tk.StringVar(value="0.5")
        self._judge_max_area_var = tk.StringVar(value="0.1")

        # ---- Tkinter variables (Report) -------------------------------------
        self._rpt_title_var = tk.StringVar(value="AOI 檢測報告")
        self._rpt_company_var = tk.StringVar(value="")
        self._rpt_operator_var = tk.StringVar(value="")
        self._rpt_line_var = tk.StringVar(value="")
        self._rpt_logo_var = tk.StringVar(value="")
        self._rpt_heatmap_var = tk.BooleanVar(value=True)
        self._rpt_histogram_var = tk.BooleanVar(value=True)
        self._rpt_spc_var = tk.BooleanVar(value=False)

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._close)

    # =================================================================== #
    #  UI construction                                                      #
    # =================================================================== #

    def _build_ui(self) -> None:
        """Build the three-tab notebook layout."""
        style = ttk.Style(self)
        style.configure("MVP.TNotebook", background=_BG)
        style.configure(
            "MVP.TNotebook.Tab",
            background=_BG_MEDIUM, foreground=_FG, padding=[12, 4],
        )
        style.map(
            "MVP.TNotebook.Tab",
            background=[("selected", _ACTIVE_BG)],
            foreground=[("selected", "#ffffff")],
        )

        self._notebook = ttk.Notebook(self, style="MVP.TNotebook")
        self._notebook.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # Tab 1 - Camera streaming
        tab1 = tk.Frame(self._notebook, bg=_BG)
        self._notebook.add(tab1, text=" 相機串流 ")
        self._build_camera_tab(tab1)

        # Tab 2 - Inspection flow
        tab2 = tk.Frame(self._notebook, bg=_BG)
        self._notebook.add(tab2, text=" 檢測流程 ")
        self._build_inspection_tab(tab2)

        # Tab 3 - PDF report
        tab3 = tk.Frame(self._notebook, bg=_BG)
        self._notebook.add(tab3, text=" PDF 報表 ")
        self._build_report_tab(tab3)

    # ------------------------------------------------------------------- #
    #  Tab 1: Camera streaming                                              #
    # ------------------------------------------------------------------- #

    def _build_camera_tab(self, parent: tk.Frame) -> None:
        """Build camera streaming UI."""
        left = tk.Frame(parent, bg=_BG, width=320)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(6, 3), pady=6)
        left.pack_propagate(False)

        right = tk.Frame(parent, bg=_BG)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(3, 6), pady=6)

        # -- Discovery section --
        tk.Label(left, text="相機探索", **_LABEL_KW, font=("", 10, "bold")).pack(
            anchor=tk.W, pady=(0, 4),
        )
        tk.Button(left, text="探索相機", command=self._on_discover_cameras,
                  **_BTN_ACCENT_KW).pack(fill=tk.X, pady=2)

        self._cam_listbox = tk.Listbox(
            left, height=6, bg=_BG_MEDIUM, fg=_FG,
            selectbackground=_ACCENT, selectforeground="#ffffff",
            relief=tk.FLAT, font=("", 9),
        )
        self._cam_listbox.pack(fill=tk.X, pady=4)

        btn_row = tk.Frame(left, bg=_BG)
        btn_row.pack(fill=tk.X, pady=2)
        tk.Button(btn_row, text="開啟", command=self._on_open_camera,
                  **_BTN_KW).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        tk.Button(btn_row, text="關閉", command=self._on_close_camera,
                  **_BTN_KW).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0))

        # -- Grab / stream buttons --
        sep1 = tk.Frame(left, bg=_FG_DIM, height=1)
        sep1.pack(fill=tk.X, pady=8)

        tk.Button(left, text="單張擷取", command=self._on_grab_single,
                  **_BTN_ACCENT_KW).pack(fill=tk.X, pady=2)

        stream_row = tk.Frame(left, bg=_BG)
        stream_row.pack(fill=tk.X, pady=2)
        tk.Button(stream_row, text="開始串流", command=self._on_start_streaming,
                  **_BTN_KW).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        tk.Button(stream_row, text="停止串流", command=self._on_stop_streaming,
                  **_BTN_KW).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0))

        # -- Parameters --
        sep2 = tk.Frame(left, bg=_FG_DIM, height=1)
        sep2.pack(fill=tk.X, pady=8)

        tk.Label(left, text="曝光時間 (us)", **_LABEL_KW).pack(anchor=tk.W)
        self._exposure_scale = tk.Scale(
            left, variable=self._exposure_var, from_=100, to=500000,
            orient=tk.HORIZONTAL, command=self._on_exposure_changed,
            **_SCALE_KW,
        )
        self._exposure_scale.pack(fill=tk.X, pady=2)

        tk.Label(left, text="增益 (dB)", **_LABEL_KW).pack(anchor=tk.W)
        self._gain_scale = tk.Scale(
            left, variable=self._gain_var, from_=0, to=48,
            orient=tk.HORIZONTAL, command=self._on_gain_changed,
            resolution=0.5, **_SCALE_KW,
        )
        self._gain_scale.pack(fill=tk.X, pady=2)

        # -- Trigger mode --
        sep3 = tk.Frame(left, bg=_FG_DIM, height=1)
        sep3.pack(fill=tk.X, pady=8)
        tk.Label(left, text="觸發模式", **_LABEL_KW).pack(anchor=tk.W)

        for text, value in [("自由運行", "freerun"),
                            ("軟體觸發", "software"),
                            ("硬體觸發", "hardware")]:
            tk.Radiobutton(
                left, text=text, variable=self._trigger_mode_var, value=value,
                command=self._on_trigger_mode_changed,
                bg=_BG, fg=_FG, selectcolor=_BG_MEDIUM,
                activebackground=_BG, activeforeground=_FG,
                font=("", 9),
            ).pack(anchor=tk.W)

        # -- Status label --
        self._cam_status_lbl = tk.Label(
            left, text="未連線", **_LABEL_KW, wraplength=300, justify=tk.LEFT,
        )
        self._cam_status_lbl.pack(anchor=tk.W, pady=(8, 0))

        # -- Preview area (right pane) --
        tk.Label(right, text="預覽", **_LABEL_KW, font=("", 10, "bold")).pack(
            anchor=tk.W, pady=(0, 4),
        )
        self._cam_preview_lbl = tk.Label(
            right, bg=_CANVAS_BG, relief=tk.FLAT,
        )
        self._cam_preview_lbl.pack(fill=tk.BOTH, expand=True)

    # ---- Camera handlers ------------------------------------------------

    def _ensure_camera_manager(self) -> Any:
        """Lazily create a CameraManager instance."""
        if self._cam_manager is not None:
            return self._cam_manager
        try:
            from dl_anomaly.core.camera import CameraManager
            self._cam_manager = CameraManager()
            return self._cam_manager
        except Exception as exc:
            messagebox.showerror("錯誤", f"無法建立 CameraManager:\n{exc}", parent=self)
            return None

    def _on_discover_cameras(self) -> None:
        mgr = self._ensure_camera_manager()
        if mgr is None:
            return
        try:
            self._discovered_cameras = mgr.discover_all()
            self._cam_listbox.delete(0, tk.END)
            if not self._discovered_cameras:
                self._cam_listbox.insert(tk.END, "(未偵測到任何相機)")
                self._set_status("未偵測到任何相機")
                return
            for cam in self._discovered_cameras:
                self._cam_listbox.insert(tk.END, str(cam))
            self._set_status(f"探索到 {len(self._discovered_cameras)} 台相機")
        except Exception as exc:
            messagebox.showerror("錯誤", f"相機探索失敗:\n{exc}", parent=self)

    def _on_open_camera(self) -> None:
        mgr = self._ensure_camera_manager()
        if mgr is None:
            return
        sel = self._cam_listbox.curselection()
        if not sel:
            messagebox.showwarning("提示", "請先選擇一台相機", parent=self)
            return
        idx = sel[0]
        if idx >= len(self._discovered_cameras):
            return
        cam_info = self._discovered_cameras[idx]
        try:
            mgr.open(cam_info.id)
            self._cam_status_lbl.config(text=f"已連線: {cam_info}")
            self._set_status(f"已開啟相機: {cam_info}")
            # Update exposure/gain ranges
            try:
                exp_min, exp_max = mgr.get_exposure_range()
                self._exposure_scale.config(from_=exp_min, to=exp_max)
            except Exception as exc:
                logger.debug("Failed to read exposure range: %s", exc)
            try:
                gain_min, gain_max = mgr.get_gain_range()
                self._gain_scale.config(from_=gain_min, to=gain_max)
            except Exception as exc:
                logger.debug("Failed to read gain range: %s", exc)
        except Exception as exc:
            messagebox.showerror("錯誤", f"開啟相機失敗:\n{exc}", parent=self)

    def _on_close_camera(self) -> None:
        if self._streaming:
            self._on_stop_streaming()
        mgr = self._ensure_camera_manager()
        if mgr is None:
            return
        try:
            mgr.close()
            self._cam_status_lbl.config(text="未連線")
            self._set_status("相機已關閉")
        except Exception as exc:
            messagebox.showerror("錯誤", f"關閉相機失敗:\n{exc}", parent=self)

    def _on_grab_single(self) -> None:
        mgr = self._ensure_camera_manager()
        if mgr is None:
            return
        if not mgr.is_open:
            messagebox.showwarning("提示", "請先開啟相機", parent=self)
            return
        try:
            frame = mgr.grab_single()
            if frame is None:
                messagebox.showwarning("提示", "擷取失敗，未取得影像", parent=self)
                return
            self._display_camera_preview(frame.image)
            self._add_pipeline_step("相機擷取", frame.image)
            self._set_status(
                f"已擷取影像: {frame.width}x{frame.height} "
                f"(曝光={frame.exposure_us:.0f}us, 增益={frame.gain_db:.1f}dB)"
            )
        except Exception as exc:
            messagebox.showerror("錯誤", f"擷取失敗:\n{exc}", parent=self)

    def _on_start_streaming(self) -> None:
        mgr = self._ensure_camera_manager()
        if mgr is None:
            return
        if not mgr.is_open:
            messagebox.showwarning("提示", "請先開啟相機", parent=self)
            return
        if self._streaming:
            return

        # Clear queue
        while not self._stream_queue.empty():
            try:
                self._stream_queue.get_nowait()
            except queue.Empty:
                break

        def _frame_callback(frame_result: Any) -> None:
            try:
                self._stream_queue.put_nowait(frame_result)
            except queue.Full:
                pass  # Drop frame if UI is behind

        try:
            mgr.start_streaming(_frame_callback)
            self._streaming = True
            self._set_status("串流已啟動")
            self._poll_stream_queue()
        except Exception as exc:
            messagebox.showerror("錯誤", f"啟動串流失敗:\n{exc}", parent=self)

    def _on_stop_streaming(self) -> None:
        self._streaming = False
        if self._stream_poll_id is not None:
            self.after_cancel(self._stream_poll_id)
            self._stream_poll_id = None
        mgr = self._ensure_camera_manager()
        if mgr is None:
            return
        try:
            if mgr.is_streaming:
                mgr.stop_streaming()
            self._set_status("串流已停止")
        except Exception as exc:
            logger.warning("停止串流失敗: %s", exc)

    def _poll_stream_queue(self) -> None:
        """Poll the stream queue using root.after() to update preview."""
        if not self._streaming:
            return
        try:
            frame = self._stream_queue.get_nowait()
            self._display_camera_preview(frame.image)
        except queue.Empty:
            pass  # no frame available yet, expected during polling
        self._stream_poll_id = self.after(33, self._poll_stream_queue)  # ~30 fps

    def _display_camera_preview(self, image: np.ndarray) -> None:
        """Render a numpy image onto the camera preview label."""
        try:
            from PIL import Image, ImageTk
        except ImportError:
            logger.warning("Pillow 未安裝，無法顯示相機預覽")
            return

        try:
            import cv2
            if image.ndim == 2:
                pil_img = Image.fromarray(image, mode="L")
            else:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)

            # Fit into preview area
            max_w = max(self._cam_preview_lbl.winfo_width(), 320)
            max_h = max(self._cam_preview_lbl.winfo_height(), 240)
            pil_img.thumbnail((max_w, max_h), Image.LANCZOS)

            photo = ImageTk.PhotoImage(pil_img)
            self._cam_preview_lbl.config(image=photo)
            self._photo_refs.clear()
            self._photo_refs.append(photo)
        except Exception as exc:
            logger.warning("顯示預覽失敗: %s", exc)

    def _on_exposure_changed(self, _value: str) -> None:
        mgr = self._cam_manager
        if mgr is None or not mgr.is_open:
            return
        try:
            mgr.set_exposure(self._exposure_var.get())
        except Exception as exc:
            logger.warning("設定曝光失敗: %s", exc)

    def _on_gain_changed(self, _value: str) -> None:
        mgr = self._cam_manager
        if mgr is None or not mgr.is_open:
            return
        try:
            mgr.set_gain(self._gain_var.get())
        except Exception as exc:
            logger.warning("設定增益失敗: %s", exc)

    def _on_trigger_mode_changed(self) -> None:
        mgr = self._cam_manager
        if mgr is None or not mgr.is_open:
            return
        mode = self._trigger_mode_var.get()
        try:
            mgr.set_trigger_mode(mode)
            labels = {"freerun": "自由運行", "software": "軟體觸發", "hardware": "硬體觸發"}
            self._set_status(f"觸發模式已切換為: {labels.get(mode, mode)}")
        except Exception as exc:
            messagebox.showerror("錯誤", f"切換觸發模式失敗:\n{exc}", parent=self)

    # ------------------------------------------------------------------- #
    #  Tab 2: Inspection flow                                               #
    # ------------------------------------------------------------------- #

    def _build_inspection_tab(self, parent: tk.Frame) -> None:
        """Build inspection flow configuration UI."""
        left = tk.Frame(parent, bg=_BG, width=340)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(6, 3), pady=6)
        left.pack_propagate(False)

        right = tk.Frame(parent, bg=_BG)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(3, 6), pady=6)

        # -- Flow name --
        tk.Label(left, text="流程名稱", **_LABEL_KW, font=("", 10, "bold")).pack(
            anchor=tk.W, pady=(0, 2),
        )
        tk.Entry(left, textvariable=self._flow_name_var, **_ENTRY_KW).pack(
            fill=tk.X, pady=(0, 6),
        )

        # -- Step type selector --
        tk.Label(left, text="新增步驟類型", **_LABEL_KW).pack(anchor=tk.W)
        step_types_frame = tk.Frame(left, bg=_BG)
        step_types_frame.pack(fill=tk.X, pady=2)

        self._step_type_combo = ttk.Combobox(
            step_types_frame, textvariable=self._step_type_var,
            values=["LocateStep", "DetectStep", "MeasureStep",
                    "ClassifyStep", "JudgeStep"],
            state="readonly", width=16,
        )
        self._step_type_combo.pack(side=tk.LEFT, padx=(0, 4))
        self._step_type_combo.bind("<<ComboboxSelected>>", self._on_step_type_changed)

        tk.Button(step_types_frame, text="新增步驟", command=self._on_add_step,
                  **_BTN_ACCENT_KW).pack(side=tk.LEFT)

        # -- Step config area (dynamic) --
        self._step_config_frame = tk.Frame(left, bg=_BG)
        self._step_config_frame.pack(fill=tk.X, pady=6)
        self._build_step_config("LocateStep")

        # -- Step list --
        sep1 = tk.Frame(left, bg=_FG_DIM, height=1)
        sep1.pack(fill=tk.X, pady=6)

        tk.Label(left, text="步驟清單", **_LABEL_KW, font=("", 10, "bold")).pack(
            anchor=tk.W, pady=(0, 2),
        )
        self._step_listbox = tk.Listbox(
            left, height=8, bg=_BG_MEDIUM, fg=_FG,
            selectbackground=_ACCENT, selectforeground="#ffffff",
            relief=tk.FLAT, font=("", 9),
        )
        self._step_listbox.pack(fill=tk.X, pady=2)

        list_btns = tk.Frame(left, bg=_BG)
        list_btns.pack(fill=tk.X, pady=2)
        tk.Button(list_btns, text="移除步驟", command=self._on_remove_step,
                  **_BTN_KW).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        tk.Button(list_btns, text="上移", command=self._on_move_step_up,
                  **_BTN_KW).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        tk.Button(list_btns, text="下移", command=self._on_move_step_down,
                  **_BTN_KW).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0))

        # -- Execute / Save / Load --
        sep2 = tk.Frame(left, bg=_FG_DIM, height=1)
        sep2.pack(fill=tk.X, pady=6)

        tk.Button(left, text="執行流程", command=self._on_run_flow,
                  **_BTN_ACCENT_KW).pack(fill=tk.X, pady=2)

        recipe_row = tk.Frame(left, bg=_BG)
        recipe_row.pack(fill=tk.X, pady=2)
        tk.Button(recipe_row, text="儲存配方", command=self._on_save_recipe,
                  **_BTN_KW).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        tk.Button(recipe_row, text="載入配方", command=self._on_load_recipe,
                  **_BTN_KW).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0))

        # -- Results area (right pane) --
        tk.Label(right, text="執行結果", **_LABEL_KW, font=("", 10, "bold")).pack(
            anchor=tk.W, pady=(0, 4),
        )
        self._flow_result_text = tk.Text(
            right, bg=_BG_MEDIUM, fg=_FG, insertbackground=_FG,
            relief=tk.FLAT, font=("Courier", 10), wrap=tk.WORD,
            state=tk.DISABLED,
        )
        self._flow_result_text.pack(fill=tk.BOTH, expand=True)

    def _build_step_config(self, step_type: str) -> None:
        """Rebuild the step-specific configuration widgets."""
        for w in self._step_config_frame.winfo_children():
            w.destroy()

        frame = self._step_config_frame
        label_map = {
            "LocateStep": "定位",
            "DetectStep": "偵測",
            "MeasureStep": "量測",
            "ClassifyStep": "分類",
            "JudgeStep": "判定",
        }
        tk.Label(frame, text=f"{label_map.get(step_type, step_type)} 設定",
                 **_LABEL_KW).pack(anchor=tk.W, pady=(0, 2))

        if step_type == "LocateStep":
            row1 = tk.Frame(frame, bg=_BG)
            row1.pack(fill=tk.X, pady=1)
            tk.Label(row1, text="匹配門檻:", **_LABEL_KW).pack(side=tk.LEFT)
            tk.Spinbox(row1, textvariable=self._loc_threshold_var,
                       from_=0.0, to=1.0, increment=0.05, width=8,
                       **_SPINBOX_KW).pack(side=tk.LEFT, padx=4)

            row2 = tk.Frame(frame, bg=_BG)
            row2.pack(fill=tk.X, pady=1)
            tk.Label(row2, text="角度範圍:", **_LABEL_KW).pack(side=tk.LEFT)
            tk.Entry(row2, textvariable=self._loc_angle_var, width=12,
                     **_ENTRY_KW).pack(side=tk.LEFT, padx=4)

        elif step_type == "DetectStep":
            row1 = tk.Frame(frame, bg=_BG)
            row1.pack(fill=tk.X, pady=1)
            tk.Label(row1, text="模型路徑:", **_LABEL_KW).pack(side=tk.LEFT)
            tk.Entry(row1, textvariable=self._det_model_path_var, width=18,
                     **_ENTRY_KW).pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)
            tk.Button(row1, text="...", command=self._on_browse_model_path,
                      **_BTN_KW, padx=4).pack(side=tk.LEFT)

            row2 = tk.Frame(frame, bg=_BG)
            row2.pack(fill=tk.X, pady=1)
            tk.Label(row2, text="異常門檻:", **_LABEL_KW).pack(side=tk.LEFT)
            tk.Spinbox(row2, textvariable=self._det_threshold_var,
                       from_=0.0, to=1.0, increment=0.05, width=8,
                       **_SPINBOX_KW).pack(side=tk.LEFT, padx=4)

        elif step_type == "MeasureStep":
            row1 = tk.Frame(frame, bg=_BG)
            row1.pack(fill=tk.X, pady=1)
            tk.Label(row1, text="量測類型:", **_LABEL_KW).pack(side=tk.LEFT)
            ttk.Combobox(
                row1, textvariable=self._meas_type_var,
                values=["edge_distance", "circle_fit", "line_fit",
                        "angle_measure", "area_measure"],
                state="readonly", width=16,
            ).pack(side=tk.LEFT, padx=4)

        elif step_type == "ClassifyStep":
            tk.Label(frame, text="(使用預設瑕疵類別)", **_LABEL_KW).pack(anchor=tk.W)

        elif step_type == "JudgeStep":
            row1 = tk.Frame(frame, bg=_BG)
            row1.pack(fill=tk.X, pady=1)
            tk.Label(row1, text="最大分數:", **_LABEL_KW).pack(side=tk.LEFT)
            tk.Entry(row1, textvariable=self._judge_max_score_var, width=8,
                     **_ENTRY_KW).pack(side=tk.LEFT, padx=4)

            row2 = tk.Frame(frame, bg=_BG)
            row2.pack(fill=tk.X, pady=1)
            tk.Label(row2, text="最大瑕疵面積:", **_LABEL_KW).pack(side=tk.LEFT)
            tk.Entry(row2, textvariable=self._judge_max_area_var, width=8,
                     **_ENTRY_KW).pack(side=tk.LEFT, padx=4)

    def _on_step_type_changed(self, _event: Any = None) -> None:
        self._build_step_config(self._step_type_var.get())

    def _on_browse_model_path(self) -> None:
        path = filedialog.askopenfilename(
            title="選擇模型檔案",
            filetypes=[("PyTorch Model", "*.pth *.pt"), ("ONNX", "*.onnx"),
                       ("所有檔案", "*.*")],
            parent=self,
        )
        if path:
            self._det_model_path_var.set(path)

    # ---- Inspection flow handlers ---------------------------------------

    def _on_add_step(self) -> None:
        step_type = self._step_type_var.get()
        step_label_map = {
            "LocateStep": "定位",
            "DetectStep": "偵測",
            "MeasureStep": "量測",
            "ClassifyStep": "分類",
            "JudgeStep": "判定",
        }

        config: Dict[str, Any] = {"step_type": step_type}
        if step_type == "LocateStep":
            config["min_score"] = float(self._loc_threshold_var.get())
            try:
                parts = self._loc_angle_var.get().split(",")
                config["angle_range"] = (float(parts[0]), float(parts[1]))
            except (ValueError, IndexError):
                config["angle_range"] = (-30.0, 30.0)
        elif step_type == "DetectStep":
            config["model_path"] = self._det_model_path_var.get() or None
            config["threshold"] = float(self._det_threshold_var.get())
        elif step_type == "MeasureStep":
            config["measurement_type"] = self._meas_type_var.get()
        elif step_type == "JudgeStep":
            try:
                config["max_score"] = float(self._judge_max_score_var.get())
            except ValueError:
                config["max_score"] = 0.5
            try:
                config["max_defect_area"] = float(self._judge_max_area_var.get())
            except ValueError:
                config["max_defect_area"] = 0.1

        self._flow_steps_data.append(config)
        label = step_label_map.get(step_type, step_type)
        display = f"[{len(self._flow_steps_data)}] {label}"
        if step_type == "DetectStep" and config.get("model_path"):
            import os
            display += f" ({os.path.basename(config['model_path'])})"
        self._step_listbox.insert(tk.END, display)
        self._set_status(f"已新增步驟: {label}")

    def _on_remove_step(self) -> None:
        sel = self._step_listbox.curselection()
        if not sel:
            messagebox.showwarning("提示", "請先選擇要移除的步驟", parent=self)
            return
        idx = sel[0]
        self._step_listbox.delete(idx)
        self._flow_steps_data.pop(idx)
        self._refresh_step_listbox()
        self._set_status("已移除步驟")

    def _on_move_step_up(self) -> None:
        sel = self._step_listbox.curselection()
        if not sel or sel[0] == 0:
            return
        idx = sel[0]
        self._flow_steps_data[idx - 1], self._flow_steps_data[idx] = (
            self._flow_steps_data[idx], self._flow_steps_data[idx - 1]
        )
        self._refresh_step_listbox()
        self._step_listbox.selection_set(idx - 1)

    def _on_move_step_down(self) -> None:
        sel = self._step_listbox.curselection()
        if not sel or sel[0] >= len(self._flow_steps_data) - 1:
            return
        idx = sel[0]
        self._flow_steps_data[idx], self._flow_steps_data[idx + 1] = (
            self._flow_steps_data[idx + 1], self._flow_steps_data[idx]
        )
        self._refresh_step_listbox()
        self._step_listbox.selection_set(idx + 1)

    def _refresh_step_listbox(self) -> None:
        """Rebuild the step listbox from internal data."""
        self._step_listbox.delete(0, tk.END)
        label_map = {
            "LocateStep": "定位", "DetectStep": "偵測",
            "MeasureStep": "量測", "ClassifyStep": "分類", "JudgeStep": "判定",
        }
        for i, cfg in enumerate(self._flow_steps_data):
            st = cfg.get("step_type", "?")
            label = label_map.get(st, st)
            display = f"[{i + 1}] {label}"
            self._step_listbox.insert(tk.END, display)

    def _build_flow_from_data(self) -> Any:
        """Build an InspectionFlow from the current step data list."""
        try:
            from dl_anomaly.core.inspection_flow import (
                ClassifyStep,
                DetectStep,
                InspectionFlow,
                JudgeStep,
                LocateStep,
                MeasureStep,
            )
        except ImportError as exc:
            messagebox.showerror(
                "錯誤", f"無法載入 inspection_flow 模組:\n{exc}", parent=self,
            )
            return None

        flow = InspectionFlow(name=self._flow_name_var.get())

        for cfg in self._flow_steps_data:
            st = cfg.get("step_type")
            step_cfg = {k: v for k, v in cfg.items() if k != "step_type"}
            if st == "LocateStep":
                flow.add_step(LocateStep(name="定位", config={
                    "min_score": step_cfg.get("min_score", 0.5),
                    "angle_start": step_cfg.get("angle_start", -30.0),
                    "angle_extent": step_cfg.get("angle_extent", 60.0),
                }))
            elif st == "DetectStep":
                flow.add_step(DetectStep(name="偵測", config={
                    "checkpoint_path": step_cfg.get("model_path", ""),
                    "threshold": step_cfg.get("threshold", 0.5),
                }))
            elif st == "MeasureStep":
                flow.add_step(MeasureStep(name="量測"))
            elif st == "ClassifyStep":
                flow.add_step(ClassifyStep(name="分類"))
            elif st == "JudgeStep":
                rules = []
                if "max_score" in step_cfg:
                    rules.append({
                        "field": "detect.anomaly_score",
                        "operator": "le",
                        "value": step_cfg["max_score"],
                    })
                if "max_defect_area" in step_cfg:
                    rules.append({
                        "field": "detect.total_defect_area",
                        "operator": "le",
                        "value": step_cfg["max_defect_area"],
                    })
                flow.add_step(JudgeStep(name="判定", config={
                    "rules": rules,
                }))

        return flow

    def _on_run_flow(self) -> None:
        if not self._flow_steps_data:
            messagebox.showwarning("提示", "請先新增至少一個步驟", parent=self)
            return

        image = self._get_current_image()
        if image is None:
            messagebox.showwarning("提示", "沒有可用的影像，請先載入影像", parent=self)
            return

        flow = self._build_flow_from_data()
        if flow is None:
            return

        self._set_status("正在執行檢測流程...")
        self._flow_result_text.config(state=tk.NORMAL)
        self._flow_result_text.delete("1.0", tk.END)
        self._flow_result_text.insert(tk.END, "執行中，請稍候...\n")
        self._flow_result_text.config(state=tk.DISABLED)

        # Clear result queue
        while not self._flow_result_queue.empty():
            try:
                self._flow_result_queue.get_nowait()
            except queue.Empty:
                break

        def _run_in_thread() -> None:
            try:
                result = flow.execute(image)
                self._flow_result_queue.put(("ok", result))
            except Exception as exc:
                self._flow_result_queue.put(("error", str(exc)))

        threading.Thread(target=_run_in_thread, daemon=True).start()
        self._poll_flow_result()

    def _poll_flow_result(self) -> None:
        """Poll the flow result queue."""
        try:
            status, data = self._flow_result_queue.get_nowait()
        except queue.Empty:
            self._flow_poll_id = self.after(100, self._poll_flow_result)
            return

        self._flow_result_text.config(state=tk.NORMAL)
        self._flow_result_text.delete("1.0", tk.END)

        if status == "ok":
            result = data
            # 摘要資訊
            verdict = "良品" if result.overall_pass else "瑕疵"
            self._flow_result_text.insert(
                tk.END,
                f"流程名稱：{result.flow_name}\n"
                f"最終判定：{verdict}\n"
                f"總耗時：{result.total_time_ms:.1f} ms\n"
                f"時間戳：{result.timestamp}\n",
            )
            if result.summary:
                self._flow_result_text.insert(tk.END, f"摘要：{result.summary}\n")

            self._flow_result_text.insert(tk.END, "\n--- 詳細結果 ---\n")
            for sr in result.steps:
                status_str = "成功" if sr.success else "失敗"
                self._flow_result_text.insert(
                    tk.END,
                    f"\n[{sr.step_name}] ({sr.step_type}) - {status_str} ({sr.elapsed_ms:.1f}ms)\n",
                )
                if sr.message:
                    self._flow_result_text.insert(tk.END, f"  訊息：{sr.message}\n")
                if isinstance(sr.data, dict):
                    for k, v in sr.data.items():
                        if isinstance(v, np.ndarray):
                            self._flow_result_text.insert(
                                tk.END, f"  {k}: ndarray {v.shape}\n",
                            )
                        else:
                            self._flow_result_text.insert(
                                tk.END, f"  {k}: {v}\n",
                            )

            # Push result image to pipeline
            try:
                self._add_pipeline_step("檢測流程", self._get_current_image())
            except Exception as exc:
                logger.debug("Failed to add pipeline step after inspection: %s", exc)

            self._set_status(
                f"流程執行完成：{verdict} ({result.total_time_ms:.1f}ms)"
            )
        else:
            self._flow_result_text.insert(tk.END, f"執行失敗:\n{data}\n")
            self._set_status("流程執行失敗")

        self._flow_result_text.config(state=tk.DISABLED)

    def _on_save_recipe(self) -> None:
        if not self._flow_steps_data:
            messagebox.showwarning("提示", "流程步驟清單為空", parent=self)
            return

        path = filedialog.asksaveasfilename(
            title="儲存檢測配方",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("所有檔案", "*.*")],
            parent=self,
        )
        if not path:
            return

        flow = self._build_flow_from_data()
        if flow is None:
            return
        try:
            flow.save(path)
            self._set_status(f"配方已儲存: {path}")
        except Exception as exc:
            messagebox.showerror("錯誤", f"儲存配方失敗:\n{exc}", parent=self)

    def _on_load_recipe(self) -> None:
        path = filedialog.askopenfilename(
            title="載入檢測配方",
            filetypes=[("JSON", "*.json"), ("所有檔案", "*.*")],
            parent=self,
        )
        if not path:
            return

        try:
            from dl_anomaly.core.inspection_flow import InspectionFlow
        except ImportError as exc:
            messagebox.showerror(
                "錯誤", f"無法載入 inspection_flow 模組:\n{exc}", parent=self,
            )
            return

        try:
            flow = InspectionFlow.load(path)
        except Exception as exc:
            messagebox.showerror("錯誤", f"載入配方失敗:\n{exc}", parent=self)
            return

        # Convert loaded flow steps back to our internal data format
        self._flow_steps_data.clear()
        step_type_map = {
            "locate": "LocateStep",
            "detect": "DetectStep",
            "measure": "MeasureStep",
            "classify": "ClassifyStep",
            "judge": "JudgeStep",
            "custom": "CustomStep",
        }
        for step in flow._steps:
            cfg = dict(step.config)
            cfg["step_type"] = step_type_map.get(step.step_type, step.step_type)
            self._flow_steps_data.append(cfg)

        self._flow_name_var.set(flow.name)
        self._refresh_step_listbox()
        self._set_status(f"已載入配方: {flow.name} ({len(flow._steps)} 個步驟)")

    # ------------------------------------------------------------------- #
    #  Tab 3: PDF report                                                    #
    # ------------------------------------------------------------------- #

    def _build_report_tab(self, parent: tk.Frame) -> None:
        """Build PDF report export UI."""
        top = tk.Frame(parent, bg=_BG)
        top.pack(fill=tk.X, padx=6, pady=6)

        bottom = tk.Frame(parent, bg=_BG)
        bottom.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

        # -- Report config section --
        config_frame = tk.LabelFrame(
            top, text=" 報表設定 ", bg=_BG, fg=_FG, font=("", 10, "bold"),
            relief=tk.GROOVE, bd=1,
        )
        config_frame.pack(fill=tk.X, pady=(0, 6))

        fields = [
            ("報表標題:", self._rpt_title_var, None),
            ("公司名稱:", self._rpt_company_var, None),
            ("操作員:", self._rpt_operator_var, None),
            ("產線名稱:", self._rpt_line_var, None),
            ("Logo 路徑:", self._rpt_logo_var, self._on_browse_logo),
        ]
        for i, (label_text, var, browse_cmd) in enumerate(fields):
            row = tk.Frame(config_frame, bg=_BG)
            row.pack(fill=tk.X, padx=8, pady=2)
            tk.Label(row, text=label_text, width=10, anchor=tk.E,
                     **_LABEL_KW).pack(side=tk.LEFT)
            entry = tk.Entry(row, textvariable=var, **_ENTRY_KW)
            entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
            if browse_cmd is not None:
                tk.Button(row, text="...", command=browse_cmd,
                          **_BTN_KW, padx=4).pack(side=tk.LEFT)

        # -- Options checkboxes --
        opts_frame = tk.Frame(config_frame, bg=_BG)
        opts_frame.pack(fill=tk.X, padx=8, pady=4)

        cb_kw = dict(
            bg=_BG, fg=_FG, selectcolor=_BG_MEDIUM,
            activebackground=_BG, activeforeground=_FG, font=("", 9),
        )
        tk.Checkbutton(opts_frame, text="包含熱力圖",
                       variable=self._rpt_heatmap_var, **cb_kw).pack(
            side=tk.LEFT, padx=(0, 12),
        )
        tk.Checkbutton(opts_frame, text="包含直方圖",
                       variable=self._rpt_histogram_var, **cb_kw).pack(
            side=tk.LEFT, padx=(0, 12),
        )
        tk.Checkbutton(opts_frame, text="包含 SPC 圖",
                       variable=self._rpt_spc_var, **cb_kw).pack(
            side=tk.LEFT,
        )

        # -- Entry list (Treeview) --
        tree_frame = tk.LabelFrame(
            bottom, text=" 檢測項目 ", bg=_BG, fg=_FG, font=("", 10, "bold"),
            relief=tk.GROOVE, bd=1,
        )
        tree_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 6))

        columns = ("index", "verdict", "score")
        self._report_tree = ttk.Treeview(
            tree_frame, columns=columns, show="headings", height=8,
        )
        self._report_tree.heading("index", text="序號")
        self._report_tree.heading("verdict", text="判定")
        self._report_tree.heading("score", text="分數")
        self._report_tree.column("index", width=60, anchor=tk.CENTER)
        self._report_tree.column("verdict", width=80, anchor=tk.CENTER)
        self._report_tree.column("score", width=100, anchor=tk.CENTER)

        # Style the treeview for dark theme
        tree_style = ttk.Style()
        tree_style.configure(
            "Treeview",
            background=_BG_MEDIUM, foreground=_FG, fieldbackground=_BG_MEDIUM,
            font=("", 9),
        )
        tree_style.configure("Treeview.Heading",
                             background=_BG, foreground=_FG, font=("", 9, "bold"))
        tree_style.map("Treeview", background=[("selected", _ACCENT)])

        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL,
                                  command=self._report_tree.yview)
        self._report_tree.configure(yscrollcommand=scrollbar.set)
        self._report_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0), pady=4)
        scrollbar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8), pady=4)

        # -- Buttons --
        btn_frame = tk.Frame(bottom, bg=_BG)
        btn_frame.pack(fill=tk.X)

        tk.Button(btn_frame, text="加入當前影像", command=self._on_add_report_entry,
                  **_BTN_KW).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(btn_frame, text="移除選取", command=self._on_remove_report_entry,
                  **_BTN_KW).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_frame, text="一鍵匯出 PDF", command=self._on_export_pdf,
                  **_BTN_ACCENT_KW).pack(side=tk.RIGHT, padx=(4, 0))

        # -- Status label --
        self._rpt_status_lbl = tk.Label(
            bottom, text="尚未加入任何檢測項目", **_LABEL_KW, anchor=tk.W,
        )
        self._rpt_status_lbl.pack(fill=tk.X, pady=(6, 0))

    # ---- Report handlers ------------------------------------------------

    def _on_browse_logo(self) -> None:
        path = filedialog.askopenfilename(
            title="選擇 Logo 圖片",
            filetypes=[("圖片檔案", "*.png *.jpg *.jpeg *.bmp"), ("所有檔案", "*.*")],
            parent=self,
        )
        if path:
            self._rpt_logo_var.set(path)

    def _on_add_report_entry(self) -> None:
        image = self._get_current_image()
        if image is None:
            messagebox.showwarning("提示", "沒有可用的影像", parent=self)
            return

        try:
            from dl_anomaly.core.report_generator import InspectionEntry
        except ImportError as exc:
            messagebox.showerror(
                "錯誤", f"無法載入 report_generator 模組:\n{exc}", parent=self,
            )
            return

        import datetime as _dt
        entry = InspectionEntry(
            original_image=image.copy(),
            anomaly_score=0.0,
            is_defective=False,
            timestamp=_dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            image_path=str(len(self._report_entries) + 1),
        )
        self._report_entries.append(entry)

        idx = len(self._report_entries)
        verdict = "FAIL" if entry.is_defective else "PASS"
        self._report_tree.insert(
            "", tk.END,
            values=(idx, verdict, f"{entry.anomaly_score:.4f}"),
        )
        self._rpt_status_lbl.config(text=f"已加入 {idx} 筆檢測項目")
        self._set_status(f"已加入第 {idx} 筆檢測項目至報表")

    def _on_remove_report_entry(self) -> None:
        selected = self._report_tree.selection()
        if not selected:
            messagebox.showwarning("提示", "請先選擇要移除的項目", parent=self)
            return

        # Collect indices to remove (from tree item values)
        indices_to_remove: List[int] = []
        for item_id in selected:
            values = self._report_tree.item(item_id, "values")
            try:
                idx = int(values[0]) - 1  # 0-based
                indices_to_remove.append(idx)
            except (ValueError, IndexError) as exc:
                logger.debug("Failed to parse report entry index: %s", exc)
            self._report_tree.delete(item_id)

        # Remove from internal list in reverse order
        for idx in sorted(indices_to_remove, reverse=True):
            if 0 <= idx < len(self._report_entries):
                self._report_entries.pop(idx)

        # Rebuild tree with updated indices
        self._rebuild_report_tree()
        count = len(self._report_entries)
        self._rpt_status_lbl.config(
            text=f"已加入 {count} 筆檢測項目" if count else "尚未加入任何檢測項目",
        )

    def _rebuild_report_tree(self) -> None:
        """Rebuild the report treeview from internal entries."""
        for item in self._report_tree.get_children():
            self._report_tree.delete(item)
        for i, entry in enumerate(self._report_entries):
            verdict = "FAIL" if entry.is_defective else "PASS"
            self._report_tree.insert(
                "", tk.END,
                values=(i + 1, verdict, f"{entry.anomaly_score:.4f}"),
            )

    def _on_export_pdf(self) -> None:
        if not self._report_entries:
            messagebox.showwarning("提示", "請先加入至少一筆檢測項目", parent=self)
            return

        output_path = filedialog.asksaveasfilename(
            title="匯出 PDF 報表",
            defaultextension=".pdf",
            filetypes=[("PDF 檔案", "*.pdf"), ("所有檔案", "*.*")],
            parent=self,
        )
        if not output_path:
            return

        try:
            from dl_anomaly.core.report_generator import (
                PDFReportGenerator,
                ReportConfig,
            )
        except ImportError as exc:
            messagebox.showerror(
                "錯誤", f"無法載入 report_generator 模組:\n{exc}", parent=self,
            )
            return

        config = ReportConfig(
            report_title=self._rpt_title_var.get(),
            company_name=self._rpt_company_var.get(),
            operator=self._rpt_operator_var.get(),
            line_id=self._rpt_line_var.get(),
            logo_path=self._rpt_logo_var.get() or "",
            include_spc=self._rpt_spc_var.get(),
            include_images=self._rpt_heatmap_var.get(),
            include_measurements=self._rpt_histogram_var.get(),
        )

        self._rpt_status_lbl.config(text="正在產生 PDF 報表...")
        self._set_status("正在匯出 PDF...")

        def _generate_in_thread() -> None:
            try:
                gen = PDFReportGenerator(config)
                for entry in self._report_entries:
                    gen.add_entry(entry)
                result_path = gen.generate(output_path)
                self.after(0, lambda: self._on_pdf_complete(result_path))
            except Exception as exc:
                self.after(0, lambda e=exc: self._on_pdf_error(e))

        threading.Thread(target=_generate_in_thread, daemon=True).start()

    def _on_pdf_complete(self, path: str) -> None:
        self._rpt_status_lbl.config(text=f"PDF 已匯出: {path}")
        self._set_status(f"PDF 報表已匯出: {path}")
        messagebox.showinfo("完成", f"PDF 報表已成功匯出:\n{path}", parent=self)

    def _on_pdf_error(self, exc: Exception) -> None:
        self._rpt_status_lbl.config(text="PDF 匯出失敗")
        self._set_status("PDF 匯出失敗")
        messagebox.showerror("錯誤", f"PDF 產生失敗:\n{exc}", parent=self)

    # =================================================================== #
    #  Cleanup                                                              #
    # =================================================================== #

    def _close(self) -> None:
        """Clean shutdown: stop streaming and close camera."""
        if self._streaming:
            self._on_stop_streaming()

        if self._flow_poll_id is not None:
            self.after_cancel(self._flow_poll_id)
            self._flow_poll_id = None

        if self._cam_manager is not None:
            try:
                self._cam_manager.close()
            except Exception as exc:
                logger.debug("Failed to close camera manager during cleanup: %s", exc)

        self.grab_release()
        self.destroy()
