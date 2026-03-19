"""Live inspection control panel for continuous production line monitoring.

Provides controls for:
- Start/Stop live inspection loop
- Interval configuration
- Image source selection (directory watch / camera)
- Real-time statistics display
- Auto-save results option
"""

from __future__ import annotations

import logging
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Callable, Optional

from shared.i18n import t

logger = logging.getLogger(__name__)

# -- Theme constants ----------------------------------------------------------

_BG = "#2b2b2b"
_FG = "#e0e0e0"
_BG_DARK = "#1e1e1e"
_FG_DIM = "#888888"
_COLOR_PASS = "#00C853"
_COLOR_FAIL = "#FF1744"
_COLOR_RUNNING = "#00C853"
_COLOR_STOPPED = "#FF1744"
_BTN_START_BG = "#1b5e20"
_BTN_STOP_BG = "#b71c1c"


class LiveInspectionPanel(tk.LabelFrame):
    """A self-contained panel for continuous live inspection.

    The panel provides source selection, interval configuration,
    start/stop controls, real-time counters, and a scrollable log
    of recent inspection results.

    Parameters
    ----------
    master:
        Parent widget.
    on_start:
        Callback ``(source: str, interval_ms: int) -> None`` invoked when
        the operator presses *Start*.
    on_stop:
        Callback ``() -> None`` invoked when the operator presses *Stop*.
    on_inspect_single:
        Callback ``() -> None`` for a one-shot inspection trigger.
    """

    _MAX_LOG_LINES: int = 20

    def __init__(
        self,
        master: tk.Misc,
        on_start: Callable[[str, int], None],
        on_stop: Callable[[], None],
        on_inspect_single: Callable[[], None],
    ) -> None:
        super().__init__(
            master,
            text=t("live.title"),
            bg=_BG,
            fg=_FG,
            font=("Helvetica", 11, "bold"),
            padx=8,
            pady=8,
        )

        self._on_start = on_start
        self._on_stop = on_stop
        self._on_inspect_single = on_inspect_single

        self._lock = threading.Lock()
        self._running = False
        self._loop_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pending_after_id: Optional[str] = None

        # Counters
        self._inspected = 0
        self._pass_count = 0
        self._fail_count = 0

        self._build_ui()

    # ================================================================== #
    #  UI construction                                                     #
    # ================================================================== #

    def _build_ui(self) -> None:
        # -- Source selection -------------------------------------------------
        src_frame = tk.Frame(self, bg=_BG)
        src_frame.pack(fill=tk.X, pady=(0, 6))

        tk.Label(
            src_frame, text=t("live.source"), bg=_BG, fg=_FG,
            font=("Helvetica", 10),
        ).pack(side=tk.LEFT)

        self._source_var = tk.StringVar(value="directory")
        rb_dir = tk.Radiobutton(
            src_frame, text=t("live.folder"), variable=self._source_var,
            value="directory", bg=_BG, fg=_FG, selectcolor=_BG_DARK,
            activebackground=_BG, activeforeground=_FG,
            font=("Helvetica", 10),
        )
        rb_dir.pack(side=tk.LEFT, padx=(8, 0))

        rb_cam = tk.Radiobutton(
            src_frame, text=t("live.camera"), variable=self._source_var,
            value="camera", bg=_BG, fg=_FG, selectcolor=_BG_DARK,
            activebackground=_BG, activeforeground=_FG,
            font=("Helvetica", 10),
        )
        rb_cam.pack(side=tk.LEFT, padx=(8, 0))

        # -- Directory path ---------------------------------------------------
        path_frame = tk.Frame(self, bg=_BG)
        path_frame.pack(fill=tk.X, pady=(0, 6))

        self._path_var = tk.StringVar()
        self._path_entry = tk.Entry(
            path_frame, textvariable=self._path_var,
            bg=_BG_DARK, fg=_FG, insertbackground=_FG,
            font=("Consolas", 10), relief=tk.FLAT,
        )
        self._path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        self._browse_btn = tk.Button(
            path_frame, text=t("live.browse"), command=self._browse_directory,
            bg=_BG_DARK, fg=_FG, activebackground="#3c3c3c",
            activeforeground=_FG, relief=tk.FLAT, padx=8,
            font=("Helvetica", 9),
        )
        self._browse_btn.pack(side=tk.RIGHT)

        # -- Interval ---------------------------------------------------------
        interval_frame = tk.Frame(self, bg=_BG)
        interval_frame.pack(fill=tk.X, pady=(0, 6))

        tk.Label(
            interval_frame, text=t("live.interval_ms"), bg=_BG, fg=_FG,
            font=("Helvetica", 10),
        ).pack(side=tk.LEFT)

        self._interval_var = tk.IntVar(value=1000)
        self._interval_spin = tk.Spinbox(
            interval_frame,
            from_=100, to=10000, increment=100,
            textvariable=self._interval_var,
            width=7,
            bg=_BG_DARK, fg=_FG, buttonbackground=_BG_DARK,
            insertbackground=_FG, font=("Consolas", 10),
            relief=tk.FLAT,
        )
        self._interval_spin.pack(side=tk.LEFT, padx=(8, 0))

        # -- Auto-save --------------------------------------------------------
        self._autosave_var = tk.BooleanVar(value=False)
        autosave_cb = tk.Checkbutton(
            interval_frame, text=t("live.auto_save"),
            variable=self._autosave_var,
            bg=_BG, fg=_FG, selectcolor=_BG_DARK,
            activebackground=_BG, activeforeground=_FG,
            font=("Helvetica", 10),
        )
        autosave_cb.pack(side=tk.RIGHT)

        # -- Control buttons --------------------------------------------------
        ctrl_frame = tk.Frame(self, bg=_BG)
        ctrl_frame.pack(fill=tk.X, pady=(0, 6))

        self._start_btn = tk.Button(
            ctrl_frame, text=f"  {t('live.start')}  ", command=self._handle_start,
            bg=_BTN_START_BG, fg=_FG, activebackground="#2e7d32",
            activeforeground=_FG, relief=tk.FLAT,
            font=("Helvetica", 11, "bold"), padx=12, pady=4,
        )
        self._start_btn.pack(side=tk.LEFT, padx=(0, 4))

        self._stop_btn = tk.Button(
            ctrl_frame, text=f"  {t('live.stop')}  ", command=self._handle_stop,
            bg=_BTN_STOP_BG, fg=_FG, activebackground="#c62828",
            activeforeground=_FG, relief=tk.FLAT,
            font=("Helvetica", 11, "bold"), padx=12, pady=4,
            state=tk.DISABLED,
        )
        self._stop_btn.pack(side=tk.LEFT, padx=(0, 4))

        self._single_btn = tk.Button(
            ctrl_frame, text=t("live.single_shot"), command=self._handle_single,
            bg=_BG_DARK, fg=_FG, activebackground="#3c3c3c",
            activeforeground=_FG, relief=tk.FLAT,
            font=("Helvetica", 10), padx=8, pady=4,
        )
        self._single_btn.pack(side=tk.LEFT, padx=(0, 4))

        # -- Status indicator -------------------------------------------------
        status_frame = tk.Frame(ctrl_frame, bg=_BG)
        status_frame.pack(side=tk.RIGHT)

        self._status_dot = tk.Canvas(
            status_frame, width=12, height=12,
            bg=_BG, highlightthickness=0,
        )
        self._status_dot.pack(side=tk.LEFT, padx=(0, 4))
        self._dot_id = self._status_dot.create_oval(
            2, 2, 10, 10, fill=_COLOR_STOPPED, outline="",
        )

        self._status_label = tk.Label(
            status_frame, text=t("live.stopped"), bg=_BG, fg=_COLOR_STOPPED,
            font=("Helvetica", 10, "bold"),
        )
        self._status_label.pack(side=tk.LEFT)

        # -- Count display ----------------------------------------------------
        count_frame = tk.Frame(self, bg=_BG_DARK, pady=4, padx=8)
        count_frame.pack(fill=tk.X, pady=(0, 6))

        self._count_label = tk.Label(
            count_frame,
            text=self._format_counts(0, 0, 0),
            bg=_BG_DARK, fg=_FG, font=("Consolas", 11),
        )
        self._count_label.pack(anchor=tk.W)

        # -- Log area ---------------------------------------------------------
        log_label = tk.Label(
            self, text=t("live.log_title"), bg=_BG, fg=_FG_DIM,
            font=("Helvetica", 9), anchor=tk.W,
        )
        log_label.pack(fill=tk.X)

        log_frame = tk.Frame(self, bg=_BG_DARK)
        log_frame.pack(fill=tk.BOTH, expand=True)

        self._log_text = tk.Text(
            log_frame,
            height=8, width=50,
            bg=_BG_DARK, fg=_FG, insertbackground=_FG,
            font=("Consolas", 9), wrap=tk.WORD,
            relief=tk.FLAT, state=tk.DISABLED,
        )
        log_scroll = tk.Scrollbar(
            log_frame, orient=tk.VERTICAL,
            command=self._log_text.yview,
        )
        self._log_text.configure(yscrollcommand=log_scroll.set)
        self._log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Tag colours for log entries.
        self._log_text.tag_configure("pass", foreground=_COLOR_PASS)
        self._log_text.tag_configure("fail", foreground=_COLOR_FAIL)
        self._log_text.tag_configure("info", foreground=_FG_DIM)

        self._log_line_count = 0

    # ================================================================== #
    #  Properties                                                          #
    # ================================================================== #

    @property
    def is_running(self) -> bool:
        """Whether the live inspection loop is currently active."""
        with self._lock:
            return self._running

    @property
    def autosave(self) -> bool:
        """Whether auto-save is enabled."""
        return self._autosave_var.get()

    # ================================================================== #
    #  Public API                                                          #
    # ================================================================== #

    def start_inspection(
        self,
        source_path: str,
        interval_ms: int = 1000,
    ) -> None:
        """Programmatically start the inspection loop.

        Parameters
        ----------
        source_path:
            Directory path or ``"camera"`` for live camera feed.
        interval_ms:
            Delay between inspections in milliseconds.
        """
        with self._lock:
            if self._running:
                return

        self._path_var.set(source_path)
        self._interval_var.set(interval_ms)
        self._start_loop()

    def stop_inspection(self) -> None:
        """Programmatically stop the inspection loop."""
        with self._lock:
            if not self._running:
                return
        self._stop_loop()

    def update_result(
        self,
        is_pass: bool,
        score: float,
        image_path: str = "",
    ) -> None:
        """Record an inspection result and update UI.

        This method is **thread-safe** -- it schedules the UI update on
        the main thread via ``after_idle``.
        """
        self.after_idle(self._apply_result, is_pass, score, image_path)

    def destroy(self) -> None:
        """Clean up resources before widget destruction."""
        # Cancel any pending after() calls.
        if self._pending_after_id is not None:
            try:
                self.after_cancel(self._pending_after_id)
            except Exception:
                pass
            self._pending_after_id = None

        # Stop inspection if running.
        with self._lock:
            if self._running:
                self._stop_event.set()
                self._running = False

        super().destroy()

    # ================================================================== #
    #  Internal handlers                                                   #
    # ================================================================== #

    def _browse_directory(self) -> None:
        path = filedialog.askdirectory(title=t("live.select_folder"))
        if path:
            self._path_var.set(path)

    def _handle_start(self) -> None:
        self._start_loop()

    def _handle_stop(self) -> None:
        self._stop_loop()

    def _handle_single(self) -> None:
        self._on_inspect_single()

    # -- Loop management -------------------------------------------------- #

    def _start_loop(self) -> None:
        source = self._path_var.get() if self._source_var.get() == "directory" else "camera"
        interval = self._interval_var.get()

        # Validate source.
        if self._source_var.get() == "directory" and not source:
            self._log_append(f"[WARN] {t('live.warn_no_folder')}", "info")
            return
        if self._source_var.get() == "directory" and not Path(source).is_dir():
            self._log_append(f"[WARN] {t('live.warn_folder_missing', path=source)}", "info")
            return

        with self._lock:
            self._running = True
        self._stop_event.clear()
        self._set_ui_running(True)
        self._log_append(f"[INFO] {t('live.log_started')}", "info")

        try:
            self._on_start(source, interval)
        except Exception as exc:
            logger.exception("on_start callback failed")
            self._log_append(f"[ERROR] {t('live.error_start_failed', error=str(exc))}", "fail")
            with self._lock:
                self._running = False
            self._set_ui_running(False)

    def _stop_loop(self) -> None:
        self._stop_event.set()
        with self._lock:
            self._running = False
        self._set_ui_running(False)
        self._log_append(f"[INFO] {t('live.log_stopped')}", "info")
        self._on_stop()

    # -- UI state --------------------------------------------------------- #

    def _set_ui_running(self, running: bool) -> None:
        if running:
            self._start_btn.configure(state=tk.DISABLED)
            self._stop_btn.configure(state=tk.NORMAL)
            self._single_btn.configure(state=tk.DISABLED)
            self._path_entry.configure(state=tk.DISABLED)
            self._browse_btn.configure(state=tk.DISABLED)
            self._interval_spin.configure(state=tk.DISABLED)
            self._status_dot.itemconfigure(self._dot_id, fill=_COLOR_RUNNING)
            self._status_label.configure(text=t("live.running"), fg=_COLOR_RUNNING)
        else:
            self._start_btn.configure(state=tk.NORMAL)
            self._stop_btn.configure(state=tk.DISABLED)
            self._single_btn.configure(state=tk.NORMAL)
            self._path_entry.configure(state=tk.NORMAL)
            self._browse_btn.configure(state=tk.NORMAL)
            self._interval_spin.configure(state=tk.NORMAL)
            self._status_dot.itemconfigure(self._dot_id, fill=_COLOR_STOPPED)
            self._status_label.configure(text=t("live.stopped"), fg=_COLOR_STOPPED)

    def _format_counts(
        self, inspected: int, passed: int, failed: int
    ) -> str:
        """Build the counter display string using i18n labels."""
        return (
            f"{t('live.inspected')} {inspected}  |  "
            f"{t('live.passed')} {passed}  |  "
            f"{t('live.failed')} {failed}"
        )

    def _apply_result(
        self,
        is_pass: bool,
        score: float,
        image_path: str,
    ) -> None:
        """Update counters, labels, and log.  Must run on main thread."""
        self._inspected += 1
        if is_pass:
            self._pass_count += 1
        else:
            self._fail_count += 1

        self._count_label.configure(
            text=self._format_counts(
                self._inspected, self._pass_count, self._fail_count
            ),
        )

        tag = "pass" if is_pass else "fail"
        result_text = "PASS" if is_pass else "FAIL"
        name = Path(image_path).name if image_path else "--"
        self._log_append(
            f"[{result_text}] score={score:.4f}  {name}",
            tag,
        )

    def _log_append(self, text: str, tag: str = "info") -> None:
        """Append a line to the log, removing old entries beyond the limit."""
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"{ts}  {text}\n"

        self._log_text.configure(state=tk.NORMAL)
        self._log_text.insert(tk.END, line, tag)
        self._log_line_count += 1

        # Trim oldest lines.
        while self._log_line_count > self._MAX_LOG_LINES:
            self._log_text.delete("1.0", "2.0")
            self._log_line_count -= 1

        self._log_text.see(tk.END)
        self._log_text.configure(state=tk.DISABLED)
