"""Real-time SPC Dashboard panel for production line monitoring.

Displays:
- Yield rate gauge (large percentage with color coding)
- Production counters (total, pass, fail)
- Yield trend mini-chart (last N inspections)
- Defect rate by time period
- Process capability indicators (Cp, Cpk if available)
"""

from __future__ import annotations

import logging
import tkinter as tk
from collections import deque
from typing import TYPE_CHECKING, Deque, Optional

from shared.i18n import t

if TYPE_CHECKING:
    from shared.core.results_db import ResultsDatabase, SPCMetrics

logger = logging.getLogger(__name__)


# -- Colour constants --------------------------------------------------------

_BG = "#2b2b2b"
_BG_DARK = "#1e1e1e"
_FG = "#e0e0e0"
_FG_DIM = "#888888"

_YIELD_GREEN = "#00C853"
_YIELD_YELLOW = "#FFD600"
_YIELD_RED = "#FF1744"

_COUNTER_PASS = "#4CAF50"
_COUNTER_FAIL = "#FF5252"
_COUNTER_TOTAL = "#FFFFFF"

_CHART_LINE = "#4fc3f7"
_CHART_REF_95 = "#00C853"
_CHART_REF_90 = "#FFD600"
_CHART_GRID = "#333333"

_TREND_WINDOW = 50


def _yield_color(rate: float) -> str:
    """Return the appropriate colour for a yield percentage."""
    if rate >= 95.0:
        return _YIELD_GREEN
    if rate >= 90.0:
        return _YIELD_YELLOW
    return _YIELD_RED


# ---------------------------------------------------------------------------
# DashboardPanel
# ---------------------------------------------------------------------------

class DashboardPanel(tk.LabelFrame):
    """Compact real-time SPC dashboard panel.

    Intended to sit in a sidebar or bottom dock of the inspection GUI.
    All drawing uses plain ``tkinter`` -- no matplotlib dependency -- so
    updates are lightweight and fast.

    Parameters
    ----------
    master:
        Parent widget.
    trend_window:
        Number of recent inspections to track for the yield trend chart.
    """

    def __init__(
        self,
        master: tk.Misc,
        *,
        trend_window: int = _TREND_WINDOW,
    ) -> None:
        super().__init__(
            master,
            text=" SPC Dashboard ",
            bg=_BG,
            fg=_FG,
            font=("Helvetica", 11, "bold"),
            labelanchor=tk.N,
            padx=6,
            pady=4,
        )

        self._trend_window = trend_window

        # -- Internal state --------------------------------------------------
        self._total: int = 0
        self._pass_count: int = 0
        self._fail_count: int = 0
        self._recent_results: Deque[bool] = deque(maxlen=trend_window)
        self._yield_history: Deque[float] = deque(maxlen=trend_window)

        # -- Build layout (top -> bottom) ------------------------------------
        self._build_yield_section()
        self._build_counters_section()
        self._build_trend_chart()
        self._build_spc_section()

    # ===================================================================== #
    #  Layout builders                                                       #
    # ===================================================================== #

    def _build_yield_section(self) -> None:
        """Large yield percentage with label."""
        frame = tk.Frame(self, bg=_BG)
        frame.pack(fill=tk.X, pady=(2, 0))

        self._yield_value_label = tk.Label(
            frame,
            text="--.--%",
            font=("Helvetica", 36, "bold"),
            fg=_FG_DIM,
            bg=_BG,
            anchor=tk.CENTER,
        )
        self._yield_value_label.pack(fill=tk.X)

        self._yield_title_label = tk.Label(
            frame,
            text=t("dashboard.yield_rate"),
            font=("Helvetica", 10),
            fg=_FG_DIM,
            bg=_BG,
            anchor=tk.CENTER,
        )
        self._yield_title_label.pack(fill=tk.X)

    def _build_counters_section(self) -> None:
        """Three-column counter grid: Total / Pass / Fail."""
        frame = tk.Frame(self, bg=_BG_DARK)
        frame.pack(fill=tk.X, pady=(6, 0))

        # Configure three equal columns.
        for col in range(3):
            frame.columnconfigure(col, weight=1)

        headers = [
            (t("dashboard.total"), _COUNTER_TOTAL),
            (t("dashboard.pass_count"), _COUNTER_PASS),
            (t("dashboard.fail_count"), _COUNTER_FAIL),
        ]
        self._counter_labels: list[tk.Label] = []

        for col, (header_text, color) in enumerate(headers):
            tk.Label(
                frame,
                text=header_text,
                font=("Helvetica", 9),
                fg=_FG_DIM,
                bg=_BG_DARK,
                anchor=tk.CENTER,
            ).grid(row=0, column=col, sticky=tk.EW, padx=2, pady=(4, 0))

            lbl = tk.Label(
                frame,
                text="0",
                font=("Consolas", 18, "bold"),
                fg=color,
                bg=_BG_DARK,
                anchor=tk.CENTER,
            )
            lbl.grid(row=1, column=col, sticky=tk.EW, padx=2, pady=(0, 4))
            self._counter_labels.append(lbl)

    def _build_trend_chart(self) -> None:
        """Mini yield trend chart drawn on a tk.Canvas."""
        frame = tk.Frame(self, bg=_BG)
        frame.pack(fill=tk.X, pady=(6, 0))

        tk.Label(
            frame,
            text="\u826f\u7387\u8d8b\u52e2 Yield Trend",
            font=("Helvetica", 9),
            fg=_FG_DIM,
            bg=_BG,
            anchor=tk.W,
        ).pack(fill=tk.X)

        self._chart_canvas = tk.Canvas(
            frame,
            bg=_BG_DARK,
            height=80,
            highlightthickness=0,
        )
        self._chart_canvas.pack(fill=tk.X, pady=(2, 0))

        # Redraw on resize so the chart fills available width.
        self._chart_canvas.bind("<Configure>", self._on_chart_resize)

    def _build_spc_section(self) -> None:
        """SPC indicator labels (hidden until populated)."""
        self._spc_frame = tk.Frame(self, bg=_BG_DARK)
        # Not packed yet -- shown on first update_from_database call.

        self._spc_labels: dict[str, tk.Label] = {}
        metrics = [
            ("mean", t("dashboard.mean_score")),
            ("ucl", "UCL"),
            ("lcl", "LCL"),
            ("cp", "Cp"),
            ("cpk", "Cpk"),
        ]
        for row_idx, (key, display) in enumerate(metrics):
            tk.Label(
                self._spc_frame,
                text=f"{display}:",
                font=("Consolas", 9),
                fg=_FG_DIM,
                bg=_BG_DARK,
                anchor=tk.W,
                padx=6,
            ).grid(row=row_idx, column=0, sticky=tk.W, pady=1)

            lbl = tk.Label(
                self._spc_frame,
                text="--",
                font=("Consolas", 9, "bold"),
                fg=_FG,
                bg=_BG_DARK,
                anchor=tk.E,
                padx=6,
            )
            lbl.grid(row=row_idx, column=1, sticky=tk.E, pady=1)
            self._spc_labels[key] = lbl

        self._spc_frame.columnconfigure(1, weight=1)
        self._spc_visible = False

    # ===================================================================== #
    #  Public API                                                            #
    # ===================================================================== #

    def update_from_result(self, is_pass: bool, score: float = 0.0) -> None:
        """Record a single new inspection result and refresh the display.

        This is the primary method called after each inspection cycle.

        Parameters
        ----------
        is_pass:
            ``True`` if the part passed inspection.
        score:
            Anomaly / confidence score (informational; not used for
            yield calculation but stored for potential SPC use).
        """
        self._total += 1
        if is_pass:
            self._pass_count += 1
        else:
            self._fail_count += 1

        self._recent_results.append(is_pass)

        # Compute rolling yield from the recent results window.
        if self._recent_results:
            rolling_pass = sum(self._recent_results)
            rolling_yield = rolling_pass / len(self._recent_results) * 100.0
        else:
            rolling_yield = 0.0
        self._yield_history.append(rolling_yield)

        # -- Update display elements (no full redraw) --------------------
        self._refresh_yield_display()
        self._refresh_counters()
        self._redraw_trend_chart()

    def update_from_database(self, db: ResultsDatabase) -> None:
        """Pull latest aggregate stats from the database and refresh.

        Useful for initialising the dashboard on startup or after a
        reconnection.  This replaces the internal counters with values
        from the database.

        Parameters
        ----------
        db:
            An open :class:`ResultsDatabase` instance.
        """
        try:
            records = db.query_records(limit=self._trend_window)
        except Exception:
            logger.exception("Failed to query records from database")
            return

        # Records come in descending timestamp order; reverse for
        # chronological processing.
        records.reverse()

        # Rebuild counters from full DB totals.
        try:
            all_records = db.query_records(limit=999_999_999)
            self._total = len(all_records)
            self._pass_count = sum(1 for r in all_records if not r.is_defective)
            self._fail_count = sum(1 for r in all_records if r.is_defective)
        except Exception:
            logger.exception("Failed to load full record count")

        # Rebuild rolling yield from the most recent N records.
        self._recent_results.clear()
        self._yield_history.clear()

        for rec in records:
            is_pass = not rec.is_defective
            self._recent_results.append(is_pass)
            if self._recent_results:
                rolling_pass = sum(self._recent_results)
                rolling_yield = rolling_pass / len(self._recent_results) * 100.0
            else:
                rolling_yield = 0.0
            self._yield_history.append(rolling_yield)

        # SPC metrics.
        try:
            spc = db.compute_spc_metrics(field="anomaly_score")
            self._refresh_spc_display(spc)
        except Exception:
            logger.debug("SPC metrics unavailable", exc_info=True)

        # Refresh all visual elements.
        self._refresh_yield_display()
        self._refresh_counters()
        self._redraw_trend_chart()

    def reset(self) -> None:
        """Clear all counters and return to the initial idle state."""
        self._total = 0
        self._pass_count = 0
        self._fail_count = 0
        self._recent_results.clear()
        self._yield_history.clear()

        # Yield display.
        self._yield_value_label.configure(text="--.--%", fg=_FG_DIM)

        # Counters.
        for lbl in self._counter_labels:
            lbl.configure(text="0")

        # Chart.
        self._chart_canvas.delete("all")

        # SPC section.
        for lbl in self._spc_labels.values():
            lbl.configure(text="--")
        if self._spc_visible:
            self._spc_frame.pack_forget()
            self._spc_visible = False

    # ===================================================================== #
    #  Internal display refresh helpers                                      #
    # ===================================================================== #

    def _refresh_yield_display(self) -> None:
        """Update the large yield percentage text and colour."""
        if self._total == 0:
            self._yield_value_label.configure(text="--.--%", fg=_FG_DIM)
            return

        overall_yield = self._pass_count / self._total * 100.0
        color = _yield_color(overall_yield)
        self._yield_value_label.configure(
            text=f"{overall_yield:.1f}%",
            fg=color,
        )

    def _refresh_counters(self) -> None:
        """Update the three counter labels."""
        self._counter_labels[0].configure(text=str(self._total))
        self._counter_labels[1].configure(text=str(self._pass_count))
        self._counter_labels[2].configure(text=str(self._fail_count))

    # -- Trend chart ------------------------------------------------------ #

    def _on_chart_resize(self, _event: tk.Event) -> None:  # type: ignore[type-arg]
        """Redraw the trend chart when the canvas is resized."""
        self._redraw_trend_chart()

    def _redraw_trend_chart(self) -> None:
        """Redraw the mini yield trend line chart on the canvas."""
        canvas = self._chart_canvas
        canvas.delete("all")

        w = canvas.winfo_width()
        h = canvas.winfo_height()

        # Guard against too-small or not-yet-mapped canvas.
        if w < 20 or h < 20:
            logger.debug("Skipping chart render: canvas too small (%dx%d).", w, h)
            return

        pad_left = 4
        pad_right = 4
        pad_top = 6
        pad_bottom = 6
        plot_w = w - pad_left - pad_right
        plot_h = h - pad_top - pad_bottom

        if plot_w <= 0 or plot_h <= 0:
            logger.debug("Skipping chart render: plot area too small (%dx%d).", plot_w, plot_h)
            return

        data = list(self._yield_history)

        # Determine Y-axis range.
        y_min = 80.0
        y_max = 100.0
        if data:
            data_min = min(data)
            data_max = max(data)
            if data_min < y_min:
                y_min = max(0.0, data_min - 5.0)
            if data_max > y_max:
                y_max = min(100.0, data_max + 5.0)

        y_range = y_max - y_min
        if y_range == 0:
            y_range = 1.0

        def _y_to_canvas(val: float) -> float:
            """Map a yield percentage to canvas Y coordinate."""
            return pad_top + plot_h * (1.0 - (val - y_min) / y_range)

        def _x_to_canvas(idx: int, n: int) -> float:
            """Map a data index to canvas X coordinate."""
            if n <= 1:
                return pad_left + plot_w / 2
            return pad_left + plot_w * idx / (n - 1)

        # -- Reference lines (90% yellow dashed, 95% green dashed) --------
        for ref_val, ref_color in [(90.0, _CHART_REF_90), (95.0, _CHART_REF_95)]:
            if y_min <= ref_val <= y_max:
                ry = _y_to_canvas(ref_val)
                canvas.create_line(
                    pad_left, ry, w - pad_right, ry,
                    fill=ref_color, dash=(4, 4), width=1,
                )
                canvas.create_text(
                    w - pad_right - 2, ry - 6,
                    text=f"{ref_val:.0f}%",
                    fill=ref_color, font=("Consolas", 7),
                    anchor=tk.E,
                )

        # -- Data line ----------------------------------------------------
        if len(data) < 2:
            if len(data) == 1:
                cx = _x_to_canvas(0, 1)
                cy = _y_to_canvas(data[0])
                canvas.create_oval(
                    cx - 3, cy - 3, cx + 3, cy + 3,
                    fill=_CHART_LINE, outline=_CHART_LINE,
                )
            return

        n = len(data)
        coords: list[float] = []
        for i, val in enumerate(data):
            coords.append(_x_to_canvas(i, n))
            coords.append(_y_to_canvas(val))

        canvas.create_line(
            *coords,
            fill=_CHART_LINE, width=1.5, smooth=True,
        )

        # Highlight the most recent point.
        last_x = coords[-2]
        last_y = coords[-1]
        canvas.create_oval(
            last_x - 3, last_y - 3, last_x + 3, last_y + 3,
            fill=_CHART_LINE, outline=_CHART_LINE,
        )

    # -- SPC indicators --------------------------------------------------- #

    def _refresh_spc_display(self, spc: SPCMetrics) -> None:
        """Update the SPC indicator labels from an SPCMetrics dataclass."""
        if spc.n_samples < 2:
            return

        # Show the frame if hidden.
        if not self._spc_visible:
            self._spc_frame.pack(fill=tk.X, pady=(6, 0))
            self._spc_visible = True

        self._spc_labels["mean"].configure(text=f"{spc.mean:.4f}")
        self._spc_labels["ucl"].configure(text=f"{spc.ucl:.4f}")
        self._spc_labels["lcl"].configure(text=f"{spc.lcl:.4f}")

        if spc.cp is not None:
            self._spc_labels["cp"].configure(text=f"{spc.cp:.3f}")
        else:
            self._spc_labels["cp"].configure(text="--")

        if spc.cpk is not None:
            cpk_color = _YIELD_GREEN if spc.cpk >= 1.33 else (
                _YIELD_YELLOW if spc.cpk >= 1.0 else _YIELD_RED
            )
            self._spc_labels["cpk"].configure(text=f"{spc.cpk:.3f}", fg=cpk_color)
        else:
            self._spc_labels["cpk"].configure(text="--", fg=_FG)
