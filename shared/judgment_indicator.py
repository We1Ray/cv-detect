"""Large OK/NG judgment indicator widget for industrial inspection.

Provides a highly visible pass/fail indicator with:
- Full-width color banner (green=PASS, red=FAIL, gray=IDLE)
- Large text (configurable font size)
- Flash animation for attention
- Sound notification support (optional)
- Statistics counters (total, pass, fail, yield rate)
- Timestamp of last judgment
"""

from __future__ import annotations

import tkinter as tk
from datetime import datetime
from typing import Any, Dict

import platform as _platform

from shared.i18n import t

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


# -- Colour constants --------------------------------------------------------

_COLOR_PASS = "#00C853"
_COLOR_FAIL = "#FF1744"
_COLOR_IDLE = "#424242"
_COLOR_FLASH = "#FFFFFF"
_COLOR_TEXT = "#FFFFFF"
_COLOR_STATS_BG = "#1e1e1e"
_COLOR_STATS_FG = "#e0e0e0"


class JudgmentIndicator(tk.Frame):
    """A large, prominent OK/NG indicator designed to be visible from a
    distance on a production line.

    Parameters
    ----------
    master:
        Parent widget.
    height:
        Desired height (pixels) of the main banner area.
    flash_duration:
        Duration in milliseconds of each flash step.
    flash_count:
        Number of on/off flash cycles when a result is set.
    font_size:
        Font size for the main result text.  Defaults to 48 for
        maximum readability at a distance.
    """

    def __init__(
        self,
        master: tk.Misc,
        *,
        height: int = 120,
        flash_duration: int = 500,
        flash_count: int = 3,
        font_size: int = 48,
    ) -> None:
        super().__init__(master, bg=_COLOR_IDLE, height=height)
        self.pack_propagate(False)

        self._flash_duration = flash_duration
        self._flash_count = flash_count
        self._flash_after_id: str | None = None

        # -- Statistics state -------------------------------------------------
        self._total = 0
        self._pass_count = 0
        self._fail_count = 0
        self._last_timestamp: str = ""

        # -- Current result colour (used by flash) ----------------------------
        self._current_color: str = _COLOR_IDLE

        # =====================================================================
        #  Layout: banner -> score row -> stats row
        # =====================================================================

        # -- Banner (result label) --------------------------------------------
        self._banner = tk.Label(
            self,
            text=t("judgment.idle"),
            font=("Helvetica", font_size, "bold"),
            fg=_COLOR_TEXT,
            bg=_COLOR_IDLE,
            anchor=tk.CENTER,
        )
        self._banner.pack(fill=tk.BOTH, expand=True)

        # -- Score / message row ----------------------------------------------
        self._score_frame = tk.Frame(self, bg=_COLOR_STATS_BG)
        self._score_frame.pack(fill=tk.X)

        self._score_label = tk.Label(
            self._score_frame,
            text=f"{t('judgment.score')}: --",
            font=(_MONO_FAMILY, 12),
            fg=_COLOR_STATS_FG,
            bg=_COLOR_STATS_BG,
            padx=8,
            pady=2,
        )
        self._score_label.pack(side=tk.LEFT)

        self._message_label = tk.Label(
            self._score_frame,
            text="",
            font=("Helvetica", 11),
            fg=_COLOR_STATS_FG,
            bg=_COLOR_STATS_BG,
            padx=8,
            pady=2,
            anchor=tk.W,
        )
        self._message_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # -- Statistics bar ---------------------------------------------------
        self._stats_frame = tk.Frame(self, bg=_COLOR_STATS_BG)
        self._stats_frame.pack(fill=tk.X)

        self._stats_label = tk.Label(
            self._stats_frame,
            text=f"{t('judgment.total')}: 0 | {t('judgment.pass_count')}: 0 | {t('judgment.fail_count')}: 0 | {t('judgment.yield_rate')}: --",
            font=(_MONO_FAMILY, 11),
            fg=_COLOR_STATS_FG,
            bg=_COLOR_STATS_BG,
            padx=8,
            pady=2,
        )
        self._stats_label.pack(side=tk.LEFT)

        self._timestamp_label = tk.Label(
            self._stats_frame,
            text=f"{t('judgment.last_time')}: --",
            font=(_MONO_FAMILY, 11),
            fg="#888888",
            bg=_COLOR_STATS_BG,
            padx=8,
            pady=2,
        )
        self._timestamp_label.pack(side=tk.RIGHT)

    # --------------------------------------------------------------------- #
    #  Public API                                                             #
    # --------------------------------------------------------------------- #

    def set_result(
        self,
        is_pass: bool,
        score: float = 0.0,
        message: str = "",
    ) -> None:
        """Display a PASS or FAIL result and update statistics.

        Parameters
        ----------
        is_pass:
            ``True`` for PASS/OK, ``False`` for FAIL/NG.
        score:
            Numeric confidence or defect score.
        message:
            Optional short description shown beneath the banner.
        """
        # Cancel any running flash animation.
        self._cancel_flash()

        # Update statistics.
        self._total += 1
        if is_pass:
            self._pass_count += 1
        else:
            self._fail_count += 1

        now = datetime.now()
        self._last_timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

        # Determine display values.
        color = _COLOR_PASS if is_pass else _COLOR_FAIL
        text = f"{t('judgment.pass')}  OK" if is_pass else f"{t('judgment.fail')}  NG"
        self._current_color = color

        # Apply to widgets.
        self._set_banner(color, text)
        self._score_label.configure(text=f"{t('judgment.score')}: {score:.4f}")
        self._message_label.configure(text=message)
        self._update_stats_display()
        self._timestamp_label.configure(text=f"{t('judgment.last_time')}: {self._last_timestamp}")

        # Start flash animation.
        self._flash(color, self._flash_count * 2)

    def reset(self) -> None:
        """Return the indicator to the idle (standby) state."""
        self._cancel_flash()
        self._current_color = _COLOR_IDLE
        self._set_banner(_COLOR_IDLE, t("judgment.idle"))
        self._score_label.configure(text=f"{t('judgment.score')}: --")
        self._message_label.configure(text="")
        self._timestamp_label.configure(text=f"{t('judgment.last_time')}: --")

    def get_statistics(self) -> Dict[str, Any]:
        """Return current inspection statistics.

        Returns
        -------
        dict
            Keys: ``total``, ``pass_count``, ``fail_count``, ``yield_rate``.
            ``yield_rate`` is a float in [0, 100] or ``None`` if no
            inspections have been recorded.
        """
        yield_rate: float | None = None
        if self._total > 0:
            yield_rate = round(self._pass_count / self._total * 100, 1)
        return {
            "total": self._total,
            "pass_count": self._pass_count,
            "fail_count": self._fail_count,
            "yield_rate": yield_rate,
        }

    def reset_statistics(self) -> None:
        """Reset all counters to zero."""
        self._total = 0
        self._pass_count = 0
        self._fail_count = 0
        self._last_timestamp = ""
        self._update_stats_display()

    # --------------------------------------------------------------------- #
    #  Internal helpers                                                       #
    # --------------------------------------------------------------------- #

    def _set_banner(self, bg: str, text: str) -> None:
        """Set the banner background colour and text."""
        self.configure(bg=bg)
        self._banner.configure(bg=bg, text=text)

    def _update_stats_display(self) -> None:
        """Refresh the statistics label from internal counters."""
        stats = self.get_statistics()
        yr = f"{stats['yield_rate']:.1f}%" if stats["yield_rate"] is not None else "--"
        self._stats_label.configure(
            text=(
                f"{t('judgment.total')}: {stats['total']} | "
                f"{t('judgment.pass_count')}: {stats['pass_count']} | "
                f"{t('judgment.fail_count')}: {stats['fail_count']} | "
                f"{t('judgment.yield_rate')}: {yr}"
            ),
        )

    # -- Flash animation -------------------------------------------------- #

    def _flash(self, color: str, remaining: int) -> None:
        """Alternate the banner between *color* and a flash colour.

        Uses ``tk.after()`` for non-blocking animation.  *remaining* counts
        the number of half-cycles still to execute.
        """
        if remaining <= 0:
            # Ensure we finish on the result colour.
            self._set_banner(self._current_color, self._banner.cget("text"))
            self._flash_after_id = None
            return

        # Odd remaining -> flash colour; even -> result colour.
        if remaining % 2 == 1:
            self._banner.configure(bg=_COLOR_FLASH, fg="#000000")
            self.configure(bg=_COLOR_FLASH)
        else:
            self._banner.configure(bg=color, fg=_COLOR_TEXT)
            self.configure(bg=color)

        try:
            self._flash_after_id = self.after(
                self._flash_duration,
                self._flash,
                color,
                remaining - 1,
            )
        except tk.TclError:
            # Widget is being destroyed; silently abandon the animation.
            self._flash_after_id = None

    def destroy(self) -> None:
        """Cancel pending flash animations before destroying the widget."""
        self._cancel_flash()
        super().destroy()

    def _cancel_flash(self) -> None:
        """Cancel any pending flash animation callback."""
        if self._flash_after_id is not None:
            self.after_cancel(self._flash_after_id)
            self._flash_after_id = None
