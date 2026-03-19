"""Widget-level tests for shared UI components.

Each test creates a temporary Tk root, exercises the widget API,
and destroys the root afterwards.  The entire module is skipped
when no display server is available (typical in headless CI).
"""

from __future__ import annotations

import os
import sys

import pytest

# Skip all GUI tests if no display available.
pytestmark = pytest.mark.skipif(
    os.environ.get("DISPLAY") is None
    and os.environ.get("WAYLAND_DISPLAY") is None
    and not os.path.exists("/tmp/.X11-unix")
    and sys.platform != "darwin",
    reason="No display available for GUI tests",
)


# -- Helpers ----------------------------------------------------------------- #


@pytest.fixture()
def tk_root():
    """Create a temporary Tk root and destroy it after the test."""
    import tkinter as tk

    root = tk.Tk()
    root.withdraw()  # Hide the window during tests.
    yield root
    root.destroy()


# =========================================================================== #
#  JudgmentIndicator tests                                                     #
# =========================================================================== #


class TestJudgmentIndicator:
    """Tests for :class:`shared.judgment_indicator.JudgmentIndicator`."""

    def _make(self, root):
        from shared.judgment_indicator import JudgmentIndicator

        return JudgmentIndicator(root, flash_count=0, flash_duration=0)

    def test_create(self, tk_root):
        """Widget can be created without error; initial state is idle."""
        indicator = self._make(tk_root)
        indicator.pack()
        tk_root.update_idletasks()

        stats = indicator.get_statistics()
        assert stats["total"] == 0
        assert stats["pass_count"] == 0
        assert stats["fail_count"] == 0
        assert stats["yield_rate"] is None

    def test_set_pass(self, tk_root):
        """set_result(True, ...) updates the display without error."""
        indicator = self._make(tk_root)
        indicator.pack()
        tk_root.update_idletasks()

        indicator.set_result(True, 0.1)
        tk_root.update_idletasks()

        stats = indicator.get_statistics()
        assert stats["total"] == 1
        assert stats["pass_count"] == 1
        assert stats["fail_count"] == 0

    def test_set_fail(self, tk_root):
        """set_result(False, ...) updates the display without error."""
        indicator = self._make(tk_root)
        indicator.pack()
        tk_root.update_idletasks()

        indicator.set_result(False, 0.8)
        tk_root.update_idletasks()

        stats = indicator.get_statistics()
        assert stats["total"] == 1
        assert stats["pass_count"] == 0
        assert stats["fail_count"] == 1

    def test_statistics(self, tk_root):
        """After 3 pass + 2 fail, get_statistics returns correct counts."""
        indicator = self._make(tk_root)
        indicator.pack()
        tk_root.update_idletasks()

        for _ in range(3):
            indicator.set_result(True, 0.05)
        for _ in range(2):
            indicator.set_result(False, 0.9)
        tk_root.update_idletasks()

        stats = indicator.get_statistics()
        assert stats["total"] == 5
        assert stats["pass_count"] == 3
        assert stats["fail_count"] == 2
        assert stats["yield_rate"] == 60.0

    def test_reset(self, tk_root):
        """After reset, indicator returns to idle display state."""
        indicator = self._make(tk_root)
        indicator.pack()
        tk_root.update_idletasks()

        indicator.set_result(True, 0.1)
        tk_root.update_idletasks()

        indicator.reset()
        tk_root.update_idletasks()

        # Banner text should be the idle text.
        banner_text = indicator._banner.cget("text")
        assert banner_text == "待機中"

    def test_reset_statistics(self, tk_root):
        """reset_statistics sets all counters to zero."""
        indicator = self._make(tk_root)
        indicator.pack()
        tk_root.update_idletasks()

        indicator.set_result(True, 0.1)
        indicator.set_result(False, 0.9)
        tk_root.update_idletasks()

        indicator.reset_statistics()
        tk_root.update_idletasks()

        stats = indicator.get_statistics()
        assert stats["total"] == 0
        assert stats["pass_count"] == 0
        assert stats["fail_count"] == 0
        assert stats["yield_rate"] is None


# =========================================================================== #
#  LiveInspectionPanel tests                                                   #
# =========================================================================== #


class TestLiveInspectionPanel:
    """Tests for :class:`shared.live_inspection_panel.LiveInspectionPanel`."""

    def _make(self, root):
        from shared.live_inspection_panel import LiveInspectionPanel

        return LiveInspectionPanel(
            root,
            on_start=lambda source, interval: None,
            on_stop=lambda: None,
            on_inspect_single=lambda: None,
        )

    def test_create(self, tk_root):
        """Widget can be created without error."""
        panel = self._make(tk_root)
        panel.pack()
        tk_root.update_idletasks()

    def test_is_running_default_false(self, tk_root):
        """Initial running state is False."""
        panel = self._make(tk_root)
        panel.pack()
        tk_root.update_idletasks()

        assert panel.is_running is False

    def test_update_result(self, tk_root):
        """update_result does not crash."""
        panel = self._make(tk_root)
        panel.pack()
        tk_root.update_idletasks()

        panel.update_result(True, 0.05, "test_image.png")
        tk_root.update_idletasks()

        # Verify counter was incremented internally.
        assert panel._inspected == 1
        assert panel._pass_count == 1


# =========================================================================== #
#  DashboardPanel tests                                                        #
# =========================================================================== #


class TestDashboardPanel:
    """Tests for :class:`shared.dashboard_panel.DashboardPanel`."""

    def _make(self, root):
        from shared.dashboard_panel import DashboardPanel

        return DashboardPanel(root, trend_window=10)

    def test_create(self, tk_root):
        """Widget can be created without error."""
        panel = self._make(tk_root)
        panel.pack()
        tk_root.update_idletasks()

        assert panel._total == 0
        assert panel._pass_count == 0
        assert panel._fail_count == 0

    def test_update_from_result(self, tk_root):
        """update_from_result correctly updates pass/fail counters."""
        panel = self._make(tk_root)
        panel.pack()
        tk_root.update_idletasks()

        panel.update_from_result(True, 0.05)
        panel.update_from_result(True, 0.03)
        panel.update_from_result(False, 0.85)
        tk_root.update_idletasks()

        assert panel._total == 3
        assert panel._pass_count == 2
        assert panel._fail_count == 1

    def test_reset(self, tk_root):
        """reset clears all state back to initial values."""
        panel = self._make(tk_root)
        panel.pack()
        tk_root.update_idletasks()

        panel.update_from_result(True, 0.1)
        panel.update_from_result(False, 0.9)
        tk_root.update_idletasks()

        panel.reset()
        tk_root.update_idletasks()

        assert panel._total == 0
        assert panel._pass_count == 0
        assert panel._fail_count == 0
        assert len(panel._recent_results) == 0
        assert len(panel._yield_history) == 0
