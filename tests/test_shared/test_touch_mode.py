"""Tests for shared.touch_mode -- touch-screen optimisation utilities."""

from __future__ import annotations

import sys
import tkinter as tk

import pytest

from shared.touch_mode import (
    LongPressDetector,
    SwipeDetector,
    TouchMode,
    disable_touch_mode,
    enable_touch_mode,
    is_touch_mode,
)

# Skip the entire module when no display server is available (CI, SSH, etc.)
_display_available = True
try:
    _root = tk.Tk()
    _root.withdraw()
    _root.destroy()
except tk.TclError:
    _display_available = False

pytestmark = pytest.mark.skipif(
    not _display_available,
    reason="No display available (headless environment)",
)


# ------------------------------------------------------------------ #
#  Fixtures                                                            #
# ------------------------------------------------------------------ #

@pytest.fixture()
def root():
    """Create and tear down a Tk root window for each test."""
    r = tk.Tk()
    r.withdraw()
    yield r
    r.destroy()


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset the TouchMode singleton between tests."""
    yield
    # Force a fresh singleton on next instantiation.
    TouchMode._instance = None
    TouchMode._enabled = False
    TouchMode._font_scale = 1.0


# ------------------------------------------------------------------ #
#  TouchMode singleton                                                 #
# ------------------------------------------------------------------ #

class TestTouchModeSingleton:
    """Verify that TouchMode is a proper singleton."""

    def test_same_instance(self) -> None:
        """Two calls to TouchMode() must return the same object."""
        a = TouchMode()
        b = TouchMode()
        assert a is b

    def test_default_disabled(self) -> None:
        """Touch mode should be disabled by default."""
        touch = TouchMode()
        assert touch.is_enabled is False


# ------------------------------------------------------------------ #
#  Enable / Disable                                                    #
# ------------------------------------------------------------------ #

class TestEnableDisable:
    """Test toggling touch mode on and off."""

    def test_enable_sets_flag(self, root: tk.Tk) -> None:
        touch = TouchMode()
        touch.enable(root)
        assert touch.is_enabled is True

    def test_disable_clears_flag(self, root: tk.Tk) -> None:
        touch = TouchMode()
        touch.enable(root)
        touch.disable(root)
        assert touch.is_enabled is False

    def test_module_functions(self, root: tk.Tk) -> None:
        """Module-level enable/disable/is_touch_mode should work."""
        assert is_touch_mode() is False
        enable_touch_mode(root)
        assert is_touch_mode() is True
        disable_touch_mode(root)
        assert is_touch_mode() is False


# ------------------------------------------------------------------ #
#  Font scaling                                                        #
# ------------------------------------------------------------------ #

class TestFontScale:
    """Verify that font_scale is stored correctly."""

    def test_default_scale(self) -> None:
        touch = TouchMode()
        assert touch.font_scale == 1.0

    def test_custom_scale(self, root: tk.Tk) -> None:
        touch = TouchMode()
        touch.enable(root, font_scale=1.5)
        assert touch.font_scale == 1.5

    def test_scale_reset_on_disable(self, root: tk.Tk) -> None:
        touch = TouchMode()
        touch.enable(root, font_scale=2.0)
        touch.disable(root)
        assert touch.font_scale == 1.0


# ------------------------------------------------------------------ #
#  SwipeDetector                                                       #
# ------------------------------------------------------------------ #

class TestSwipeDetector:
    """Basic construction tests for SwipeDetector."""

    def test_init_no_error(self, root: tk.Tk) -> None:
        """SwipeDetector should be created without raising."""
        frame = tk.Frame(root)
        detector = SwipeDetector(frame)
        assert detector.min_distance == 50

    def test_custom_params(self, root: tk.Tk) -> None:
        """Custom parameters should be stored."""
        frame = tk.Frame(root)
        detector = SwipeDetector(
            frame,
            min_distance=100,
            max_time=1.0,
        )
        assert detector.min_distance == 100
        assert detector.max_time == 1.0


# ------------------------------------------------------------------ #
#  LongPressDetector                                                   #
# ------------------------------------------------------------------ #

class TestLongPressDetector:
    """Basic construction tests for LongPressDetector."""

    def test_init_no_error(self, root: tk.Tk) -> None:
        """LongPressDetector should be created without raising."""
        frame = tk.Frame(root)
        detector = LongPressDetector(frame)
        assert detector.delay == 500

    def test_custom_delay(self, root: tk.Tk) -> None:
        """Custom delay should be stored."""
        frame = tk.Frame(root)
        detector = LongPressDetector(frame, delay=1000, motion_threshold=20)
        assert detector.delay == 1000
        assert detector.motion_threshold == 20
