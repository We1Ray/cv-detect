"""Touch-screen optimization for factory floor displays.

Provides utilities to convert standard tkinter UIs for touch-screen use:
- Enlarged button/widget sizes
- Larger fonts
- Touch-friendly spacing
- Swipe gesture detection
- Long-press context menu
"""

from __future__ import annotations

import threading
import time
import tkinter as tk
from tkinter import ttk
import platform as _platform
from typing import Callable, Optional

_SYS = _platform.system()
if _SYS == "Darwin":
    _FONT_FAMILY = "Helvetica Neue"
    _MONO_FAMILY = "Menlo"
elif _SYS == "Linux":
    _FONT_FAMILY = "DejaVu Sans"
    _MONO_FAMILY = "DejaVu Sans Mono"
else:
    _FONT_FAMILY = "Segoe UI"
    _MONO_FAMILY = "Consolas"


# -- Dark theme colour constants (match project palette) -------------------

_COLOR_BG = "#2b2b2b"
_COLOR_BUTTON_BG = "#3c3c3c"
_COLOR_FG = "#e0e0e0"
_COLOR_ACTIVE_BG = "#505050"
_COLOR_TROUGH = "#1e1e1e"


# ======================================================================== #
#  TouchMode singleton                                                      #
# ======================================================================== #

class TouchMode:
    """Singleton manager for touch-screen optimisation.

    Applies enlarged widget sizes, fonts, and spacing suitable for
    factory floor touch panels where operators may be wearing gloves
    and screens are mounted at arm's length.

    Usage
    -----
    ::

        touch = TouchMode()
        touch.enable(root, min_button_height=48, font_scale=1.3)
        # ...later...
        touch.disable(root)
    """

    _instance: Optional[TouchMode] = None
    _lock = threading.Lock()
    _enabled: bool = False
    _font_scale: float = 1.0

    def __new__(cls) -> TouchMode:
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    def __init__(self) -> None:
        # Guard against re-initialisation on repeated __init__ calls.
        if hasattr(self, "_initialised"):
            return
        self._initialised = True
        self._original_styles: dict[str, dict] = {}

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def enable(
        self,
        root: tk.Tk,
        *,
        min_button_height: int = 48,
        font_scale: float = 1.3,
    ) -> None:
        """Enable touch-optimised styles on *root* and all descendants.

        Parameters
        ----------
        root:
            The top-level Tk window.
        min_button_height:
            Minimum button height in pixels (default 48, following
            Android accessibility guidelines).
        font_scale:
            Multiplier applied to base font sizes.
        """
        if min_button_height < 20:
            raise ValueError("min_button_height must be >= 20")
        if font_scale < 0.5 or font_scale > 3.0:
            raise ValueError("font_scale must be between 0.5 and 3.0")
        self._enabled = True
        self._font_scale = font_scale
        style = ttk.Style(root)
        self.configure_styles(style, min_button_height=min_button_height)

    def disable(self, root: tk.Tk) -> None:
        """Revert touch-optimised styles and restore defaults."""
        self._enabled = False
        self._font_scale = 1.0
        style = ttk.Style(root)
        self._restore_styles(style)

    @property
    def is_enabled(self) -> bool:
        """Whether touch mode is currently active."""
        return self._enabled

    @property
    def font_scale(self) -> float:
        """Current font scaling factor."""
        return self._font_scale

    def configure_styles(
        self,
        style: ttk.Style,
        *,
        min_button_height: int = 48,
    ) -> None:
        """Override ttk styles for touch-friendly sizes.

        Parameters
        ----------
        style:
            A ``ttk.Style`` instance to configure.
        min_button_height:
            Minimum button height in pixels.
        """
        scale = self._font_scale

        # Save originals for restore (best-effort).
        self._save_original(style, "TButton")
        self._save_original(style, "TEntry")
        self._save_original(style, "Treeview")
        self._save_original(style, "TCombobox")
        self._save_original(style, "Vertical.TScrollbar")
        self._save_original(style, "Horizontal.TScrollbar")
        self._save_original(style, "TSpinbox")

        # -- TButton --------------------------------------------------- #
        btn_font_size = int(11 * scale)
        # Vertical padding is used to enforce minimum height since ttk
        # buttons do not support a direct height option.
        v_pad = max(8, (min_button_height - btn_font_size - 8) // 2)
        style.configure(
            "TButton",
            font=(_FONT_FAMILY, btn_font_size),
            padding=(12, v_pad),
            background=_COLOR_BUTTON_BG,
            foreground=_COLOR_FG,
        )
        style.map(
            "TButton",
            background=[("active", _COLOR_ACTIVE_BG)],
        )

        # -- TEntry ---------------------------------------------------- #
        entry_font_size = int(11 * scale * (1.2 / 1.3))  # scale * ~1.2
        style.configure(
            "TEntry",
            font=(_FONT_FAMILY, entry_font_size),
            padding=(8, 6),
            foreground=_COLOR_FG,
            fieldbackground=_COLOR_BUTTON_BG,
        )

        # -- Treeview -------------------------------------------------- #
        style.configure(
            "Treeview",
            rowheight=36,
            font=(_FONT_FAMILY, int(10 * scale)),
            background=_COLOR_BG,
            foreground=_COLOR_FG,
            fieldbackground=_COLOR_BG,
        )
        style.configure(
            "Treeview.Heading",
            font=(_FONT_FAMILY, int(10 * scale), "bold"),
            padding=(4, 6),
        )

        # -- TCombobox ------------------------------------------------- #
        combo_font_size = int(11 * scale * (1.2 / 1.3))
        style.configure(
            "TCombobox",
            font=(_FONT_FAMILY, combo_font_size),
            padding=(8, 6),
            foreground=_COLOR_FG,
            fieldbackground=_COLOR_BUTTON_BG,
        )

        # -- Scrollbar (wider for gloved fingers) ---------------------- #
        style.configure(
            "Vertical.TScrollbar",
            width=20,
            troughcolor=_COLOR_TROUGH,
            background=_COLOR_BUTTON_BG,
        )
        style.configure(
            "Horizontal.TScrollbar",
            width=20,
            troughcolor=_COLOR_TROUGH,
            background=_COLOR_BUTTON_BG,
        )

        # -- TSpinbox -------------------------------------------------- #
        spin_font_size = int(11 * scale * (1.2 / 1.3))
        style.configure(
            "TSpinbox",
            font=(_FONT_FAMILY, spin_font_size),
            padding=(8, 6),
            foreground=_COLOR_FG,
            fieldbackground=_COLOR_BUTTON_BG,
        )

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _save_original(self, style: ttk.Style, widget_style: str) -> None:
        """Best-effort snapshot of a style's current configuration."""
        if widget_style in self._original_styles:
            return
        try:
            cfg = style.configure(widget_style)
            if cfg:
                self._original_styles[widget_style] = dict(cfg)
        except tk.TclError:
            pass

    def _restore_styles(self, style: ttk.Style) -> None:
        """Restore previously saved style configurations."""
        for widget_style, cfg in self._original_styles.items():
            try:
                style.configure(widget_style, **cfg)
            except tk.TclError:
                pass
        self._original_styles.clear()


# ======================================================================== #
#  SwipeDetector                                                            #
# ======================================================================== #

class SwipeDetector:
    """Detect horizontal and vertical swipe gestures on a widget.

    Binds ``<ButtonPress-1>`` and ``<ButtonRelease-1>`` to track finger
    movement.  When a release is detected within *max_time* milliseconds
    and the distance exceeds *min_distance* pixels, the appropriate
    callback is invoked.

    Parameters
    ----------
    widget:
        The tkinter widget to monitor.
    min_distance:
        Minimum pixel distance for a gesture to qualify as a swipe.
    max_time:
        Maximum elapsed time (seconds) for the gesture.
    on_swipe_left / on_swipe_right / on_swipe_up / on_swipe_down:
        Callbacks invoked with no arguments when the corresponding
        gesture is detected.
    """

    def __init__(
        self,
        widget: tk.Widget,
        *,
        min_distance: int = 50,
        max_time: float = 0.5,
        on_swipe_left: Optional[Callable[[], None]] = None,
        on_swipe_right: Optional[Callable[[], None]] = None,
        on_swipe_up: Optional[Callable[[], None]] = None,
        on_swipe_down: Optional[Callable[[], None]] = None,
    ) -> None:
        self.widget = widget
        self.min_distance = min_distance
        self.max_time = max_time

        self.on_swipe_left = on_swipe_left
        self.on_swipe_right = on_swipe_right
        self.on_swipe_up = on_swipe_up
        self.on_swipe_down = on_swipe_down

        self._start_x: int = 0
        self._start_y: int = 0
        self._start_time: float = 0.0

        self._press_id: Optional[str] = widget.bind("<ButtonPress-1>", self._on_press, add=True)
        self._release_id: Optional[str] = widget.bind("<ButtonRelease-1>", self._on_release, add=True)

    # ------------------------------------------------------------------ #

    def unbind(self) -> None:
        """Remove only swipe-related event bindings from the widget."""
        if self._press_id:
            self.widget.unbind("<ButtonPress-1>", self._press_id)
            self._press_id = None
        if self._release_id:
            self.widget.unbind("<ButtonRelease-1>", self._release_id)
            self._release_id = None

    def _on_press(self, event: tk.Event) -> None:
        self._start_x = event.x
        self._start_y = event.y
        self._start_time = time.monotonic()

    def _on_release(self, event: tk.Event) -> None:
        elapsed = time.monotonic() - self._start_time
        if elapsed > self.max_time:
            return

        dx = event.x - self._start_x
        dy = event.y - self._start_y

        # Determine dominant axis.
        if abs(dx) > abs(dy) and abs(dx) >= self.min_distance:
            if dx < 0 and self.on_swipe_left:
                self.on_swipe_left()
            elif dx > 0 and self.on_swipe_right:
                self.on_swipe_right()
        elif abs(dy) >= self.min_distance:
            if dy < 0 and self.on_swipe_up:
                self.on_swipe_up()
            elif dy > 0 and self.on_swipe_down:
                self.on_swipe_down()


# ======================================================================== #
#  LongPressDetector                                                        #
# ======================================================================== #

class LongPressDetector:
    """Detect long-press (press-and-hold) gestures on a widget.

    A long press is registered when the user presses and holds for at
    least *delay* milliseconds without moving the pointer beyond
    *motion_threshold* pixels.

    Parameters
    ----------
    widget:
        The tkinter widget to monitor.
    delay:
        Hold duration in milliseconds before firing the callback.
    motion_threshold:
        Maximum pointer movement (pixels) allowed during the hold.
    on_long_press:
        Callback invoked with the original ``tk.Event`` when a long
        press is detected.
    """

    def __init__(
        self,
        widget: tk.Widget,
        *,
        delay: int = 500,
        motion_threshold: int = 10,
        on_long_press: Optional[Callable[[tk.Event], None]] = None,
    ) -> None:
        self.widget = widget
        self.delay = delay
        self.motion_threshold = motion_threshold
        self.on_long_press = on_long_press

        self._after_id: Optional[int] = None
        self._start_x: int = 0
        self._start_y: int = 0
        self._press_event: Optional[tk.Event] = None

        self._press_id: Optional[str] = widget.bind("<ButtonPress-1>", self._on_press, add=True)
        self._release_id: Optional[str] = widget.bind("<ButtonRelease-1>", self._on_release, add=True)
        self._motion_id: Optional[str] = widget.bind("<B1-Motion>", self._on_motion, add=True)

    # ------------------------------------------------------------------ #

    def unbind(self) -> None:
        """Remove only long-press event bindings and cancel any pending timer."""
        if self._press_id:
            self.widget.unbind("<ButtonPress-1>", self._press_id)
            self._press_id = None
        if self._release_id:
            self.widget.unbind("<ButtonRelease-1>", self._release_id)
            self._release_id = None
        if self._motion_id:
            self.widget.unbind("<B1-Motion>", self._motion_id)
            self._motion_id = None
        self._cancel()

    def _on_press(self, event: tk.Event) -> None:
        self._cancel()
        self._start_x = event.x
        self._start_y = event.y
        self._press_event = event
        self._after_id = self.widget.after(self.delay, self._fire)

    def _on_release(self, _event: tk.Event) -> None:
        self._cancel()

    def _on_motion(self, event: tk.Event) -> None:
        dx = abs(event.x - self._start_x)
        dy = abs(event.y - self._start_y)
        if dx > self.motion_threshold or dy > self.motion_threshold:
            self._cancel()

    def _fire(self) -> None:
        self._after_id = None
        if self.on_long_press and self._press_event is not None:
            self.on_long_press(self._press_event)

    def _cancel(self) -> None:
        if self._after_id is not None:
            self.widget.after_cancel(self._after_id)
            self._after_id = None


# ======================================================================== #
#  Module-level convenience functions                                       #
# ======================================================================== #

def enable_touch_mode(
    root: tk.Tk,
    *,
    min_button_height: int = 48,
    font_scale: float = 1.3,
) -> None:
    """Enable touch-screen optimisation on the given root window.

    This is a convenience wrapper around ``TouchMode().enable()``.
    """
    TouchMode().enable(
        root,
        min_button_height=min_button_height,
        font_scale=font_scale,
    )


def disable_touch_mode(root: tk.Tk) -> None:
    """Disable touch-screen optimisation and restore default styles.

    This is a convenience wrapper around ``TouchMode().disable()``.
    """
    TouchMode().disable(root)


def is_touch_mode() -> bool:
    """Return whether touch mode is currently enabled."""
    return TouchMode().is_enabled
