"""Platform-aware keyboard shortcut utilities.

On macOS the primary modifier is Command (⌘), on Windows/Linux it is Control.
This module provides a single source of truth for:

1. Tkinter event sequences   – used in ``bind_all()``
2. Menu accelerator strings  – displayed in menus and tooltips
3. Human-readable labels     – displayed in the shortcuts dialog and welcome overlay
"""

from __future__ import annotations

import sys

IS_MAC = sys.platform == "darwin"

# --- Tkinter modifier tokens ---
# On macOS, Tk maps the Command key to <Command-...> (also <Mod1-...>).
# On Windows/Linux the standard modifier is <Control-...>.
_MOD = "Command" if IS_MAC else "Control"
_MOD_SHIFT = f"{_MOD}-Shift" if IS_MAC else "Control-Shift"

# --- Display strings ---
# Menu accelerator label (shown right-aligned in menus)
MOD_LABEL = "\u2318" if IS_MAC else "Ctrl"  # ⌘ vs Ctrl
MOD_SHIFT_LABEL = f"\u2318\u21e7" if IS_MAC else "Ctrl+Shift"  # ⌘⇧ vs Ctrl+Shift

# Human-readable for dialog / tooltip
MOD_DISPLAY = "\u2318" if IS_MAC else "Ctrl"
SHIFT_SYMBOL = "\u21e7" if IS_MAC else "Shift"
DELETE_LABEL = "\u232b" if IS_MAC else "Delete"


def _seq(modifier: str, key: str) -> str:
    """Build a Tkinter event sequence string, e.g. ``<Command-o>``."""
    return f"<{modifier}-{key}>"


# ------------------------------------------------------------------
# Public helpers
# ------------------------------------------------------------------

def mod_key(key: str) -> str:
    """Return Tkinter event sequence for Mod+key (e.g. ``<Command-o>`` on macOS)."""
    return _seq(_MOD, key)


def mod_shift_key(key: str) -> str:
    """Return Tkinter event sequence for Mod+Shift+key."""
    return _seq(_MOD_SHIFT, key)


def accel(key: str) -> str:
    """Return a menu accelerator display string for Mod+key.

    Examples (macOS): ``⌘O``, ``⌘S``
    Examples (other): ``Ctrl+O``, ``Ctrl+S``
    """
    if IS_MAC:
        return f"{MOD_LABEL}{key.upper()}"
    return f"{MOD_LABEL}+{key.upper()}"


def accel_shift(key: str) -> str:
    """Return a menu accelerator display string for Mod+Shift+key.

    Examples (macOS): ``⌘⇧P``
    Examples (other): ``Ctrl+Shift+P``
    """
    if IS_MAC:
        return f"{MOD_SHIFT_LABEL}{key.upper()}"
    return f"{MOD_SHIFT_LABEL}+{key.upper()}"


def display(key: str) -> str:
    """Return a human-friendly label for Mod+key (for dialogs, tooltips)."""
    if IS_MAC:
        return f"{MOD_DISPLAY}{key.upper()}"
    return f"{MOD_DISPLAY}+{key.upper()}"


def display_shift(key: str) -> str:
    """Return a human-friendly label for Mod+Shift+key."""
    if IS_MAC:
        return f"{MOD_DISPLAY}{SHIFT_SYMBOL}{key.upper()}"
    return f"{MOD_DISPLAY}+{SHIFT_SYMBOL}+{key.upper()}"


def bind_mod(widget, key: str, callback, *, bind_all: bool = False):
    """Bind Mod+key (both cases) to *callback* on *widget*.

    On macOS binds ``<Command-key>`` and ``<Command-KEY>``.
    On other platforms binds ``<Control-key>`` and ``<Control-KEY>``.

    Parameters
    ----------
    widget : tk widget
    key : single character (e.g. ``"o"``)
    callback : event handler ``(event) -> None``
    bind_all : if True, use ``bind_all`` instead of ``bind``
    """
    binder = widget.bind_all if bind_all else widget.bind
    binder(_seq(_MOD, key.lower()), callback)
    binder(_seq(_MOD, key.upper()), callback)


def bind_mod_shift(widget, key: str, callback, *, bind_all: bool = False):
    """Bind Mod+Shift+key (both cases) to *callback* on *widget*."""
    binder = widget.bind_all if bind_all else widget.bind
    binder(_seq(_MOD_SHIFT, key.lower()), callback)
    binder(_seq(_MOD_SHIFT, key.upper()), callback)
