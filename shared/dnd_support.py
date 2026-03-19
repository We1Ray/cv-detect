"""Drag-and-drop support utilities for tkinter.

Provides cross-platform file drop support using tkinterdnd2 if available,
with a graceful fallback to a paste-from-clipboard approach.
"""

from __future__ import annotations

import logging
import os
import platform
import re
import tkinter as tk
from pathlib import Path
from typing import Callable, List, Optional, Sequence

logger = logging.getLogger(__name__)

# Supported image extensions for filtering dropped / pasted paths.
IMAGE_EXTENSIONS: frozenset[str] = frozenset({
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif", ".webp",
})

# ---------------------------------------------------------------------------
# Try to import tkinterdnd2
# ---------------------------------------------------------------------------

_HAS_DND2: bool = False
try:
    import tkinterdnd2  # noqa: F401

    _HAS_DND2 = True
    logger.debug("tkinterdnd2 is available; native drag-and-drop enabled.")
except ImportError:
    logger.info(
        "tkinterdnd2 not installed. Falling back to clipboard-paste "
        "file-drop support (Ctrl+V / Cmd+V)."
    )


def has_native_dnd() -> bool:
    """Return True if tkinterdnd2 is importable."""
    return _HAS_DND2


# ---------------------------------------------------------------------------
# Path parsing helpers
# ---------------------------------------------------------------------------

def _parse_drop_data(data: str) -> List[str]:
    """Parse the raw string from a DnD drop event into file paths.

    tkinterdnd2 encodes paths differently per platform:
    - Windows: ``{C:/path/to file} {D:/other}`` (braces around spaces)
    - Linux/macOS: ``file:///path/to%20file\\r\\nfile:///other``

    Returns a list of normalised, existing file paths.
    """
    paths: List[str] = []

    # Brace-delimited (Windows style)
    brace_matches = re.findall(r"\{([^}]+)\}", data)
    if brace_matches:
        paths.extend(brace_matches)
    else:
        # Try newline-separated file URIs (Linux / macOS)
        for token in re.split(r"[\r\n]+", data):
            token = token.strip()
            if not token:
                continue
            if token.startswith("file://"):
                # Strip the file:// prefix.  On Linux it is file:///path,
                # on macOS it may be file://hostname/path.
                token = re.sub(r"^file://(localhost)?", "", token)
                # URL-decode percent-encoded characters
                from urllib.parse import unquote

                token = unquote(token)
            paths.append(token)

    # If no brace / URI patterns matched, try space-split (simple case)
    if not paths:
        paths = data.strip().split()

    return [p for p in paths if os.path.isfile(p)]


def _parse_clipboard_text(text: str) -> List[str]:
    """Extract valid file paths from clipboard text.

    Handles both newline-separated paths and a single path.
    """
    candidates: List[str] = []
    for line in text.splitlines():
        line = line.strip().strip('"').strip("'")
        if line and os.path.isfile(line):
            candidates.append(line)
    return candidates


def filter_image_paths(
    paths: Sequence[str],
    extensions: frozenset[str] = IMAGE_EXTENSIONS,
) -> List[str]:
    """Keep only paths whose suffix matches *extensions* (case-insensitive)."""
    return [
        p for p in paths
        if Path(p).suffix.lower() in extensions
    ]


# ---------------------------------------------------------------------------
# DnDMixin -- mix into any tk.Widget / ttk.Frame
# ---------------------------------------------------------------------------

class DnDMixin:
    """Mixin that adds drag-and-drop (or clipboard-paste) file support.

    Usage::

        class MyViewer(DnDMixin, ttk.Frame):
            def __init__(self, master, **kw):
                super().__init__(master, **kw)
                self.setup_drop(self, self._on_files_dropped)

            def _on_files_dropped(self, paths: List[str]) -> None:
                print("Received:", paths)

    The mixin will use tkinterdnd2 when available, otherwise it binds
    Ctrl+V (or Cmd+V on macOS) to read file paths from the clipboard.
    """

    def setup_drop(
        self,
        widget: tk.Widget,
        callback: Callable[[List[str]], None],
        *,
        image_only: bool = True,
    ) -> None:
        """Register *widget* to accept dropped files.

        Parameters
        ----------
        widget:
            The tk widget that should accept drops.
        callback:
            Called with a list of absolute file paths when files are dropped
            or pasted.
        image_only:
            If True (default), filter paths to image files only.
        """
        self._dnd_callback = callback
        self._dnd_image_only = image_only
        self._dnd_widget = widget

        if _HAS_DND2:
            self._setup_native_dnd(widget, callback, image_only)
        else:
            self._setup_clipboard_fallback(widget, callback, image_only)

    # -- Native DnD via tkinterdnd2 ----------------------------------------

    @staticmethod
    def _setup_native_dnd(
        widget: tk.Widget,
        callback: Callable[[List[str]], None],
        image_only: bool,
    ) -> None:
        """Register native drop target using tkinterdnd2."""
        try:
            from tkinterdnd2 import DND_FILES

            widget.drop_target_register(DND_FILES)  # type: ignore[attr-defined]

            def _on_drop(event: tk.Event) -> None:
                raw: str = event.data  # type: ignore[attr-defined]
                paths = _parse_drop_data(raw)
                if image_only:
                    paths = filter_image_paths(paths)
                if paths:
                    logger.info("Native DnD: received %d file(s).", len(paths))
                    callback(paths)

            widget.dnd_bind("<<Drop>>", _on_drop)  # type: ignore[attr-defined]
            logger.debug("Native DnD target registered on %s.", widget)
        except Exception:
            logger.warning(
                "Failed to register native DnD target; "
                "falling back to clipboard paste.",
                exc_info=True,
            )
            DnDMixin._setup_clipboard_fallback(widget, callback, image_only)

    # -- Clipboard paste fallback ------------------------------------------

    @staticmethod
    def _setup_clipboard_fallback(
        widget: tk.Widget,
        callback: Callable[[List[str]], None],
        image_only: bool,
    ) -> None:
        """Bind Ctrl+V / Cmd+V to paste file paths from clipboard."""
        is_mac = platform.system() == "Darwin"
        modifier = "Command" if is_mac else "Control"

        def _on_paste(event: tk.Event) -> Optional[str]:
            try:
                text = widget.clipboard_get()
            except tk.TclError:
                logger.debug("Clipboard is empty or inaccessible.")
                return None

            paths = _parse_clipboard_text(text)
            if image_only:
                paths = filter_image_paths(paths)
            if paths:
                logger.info(
                    "Clipboard paste: received %d file path(s).", len(paths),
                )
                callback(paths)
            else:
                logger.debug(
                    "Clipboard text did not contain valid file paths."
                )
            return "break"

        widget.bind(f"<{modifier}-v>", _on_paste)
        widget.bind(f"<{modifier}-V>", _on_paste)
        logger.debug(
            "Clipboard-paste fallback bound on %s (%s+V).", widget, modifier,
        )


# ---------------------------------------------------------------------------
# Standalone helper function
# ---------------------------------------------------------------------------

def setup_drop(
    widget: tk.Widget,
    callback: Callable[[List[str]], None],
    *,
    image_only: bool = True,
) -> None:
    """Convenience function: register *widget* as a file-drop target.

    Uses tkinterdnd2 if available, otherwise binds Ctrl+V / Cmd+V.

    Parameters
    ----------
    widget:
        The tk widget to accept drops.
    callback:
        Called with a ``List[str]`` of absolute file paths.
    image_only:
        If True, only image files are passed through.
    """
    if _HAS_DND2:
        DnDMixin._setup_native_dnd(widget, callback, image_only)
    else:
        DnDMixin._setup_clipboard_fallback(widget, callback, image_only)
