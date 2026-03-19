"""
gui/pixel_inspector.py - Floating pixel value inspector window.

Provides an NxN grid of pixel values around the cursor position,
similar to Vision gray value inspection tools. Non-modal, stays on top.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Optional

import numpy as np


class PixelInspector(tk.Toplevel):
    """Floating pixel value inspector showing an NxN neighbourhood grid.

    The window is non-modal and stays on top of other windows.  Call
    :meth:`update_values` from the main application whenever the mouse
    moves over the image viewer.
    """

    # Theme colours
    _BG = "#2b2b2b"
    _FG = "#cccccc"
    _CANVAS_BG = "#1e1e1e"
    _ACCENT = "#0078d4"
    _BORDER_NORMAL = "#555555"
    _BORDER_CENTER = "#0078d4"

    def __init__(self, master: tk.Widget) -> None:
        super().__init__(master)
        self.title("Pixel Inspector")
        self.geometry("380x380")
        self.resizable(True, True)
        self.configure(bg=self._BG)

        # Non-modal: stay on top, no grab
        self.attributes("-topmost", True)

        # Current grid half-size (radius). Full size = 2*radius + 1.
        self._grid_size: int = 7

        # Cell label widgets (row, col) -> tk.Label
        self._cells: dict[tuple[int, int], tk.Label] = {}

        self._build_ui()
        self._rebuild_grid()

        self.protocol("WM_DELETE_WINDOW", self.withdraw)

    # ------------------------------------------------------------------ #
    #  UI construction                                                     #
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        """Build the top control bar and the grid container frame."""
        # -- Top bar: coordinate display + grid size selector --
        top_frame = tk.Frame(self, bg=self._BG)
        top_frame.pack(fill=tk.X, padx=6, pady=(6, 2))

        self._coord_label = tk.Label(
            top_frame,
            text="X: --  Y: --",
            bg=self._BG,
            fg=self._FG,
            font=("Consolas", 10),
            anchor=tk.W,
        )
        self._coord_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        tk.Label(
            top_frame,
            text="Grid:",
            bg=self._BG,
            fg=self._FG,
            font=("Consolas", 9),
        ).pack(side=tk.LEFT, padx=(8, 2))

        self._size_var = tk.StringVar(value=str(self._grid_size))
        size_combo = ttk.Combobox(
            top_frame,
            textvariable=self._size_var,
            values=["3", "5", "7", "9", "11"],
            state="readonly",
            width=4,
        )
        size_combo.pack(side=tk.LEFT)
        size_combo.bind("<<ComboboxSelected>>", self._on_size_changed)

        # -- Grid container --
        self._grid_frame = tk.Frame(self, bg=self._CANVAS_BG)
        self._grid_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

    # ------------------------------------------------------------------ #
    #  Grid management                                                     #
    # ------------------------------------------------------------------ #

    def _rebuild_grid(self) -> None:
        """Destroy existing cells and create a new NxN grid of labels."""
        # Destroy old widgets
        for widget in self._grid_frame.winfo_children():
            widget.destroy()
        self._cells.clear()

        n = self._grid_size

        # Configure equal weights so cells expand evenly
        for i in range(n):
            self._grid_frame.rowconfigure(i, weight=1)
            self._grid_frame.columnconfigure(i, weight=1)

        center = n // 2

        for r in range(n):
            for c in range(n):
                is_center = (r == center and c == center)
                bd_color = self._BORDER_CENTER if is_center else self._BORDER_NORMAL
                bd_width = 2 if is_center else 1

                # Wrapper frame for coloured border
                wrapper = tk.Frame(
                    self._grid_frame,
                    bg=bd_color,
                    padx=bd_width,
                    pady=bd_width,
                )
                wrapper.grid(row=r, column=c, sticky="nsew", padx=0, pady=0)
                wrapper.rowconfigure(0, weight=1)
                wrapper.columnconfigure(0, weight=1)

                cell = tk.Label(
                    wrapper,
                    text="--",
                    bg="#1a1a1a",
                    fg=self._FG,
                    font=("Consolas", 8),
                    anchor=tk.CENTER,
                )
                cell.grid(row=0, column=0, sticky="nsew")
                self._cells[(r, c)] = cell

    def _on_size_changed(self, _event: tk.Event) -> None:
        """Handle grid size combobox selection."""
        try:
            new_size = int(self._size_var.get())
        except ValueError:
            return
        if new_size != self._grid_size and new_size in (3, 5, 7, 9, 11):
            self._grid_size = new_size
            self._rebuild_grid()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def update_values(self, image: np.ndarray, cx: int, cy: int) -> None:
        """Update the inspector grid with pixel values around (*cx*, *cy*).

        Args:
            image: Source image (grayscale or BGR colour, any dtype).
            cx: X coordinate (column) in the image.
            cy: Y coordinate (row) in the image.
        """
        self._coord_label.configure(text=f"X: {cx}  Y: {cy}")

        h, w = image.shape[:2]
        is_color = image.ndim == 3
        n = self._grid_size
        radius = n // 2

        for r in range(n):
            for c in range(n):
                iy = cy - radius + r
                ix = cx - radius + c
                cell = self._cells.get((r, c))
                if cell is None:
                    continue

                if 0 <= ix < w and 0 <= iy < h:
                    pixel = image[iy, ix]
                    if is_color:
                        # BGR order from OpenCV -- display as R,G,B
                        if isinstance(pixel, np.ndarray) and pixel.size >= 3:
                            b_val = int(pixel[0])
                            g_val = int(pixel[1])
                            r_val = int(pixel[2])
                            text = f"{r_val},{g_val},{b_val}"
                            # Perceived luminance for background
                            lum = int(0.299 * r_val + 0.587 * g_val + 0.114 * b_val)
                        else:
                            val = int(pixel)
                            text = str(val)
                            lum = val
                    else:
                        # Grayscale
                        if image.dtype == np.uint8:
                            val = int(pixel)
                        else:
                            # Normalise to 0-255 for display
                            img_min = image.min()
                            img_max = image.max()
                            if img_max - img_min > 0:
                                val = int(
                                    (float(pixel) - img_min)
                                    / (img_max - img_min)
                                    * 255
                                )
                            else:
                                val = 0
                        text = str(int(pixel)) if image.dtype == np.uint8 else f"{float(pixel):.1f}"
                        lum = val

                    # Compute background and foreground for contrast
                    bg_hex = self._lum_to_bg(lum)
                    fg_hex = "#000000" if lum > 128 else "#ffffff"

                    cell.configure(text=text, bg=bg_hex, fg=fg_hex)
                else:
                    # Out of bounds
                    cell.configure(text="--", bg="#1a1a1a", fg="#555555")

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _lum_to_bg(lum: int) -> str:
        """Map a luminance value (0-255) to a background hex colour.

        Dark pixels get a dark bg, bright pixels get a lighter bg.
        The mapping is slightly compressed so that pure black and pure
        white remain readable.
        """
        # Compress to 20..235 range for readability
        mapped = int(20 + (lum / 255) * 215)
        mapped = max(0, min(255, mapped))
        return f"#{mapped:02x}{mapped:02x}{mapped:02x}"
