"""
gui/threshold_dialog.py - Interactive threshold segmentation dialog.

Provides a modal dialog with:
- 256-bin histogram with highlighted selection range
- Min/max gray level sliders
- Live preview with red overlay on selected pixels
- Otsu auto-threshold button
- Accept/cancel producing a Region object
"""

from __future__ import annotations

import logging
import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

logger = logging.getLogger(__name__)

import platform as _platform
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

# Theme constants
_BG = "#2b2b2b"
_BG_MEDIUM = "#3c3c3c"
_FG = "#cccccc"
_FG_WHITE = "#e0e0e0"
_CANVAS_BG = "#1e1e1e"
_ACCENT = "#0078d4"
_HIST_BAR_INACTIVE = "#555555"
_HIST_BAR_ACTIVE = "#2a7ad4"
_OVERLAY_COLOR = (255, 0, 0)  # Red overlay in RGB


class ThresholdDialog(tk.Toplevel):
    """Interactive threshold segmentation dialog with histogram and live preview.

    Parameters
    ----------
    master : tk.Widget
        Parent widget.
    image : np.ndarray
        Source image (BGR colour or grayscale, any dtype).
    on_accept : callable, optional
        Callback invoked on accept with signature
        ``on_accept(region, display_image, name)``.
    """

    def __init__(
        self,
        master: tk.Widget,
        image: np.ndarray,
        on_accept: Optional[Callable] = None,
    ) -> None:
        super().__init__(master)
        self.title("Threshold Segmentation")
        self.geometry("520x720")
        self.resizable(True, True)
        self.configure(bg=_BG)

        # Modal behaviour
        self.transient(master)
        self.grab_set()

        self._source_image = image.copy()
        self._gray = self._to_gray(image)
        self._on_accept = on_accept

        # Histogram data (computed once)
        self._hist = cv2.calcHist(
            [self._gray], [0], None, [256], [0, 256]
        ).flatten()
        self._total_pixels = int(self._gray.shape[0] * self._gray.shape[1])

        # Slider variables
        self._min_var = tk.IntVar(value=0)
        self._max_var = tk.IntVar(value=255)

        # Debounce timer id
        self._update_after_id: Optional[str] = None

        # PhotoImage references (prevent GC)
        self._preview_photo: Optional[ImageTk.PhotoImage] = None

        self._build_ui()

        # Initial draw after the window is mapped
        self.after(50, self._full_update)

        self.protocol("WM_DELETE_WINDOW", self._cancel)

    # ------------------------------------------------------------------ #
    #  UI construction                                                     #
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        """Build all UI components top to bottom."""
        # -- 1. Histogram canvas --
        hist_label = tk.Label(
            self, text="直方圖", bg=_BG, fg=_FG,
            font=("", 9, "bold"), anchor=tk.W,
        )
        hist_label.pack(fill=tk.X, padx=10, pady=(8, 0))

        self._hist_canvas = tk.Canvas(
            self, bg=_CANVAS_BG, highlightthickness=0, height=130,
        )
        self._hist_canvas.pack(fill=tk.X, padx=10, pady=(2, 4))
        self._hist_canvas.bind("<Configure>", lambda e: self._draw_histogram())

        # -- 2. Auto-threshold buttons --
        auto_frame = tk.Frame(self, bg=_BG)
        auto_frame.pack(fill=tk.X, padx=10, pady=(0, 4))

        tk.Label(
            auto_frame, text="自動：", bg=_BG, fg=_FG, font=("", 9),
        ).pack(side=tk.LEFT, padx=(0, 6))

        otsu_btn = tk.Button(
            auto_frame,
            text="Otsu",
            bg=_BG_MEDIUM,
            fg=_FG_WHITE,
            activebackground=_ACCENT,
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=10,
            pady=2,
            command=self._apply_otsu,
        )
        otsu_btn.pack(side=tk.LEFT, padx=2)

        adaptive_btn = tk.Button(
            auto_frame,
            text="\u81ea\u9069\u61c9",  # "自適應"
            bg=_BG_MEDIUM,
            fg="#777777",
            activebackground=_BG_MEDIUM,
            activeforeground="#777777",
            relief=tk.FLAT,
            padx=10,
            pady=2,
            state=tk.DISABLED,
        )
        adaptive_btn.pack(side=tk.LEFT, padx=2)

        # -- 3. Sliders frame --
        slider_frame = tk.Frame(self, bg=_BG)
        slider_frame.pack(fill=tk.X, padx=10, pady=4)

        # Min slider row
        min_row = tk.Frame(slider_frame, bg=_BG)
        min_row.pack(fill=tk.X, pady=2)

        tk.Label(
            min_row, text="\u6700\u5c0f\u7070\u5ea6:",  # "最小灰度:"
            bg=_BG, fg=_FG, font=("", 9), width=8, anchor=tk.E,
        ).pack(side=tk.LEFT)

        self._min_scale = tk.Scale(
            min_row,
            from_=0,
            to=255,
            orient=tk.HORIZONTAL,
            variable=self._min_var,
            bg=_BG,
            fg=_FG,
            troughcolor=_BG_MEDIUM,
            highlightthickness=0,
            sliderrelief=tk.FLAT,
            showvalue=False,
            command=self._on_slider_moved,
        )
        self._min_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        self._min_val_label = tk.Label(
            min_row, text="0", bg=_BG, fg=_FG_WHITE,
            font=(_MONO_FAMILY, 10), width=4, anchor=tk.W,
        )
        self._min_val_label.pack(side=tk.LEFT)

        # Max slider row
        max_row = tk.Frame(slider_frame, bg=_BG)
        max_row.pack(fill=tk.X, pady=2)

        tk.Label(
            max_row, text="\u6700\u5927\u7070\u5ea6:",  # "最大灰度:"
            bg=_BG, fg=_FG, font=("", 9), width=8, anchor=tk.E,
        ).pack(side=tk.LEFT)

        self._max_scale = tk.Scale(
            max_row,
            from_=0,
            to=255,
            orient=tk.HORIZONTAL,
            variable=self._max_var,
            bg=_BG,
            fg=_FG,
            troughcolor=_BG_MEDIUM,
            highlightthickness=0,
            sliderrelief=tk.FLAT,
            showvalue=False,
            command=self._on_slider_moved,
        )
        self._max_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        self._max_val_label = tk.Label(
            max_row, text="255", bg=_BG, fg=_FG_WHITE,
            font=(_MONO_FAMILY, 10), width=4, anchor=tk.W,
        )
        self._max_val_label.pack(side=tk.LEFT)

        # -- 4. Info label --
        self._info_label = tk.Label(
            self,
            text="\u9078\u53d6\u50cf\u7d20: -- / -- (--%)",  # "選取像素: ..."
            bg=_BG,
            fg=_FG,
            font=(_MONO_FAMILY, 9),
            anchor=tk.W,
        )
        self._info_label.pack(fill=tk.X, padx=10, pady=(2, 4))

        # -- 5. Preview canvas --
        preview_label = tk.Label(
            self, text="預覽", bg=_BG, fg=_FG,
            font=("", 9, "bold"), anchor=tk.W,
        )
        preview_label.pack(fill=tk.X, padx=10, pady=(4, 0))

        self._preview_canvas = tk.Canvas(
            self, bg=_CANVAS_BG, highlightthickness=0,
        )
        self._preview_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=(2, 6))
        self._preview_canvas.bind("<Configure>", lambda e: self._schedule_update())

        # -- 6. Buttons --
        btn_frame = tk.Frame(self, bg=_BG)
        btn_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        accept_btn = tk.Button(
            btn_frame,
            text="\u78ba\u5b9a",  # "確定"
            bg=_ACCENT,
            fg="#ffffff",
            activebackground="#005a9e",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=20,
            pady=4,
            font=("", 10),
            command=self._accept,
        )
        accept_btn.pack(side=tk.RIGHT, padx=(6, 0))

        cancel_btn = tk.Button(
            btn_frame,
            text="\u53d6\u6d88",  # "取消"
            bg=_BG_MEDIUM,
            fg=_FG_WHITE,
            activebackground="#555555",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=20,
            pady=4,
            font=("", 10),
            command=self._cancel,
        )
        cancel_btn.pack(side=tk.RIGHT)

    # ------------------------------------------------------------------ #
    #  Histogram drawing                                                   #
    # ------------------------------------------------------------------ #

    def _draw_histogram(self) -> None:
        """Draw the 256-bin histogram on the canvas, highlighting the
        selected [min, max] range in accent colour."""
        self._hist_canvas.delete("all")
        cw = self._hist_canvas.winfo_width()
        ch = self._hist_canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            return

        margin_bottom = 18
        margin_top = 4
        plot_h = ch - margin_bottom - margin_top

        max_count = self._hist.max()
        if max_count == 0:
            return

        lo = self._min_var.get()
        hi = self._max_var.get()

        bar_w = cw / 256.0

        for i in range(256):
            bar_height = (self._hist[i] / max_count) * plot_h
            x1 = i * bar_w
            x2 = (i + 1) * bar_w
            y1 = ch - margin_bottom - bar_height
            y2 = ch - margin_bottom

            if lo <= i <= hi:
                fill = _HIST_BAR_ACTIVE
            else:
                fill = _HIST_BAR_INACTIVE

            self._hist_canvas.create_rectangle(
                x1, y1, x2, y2, fill=fill, outline="", width=0,
            )

        # Draw range indicator lines
        lo_x = lo * bar_w
        hi_x = (hi + 1) * bar_w
        self._hist_canvas.create_line(
            lo_x, margin_top, lo_x, ch - margin_bottom,
            fill="#ff6666", width=1, dash=(3, 2),
        )
        self._hist_canvas.create_line(
            hi_x, margin_top, hi_x, ch - margin_bottom,
            fill="#ff6666", width=1, dash=(3, 2),
        )

        # Axis labels
        self._hist_canvas.create_text(
            2, ch - 2, text="0", fill="#888888", font=("", 7), anchor=tk.SW,
        )
        self._hist_canvas.create_text(
            cw - 2, ch - 2, text="255", fill="#888888", font=("", 7), anchor=tk.SE,
        )

    # ------------------------------------------------------------------ #
    #  Preview update                                                      #
    # ------------------------------------------------------------------ #

    def _update_preview(self) -> None:
        """Create a preview image with red overlay on selected pixels and
        update the info label with pixel counts."""
        self._update_after_id = None

        lo = self._min_var.get()
        hi = self._max_var.get()

        # Update value labels
        self._min_val_label.configure(text=str(lo))
        self._max_val_label.configure(text=str(hi))

        # Binary mask of selected pixels
        mask = cv2.inRange(self._gray, lo, hi)  # uint8 0/255

        # Count selected pixels
        selected_count = int(cv2.countNonZero(mask))
        if self._total_pixels > 0:
            pct = selected_count / self._total_pixels * 100
        else:
            pct = 0.0
        self._info_label.configure(
            text=(
                f"\u9078\u53d6\u50cf\u7d20: {selected_count} / "
                f"{self._total_pixels} ({pct:.1f}%)"
            ),
        )

        # Build composite preview: original with red semi-transparent overlay
        # Work in RGB for PIL
        src = self._source_image
        if src.ndim == 2:
            rgb = cv2.cvtColor(src, cv2.COLOR_GRAY2RGB)
        elif src.ndim == 3 and src.shape[2] == 3:
            rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        elif src.ndim == 3 and src.shape[2] == 4:
            rgb = cv2.cvtColor(src, cv2.COLOR_BGRA2RGB)
        else:
            rgb = src.copy()

        # Normalise to uint8 if necessary
        if rgb.dtype != np.uint8:
            rmin, rmax = rgb.min(), rgb.max()
            if rmax - rmin > 0:
                rgb = ((rgb - rmin) / (rmax - rmin) * 255).astype(np.uint8)
            else:
                rgb = np.zeros_like(rgb, dtype=np.uint8)

        # Create red overlay using PIL for alpha compositing
        base_pil = Image.fromarray(rgb)
        overlay_arr = np.zeros_like(rgb)
        overlay_arr[mask > 0] = _OVERLAY_COLOR
        overlay_pil = Image.fromarray(overlay_arr)

        # Alpha mask: 0 where no selection, ~100 where selected
        alpha_arr = np.zeros(mask.shape, dtype=np.uint8)
        alpha_arr[mask > 0] = 100
        alpha_pil = Image.fromarray(alpha_arr, mode="L")

        composite = Image.composite(overlay_pil, base_pil, alpha_pil)

        # Fit to canvas
        self._preview_canvas.update_idletasks()
        cw = max(self._preview_canvas.winfo_width(), 50)
        ch_canvas = max(self._preview_canvas.winfo_height(), 50)

        img_w, img_h = composite.size
        scale = min(cw / img_w, ch_canvas / img_h)
        new_w = max(1, int(img_w * scale))
        new_h = max(1, int(img_h * scale))
        composite = composite.resize((new_w, new_h), Image.LANCZOS)

        self._preview_photo = ImageTk.PhotoImage(composite)
        self._preview_canvas.delete("all")
        self._preview_canvas.create_image(
            cw // 2, ch_canvas // 2,
            image=self._preview_photo,
            anchor=tk.CENTER,
        )

        # Redraw histogram with updated range
        self._draw_histogram()

    def _schedule_update(self) -> None:
        """Schedule a debounced preview update (50 ms)."""
        if self._update_after_id is not None:
            self.after_cancel(self._update_after_id)
        self._update_after_id = self.after(50, self._update_preview)

    def _full_update(self) -> None:
        """Force an immediate full redraw."""
        self._draw_histogram()
        self._update_preview()

    # ------------------------------------------------------------------ #
    #  Slider events                                                       #
    # ------------------------------------------------------------------ #

    def _on_slider_moved(self, _value: str) -> None:
        """Called when either slider is moved. Clamp and schedule update."""
        lo = self._min_var.get()
        hi = self._max_var.get()

        # Ensure min <= max
        if lo > hi:
            # Decide which one to adjust based on which is closer to changing
            self._min_var.set(hi)

        self._schedule_update()

    # ------------------------------------------------------------------ #
    #  Auto-threshold                                                      #
    # ------------------------------------------------------------------ #

    def _apply_otsu(self) -> None:
        """Compute Otsu threshold and set the min slider to that value."""
        thresh_val, _ = cv2.threshold(
            self._gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        thresh_int = int(thresh_val)
        self._min_var.set(thresh_int)
        self._max_var.set(255)
        self._schedule_update()

    # ------------------------------------------------------------------ #
    #  Accept / Cancel                                                     #
    # ------------------------------------------------------------------ #

    def _accept(self) -> None:
        """Create a Region from the current threshold and invoke callback."""
        lo = self._min_var.get()
        hi = self._max_var.get()

        # Build binary mask
        mask = cv2.inRange(self._gray, lo, hi)  # 0/255 uint8

        try:
            from core.region_ops import threshold as region_threshold
            from core.region_ops import region_to_display_image

            region = region_threshold(self._source_image, lo, hi)
            display_image = region_to_display_image(region, self._source_image)
            name = f"Threshold [{lo}, {hi}]"

            if self._on_accept:
                self._on_accept(region, display_image, name)

        except ImportError:
            # core.region_ops not available -- fall back to constructing
            # a Region manually from the binary mask.
            logger.warning(
                "core.region_ops not found; building Region from mask directly."
            )
            from core.region import Region

            num_labels, labels = cv2.connectedComponents(mask, connectivity=8)
            labels = labels.astype(np.int32)

            region = Region(
                labels=labels,
                num_regions=num_labels - 1,
                properties=[],
                source_image=self._gray,
                source_shape=self._gray.shape[:2],
            )

            # Build a simple display image: red overlay on source
            display_image = self._build_display_image(mask)
            name = f"Threshold [{lo}, {hi}]"

            if self._on_accept:
                self._on_accept(region, display_image, name)

        self.grab_release()
        self.destroy()

    def _cancel(self) -> None:
        """Close the dialog without accepting."""
        self.grab_release()
        self.destroy()

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        """Convert an image to uint8 grayscale.

        Handles colour images (ndim == 3) via ``cv2.cvtColor`` and
        non-uint8 dtypes via min-max normalisation.
        """
        if image.ndim == 3:
            if image.shape[2] == 4:
                gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            else:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        if gray.dtype != np.uint8:
            gmin, gmax = gray.min(), gray.max()
            if gmax - gmin > 0:
                gray = ((gray - gmin) / (gmax - gmin) * 255).astype(np.uint8)
            else:
                gray = np.zeros_like(gray, dtype=np.uint8)

        return gray

    def _build_display_image(self, mask: np.ndarray) -> np.ndarray:
        """Build a BGR display image with red overlay on masked pixels.

        This is the fallback when ``core.region_ops.region_to_display_image``
        is not available.
        """
        src = self._source_image
        if src.ndim == 2:
            bgr = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
        elif src.ndim == 3 and src.shape[2] == 4:
            bgr = cv2.cvtColor(src, cv2.COLOR_BGRA2BGR)
        else:
            bgr = src.copy()

        if bgr.dtype != np.uint8:
            bmin, bmax = bgr.min(), bgr.max()
            if bmax - bmin > 0:
                bgr = ((bgr - bmin) / (bmax - bmin) * 255).astype(np.uint8)
            else:
                bgr = np.zeros_like(bgr, dtype=np.uint8)

        # Red overlay: blend selected pixels toward red
        overlay = bgr.copy()
        overlay[mask > 0] = [0, 0, 255]  # BGR red
        alpha = 0.4
        display = cv2.addWeighted(bgr, 1.0 - alpha, overlay, alpha, 0)
        return display
