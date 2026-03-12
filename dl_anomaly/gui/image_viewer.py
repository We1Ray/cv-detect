"""Enhanced zoomable, pannable image viewer with HALCON HDevelop-style features.

Supports NumPy/PIL images with:
- Mouse-wheel zoom centred on cursor
- Click-and-drag pan
- Fit-to-window (double-click or keyboard shortcut)
- Zoom-to-selection (right-drag rectangle)
- Real-time pixel coordinate + value feedback
- Overlay support for defect region bounding boxes
- Grid and crosshair overlays
- Collapsible loss-curve panel at the bottom (matplotlib FigureCanvasTkAgg)
"""

from __future__ import annotations

import tkinter as tk
from enum import Enum, auto
from tkinter import ttk
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageTk


class ActiveTool(Enum):
    """Active interaction tool for the image viewer."""
    PAN = auto()
    PIXEL_INSPECT = auto()
    REGION_SELECT = auto()

# matplotlib is imported lazily inside the loss-curve section to avoid
# startup cost when the panel is collapsed.


class ImageViewer(ttk.Frame):
    """A ``ttk.Frame`` containing a ``tk.Canvas`` for interactive image display
    and an optional collapsible loss-curve panel below it.

    Usage::

        viewer = ImageViewer(parent)
        viewer.pack(fill=tk.BOTH, expand=True)
        viewer.set_image(some_numpy_array)

    Callbacks
    ---------
    coord_callback(x, y, pixel_value):
        Invoked whenever the mouse moves over the canvas. *pixel_value*
        is a tuple of channel values at that coordinate (R, G, B) or (L,).
    zoom_callback(zoom_pct):
        Invoked whenever the zoom level changes.
    """

    MIN_SCALE = 0.02
    MAX_SCALE = 40.0

    def __init__(
        self,
        master: tk.Misc,
        coord_callback: Optional[Callable] = None,
        zoom_callback: Optional[Callable] = None,
        click_callback: Optional[Callable] = None,
        region_callback: Optional[Callable] = None,
        show_loss_panel: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(master, **kwargs)
        self._coord_callback = coord_callback
        self._zoom_callback = zoom_callback
        self._click_callback = click_callback
        self._region_callback = region_callback

        # Image state
        self._source_image: Optional[Image.Image] = None
        self._source_array: Optional[np.ndarray] = None  # keep raw for pixel readout
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._image_id: Optional[int] = None
        self._scale: float = 1.0

        # Active tool mode
        self._active_tool: ActiveTool = ActiveTool.PAN

        # Overlay state
        self._overlay_rects: List[Dict] = []  # list of {bbox, color, tag}
        self._show_grid: bool = False
        self._show_crosshair: bool = False
        self._crosshair_pos: Optional[Tuple[int, int]] = None

        # Pan state
        self._pan_start_x: int = 0
        self._pan_start_y: int = 0

        # Zoom-to-selection state
        self._sel_rect_id: Optional[int] = None
        self._sel_start: Optional[Tuple[int, int]] = None

        # Region selection state (REGION_SELECT tool)
        self._region_sel_start: Optional[Tuple[int, int]] = None
        self._current_region_sel: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h) in image coords

        # Image offset (top-left corner of the displayed image in canvas coords)
        self._img_offset_x: float = 0.0
        self._img_offset_y: float = 0.0

        # Build layout
        self._build_canvas()

        # Optional loss-curve panel
        self._loss_panel_visible = False
        self._loss_frame: Optional[ttk.Frame] = None
        self._loss_canvas_mpl = None
        self._loss_fig = None
        self._loss_ax = None
        if show_loss_panel:
            self._build_loss_panel()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_canvas(self) -> None:
        self._canvas = tk.Canvas(
            self,
            bg="#1e1e1e",
            highlightthickness=0,
            cursor="crosshair",
        )
        self._canvas.pack(fill=tk.BOTH, expand=True)

        # Bindings
        self._canvas.bind("<MouseWheel>", self._on_mousewheel)
        self._canvas.bind("<Button-4>", self._on_mousewheel_linux_up)
        self._canvas.bind("<Button-5>", self._on_mousewheel_linux_down)
        self._canvas.bind("<ButtonPress-1>", self._on_button1_press)
        self._canvas.bind("<B1-Motion>", self._on_button1_motion)
        self._canvas.bind("<ButtonRelease-1>", self._on_button1_release)
        self._canvas.bind("<Double-Button-1>", self._on_double_click)
        self._canvas.bind("<ButtonPress-3>", self._on_sel_start)
        self._canvas.bind("<B3-Motion>", self._on_sel_move)
        self._canvas.bind("<ButtonRelease-3>", self._on_sel_end)
        self._canvas.bind("<Motion>", self._on_motion)
        self._canvas.bind("<Configure>", self._on_configure)

    def _build_loss_panel(self) -> None:
        """Build a collapsible matplotlib loss-curve frame below the canvas."""
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        self._loss_frame = ttk.Frame(self)

        # Toggle button
        self._loss_toggle_var = tk.StringVar(value="-- \u25BC \u640d\u5931\u66f2\u7dda --")
        self._loss_toggle_btn = ttk.Button(
            self,
            textvariable=self._loss_toggle_var,
            command=self._toggle_loss_panel,
        )
        self._loss_toggle_btn.pack(fill=tk.X, side=tk.BOTTOM)

        # Figure
        self._loss_fig = Figure(figsize=(6, 2.2), dpi=90)
        self._loss_fig.patch.set_facecolor("#2b2b2b")
        self._loss_ax = self._loss_fig.add_subplot(111)
        self._style_loss_ax()
        self._loss_fig.tight_layout(pad=1.0)

        self._loss_canvas_mpl = FigureCanvasTkAgg(self._loss_fig, master=self._loss_frame)
        self._loss_canvas_mpl.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _style_loss_ax(self) -> None:
        if self._loss_ax is None:
            return
        ax = self._loss_ax
        ax.set_facecolor("#2b2b2b")
        ax.tick_params(colors="#cccccc", labelsize=8)
        ax.xaxis.label.set_color("#cccccc")
        ax.yaxis.label.set_color("#cccccc")
        ax.title.set_color("#cccccc")
        for spine in ax.spines.values():
            spine.set_color("#555555")
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel("Loss", fontsize=8)
        ax.grid(True, alpha=0.2, color="#888888")

    def _toggle_loss_panel(self) -> None:
        if self._loss_frame is None:
            return
        if self._loss_panel_visible:
            self._loss_frame.pack_forget()
            self._loss_toggle_var.set("-- \u25BC \u640d\u5931\u66f2\u7dda --")
            self._loss_panel_visible = False
        else:
            self._loss_frame.pack(fill=tk.X, side=tk.BOTTOM, before=self._loss_toggle_btn)
            self._loss_toggle_var.set("-- \u25B2 \u640d\u5931\u66f2\u7dda --")
            self._loss_panel_visible = True

    def show_loss_panel(self) -> None:
        if not self._loss_panel_visible and self._loss_frame is not None:
            self._toggle_loss_panel()

    def hide_loss_panel(self) -> None:
        if self._loss_panel_visible:
            self._toggle_loss_panel()

    # ------------------------------------------------------------------
    # Loss plot update
    # ------------------------------------------------------------------

    def update_loss_plot(
        self,
        train_losses: list,
        val_losses: list,
        title: str = "",
    ) -> None:
        """Redraw the loss curve with new data. Must be called on the main thread."""
        if self._loss_ax is None:
            return
        ax = self._loss_ax
        ax.clear()
        self._style_loss_ax()
        epochs = list(range(1, len(train_losses) + 1))
        if train_losses:
            ax.plot(epochs, train_losses, label="Train", linewidth=1.2, color="#4fc3f7")
        if val_losses:
            ax.plot(epochs, val_losses, label="Val", linewidth=1.2, color="#ff8a65")
        if train_losses or val_losses:
            ax.legend(fontsize=7, facecolor="#2b2b2b", edgecolor="#555555", labelcolor="#cccccc")
        if title:
            ax.set_title(title, fontsize=9)
        self._loss_fig.tight_layout(pad=1.0)
        self._loss_canvas_mpl.draw_idle()

    # ------------------------------------------------------------------
    # Public API -- Image
    # ------------------------------------------------------------------

    def display_array(self, array: np.ndarray) -> None:
        """Alias for :meth:`set_image` (used by region highlight)."""
        self.set_image(array)

    def set_image(self, array: np.ndarray) -> None:
        """Display a NumPy image ``(H, W)`` or ``(H, W, 3)`` (RGB uint8)."""
        self._source_array = array.copy()
        if array.ndim == 2:
            self._source_image = Image.fromarray(array, mode="L")
        elif array.ndim == 3 and array.shape[2] == 1:
            self._source_image = Image.fromarray(array[:, :, 0], mode="L")
        else:
            self._source_image = Image.fromarray(array, mode="RGB")
        self.fit_to_window()

    def fit_to_window(self) -> None:
        """Resize image to fill canvas while preserving aspect ratio."""
        if self._source_image is None:
            return
        cw = self._canvas.winfo_width() or 400
        ch = self._canvas.winfo_height() or 400
        iw, ih = self._source_image.size
        if iw == 0 or ih == 0:
            return
        self._scale = min(cw / iw, ch / ih)
        # Centre
        disp_w = iw * self._scale
        disp_h = ih * self._scale
        self._img_offset_x = (cw - disp_w) / 2
        self._img_offset_y = (ch - disp_h) / 2
        self._refresh()

    def zoom_to_actual(self) -> None:
        """Show image at 1:1 pixel mapping."""
        if self._source_image is None:
            return
        cw = self._canvas.winfo_width() or 400
        ch = self._canvas.winfo_height() or 400
        iw, ih = self._source_image.size
        self._scale = 1.0
        self._img_offset_x = (cw - iw) / 2
        self._img_offset_y = (ch - ih) / 2
        self._refresh()

    def zoom_in(self) -> None:
        cw = self._canvas.winfo_width() or 400
        ch = self._canvas.winfo_height() or 400
        self._zoom(1.25, cw // 2, ch // 2)

    def zoom_out(self) -> None:
        cw = self._canvas.winfo_width() or 400
        ch = self._canvas.winfo_height() or 400
        self._zoom(0.8, cw // 2, ch // 2)

    def clear(self) -> None:
        """Remove displayed image and all overlays."""
        self._canvas.delete("all")
        self._source_image = None
        self._source_array = None
        self._photo = None
        self._image_id = None
        self._overlay_rects.clear()

    def get_zoom_percent(self) -> float:
        return self._scale * 100.0

    # ------------------------------------------------------------------
    # Tool mode
    # ------------------------------------------------------------------

    def set_active_tool(self, tool: ActiveTool) -> None:
        """Switch the active interaction tool."""
        self._active_tool = tool
        cursor_map = {
            ActiveTool.PAN: "fleur",
            ActiveTool.PIXEL_INSPECT: "crosshair",
            ActiveTool.REGION_SELECT: "cross",
        }
        self._canvas.configure(cursor=cursor_map.get(tool, "crosshair"))
        if tool != ActiveTool.REGION_SELECT:
            self._clear_region_selection()

    def get_active_tool(self) -> ActiveTool:
        return self._active_tool

    def get_region_selection(self) -> Optional[Tuple[int, int, int, int]]:
        """Return (x, y, w, h) in image coords, or None."""
        return self._current_region_sel

    def clear_region_selection(self) -> None:
        self._clear_region_selection()

    # ------------------------------------------------------------------
    # Overlays
    # ------------------------------------------------------------------

    def set_overlay_rects(self, rects: List[Dict]) -> None:
        """Set bounding-box overlays. Each dict: {bbox: (x,y,w,h), color: str}."""
        self._overlay_rects = rects
        self._draw_overlays()

    def set_grid(self, show: bool) -> None:
        self._show_grid = show
        self._draw_overlays()

    def set_region_overlay(self, region, alpha: float = 0.5) -> None:
        """Set Region color overlay."""
        if region is None or self._source_array is None:
            self._redraw()
            return
        try:
            from dl_anomaly.core.region_ops import region_to_display_image
            overlay = region_to_display_image(region, self._source_array)
            # Convert BGR to RGB for display
            if overlay.ndim == 3 and overlay.shape[2] == 3:
                import cv2
                overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            self.set_image(overlay)
        except Exception:
            pass

    def set_crosshair(self, show: bool) -> None:
        self._show_crosshair = show
        self._draw_overlays()

    # ------------------------------------------------------------------
    # Internal rendering
    # ------------------------------------------------------------------

    def _refresh(self) -> None:
        if self._source_image is None:
            return

        iw, ih = self._source_image.size
        new_w = max(1, int(iw * self._scale))
        new_h = max(1, int(ih * self._scale))

        # Use NEAREST for large zoom to see pixel grid, LANCZOS otherwise
        if self._scale > 4.0:
            resample = Image.NEAREST
        else:
            resample = Image.LANCZOS

        resized = self._source_image.resize((new_w, new_h), resample)
        self._photo = ImageTk.PhotoImage(resized)

        self._canvas.delete("all")
        self._image_id = self._canvas.create_image(
            int(self._img_offset_x),
            int(self._img_offset_y),
            image=self._photo,
            anchor=tk.NW,
        )

        self._draw_overlays()

        if self._zoom_callback:
            self._zoom_callback(self.get_zoom_percent())

    def _draw_overlays(self) -> None:
        self._canvas.delete("overlay")

        if self._source_image is None:
            return

        # Defect region rectangles
        for rect_info in self._overlay_rects:
            x, y, w, h = rect_info["bbox"]
            color = rect_info.get("color", "#ff0000")
            # Convert image coords to canvas coords
            cx1 = self._img_offset_x + x * self._scale
            cy1 = self._img_offset_y + y * self._scale
            cx2 = cx1 + w * self._scale
            cy2 = cy1 + h * self._scale
            self._canvas.create_rectangle(
                cx1, cy1, cx2, cy2,
                outline=color, width=2, tags="overlay",
            )

        # Grid
        if self._show_grid and self._scale > 3.0:
            iw, ih = self._source_image.size
            disp_w = iw * self._scale
            disp_h = ih * self._scale
            ox = self._img_offset_x
            oy = self._img_offset_y
            step = self._scale  # one pixel
            for px in range(iw + 1):
                gx = ox + px * step
                self._canvas.create_line(
                    gx, oy, gx, oy + disp_h,
                    fill="#555555", tags="overlay",
                )
            for py in range(ih + 1):
                gy = oy + py * step
                self._canvas.create_line(
                    ox, gy, ox + disp_w, gy,
                    fill="#555555", tags="overlay",
                )

        # Crosshair
        if self._show_crosshair and self._crosshair_pos is not None:
            mx, my = self._crosshair_pos
            cw = self._canvas.winfo_width()
            ch = self._canvas.winfo_height()
            self._canvas.create_line(mx, 0, mx, ch, fill="#00ff00", dash=(4, 4), tags="overlay")
            self._canvas.create_line(0, my, cw, my, fill="#00ff00", dash=(4, 4), tags="overlay")

        # Redraw active region selection (survives canvas refresh)
        if self._current_region_sel is not None:
            ix, iy, w, h = self._current_region_sel
            self._draw_final_region_rect(ix, iy, w, h)

    # ------------------------------------------------------------------
    # Coordinate conversion
    # ------------------------------------------------------------------

    def _canvas_to_image(self, cx: int, cy: int) -> Optional[Tuple[int, int]]:
        """Convert canvas coords to original image pixel coords."""
        if self._source_image is None or self._scale == 0:
            return None
        ix = (cx - self._img_offset_x) / self._scale
        iy = (cy - self._img_offset_y) / self._scale
        iw, ih = self._source_image.size
        ix_int = int(ix)
        iy_int = int(iy)
        if 0 <= ix_int < iw and 0 <= iy_int < ih:
            return (ix_int, iy_int)
        return None

    def _get_pixel_value(self, ix: int, iy: int) -> Tuple:
        """Return the pixel value at image coords (ix, iy)."""
        if self._source_array is None:
            return ()
        arr = self._source_array
        if arr.ndim == 2:
            return (int(arr[iy, ix]),)
        elif arr.ndim == 3:
            return tuple(int(v) for v in arr[iy, ix])
        return ()

    def _image_to_canvas(self, ix: int, iy: int) -> Tuple[float, float]:
        """Convert image pixel coords to canvas coords."""
        cx = self._img_offset_x + ix * self._scale
        cy = self._img_offset_y + iy * self._scale
        return (cx, cy)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_mousewheel(self, event: tk.Event) -> None:
        factor = 1.15 if event.delta > 0 else 1.0 / 1.15
        self._zoom(factor, event.x, event.y)

    def _on_mousewheel_linux_up(self, event: tk.Event) -> None:
        self._zoom(1.15, event.x, event.y)

    def _on_mousewheel_linux_down(self, event: tk.Event) -> None:
        self._zoom(1.0 / 1.15, event.x, event.y)

    def _zoom(self, factor: float, cx: int, cy: int) -> None:
        if self._source_image is None:
            return
        new_scale = self._scale * factor
        new_scale = max(self.MIN_SCALE, min(new_scale, self.MAX_SCALE))

        img_x = (cx - self._img_offset_x) / self._scale
        img_y = (cy - self._img_offset_y) / self._scale
        self._scale = new_scale
        self._img_offset_x = cx - img_x * new_scale
        self._img_offset_y = cy - img_y * new_scale
        self._refresh()

    # ------------------------------------------------------------------
    # Button-1 dispatchers (tool-aware)
    # ------------------------------------------------------------------

    def _on_button1_press(self, event: tk.Event) -> None:
        if self._active_tool == ActiveTool.PAN:
            self._on_pan_start(event)
        elif self._active_tool == ActiveTool.PIXEL_INSPECT:
            self._on_pixel_inspect_click(event)
        elif self._active_tool == ActiveTool.REGION_SELECT:
            self._on_region_sel_start(event)

    def _on_button1_motion(self, event: tk.Event) -> None:
        if self._active_tool == ActiveTool.PAN:
            self._on_pan_move(event)
        elif self._active_tool == ActiveTool.REGION_SELECT:
            self._on_region_sel_move(event)

    def _on_button1_release(self, event: tk.Event) -> None:
        if self._active_tool == ActiveTool.PAN:
            self._on_pan_end(event)
        elif self._active_tool == ActiveTool.REGION_SELECT:
            self._on_region_sel_end(event)

    # ------------------------------------------------------------------
    # Pan handlers
    # ------------------------------------------------------------------

    def _on_pan_start(self, event: tk.Event) -> None:
        self._pan_start_x = event.x
        self._pan_start_y = event.y

    def _on_pan_move(self, event: tk.Event) -> None:
        if self._source_image is None:
            return
        dx = event.x - self._pan_start_x
        dy = event.y - self._pan_start_y
        self._img_offset_x += dx
        self._img_offset_y += dy
        self._pan_start_x = event.x
        self._pan_start_y = event.y
        self._refresh()

    def _on_pan_end(self, event: tk.Event) -> None:
        pass

    def _on_double_click(self, event: tk.Event) -> None:
        self.fit_to_window()

    # ------------------------------------------------------------------
    # Pixel inspect click handler
    # ------------------------------------------------------------------

    def _on_pixel_inspect_click(self, event: tk.Event) -> None:
        """Handle single click in PIXEL_INSPECT tool mode."""
        coords = self._canvas_to_image(event.x, event.y)
        if coords is None:
            return
        ix, iy = coords
        pv = self._get_pixel_value(ix, iy)
        if self._click_callback:
            self._click_callback(ix, iy, pv)

    # ------------------------------------------------------------------
    # Region selection handlers
    # ------------------------------------------------------------------

    def _on_region_sel_start(self, event: tk.Event) -> None:
        self._region_sel_start = (event.x, event.y)
        self._canvas.delete("region_sel")

    def _on_region_sel_move(self, event: tk.Event) -> None:
        if self._region_sel_start is None:
            return
        self._canvas.delete("region_sel")
        sx, sy = self._region_sel_start
        self._canvas.create_rectangle(
            sx, sy, event.x, event.y,
            outline="#ffcc00", width=2, dash=(4, 4), tags="region_sel",
        )
        # Live info label
        self._update_region_info_label(sx, sy, event.x, event.y)

    def _on_region_sel_end(self, event: tk.Event) -> None:
        if self._region_sel_start is None:
            return
        sx, sy = self._region_sel_start
        ex, ey = event.x, event.y
        self._region_sel_start = None

        if abs(ex - sx) < 5 or abs(ey - sy) < 5:
            self._canvas.delete("region_sel")
            return

        # Convert to image coords, clamping to image bounds
        tl, br = self._clamp_region_to_image(
            min(sx, ex), min(sy, ey), max(sx, ex), max(sy, ey)
        )
        if tl is None or br is None:
            self._canvas.delete("region_sel")
            return

        ix1, iy1 = tl
        ix2, iy2 = br
        w = ix2 - ix1
        h = iy2 - iy1
        if w <= 0 or h <= 0:
            self._canvas.delete("region_sel")
            return

        self._current_region_sel = (ix1, iy1, w, h)
        self._draw_final_region_rect(ix1, iy1, w, h)

        if self._region_callback:
            self._region_callback(ix1, iy1, w, h)

    def _update_region_info_label(self, sx: int, sy: int, ex: int, ey: int) -> None:
        """Show live info label near the region rectangle during drag."""
        tl = self._canvas_to_image(min(sx, ex), min(sy, ey))
        br = self._canvas_to_image(max(sx, ex), max(sy, ey))
        if tl and br:
            ix1, iy1 = tl
            ix2, iy2 = br
            w = max(0, ix2 - ix1)
            h = max(0, iy2 - iy1)
            text = f"({ix1},{iy1}) {w}\u00d7{h} = {w * h}px"
        else:
            text = ""
        label_x = max(sx, ex) + 4
        label_y = max(sy, ey) + 4
        self._canvas.create_text(
            label_x, label_y,
            text=text, fill="#ffcc00", anchor=tk.NW,
            font=("Consolas", 9), tags="region_sel",
        )

    def _draw_final_region_rect(self, ix: int, iy: int, w: int, h: int) -> None:
        """Draw finalized region rectangle snapped to image pixel grid."""
        self._canvas.delete("region_sel")
        cx1, cy1 = self._image_to_canvas(ix, iy)
        cx2, cy2 = self._image_to_canvas(ix + w, iy + h)
        self._canvas.create_rectangle(
            cx1, cy1, cx2, cy2,
            outline="#ffcc00", width=2, tags="region_sel",
        )
        text = f"({ix},{iy})  {w} \u00d7 {h}  Area={w * h}"
        self._canvas.create_text(
            cx2 + 4, cy2 + 4,
            text=text, fill="#ffcc00", anchor=tk.NW,
            font=("Consolas", 9, "bold"), tags="region_sel",
        )

    def _clear_region_selection(self) -> None:
        self._canvas.delete("region_sel")
        self._current_region_sel = None
        self._region_sel_start = None

    def _clamp_region_to_image(
        self, x1: int, y1: int, x2: int, y2: int
    ) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        """Convert canvas coords to image coords, clamping to image bounds."""
        if self._source_image is None:
            return (None, None)
        iw, ih = self._source_image.size

        def clamp(cx: int, cy: int) -> Tuple[int, int]:
            raw = self._canvas_to_image(cx, cy)
            if raw is not None:
                return raw
            ix = int((cx - self._img_offset_x) / self._scale)
            iy = int((cy - self._img_offset_y) / self._scale)
            ix = max(0, min(ix, iw - 1))
            iy = max(0, min(iy, ih - 1))
            return (ix, iy)

        return (clamp(x1, y1), clamp(x2, y2))

    # ------------------------------------------------------------------
    # Right-drag for zoom-to-selection
    # ------------------------------------------------------------------

    def _on_sel_start(self, event: tk.Event) -> None:
        self._sel_start = (event.x, event.y)
        self._sel_rect_id = None

    def _on_sel_move(self, event: tk.Event) -> None:
        if self._sel_start is None:
            return
        self._canvas.delete("sel_rect")
        sx, sy = self._sel_start
        self._sel_rect_id = self._canvas.create_rectangle(
            sx, sy, event.x, event.y,
            outline="#00ff00", width=1, dash=(3, 3), tags="sel_rect",
        )

    def _on_sel_end(self, event: tk.Event) -> None:
        self._canvas.delete("sel_rect")
        if self._sel_start is None or self._source_image is None:
            return

        sx, sy = self._sel_start
        ex, ey = event.x, event.y
        self._sel_start = None

        # Minimum drag size
        if abs(ex - sx) < 10 or abs(ey - sy) < 10:
            return

        # Selection rectangle in canvas coords
        x1, x2 = min(sx, ex), max(sx, ex)
        y1, y2 = min(sy, ey), max(sy, ey)
        sel_w = x2 - x1
        sel_h = y2 - y1

        # Centre of selection in image coords
        centre_img_x = (((x1 + x2) / 2) - self._img_offset_x) / self._scale
        centre_img_y = (((y1 + y2) / 2) - self._img_offset_y) / self._scale

        # Desired scale: fit selection to canvas
        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        img_sel_w = sel_w / self._scale
        img_sel_h = sel_h / self._scale
        new_scale = min(cw / img_sel_w, ch / img_sel_h)
        new_scale = max(self.MIN_SCALE, min(new_scale, self.MAX_SCALE))

        self._scale = new_scale
        self._img_offset_x = cw / 2 - centre_img_x * new_scale
        self._img_offset_y = ch / 2 - centre_img_y * new_scale
        self._refresh()

    # ------------------------------------------------------------------
    # Motion and configure
    # ------------------------------------------------------------------

    def _on_motion(self, event: tk.Event) -> None:
        if self._show_crosshair:
            self._crosshair_pos = (event.x, event.y)
            self._draw_overlays()

        if self._coord_callback is None or self._source_image is None:
            return
        coords = self._canvas_to_image(event.x, event.y)
        if coords is not None:
            ix, iy = coords
            pv = self._get_pixel_value(ix, iy)
            self._coord_callback(ix, iy, pv)

    def _on_configure(self, _event: tk.Event) -> None:
        if self._source_image is not None:
            self.fit_to_window()
