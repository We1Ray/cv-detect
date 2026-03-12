"""Right panel (top): image properties display.

Shows metadata about the currently viewed image: name, path, dimensions,
data type, channel count, min/max/mean pixel values, and a mini histogram.

When a Region step is selected, additionally shows a scrollable table of
per-region properties (area, centroid, circularity, etc.).
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageTk


class PropertiesPanel(ttk.LabelFrame):
    """Displays properties of the currently selected pipeline step image."""

    def __init__(self, master: tk.Misc, **kwargs) -> None:
        kwargs.setdefault("text", "\u5716\u7247\u5c6c\u6027")  # "Image Properties"
        super().__init__(master, **kwargs)
        self._histogram_photo: Optional[ImageTk.PhotoImage] = None
        self._on_region_highlight: Optional[Callable[[int], None]] = None
        self._on_region_remove: Optional[Callable[[int], None]] = None
        self._selected_region_index: Optional[int] = None
        self._build_ui()

    def _build_ui(self) -> None:
        # Grid of label pairs
        self._vars = {}
        fields = [
            ("name", "\u540d\u7a31:"),       # "Name:"
            ("size", "\u5c3a\u5bf8:"),       # "Size:"
            ("type", "\u578b\u5225:"),       # "Type:"
            ("channels", "\u901a\u9053:"),   # "Channels:"
            ("min_val", "\u6700\u5c0f\u503c:"),  # "Min:"
            ("max_val", "\u6700\u5927\u503c:"),  # "Max:"
            ("mean_val", "\u5e73\u5747\u503c:"),  # "Mean:"
            ("region", "\u5340\u57df:"),     # "Region:"
        ]

        for i, (key, label_text) in enumerate(fields):
            ttk.Label(self, text=label_text, font=("Segoe UI", 8)).grid(
                row=i, column=0, sticky=tk.W, padx=(6, 2), pady=1,
            )
            var = tk.StringVar(value="--")
            self._vars[key] = var
            ttk.Label(self, textvariable=var, font=("Segoe UI", 8)).grid(
                row=i, column=1, sticky=tk.W, padx=2, pady=1,
            )

        # Mini histogram
        row = len(fields)
        ttk.Label(self, text="\u76f4\u65b9\u5716:", font=("Segoe UI", 8)).grid(
            row=row, column=0, sticky=tk.NW, padx=(6, 2), pady=(4, 1),
        )
        self._hist_label = ttk.Label(self)
        self._hist_label.grid(row=row, column=1, sticky=tk.W, padx=2, pady=(4, 1))

        # Region feature table (initially hidden)
        row += 1
        self._region_table_label = ttk.Label(
            self, text="\u5340\u57df\u8a73\u60c5:", font=("Segoe UI", 8, "bold"),
        )
        self._region_table_label.grid(
            row=row, column=0, columnspan=2, padx=6, pady=(6, 1), sticky=tk.W,
        )

        row += 1
        table_frame = ttk.Frame(self)
        table_frame.grid(
            row=row, column=0, columnspan=2, padx=6, pady=2, sticky=tk.NSEW,
        )

        columns = ("#", "Area", "Cx", "Cy", "Circ", "Rect", "AR")
        self._region_tree = ttk.Treeview(
            table_frame, columns=columns, show="headings", height=6,
        )

        col_configs = {
            "#": ("\u5e8f\u865f", 35),
            "Area": ("\u9762\u7a4d", 55),
            "Cx": ("\u4e2d\u5fc3X", 50),
            "Cy": ("\u4e2d\u5fc3Y", 50),
            "Circ": ("\u5713\u5ea6", 50),
            "Rect": ("\u77e9\u5f62\u5ea6", 50),
            "AR": ("\u9577\u5bec\u6bd4", 50),
        }
        for col_id, (heading, width) in col_configs.items():
            self._region_tree.heading(col_id, text=heading)
            self._region_tree.column(col_id, width=width, anchor=tk.CENTER, minwidth=30)

        tree_scroll = ttk.Scrollbar(
            table_frame, orient=tk.VERTICAL, command=self._region_tree.yview,
        )
        self._region_tree.configure(yscrollcommand=tree_scroll.set)
        self._region_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self._region_tree.bind("<<TreeviewSelect>>", self._on_region_row_selected)

        # Button bar below region table
        row += 1
        btn_frame = ttk.Frame(self)
        btn_frame.grid(
            row=row, column=0, columnspan=2, padx=6, pady=(2, 4), sticky=tk.W,
        )
        self._remove_btn = ttk.Button(
            btn_frame, text="\u79fb\u9664\u9078\u53d6", command=self._on_remove_clicked,
            state=tk.DISABLED,
        )
        self._remove_btn.pack(side=tk.LEFT)
        self._region_btn_frame = btn_frame
        btn_frame.grid_remove()

        # Initially hide the region table
        self._region_table_label.grid_remove()
        table_frame.grid_remove()
        self._region_table_frame = table_frame
        self._region_table_row = row - 1

        self.columnconfigure(1, weight=1)
        self.rowconfigure(row - 1, weight=1)  # table_frame row gets expansion

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_region_highlight_callback(
        self, callback: Optional[Callable[[int], None]]
    ) -> None:
        """Register callback invoked when user clicks a region row.

        ``callback(region_index)`` receives the 1-based region index.
        """
        self._on_region_highlight = callback

    def set_region_remove_callback(
        self, callback: Optional[Callable[[int], None]]
    ) -> None:
        """Register callback invoked when user clicks Remove.

        ``callback(region_index)`` receives the 1-based region index.
        """
        self._on_region_remove = callback

    def update_properties(
        self,
        name: str,
        array: np.ndarray,
        path: str = "",
        region=None,
    ) -> None:
        """Update displayed properties from a numpy array."""
        h, w = array.shape[:2]
        channels = array.shape[2] if array.ndim == 3 else 1
        dtype_str = str(array.dtype)

        self._vars["name"].set(name[:30])
        self._vars["size"].set(f"{w} x {h}")
        self._vars["type"].set(dtype_str)
        self._vars["channels"].set(str(channels))
        self._vars["min_val"].set(f"{array.min():.2f}" if array.dtype.kind == "f" else str(array.min()))
        self._vars["max_val"].set(f"{array.max():.2f}" if array.dtype.kind == "f" else str(array.max()))
        self._vars["mean_val"].set(f"{array.mean():.4f}" if array.dtype.kind == "f" else f"{array.mean():.1f}")

        # Region info
        if region is not None and hasattr(region, "num_regions"):
            total_area = sum(p.area for p in region.properties) if region.properties else 0
            self._vars["region"].set(f"{region.num_regions} \u500b (\u9762\u7a4d={total_area})")
        else:
            self._vars["region"].set("--")

        self._draw_histogram(array)

        # Region feature table
        if (
            region is not None
            and hasattr(region, "properties")
            and region.properties
        ):
            self._region_table_label.grid()
            self._region_table_frame.grid()
            self._region_btn_frame.grid()
            self._populate_region_table(region)
        else:
            self._hide_region_table()

    def update_measurement(
        self, x: int, y: int, w: int, h: int, roi_stats: Optional[dict] = None,
    ) -> None:
        """Update the region field with measurement info from a selection."""
        area = w * h
        info = f"({x},{y}) {w}\u00d7{h} Area={area}"
        if roi_stats:
            mean_v = roi_stats.get("mean")
            if mean_v is not None:
                info += f" Mean={mean_v:.1f}"
        self._vars["region"].set(info)

    def clear(self) -> None:
        for var in self._vars.values():
            var.set("--")
        self._hist_label.configure(image="")
        self._histogram_photo = None
        self._hide_region_table()

    # ------------------------------------------------------------------
    # Region table
    # ------------------------------------------------------------------

    def _populate_region_table(self, region) -> None:
        """Fill the region feature table from region.properties."""
        for item in self._region_tree.get_children():
            self._region_tree.delete(item)

        for p in region.properties:
            cx, cy = p.centroid
            self._region_tree.insert(
                "", tk.END,
                iid=str(p.index),
                values=(
                    p.index,
                    p.area,
                    f"{cx:.1f}",
                    f"{cy:.1f}",
                    f"{p.circularity:.3f}",
                    f"{p.rectangularity:.3f}",
                    f"{p.aspect_ratio:.2f}",
                ),
            )

    def _hide_region_table(self) -> None:
        self._region_table_label.grid_remove()
        self._region_table_frame.grid_remove()
        self._region_btn_frame.grid_remove()
        self._selected_region_index = None
        self._remove_btn.configure(state=tk.DISABLED)
        for item in self._region_tree.get_children():
            self._region_tree.delete(item)

    def _on_region_row_selected(self, _event: tk.Event) -> None:
        selection = self._region_tree.selection()
        if not selection:
            self._selected_region_index = None
            self._remove_btn.configure(state=tk.DISABLED)
            return
        try:
            region_index = int(selection[0])
        except (ValueError, IndexError):
            self._selected_region_index = None
            self._remove_btn.configure(state=tk.DISABLED)
            return
        self._selected_region_index = region_index
        self._remove_btn.configure(state=tk.NORMAL)
        if self._on_region_highlight:
            self._on_region_highlight(region_index)

    def _on_remove_clicked(self) -> None:
        """Handle 'Remove Selected' button click."""
        if self._selected_region_index is None:
            return
        idx = self._selected_region_index
        if self._on_region_remove:
            self._on_region_remove(idx)
        self._selected_region_index = None
        self._remove_btn.configure(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # Histogram
    # ------------------------------------------------------------------

    def _draw_histogram(self, array: np.ndarray) -> None:
        """Draw a compact histogram using PIL."""
        hist_w, hist_h = 160, 50

        if array.dtype.kind == "f":
            vmin, vmax = float(array.min()), float(array.max())
            if vmax - vmin < 1e-8:
                data = np.zeros(array.shape[:2], dtype=np.uint8)
            else:
                data = ((array.astype(np.float64) - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        else:
            data = array.astype(np.uint8)

        if data.ndim == 3:
            data = np.mean(data, axis=2).astype(np.uint8)

        hist, _ = np.histogram(data.ravel(), bins=64, range=(0, 255))
        if hist.max() == 0:
            hist_norm = np.zeros_like(hist, dtype=float)
        else:
            hist_norm = hist.astype(float) / hist.max()

        img = Image.new("RGB", (hist_w, hist_h), (30, 30, 30))
        draw = ImageDraw.Draw(img)

        bar_w = hist_w / len(hist_norm)
        for i, v in enumerate(hist_norm):
            x0 = int(i * bar_w)
            x1 = int((i + 1) * bar_w)
            bar_h = int(v * (hist_h - 4))
            if bar_h > 0:
                draw.rectangle(
                    [x0, hist_h - bar_h - 2, x1 - 1, hist_h - 2],
                    fill=(79, 195, 247),
                )

        self._histogram_photo = ImageTk.PhotoImage(img)
        self._hist_label.configure(image=self._histogram_photo)
