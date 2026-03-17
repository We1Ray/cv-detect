"""Left panel: processing step list with thumbnails.

Each inspection or operation produces a *pipeline step* (a named image).
This panel shows them as a vertical list with small thumbnails. Clicking
a step displays that image in the main viewer.

Steps are visually distinguished by type:
- **Image** steps (blue ``I`` tag) — standard image processing results.
- **Region** steps (orange ``R`` tag) — threshold / blob analysis results
  that carry a Region object.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageTk

from dl_anomaly.gui.platform_keys import display


class PipelineStep:
    """Data class for a single pipeline step."""

    __slots__ = ("name", "array", "thumbnail", "photo", "region", "op_meta")

    def __init__(self, name: str, array: np.ndarray) -> None:
        self.name = name
        self.array = array  # original full-res numpy array (H,W) or (H,W,3) uint8
        self.thumbnail: Optional[Image.Image] = None
        self.photo: Optional[ImageTk.PhotoImage] = None
        self.region = None
        self.op_meta: Optional[dict] = None

    @property
    def is_region(self) -> bool:
        """Return ``True`` if this step carries a Region."""
        return self.region is not None

    def make_thumbnail(self, size: int = 64) -> ImageTk.PhotoImage:
        if self.array.ndim == 2:
            img = Image.fromarray(self.array, mode="L").convert("RGB")
        elif self.array.ndim == 3 and self.array.shape[2] == 1:
            img = Image.fromarray(self.array[:, :, 0], mode="L").convert("RGB")
        else:
            img = Image.fromarray(self.array, mode="RGB")
        img.thumbnail((size, size), Image.LANCZOS)
        # Pad to exact size with dark background
        bg = Image.new("RGB", (size, size), (30, 30, 30))
        ox = (size - img.width) // 2
        oy = (size - img.height) // 2
        bg.paste(img, (ox, oy))
        self.thumbnail = bg
        self.photo = ImageTk.PhotoImage(bg)
        return self.photo


class PipelinePanel(ttk.Frame):
    """Left panel showing the processing pipeline as a step list with thumbnails.

    Parameters
    ----------
    on_step_selected : callable(index: int)
        Called when the user clicks a step.
    on_step_delete : callable(index: int)
        Called when the user requests deletion (right-click menu).
    on_step_export : callable(index: int)
        Called when the user exports a step image.
    """

    THUMB_SIZE = 64

    # Colours for the type tags
    _TAG_IMG_BG = "#0078d4"   # blue for Image
    _TAG_RGN_BG = "#d45500"   # orange for Region
    _TAG_FG = "#ffffff"

    # Row colours
    _BG = "#2b2b2b"
    _BG_SEL = "#3a3a5c"

    def __init__(
        self,
        master: tk.Misc,
        on_step_selected: Optional[Callable[[int], None]] = None,
        on_step_delete: Optional[Callable[[int], None]] = None,
        on_step_export: Optional[Callable[[int], None]] = None,
        **kwargs,
    ) -> None:
        super().__init__(master, **kwargs)
        self._on_step_selected = on_step_selected
        self._on_step_delete = on_step_delete
        self._on_step_export = on_step_export

        self._steps: List[PipelineStep] = []
        self._selected_index: int = -1
        self._filter_mode: str = "all"  # "all" | "image" | "region"

        # Drag-and-drop state
        self._drag_source_index: Optional[int] = None
        self._drag_indicator: Optional[tk.Frame] = None

        # Empty state hint
        self._empty_hint: Optional[tk.Label] = None

        self._build_ui()
        self._show_empty_hint()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # Title
        title_frame = ttk.Frame(self)
        title_frame.pack(fill=tk.X, padx=2, pady=(4, 2))
        ttk.Label(
            title_frame,
            text="\u8655\u7406\u6d41\u7a0b",  # "Processing Pipeline"
            font=("Segoe UI", 10, "bold"),
        ).pack(side=tk.LEFT, padx=4)

        self._clear_btn = ttk.Button(
            title_frame,
            text="\u6e05\u9664",  # "Clear"
            width=5,
            command=self._on_clear,
        )
        self._clear_btn.pack(side=tk.RIGHT, padx=2)

        # Filter bar
        filter_frame = tk.Frame(self, bg=self._BG)
        filter_frame.pack(fill=tk.X, padx=2, pady=(0, 2))

        self._filter_btns: Dict[str, tk.Button] = {}
        for mode, label in [("all", "全部"), ("image", "影像"), ("region", "區域")]:
            btn = tk.Button(
                filter_frame,
                text=label,
                width=5,
                font=("Segoe UI", 8),
                bg="#3c3c3c",
                fg="#e0e0e0",
                activebackground="#4a4a6c",
                activeforeground="#ffffff",
                relief=tk.FLAT,
                command=lambda m=mode: self._set_filter(m),
            )
            btn.pack(side=tk.LEFT, padx=1)
            self._filter_btns[mode] = btn
        self._update_filter_buttons()

        # Scrollable step list
        list_container = ttk.Frame(self)
        list_container.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        self._list_canvas = tk.Canvas(
            list_container,
            bg="#1e1e1e",
            highlightthickness=0,
            width=200,
        )
        self._scrollbar = ttk.Scrollbar(
            list_container, orient=tk.VERTICAL, command=self._list_canvas.yview
        )
        self._list_canvas.configure(yscrollcommand=self._scrollbar.set)

        self._scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._list_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._inner_frame = ttk.Frame(self._list_canvas)
        self._inner_window = self._list_canvas.create_window(
            (0, 0), window=self._inner_frame, anchor=tk.NW
        )

        self._inner_frame.bind("<Configure>", self._on_inner_configure)
        self._list_canvas.bind("<Configure>", self._on_canvas_configure)
        self._list_canvas.bind("<MouseWheel>", self._on_mousewheel)

        # Thumbnail grid at the bottom
        thumb_label = ttk.Label(self, text="\u7e2e\u5716\u9810\u89bd", font=("Segoe UI", 9))
        thumb_label.pack(fill=tk.X, padx=6, pady=(4, 0))
        self._thumb_frame = ttk.Frame(self)
        self._thumb_frame.pack(fill=tk.X, padx=2, pady=2)

    def _on_inner_configure(self, _event: tk.Event) -> None:
        self._list_canvas.configure(scrollregion=self._list_canvas.bbox("all"))

    def _on_canvas_configure(self, event: tk.Event) -> None:
        self._list_canvas.itemconfig(self._inner_window, width=event.width)

    def _on_mousewheel(self, event: tk.Event) -> None:
        self._list_canvas.yview_scroll(-1 * (event.delta // 120), "units")

    # ------------------------------------------------------------------
    # Filter
    # ------------------------------------------------------------------

    def _set_filter(self, mode: str) -> None:
        self._filter_mode = mode
        self._update_filter_buttons()
        self._rebuild_step_list()
        self._update_selection_visual()

    def _update_filter_buttons(self) -> None:
        for mode, btn in self._filter_btns.items():
            if mode == self._filter_mode:
                btn.configure(bg="#3a3a5c", relief=tk.SUNKEN)
            else:
                btn.configure(bg="#3c3c3c", relief=tk.FLAT)

    def _is_step_visible(self, step: PipelineStep) -> bool:
        if self._filter_mode == "all":
            return True
        if self._filter_mode == "region":
            return step.is_region
        return not step.is_region  # "image" mode

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_step(self, name: str, array: np.ndarray, select: bool = True, region=None, op_meta=None) -> int:
        """Add a new pipeline step and return its index."""
        self._hide_empty_hint()
        step = PipelineStep(name, array)
        step.region = region
        step.op_meta = op_meta
        step.make_thumbnail(self.THUMB_SIZE)
        self._steps.append(step)
        idx = len(self._steps) - 1

        if self._is_step_visible(step):
            self._create_step_widget(idx, step)
        self._update_thumbnails()

        if select:
            self.select_step(idx)

        return idx

    def _on_clear(self) -> None:
        """Ask for confirmation before clearing all steps."""
        if not self._steps:
            return
        from tkinter import messagebox
        if messagebox.askyesno(
            "\u78ba\u8a8d\u6e05\u9664",
            f"\u78ba\u5b9a\u8981\u6e05\u9664\u6240\u6709 {len(self._steps)} \u500b\u8655\u7406\u6b65\u9a5f\uff1f\n\u6b64\u64cd\u4f5c\u7121\u6cd5\u5fa9\u539f\u3002",
            icon="warning",
            parent=self,
        ):
            self.clear_all()

    def clear_all(self) -> None:
        """Remove all steps."""
        self._steps.clear()
        self._selected_index = -1
        for child in self._inner_frame.winfo_children():
            child.destroy()
        self._update_thumbnails()
        self._show_empty_hint()

    def _show_empty_hint(self) -> None:
        """Show hint when pipeline is empty."""
        if hasattr(self, '_empty_hint') and self._empty_hint is not None:
            return
        self._empty_hint = tk.Label(
            self._inner_frame,
            text=f"\u958b\u555f\u5716\u7247\u4ee5\u958b\u59cb\n{display('O')}",
            bg="#2b2b2b",
            fg="#555555",
            font=("Segoe UI", 10),
            justify=tk.CENTER,
            pady=40,
        )
        self._empty_hint.pack(fill=tk.X)

    def _hide_empty_hint(self) -> None:
        """Remove the empty-state hint label."""
        if hasattr(self, '_empty_hint') and self._empty_hint is not None:
            self._empty_hint.destroy()
            self._empty_hint = None

    def get_step(self, index: int) -> Optional[PipelineStep]:
        if 0 <= index < len(self._steps):
            return self._steps[index]
        return None

    def get_current_step(self) -> Optional[PipelineStep]:
        return self.get_step(self._selected_index)

    def get_current_index(self) -> int:
        return self._selected_index

    def get_step_count(self) -> int:
        return len(self._steps)

    def select_step(self, index: int) -> None:
        if index < 0 or index >= len(self._steps):
            return
        self._selected_index = index
        self._update_selection_visual()
        if self._on_step_selected:
            self._on_step_selected(index)

    def delete_step(self, index: int) -> None:
        if index < 0 or index >= len(self._steps):
            return
        del self._steps[index]
        self._rebuild_step_list()
        self._update_thumbnails()
        if len(self._steps) == 0:
            self._selected_index = -1
            self._show_empty_hint()
            return
        if self._selected_index >= len(self._steps):
            self._selected_index = len(self._steps) - 1
        if self._selected_index >= 0:
            self.select_step(self._selected_index)

    def get_all_steps(self) -> List[PipelineStep]:
        return list(self._steps)

    # ------------------------------------------------------------------
    # Internal: step widget creation
    # ------------------------------------------------------------------

    def _create_step_widget(self, index: int, step: PipelineStep) -> None:
        frame = tk.Frame(
            self._inner_frame,
            bg=self._BG,
            cursor="hand2",
            padx=4,
            pady=3,
        )
        frame.pack(fill=tk.X, padx=2, pady=1)
        frame._step_index = index  # type: ignore[attr-defined]

        # Indicator arrow
        indicator = tk.Label(
            frame,
            text="",
            fg="#4fc3f7",
            bg=self._BG,
            font=("Segoe UI", 10),
            width=2,
        )
        indicator.pack(side=tk.LEFT)
        frame._indicator = indicator  # type: ignore[attr-defined]

        # Type tag
        is_region = step.is_region
        tag_text = "R" if is_region else "I"
        tag_bg = self._TAG_RGN_BG if is_region else self._TAG_IMG_BG
        type_tag = tk.Label(
            frame,
            text=tag_text,
            bg=tag_bg,
            fg=self._TAG_FG,
            font=("Segoe UI", 7, "bold"),
            width=2,
            anchor=tk.CENTER,
            padx=1,
        )
        type_tag.pack(side=tk.LEFT, padx=(0, 3))
        frame._type_tag = type_tag  # type: ignore[attr-defined]

        # Step number + name
        text = f"{index + 1}. {step.name}"
        label = tk.Label(
            frame,
            text=text,
            fg="#e0e0e0",
            bg=self._BG,
            font=("Segoe UI", 9),
            anchor=tk.W,
        )
        label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Bind click and drag on the whole row
        for widget in (frame, indicator, type_tag, label):
            widget.bind("<Button-1>", lambda e, i=index: self._on_row_click(e, i))
            widget.bind("<B1-Motion>", lambda e, i=index: self._on_drag_motion(e, i))
            widget.bind("<ButtonRelease-1>", lambda e, i=index: self._on_drag_release(e, i))
            widget.bind("<Button-3>", lambda e, i=index: self._show_context_menu(e, i))

    def _rebuild_step_list(self) -> None:
        for child in self._inner_frame.winfo_children():
            child.destroy()
        for idx, step in enumerate(self._steps):
            if self._is_step_visible(step):
                self._create_step_widget(idx, step)

    def _update_selection_visual(self) -> None:
        for child in self._inner_frame.winfo_children():
            idx = getattr(child, "_step_index", -1)
            indicator = getattr(child, "_indicator", None)
            type_tag = getattr(child, "_type_tag", None)
            is_selected = idx == self._selected_index
            bg = self._BG_SEL if is_selected else self._BG
            child.configure(bg=bg)
            if indicator:
                indicator.configure(text="\u25B6" if is_selected else "", bg=bg)
            for w in child.winfo_children():
                # Don't change type_tag background
                if w is type_tag:
                    continue
                try:
                    w.configure(bg=bg)
                except tk.TclError:
                    pass

    def _update_thumbnails(self) -> None:
        """Show small thumbnails in the thumbnail grid at the bottom."""
        for child in self._thumb_frame.winfo_children():
            child.destroy()

        max_thumbs = 6
        steps_to_show = self._steps[-max_thumbs:]
        start_idx = max(0, len(self._steps) - max_thumbs)

        row_frame = None
        for i, step in enumerate(steps_to_show):
            real_idx = start_idx + i
            if i % 3 == 0:
                row_frame = ttk.Frame(self._thumb_frame)
                row_frame.pack(fill=tk.X, pady=1)

            thumb_size = 56
            step.make_thumbnail(thumb_size)
            btn = tk.Label(
                row_frame,
                image=step.photo,
                bg="#1e1e1e",
                cursor="hand2",
                borderwidth=1,
                relief=tk.RIDGE,
            )
            btn.pack(side=tk.LEFT, padx=1, pady=1)
            btn.bind("<Button-1>", lambda e, idx=real_idx: self.select_step(idx))

    # ------------------------------------------------------------------
    # Drag-and-drop reordering
    # ------------------------------------------------------------------

    def _on_row_click(self, event: tk.Event, index: int) -> None:
        """Record initial click position and select the step."""
        self._drag_source_index = None  # reset; only becomes a drag on motion
        self._drag_start_y = event.y_root
        self.select_step(index)

    def _on_drag_motion(self, event: tk.Event, source_index: int) -> None:
        """Track mouse movement during drag and show drop indicator."""
        # Only start drag after a small threshold to avoid accidental drags
        if self._drag_source_index is None:
            dy = abs(event.y_root - getattr(self, "_drag_start_y", event.y_root))
            if dy < 5:
                return
            self._drag_source_index = source_index

        # Determine which row the cursor is over
        target_index = self._row_index_at_y(event.y_root)

        # Show / move the drop indicator line
        self._show_drag_indicator(target_index)

    def _on_drag_release(self, event: tk.Event, source_index: int) -> None:
        """Complete the drag-and-drop: move the step in the list."""
        self._remove_drag_indicator()

        if self._drag_source_index is None:
            return  # was just a click, not a drag

        src = self._drag_source_index
        self._drag_source_index = None

        target = self._row_index_at_y(event.y_root)
        if target is None or target == src:
            return

        # Move the step in the internal list
        step = self._steps.pop(src)
        self._steps.insert(target, step)

        # Adjust selected index to follow the moved step
        if self._selected_index == src:
            self._selected_index = target
        elif src < self._selected_index <= target:
            self._selected_index -= 1
        elif target <= self._selected_index < src:
            self._selected_index += 1

        self._rebuild_step_list()
        self._update_selection_visual()
        self._update_thumbnails()

    def _row_index_at_y(self, y_root: int) -> Optional[int]:
        """Return the step index corresponding to *y_root* screen coordinate."""
        children = self._inner_frame.winfo_children()
        if not children:
            return None
        for child in children:
            try:
                wy = child.winfo_rooty()
                wh = child.winfo_height()
                if y_root < wy + wh // 2:
                    return getattr(child, "_step_index", 0)
            except tk.TclError:
                continue
        # Past the last row → return last index
        last = children[-1]
        return getattr(last, "_step_index", len(self._steps) - 1)

    def _show_drag_indicator(self, target_index: Optional[int]) -> None:
        """Draw a thin blue line between rows to indicate drop position."""
        self._remove_drag_indicator()
        if target_index is None:
            return
        children = self._inner_frame.winfo_children()
        if not children:
            return

        # Find the widget for target_index
        ref_widget = None
        for child in children:
            if getattr(child, "_step_index", -1) == target_index:
                ref_widget = child
                break

        self._drag_indicator = tk.Frame(
            self._inner_frame, bg="#4fc3f7", height=2,
        )
        if ref_widget is not None:
            # Place just before the target widget
            self._drag_indicator.pack(fill=tk.X, before=ref_widget, pady=0)
        else:
            self._drag_indicator.pack(fill=tk.X, pady=0)

    def _remove_drag_indicator(self) -> None:
        """Remove the drop indicator line."""
        if self._drag_indicator is not None:
            self._drag_indicator.destroy()
            self._drag_indicator = None

    # ------------------------------------------------------------------
    # Context menu
    # ------------------------------------------------------------------

    def _show_context_menu(self, event: tk.Event, index: int) -> None:
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(
            label="\u6aa2\u8996\u539f\u5716",
            command=lambda: self.select_step(index),
        )
        menu.add_command(
            label="\u532f\u51fa\u5716\u7247",
            command=lambda: self._export_step(index),
        )
        menu.add_separator()
        menu.add_command(
            label="\u522a\u9664\u6b65\u9a5f",
            command=lambda: self._delete_step_action(index),
        )
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def _export_step(self, index: int) -> None:
        if self._on_step_export:
            self._on_step_export(index)

    def _delete_step_action(self, index: int) -> None:
        if self._on_step_delete:
            self._on_step_delete(index)
        else:
            self.delete_step(index)
