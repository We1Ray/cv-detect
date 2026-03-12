"""ROI Manager Dialog for the CV defect detection application.

Provides a Tkinter Toplevel dialog for creating, editing, deleting, and
applying ROIs (Regions of Interest).  Supports rectangle, rotated rectangle,
circle, ellipse, polygon, and ring ROI types with JSON persistence.
"""
from __future__ import annotations

import copy
import logging
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from core.roi_manager import ROI, ROIManager, draw_rois

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Theme constants
# ---------------------------------------------------------------------------
_BG = "#2b2b2b"
_FG = "#e0e0e0"
_BG_BUTTON = "#3c3c3c"
_BG_ACTIVE = "#3a3a5c"
_BG_ENTRY = "#3c3c3c"
_ACCENT = "#0078d4"

# ---------------------------------------------------------------------------
#  ROI type labels (Traditional Chinese -> internal key)
# ---------------------------------------------------------------------------
_TYPE_LABELS: List[str] = [
    "矩形",
    "旋轉矩形",
    "圓形",
    "橢圓",
    "多邊形",
    "環形",
]

_TYPE_MAP: Dict[str, str] = {
    "矩形": "rectangle",
    "旋轉矩形": "rotated_rectangle",
    "圓形": "circle",
    "橢圓": "ellipse",
    "多邊形": "polygon",
    "環形": "ring",
}

# Parameter definitions per type: list of (label, param_key)
_PARAM_DEFS: Dict[str, List[Tuple[str, str]]] = {
    "矩形": [("X", "x"), ("Y", "y"), ("寬度", "width"), ("高度", "height")],
    "旋轉矩形": [
        ("中心X", "cx"),
        ("中心Y", "cy"),
        ("寬度", "width"),
        ("高度", "height"),
        ("角度(\u00b0)", "angle"),
    ],
    "圓形": [("中心X", "cx"), ("中心Y", "cy"), ("半徑", "radius")],
    "橢圓": [
        ("中心X", "cx"),
        ("中心Y", "cy"),
        ("半長軸", "rx"),
        ("半短軸", "ry"),
        ("角度(\u00b0)", "angle"),
    ],
    "多邊形": [],  # special: uses a text area
    "環形": [
        ("中心X", "cx"),
        ("中心Y", "cy"),
        ("內半徑", "inner_radius"),
        ("外半徑", "outer_radius"),
    ],
}

# ---------------------------------------------------------------------------
#  Colour map
# ---------------------------------------------------------------------------
COLOR_MAP: Dict[str, Tuple[int, int, int]] = {
    "綠": (0, 255, 0),
    "紅": (0, 0, 255),
    "藍": (255, 0, 0),
    "黃": (0, 255, 255),
    "青": (255, 255, 0),
    "紫": (255, 0, 255),
}

_COLOR_NAMES: List[str] = list(COLOR_MAP.keys())


# ======================================================================== #
#  ROIManagerDialog                                                         #
# ======================================================================== #


class ROIManagerDialog(tk.Toplevel):
    """Modal dialog for managing ROIs.

    Parameters
    ----------
    master : tk.Widget
        Parent widget.
    get_current_image : callable
        Returns the current ``np.ndarray`` image (BGR or grayscale).
    add_pipeline_step : callable
        ``add_pipeline_step(image, name)`` to push a result into the
        processing pipeline.
    set_status : callable
        ``set_status(text)`` to update the main window status bar.
    viewer : optional
        An ``ImageViewer`` instance that provides ``get_region_selection()``
        returning ``(x, y, w, h)`` or ``None``.
    """

    def __init__(
        self,
        master: tk.Widget,
        get_current_image: Callable[[], Optional[np.ndarray]],
        add_pipeline_step: Callable[[np.ndarray, str], None],
        set_status: Callable[[str], None],
        viewer: Any = None,
    ) -> None:
        super().__init__(master)

        self.title("ROI 管理")
        self.geometry("800x550")
        self.minsize(750, 500)
        self.configure(bg=_BG)
        self.transient(master)
        self.grab_set()

        self._get_current_image = get_current_image
        self._add_pipeline_step = add_pipeline_step
        self._set_status = set_status
        self._viewer = viewer

        self._roi_manager = ROIManager()
        self._param_widgets: List[Tuple[tk.Label, tk.Entry, str]] = []
        self._polygon_text: Optional[tk.Text] = None

        self._build_ui()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------ #
    #  UI construction                                                     #
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        # Main two-column layout: left = list, right = controls
        left = tk.Frame(self, bg=_BG)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 4), pady=8)

        right = tk.Frame(self, bg=_BG)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(4, 8), pady=8)

        self._build_roi_list(left)
        self._build_controls(right)

    # ---- Left column: ROI list ----------------------------------------

    def _build_roi_list(self, parent: tk.Frame) -> None:
        header = tk.Label(
            parent, text="ROI 列表", bg=_BG, fg=_FG,
            font=("", 10, "bold"), anchor=tk.W,
        )
        header.pack(fill=tk.X, pady=(0, 4))

        columns = ("#", "名稱", "類型", "面積", "可見")
        self._tree = ttk.Treeview(
            parent, columns=columns, show="headings",
            selectmode="browse", height=14,
        )

        col_widths = {"#": 40, "名稱": 100, "類型": 80, "面積": 90, "可見": 50}
        for col in columns:
            self._tree.heading(col, text=col)
            self._tree.column(
                col, width=col_widths.get(col, 80),
                anchor=tk.CENTER, minwidth=35,
            )

        tree_scroll = ttk.Scrollbar(
            parent, orient=tk.VERTICAL, command=self._tree.yview,
        )
        self._tree.configure(yscrollcommand=tree_scroll.set)
        self._tree.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self._tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        # Buttons below treeview
        btn_frame = tk.Frame(parent, bg=_BG)
        btn_frame.pack(fill=tk.X, pady=(6, 0))

        for text, cmd in [
            ("刪除", self._delete_roi),
            ("複製", self._duplicate_roi),
            ("全部清除", self._clear_all),
        ]:
            tk.Button(
                btn_frame, text=text, command=cmd,
                bg=_BG_BUTTON, fg=_FG, activebackground=_BG_ACTIVE,
                relief=tk.FLAT, padx=8, pady=3,
            ).pack(side=tk.LEFT, padx=(0, 6))

    # ---- Right column: controls ---------------------------------------

    def _build_controls(self, parent: tk.Frame) -> None:
        # Make the right column scrollable via a canvas if needed,
        # but 550px height should suffice.  Use pack for sections.

        # === Add ROI ===
        add_frame = tk.LabelFrame(
            parent, text=" 新增 ROI ", bg=_BG, fg=_FG,
            font=("", 9, "bold"), padx=6, pady=4,
        )
        add_frame.pack(fill=tk.X, pady=(0, 6))

        # Type combobox
        type_row = tk.Frame(add_frame, bg=_BG)
        type_row.pack(fill=tk.X, pady=2)
        tk.Label(type_row, text="類型:", bg=_BG, fg=_FG, width=8, anchor=tk.E).pack(side=tk.LEFT)
        self._type_var = tk.StringVar(value=_TYPE_LABELS[0])
        self._type_combo = ttk.Combobox(
            type_row, textvariable=self._type_var,
            values=_TYPE_LABELS, state="readonly", width=14,
        )
        self._type_combo.pack(side=tk.LEFT, padx=4)
        self._type_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_type_changed())

        # Name entry
        name_row = tk.Frame(add_frame, bg=_BG)
        name_row.pack(fill=tk.X, pady=2)
        tk.Label(name_row, text="名稱:", bg=_BG, fg=_FG, width=8, anchor=tk.E).pack(side=tk.LEFT)
        self._name_var = tk.StringVar(value="")
        tk.Entry(
            name_row, textvariable=self._name_var, width=16,
            bg=_BG_ENTRY, fg=_FG, insertbackground=_FG, relief=tk.FLAT,
        ).pack(side=tk.LEFT, padx=4)

        # Colour combobox
        color_row = tk.Frame(add_frame, bg=_BG)
        color_row.pack(fill=tk.X, pady=2)
        tk.Label(color_row, text="顏色:", bg=_BG, fg=_FG, width=8, anchor=tk.E).pack(side=tk.LEFT)
        self._color_var = tk.StringVar(value=_COLOR_NAMES[0])
        ttk.Combobox(
            color_row, textvariable=self._color_var,
            values=_COLOR_NAMES, state="readonly", width=8,
        ).pack(side=tk.LEFT, padx=4)

        # Dynamic parameter area (grid layout)
        self._param_frame = tk.Frame(add_frame, bg=_BG)
        self._param_frame.pack(fill=tk.X, pady=(4, 2))

        self._on_type_changed()  # populate initial params

        # From selection + Add buttons
        btn_row = tk.Frame(add_frame, bg=_BG)
        btn_row.pack(fill=tk.X, pady=(4, 0))

        tk.Button(
            btn_row, text="從選取區域建立", command=self._from_selection,
            bg=_BG_BUTTON, fg=_FG, activebackground=_BG_ACTIVE,
            relief=tk.FLAT, padx=6, pady=3,
        ).pack(side=tk.LEFT, padx=(0, 6))

        tk.Button(
            btn_row, text="新增", command=self._add_roi,
            bg=_ACCENT, fg="#ffffff", activebackground="#005a9e",
            relief=tk.FLAT, padx=10, pady=3,
        ).pack(side=tk.LEFT)

        # === ROI Properties ===
        prop_frame = tk.LabelFrame(
            parent, text=" ROI 屬性 ", bg=_BG, fg=_FG,
            font=("", 9, "bold"), padx=6, pady=4,
        )
        prop_frame.pack(fill=tk.X, pady=(0, 6))

        # Name
        pname_row = tk.Frame(prop_frame, bg=_BG)
        pname_row.pack(fill=tk.X, pady=2)
        tk.Label(pname_row, text="名稱:", bg=_BG, fg=_FG, width=8, anchor=tk.E).pack(side=tk.LEFT)
        self._prop_name_var = tk.StringVar()
        tk.Entry(
            pname_row, textvariable=self._prop_name_var, width=16,
            bg=_BG_ENTRY, fg=_FG, insertbackground=_FG, relief=tk.FLAT,
        ).pack(side=tk.LEFT, padx=4)

        # Visible
        self._prop_visible_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            prop_frame, text="可見", variable=self._prop_visible_var,
            bg=_BG, fg=_FG, selectcolor=_BG_BUTTON, activebackground=_BG,
            activeforeground=_FG,
        ).pack(anchor=tk.W, pady=1)

        # Locked
        self._prop_locked_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            prop_frame, text="鎖定", variable=self._prop_locked_var,
            bg=_BG, fg=_FG, selectcolor=_BG_BUTTON, activebackground=_BG,
            activeforeground=_FG,
        ).pack(anchor=tk.W, pady=1)

        # Update button
        tk.Button(
            prop_frame, text="更新", command=self._update_roi_properties,
            bg=_BG_BUTTON, fg=_FG, activebackground=_BG_ACTIVE,
            relief=tk.FLAT, padx=8, pady=3,
        ).pack(anchor=tk.W, pady=(4, 0))

        # === Operations ===
        ops_frame = tk.LabelFrame(
            parent, text=" 操作 ", bg=_BG, fg=_FG,
            font=("", 9, "bold"), padx=6, pady=4,
        )
        ops_frame.pack(fill=tk.X, pady=(0, 6))

        op_buttons: List[Tuple[str, Callable[[], None]]] = [
            ("套用到圖片", self._apply_to_image),
            ("裁切", self._crop),
            ("反轉遮罩", self._inverse_mask),
            ("繪製所有 ROI", self._draw_all_rois),
        ]
        for text, cmd in op_buttons:
            tk.Button(
                ops_frame, text=text, command=cmd,
                bg=_BG_BUTTON, fg=_FG, activebackground=_BG_ACTIVE,
                relief=tk.FLAT, padx=6, pady=3, anchor=tk.W,
            ).pack(fill=tk.X, pady=1)

        ttk.Separator(ops_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)

        for text, cmd in [
            ("儲存 ROI...", self._save_rois),
            ("載入 ROI...", self._load_rois),
        ]:
            tk.Button(
                ops_frame, text=text, command=cmd,
                bg=_BG_BUTTON, fg=_FG, activebackground=_BG_ACTIVE,
                relief=tk.FLAT, padx=6, pady=3, anchor=tk.W,
            ).pack(fill=tk.X, pady=1)

    # ------------------------------------------------------------------ #
    #  Dynamic parameter entries                                           #
    # ------------------------------------------------------------------ #

    def _on_type_changed(self) -> None:
        """Rebuild parameter entry widgets when the ROI type changes."""
        # Clear existing widgets
        for child in self._param_frame.winfo_children():
            child.destroy()
        self._param_widgets.clear()
        self._polygon_text = None

        roi_type_label = self._type_var.get()

        if roi_type_label == "多邊形":
            tk.Label(
                self._param_frame, text="頂點 (x1,y1; x2,y2; ...):",
                bg=_BG, fg=_FG, font=("", 8),
            ).pack(anchor=tk.W)
            self._polygon_text = tk.Text(
                self._param_frame, height=3, width=30,
                bg=_BG_ENTRY, fg=_FG, insertbackground=_FG,
                relief=tk.FLAT, font=("Consolas", 9),
            )
            self._polygon_text.pack(fill=tk.X, pady=2)
            return

        params = _PARAM_DEFS.get(roi_type_label, [])
        for row_idx, (label_text, key) in enumerate(params):
            lbl = tk.Label(
                self._param_frame, text=f"{label_text}:",
                bg=_BG, fg=_FG, width=10, anchor=tk.E,
            )
            lbl.grid(row=row_idx, column=0, sticky=tk.E, pady=1)

            entry = tk.Entry(
                self._param_frame, width=10,
                bg=_BG_ENTRY, fg=_FG, insertbackground=_FG,
                relief=tk.FLAT,
            )
            entry.grid(row=row_idx, column=1, sticky=tk.W, padx=4, pady=1)
            entry.insert(0, "0")

            self._param_widgets.append((lbl, entry, key))

    # ------------------------------------------------------------------ #
    #  Treeview helpers                                                    #
    # ------------------------------------------------------------------ #

    def _refresh_tree(self) -> None:
        """Rebuild the treeview from the ROIManager state."""
        for item in self._tree.get_children():
            self._tree.delete(item)

        for idx, roi in enumerate(self._roi_manager.get_all_rois()):
            # Map internal type to display label
            display_type = roi.roi_type
            for label, key in _TYPE_MAP.items():
                if key == roi.roi_type:
                    display_type = label
                    break

            area_str = f"{roi.area():.0f}"
            visible_str = "是" if roi.visible else "否"
            self._tree.insert(
                "", tk.END, iid=str(idx),
                values=(idx, roi.name, display_type, area_str, visible_str),
            )

    def _selected_index(self) -> Optional[int]:
        """Return the currently selected ROI index, or None."""
        sel = self._tree.selection()
        if not sel:
            return None
        return int(sel[0])

    def _on_tree_select(self, _event: tk.Event) -> None:
        """Populate the properties panel when a ROI is selected."""
        idx = self._selected_index()
        if idx is None:
            return
        try:
            roi = self._roi_manager.get_roi(idx)
        except IndexError:
            return

        self._prop_name_var.set(roi.name)
        self._prop_visible_var.set(roi.visible)
        self._prop_locked_var.set(roi.locked)

    # ------------------------------------------------------------------ #
    #  ROI list buttons                                                    #
    # ------------------------------------------------------------------ #

    def _delete_roi(self) -> None:
        idx = self._selected_index()
        if idx is None:
            messagebox.showwarning("提示", "請先選取一個 ROI", parent=self)
            return
        try:
            self._roi_manager.remove_roi(idx)
        except IndexError:
            return
        self._refresh_tree()
        self._set_status(f"已刪除 ROI #{idx}")

    def _duplicate_roi(self) -> None:
        idx = self._selected_index()
        if idx is None:
            messagebox.showwarning("提示", "請先選取一個 ROI", parent=self)
            return
        try:
            new_idx = self._roi_manager.duplicate_roi(idx)
        except IndexError:
            return
        self._refresh_tree()
        self._set_status(f"已複製 ROI #{idx} -> #{new_idx}")

    def _clear_all(self) -> None:
        if len(self._roi_manager) == 0:
            return
        if not messagebox.askyesno("確認", "確定要清除所有 ROI？", parent=self):
            return
        self._roi_manager.clear()
        self._refresh_tree()
        self._set_status("已清除所有 ROI")

    # ------------------------------------------------------------------ #
    #  Add ROI                                                             #
    # ------------------------------------------------------------------ #

    def _add_roi(self) -> None:
        """Create a new ROI from the parameter entries and add it."""
        roi_type_label = self._type_var.get()
        internal_type = _TYPE_MAP.get(roi_type_label)
        if internal_type is None:
            messagebox.showerror("錯誤", f"未知的 ROI 類型: {roi_type_label}", parent=self)
            return

        # Build params dict
        try:
            if roi_type_label == "多邊形":
                params = self._parse_polygon_params()
            else:
                params = self._parse_entry_params()
        except ValueError as exc:
            messagebox.showerror("參數錯誤", str(exc), parent=self)
            return

        # Name and colour
        name = self._name_var.get().strip()
        if not name:
            name = f"ROI_{len(self._roi_manager)}"

        color_label = self._color_var.get()
        color = COLOR_MAP.get(color_label, (0, 255, 0))

        try:
            roi = ROI(
                roi_type=internal_type,
                params=params,
                name=name,
                color=color,
            )
            self._roi_manager.add_roi(roi)
        except (ValueError, TypeError) as exc:
            messagebox.showerror("錯誤", str(exc), parent=self)
            return

        # Update image shape if possible
        img = self._get_current_image()
        if img is not None:
            self._roi_manager.set_image_shape(img.shape[:2])

        self._refresh_tree()
        self._set_status(f"已新增 ROI: {roi.name}")

    def _parse_entry_params(self) -> Dict[str, Any]:
        """Parse numeric parameters from the grid entries."""
        params: Dict[str, Any] = {}
        for _lbl, entry, key in self._param_widgets:
            text = entry.get().strip()
            if not text:
                raise ValueError(f"參數 '{key}' 不可為空")
            try:
                params[key] = float(text)
            except ValueError:
                raise ValueError(f"參數 '{key}' 必須為數值，目前為: {text}")
        return params

    def _parse_polygon_params(self) -> Dict[str, Any]:
        """Parse polygon points from the text area."""
        if self._polygon_text is None:
            raise ValueError("找不到多邊形輸入區域")
        raw = self._polygon_text.get("1.0", tk.END).strip()
        if not raw:
            raise ValueError("請輸入多邊形頂點，格式: x1,y1; x2,y2; ...")

        points: List[Tuple[float, float]] = []
        for part in raw.split(";"):
            part = part.strip()
            if not part:
                continue
            coords = part.split(",")
            if len(coords) != 2:
                raise ValueError(f"頂點格式錯誤: '{part}'，應為 x,y")
            try:
                x = float(coords[0].strip())
                y = float(coords[1].strip())
            except ValueError:
                raise ValueError(f"頂點座標必須為數值: '{part}'")
            points.append((x, y))

        if len(points) < 3:
            raise ValueError("多邊形至少需要 3 個頂點")

        return {"points": points}

    # ------------------------------------------------------------------ #
    #  From selection                                                      #
    # ------------------------------------------------------------------ #

    def _from_selection(self) -> None:
        """Create a rectangle ROI from the viewer's current selection."""
        if self._viewer is None:
            messagebox.showwarning("提示", "無可用的影像檢視器", parent=self)
            return

        sel = self._viewer.get_region_selection()
        if sel is None:
            messagebox.showwarning("提示", "請先在影像上框選一個區域", parent=self)
            return

        x, y, w, h = sel

        # Populate the rectangle parameter entries
        self._type_var.set("矩形")
        self._on_type_changed()

        # Fill entries with the selection values
        values = {"x": x, "y": y, "width": w, "height": h}
        for _lbl, entry, key in self._param_widgets:
            if key in values:
                entry.delete(0, tk.END)
                entry.insert(0, str(int(values[key])))

        self._set_status(f"已從選取區域填入參數: ({x}, {y}, {w}, {h})")

    # ------------------------------------------------------------------ #
    #  Update properties                                                   #
    # ------------------------------------------------------------------ #

    def _update_roi_properties(self) -> None:
        """Apply the edited properties to the selected ROI."""
        idx = self._selected_index()
        if idx is None:
            messagebox.showwarning("提示", "請先選取一個 ROI", parent=self)
            return

        try:
            roi = self._roi_manager.get_roi(idx)
        except IndexError:
            return

        new_name = self._prop_name_var.get().strip()
        if new_name:
            roi.name = new_name
        roi.visible = self._prop_visible_var.get()
        roi.locked = self._prop_locked_var.get()

        self._refresh_tree()
        self._set_status(f"已更新 ROI #{idx} 屬性")

    # ------------------------------------------------------------------ #
    #  Operations                                                          #
    # ------------------------------------------------------------------ #

    def _require_image(self) -> Optional[np.ndarray]:
        """Get the current image, showing a warning if unavailable."""
        img = self._get_current_image()
        if img is None:
            messagebox.showwarning("提示", "目前沒有載入圖片", parent=self)
        return img

    def _apply_to_image(self) -> None:
        """Mask the current image with the selected ROI."""
        idx = self._selected_index()
        if idx is None:
            messagebox.showwarning("提示", "請先選取一個 ROI", parent=self)
            return
        img = self._require_image()
        if img is None:
            return

        self._roi_manager.set_image_shape(img.shape[:2])
        try:
            result = self._roi_manager.apply_roi_to_image(img, idx)
        except (IndexError, RuntimeError) as exc:
            messagebox.showerror("錯誤", str(exc), parent=self)
            return

        roi = self._roi_manager.get_roi(idx)
        self._add_pipeline_step(result, f"ROI 遮罩: {roi.name}")
        self._set_status(f"已套用 ROI '{roi.name}' 遮罩到圖片")

    def _crop(self) -> None:
        """Crop the current image to the selected ROI bounding box."""
        idx = self._selected_index()
        if idx is None:
            messagebox.showwarning("提示", "請先選取一個 ROI", parent=self)
            return
        img = self._require_image()
        if img is None:
            return

        self._roi_manager.set_image_shape(img.shape[:2])
        try:
            cropped, offset = self._roi_manager.crop_roi(img, idx)
        except (IndexError, RuntimeError) as exc:
            messagebox.showerror("錯誤", str(exc), parent=self)
            return

        roi = self._roi_manager.get_roi(idx)
        self._add_pipeline_step(cropped, f"ROI 裁切: {roi.name}")
        self._set_status(
            f"已裁切 ROI '{roi.name}' (偏移: {offset[0]}, {offset[1]})"
        )

    def _inverse_mask(self) -> None:
        """Apply an inverse mask of all visible ROIs to the current image."""
        img = self._require_image()
        if img is None:
            return
        if len(self._roi_manager) == 0:
            messagebox.showwarning("提示", "尚未新增任何 ROI", parent=self)
            return

        self._roi_manager.set_image_shape(img.shape[:2])
        try:
            inv_mask = self._roi_manager.get_inverse_mask()
        except RuntimeError as exc:
            messagebox.showerror("錯誤", str(exc), parent=self)
            return

        result = img.copy()
        if result.ndim == 3:
            result[inv_mask == 0] = 0
        else:
            result[inv_mask == 0] = 0

        self._add_pipeline_step(result, "ROI 反轉遮罩")
        self._set_status("已套用反轉遮罩")

    def _draw_all_rois(self) -> None:
        """Draw all ROI outlines on the current image."""
        img = self._require_image()
        if img is None:
            return
        if len(self._roi_manager) == 0:
            messagebox.showwarning("提示", "尚未新增任何 ROI", parent=self)
            return

        # Ensure BGR for drawing
        if img.ndim == 2:
            draw_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            draw_img = img.copy()

        result = draw_rois(
            draw_img, self._roi_manager.get_all_rois(),
            show_names=True, show_handles=False, thickness=2,
        )

        self._add_pipeline_step(result, "繪製所有 ROI")
        self._set_status(f"已繪製 {len(self._roi_manager)} 個 ROI")

    # ------------------------------------------------------------------ #
    #  Save / Load                                                         #
    # ------------------------------------------------------------------ #

    def _save_rois(self) -> None:
        """Save all ROIs to a JSON file."""
        if len(self._roi_manager) == 0:
            messagebox.showwarning("提示", "尚未新增任何 ROI", parent=self)
            return

        path = filedialog.asksaveasfilename(
            title="儲存 ROI",
            defaultextension=".json",
            filetypes=[("JSON 檔案", "*.json"), ("所有檔案", "*.*")],
            parent=self,
        )
        if not path:
            return

        try:
            self._roi_manager.save(path)
        except OSError as exc:
            messagebox.showerror("儲存錯誤", str(exc), parent=self)
            return

        self._set_status(f"已儲存 {len(self._roi_manager)} 個 ROI 至 {path}")

    def _load_rois(self) -> None:
        """Load ROIs from a JSON file."""
        path = filedialog.askopenfilename(
            title="載入 ROI",
            filetypes=[("JSON 檔案", "*.json"), ("所有檔案", "*.*")],
            parent=self,
        )
        if not path:
            return

        try:
            self._roi_manager.load(path)
        except (OSError, ValueError, KeyError) as exc:
            messagebox.showerror("載入錯誤", str(exc), parent=self)
            return

        self._refresh_tree()
        self._set_status(f"已載入 {len(self._roi_manager)} 個 ROI 從 {path}")

    # ------------------------------------------------------------------ #
    #  Close                                                               #
    # ------------------------------------------------------------------ #

    def _on_close(self) -> None:
        self.grab_release()
        self.destroy()
