"""
gui/blob_analysis.py - Industrial Vision-style Blob Analysis dialog.

Provides a complete image processing pipeline dialog for industrial defect
detection.  The pipeline executes the following steps when the user clicks
the "Execute Analysis" button:

1. Convert to grayscale
2. Optional Gaussian blur
3. Threshold (manual min/max or Otsu auto)
4. Optional morphological opening / closing
5. Connected-component labeling
6. Feature computation
7. Area-based filtering
8. Overlay visualization (coloured regions + bbox + labels + centroid cross)

Results are displayed in an interactive Treeview table.  Clicking a row
highlights the corresponding region on the preview canvas.  The user can
export the table to CSV or push the entire pipeline into the parent
application via the ``on_accept`` callback.
"""
from __future__ import annotations

import csv
import logging
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk

from dl_anomaly.core.region import Region, RegionProperties
from dl_anomaly.core.region_ops import (
    binary_threshold,
    compute_region_properties,
    connection,
    region_to_display_image,
    select_shape,
    threshold,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Theme constants
# ---------------------------------------------------------------------------
_BG = "#2b2b2b"
_BG_MEDIUM = "#3c3c3c"
_FG = "#cccccc"
_FG_WHITE = "#e0e0e0"
_ACCENT = "#0078d4"
_CANVAS_BG = "#1e1e1e"


# ======================================================================== #
#  BlobAnalysisDialog                                                       #
# ======================================================================== #

class BlobAnalysisDialog(tk.Toplevel):
    """Industrial Vision-style Blob Analysis dialog.

    Parameters
    ----------
    master : tk.Widget
        Parent widget.
    image : np.ndarray
        Source image (BGR colour or single-channel grayscale).
    on_accept : callable, optional
        Callback invoked when the user clicks "Add to Pipeline".
        Signature: ``on_accept(steps)`` where *steps* is
        ``List[Tuple[str, np.ndarray, Optional[Region]]]``.
    """

    def __init__(
        self,
        master: tk.Widget,
        image: np.ndarray,
        on_accept: Optional[Callable] = None,
    ) -> None:
        super().__init__(master)
        self.title("Blob Analysis")
        self.geometry("920x800")
        self.resizable(True, True)
        self.configure(bg=_BG)

        # Modal behaviour
        self.transient(master)
        self.grab_set()

        self._source_image = image.copy()
        self._gray = self._to_gray(image)
        self._on_accept = on_accept

        # Pipeline results -------------------------------------------------
        self._pipeline_steps: List[Tuple[str, np.ndarray, Optional[Region]]] = []
        self._result_region: Optional[Region] = None
        self._display_image: Optional[np.ndarray] = None

        # PhotoImage reference (prevent GC)
        self._preview_photo: Optional[ImageTk.PhotoImage] = None

        # Currently highlighted region index (1-based), None = none
        self._highlight_index: Optional[int] = None

        # ------------------------------------------------------------------
        # Tkinter variables for parameters
        # ------------------------------------------------------------------
        self._threshold_mode_var = tk.StringVar(value="手動")
        self._thresh_min_var = tk.StringVar(value="0")
        self._thresh_max_var = tk.StringVar(value="255")

        self._blur_kernel_var = tk.IntVar(value=0)

        self._morph_mode_var = tk.StringVar(value="無")
        self._morph_kernel_var = tk.IntVar(value=3)

        self._area_min_var = tk.StringVar(value="0")
        self._area_max_var = tk.StringVar(value="999999999")

        # Build UI
        self._build_ui()

        self.protocol("WM_DELETE_WINDOW", self._close)

    # ------------------------------------------------------------------ #
    #  UI construction                                                     #
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        """Construct all widgets from top to bottom."""
        # ========== 1. Parameters frame ========== #
        param_frame = tk.LabelFrame(
            self,
            text=" 分析參數 ",
            bg=_BG,
            fg=_FG,
            font=("", 10, "bold"),
            padx=8,
            pady=6,
        )
        param_frame.pack(fill=tk.X, padx=10, pady=(8, 4))

        # --- Row 1: Threshold mode + min / max ---
        row1 = tk.Frame(param_frame, bg=_BG)
        row1.pack(fill=tk.X, pady=2)

        tk.Label(
            row1, text="閾值模式:", bg=_BG, fg=_FG, font=("", 9),
        ).pack(side=tk.LEFT, padx=(0, 4))

        self._threshold_combo = ttk.Combobox(
            row1,
            textvariable=self._threshold_mode_var,
            values=["手動", "Otsu"],
            state="readonly",
            width=8,
        )
        self._threshold_combo.pack(side=tk.LEFT, padx=(0, 12))
        self._threshold_combo.bind(
            "<<ComboboxSelected>>", self._on_threshold_mode_changed,
        )

        tk.Label(
            row1, text="最小:", bg=_BG, fg=_FG, font=("", 9),
        ).pack(side=tk.LEFT, padx=(0, 2))
        self._thresh_min_entry = tk.Entry(
            row1,
            textvariable=self._thresh_min_var,
            width=6,
            bg=_BG_MEDIUM,
            fg=_FG_WHITE,
            insertbackground=_FG_WHITE,
            relief=tk.FLAT,
        )
        self._thresh_min_entry.pack(side=tk.LEFT, padx=(0, 8))

        tk.Label(
            row1, text="最大:", bg=_BG, fg=_FG, font=("", 9),
        ).pack(side=tk.LEFT, padx=(0, 2))
        self._thresh_max_entry = tk.Entry(
            row1,
            textvariable=self._thresh_max_var,
            width=6,
            bg=_BG_MEDIUM,
            fg=_FG_WHITE,
            insertbackground=_FG_WHITE,
            relief=tk.FLAT,
        )
        self._thresh_max_entry.pack(side=tk.LEFT)

        # --- Row 2: Blur kernel + Morphology mode + kernel ---
        row2 = tk.Frame(param_frame, bg=_BG)
        row2.pack(fill=tk.X, pady=2)

        tk.Label(
            row2, text="模糊核心:", bg=_BG, fg=_FG, font=("", 9),
        ).pack(side=tk.LEFT, padx=(0, 4))
        self._blur_spin = tk.Spinbox(
            row2,
            textvariable=self._blur_kernel_var,
            values=(0, 3, 5, 7, 9, 11),
            width=4,
            bg=_BG_MEDIUM,
            fg=_FG_WHITE,
            buttonbackground=_BG_MEDIUM,
            insertbackground=_FG_WHITE,
            relief=tk.FLAT,
        )
        self._blur_spin.pack(side=tk.LEFT, padx=(0, 16))

        tk.Label(
            row2, text="形態學:", bg=_BG, fg=_FG, font=("", 9),
        ).pack(side=tk.LEFT, padx=(0, 4))
        self._morph_combo = ttk.Combobox(
            row2,
            textvariable=self._morph_mode_var,
            values=["無", "開運算", "閉運算"],
            state="readonly",
            width=8,
        )
        self._morph_combo.pack(side=tk.LEFT, padx=(0, 8))

        tk.Label(
            row2, text="形態學核心:", bg=_BG, fg=_FG, font=("", 9),
        ).pack(side=tk.LEFT, padx=(0, 4))
        self._morph_kernel_spin = tk.Spinbox(
            row2,
            textvariable=self._morph_kernel_var,
            from_=3,
            to=31,
            increment=2,
            width=4,
            bg=_BG_MEDIUM,
            fg=_FG_WHITE,
            buttonbackground=_BG_MEDIUM,
            insertbackground=_FG_WHITE,
            relief=tk.FLAT,
        )
        self._morph_kernel_spin.pack(side=tk.LEFT)

        # --- Row 3: Area filter ---
        row3 = tk.Frame(param_frame, bg=_BG)
        row3.pack(fill=tk.X, pady=2)

        tk.Label(
            row3, text="最小面積:", bg=_BG, fg=_FG, font=("", 9),
        ).pack(side=tk.LEFT, padx=(0, 4))
        self._area_min_entry = tk.Entry(
            row3,
            textvariable=self._area_min_var,
            width=10,
            bg=_BG_MEDIUM,
            fg=_FG_WHITE,
            insertbackground=_FG_WHITE,
            relief=tk.FLAT,
        )
        self._area_min_entry.pack(side=tk.LEFT, padx=(0, 16))

        tk.Label(
            row3, text="最大面積:", bg=_BG, fg=_FG, font=("", 9),
        ).pack(side=tk.LEFT, padx=(0, 4))
        self._area_max_entry = tk.Entry(
            row3,
            textvariable=self._area_max_var,
            width=10,
            bg=_BG_MEDIUM,
            fg=_FG_WHITE,
            insertbackground=_FG_WHITE,
            relief=tk.FLAT,
        )
        self._area_max_entry.pack(side=tk.LEFT)

        # --- Row 4: Execute button ---
        row4 = tk.Frame(param_frame, bg=_BG)
        row4.pack(fill=tk.X, pady=(6, 2))

        self._run_btn = tk.Button(
            row4,
            text="執行分析",
            bg=_ACCENT,
            fg="#ffffff",
            activebackground="#005a9e",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=24,
            pady=4,
            font=("", 10, "bold"),
            command=self._run_analysis,
        )
        self._run_btn.pack(side=tk.LEFT)

        # ========== 2. Preview canvas ========== #
        preview_label = tk.Label(
            self, text="預覽", bg=_BG, fg=_FG,
            font=("", 9, "bold"), anchor=tk.W,
        )
        preview_label.pack(fill=tk.X, padx=10, pady=(6, 0))

        self._preview_canvas = tk.Canvas(
            self, bg=_CANVAS_BG, highlightthickness=0, height=260,
        )
        self._preview_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=(2, 4))
        self._preview_canvas.bind("<Configure>", lambda _e: self._refresh_preview())

        # ========== 3. Results table ========== #
        table_frame = tk.LabelFrame(
            self,
            text=" 結果 ",
            bg=_BG,
            fg=_FG,
            font=("", 9, "bold"),
            padx=4,
            pady=2,
        )
        table_frame.pack(fill=tk.BOTH, padx=10, pady=(0, 4))

        columns = (
            "index", "area", "cx", "cy", "bbox",
            "circularity", "rectangularity", "convexity", "mean_gray",
        )
        self._tree = ttk.Treeview(
            table_frame, columns=columns, show="headings", height=7,
        )
        self._tree.heading("index", text="#")
        self._tree.heading("area", text="面積")
        self._tree.heading("cx", text="重心 X")
        self._tree.heading("cy", text="重心 Y")
        self._tree.heading("bbox", text="邊界框")
        self._tree.heading("circularity", text="圓形度")
        self._tree.heading("rectangularity", text="矩形度")
        self._tree.heading("convexity", text="凸性")
        self._tree.heading("mean_gray", text="平均灰度")

        self._tree.column("index", width=36, anchor=tk.CENTER)
        self._tree.column("area", width=64, anchor=tk.CENTER)
        self._tree.column("cx", width=72, anchor=tk.CENTER)
        self._tree.column("cy", width=72, anchor=tk.CENTER)
        self._tree.column("bbox", width=110, anchor=tk.CENTER)
        self._tree.column("circularity", width=78, anchor=tk.CENTER)
        self._tree.column("rectangularity", width=90, anchor=tk.CENTER)
        self._tree.column("convexity", width=72, anchor=tk.CENTER)
        self._tree.column("mean_gray", width=72, anchor=tk.CENTER)

        tree_scroll = ttk.Scrollbar(
            table_frame, orient=tk.VERTICAL, command=self._tree.yview,
        )
        self._tree.configure(yscrollcommand=tree_scroll.set)
        self._tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Clicking a row highlights the corresponding region
        self._tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        # ========== 4. Summary frame ========== #
        self._summary_var = tk.StringVar(
            value="區域數: --  面積平均: --  面積總和: --  最大: --  最小: --",
        )
        summary_frame = tk.Frame(self, bg=_BG)
        summary_frame.pack(fill=tk.X, padx=10, pady=(0, 4))

        tk.Label(
            summary_frame,
            textvariable=self._summary_var,
            bg=_BG,
            fg="#88cc88",
            font=("Consolas", 9),
            anchor=tk.W,
        ).pack(fill=tk.X)

        # ========== 5. Button frame ========== #
        btn_frame = tk.Frame(self, bg=_BG)
        btn_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        tk.Button(
            btn_frame,
            text="匯出 CSV",
            bg=_BG_MEDIUM,
            fg=_FG_WHITE,
            activebackground="#555555",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=14,
            pady=4,
            font=("", 9),
            command=self._export_csv,
        ).pack(side=tk.LEFT, padx=(0, 6))

        tk.Button(
            btn_frame,
            text="加入 Pipeline",
            bg=_ACCENT,
            fg="#ffffff",
            activebackground="#005a9e",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=14,
            pady=4,
            font=("", 10),
            command=self._add_to_pipeline,
        ).pack(side=tk.RIGHT, padx=(6, 0))

        tk.Button(
            btn_frame,
            text="關閉",
            bg=_BG_MEDIUM,
            fg=_FG_WHITE,
            activebackground="#555555",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=14,
            pady=4,
            font=("", 10),
            command=self._close,
        ).pack(side=tk.RIGHT)

    # ------------------------------------------------------------------ #
    #  Threshold mode toggle                                               #
    # ------------------------------------------------------------------ #

    def _on_threshold_mode_changed(self, _event: object = None) -> None:
        """Enable or disable the manual threshold entries depending on the
        selected mode."""
        mode = self._threshold_mode_var.get()
        if mode == "Otsu":
            self._thresh_min_entry.configure(state=tk.DISABLED)
            self._thresh_max_entry.configure(state=tk.DISABLED)
        else:
            self._thresh_min_entry.configure(state=tk.NORMAL)
            self._thresh_max_entry.configure(state=tk.NORMAL)

    # ------------------------------------------------------------------ #
    #  Pipeline execution                                                  #
    # ------------------------------------------------------------------ #

    def _run_analysis(self) -> None:
        """Execute the full blob analysis pipeline."""
        self._pipeline_steps.clear()
        self._highlight_index = None

        try:
            # Step 1 -- Grayscale conversion
            gray = self._gray.copy()
            self._pipeline_steps.append(("Grayscale", gray.copy(), None))

            # Step 2 -- Optional Gaussian blur
            blur_k = self._blur_kernel_var.get()
            if blur_k >= 3:
                # Ensure odd kernel size
                if blur_k % 2 == 0:
                    blur_k += 1
                gray = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
                self._pipeline_steps.append(
                    (f"GaussianBlur k={blur_k}", gray.copy(), None),
                )

            # Step 3 -- Threshold
            mode = self._threshold_mode_var.get()
            if mode == "Otsu":
                region = binary_threshold(gray, method="otsu")
                step_name = "BinaryThreshold (Otsu)"
            else:
                t_min = self._safe_int(self._thresh_min_var.get(), 0)
                t_max = self._safe_int(self._thresh_max_var.get(), 255)
                region = threshold(gray, t_min, t_max)
                step_name = f"Threshold [{t_min}, {t_max}]"
            self._pipeline_steps.append(
                (step_name, region.to_binary_mask(), region),
            )

            # Step 4 -- Optional morphological opening / closing
            morph_mode = self._morph_mode_var.get()
            if morph_mode != "無":
                mk = self._morph_kernel_var.get()
                if mk < 3:
                    mk = 3
                if mk % 2 == 0:
                    mk += 1
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (mk, mk),
                )
                mask = region.to_binary_mask()
                if morph_mode == "開運算":
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                    op_label = f"Opening k={mk}"
                else:
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                    op_label = f"Closing k={mk}"

                # Rebuild region from morphed mask
                num, labels = cv2.connectedComponents(mask, connectivity=8)
                labels = labels.astype(np.int32)
                props = compute_region_properties(labels, gray)
                region = Region(
                    labels=labels,
                    num_regions=num - 1,
                    properties=props,
                    source_image=gray,
                    source_shape=gray.shape[:2],
                )
                self._pipeline_steps.append(
                    (op_label, mask.copy(), region),
                )

            # Step 5 -- Connected components
            region = connection(region)
            self._pipeline_steps.append(
                ("Connection", region.to_binary_mask(), region),
            )

            # Step 6 -- Compute properties
            if not region.properties and region.num_regions > 0:
                props = compute_region_properties(
                    region.labels, region.source_image,
                )
                region = Region(
                    labels=region.labels,
                    num_regions=region.num_regions,
                    properties=props,
                    source_image=region.source_image,
                    source_shape=region.source_shape,
                )
            self._pipeline_steps.append(
                ("ComputeProperties", region.to_binary_mask(), region),
            )

            # Step 7 -- Area filter
            area_min = self._safe_int(self._area_min_var.get(), 0)
            area_max = self._safe_int(self._area_max_var.get(), 999_999_999)
            if area_min > 0 or area_max < 999_999_999:
                region = select_shape(region, "area", area_min, area_max)
                self._pipeline_steps.append(
                    (
                        f"SelectShape area[{area_min},{area_max}]",
                        region.to_binary_mask(),
                        region,
                    ),
                )

            # Step 8 -- Overlay visualization
            display = region_to_display_image(
                region,
                self._source_image,
                show_labels=True,
                show_bbox=True,
                show_cross=True,
                alpha=0.45,
            )
            self._pipeline_steps.append(("Overlay", display, region))

            # Store final results
            self._result_region = region
            self._display_image = display

            # Update UI
            self._refresh_preview()
            self._populate_table()
            self._update_summary()

        except Exception as exc:
            logger.exception("Blob analysis failed")
            messagebox.showerror(
                "分析錯誤",
                f"執行分析時發生錯誤:\n{exc}",
                parent=self,
            )

    # ------------------------------------------------------------------ #
    #  Preview                                                             #
    # ------------------------------------------------------------------ #

    def _refresh_preview(self) -> None:
        """Render the current display image onto the preview canvas."""
        if self._display_image is None and self._result_region is None:
            return

        # If a row is highlighted, regenerate overlay with highlight
        if (
            self._highlight_index is not None
            and self._result_region is not None
        ):
            image = region_to_display_image(
                self._result_region,
                self._source_image,
                show_labels=True,
                show_bbox=True,
                show_cross=True,
                alpha=0.45,
                highlight_indices=[self._highlight_index],
                highlight_color=(0, 255, 255),
            )
        elif self._display_image is not None:
            image = self._display_image
        else:
            return

        self._show_on_canvas(image)

    def _show_on_canvas(self, image: np.ndarray) -> None:
        """Scale *image* to fit the preview canvas and display it."""
        self._preview_canvas.update_idletasks()
        cw = max(self._preview_canvas.winfo_width(), 100)
        ch = max(self._preview_canvas.winfo_height(), 100)

        # BGR -> RGB for PIL
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        scale = min(cw / w, ch / h) * 0.96
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        pil_img = Image.fromarray(rgb).resize((new_w, new_h), Image.LANCZOS)
        self._preview_photo = ImageTk.PhotoImage(pil_img)

        self._preview_canvas.delete("all")
        self._preview_canvas.create_image(
            cw // 2, ch // 2,
            image=self._preview_photo,
            anchor=tk.CENTER,
        )

    # ------------------------------------------------------------------ #
    #  Results table                                                       #
    # ------------------------------------------------------------------ #

    def _populate_table(self) -> None:
        """Fill the Treeview with properties of all detected regions."""
        # Clear existing rows
        for item in self._tree.get_children():
            self._tree.delete(item)

        if self._result_region is None:
            return

        for p in self._result_region.properties:
            bx, by, bw, bh = p.bbox
            bbox_str = f"({bx},{by},{bw},{bh})"
            self._tree.insert("", tk.END, values=(
                p.index,
                p.area,
                f"{p.centroid[0]:.1f}",
                f"{p.centroid[1]:.1f}",
                bbox_str,
                f"{p.circularity:.4f}",
                f"{p.rectangularity:.4f}",
                f"{p.convexity:.4f}",
                f"{p.mean_value:.1f}",
            ))

    def _on_tree_select(self, _event: object = None) -> None:
        """Highlight the region corresponding to the selected table row."""
        selection = self._tree.selection()
        if not selection:
            self._highlight_index = None
            self._refresh_preview()
            return

        item = selection[0]
        values = self._tree.item(item, "values")
        if values:
            try:
                self._highlight_index = int(values[0])
            except (ValueError, IndexError):
                self._highlight_index = None
        self._refresh_preview()

    # ------------------------------------------------------------------ #
    #  Summary                                                             #
    # ------------------------------------------------------------------ #

    def _update_summary(self) -> None:
        """Update the summary label with aggregate statistics."""
        if self._result_region is None or not self._result_region.properties:
            self._summary_var.set(
                "區域數: 0  面積平均: --  面積總和: --  最大: --  最小: --",
            )
            return

        props = self._result_region.properties
        areas = [p.area for p in props]
        count = len(areas)
        total = sum(areas)
        avg = total / count if count > 0 else 0
        max_a = max(areas)
        min_a = min(areas)

        self._summary_var.set(
            f"區域數: {count}  "
            f"面積平均: {avg:.1f}  "
            f"面積總和: {total}  "
            f"最大: {max_a}  "
            f"最小: {min_a}"
        )

    # ------------------------------------------------------------------ #
    #  Export CSV                                                           #
    # ------------------------------------------------------------------ #

    def _export_csv(self) -> None:
        """Save the current results table to a CSV file."""
        if self._result_region is None or not self._result_region.properties:
            messagebox.showwarning(
                "無資料", "請先執行分析以產生結果。", parent=self,
            )
            return

        path = filedialog.asksaveasfilename(
            parent=self,
            title="匯出 CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*")],
        )
        if not path:
            return

        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "#", "Area", "Centroid_X", "Centroid_Y",
                    "BBox_X", "BBox_Y", "BBox_W", "BBox_H",
                    "Circularity", "Rectangularity", "Aspect_Ratio",
                    "Compactness", "Convexity", "Perimeter",
                    "Orientation", "Mean_Gray", "Min_Gray", "Max_Gray",
                ])
                for p in self._result_region.properties:
                    bx, by, bw, bh = p.bbox
                    writer.writerow([
                        p.index, p.area,
                        f"{p.centroid[0]:.2f}", f"{p.centroid[1]:.2f}",
                        bx, by, bw, bh,
                        f"{p.circularity:.4f}",
                        f"{p.rectangularity:.4f}",
                        f"{p.aspect_ratio:.4f}",
                        f"{p.compactness:.4f}",
                        f"{p.convexity:.4f}",
                        f"{p.perimeter:.2f}",
                        f"{p.orientation:.2f}",
                        f"{p.mean_value:.2f}",
                        f"{p.min_value:.2f}",
                        f"{p.max_value:.2f}",
                    ])
            logger.info("Exported blob analysis results to %s", path)
            messagebox.showinfo(
                "匯出完成",
                f"已匯出 {len(self._result_region.properties)} 筆資料至:\n{path}",
                parent=self,
            )
        except OSError as exc:
            logger.exception("CSV export failed")
            messagebox.showerror(
                "匯出失敗", f"無法寫入檔案:\n{exc}", parent=self,
            )

    # ------------------------------------------------------------------ #
    #  Add to Pipeline                                                     #
    # ------------------------------------------------------------------ #

    def _add_to_pipeline(self) -> None:
        """Invoke the on_accept callback with the accumulated pipeline
        steps and close the dialog."""
        if not self._pipeline_steps:
            messagebox.showwarning(
                "無結果", "請先執行分析。", parent=self,
            )
            return

        if self._on_accept is not None:
            self._on_accept(self._pipeline_steps)

        self.grab_release()
        self.destroy()

    # ------------------------------------------------------------------ #
    #  Close                                                               #
    # ------------------------------------------------------------------ #

    def _close(self) -> None:
        """Close the dialog without accepting."""
        self.grab_release()
        self.destroy()

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        """Convert an image to uint8 grayscale."""
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

    @staticmethod
    def _safe_int(value: str, default: int) -> int:
        """Parse *value* as an integer, returning *default* on failure."""
        try:
            return int(value)
        except (ValueError, TypeError):
            return default
