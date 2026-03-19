"""Dialog-opening mixin for InspectorApp.

Covers: pixel inspector, threshold dialogs, binarize, adaptive threshold,
subtract dialog, contour detection, dynamic threshold, script editor,
and tool dialogs (shape matching, metrology, ROI, advanced models, etc.).
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from dl_anomaly.gui.inspector_app import InspectorApp


class DialogMixin:
    """All dialog-opening methods for InspectorApp."""

    # ==================================================================
    # Pixel Inspector
    # ==================================================================

    def _toggle_pixel_inspector(self: "InspectorApp", state: bool = None) -> None:
        """Toggle the pixel inspector window."""
        if self._pixel_inspector is not None and self._pixel_inspector.winfo_exists():
            self._pixel_inspector.destroy()
            self._pixel_inspector = None
        else:
            from dl_anomaly.gui.pixel_inspector import PixelInspector
            self._pixel_inspector = PixelInspector(self)

    # ==================================================================
    # Threshold dialog
    # ==================================================================

    def _open_threshold_dialog(self: "InspectorApp") -> None:
        """Open threshold segmentation dialog."""
        img = self._get_current_image()
        if img is None:
            messagebox.showwarning("\u8b66\u544a", "\u8acb\u5148\u8f09\u5165\u5716\u7247\u3002")
            return

        from dl_anomaly.gui.threshold_dialog import ThresholdDialog

        def on_accept(region, display_image, name):
            self._current_region = region
            import re
            m = re.search(r"\[(\d+),\s*(\d+)\]", name)
            if m:
                min_val, max_val = int(m.group(1)), int(m.group(2))
                op_meta = {"category": "threshold", "op": "manual",
                           "params": {"min_val": min_val, "max_val": max_val}}
            else:
                op_meta = {"category": "threshold", "op": "manual", "params": {}}
            self._pipeline_panel.add_step(name, display_image, region=region, op_meta=op_meta)
            self.set_status(f"\u95be\u503c\u5206\u5272\u5b8c\u6210: {region.num_regions} \u500b\u5340\u57df")

        ThresholdDialog(self, img, on_accept=on_accept)

    # ==================================================================
    # Binarize
    # ==================================================================

    def _open_binarize_dialog(self: "InspectorApp"):
        """Open binarization dialog -- produces a binary IMAGE (not Region)."""
        img = self._get_current_image()
        if img is None:
            messagebox.showwarning("\u8b66\u544a", "\u8acb\u5148\u8f09\u5165\u5716\u7247\u3002")
            return

        methods = ["THRESH_BINARY", "THRESH_BINARY_INV", "THRESH_OTSU", "THRESH_TRIANGLE"]
        self._open_param_dialog("\u4e8c\u503c\u5316", [
            {"label": "\u95be\u503c (0-255):", "key": "thresh", "type": "int", "default": 128},
            {"label": "\u65b9\u6cd5:", "key": "method", "type": "combo", "default": "THRESH_BINARY", "values": methods},
        ], lambda p: self._apply_binarize(img, p))

    def _apply_binarize(self: "InspectorApp", img, p):
        method_map = {
            "THRESH_BINARY": cv2.THRESH_BINARY,
            "THRESH_BINARY_INV": cv2.THRESH_BINARY_INV,
            "THRESH_OTSU": cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            "THRESH_TRIANGLE": cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE,
        }
        thresh_val = p["thresh"]
        cv_method = method_map.get(p["method"], cv2.THRESH_BINARY)

        def _compute():
            from dl_anomaly.core.vision_ops import _ensure_gray
            gray = _ensure_gray(img)
            _, binary = cv2.threshold(gray, thresh_val, 255, cv_method)
            return binary

        def _done(binary):
            name = f"\u4e8c\u503c\u5316 t={thresh_val} {p['method']}"
            self._add_pipeline_step(name, binary)
            self.set_status(f"\u5b8c\u6210: {name}")

        self._run_in_bg(_compute, on_done=_done, status_msg="\u4e8c\u503c\u5316\u8655\u7406\u4e2d...")

    # ==================================================================
    # Adaptive Threshold
    # ==================================================================

    def _open_adaptive_threshold_dialog(self: "InspectorApp"):
        """Open adaptive threshold dialog with parameter control."""
        img = self._get_current_image()
        if img is None:
            messagebox.showwarning("\u8b66\u544a", "\u8acb\u5148\u8f09\u5165\u5716\u7247\u3002")
            return

        self._open_param_dialog("\u81ea\u9069\u61c9\u95be\u503c", [
            {"label": "\u5340\u584a\u5927\u5c0f (block_size):", "key": "block_size", "type": "int", "default": 15},
            {"label": "\u5e38\u6578 C:", "key": "c_value", "type": "int", "default": 5},
            {"label": "\u65b9\u6cd5:", "key": "method", "type": "combo",
             "default": "GAUSSIAN_C", "values": ["MEAN_C", "GAUSSIAN_C"]},
        ], lambda p: self._apply_adaptive_threshold(img, p))

    def _apply_adaptive_threshold(self: "InspectorApp", img, p):
        block_size = p["block_size"]
        c_value = p["c_value"]
        adapt_method = p["method"]

        def _compute():
            from dl_anomaly.core.region_ops import binary_threshold
            cv_method = (cv2.ADAPTIVE_THRESH_GAUSSIAN_C if adapt_method == "GAUSSIAN_C"
                         else cv2.ADAPTIVE_THRESH_MEAN_C)
            bs = block_size
            if bs % 2 == 0:
                bs += 1
            if bs < 3:
                bs = 3
            region = binary_threshold(img, method="adaptive", block_size=bs, c_value=c_value)
            return region, bs

        def _done(result):
            region, bs = result
            self._current_region = region
            name = f"\u81ea\u9069\u61c9\u95be\u503c bs={bs} C={c_value} {adapt_method}"
            self._pipeline_panel.add_step(name, region.to_binary_mask(), region=region)
            self.set_status(f"{name}: {region.num_regions} \u500b\u5340\u57df")

        self._run_in_bg(_compute, on_done=_done, status_msg="\u81ea\u9069\u61c9\u95be\u503c\u5206\u5272\u4e2d...")

    # ==================================================================
    # Subtract dialog
    # ==================================================================

    def _open_subtract_dialog(self: "InspectorApp"):
        """Open image subtraction dialog -- select two pipeline steps."""
        steps = self._pipeline_panel.get_all_steps()
        if len(steps) < 2:
            messagebox.showwarning("\u8b66\u544a", "\u9700\u8981\u81f3\u5c11\u5169\u500b\u7ba1\u7dda\u6b65\u9a5f\u624d\u80fd\u9032\u884c\u5716\u50cf\u76f8\u6e1b\u3002")
            return

        step_names = [f"[{i}] {s.name}" for i, s in enumerate(steps)]

        dlg = tk.Toplevel(self)
        dlg.title("\u5716\u50cf\u76f8\u6e1b")
        dlg.configure(bg="#2b2b2b")
        dlg.resizable(False, False)
        dlg.grab_set()

        tk.Label(dlg, text="\u5716\u50cf\u76f8\u6e1b (A - B) * mult + add",
                 bg="#2b2b2b", fg="#e0e0e0", font=("", 10, "bold")).grid(
            row=0, column=0, columnspan=2, pady=(10, 6), padx=10)

        tk.Label(dlg, text="\u5716\u50cf A:", bg="#2b2b2b", fg="#e0e0e0").grid(
            row=1, column=0, sticky="e", padx=(10, 4), pady=4)
        a_var = tk.StringVar(value=step_names[0] if step_names else "")
        ttk.Combobox(dlg, textvariable=a_var, width=30,
                     values=step_names, state="readonly").grid(
            row=1, column=1, padx=(0, 10), pady=4)

        tk.Label(dlg, text="\u5716\u50cf B:", bg="#2b2b2b", fg="#e0e0e0").grid(
            row=2, column=0, sticky="e", padx=(10, 4), pady=4)
        b_var = tk.StringVar(value=step_names[-1] if len(step_names) > 1 else step_names[0])
        ttk.Combobox(dlg, textvariable=b_var, width=30,
                     values=step_names, state="readonly").grid(
            row=2, column=1, padx=(0, 10), pady=4)

        tk.Label(dlg, text="\u4e58\u6578 (mult):", bg="#2b2b2b", fg="#e0e0e0").grid(
            row=3, column=0, sticky="e", padx=(10, 4), pady=4)
        mult_var = tk.StringVar(value="1.0")
        tk.Entry(dlg, textvariable=mult_var, width=10,
                 bg="#3c3c3c", fg="#e0e0e0", insertbackground="#e0e0e0").grid(
            row=3, column=1, sticky="w", padx=(0, 10), pady=4)

        tk.Label(dlg, text="\u504f\u79fb (add):", bg="#2b2b2b", fg="#e0e0e0").grid(
            row=4, column=0, sticky="e", padx=(10, 4), pady=4)
        add_var = tk.StringVar(value="0.0")
        tk.Entry(dlg, textvariable=add_var, width=10,
                 bg="#3c3c3c", fg="#e0e0e0", insertbackground="#e0e0e0").grid(
            row=4, column=1, sticky="w", padx=(0, 10), pady=4)

        def _apply():
            try:
                mult = float(mult_var.get())
                add = float(add_var.get())
            except ValueError:
                messagebox.showwarning("\u8b66\u544a", "\u8acb\u8f38\u5165\u6709\u6548\u7684\u6578\u503c\u3002", parent=dlg)
                return
            a_sel = a_var.get()
            b_sel = b_var.get()
            a_idx = step_names.index(a_sel) if a_sel in step_names else 0
            b_idx = step_names.index(b_sel) if b_sel in step_names else 0
            img_a = steps[a_idx].array.copy()
            img_b = steps[b_idx].array.copy()
            dlg.destroy()

            def _compute():
                from dl_anomaly.core import vision_ops as hops
                return hops.sub_image(img_a, img_b, mult, add)

            def _done(result):
                name = f"\u5716\u50cf\u76f8\u6e1b [{a_idx}]-[{b_idx}] m={mult} a={add}"
                self._add_pipeline_step(name, result)
                self.set_status(f"\u5b8c\u6210: {name}")

            self._run_in_bg(_compute, on_done=_done, status_msg="\u5716\u50cf\u76f8\u6e1b\u4e2d...")

        btn_frame = tk.Frame(dlg, bg="#2b2b2b")
        btn_frame.grid(row=5, column=0, columnspan=2, pady=(6, 10))
        tk.Button(btn_frame, text="\u57f7\u884c", command=_apply, width=8,
                  bg="#3a3a5c", fg="#e0e0e0", activebackground="#4a4a6c",
                  activeforeground="#ffffff").pack(side="left", padx=4)
        tk.Button(btn_frame, text="\u53d6\u6d88", command=dlg.destroy, width=8,
                  bg="#3a3a5c", fg="#e0e0e0", activebackground="#4a4a6c",
                  activeforeground="#ffffff").pack(side="left", padx=4)

        dlg.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() - dlg.winfo_width()) // 2
        y = self.winfo_y() + (self.winfo_height() - dlg.winfo_height()) // 2
        dlg.geometry(f"+{x}+{y}")

    # ==================================================================
    # Contour detection
    # ==================================================================

    def _open_contour_detection_dialog(self: "InspectorApp"):
        """Open contour detection dialog with area filter."""
        img = self._get_current_image()
        if img is None:
            messagebox.showwarning("\u8b66\u544a", "\u8acb\u5148\u8f09\u5165\u5716\u7247\u3002")
            return

        modes = ["RETR_LIST", "RETR_EXTERNAL"]
        self._open_param_dialog("\u8f2a\u5ed3\u6aa2\u6e2c", [
            {"label": "\u6700\u5c0f\u9762\u7a4d:", "key": "min_area", "type": "int", "default": 0},
            {"label": "\u6700\u5927\u9762\u7a4d:", "key": "max_area", "type": "int", "default": 10000},
            {"label": "\u6a21\u5f0f:", "key": "mode", "type": "combo", "default": "RETR_LIST", "values": modes},
        ], lambda p: self._apply_contour_detection(img, p))

    def _apply_contour_detection(self: "InspectorApp", img, p):
        min_area = p["min_area"]
        max_area = p["max_area"]
        mode_str = p["mode"]

        def _compute():
            from dl_anomaly.core.vision_ops import _ensure_gray
            mode_map = {"RETR_LIST": cv2.RETR_LIST, "RETR_EXTERNAL": cv2.RETR_EXTERNAL}
            cv_mode = mode_map.get(mode_str, cv2.RETR_LIST)

            gray = _ensure_gray(img)
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv_mode, cv2.CHAIN_APPROX_SIMPLE)

            filtered = [c for c in contours if min_area <= cv2.contourArea(c) <= max_area]

            vis = img.copy()
            if vis.ndim == 2:
                vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(vis, filtered, -1, (0, 255, 0), 2)
            return vis, len(filtered)

        def _done(result):
            vis, count = result
            name = f"\u8f2a\u5ed3\u6aa2\u6e2c area=[{min_area},{max_area}] ({count})"
            self._add_pipeline_step(name, vis)
            self.set_status(f"\u8f2a\u5ed3\u6aa2\u6e2c\u5b8c\u6210: \u627e\u5230 {count} \u500b\u8f2a\u5ed3")

        self._run_in_bg(_compute, on_done=_done, status_msg="\u8f2a\u5ed3\u6aa2\u6e2c\u4e2d...")

    # ==================================================================
    # Dynamic threshold dialog
    # ==================================================================

    def _open_dyn_threshold_dialog(self: "InspectorApp") -> None:
        """Open dialog for dynamic threshold segmentation (vision style)."""
        img = self._get_current_image()
        if img is None:
            messagebox.showwarning("\u8b66\u544a", "\u8acb\u5148\u8f09\u5165\u5716\u7247\u3002")
            return

        dlg = tk.Toplevel(self)
        dlg.title("\u52d5\u614b\u95be\u503c\u5206\u5272")
        dlg.configure(bg="#2b2b2b")
        dlg.resizable(False, False)
        dlg.grab_set()

        tk.Label(dlg, text="\u52d5\u614b\u95be\u503c\u5206\u5272 (dyn_threshold)",
                 bg="#2b2b2b", fg="#e0e0e0", font=("", 10, "bold")).grid(
            row=0, column=0, columnspan=2, pady=(10, 6), padx=10)

        tk.Label(dlg, text="\u6838\u5927\u5c0f (ksize):", bg="#2b2b2b", fg="#e0e0e0").grid(
            row=1, column=0, sticky="e", padx=(10, 4), pady=4)
        ksize_var = tk.StringVar(value="7")
        tk.Entry(dlg, textvariable=ksize_var, width=8,
                 bg="#3c3c3c", fg="#e0e0e0", insertbackground="#e0e0e0").grid(
            row=1, column=1, padx=(0, 10), pady=4)

        tk.Label(dlg, text="\u6838\u5f62\u72c0:", bg="#2b2b2b", fg="#e0e0e0").grid(
            row=2, column=0, sticky="e", padx=(10, 4), pady=4)
        shape_var = tk.StringVar(value="octagon")
        shape_combo = ttk.Combobox(dlg, textvariable=shape_var, width=10,
                                   values=["octagon", "rectangle", "ellipse", "cross"],
                                   state="readonly")
        shape_combo.grid(row=2, column=1, padx=(0, 10), pady=4)

        tk.Label(dlg, text="\u504f\u79fb\u91cf (offset):", bg="#2b2b2b", fg="#e0e0e0").grid(
            row=3, column=0, sticky="e", padx=(10, 4), pady=4)
        offset_var = tk.StringVar(value="75")
        tk.Entry(dlg, textvariable=offset_var, width=8,
                 bg="#3c3c3c", fg="#e0e0e0", insertbackground="#e0e0e0").grid(
            row=3, column=1, padx=(0, 10), pady=4)

        tk.Label(dlg, text="\u6a21\u5f0f:", bg="#2b2b2b", fg="#e0e0e0").grid(
            row=4, column=0, sticky="e", padx=(10, 4), pady=4)
        mode_var = tk.StringVar(value="not_equal")
        mode_combo = ttk.Combobox(dlg, textvariable=mode_var, width=10,
                                  values=["not_equal", "light", "dark", "equal"],
                                  state="readonly")
        mode_combo.grid(row=4, column=1, padx=(0, 10), pady=4)

        def _apply():
            try:
                ks = int(ksize_var.get())
                ofs = float(offset_var.get())
            except ValueError:
                messagebox.showwarning("\u8b66\u544a", "\u8acb\u8f38\u5165\u6709\u6548\u7684\u6578\u503c\u3002", parent=dlg)
                return
            sh = shape_var.get()
            md = mode_var.get()
            dlg.destroy()

            src = img

            def _compute():
                from dl_anomaly.core import vision_ops as hops
                from dl_anomaly.core.region import Region
                from dl_anomaly.core.region_ops import compute_region_properties

                gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY) if src.ndim == 3 else src
                img_opening = hops.gray_opening_shape(gray, ks, ks, sh)
                img_closing = hops.gray_closing_shape(gray, ks, ks, sh)
                mask = hops.dyn_threshold(img_opening, img_closing, ofs, md)

                num, labels = cv2.connectedComponents(mask, connectivity=8)
                labels = labels.astype(np.int32)
                props = compute_region_properties(labels, gray)
                return Region(
                    labels=labels, num_regions=num - 1, properties=props,
                    source_image=gray, source_shape=gray.shape[:2],
                )

            def _done(region):
                self._current_region = region
                name = f"\u52d5\u614b\u95be\u503c k={ks} {sh} ofs={ofs} {md}"
                self._pipeline_panel.add_step(name, region.to_binary_mask(), region=region)
                self.set_status(f"{name}: {region.num_regions} \u500b\u5340\u57df")

            self._run_in_bg(_compute, on_done=_done, status_msg="\u52d5\u614b\u95be\u503c\u5206\u5272\u4e2d...")

        btn_frame = tk.Frame(dlg, bg="#2b2b2b")
        btn_frame.grid(row=5, column=0, columnspan=2, pady=(6, 10))
        tk.Button(btn_frame, text="\u57f7\u884c", command=_apply, width=8,
                  bg="#3a3a5c", fg="#e0e0e0", activebackground="#4a4a6c",
                  activeforeground="#ffffff").pack(side="left", padx=4)
        tk.Button(btn_frame, text="\u53d6\u6d88", command=dlg.destroy, width=8,
                  bg="#3a3a5c", fg="#e0e0e0", activebackground="#4a4a6c",
                  activeforeground="#ffffff").pack(side="left", padx=4)

        dlg.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() - dlg.winfo_width()) // 2
        y = self.winfo_y() + (self.winfo_height() - dlg.winfo_height()) // 2
        dlg.geometry(f"+{x}+{y}")

    # ==================================================================
    # Script Editor
    # ==================================================================

    def _toggle_script_editor(self: "InspectorApp", state: bool = None) -> None:
        """Toggle script editor panel."""
        if self._script_editor_visible:
            if self._script_editor is not None:
                self._script_editor.pack_forget()
            self._script_editor_visible = False
            self.set_status("\u8173\u672c\u7de8\u8f2f\u5668\u5df2\u95dc\u9589")
        else:
            try:
                from dl_anomaly.gui.script_editor import ScriptEditor
                if self._script_editor is None:
                    self._script_editor = ScriptEditor(self, app=self)
                self._script_editor.pack(fill=tk.BOTH, expand=False, side=tk.BOTTOM)
                self._script_editor.pack_configure(fill=tk.BOTH, expand=False)
            except Exception as exc:
                self._show_error("\u8173\u672c\u7de8\u8f2f\u5668\u8f09\u5165\u5931\u6557", exc)
                return
            self._script_editor_visible = True
            self.set_status("\u8173\u672c\u7de8\u8f2f\u5668\u5df2\u958b\u555f (F9 \u57f7\u884c)")

    def _run_script(self: "InspectorApp") -> None:
        """Run script from the script editor."""
        if self._script_editor is not None and self._script_editor_visible:
            self._script_editor.run_script()

    # ==================================================================
    # Phase 1 Tools: Shape Matching, Metrology, ROI Manager
    # ==================================================================

    def _open_shape_matching(self: "InspectorApp") -> None:
        """Open Shape-Based Matching dialog."""
        from dl_anomaly.gui.shape_matching_dialog import ShapeMatchingDialog

        ShapeMatchingDialog(
            self,
            get_current_image=self._get_current_image,
            add_pipeline_step=self._add_pipeline_step,
            set_status=self.set_status,
        )

    def _open_metrology(self: "InspectorApp") -> None:
        """Open Metrology / Sub-pixel Measurement dialog."""
        from dl_anomaly.gui.metrology_dialog import MetrologyDialog

        MetrologyDialog(
            self,
            get_current_image=self._get_current_image,
            add_pipeline_step=self._add_pipeline_step,
            set_status=self.set_status,
        )

    def _open_roi_manager(self: "InspectorApp") -> None:
        """Open ROI Manager dialog."""
        from dl_anomaly.gui.roi_dialog import ROIManagerDialog

        ROIManagerDialog(
            self,
            get_current_image=self._get_current_image,
            add_pipeline_step=self._add_pipeline_step,
            set_status=self.set_status,
            viewer=self._viewer,
        )

    def _open_advanced_models(self: "InspectorApp") -> None:
        """Open PatchCore / ONNX advanced models dialog."""
        from dl_anomaly.gui.advanced_models_dialog import AdvancedModelsDialog

        AdvancedModelsDialog(
            self,
            config=self.config,
            get_current_image=self._get_current_image,
            add_pipeline_step=self._add_pipeline_step,
            set_status=self.set_status,
        )

    def _open_inspection_tools(self: "InspectorApp") -> None:
        """Open FFT / Color / OCR / Barcode inspection tools dialog."""
        from dl_anomaly.gui.inspection_tools_dialog import InspectionToolsDialog

        InspectionToolsDialog(
            self,
            get_current_image=self._get_current_image,
            add_pipeline_step=self._add_pipeline_step,
            set_status=self.set_status,
        )

    def _open_engineering_tools(self: "InspectorApp") -> None:
        """Open Calibration / Pipeline / SPC / Stitching engineering tools dialog."""
        from dl_anomaly.gui.engineering_tools_dialog import EngineeringToolsDialog

        EngineeringToolsDialog(
            self,
            get_current_image=self._get_current_image,
            add_pipeline_step=self._add_pipeline_step,
            set_status=self.set_status,
        )

    def _open_mvp_tools(self: "InspectorApp") -> None:
        """Open MVP tools dialog (Camera / Inspection Flow / Report)."""
        try:
            from dl_anomaly.gui.mvp_tools_dialog import MVPToolsDialog
        except ImportError as exc:
            from tkinter import messagebox as _mb
            _mb.showerror("\u532f\u5165\u932f\u8aa4", f"\u7121\u6cd5\u8f09\u5165 MVP \u5de5\u5177\u6a21\u7d44\uff1a\n{exc}")
            return
        MVPToolsDialog(
            self,
            get_current_image=self._get_current_image,
            add_pipeline_step=self._add_pipeline_step,
            set_status=self.set_status,
        )

    # ==================================================================
    # Pipeline Model (save / load / manage)
    # ==================================================================

    def _cmd_save_pipeline_model(self: "InspectorApp") -> None:
        """Save current pipeline + flow as a .cpmodel file."""
        from dl_anomaly.gui.pipeline_model_dialog import save_pipeline_model_dialog
        save_pipeline_model_dialog(
            self,
            pipeline_panel=self._pipeline_panel,
            set_status=self.set_status,
        )

    def _cmd_load_pipeline_model(self: "InspectorApp") -> None:
        """Load and execute a .cpmodel pipeline model."""
        from dl_anomaly.gui.pipeline_model_dialog import load_pipeline_model_dialog
        load_pipeline_model_dialog(
            self,
            get_current_image=self._get_current_image,
            add_pipeline_step=self._add_pipeline_step,
            set_status=self.set_status,
        )

    def _open_pipeline_model_manager(self: "InspectorApp") -> None:
        """Open the pipeline model registry manager dialog."""
        from dl_anomaly.gui.pipeline_model_dialog import PipelineModelManagerDialog
        PipelineModelManagerDialog(
            self,
            get_current_image=self._get_current_image,
            add_pipeline_step=self._add_pipeline_step,
            set_status=self.set_status,
        )

    # ==================================================================
    # Auto-Tune
    # ==================================================================

    def _open_auto_tune(self: "InspectorApp") -> None:
        """Open automatic threshold calibration dialog."""
        from dl_anomaly.gui.auto_tune_dialog import AutoTuneDialog
        AutoTuneDialog(self, set_status=self.set_status)
