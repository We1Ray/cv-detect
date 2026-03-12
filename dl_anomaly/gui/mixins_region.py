"""Region operations mixin for HalconApp.

Covers: threshold operations (ensure_region, otsu, adaptive),
connection, fill_up, shape_trans, morphology, region filter,
select_gray, sort, set_op, blob analysis,
domain operations (reduce, crop), region highlight/remove.
"""

from __future__ import annotations

import copy
import tkinter as tk
from tkinter import messagebox, ttk
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from dl_anomaly.gui.halcon_app import HalconApp


class RegionMixin:
    """All region / morphology operations for HalconApp."""

    # ==================================================================
    # Region helpers
    # ==================================================================

    def _ensure_region(self: "HalconApp") -> bool:
        """Ensure ``_current_region`` exists.

        If no region has been created yet, automatically run Otsu threshold
        on the current image to produce a meaningful initial region.
        """
        if self._current_region is not None:
            return True
        img = self._get_current_image()
        if img is None:
            messagebox.showwarning("\u8b66\u544a", "\u8acb\u5148\u8f09\u5165\u5716\u7247\u3002")
            return False

        from dl_anomaly.core.region_ops import binary_threshold
        self._current_region = binary_threshold(img, method="otsu")
        self._pipeline_panel.add_step(
            "Otsu \u95be\u503c (\u81ea\u52d5)",
            self._current_region.to_binary_mask(),
            region=self._current_region,
            op_meta={"category": "threshold", "op": "otsu", "params": {}},
        )
        return True

    # ==================================================================
    # Threshold operations
    # ==================================================================

    def _auto_threshold_otsu(self: "HalconApp") -> None:
        """Auto Otsu threshold segmentation."""
        img = self._get_current_image()
        if img is None:
            messagebox.showwarning("\u8b66\u544a", "\u8acb\u5148\u8f09\u5165\u5716\u7247\u3002")
            return

        def _compute():
            from dl_anomaly.core.region_ops import binary_threshold
            region = binary_threshold(img, method="otsu")
            return region

        def _done(region):
            self._current_region = region
            op_meta = {"category": "threshold", "op": "otsu", "params": {}}
            self._pipeline_panel.add_step(
                "Otsu \u95be\u503c", region.to_binary_mask(), region=region, op_meta=op_meta)
            self.set_status(f"Otsu \u95be\u503c: {region.num_regions} \u500b\u5340\u57df")

        self._run_in_bg(_compute, on_done=_done, status_msg="Otsu \u95be\u503c\u5206\u5272\u4e2d...")

    def _auto_threshold_adaptive(self: "HalconApp") -> None:
        """Auto adaptive threshold segmentation."""
        img = self._get_current_image()
        if img is None:
            messagebox.showwarning("\u8b66\u544a", "\u8acb\u5148\u8f09\u5165\u5716\u7247\u3002")
            return

        def _compute():
            from dl_anomaly.core.region_ops import binary_threshold
            region = binary_threshold(img, method="adaptive")
            return region

        def _done(region):
            self._current_region = region
            op_meta = {"category": "threshold", "op": "adaptive", "params": {}}
            self._pipeline_panel.add_step(
                "\u81ea\u9069\u61c9\u95be\u503c", region.to_binary_mask(), region=region, op_meta=op_meta)
            self.set_status(f"\u81ea\u9069\u61c9\u95be\u503c: {region.num_regions} \u500b\u5340\u57df")

        self._run_in_bg(_compute, on_done=_done, status_msg="\u81ea\u9069\u61c9\u95be\u503c\u5206\u5272\u4e2d...")

    # ==================================================================
    # Connection
    # ==================================================================

    def _region_connection(self: "HalconApp") -> None:
        """Connection operation."""
        if not self._ensure_region():
            return

        cur_region = self._current_region
        img = self._get_current_image()

        def _compute():
            from dl_anomaly.core.region_ops import connection
            return connection(cur_region)

        def _done(region):
            self._current_region = region
            n = max(region.num_regions, 1)
            vis = np.zeros(region.labels.shape, dtype=np.uint8)
            for i in range(1, n + 1):
                gray_val = int(55 + 200 * i / n)
                vis[region.labels == i] = gray_val
            op_meta = {"category": "region", "op": "connection", "params": {}}
            self._pipeline_panel.add_step(
                f"Connection ({region.num_regions})", vis, region=region, op_meta=op_meta)
            self.set_status(f"\u6253\u6563: {region.num_regions} \u500b\u7368\u7acb\u5340\u57df")

        self._run_in_bg(_compute, on_done=_done, status_msg="Connection...")

    # ==================================================================
    # Fill Up
    # ==================================================================

    def _region_fill_up(self: "HalconApp") -> None:
        """Fill region holes with parameter dialog."""
        if not self._ensure_region():
            return

        dlg = tk.Toplevel(self)
        dlg.title("\u586b\u5145 (Fill Up)")
        dlg.configure(bg="#2b2b2b")
        dlg.resizable(False, False)
        dlg.grab_set()

        tk.Label(dlg, text="\u586b\u5145\u5340\u57df\u5167\u90e8\u7a7a\u6d1e",
                 bg="#2b2b2b", fg="#e0e0e0", font=("", 10, "bold")).grid(
            row=0, column=0, columnspan=2, pady=(10, 6), padx=10)

        tk.Label(dlg, text="\u6700\u5c0f\u7a7a\u6d1e\u9762\u7a4d:", bg="#2b2b2b", fg="#e0e0e0").grid(
            row=1, column=0, sticky="e", padx=(10, 4), pady=4)
        min_area_var = tk.StringVar(value="0")
        tk.Entry(dlg, textvariable=min_area_var, width=10,
                 bg="#3c3c3c", fg="#e0e0e0", insertbackground="#e0e0e0").grid(
            row=1, column=1, padx=(0, 10), pady=4)
        tk.Label(dlg, text="(0 = \u586b\u5145\u6240\u6709\u7a7a\u6d1e)", bg="#2b2b2b", fg="#999999",
                 font=("", 8)).grid(row=2, column=0, columnspan=2, padx=10)

        def _apply():
            try:
                min_area = int(min_area_var.get())
            except ValueError:
                messagebox.showwarning("\u8b66\u544a", "\u8acb\u8f38\u5165\u6709\u6548\u7684\u6578\u503c\u3002", parent=dlg)
                return
            dlg.destroy()
            cur_region = self._current_region

            def _compute():
                from dl_anomaly.core.region_ops import compute_region_properties
                from dl_anomaly.core.region import Region

                labels_in = cur_region.labels
                filled_labels = np.zeros_like(labels_in)
                new_id = 1

                for lbl in range(1, cur_region.num_regions + 1):
                    comp = ((labels_in == lbl) * 255).astype(np.uint8)
                    if not np.any(comp):
                        continue

                    h, w = comp.shape
                    border = np.full((h + 2, w + 2), 255, dtype=np.uint8)
                    border[1:-1, 1:-1] = cv2.bitwise_not(comp)
                    cv2.floodFill(border, None, (0, 0), 0)
                    holes = border[1:-1, 1:-1]

                    if min_area > 0:
                        hole_n, hole_labels = cv2.connectedComponents(holes, connectivity=8)
                        for hi in range(1, hole_n):
                            hole_mask = (hole_labels == hi)
                            if hole_mask.sum() >= min_area:
                                comp[hole_mask] = 255
                    else:
                        comp = comp | holes

                    filled_labels[comp > 0] = new_id
                    new_id += 1

                num, labels_out = cv2.connectedComponents(
                    (filled_labels > 0).astype(np.uint8) * 255, connectivity=8)
                labels_out = labels_out.astype(np.int32)
                props = compute_region_properties(labels_out, cur_region.source_image)
                return Region(labels=labels_out, num_regions=num - 1, properties=props,
                              source_image=cur_region.source_image,
                              source_shape=cur_region.source_shape)

            def _done(region):
                self._current_region = region
                name = f"Fill Up (min={min_area})" if min_area > 0 else "Fill Up"
                op_meta = {"category": "region", "op": "fill_up",
                           "params": {"min_area": min_area}}
                self._pipeline_panel.add_step(
                    name, region.to_binary_mask(), region=region, op_meta=op_meta)
                self.set_status(f"\u586b\u5145\u5b8c\u6210: {region.num_regions} \u500b\u5340\u57df")

            self._run_in_bg(_compute, on_done=_done, status_msg="Fill Up...")

        btn_frame = tk.Frame(dlg, bg="#2b2b2b")
        btn_frame.grid(row=3, column=0, columnspan=2, pady=(6, 10))
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
    # Shape transformation
    # ==================================================================

    def _region_shape_trans(self: "HalconApp", shape_type: str) -> None:
        """Shape transformation."""
        if not self._ensure_region():
            return

        cur_region = self._current_region
        img = self._get_current_image()

        def _compute():
            from dl_anomaly.core.region_ops import compute_region_properties
            from dl_anomaly.core.region import Region
            mask = cur_region.to_binary_mask()
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            result = np.zeros_like(mask)
            for cnt in contours:
                if shape_type == "convex":
                    hull = cv2.convexHull(cnt)
                    cv2.drawContours(result, [hull], -1, 255, -1)
                elif shape_type == "rectangle":
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(result, (x, y), (x + w, y + h), 255, -1)
                elif shape_type == "circle":
                    (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                    cv2.circle(result, (int(cx), int(cy)), int(radius), 255, -1)
                elif shape_type == "ellipse":
                    if len(cnt) >= 5:
                        ellipse = cv2.fitEllipse(cnt)
                        cv2.ellipse(result, ellipse, 255, -1)
            num, labels = cv2.connectedComponents(result, connectivity=8)
            labels = labels.astype(np.int32)
            props = compute_region_properties(labels, cur_region.source_image)
            return Region(labels=labels, num_regions=num - 1, properties=props,
                          source_image=cur_region.source_image,
                          source_shape=cur_region.source_shape)

        def _done(region):
            self._current_region = region
            self._pipeline_panel.add_step(f"Shape Trans ({shape_type})", region.to_binary_mask(), region=region)

        self._run_in_bg(_compute, on_done=_done, status_msg=f"Shape Trans ({shape_type})...")

    # ==================================================================
    # Morphology
    # ==================================================================

    def _region_morphology(self: "HalconApp", op: str) -> None:
        """Region morphology with parameter dialog."""
        if not self._ensure_region():
            return

        op_names = {
            "erosion": "\u5340\u57df\u4fb5\u8755",
            "dilation": "\u5340\u57df\u81a8\u8139",
            "opening": "\u5340\u57df\u958b\u904b\u7b97",
            "closing": "\u5340\u57df\u9589\u904b\u7b97",
        }
        title = op_names.get(op, op)

        dlg = tk.Toplevel(self)
        dlg.title(title)
        dlg.configure(bg="#2b2b2b")
        dlg.resizable(False, False)
        dlg.grab_set()

        tk.Label(dlg, text=title, bg="#2b2b2b", fg="#e0e0e0",
                 font=("", 10, "bold")).grid(
            row=0, column=0, columnspan=2, pady=(10, 6), padx=10)

        tk.Label(dlg, text="\u6838\u5927\u5c0f (ksize):", bg="#2b2b2b", fg="#e0e0e0").grid(
            row=1, column=0, sticky="e", padx=(10, 4), pady=4)
        ksize_var = tk.StringVar(value="5")
        tk.Entry(dlg, textvariable=ksize_var, width=8,
                 bg="#3c3c3c", fg="#e0e0e0", insertbackground="#e0e0e0").grid(
            row=1, column=1, padx=(0, 10), pady=4)

        tk.Label(dlg, text="\u6838\u5f62\u72c0:", bg="#2b2b2b", fg="#e0e0e0").grid(
            row=2, column=0, sticky="e", padx=(10, 4), pady=4)
        shape_var = tk.StringVar(value="ellipse")
        ttk.Combobox(dlg, textvariable=shape_var, width=10,
                     values=["ellipse", "rectangle", "cross"],
                     state="readonly").grid(row=2, column=1, padx=(0, 10), pady=4)

        tk.Label(dlg, text="\u8fed\u4ee3\u6b21\u6578:", bg="#2b2b2b", fg="#e0e0e0").grid(
            row=3, column=0, sticky="e", padx=(10, 4), pady=4)
        iter_var = tk.StringVar(value="1")
        tk.Entry(dlg, textvariable=iter_var, width=8,
                 bg="#3c3c3c", fg="#e0e0e0", insertbackground="#e0e0e0").grid(
            row=3, column=1, padx=(0, 10), pady=4)

        def _apply():
            try:
                ks = int(ksize_var.get())
                iters = int(iter_var.get())
            except ValueError:
                messagebox.showwarning("\u8b66\u544a", "\u8acb\u8f38\u5165\u6709\u6548\u7684\u6578\u503c\u3002", parent=dlg)
                return
            if ks < 1:
                ks = 1
            if ks % 2 == 0:
                ks += 1
            if iters < 1:
                iters = 1
            sh = shape_var.get()
            dlg.destroy()

            cur_region = self._current_region
            shape_map = {"ellipse": cv2.MORPH_ELLIPSE,
                         "rectangle": cv2.MORPH_RECT,
                         "cross": cv2.MORPH_CROSS}

            def _compute():
                from dl_anomaly.core.region_ops import compute_region_properties
                from dl_anomaly.core.region import Region
                mask = cur_region.to_binary_mask()
                morph = shape_map.get(sh, cv2.MORPH_ELLIPSE)
                kernel = cv2.getStructuringElement(morph, (ks, ks))
                if op == "erosion":
                    result = cv2.erode(mask, kernel, iterations=iters)
                elif op == "dilation":
                    result = cv2.dilate(mask, kernel, iterations=iters)
                elif op == "opening":
                    result = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iters)
                elif op == "closing":
                    result = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iters)
                else:
                    return None
                num, labels = cv2.connectedComponents(result, connectivity=8)
                labels = labels.astype(np.int32)
                props = compute_region_properties(labels, cur_region.source_image)
                region = Region(labels=labels, num_regions=num - 1, properties=props,
                              source_image=cur_region.source_image,
                              source_shape=cur_region.source_shape)
                name = f"{title} k={ks} {sh} x{iters}"
                return region, name

            def _done(result):
                if result is None:
                    return
                region, name = result
                self._current_region = region
                self._pipeline_panel.add_step(name, region.to_binary_mask(), region=region)
                self.set_status(f"{name}: {region.num_regions} \u500b\u5340\u57df")

            self._run_in_bg(_compute, on_done=_done, status_msg=f"{title}...")

        btn_frame = tk.Frame(dlg, bg="#2b2b2b")
        btn_frame.grid(row=4, column=0, columnspan=2, pady=(6, 10))
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
    # Region filter
    # ==================================================================

    def _open_region_filter(self: "HalconApp") -> None:
        """Open region filter dialog."""
        if not self._ensure_region():
            return
        img = self._get_current_image()
        if img is None:
            return
        try:
            from dl_anomaly.gui.region_filter_dialog import RegionFilterDialog

            def on_accept(filtered_region, display_image, name, conditions_list=None):
                self._current_region = filtered_region
                op_meta = {"category": "region", "op": "select_shape",
                           "params": {"conditions": conditions_list or []}}
                self._pipeline_panel.add_step(
                    name, display_image, region=filtered_region, op_meta=op_meta)
                self.set_status(f"\u7be9\u9078\u5b8c\u6210: {filtered_region.num_regions} \u500b\u5340\u57df")

            RegionFilterDialog(self, self._current_region, img, on_accept=on_accept)
        except Exception as exc:
            self._show_error("\u958b\u555f\u7be9\u9078\u5c0d\u8a71\u6846\u5931\u6557", exc)

    # ==================================================================
    # Select by gray
    # ==================================================================

    def _region_select_gray(self: "HalconApp") -> None:
        """Filter regions by gray value."""
        if not self._ensure_region():
            return
        img = self._get_current_image()
        if img is None:
            return

        dlg = tk.Toplevel(self)
        dlg.title("\u4f9d\u7070\u5ea6\u7be9\u9078")
        dlg.configure(bg="#2b2b2b")
        dlg.resizable(False, False)
        dlg.grab_set()

        tk.Label(dlg, text="\u7be9\u9078\u5340\u57df\u7684\u5e73\u5747\u7070\u5ea6\u7bc4\u570d",
                 bg="#2b2b2b", fg="#e0e0e0", font=("", 10, "bold")).grid(
            row=0, column=0, columnspan=2, pady=(10, 6), padx=10)

        tk.Label(dlg, text="\u6700\u5c0f\u7070\u5ea6:", bg="#2b2b2b", fg="#e0e0e0").grid(
            row=1, column=0, sticky="e", padx=(10, 4), pady=4)
        min_var = tk.StringVar(value="0")
        min_entry = tk.Entry(dlg, textvariable=min_var, width=8,
                             bg="#3c3c3c", fg="#e0e0e0", insertbackground="#e0e0e0")
        min_entry.grid(row=1, column=1, padx=(0, 10), pady=4)

        tk.Label(dlg, text="\u6700\u5927\u7070\u5ea6:", bg="#2b2b2b", fg="#e0e0e0").grid(
            row=2, column=0, sticky="e", padx=(10, 4), pady=4)
        max_var = tk.StringVar(value="128")
        max_entry = tk.Entry(dlg, textvariable=max_var, width=8,
                             bg="#3c3c3c", fg="#e0e0e0", insertbackground="#e0e0e0")
        max_entry.grid(row=2, column=1, padx=(0, 10), pady=4)

        def _apply():
            try:
                mn = float(min_var.get())
                mx = float(max_var.get())
            except ValueError:
                messagebox.showwarning("\u8b66\u544a", "\u8acb\u8f38\u5165\u6709\u6548\u7684\u6578\u503c\u3002", parent=dlg)
                return
            dlg.destroy()

            cur_region = self._current_region

            def _compute():
                from dl_anomaly.core.region_ops import select_shape
                return select_shape(cur_region, "mean_value", mn, mx)

            def _done(filtered):
                self._current_region = filtered
                self._pipeline_panel.add_step(
                    f"\u7070\u5ea6\u7be9\u9078 [{mn:.0f}-{mx:.0f}] ({filtered.num_regions})",
                    filtered.to_binary_mask(), region=filtered)
                self.set_status(f"\u7070\u5ea6\u7be9\u9078: {filtered.num_regions} \u500b\u5340\u57df")

            self._run_in_bg(_compute, on_done=_done, status_msg="\u7070\u5ea6\u7be9\u9078\u4e2d...")

        btn_frame = tk.Frame(dlg, bg="#2b2b2b")
        btn_frame.grid(row=3, column=0, columnspan=2, pady=(6, 10))
        tk.Button(btn_frame, text="\u78ba\u5b9a", command=_apply, width=8,
                  bg="#3a3a5c", fg="#e0e0e0", activebackground="#4a4a6c",
                  activeforeground="#ffffff").pack(side="left", padx=4)
        tk.Button(btn_frame, text="\u53d6\u6d88", command=dlg.destroy, width=8,
                  bg="#3a3a5c", fg="#e0e0e0", activebackground="#4a4a6c",
                  activeforeground="#ffffff").pack(side="left", padx=4)

        dlg.update_idletasks()
        x = self.winfo_x() + (self.winfo_width() - dlg.winfo_width()) // 2
        y = self.winfo_y() + (self.winfo_height() - dlg.winfo_height()) // 2
        dlg.geometry(f"+{x}+{y}")
        min_entry.focus_set()

    # ==================================================================
    # Sort
    # ==================================================================

    def _region_sort(self: "HalconApp") -> None:
        """Sort regions by area."""
        if not self._ensure_region():
            return
        try:
            from dl_anomaly.core.region import Region
            sorted_props = sorted(self._current_region.properties, key=lambda p: p.area, reverse=True)
            new_labels = np.zeros_like(self._current_region.labels)
            new_props = []
            for new_idx, p in enumerate(sorted_props, 1):
                new_labels[self._current_region.labels == p.index] = new_idx
                p_copy = copy.copy(p)
                p_copy.index = new_idx
                new_props.append(p_copy)
            region = Region(labels=new_labels, num_regions=len(new_props), properties=new_props,
                          source_image=self._current_region.source_image,
                          source_shape=self._current_region.source_shape)
            self._current_region = region
            self._pipeline_panel.add_step("\u6392\u5e8f\u5340\u57df (\u9762\u7a4d)", region.to_binary_mask(), region=region)
        except Exception as exc:
            self._show_error("\u6392\u5e8f\u5931\u6557", exc)

    # ==================================================================
    # Set operations
    # ==================================================================

    def _region_set_op(self: "HalconApp", op: str) -> None:
        """Region set operations."""
        if not self._ensure_region():
            return
        try:
            if op == "complement":
                region = self._current_region.complement()
            else:
                messagebox.showinfo("\u63d0\u793a", f"\u5340\u57df{op}\u9700\u8981\u5169\u500b\u5340\u57df\uff0c\u76ee\u524d\u50c5\u6709\u4e00\u500b\u3002\n\u8acb\u5148\u5206\u5225\u7522\u751f\u5169\u500b\u5340\u57df\u6b65\u9a5f\u3002")
                return
            self._current_region = region
            self._pipeline_panel.add_step(f"\u5340\u57df\u88dc\u96c6 ({region.num_regions})", region.to_binary_mask(), region=region)
        except Exception as exc:
            self._show_error("\u96c6\u5408\u64cd\u4f5c\u5931\u6557", exc)

    # ==================================================================
    # Blob analysis
    # ==================================================================

    def _open_blob_analysis(self: "HalconApp") -> None:
        """Open blob analysis dialog."""
        img = self._get_current_image()
        if img is None:
            messagebox.showwarning("\u8b66\u544a", "\u8acb\u5148\u8f09\u5165\u5716\u7247\u3002")
            return
        try:
            from dl_anomaly.gui.blob_analysis import BlobAnalysisDialog

            def on_accept(steps):
                for step_name, step_img, step_region in steps:
                    self._pipeline_panel.add_step(step_name, step_img, region=step_region)
                if steps:
                    last_region = steps[-1][2]
                    if last_region is not None:
                        self._current_region = last_region
                self.set_status("Blob \u5206\u6790\u5b8c\u6210")

            BlobAnalysisDialog(self, img, on_accept=on_accept)
        except Exception as exc:
            self._show_error("Blob \u5206\u6790\u958b\u555f\u5931\u6557", exc)

    # ==================================================================
    # Domain operations
    # ==================================================================

    def _cmd_reduce_domain(self: "HalconApp"):
        """Restrict current image to the domain of the current region."""
        if self._current_region is None:
            messagebox.showwarning("\u8b66\u544a", "\u8acb\u5148\u5efa\u7acb\u5340\u57df (\u95be\u503c\u5206\u5272\u3001Blob \u5206\u6790\u7b49)\u3002")
            return
        img = self._get_current_image()
        if img is None:
            messagebox.showwarning("\u8b66\u544a", "\u8acb\u5148\u8f09\u5165\u5716\u7247\u3002")
            return

        from dl_anomaly.core.halcon_ops import reduce_domain
        result = reduce_domain(img, self._current_region)
        self._add_pipeline_step("\u7e2e\u6e1b\u57df", result)
        self.set_status("\u5b8c\u6210: \u7e2e\u6e1b\u57df (Reduce Domain)")

    def _cmd_crop_domain(self: "HalconApp"):
        """Crop current image to the bounding box of the current region."""
        if self._current_region is None:
            messagebox.showwarning("\u8b66\u544a", "\u8acb\u5148\u5efa\u7acb\u5340\u57df (\u95be\u503c\u5206\u5272\u3001Blob \u5206\u6790\u7b49)\u3002")
            return
        img = self._get_current_image()
        if img is None:
            messagebox.showwarning("\u8b66\u544a", "\u8acb\u5148\u8f09\u5165\u5716\u7247\u3002")
            return

        from dl_anomaly.core.halcon_ops import crop_domain
        result = crop_domain(img, self._current_region)
        self._add_pipeline_step("\u88c1\u5207\u57df", result)
        self.set_status(f"\u5b8c\u6210: \u88c1\u5207\u57df ({result.shape[1]}x{result.shape[0]})")

    # ==================================================================
    # Region highlight / remove (from properties table click)
    # ==================================================================

    def _on_region_highlight(self: "HalconApp", region_index: int) -> None:
        """Re-render viewer with specific region highlighted in yellow."""
        if self._current_region is None:
            return
        step = self._pipeline_panel.get_current_step()
        if step is None:
            return
        try:
            from dl_anomaly.core.region_ops import region_to_display_image
            display = region_to_display_image(
                self._current_region,
                step.array,
                highlight_indices=[region_index],
                highlight_color=(255, 255, 0),
            )
            if display.ndim == 3 and display.shape[2] == 3:
                display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            self._viewer.set_image(display)
        except Exception:
            pass

    def _on_region_remove(self: "HalconApp", region_index: int) -> None:
        """Remove a single region by its 1-based index and add a new pipeline step."""
        if self._current_region is None:
            return
        step = self._pipeline_panel.get_current_step()
        if step is None:
            return
        try:
            remaining = [
                p.index for p in self._current_region.properties
                if p.index != region_index
            ]
            if not remaining:
                return
            new_region = self._current_region._keep_indices(remaining)
            self._current_region = new_region

            ci = self._pipeline_panel.get_current_index()
            if ci >= 0:
                self._undo_stack.append(ci)
            self._redo_stack.clear()

            from dl_anomaly.core.region_ops import region_to_display_image
            display = region_to_display_image(new_region, step.array)
            if display.ndim == 3 and display.shape[2] == 3:
                display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)

            self._pipeline_panel.add_step(
                f"\u79fb\u9664\u5340\u57df #{region_index}",
                display,
                region=new_region,
                op_meta={"category": "region", "op": "remove", "params": {"index": region_index}},
            )
            self.set_status(f"\u5df2\u79fb\u9664\u5340\u57df #{region_index}\uff0c\u5269\u9918 {new_region.num_regions} \u500b")
        except Exception as exc:
            self._show_error("\u79fb\u9664\u5340\u57df\u5931\u6557", exc)
