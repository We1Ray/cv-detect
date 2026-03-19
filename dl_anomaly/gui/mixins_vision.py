"""Vision operator mixin for InspectorApp.

Covers: reusable parameter dialog helper, filter dialog helper,
all _dlg_* filter dialog launchers, _apply_vision_op, _vision_op_compute.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from dl_anomaly.gui.inspector_app import InspectorApp


class VisionOpsMixin:
    """Vision-specific operators, parameter dialogs, and filter dialogs."""

    # ==================================================================
    # Reusable parameter dialog helper
    # ==================================================================

    def _open_param_dialog(self: "InspectorApp", title, params, on_apply):
        """Build a dark-themed parameter dialog from a specification list.

        Parameters
        ----------
        title : str
            Dialog window title.
        params : list of dict
            Each dict has keys: label, key, type ("int"|"float"|"combo"),
            default, and optionally values (for combo).
        on_apply : callable(dict)
            Called with {key: value, ...} when user clicks Apply.
        """
        dlg = tk.Toplevel(self)
        dlg.title(title)
        dlg.configure(bg="#2b2b2b")
        dlg.resizable(False, False)
        dlg.grab_set()

        tk.Label(dlg, text=title, bg="#2b2b2b", fg="#e0e0e0",
                 font=("", 10, "bold")).grid(
            row=0, column=0, columnspan=2, pady=(10, 6), padx=10)

        vars_map = {}
        for i, p in enumerate(params, start=1):
            tk.Label(dlg, text=p["label"], bg="#2b2b2b", fg="#e0e0e0").grid(
                row=i, column=0, sticky="e", padx=(10, 4), pady=4)
            var = tk.StringVar(value=str(p["default"]))
            vars_map[p["key"]] = (var, p["type"])
            if p["type"] == "combo":
                combo = ttk.Combobox(dlg, textvariable=var, width=14,
                                     values=p.get("values", []), state="readonly")
                combo.grid(row=i, column=1, padx=(0, 10), pady=4)
            else:
                tk.Entry(dlg, textvariable=var, width=10,
                         bg="#3c3c3c", fg="#e0e0e0",
                         insertbackground="#e0e0e0").grid(
                    row=i, column=1, padx=(0, 10), pady=4)

        def _do_apply():
            result = {}
            for key, (var, vtype) in vars_map.items():
                raw = var.get()
                try:
                    if vtype == "int":
                        result[key] = int(raw)
                    elif vtype == "float":
                        result[key] = float(raw)
                    else:
                        result[key] = raw
                except ValueError:
                    messagebox.showwarning("\u8b66\u544a", f"\u53c3\u6578 '{key}' \u7684\u503c\u7121\u6548: {raw}", parent=dlg)
                    return
            dlg.destroy()
            on_apply(result)

        btn_frame = tk.Frame(dlg, bg="#2b2b2b")
        btn_frame.grid(row=len(params) + 1, column=0, columnspan=2, pady=(6, 10))
        tk.Button(btn_frame, text="\u57f7\u884c", command=_do_apply, width=8,
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
    # Parameterized vision filter dialogs
    # ==================================================================

    def _open_filter_dialog(self: "InspectorApp", op_label, params, apply_func):
        """Open a parameter dialog for a filter operation.

        Parameters
        ----------
        op_label : str
            Display name for the operation (used in pipeline step name).
        params : list of dict
            Parameter spec for _open_param_dialog.
        apply_func : callable(img, param_dict) -> ndarray
            Function that applies the filter and returns result image.
        """
        img = self._get_current_image()
        if img is None:
            messagebox.showwarning("\u8b66\u544a", "\u8acb\u5148\u8f09\u5165\u5716\u7247\u3002")
            return

        def _on_apply(p):
            def _compute():
                return apply_func(img, p)

            def _done(result):
                param_str = " ".join(f"{k}={v}" for k, v in p.items())
                name = f"{op_label} {param_str}"
                op_meta = {"category": "dialog_op", "op": op_label, "params": dict(p)}
                self._add_pipeline_step(name, result, op_meta=op_meta)
                self.set_status(f"\u5b8c\u6210: {name}")

            self._run_in_bg(_compute, on_done=_done, status_msg=f"\u57f7\u884c {op_label}...")

        self._open_param_dialog(op_label, params, _on_apply)

    # ------------------------------------------------------------------
    # Individual filter dialog launchers
    # ------------------------------------------------------------------

    def _dlg_mean_image(self: "InspectorApp"):
        from dl_anomaly.core import vision_ops as hops
        self._open_filter_dialog("\u5747\u503c\u6ffe\u6ce2", [
            {"label": "\u6838\u5927\u5c0f (ksize):", "key": "ksize", "type": "int", "default": 5},
        ], lambda img, p: hops.mean_image(img, p["ksize"]))

    def _dlg_median_image(self: "InspectorApp"):
        from dl_anomaly.core import vision_ops as hops
        self._open_filter_dialog("\u4e2d\u503c\u6ffe\u6ce2", [
            {"label": "\u6838\u5927\u5c0f (ksize):", "key": "ksize", "type": "int", "default": 5},
        ], lambda img, p: hops.median_image(img, p["ksize"]))

    def _dlg_gauss_blur(self: "InspectorApp"):
        from dl_anomaly.core import vision_ops as hops
        self._open_filter_dialog("\u9ad8\u65af\u6a21\u7cca", [
            {"label": "\u6838\u5927\u5c0f (ksize):", "key": "ksize", "type": "int", "default": 5},
            {"label": "Sigma:", "key": "sigma", "type": "float", "default": 0},
        ], lambda img, p: (
            hops.gauss_blur(img, p["ksize"]) if p["sigma"] == 0
            else hops.gauss_filter(img, p["sigma"])
        ))

    def _dlg_bilateral_filter(self: "InspectorApp"):
        from dl_anomaly.core import vision_ops as hops
        self._open_filter_dialog("\u96d9\u908a\u6ffe\u6ce2", [
            {"label": "d:", "key": "d", "type": "int", "default": 9},
            {"label": "Sigma Color:", "key": "sigma_color", "type": "float", "default": 75},
            {"label": "Sigma Space:", "key": "sigma_space", "type": "float", "default": 75},
        ], lambda img, p: hops.bilateral_filter(img, p["d"], p["sigma_color"], p["sigma_space"]))

    def _dlg_sharpen(self: "InspectorApp"):
        from dl_anomaly.core import vision_ops as hops
        self._open_filter_dialog("\u92b3\u5316", [
            {"label": "\u5f37\u5ea6 (amount):", "key": "amount", "type": "float", "default": 0.5},
        ], lambda img, p: hops.sharpen_image(img, p["amount"]))

    def _dlg_canny(self: "InspectorApp"):
        from dl_anomaly.core import vision_ops as hops
        self._open_filter_dialog("Canny \u908a\u7de3", [
            {"label": "\u4f4e\u95be\u503c (low):", "key": "low", "type": "float", "default": 50},
            {"label": "\u9ad8\u95be\u503c (high):", "key": "high", "type": "float", "default": 150},
            {"label": "Sigma:", "key": "sigma", "type": "float", "default": 1.0},
        ], lambda img, p: hops.edges_canny(img, p["low"], p["high"], p["sigma"]))

    def _dlg_gray_erosion(self: "InspectorApp"):
        from dl_anomaly.core import vision_ops as hops
        self._open_filter_dialog("\u7070\u5ea6\u4fb5\u8755", [
            {"label": "\u6838\u5927\u5c0f (ksize):", "key": "ksize", "type": "int", "default": 5},
        ], lambda img, p: hops.gray_erosion(img, p["ksize"]))

    def _dlg_gray_dilation(self: "InspectorApp"):
        from dl_anomaly.core import vision_ops as hops
        self._open_filter_dialog("\u7070\u5ea6\u81a8\u8139", [
            {"label": "\u6838\u5927\u5c0f (ksize):", "key": "ksize", "type": "int", "default": 5},
        ], lambda img, p: hops.gray_dilation(img, p["ksize"]))

    def _dlg_gray_opening(self: "InspectorApp"):
        from dl_anomaly.core import vision_ops as hops
        self._open_filter_dialog("\u7070\u5ea6\u958b\u904b\u7b97", [
            {"label": "\u6838\u5927\u5c0f (ksize):", "key": "ksize", "type": "int", "default": 5},
        ], lambda img, p: hops.gray_opening(img, p["ksize"]))

    def _dlg_gray_closing(self: "InspectorApp"):
        from dl_anomaly.core import vision_ops as hops
        self._open_filter_dialog("\u7070\u5ea6\u9589\u904b\u7b97", [
            {"label": "\u6838\u5927\u5c0f (ksize):", "key": "ksize", "type": "int", "default": 5},
        ], lambda img, p: hops.gray_closing(img, p["ksize"]))

    def _dlg_top_hat(self: "InspectorApp"):
        from dl_anomaly.core import vision_ops as hops
        self._open_filter_dialog("Top-hat", [
            {"label": "\u6838\u5927\u5c0f (ksize):", "key": "ksize", "type": "int", "default": 9},
        ], lambda img, p: hops.top_hat(img, p["ksize"]))

    def _dlg_bottom_hat(self: "InspectorApp"):
        from dl_anomaly.core import vision_ops as hops
        self._open_filter_dialog("Bottom-hat", [
            {"label": "\u6838\u5927\u5c0f (ksize):", "key": "ksize", "type": "int", "default": 9},
        ], lambda img, p: hops.bottom_hat(img, p["ksize"]))

    def _dlg_emphasize(self: "InspectorApp"):
        from dl_anomaly.core import vision_ops as hops
        self._open_filter_dialog("\u5f37\u8abf", [
            {"label": "\u6838\u5927\u5c0f (ksize):", "key": "ksize", "type": "int", "default": 7},
            {"label": "\u5f37\u5ea6 (factor):", "key": "factor", "type": "float", "default": 1.5},
        ], lambda img, p: hops.emphasize(img, p["ksize"], p["factor"]))

    def _dlg_scale_image(self: "InspectorApp"):
        from dl_anomaly.core import vision_ops as hops
        self._open_filter_dialog("\u4eae\u5ea6/\u5c0d\u6bd4\u5ea6\u8abf\u6574", [
            {"label": "\u4e58\u6578 (mult):", "key": "mult", "type": "float", "default": 1.0},
            {"label": "\u504f\u79fb (add):", "key": "add", "type": "float", "default": 0},
        ], lambda img, p: hops.scale_image(img, p["mult"], p["add"]))

    def _dlg_log_image(self: "InspectorApp"):
        from dl_anomaly.core import vision_ops as hops
        self._open_filter_dialog("\u5c0d\u6578\u8b8a\u63db", [
            {"label": "\u5e95\u6578 (base):", "key": "base", "type": "combo",
             "default": "e", "values": ["e", "2", "10"]},
        ], lambda img, p: hops.log_image(img, p["base"]))

    def _dlg_exp_image(self: "InspectorApp"):
        from dl_anomaly.core import vision_ops as hops
        self._open_filter_dialog("\u6307\u6578\u8b8a\u63db", [
            {"label": "\u5e95\u6578 (base):", "key": "base", "type": "combo",
             "default": "e", "values": ["e", "2", "10"]},
        ], lambda img, p: hops.exp_image(img, p["base"]))

    def _dlg_gamma_image(self: "InspectorApp"):
        from dl_anomaly.core import vision_ops as hops
        self._open_filter_dialog("Gamma \u6821\u6b63", [
            {"label": "Gamma:", "key": "gamma", "type": "float", "default": 1.0},
        ], lambda img, p: hops.gamma_image(img, p["gamma"]))

    def _dlg_var_threshold(self: "InspectorApp"):
        from dl_anomaly.core import vision_ops as hops
        img = self._get_current_image()
        if img is None:
            messagebox.showwarning("\u8b66\u544a", "\u8acb\u5148\u8f09\u5165\u5716\u7247\u3002")
            return

        def _on_apply(p):
            def _compute():
                return hops.var_threshold(
                    img, p["width"], p["height"], p["std_mult"],
                    p["abs_thresh"], p["light_dark"])

            def _done(result):
                from dl_anomaly.core.vision_ops import Region
                name = f"\u53ef\u8b8a\u95be\u503c w={p['width']} h={p['height']} s={p['std_mult']}"
                region = Region(mask=result)
                self._pipeline_panel.add_step(name, img, region=region)
                self._current_region = region
                overlay = img.copy()
                if overlay.ndim == 2:
                    overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
                overlay[result > 0] = (0, 0, 255)
                self._viewer.display_array(overlay)
                self.set_status(f"\u5b8c\u6210: {name}")

            self._run_in_bg(_compute, on_done=_done, status_msg="\u57f7\u884c\u53ef\u8b8a\u95be\u503c...")

        self._open_param_dialog("\u53ef\u8b8a\u95be\u503c", [
            {"label": "\u5bec\u5ea6 (width):", "key": "width", "type": "int", "default": 15},
            {"label": "\u9ad8\u5ea6 (height):", "key": "height", "type": "int", "default": 15},
            {"label": "\u6a19\u6e96\u5dee\u500d\u6578:", "key": "std_mult", "type": "float", "default": 0.2},
            {"label": "\u7d55\u5c0d\u95be\u503c:", "key": "abs_thresh", "type": "float", "default": 2},
            {"label": "\u660e\u6697\u6a21\u5f0f:", "key": "light_dark", "type": "combo",
             "default": "dark", "values": ["dark", "light", "equal", "not_equal"]},
        ], _on_apply)

    def _dlg_local_threshold(self: "InspectorApp"):
        from dl_anomaly.core import vision_ops as hops
        img = self._get_current_image()
        if img is None:
            messagebox.showwarning("\u8b66\u544a", "\u8acb\u5148\u8f09\u5165\u5716\u7247\u3002")
            return

        def _on_apply(p):
            def _compute():
                return hops.local_threshold(
                    img, p["method"], p["light_dark"], p["ksize"], p["scale"])

            def _done(result):
                from dl_anomaly.core.vision_ops import Region
                name = f"\u5c40\u90e8\u95be\u503c k={p['ksize']} s={p['scale']}"
                region = Region(mask=result)
                self._pipeline_panel.add_step(name, img, region=region)
                self._current_region = region
                overlay = img.copy()
                if overlay.ndim == 2:
                    overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
                overlay[result > 0] = (0, 0, 255)
                self._viewer.display_array(overlay)
                self.set_status(f"\u5b8c\u6210: {name}")

            self._run_in_bg(_compute, on_done=_done, status_msg="\u57f7\u884c\u5c40\u90e8\u95be\u503c...")

        self._open_param_dialog("\u5c40\u90e8\u95be\u503c", [
            {"label": "\u65b9\u6cd5:", "key": "method", "type": "combo",
             "default": "adapted_std_deviation",
             "values": ["adapted_std_deviation", "mean"]},
            {"label": "\u660e\u6697\u6a21\u5f0f:", "key": "light_dark", "type": "combo",
             "default": "dark", "values": ["dark", "light", "not_equal"]},
            {"label": "\u6838\u5927\u5c0f (ksize):", "key": "ksize", "type": "int", "default": 15},
            {"label": "\u6bd4\u4f8b (scale):", "key": "scale", "type": "float", "default": 0.2},
        ], _on_apply)

    def _dlg_fft(self: "InspectorApp"):
        from dl_anomaly.core import vision_ops as hops
        self._open_filter_dialog("FFT \u983b\u8b5c", [],
                                 lambda img, p: hops.fft_image(img))

    def _dlg_freq_filter(self: "InspectorApp", filter_type="lowpass"):
        from dl_anomaly.core import vision_ops as hops
        label = "\u4f4e\u901a\u6ffe\u6ce2" if filter_type == "lowpass" else "\u9ad8\u901a\u6ffe\u6ce2"
        self._open_filter_dialog(label, [
            {"label": "\u622a\u6b62\u983b\u7387 (sigma):", "key": "cutoff", "type": "float", "default": 30},
        ], lambda img, p: hops.freq_filter(img, filter_type, p["cutoff"]))

    def _dlg_derivative_gauss(self: "InspectorApp"):
        from dl_anomaly.core import vision_ops as hops
        self._open_filter_dialog("\u9ad8\u65af\u5c0e\u6578", [
            {"label": "Sigma:", "key": "sigma", "type": "float", "default": 1.0},
            {"label": "\u65b9\u5411:", "key": "component", "type": "combo",
             "default": "x", "values": ["x", "y"]},
        ], lambda img, p: hops.derivative_gauss(img, p["sigma"], p["component"]))

    def _dlg_watersheds(self: "InspectorApp"):
        from dl_anomaly.core import vision_ops as hops
        self._open_filter_dialog("\u5206\u6c34\u5dba", [
            {"label": "\u6a19\u8a18\u95be\u503c:", "key": "marker_thresh", "type": "float", "default": 0.5},
        ], lambda img, p: hops.watersheds(img, p["marker_thresh"]))

    def _dlg_distance_transform(self: "InspectorApp"):
        from dl_anomaly.core import vision_ops as hops
        self._open_filter_dialog("\u8ddd\u96e2\u8b8a\u63db", [
            {"label": "\u65b9\u6cd5:", "key": "method", "type": "combo",
             "default": "L2", "values": ["L1", "L2", "C"]},
        ], lambda img, p: hops.distance_transform(img, p["method"]))

    def _dlg_points_harris(self: "InspectorApp"):
        from dl_anomaly.core import vision_ops as hops
        self._open_filter_dialog("Harris \u89d2\u9ede", [
            {"label": "\u5340\u584a\u5927\u5c0f:", "key": "block_size", "type": "int", "default": 2},
            {"label": "ksize:", "key": "ksize", "type": "int", "default": 3},
            {"label": "k:", "key": "k", "type": "float", "default": 0.04},
            {"label": "\u95be\u503c:", "key": "threshold", "type": "float", "default": 0.01},
        ], lambda img, p: hops.points_harris(img, p["block_size"], p["ksize"], p["k"], p["threshold"]))

    def _dlg_points_shi_tomasi(self: "InspectorApp"):
        from dl_anomaly.core import vision_ops as hops
        self._open_filter_dialog("Shi-Tomasi \u7279\u5fb5\u9ede", [
            {"label": "\u6700\u5927\u89d2\u9ede\u6578:", "key": "max_corners", "type": "int", "default": 100},
            {"label": "\u54c1\u8cea:", "key": "quality", "type": "float", "default": 0.01},
            {"label": "\u6700\u5c0f\u8ddd\u96e2:", "key": "min_distance", "type": "float", "default": 10},
        ], lambda img, p: hops.points_shi_tomasi(img, p["max_corners"], p["quality"], p["min_distance"]))

    def _dlg_hough_lines(self: "InspectorApp"):
        from dl_anomaly.core import vision_ops as hops
        self._open_filter_dialog("Hough \u76f4\u7dda", [
            {"label": "rho (\u50cf\u7d20):", "key": "rho", "type": "float", "default": 1.0},
            {"label": "theta (\u5ea6):", "key": "theta_deg", "type": "float", "default": 1.0},
            {"label": "\u95be\u503c:", "key": "threshold", "type": "int", "default": 100},
        ], lambda img, p: hops.hough_lines(img, p["rho"], p["theta_deg"], p["threshold"]))

    def _dlg_hough_circles(self: "InspectorApp"):
        from dl_anomaly.core import vision_ops as hops
        self._open_filter_dialog("Hough \u5713", [
            {"label": "dp:", "key": "dp", "type": "float", "default": 1.2},
            {"label": "\u6700\u5c0f\u8ddd\u96e2:", "key": "min_dist", "type": "float", "default": 30},
            {"label": "param1:", "key": "param1", "type": "float", "default": 50},
            {"label": "param2:", "key": "param2", "type": "float", "default": 30},
        ], lambda img, p: hops.hough_circles(img, p["dp"], p["min_dist"], p["param1"], p["param2"]))

    def _dlg_clahe(self: "InspectorApp"):
        from dl_anomaly.core import vision_ops as hops
        self._open_filter_dialog("CLAHE", [
            {"label": "\u5c0d\u6bd4\u5ea6\u9650\u5236:", "key": "clip_limit", "type": "float", "default": 2.0},
            {"label": "\u683c\u5b50\u5927\u5c0f:", "key": "tile_size", "type": "int", "default": 8},
        ], lambda img, p: hops.clahe(img, p["clip_limit"], p["tile_size"]))

    def _dlg_estimate_noise(self: "InspectorApp"):
        from dl_anomaly.core import vision_ops as hops
        img = self._get_current_image()
        if img is None:
            messagebox.showwarning("\u8b66\u544a", "\u8acb\u5148\u8f09\u5165\u5716\u7247\u3002")
            return
        sigma = hops.estimate_noise(img)
        messagebox.showinfo("\u566a\u8072\u4f30\u8a08", f"\u4f30\u8a08\u566a\u8072\u6a19\u6e96\u5dee \u03c3 = {sigma:.4f}")

    def _dlg_gen_gauss_pyramid(self: "InspectorApp"):
        from dl_anomaly.core import vision_ops as hops
        self._open_filter_dialog("\u9ad8\u65af\u91d1\u5b57\u5854", [
            {"label": "\u5c64\u6578:", "key": "levels", "type": "int", "default": 4},
        ], lambda img, p: hops.gen_gauss_pyramid(img, p["levels"])[-1])

    # ==================================================================
    # Apply vision op (generic dispatcher)
    # ==================================================================

    def _apply_vision_op(self: "InspectorApp", op: str) -> None:
        """Apply a vision operator in a background thread."""
        img = self._get_current_image()
        if img is None:
            messagebox.showwarning("\u8b66\u544a", "\u8acb\u5148\u8f09\u5165\u5716\u7247\u3002")
            return

        def _compute():
            return self._vision_op_compute(op, img)

        def _done(pair):
            name, result = pair
            if result is not None:
                op_meta = {"category": "vision", "op": op, "params": {}}
                self._add_pipeline_step(name, result, op_meta=op_meta)
                self.set_status_success(f"\u5b8c\u6210: {name}")
                self._history_panel.add_entry(name, f"op={op}")

        self._run_in_bg(_compute, on_done=_done, status_msg=f"\u57f7\u884c {op}...")

    def _vision_op_compute(self: "InspectorApp", op: str, img: np.ndarray):
        """Pure computation for a vision op (runs in background thread).
        Returns (name, result_array) or (name, None).
        """
        from dl_anomaly.core import vision_ops as hops

        result = None
        name = ""

        if op == "mean_image":
            result = hops.mean_image(img, 5)
            name = "\u5747\u503c\u6ffe\u6ce2 k=5"
        elif op == "median_image":
            result = hops.median_image(img, 5)
            name = "\u4e2d\u503c\u6ffe\u6ce2 k=5"
        elif op == "gauss_filter":
            result = hops.gauss_filter(img, 1.5)
            name = "\u9ad8\u65af\u6ffe\u6ce2 \u03c3=1.5"
        elif op == "bilateral_filter":
            result = hops.bilateral_filter(img, 9, 75, 75)
            name = "\u96d9\u908a\u6ffe\u6ce2"
        elif op == "sharpen_image":
            result = hops.sharpen_image(img, 0.5)
            name = "\u92b3\u5316 0.5"
        elif op == "emphasize":
            result = hops.emphasize(img, 7, 1.5)
            name = "\u5f37\u8abf\u6ffe\u6ce2"
        elif op == "laplace_filter":
            result = hops.laplace_filter(img)
            name = "Laplacian"
        elif op == "edges_canny":
            result = hops.edges_canny(img, 50, 150, 1.0)
            name = "Canny \u908a\u7de3"
        elif op == "sobel_filter":
            result = hops.sobel_filter(img, "both")
            name = "Sobel \u908a\u7de3"
        elif op == "prewitt_filter":
            result = hops.prewitt_filter(img)
            name = "Prewitt \u908a\u7de3"
        elif op == "zero_crossing":
            result = hops.zero_crossing(img)
            name = "\u96f6\u4ea4\u53c9"
        elif op == "gray_erosion":
            result = hops.gray_erosion(img, 5)
            name = "\u7070\u5ea6\u4fb5\u8755 k=5"
        elif op == "gray_dilation":
            result = hops.gray_dilation(img, 5)
            name = "\u7070\u5ea6\u81a8\u8139 k=5"
        elif op == "gray_opening":
            result = hops.gray_opening(img, 5)
            name = "\u7070\u5ea6\u958b\u904b\u7b97 k=5"
        elif op == "gray_closing":
            result = hops.gray_closing(img, 5)
            name = "\u7070\u5ea6\u9589\u904b\u7b97 k=5"
        elif op == "top_hat":
            result = hops.top_hat(img, 9)
            name = "Top-hat k=9"
        elif op == "bottom_hat":
            result = hops.bottom_hat(img, 9)
            name = "Bottom-hat k=9"
        elif op == "rotate_90":
            result = hops.rotate_image(img, 90, "constant")
            name = "\u65cb\u8f49 90\u00b0"
        elif op == "rotate_180":
            result = hops.rotate_image(img, 180, "constant")
            name = "\u65cb\u8f49 180\u00b0"
        elif op == "rotate_270":
            result = hops.rotate_image(img, 270, "constant")
            name = "\u65cb\u8f49 270\u00b0"
        elif op == "mirror_h":
            result = hops.mirror_image(img, "horizontal")
            name = "\u6c34\u5e73\u93e1\u50cf"
        elif op == "mirror_v":
            result = hops.mirror_image(img, "vertical")
            name = "\u5782\u76f4\u93e1\u50cf"
        elif op == "zoom_50":
            result = hops.zoom_image(img, 0.5, 0.5)
            name = "\u7e2e\u653e 50%"
        elif op == "zoom_200":
            result = hops.zoom_image(img, 2.0, 2.0)
            name = "\u7e2e\u653e 200%"
        elif op == "rgb_to_gray":
            result = hops.rgb_to_gray(img)
            name = "\u8f49\u7070\u968e"
        elif op == "rgb_to_hsv":
            result = hops.rgb_to_hsv(img)
            name = "\u8f49 HSV"
        elif op == "rgb_to_hls":
            result = hops.rgb_to_hls(img)
            name = "\u8f49 HLS"
        elif op in ("histogram_eq", "histogram_eq_halcon"):
            result = hops.histogram_eq(img)
            name = "\u76f4\u65b9\u5716\u5747\u8861"
        elif op == "invert_image":
            result = hops.invert_image(img)
            name = "\u53cd\u8272"
        elif op == "illuminate":
            result = hops.illuminate(img, 41, 1.0)
            name = "\u5149\u7167\u6821\u6b63"
        elif op == "abs_image":
            result = hops.abs_image(img)
            name = "\u7d55\u5c0d\u503c"
        elif op == "bright_up":
            result = hops.scale_image(img, 1.0, 30)
            name = "\u4eae\u5ea6 +30"
        elif op == "bright_down":
            result = hops.scale_image(img, 1.0, -30)
            name = "\u4eae\u5ea6 -30"
        elif op == "contrast_up":
            result = hops.scale_image(img, 1.3, 0)
            name = "\u5c0d\u6bd4\u5ea6\u589e\u5f37"
        elif op == "entropy_image":
            result = hops.entropy_image(img, 5)
            name = "\u71b5\u5f71\u50cf k=5"
        elif op == "deviation_image":
            result = hops.deviation_image(img, 5)
            name = "\u6a19\u6e96\u5dee\u5f71\u50cf k=5"
        elif op == "local_min":
            result = hops.local_min(img, 5)
            name = "\u5c40\u90e8\u6700\u5c0f k=5"
        elif op == "local_max":
            result = hops.local_max(img, 5)
            name = "\u5c40\u90e8\u6700\u5927 k=5"
        elif op == "find_barcode":
            results = hops.find_barcode(img)
            if results:
                result = img.copy()
                if result.ndim == 2:
                    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
                for r in results:
                    pts = r.get("points")
                    data = r.get("data", "")
                    if pts is not None:
                        pts_arr = np.array(pts, dtype=np.int32)
                        cv2.polylines(result, [pts_arr], True, (0, 255, 0), 2)
                        cv2.putText(result, data, (pts_arr[0][0], pts_arr[0][1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                name = f"\u689d\u78bc ({len(results)})"
        elif op == "find_qrcode":
            results = hops.find_qrcode(img)
            if results:
                result = img.copy()
                if result.ndim == 2:
                    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
                for r in results:
                    pts = r.get("points")
                    data = r.get("data", "")
                    if pts is not None:
                        pts_arr = np.array(pts, dtype=np.int32)
                        cv2.polylines(result, [pts_arr], True, (0, 0, 255), 2)
                        cv2.putText(result, data[:30], (pts_arr[0][0], pts_arr[0][1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                name = f"QR Code ({len(results)})"
        elif op == "find_datamatrix":
            results = hops.find_datamatrix(img)
            if results:
                result = img.copy()
                if result.ndim == 2:
                    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
                for r in results:
                    pts = r.get("points")
                    if pts is not None:
                        pts_arr = np.array(pts, dtype=np.int32)
                        cv2.polylines(result, [pts_arr], True, (255, 0, 0), 2)
                name = f"DataMatrix ({len(results)})"
        elif op == "skeleton":
            result = hops.skeleton(img)
            name = "\u9aa8\u67b6\u5316"
        elif op == "grab_image":
            result = hops.grab_image(0)
            if result is not None:
                name = "\u76f8\u6a5f\u64f7\u53d6"
            else:
                return ("", None)
        elif op == "abs_diff_image":
            ci = self._pipeline_panel.get_current_index()
            if ci is not None and ci > 0:
                prev_step = self._pipeline_panel.get_step(ci - 1)
                if prev_step is not None:
                    prev = prev_step.array.copy()
                    result = hops.abs_diff_image(img, prev)
                    name = "\u7d55\u5c0d\u5dee\u5206"

        return (name, result)
