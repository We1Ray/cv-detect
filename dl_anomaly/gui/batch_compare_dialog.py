"""Batch image comparison dialog (1-to-N).

Compares a single reference image against multiple target images using
the same subtraction logic as :class:`CompareDialog`, with support for
the ignore-small-area mechanism.
"""

from __future__ import annotations

import logging
import tkinter as tk
from dataclasses import dataclass, field
from tkinter import ttk, messagebox
from typing import Any, Callable, Dict, List, Optional, Tuple

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
    _FONT_FAMILY = _FONT_FAMILY
    _MONO_FAMILY = "Consolas"


@dataclass
class _CompareResult:
    """Cache one comparison result."""
    name: str
    excess_px: int
    missing_px: int
    adj_excess: int
    adj_missing: int
    total_px: int
    passed: bool
    excess_mask: np.ndarray
    missing_mask: np.ndarray
    base_gray: np.ndarray


class BatchCompareDialog(tk.Toplevel):
    """Dialog for 1-to-N image comparison with rule parameters.

    Parameters
    ----------
    master : tk.Misc
        Parent widget.
    steps : list[tuple[int, str, np.ndarray]]
        Pipeline steps as ``(index, name, array)`` triples.
    fetch_steps_cb : callable
        Callback to re-fetch steps from the pipeline.
    """

    _CANVAS_W = 560
    _CANVAS_H = 400

    def __init__(
        self,
        master: tk.Misc,
        steps: List[Tuple[int, str, np.ndarray]],
        fetch_steps_cb: Optional[Callable] = None,
    ) -> None:
        super().__init__(master)
        self.title("批次圖片比對（1 對 N）")
        self.configure(bg="#1e1e1e")
        self.transient(master)
        self.resizable(True, True)
        self.geometry("1050x720")

        self._steps = steps
        self._names = [f"[{idx}] {name}" for idx, name, _ in steps]
        self._fetch_steps_cb = fetch_steps_cb

        self._photo: Optional[ImageTk.PhotoImage] = None
        self._results: List[_CompareResult] = []

        # Checkbox vars for target selection
        self._check_vars: List[tk.BooleanVar] = []

        self._build_ui()

        if self._names:
            self._combo_ref.current(0)
        self._refresh_targets()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # ── Top: reference + rule selectors ──
        top = ttk.Frame(self)
        top.pack(fill=tk.X, padx=10, pady=(10, 4))

        ttk.Label(top, text="基準圖片:").pack(side=tk.LEFT, padx=(0, 4))
        self._combo_ref = ttk.Combobox(
            top, values=self._names, state="readonly", width=28)
        self._combo_ref.pack(side=tk.LEFT, padx=(0, 12))

        ttk.Button(top, text="刷新清單",
                   command=self._refresh_steps).pack(side=tk.LEFT, padx=(0, 12))

        # ── Rule parameters row ──
        rp = ttk.LabelFrame(self, text="比對參數")
        rp.pack(fill=tk.X, padx=10, pady=4)

        r0 = ttk.Frame(rp)
        r0.pack(fill=tk.X, padx=8, pady=(6, 2))

        ttk.Label(r0, text="差異閾值:").pack(side=tk.LEFT, padx=(0, 4))
        self._diff_thresh_var = tk.IntVar(value=30)
        ttk.Scale(
            r0, from_=1, to=128, variable=self._diff_thresh_var,
            orient=tk.HORIZONTAL, length=120,
            command=lambda _: self._diff_thresh_label.configure(
                text=str(self._diff_thresh_var.get())),
        ).pack(side=tk.LEFT, padx=(0, 2))
        self._diff_thresh_label = ttk.Label(r0, text="30", width=4)
        self._diff_thresh_label.pack(side=tk.LEFT, padx=(0, 12))

        ttk.Label(r0, text="面積閾值 (px):").pack(side=tk.LEFT, padx=(0, 4))
        self._area_thresh_var = tk.IntVar(value=500)
        ttk.Entry(r0, textvariable=self._area_thresh_var, width=8).pack(
            side=tk.LEFT, padx=(0, 12))

        ttk.Label(r0, text="忽略超出 ≤").pack(side=tk.LEFT, padx=(0, 4))
        self._excess_ignore_var = tk.IntVar(value=0)
        ttk.Entry(r0, textvariable=self._excess_ignore_var, width=8).pack(
            side=tk.LEFT, padx=(0, 4))
        ttk.Label(r0, text="px").pack(side=tk.LEFT, padx=(0, 12))

        ttk.Label(r0, text="忽略缺少 ≤").pack(side=tk.LEFT, padx=(0, 4))
        self._missing_ignore_var = tk.IntVar(value=0)
        ttk.Entry(r0, textvariable=self._missing_ignore_var, width=8).pack(
            side=tk.LEFT, padx=(0, 4))
        ttk.Label(r0, text="px").pack(side=tk.LEFT)

        # ── Middle: PanedWindow (target list | preview canvas) ──
        pw = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        pw.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

        # Left: checkbox target list
        left_frame = ttk.Frame(pw)
        pw.add(left_frame, weight=1)

        btn_row = ttk.Frame(left_frame)
        btn_row.pack(fill=tk.X, padx=4, pady=4)
        ttk.Button(btn_row, text="全選",
                   command=self._select_all).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(btn_row, text="取消全選",
                   command=self._deselect_all).pack(side=tk.LEFT)

        list_container = ttk.Frame(left_frame)
        list_container.pack(fill=tk.BOTH, expand=True, padx=4)
        self._target_canvas = tk.Canvas(
            list_container, bg="#2b2b2b", highlightthickness=0)
        target_scrollbar = ttk.Scrollbar(
            list_container, orient=tk.VERTICAL,
            command=self._target_canvas.yview)
        self._target_inner = ttk.Frame(self._target_canvas)
        self._target_inner.bind(
            "<Configure>",
            lambda e: self._target_canvas.configure(
                scrollregion=self._target_canvas.bbox("all")))
        self._target_canvas.create_window(
            (0, 0), window=self._target_inner, anchor="nw")
        self._target_canvas.configure(yscrollcommand=target_scrollbar.set)
        self._target_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        target_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Right: diff preview canvas
        right_frame = ttk.Frame(pw)
        pw.add(right_frame, weight=2)
        self._preview = tk.Canvas(
            right_frame, width=self._CANVAS_W, height=self._CANVAS_H,
            bg="#2b2b2b", highlightthickness=0)
        self._preview.pack(fill=tk.BOTH, expand=True)

        # ── Action row ──
        action = ttk.Frame(self)
        action.pack(fill=tk.X, padx=10, pady=4)
        ttk.Button(action, text="全部比對",
                   command=self._run_all).pack(side=tk.LEFT, padx=(0, 12))
        self._summary_var = tk.StringVar(value="")
        ttk.Label(action, textvariable=self._summary_var,
                  font=(_FONT_FAMILY, 10, "bold")).pack(side=tk.LEFT)

        # ── Result Treeview ──
        tree_frame = ttk.Frame(self)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(4, 10))

        cols = ("name", "excess", "missing", "total", "verdict")
        self._tree = ttk.Treeview(
            tree_frame, columns=cols, show="headings", height=8)
        self._tree.heading("name", text="圖片名稱")
        self._tree.heading("excess", text="超出面積")
        self._tree.heading("missing", text="缺少面積")
        self._tree.heading("total", text="總差異")
        self._tree.heading("verdict", text="判定")
        self._tree.column("name", width=220)
        self._tree.column("excess", width=100, anchor="center")
        self._tree.column("missing", width=100, anchor="center")
        self._tree.column("total", width=100, anchor="center")
        self._tree.column("verdict", width=80, anchor="center")

        # Row tags for coloring
        self._tree.tag_configure("pass", foreground="#4EC94E")
        self._tree.tag_configure("fail", foreground="#FF4444")

        tree_scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL,
                                    command=self._tree.yview)
        self._tree.configure(yscrollcommand=tree_scroll.set)
        self._tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self._tree.bind("<<TreeviewSelect>>", self._on_tree_select)

    # ------------------------------------------------------------------
    # Refresh helpers
    # ------------------------------------------------------------------

    def _refresh_steps(self) -> None:
        if self._fetch_steps_cb is None:
            return
        new_steps = self._fetch_steps_cb()
        if new_steps == self._steps:
            return
        self._steps = new_steps
        self._names = [f"[{idx}] {name}" for idx, name, _ in new_steps]
        old_ref = self._combo_ref.get()
        self._combo_ref["values"] = self._names
        if old_ref in self._names:
            self._combo_ref.set(old_ref)
        elif self._names:
            self._combo_ref.current(0)
        self._refresh_targets()

    def _refresh_targets(self) -> None:
        """Rebuild the checkbox list for target images."""
        for w in self._target_inner.winfo_children():
            w.destroy()
        self._check_vars.clear()

        for i, name in enumerate(self._names):
            var = tk.BooleanVar(value=True)
            self._check_vars.append(var)
            cb = ttk.Checkbutton(self._target_inner, text=name, variable=var)
            cb.pack(anchor="w", padx=4, pady=1)

    def _select_all(self) -> None:
        for v in self._check_vars:
            v.set(True)

    def _deselect_all(self) -> None:
        for v in self._check_vars:
            v.set(False)

    # ------------------------------------------------------------------
    # Comparison logic (same as CompareDialog)
    # ------------------------------------------------------------------

    @staticmethod
    def _to_gray(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 2:
            return arr.astype(np.float64)
        if arr.ndim == 3 and arr.shape[2] == 1:
            return arr[:, :, 0].astype(np.float64)
        return (
            0.2989 * arr[:, :, 0].astype(np.float64)
            + 0.5870 * arr[:, :, 1].astype(np.float64)
            + 0.1140 * arr[:, :, 2].astype(np.float64)
        )

    @staticmethod
    def _match_sizes(
        a: np.ndarray, b: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        return a[:h, :w], b[:h, :w]

    def _compare_one(
        self,
        ref_gray: np.ndarray,
        target_gray: np.ndarray,
        target_name: str,
    ) -> _CompareResult:
        diff_thresh = self._diff_thresh_var.get()
        area_thresh = self._area_thresh_var.get()
        excess_ignore = self._excess_ignore_var.get()
        missing_ignore = self._missing_ignore_var.get()

        a, b = self._match_sizes(ref_gray, target_gray)
        diff = b - a
        excess_mask = diff > diff_thresh
        missing_mask = diff < -diff_thresh

        excess_px = int(np.count_nonzero(excess_mask))
        missing_px = int(np.count_nonzero(missing_mask))

        adj_excess = 0 if excess_px <= excess_ignore else excess_px
        adj_missing = 0 if missing_px <= missing_ignore else missing_px
        total_px = adj_excess + adj_missing
        passed = total_px <= area_thresh

        # Hide color overlay for ignored areas
        draw_excess = excess_mask if adj_excess > 0 else np.zeros_like(excess_mask)
        draw_missing = missing_mask if adj_missing > 0 else np.zeros_like(missing_mask)

        return _CompareResult(
            name=target_name,
            excess_px=excess_px,
            missing_px=missing_px,
            adj_excess=adj_excess,
            adj_missing=adj_missing,
            total_px=total_px,
            passed=passed,
            excess_mask=draw_excess,
            missing_mask=draw_missing,
            base_gray=a,
        )

    # ------------------------------------------------------------------
    # Run all comparisons
    # ------------------------------------------------------------------

    def _run_all(self) -> None:
        ref_idx = self._combo_ref.current()
        if ref_idx < 0:
            messagebox.showwarning("警告", "請選擇基準圖片", parent=self)
            return

        ref_gray = self._to_gray(self._steps[ref_idx][2])
        self._results.clear()

        # Clear the treeview
        for item in self._tree.get_children():
            self._tree.delete(item)

        selected_any = False
        for i, (idx, name, arr) in enumerate(self._steps):
            if i == ref_idx:
                continue
            if not self._check_vars[i].get():
                continue
            selected_any = True
            target_gray = self._to_gray(arr)
            result = self._compare_one(ref_gray, target_gray, self._names[i])
            self._results.append(result)

        if not selected_any:
            messagebox.showwarning("警告", "請至少勾選一張比對目標圖片",
                                   parent=self)
            return

        # Populate treeview
        pass_count = 0
        for r in self._results:
            tag = "pass" if r.passed else "fail"
            verdict = "合格" if r.passed else "不合格"

            excess_text = str(r.excess_px)
            if r.excess_px > 0 and r.adj_excess == 0:
                excess_text += " (忽略)"
            missing_text = str(r.missing_px)
            if r.missing_px > 0 and r.adj_missing == 0:
                missing_text += " (忽略)"

            self._tree.insert(
                "", tk.END,
                values=(r.name, excess_text, missing_text,
                        str(r.total_px), verdict),
                tags=(tag,),
            )
            if r.passed:
                pass_count += 1

        total = len(self._results)
        fail_count = total - pass_count
        self._summary_var.set(
            f"合格: {pass_count} / {total}　不合格: {fail_count}")

    # ------------------------------------------------------------------
    # Tree selection → preview
    # ------------------------------------------------------------------

    def _on_tree_select(self, _event: Any = None) -> None:
        sel = self._tree.selection()
        if not sel:
            return
        item = sel[0]
        # Determine index within results by treeview row order
        row_idx = self._tree.index(item)
        if row_idx >= len(self._results):
            return
        result = self._results[row_idx]
        self._draw_diff(result.base_gray, result.excess_mask,
                        result.missing_mask)

    def _draw_diff(
        self,
        base_gray: np.ndarray,
        excess_mask: np.ndarray,
        missing_mask: np.ndarray,
    ) -> None:
        h, w = base_gray.shape[:2]
        gray_u8 = np.clip(base_gray, 0, 255).astype(np.uint8)
        rgb = np.stack([gray_u8, gray_u8, gray_u8], axis=-1)

        rgb[excess_mask] = [255, 60, 60]
        rgb[missing_mask] = [60, 100, 255]

        img = Image.fromarray(rgb, mode="RGB")
        canvas_w = self._preview.winfo_width() or self._CANVAS_W
        canvas_h = self._preview.winfo_height() or self._CANVAS_H
        scale = min(canvas_w / w, canvas_h / h, 1.0)
        disp_w = max(1, int(w * scale))
        disp_h = max(1, int(h * scale))
        img = img.resize((disp_w, disp_h), Image.NEAREST)

        ox = (canvas_w - disp_w) // 2
        oy = (canvas_h - disp_h) // 2

        self._photo = ImageTk.PhotoImage(img)
        self._preview.delete("all")
        self._preview.create_image(ox, oy, anchor=tk.NW, image=self._photo)

        # Legend
        ly = oy + 4
        self._preview.create_rectangle(
            ox + 4, ly, ox + 18, ly + 14, fill="#FF3C3C", outline="")
        self._preview.create_text(
            ox + 22, ly + 7, text="超出 (B > A)", fill="white",
            anchor=tk.W, font=(_FONT_FAMILY, 9))
        ly += 18
        self._preview.create_rectangle(
            ox + 4, ly, ox + 18, ly + 14, fill="#3C64FF", outline="")
        self._preview.create_text(
            ox + 22, ly + 7, text="缺少 (A > B)", fill="white",
            anchor=tk.W, font=(_FONT_FAMILY, 9))
