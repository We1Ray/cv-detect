"""Image subtraction comparison dialog with rule-based judgment.

Subtracts two images pixel-wise, highlights excess / missing regions,
and lets the user save / load reusable inspection rules (threshold +
area limit) to determine pass or fail.
"""

from __future__ import annotations

import json
import logging
import tkinter as tk
from pathlib import Path
from tkinter import ttk, messagebox, filedialog
from typing import Any, Dict, List, Optional, Tuple

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

# Default location for saved rules
_RULES_DIR = Path.home() / ".dl_anomaly"
_RULES_FILE = _RULES_DIR / "compare_rules.json"


# ------------------------------------------------------------------
# Rule persistence helpers
# ------------------------------------------------------------------

def _load_rules() -> Dict[str, Dict[str, Any]]:
    """Return ``{name: {diff_threshold, area_threshold}}`` from disk."""
    if not _RULES_FILE.exists():
        return {}
    try:
        data = json.loads(_RULES_FILE.read_text(encoding="utf-8"))
        return data.get("rules", {})
    except Exception:
        logger.warning("Failed to read rules file %s", _RULES_FILE)
        return {}


def _save_rules(rules: Dict[str, Dict[str, Any]]) -> None:
    _RULES_DIR.mkdir(parents=True, exist_ok=True)
    _RULES_FILE.write_text(
        json.dumps({"rules": rules}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ------------------------------------------------------------------
# Dialog
# ------------------------------------------------------------------

class CompareDialog(tk.Toplevel):
    """Dialog for image subtraction comparison with rule management.

    Parameters
    ----------
    master : tk.Misc
        Parent widget.
    steps : list[tuple[int, str, np.ndarray]]
        Selectable pipeline image steps as ``(index, name, array)`` triples.
    """

    _CANVAS_W = 720
    _CANVAS_H = 480

    def __init__(
        self,
        master: tk.Misc,
        steps: List[Tuple[int, str, np.ndarray]],
        fetch_steps_cb: Optional[Any] = None,
    ) -> None:
        super().__init__(master)
        self.title("圖片比對（相減）")
        self.configure(bg="#1e1e1e")
        self.transient(master)
        self.resizable(False, False)

        self._steps = steps
        self._names = [f"[{idx}] {name}" for idx, name, _ in steps]
        self._fetch_steps_cb = fetch_steps_cb

        # Cached grayscale arrays
        self._gray_a: Optional[np.ndarray] = None
        self._gray_b: Optional[np.ndarray] = None

        # Tk photo ref (prevent GC)
        self._photo: Optional[ImageTk.PhotoImage] = None

        # Rules cache
        self._rules: Dict[str, Dict[str, Any]] = _load_rules()

        self._build_ui()

        # Default selections: first and last
        self._combo_a.current(0)
        self._combo_b.current(len(steps) - 1)
        self._refresh_rule_combo()
        self._on_selection_changed()

        # Auto-refresh when dialog gains focus
        self.bind("<FocusIn>", lambda _: self._refresh_steps())

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # ── Row 1: image selectors ──
        top = ttk.Frame(self)
        top.pack(fill=tk.X, padx=10, pady=(10, 4))

        ttk.Label(top, text="圖片 A:").pack(side=tk.LEFT, padx=(0, 4))
        self._combo_a = ttk.Combobox(
            top, values=self._names, state="readonly", width=28,
        )
        self._combo_a.pack(side=tk.LEFT, padx=(0, 16))
        self._combo_a.bind("<<ComboboxSelected>>",
                           lambda _: self._on_selection_changed())

        ttk.Label(top, text="圖片 B:").pack(side=tk.LEFT, padx=(0, 4))
        self._combo_b = ttk.Combobox(
            top, values=self._names, state="readonly", width=28,
        )
        self._combo_b.pack(side=tk.LEFT)
        self._combo_b.bind("<<ComboboxSelected>>",
                           lambda _: self._on_selection_changed())

        ttk.Button(top, text="刷新清單",
                   command=self._refresh_steps).pack(side=tk.LEFT, padx=(16, 0))

        # ── Row 2: canvas ──
        self._canvas = tk.Canvas(
            self, width=self._CANVAS_W, height=self._CANVAS_H,
            bg="#2b2b2b", highlightthickness=0,
        )
        self._canvas.pack(padx=10, pady=4)

        # ── Row 3: rule settings ──
        rule_frame = ttk.LabelFrame(self, text="規則設定")
        rule_frame.pack(fill=tk.X, padx=10, pady=4)

        r0 = ttk.Frame(rule_frame)
        r0.pack(fill=tk.X, padx=8, pady=(6, 2))

        ttk.Label(r0, text="規則名稱:").pack(side=tk.LEFT, padx=(0, 4))
        self._rule_name_var = tk.StringVar()
        ttk.Entry(r0, textvariable=self._rule_name_var, width=16).pack(
            side=tk.LEFT, padx=(0, 12))

        ttk.Label(r0, text="差異閾值:").pack(side=tk.LEFT, padx=(0, 4))
        self._diff_thresh_var = tk.IntVar(value=30)
        ttk.Scale(
            r0, from_=1, to=128, variable=self._diff_thresh_var,
            orient=tk.HORIZONTAL, length=160,
            command=lambda _: self._on_threshold_changed(),
        ).pack(side=tk.LEFT, padx=(0, 2))
        self._diff_thresh_label = ttk.Label(r0, text="30", width=4)
        self._diff_thresh_label.pack(side=tk.LEFT, padx=(0, 12))

        ttk.Label(r0, text="面積閾值 (px):").pack(side=tk.LEFT, padx=(0, 4))
        self._area_thresh_var = tk.IntVar(value=500)
        ttk.Entry(r0, textvariable=self._area_thresh_var, width=8).pack(
            side=tk.LEFT, padx=(0, 4))

        # ── Row: ignore small area thresholds ──
        r0b = ttk.Frame(rule_frame)
        r0b.pack(fill=tk.X, padx=8, pady=(2, 2))

        ttk.Label(r0b, text="忽略超出面積 ≤").pack(side=tk.LEFT, padx=(0, 4))
        self._excess_ignore_var = tk.IntVar(value=0)
        ttk.Entry(r0b, textvariable=self._excess_ignore_var, width=8).pack(
            side=tk.LEFT, padx=(0, 4))
        ttk.Label(r0b, text="px").pack(side=tk.LEFT, padx=(0, 16))

        ttk.Label(r0b, text="忽略缺少面積 ≤").pack(side=tk.LEFT, padx=(0, 4))
        self._missing_ignore_var = tk.IntVar(value=0)
        ttk.Entry(r0b, textvariable=self._missing_ignore_var, width=8).pack(
            side=tk.LEFT, padx=(0, 4))
        ttk.Label(r0b, text="px").pack(side=tk.LEFT, padx=(0, 4))

        # Rule buttons row
        r1 = ttk.Frame(rule_frame)
        r1.pack(fill=tk.X, padx=8, pady=(2, 6))

        ttk.Button(r1, text="儲存規則", command=self._save_rule).pack(
            side=tk.LEFT, padx=(0, 6))

        ttk.Label(r1, text="載入規則:").pack(side=tk.LEFT, padx=(0, 4))
        self._rule_combo = ttk.Combobox(r1, state="readonly", width=16)
        self._rule_combo.pack(side=tk.LEFT, padx=(0, 6))
        self._rule_combo.bind("<<ComboboxSelected>>",
                              lambda _: self._load_selected_rule())

        ttk.Button(r1, text="刪除規則", command=self._delete_rule).pack(
            side=tk.LEFT, padx=(0, 6))
        ttk.Button(r1, text="匯出規則…", command=self._export_rules).pack(
            side=tk.LEFT, padx=(0, 6))
        ttk.Button(r1, text="匯入規則…", command=self._import_rules).pack(
            side=tk.LEFT)

        # ── Row 4: results ──
        res = ttk.Frame(self)
        res.pack(fill=tk.X, padx=10, pady=4)

        self._lbl_excess = ttk.Label(res, text="超出面積: —")
        self._lbl_excess.pack(side=tk.LEFT, padx=(0, 20))
        self._lbl_missing = ttk.Label(res, text="缺少面積: —")
        self._lbl_missing.pack(side=tk.LEFT, padx=(0, 20))
        self._lbl_total = ttk.Label(res, text="總差異: —")
        self._lbl_total.pack(side=tk.LEFT, padx=(0, 20))
        self._lbl_result = ttk.Label(
            res, text="", font=(_FONT_FAMILY, 12, "bold"))
        self._lbl_result.pack(side=tk.LEFT)

        # ── Row 5: close ──
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=10, pady=(4, 10))
        ttk.Button(btn_frame, text="重新計算",
                   command=self._recompute).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="關閉",
                   command=self.destroy).pack(side=tk.RIGHT)

    # ------------------------------------------------------------------
    # Refresh steps from pipeline
    # ------------------------------------------------------------------

    def _refresh_steps(self) -> None:
        """Re-fetch steps from the pipeline and update combo boxes."""
        if self._fetch_steps_cb is None:
            return
        new_steps = self._fetch_steps_cb()
        if new_steps == self._steps:
            return

        # Remember current selections by name
        old_a = self._combo_a.get()
        old_b = self._combo_b.get()

        self._steps = new_steps
        self._names = [f"[{idx}] {name}" for idx, name, _ in new_steps]
        self._combo_a["values"] = self._names
        self._combo_b["values"] = self._names

        # Restore previous selections if still available
        if old_a in self._names:
            self._combo_a.set(old_a)
        elif self._names:
            self._combo_a.current(0)
        if old_b in self._names:
            self._combo_b.set(old_b)
        elif self._names:
            self._combo_b.current(len(self._names) - 1)

        self._on_selection_changed()

    # ------------------------------------------------------------------
    # Rule management
    # ------------------------------------------------------------------

    def _refresh_rule_combo(self) -> None:
        names = list(self._rules.keys())
        self._rule_combo["values"] = names
        if names:
            self._rule_combo.set("")

    def _save_rule(self) -> None:
        name = self._rule_name_var.get().strip()
        if not name:
            messagebox.showwarning("警告", "請輸入規則名稱", parent=self)
            return
        self._rules[name] = {
            "diff_threshold": self._diff_thresh_var.get(),
            "area_threshold": self._area_thresh_var.get(),
            "excess_ignore": self._excess_ignore_var.get(),
            "missing_ignore": self._missing_ignore_var.get(),
        }
        _save_rules(self._rules)
        self._refresh_rule_combo()
        messagebox.showinfo("資訊", f"規則「{name}」已儲存", parent=self)

    def _load_selected_rule(self) -> None:
        name = self._rule_combo.get()
        if name not in self._rules:
            return
        r = self._rules[name]
        self._rule_name_var.set(name)
        self._diff_thresh_var.set(r["diff_threshold"])
        self._area_thresh_var.set(r["area_threshold"])
        self._excess_ignore_var.set(r.get("excess_ignore", 0))
        self._missing_ignore_var.set(r.get("missing_ignore", 0))
        self._diff_thresh_label.configure(text=str(r["diff_threshold"]))
        self._recompute()

    def _delete_rule(self) -> None:
        name = self._rule_combo.get()
        if not name or name not in self._rules:
            messagebox.showwarning("警告", "請先從下拉選單選取規則", parent=self)
            return
        if not messagebox.askyesno("確認", f"確定刪除規則「{name}」？",
                                   parent=self):
            return
        del self._rules[name]
        _save_rules(self._rules)
        self._refresh_rule_combo()
        self._rule_name_var.set("")

    def _export_rules(self) -> None:
        path = filedialog.asksaveasfilename(
            parent=self, title="匯出規則",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
            initialfile="compare_rules.json",
        )
        if not path:
            return
        Path(path).write_text(
            json.dumps({"rules": self._rules}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        messagebox.showinfo("資訊", f"已匯出 {len(self._rules)} 條規則",
                            parent=self)

    def _import_rules(self) -> None:
        path = filedialog.askopenfilename(
            parent=self, title="匯入規則",
            filetypes=[("JSON", "*.json"), ("所有檔案", "*")],
        )
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            imported = data.get("rules", {})
        except Exception as exc:
            messagebox.showerror("錯誤", f"讀取失敗:\n{exc}", parent=self)
            return
        self._rules.update(imported)
        _save_rules(self._rules)
        self._refresh_rule_combo()
        messagebox.showinfo("資訊", f"已匯入 {len(imported)} 條規則",
                            parent=self)

    # ------------------------------------------------------------------
    # Image comparison logic
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

    def _on_selection_changed(self) -> None:
        ia = self._combo_a.current()
        ib = self._combo_b.current()
        if ia < 0 or ib < 0:
            return
        self._gray_a = self._to_gray(self._steps[ia][2])
        self._gray_b = self._to_gray(self._steps[ib][2])
        self._recompute()

    def _on_threshold_changed(self) -> None:
        self._diff_thresh_label.configure(
            text=str(self._diff_thresh_var.get()))
        self._recompute()

    def _recompute(self) -> None:
        if self._gray_a is None or self._gray_b is None:
            return

        a, b = self._match_sizes(self._gray_a, self._gray_b)
        diff_thresh = self._diff_thresh_var.get()
        area_thresh = self._area_thresh_var.get()

        diff = b - a
        excess_mask = diff > diff_thresh
        missing_mask = diff < -diff_thresh

        excess_px = int(np.count_nonzero(excess_mask))
        missing_px = int(np.count_nonzero(missing_mask))

        # Apply ignore thresholds for small area differences
        excess_ignore = self._excess_ignore_var.get()
        missing_ignore = self._missing_ignore_var.get()
        adj_excess = 0 if excess_px <= excess_ignore else excess_px
        adj_missing = 0 if missing_px <= missing_ignore else missing_px
        total_px = adj_excess + adj_missing

        excess_text = f"超出面積: {excess_px} px"
        missing_text = f"缺少面積: {missing_px} px"
        if excess_px > 0 and adj_excess == 0:
            excess_text += " (已忽略)"
        if missing_px > 0 and adj_missing == 0:
            missing_text += " (已忽略)"

        self._lbl_excess.configure(text=excess_text)
        self._lbl_missing.configure(text=missing_text)
        self._lbl_total.configure(text=f"總差異: {total_px} px")

        if total_px <= area_thresh:
            self._lbl_result.configure(text="合格", foreground="#4EC94E")
        else:
            self._lbl_result.configure(text="不合格", foreground="#FF4444")

        # Hide color overlay for ignored areas
        draw_excess = excess_mask if adj_excess > 0 else np.zeros_like(excess_mask)
        draw_missing = missing_mask if adj_missing > 0 else np.zeros_like(missing_mask)
        self._draw_diff(a, draw_excess, draw_missing)

    @staticmethod
    def _match_sizes(
        a: np.ndarray, b: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        return a[:h, :w], b[:h, :w]

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
        scale = min(self._CANVAS_W / w, self._CANVAS_H / h, 1.0)
        disp_w = max(1, int(w * scale))
        disp_h = max(1, int(h * scale))
        img = img.resize((disp_w, disp_h), Image.NEAREST)

        ox = (self._CANVAS_W - disp_w) // 2
        oy = (self._CANVAS_H - disp_h) // 2

        self._photo = ImageTk.PhotoImage(img)
        self._canvas.delete("all")
        self._canvas.create_image(ox, oy, anchor=tk.NW, image=self._photo)

        # Legend
        ly = oy + 4
        self._canvas.create_rectangle(
            ox + 4, ly, ox + 18, ly + 14, fill="#FF3C3C", outline="")
        self._canvas.create_text(
            ox + 22, ly + 7, text="超出 (B > A)", fill="white",
            anchor=tk.W, font=(_FONT_FAMILY, 9))
        ly += 18
        self._canvas.create_rectangle(
            ox + 4, ly, ox + 18, ly + 14, fill="#3C64FF", outline="")
        self._canvas.create_text(
            ox + 22, ly + 7, text="缺少 (A > B)", fill="white",
            anchor=tk.W, font=(_FONT_FAMILY, 9))
