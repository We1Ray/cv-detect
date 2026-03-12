"""Dialog for applying a recipe to multiple selected images.

Opens a window where the user can pick a recipe JSON file, select
multiple images, and apply the recipe to each one.  Only the final
step result is added to the pipeline.
"""

from __future__ import annotations

import logging
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk, filedialog, messagebox
from typing import Any, Callable, List, Optional

import numpy as np
from PIL import Image as PILImage

logger = logging.getLogger(__name__)


class RecipeApplyDialog(tk.Toplevel):
    """Toplevel dialog for applying a recipe to a batch of images.

    Parameters
    ----------
    master : tk.Misc
        Parent widget.
    add_step_cb : callable(name, array, region=None)
        Callback to add a result step to the pipeline.
    set_status_cb : callable(str)
        Callback to set the status bar text.
    """

    def __init__(
        self,
        master: tk.Misc,
        add_step_cb: Callable,
        set_status_cb: Callable[[str], None],
    ) -> None:
        super().__init__(master)
        self.title("套用流程至圖片")
        self.configure(bg="#1e1e1e")
        self.transient(master)
        self.resizable(False, False)

        self._add_step_cb = add_step_cb
        self._set_status_cb = set_status_cb

        self._recipe_path: Optional[str] = None
        self._image_paths: List[str] = []
        self._running = False

        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        pad = dict(padx=10, pady=4)

        # ── Row 1: Recipe file ──
        r1 = ttk.Frame(self)
        r1.pack(fill=tk.X, **pad)
        ttk.Label(r1, text="流程檔案:").pack(side=tk.LEFT, padx=(0, 4))
        self._recipe_var = tk.StringVar(value="（未選擇）")
        ttk.Label(r1, textvariable=self._recipe_var, width=50,
                  relief="sunken").pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(r1, text="瀏覽...", command=self._browse_recipe).pack(
            side=tk.LEFT)

        # ── Row 2: Image selection buttons ──
        r2 = ttk.Frame(self)
        r2.pack(fill=tk.X, **pad)
        ttk.Button(r2, text="選擇圖片...", command=self._browse_images).pack(
            side=tk.LEFT, padx=(0, 6))
        ttk.Button(r2, text="清除", command=self._clear_images).pack(
            side=tk.LEFT, padx=(0, 12))
        self._count_var = tk.StringVar(value="已選: 0 張")
        ttk.Label(r2, textvariable=self._count_var).pack(side=tk.LEFT)

        # ── Row 3: Image list ──
        list_frame = ttk.Frame(self)
        list_frame.pack(fill=tk.BOTH, expand=True, **pad)
        self._listbox = tk.Listbox(
            list_frame, height=10, width=70, bg="#2b2b2b", fg="#e0e0e0",
            selectbackground="#3a3a5c", selectforeground="#ffffff",
        )
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL,
                                  command=self._listbox.yview)
        self._listbox.configure(yscrollcommand=scrollbar.set)
        self._listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # ── Row 4: Progress ──
        r4 = ttk.Frame(self)
        r4.pack(fill=tk.X, **pad)
        self._progress = ttk.Progressbar(r4, mode="determinate", length=400)
        self._progress.pack(side=tk.LEFT, padx=(0, 8))
        self._status_var = tk.StringVar(value="")
        ttk.Label(r4, textvariable=self._status_var).pack(side=tk.LEFT)

        # ── Row 5: Action buttons ──
        r5 = ttk.Frame(self)
        r5.pack(fill=tk.X, padx=10, pady=(4, 10))
        self._btn_run = ttk.Button(r5, text="執行", command=self._run)
        self._btn_run.pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(r5, text="關閉", command=self.destroy).pack(side=tk.RIGHT)

    # ------------------------------------------------------------------
    # Browse helpers
    # ------------------------------------------------------------------

    def _browse_recipe(self) -> None:
        path = filedialog.askopenfilename(
            parent=self, title="選擇流程檔案",
            filetypes=[("JSON", "*.json"), ("所有檔案", "*.*")],
        )
        if path:
            self._recipe_path = path
            self._recipe_var.set(Path(path).name)

    def _browse_images(self) -> None:
        paths = filedialog.askopenfilenames(
            parent=self, title="選擇圖片（可複選）",
            filetypes=[
                ("圖片", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"),
                ("所有檔案", "*.*"),
            ],
        )
        if paths:
            self._image_paths = list(paths)
            self._listbox.delete(0, tk.END)
            for p in self._image_paths:
                self._listbox.insert(tk.END, Path(p).name)
            self._count_var.set(f"已選: {len(self._image_paths)} 張")

    def _clear_images(self) -> None:
        self._image_paths.clear()
        self._listbox.delete(0, tk.END)
        self._count_var.set("已選: 0 張")

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

    def _run(self) -> None:
        if self._running:
            return
        if not self._recipe_path:
            messagebox.showwarning("警告", "請先選擇流程檔案", parent=self)
            return
        if not self._image_paths:
            messagebox.showwarning("警告", "請先選擇圖片", parent=self)
            return

        from dl_anomaly.core.recipe import Recipe, replay_recipe

        try:
            recipe = Recipe.load(self._recipe_path)
        except Exception as exc:
            messagebox.showerror("錯誤", f"無法載入流程:\n{exc}", parent=self)
            return

        self._running = True
        self._btn_run.configure(state="disabled")
        total = len(self._image_paths)
        self._progress["maximum"] = total
        self._progress["value"] = 0

        paths = list(self._image_paths)

        def _worker():
            MAX_SIDE = 1080
            results = []  # list of (filename, array, region)
            for i, img_path in enumerate(paths):
                try:
                    pil = PILImage.open(img_path)
                    w, h = pil.size
                    if max(w, h) > MAX_SIDE:
                        scale = MAX_SIDE / max(w, h)
                        pil = pil.resize(
                            (int(w * scale), int(h * scale)), PILImage.LANCZOS)
                    if pil.mode == "L":
                        arr = np.array(pil)
                    else:
                        arr = np.array(pil.convert("RGB"))

                    step_results = replay_recipe(recipe, arr)
                    if step_results:
                        _name, final_arr, region = step_results[-1]
                        fname = Path(img_path).stem
                        results.append((f"原圖: {fname}", final_arr, region))
                except Exception:
                    logger.exception("RecipeApply: failed on %s", img_path)

                # Report progress on the main thread
                self.after(0, self._update_progress, i + 1, total)

            self.after(0, self._on_done, results)

        threading.Thread(target=_worker, daemon=True).start()

    def _update_progress(self, current: int, total: int) -> None:
        self._progress["value"] = current
        self._status_var.set(f"處理中... {current}/{total}")

    def _on_done(self, results: list) -> None:
        self._running = False
        self._btn_run.configure(state="normal")
        self._status_var.set(f"完成: {len(results)} 張")
        self._progress["value"] = self._progress["maximum"]

        for name, arr, region in results:
            self._add_step_cb(name, arr, region=region)

        self._set_status_cb(f"流程套用完成: {len(results)} 張圖片")
        messagebox.showinfo(
            "完成",
            f"已將 {len(results)} 張圖片的最終結果加入 pipeline。",
            parent=self,
        )
