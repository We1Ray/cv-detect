"""Pipeline Model dialogs — save, load, and manage .cpmodel files.

Provides:
- ``save_pipeline_model_dialog``: Save current pipeline as a .cpmodel.
- ``load_pipeline_model_dialog``: Load and execute a .cpmodel on current image.
- ``PipelineModelManagerDialog``: Browse / load / delete pipeline models.
"""

from __future__ import annotations

import logging
import os
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Shared dark-theme colours.
_BG = "#2b2b2b"
_FG = "#e0e0e0"
_ENTRY_BG = "#3c3c3c"
_BTN_BG = "#3a3a5c"
_BTN_FG = "#e0e0e0"
_BTN_ACTIVE_BG = "#4a4a6c"
_BTN_ACTIVE_FG = "#ffffff"
_HIGHLIGHT = "#3a3a5c"


def _make_label(parent: tk.Widget, text: str, **kw) -> tk.Label:
    return tk.Label(parent, text=text, bg=_BG, fg=_FG, **kw)


def _make_entry(parent: tk.Widget, var: tk.StringVar, **kw) -> tk.Entry:
    return tk.Entry(
        parent, textvariable=var, bg=_ENTRY_BG, fg=_FG,
        insertbackground=_FG, **kw,
    )


def _make_button(parent: tk.Widget, text: str, command: Callable, **kw) -> tk.Button:
    return tk.Button(
        parent, text=text, command=command,
        bg=_BTN_BG, fg=_BTN_FG,
        activebackground=_BTN_ACTIVE_BG, activeforeground=_BTN_ACTIVE_FG,
        **kw,
    )


# ====================================================================== #
#  Save dialog                                                             #
# ====================================================================== #


def save_pipeline_model_dialog(
    parent: tk.Widget,
    pipeline_panel: Any,
    set_status: Callable[[str], None],
) -> None:
    """Open a dialog to save the current pipeline as a ``.cpmodel``."""
    dlg = tk.Toplevel(parent)
    dlg.title("儲存管線模型")
    dlg.configure(bg=_BG)
    dlg.resizable(False, False)
    dlg.grab_set()

    _make_label(dlg, "將當前管線流程打包為 .cpmodel 檔案",
                font=("", 10, "bold")).grid(
        row=0, column=0, columnspan=2, pady=(10, 8), padx=10)

    # --- Fields ---
    fields = [
        ("模型名稱:", "name", "My Pipeline Model"),
        ("版本:", "version", "1.0.0"),
        ("作者:", "author", ""),
        ("目標產品:", "target_product", ""),
        ("說明:", "description", ""),
    ]
    vars_: dict[str, tk.StringVar] = {}
    for i, (label, key, default) in enumerate(fields, start=1):
        _make_label(dlg, label).grid(row=i, column=0, sticky="e", padx=(10, 4), pady=3)
        var = tk.StringVar(value=default)
        vars_[key] = var
        _make_entry(dlg, var, width=36).grid(row=i, column=1, padx=(0, 10), pady=3)

    def _do_save():
        name = vars_["name"].get().strip()
        if not name:
            messagebox.showwarning("警告", "請輸入模型名稱。", parent=dlg)
            return

        path = filedialog.asksaveasfilename(
            title="儲存管線模型",
            defaultextension=".cpmodel",
            filetypes=[("Pipeline Model", "*.cpmodel")],
            initialfile=f"{name.replace(' ', '_')}.cpmodel",
            parent=dlg,
        )
        if not path:
            return

        dlg.destroy()
        set_status("正在儲存管線模型...")

        def _save():
            from shared.core.pipeline_model import PipelineModel

            # Build recipe from current pipeline panel
            recipe = None
            try:
                from dl_anomaly.core.recipe import Recipe
                recipe = Recipe.from_pipeline(pipeline_panel)
                if not recipe.steps:
                    recipe = None
            except Exception:
                logger.debug("No recipe steps found in pipeline panel.")

            model = PipelineModel.build(
                name=name,
                recipe=recipe,
                author=vars_["author"].get().strip(),
                description=vars_["description"].get().strip(),
                target_product=vars_["target_product"].get().strip(),
                version=vars_["version"].get().strip(),
            )
            model.save(path)
            return path

        def _run():
            try:
                saved_path = _save()
                parent.after(0, lambda: set_status(
                    f"管線模型已儲存: {os.path.basename(saved_path)}"
                ))
                parent.after(0, lambda: messagebox.showinfo(
                    "完成", f"管線模型已儲存至:\n{saved_path}"
                ))
            except Exception as exc:
                logger.exception("Failed to save pipeline model.")
                parent.after(0, lambda: set_status("管線模型儲存失敗"))
                parent.after(0, lambda e=exc: messagebox.showerror(
                    "錯誤", f"儲存管線模型失敗:\n{e}"
                ))

        threading.Thread(target=_run, daemon=True).start()

    btn_frame = tk.Frame(dlg, bg=_BG)
    btn_frame.grid(row=len(fields) + 1, column=0, columnspan=2, pady=(8, 10))
    _make_button(btn_frame, "儲存", _do_save, width=10).pack(side="left", padx=4)
    _make_button(btn_frame, "取消", dlg.destroy, width=10).pack(side="left", padx=4)

    dlg.update_idletasks()
    x = parent.winfo_x() + (parent.winfo_width() - dlg.winfo_width()) // 2
    y = parent.winfo_y() + (parent.winfo_height() - dlg.winfo_height()) // 2
    dlg.geometry(f"+{x}+{y}")


# ====================================================================== #
#  Load dialog                                                             #
# ====================================================================== #


def load_pipeline_model_dialog(
    parent: tk.Widget,
    get_current_image: Callable,
    add_pipeline_step: Callable,
    set_status: Callable[[str], None],
) -> None:
    """Open file dialog to load a ``.cpmodel`` and optionally execute it."""
    path = filedialog.askopenfilename(
        title="載入管線模型",
        filetypes=[("Pipeline Model", "*.cpmodel"), ("All files", "*")],
        parent=parent,
    )
    if not path:
        return

    set_status(f"載入管線模型: {os.path.basename(path)}...")

    def _load_and_show():
        try:
            from shared.core.pipeline_model import PipelineModel
            model = PipelineModel.load(path)
            info = model.info()

            def _show_info():
                _show_model_info_and_execute(
                    parent, model, info,
                    get_current_image, add_pipeline_step, set_status,
                )
            parent.after(0, _show_info)

        except Exception as exc:
            logger.exception("Failed to load pipeline model.")
            parent.after(0, lambda: set_status("管線模型載入失敗"))
            parent.after(0, lambda e=exc: messagebox.showerror(
                "錯誤", f"載入管線模型失敗:\n{e}"
            ))

    threading.Thread(target=_load_and_show, daemon=True).start()


def _show_model_info_and_execute(
    parent: tk.Widget,
    model: Any,
    info: dict,
    get_current_image: Callable,
    add_pipeline_step: Callable,
    set_status: Callable[[str], None],
) -> None:
    """Show model info dialog with option to execute on current image."""
    dlg = tk.Toplevel(parent)
    dlg.title(f"管線模型: {info.get('name', '?')}")
    dlg.configure(bg=_BG)
    dlg.resizable(False, False)
    dlg.grab_set()

    _make_label(dlg, "管線模型資訊", font=("", 11, "bold")).grid(
        row=0, column=0, columnspan=2, pady=(10, 6), padx=10)

    display_fields = [
        ("名稱", info.get("name", "")),
        ("版本", info.get("version", "")),
        ("作者", info.get("author", "")),
        ("目標產品", info.get("target_product", "")),
        ("說明", info.get("description", "")),
        ("建立時間", info.get("created_at", "")[:19]),
        ("含前處理", "是" if info.get("has_recipe") else "否"),
        ("含檢測流程", "是" if info.get("has_flow") else "否"),
    ]
    if "num_steps" in info:
        display_fields.append(("流程步驟數", str(info["num_steps"])))
    if "step_names" in info:
        display_fields.append(("步驟", " → ".join(info["step_names"])))

    for i, (label, value) in enumerate(display_fields, start=1):
        _make_label(dlg, f"{label}:").grid(
            row=i, column=0, sticky="ne", padx=(10, 4), pady=2)
        _make_label(dlg, value, wraplength=300, justify="left").grid(
            row=i, column=1, sticky="nw", padx=(0, 10), pady=2)

    # Validate
    warnings = model.validate()
    if warnings:
        warn_text = "\n".join(f"  - {w}" for w in warnings[:5])
        _make_label(dlg, f"警告:\n{warn_text}", fg="#ffaa00",
                    wraplength=350, justify="left").grid(
            row=len(display_fields) + 1, column=0, columnspan=2,
            padx=10, pady=(6, 2))

    def _execute():
        img = get_current_image()
        if img is None:
            messagebox.showwarning("警告", "請先載入圖片再執行管線模型。", parent=dlg)
            return
        dlg.destroy()
        set_status("正在執行管線模型...")

        def _run():
            try:
                result = model.execute(img)
                verdict = "PASS" if result.overall_pass else "FAIL"
                msg = (
                    f"管線模型執行完成: {verdict} "
                    f"({result.total_time_ms:.0f} ms, "
                    f"{len(result.steps)} 步驟)"
                )

                def _update_ui():
                    set_status(msg)
                    # Add step visualisations to pipeline panel
                    for sr in result.steps:
                        if sr.image is not None:
                            add_pipeline_step(
                                f"[{sr.step_type}] {sr.step_name}",
                                sr.image,
                            )
                    messagebox.showinfo("管線模型結果", msg)

                parent.after(0, _update_ui)
            except Exception as exc:
                logger.exception("Pipeline model execution failed.")
                parent.after(0, lambda: set_status("管線模型執行失敗"))
                parent.after(0, lambda e=exc: messagebox.showerror(
                    "錯誤", f"執行失敗:\n{e}"
                ))
            finally:
                try:
                    model.close()
                except Exception:
                    pass

        threading.Thread(target=_run, daemon=True).start()

    def _close():
        try:
            model.close()
        except Exception:
            pass
        dlg.destroy()

    btn_frame = tk.Frame(dlg, bg=_BG)
    btn_frame.grid(
        row=len(display_fields) + 2, column=0, columnspan=2, pady=(8, 10))
    _make_button(btn_frame, "執行", _execute, width=10).pack(side="left", padx=4)
    _make_button(btn_frame, "關閉", _close, width=10).pack(side="left", padx=4)

    dlg.update_idletasks()
    x = parent.winfo_x() + (parent.winfo_width() - dlg.winfo_width()) // 2
    y = parent.winfo_y() + (parent.winfo_height() - dlg.winfo_height()) // 2
    dlg.geometry(f"+{x}+{y}")

    set_status(f"已載入管線模型: {info.get('name', '?')}")


# ====================================================================== #
#  Manager dialog                                                          #
# ====================================================================== #


class PipelineModelManagerDialog(tk.Toplevel):
    """Browse, load, and delete pipeline models from a registry directory.

    Parameters
    ----------
    parent:
        Parent widget.
    get_current_image:
        Callback returning the current image array.
    add_pipeline_step:
        Callback to add a step to the pipeline panel.
    set_status:
        Callback to set status bar text.
    """

    def __init__(
        self,
        parent: tk.Widget,
        get_current_image: Callable,
        add_pipeline_step: Callable,
        set_status: Callable[[str], None],
    ) -> None:
        super().__init__(parent)
        self.title("管線模型管理")
        self.configure(bg=_BG)
        self.geometry("640x480")

        self._parent = parent
        self._get_current_image = get_current_image
        self._add_pipeline_step = add_pipeline_step
        self._set_status = set_status
        self._registry = None
        self._models_list: list[dict] = []

        self._build_ui()
        self._init_registry()

    def _build_ui(self) -> None:
        # Top bar — directory selector
        top = tk.Frame(self, bg=_BG)
        top.pack(fill="x", padx=8, pady=(8, 4))

        _make_label(top, "模型目錄:").pack(side="left")
        self._dir_var = tk.StringVar(value=str(Path.home() / ".cv-detect" / "pipeline_models"))
        _make_entry(top, self._dir_var, width=40).pack(side="left", padx=4)
        _make_button(top, "瀏覽", self._browse_dir, width=6).pack(side="left", padx=2)
        _make_button(top, "重新整理", self._refresh, width=8).pack(side="left", padx=2)

        # Model list (Treeview)
        tree_frame = tk.Frame(self, bg=_BG)
        tree_frame.pack(fill="both", expand=True, padx=8, pady=4)

        columns = ("name", "version", "author", "steps", "size", "created")
        self._tree = ttk.Treeview(
            tree_frame, columns=columns, show="headings", height=12,
        )
        headers = {
            "name": ("名稱", 140),
            "version": ("版本", 60),
            "author": ("作者", 80),
            "steps": ("步驟", 40),
            "size": ("大小", 60),
            "created": ("建立時間", 140),
        }
        for col, (heading, width) in headers.items():
            self._tree.heading(col, text=heading)
            self._tree.column(col, width=width, minwidth=30)

        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=scrollbar.set)
        self._tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Info label
        self._info_var = tk.StringVar(value="選擇一個模型查看詳情")
        _make_label(self, textvariable=self._info_var, wraplength=600,
                    justify="left").pack(fill="x", padx=8, pady=2)

        # Buttons
        btn_frame = tk.Frame(self, bg=_BG)
        btn_frame.pack(fill="x", padx=8, pady=(4, 8))

        _make_button(btn_frame, "載入執行", self._on_load, width=10).pack(side="left", padx=4)
        _make_button(btn_frame, "刪除", self._on_delete, width=8).pack(side="left", padx=4)
        _make_button(btn_frame, "匯入...", self._on_import, width=8).pack(side="left", padx=4)
        _make_button(btn_frame, "關閉", self.destroy, width=8).pack(side="right", padx=4)

        # Bind selection
        self._tree.bind("<<TreeviewSelect>>", self._on_select)

    def _browse_dir(self) -> None:
        d = filedialog.askdirectory(
            title="選擇管線模型目錄", parent=self,
        )
        if d:
            self._dir_var.set(d)
            self._init_registry()

    def _init_registry(self) -> None:
        from shared.core.pipeline_model import PipelineModelRegistry

        reg_dir = self._dir_var.get()
        self._registry = PipelineModelRegistry(reg_dir)
        self._refresh()

    def _refresh(self) -> None:
        if self._registry is None:
            return

        self._registry.scan()
        self._models_list = self._registry.list_models()

        # Clear tree
        for item in self._tree.get_children():
            self._tree.delete(item)

        for entry in self._models_list:
            meta = entry.get("metadata", {})
            self._tree.insert("", "end", values=(
                meta.get("name", "?"),
                meta.get("version", "?"),
                meta.get("author", ""),
                "",  # steps not available from manifest alone
                f"{entry.get('file_size_mb', 0):.1f} MB",
                meta.get("created_at", "")[:19],
            ))

        self._info_var.set(f"共 {len(self._models_list)} 個管線模型")

    def _get_selected_entry(self) -> Optional[dict]:
        sel = self._tree.selection()
        if not sel:
            return None
        idx = self._tree.index(sel[0])
        if 0 <= idx < len(self._models_list):
            return self._models_list[idx]
        return None

    def _on_select(self, _event: Any = None) -> None:
        entry = self._get_selected_entry()
        if entry is None:
            return
        meta = entry.get("metadata", {})
        desc = meta.get("description", "")
        product = meta.get("target_product", "")
        parts = [f"名稱: {meta.get('name', '?')}"]
        if product:
            parts.append(f"目標產品: {product}")
        if desc:
            parts.append(f"說明: {desc}")
        self._info_var.set(" | ".join(parts))

    def _on_load(self) -> None:
        entry = self._get_selected_entry()
        if entry is None:
            messagebox.showwarning("警告", "請先選擇一個模型。", parent=self)
            return

        filename = entry.get("filename", "")
        if not filename:
            return

        self._set_status(f"載入管線模型: {filename}...")
        self.destroy()

        def _load():
            try:
                from shared.core.pipeline_model import PipelineModel
                model = PipelineModel.load(entry["path"])
                info = model.info()

                def _show():
                    _show_model_info_and_execute(
                        self._parent, model, info,
                        self._get_current_image,
                        self._add_pipeline_step,
                        self._set_status,
                    )
                self._parent.after(0, _show)
            except Exception as exc:
                logger.exception("Failed to load pipeline model from registry.")
                self._parent.after(0, lambda: self._set_status("載入失敗"))
                self._parent.after(0, lambda e=exc: messagebox.showerror(
                    "錯誤", f"載入管線模型失敗:\n{e}"
                ))

        threading.Thread(target=_load, daemon=True).start()

    def _on_delete(self) -> None:
        entry = self._get_selected_entry()
        if entry is None:
            messagebox.showwarning("警告", "請先選擇一個模型。", parent=self)
            return

        filename = entry.get("filename", "")
        name = entry.get("metadata", {}).get("name", filename)
        if not messagebox.askyesno(
            "確認刪除", f"確定要刪除管線模型 '{name}'？", parent=self
        ):
            return

        try:
            self._registry.delete_model(filename)
            self._refresh()
            self._set_status(f"已刪除管線模型: {name}")
        except Exception as exc:
            messagebox.showerror("錯誤", f"刪除失敗:\n{exc}", parent=self)

    def _on_import(self) -> None:
        """Import a .cpmodel file into the registry directory."""
        path = filedialog.askopenfilename(
            title="匯入管線模型",
            filetypes=[("Pipeline Model", "*.cpmodel")],
            parent=self,
        )
        if not path:
            return

        try:
            from shared.core.pipeline_model import PipelineModel
            import shutil

            dest = Path(self._dir_var.get()) / os.path.basename(path)
            if dest.exists():
                if not messagebox.askyesno(
                    "覆蓋確認", f"'{dest.name}' 已存在，是否覆蓋？", parent=self
                ):
                    return
            shutil.copy2(path, dest)
            self._refresh()
            self._set_status(f"已匯入: {dest.name}")
        except Exception as exc:
            messagebox.showerror("錯誤", f"匯入失敗:\n{exc}", parent=self)
