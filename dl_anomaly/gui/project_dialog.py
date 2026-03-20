"""Project management dialog for the CV defect detection application.

Provides a Tkinter Toplevel dialog for creating, opening, and saving
inspection projects (.cvproj archives).  Uses the dark theme consistent
with the rest of the application.
"""

from __future__ import annotations

import logging
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable, Dict, List, Optional

from shared.project_manager import (
    PROJECT_EXT,
    ProjectError,
    ProjectInfo,
    ProjectManager,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Theme constants (matching roi_dialog.py / app conventions)
# ---------------------------------------------------------------------------
_BG = "#2b2b2b"
_BG_DARKER = "#232323"
_FG = "#e0e0e0"
_FG_DIM = "#999999"
_BG_BUTTON = "#3c3c3c"
_BG_ACTIVE = "#3a3a5c"
_BG_ENTRY = "#3c3c3c"
_ACCENT = "#0078d4"
_ACCENT_ACTIVE = "#005a9e"
_SUCCESS = "#2ea043"
_SEPARATOR = "#444444"


# ======================================================================== #
#  ProjectDialog                                                            #
# ======================================================================== #


class ProjectDialog(tk.Toplevel):
    """Modal dialog for project management (new / open / save).

    Parameters
    ----------
    master : tk.Widget
        Parent widget.
    get_current_state : callable
        Returns a dict describing the app's current state::

            {
                "dl_config": {...},
                "vm_config": {...},
                "dl_model_path": str | None,
                "vm_model_path": str | None,
                "recipe_data": dict | None,
                "roi_data": list[dict] | None,
            }
    on_project_loaded : callable(ProjectInfo, dict[str, Path | None])
        Called when a project has been successfully loaded.
    on_project_saved : callable(Path)
        Called after a project has been saved.
    current_info : ProjectInfo | None
        Currently active project info, if any.
    """

    def __init__(
        self,
        master: tk.Widget,
        get_current_state: Callable[[], Dict[str, Any]],
        on_project_loaded: Callable[[ProjectInfo, Dict[str, Optional[Path]]], None],
        on_project_saved: Callable[[Path], None],
        current_info: Optional[ProjectInfo] = None,
    ) -> None:
        super().__init__(master)

        self.title("專案管理")
        self.geometry("500x400")
        self.minsize(460, 380)
        self.configure(bg=_BG)
        self.resizable(False, False)
        self.transient(master)
        self.grab_set()

        self._get_current_state = get_current_state
        self._on_project_loaded = on_project_loaded
        self._on_project_saved = on_project_saved
        self._current_info = current_info

        self._build_ui()
        self._refresh_info_panel()
        self._refresh_recent_list()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------ #
    #  UI construction                                                     #
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        # Top action buttons
        self._build_action_bar()

        # Separator
        sep = tk.Frame(self, bg=_SEPARATOR, height=1)
        sep.pack(fill=tk.X, padx=12, pady=(0, 8))

        # Main content: left = project info, right = recent list
        content = tk.Frame(self, bg=_BG)
        content.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 8))

        self._build_info_panel(content)
        self._build_recent_panel(content)

    def _build_action_bar(self) -> None:
        bar = tk.Frame(self, bg=_BG)
        bar.pack(fill=tk.X, padx=12, pady=(12, 8))

        buttons = [
            ("新增專案", self._new_project, _ACCENT),
            ("開啟專案", self._open_project, _BG_BUTTON),
            ("儲存專案", self._save_project, _SUCCESS),
        ]

        for text, cmd, bg_color in buttons:
            btn = tk.Button(
                bar,
                text=text,
                command=cmd,
                bg=bg_color,
                fg="#ffffff",
                activebackground=_ACCENT_ACTIVE,
                relief=tk.FLAT,
                padx=14,
                pady=6,
                font=("", 10, "bold"),
                cursor="hand2",
            )
            btn.pack(side=tk.LEFT, padx=(0, 8))

    def _build_info_panel(self, parent: tk.Frame) -> None:
        """Project information panel (top half of content area)."""
        frame = tk.LabelFrame(
            parent,
            text=" 專案資訊 ",
            bg=_BG,
            fg=_FG,
            font=("", 9, "bold"),
            padx=8,
            pady=6,
        )
        frame.pack(fill=tk.X, pady=(0, 8))

        self._info_labels: Dict[str, tk.Label] = {}

        rows = [
            ("name", "名稱:"),
            ("description", "描述:"),
            ("product_line", "產品線:"),
            ("created", "建立時間:"),
            ("modified", "修改時間:"),
            ("contents", "包含內容:"),
        ]

        for row_idx, (key, label_text) in enumerate(rows):
            lbl = tk.Label(
                frame,
                text=label_text,
                bg=_BG,
                fg=_FG_DIM,
                font=("", 9),
                anchor=tk.E,
                width=10,
            )
            lbl.grid(row=row_idx, column=0, sticky=tk.E, pady=1, padx=(0, 4))

            value_lbl = tk.Label(
                frame,
                text="--",
                bg=_BG,
                fg=_FG,
                font=("", 9),
                anchor=tk.W,
            )
            value_lbl.grid(row=row_idx, column=1, sticky=tk.W, pady=1)

            self._info_labels[key] = value_lbl

    def _build_recent_panel(self, parent: tk.Frame) -> None:
        """Recent projects list (bottom half of content area)."""
        frame = tk.LabelFrame(
            parent,
            text=" 最近開啟的專案 ",
            bg=_BG,
            fg=_FG,
            font=("", 9, "bold"),
            padx=8,
            pady=6,
        )
        frame.pack(fill=tk.BOTH, expand=True)

        # Listbox + scrollbar
        list_frame = tk.Frame(frame, bg=_BG)
        list_frame.pack(fill=tk.BOTH, expand=True)

        self._recent_listbox = tk.Listbox(
            list_frame,
            bg=_BG_DARKER,
            fg=_FG,
            selectbackground=_ACCENT,
            selectforeground="#ffffff",
            relief=tk.FLAT,
            font=("", 9),
            activestyle="none",
            highlightthickness=0,
            borderwidth=0,
        )
        scrollbar = ttk.Scrollbar(
            list_frame,
            orient=tk.VERTICAL,
            command=self._recent_listbox.yview,
        )
        self._recent_listbox.configure(yscrollcommand=scrollbar.set)

        self._recent_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._recent_listbox.bind("<Double-1>", self._on_recent_double_click)

        # Store paths alongside display text
        self._recent_paths: List[str] = []

    # ------------------------------------------------------------------ #
    #  Refresh helpers                                                     #
    # ------------------------------------------------------------------ #

    def _refresh_info_panel(self) -> None:
        """Update the info panel with current project metadata."""
        info = self._current_info

        if info is None:
            for lbl in self._info_labels.values():
                lbl.configure(text="--")
            return

        self._info_labels["name"].configure(text=info.name or "--")
        self._info_labels["description"].configure(
            text=info.description or "(none)"
        )
        self._info_labels["product_line"].configure(
            text=info.product_line or "(none)"
        )

        # Format timestamps for display
        self._info_labels["created"].configure(
            text=self._format_timestamp(info.created)
        )
        self._info_labels["modified"].configure(
            text=self._format_timestamp(info.modified)
        )

        # Contents summary
        parts: List[str] = []
        if info.has_dl_model:
            parts.append("DL Model")
        if info.has_vm_model:
            parts.append("VM Model")
        if info.has_recipe:
            parts.append("Recipe")
        if info.has_rois:
            parts.append("ROIs")

        contents_text = ", ".join(parts) if parts else "(empty)"
        self._info_labels["contents"].configure(text=contents_text)

    def _refresh_recent_list(self) -> None:
        """Reload the recent-projects listbox."""
        self._recent_listbox.delete(0, tk.END)
        self._recent_paths.clear()

        recent = ProjectManager.list_recent_projects()
        if not recent:
            self._recent_listbox.insert(tk.END, "  (no recent projects)")
            return

        for entry in recent:
            name = entry.get("name", "Untitled")
            path = entry.get("path", "")
            ts = entry.get("timestamp", "")
            display = f"  {name}  --  {Path(path).name}  ({self._format_timestamp(ts)})"
            self._recent_listbox.insert(tk.END, display)
            self._recent_paths.append(path)

    @staticmethod
    def _format_timestamp(iso_str: str) -> str:
        """Format an ISO timestamp for display."""
        if not iso_str or iso_str == "--":
            return "--"
        try:
            # Take first 19 characters (YYYY-MM-DDTHH:MM:SS) and reformat
            dt_part = iso_str[:19].replace("T", " ")
            return dt_part
        except Exception:
            return iso_str

    # ------------------------------------------------------------------ #
    #  Actions                                                             #
    # ------------------------------------------------------------------ #

    def _new_project(self) -> None:
        """Create a new project with user-supplied metadata."""
        dlg = _NewProjectDialog(self)
        self.wait_window(dlg)

        if dlg.result is None:
            return

        name, description, product_line = dlg.result
        self._current_info = ProjectInfo(
            name=name,
            description=description,
            product_line=product_line,
        )
        self._refresh_info_panel()
        logger.info("Created new project: %s", name)

    def _open_project(self) -> None:
        """Open a .cvproj file via file dialog."""
        path = filedialog.askopenfilename(
            title="開啟專案",
            filetypes=[
                ("CV Project", f"*{PROJECT_EXT}"),
                ("All files", "*"),
            ],
            parent=self,
        )
        if not path:
            return

        self._load_from_path(path)

    def _load_from_path(self, path: str) -> None:
        """Load a project from a file path and update the UI."""
        try:
            info, paths = ProjectManager.load_project(path)
        except ProjectError as exc:
            messagebox.showerror("開啟失敗", str(exc), parent=self)
            return

        self._current_info = info
        self._refresh_info_panel()
        self._refresh_recent_list()

        self._on_project_loaded(info, paths)

        messagebox.showinfo(
            "開啟成功",
            f"已載入專案: {info.name}",
            parent=self,
        )
        logger.info("Loaded project '%s' from %s", info.name, path)

    def _save_project(self) -> None:
        """Save the current state as a .cvproj archive."""
        if self._current_info is None:
            messagebox.showwarning(
                "提示", "請先建立或開啟一個專案", parent=self
            )
            return

        path = filedialog.asksaveasfilename(
            title="儲存專案",
            defaultextension=PROJECT_EXT,
            filetypes=[
                ("CV Project", f"*{PROJECT_EXT}"),
                ("All files", "*"),
            ],
            initialfile=f"{self._current_info.name}{PROJECT_EXT}",
            parent=self,
        )
        if not path:
            return

        state = self._get_current_state()

        # Merge current config into project info
        self._current_info.dl_config = state.get("dl_config", {})
        self._current_info.vm_config = state.get("vm_config", {})

        try:
            saved_path = ProjectManager.save_project(
                path=path,
                info=self._current_info,
                dl_model_path=state.get("dl_model_path"),
                vm_model_path=state.get("vm_model_path"),
                recipe_data=state.get("recipe_data"),
                roi_data=state.get("roi_data"),
            )
        except ProjectError as exc:
            messagebox.showerror("儲存失敗", str(exc), parent=self)
            return

        self._refresh_info_panel()
        self._refresh_recent_list()

        self._on_project_saved(saved_path)

        messagebox.showinfo(
            "儲存成功",
            f"專案已儲存至:\n{saved_path}",
            parent=self,
        )
        logger.info("Saved project '%s' to %s", self._current_info.name, saved_path)

    def _on_recent_double_click(self, _event: tk.Event) -> None:
        """Open a project from the recent-projects list on double-click."""
        sel = self._recent_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        if idx >= len(self._recent_paths):
            return

        path = self._recent_paths[idx]
        if not path:
            return

        self._load_from_path(path)

    # ------------------------------------------------------------------ #
    #  Close                                                               #
    # ------------------------------------------------------------------ #

    def _on_close(self) -> None:
        self.grab_release()
        self.destroy()

    @property
    def project_info(self) -> Optional[ProjectInfo]:
        """Return the current project info (may be None)."""
        return self._current_info


# ======================================================================== #
#  _NewProjectDialog (internal)                                             #
# ======================================================================== #


class _NewProjectDialog(tk.Toplevel):
    """Small modal dialog for entering new project details.

    After the dialog is closed, ``self.result`` is either
    ``(name, description, product_line)`` or ``None`` if cancelled.
    """

    def __init__(self, master: tk.Widget) -> None:
        super().__init__(master)

        self.title("新增專案")
        self.geometry("360x240")
        self.resizable(False, False)
        self.configure(bg=_BG)
        self.transient(master)
        self.grab_set()

        self.result: Optional[tuple] = None

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._cancel)

        # Focus the name entry
        self._name_entry.focus_set()

    def _build_ui(self) -> None:
        pad_frame = tk.Frame(self, bg=_BG)
        pad_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=16)

        # Title
        tk.Label(
            pad_frame,
            text="建立新專案",
            bg=_BG,
            fg=_FG,
            font=("", 12, "bold"),
        ).pack(anchor=tk.W, pady=(0, 12))

        # Name
        name_row = tk.Frame(pad_frame, bg=_BG)
        name_row.pack(fill=tk.X, pady=3)
        tk.Label(
            name_row, text="專案名稱:", bg=_BG, fg=_FG, width=10, anchor=tk.E
        ).pack(side=tk.LEFT)
        self._name_var = tk.StringVar()
        self._name_entry = tk.Entry(
            name_row,
            textvariable=self._name_var,
            width=24,
            bg=_BG_ENTRY,
            fg=_FG,
            insertbackground=_FG,
            relief=tk.FLAT,
        )
        self._name_entry.pack(side=tk.LEFT, padx=4)

        # Description
        desc_row = tk.Frame(pad_frame, bg=_BG)
        desc_row.pack(fill=tk.X, pady=3)
        tk.Label(
            desc_row, text="描述:", bg=_BG, fg=_FG, width=10, anchor=tk.E
        ).pack(side=tk.LEFT)
        self._desc_var = tk.StringVar()
        tk.Entry(
            desc_row,
            textvariable=self._desc_var,
            width=24,
            bg=_BG_ENTRY,
            fg=_FG,
            insertbackground=_FG,
            relief=tk.FLAT,
        ).pack(side=tk.LEFT, padx=4)

        # Product line
        prod_row = tk.Frame(pad_frame, bg=_BG)
        prod_row.pack(fill=tk.X, pady=3)
        tk.Label(
            prod_row, text="產品線:", bg=_BG, fg=_FG, width=10, anchor=tk.E
        ).pack(side=tk.LEFT)
        self._prod_var = tk.StringVar()
        tk.Entry(
            prod_row,
            textvariable=self._prod_var,
            width=24,
            bg=_BG_ENTRY,
            fg=_FG,
            insertbackground=_FG,
            relief=tk.FLAT,
        ).pack(side=tk.LEFT, padx=4)

        # Buttons
        btn_row = tk.Frame(pad_frame, bg=_BG)
        btn_row.pack(fill=tk.X, pady=(16, 0))

        tk.Button(
            btn_row,
            text="取消",
            command=self._cancel,
            bg=_BG_BUTTON,
            fg=_FG,
            activebackground=_BG_ACTIVE,
            relief=tk.FLAT,
            padx=12,
            pady=4,
        ).pack(side=tk.RIGHT, padx=(6, 0))

        tk.Button(
            btn_row,
            text="建立",
            command=self._confirm,
            bg=_ACCENT,
            fg="#ffffff",
            activebackground=_ACCENT_ACTIVE,
            relief=tk.FLAT,
            padx=12,
            pady=4,
        ).pack(side=tk.RIGHT)

    def _confirm(self) -> None:
        name = self._name_var.get().strip()
        if not name:
            messagebox.showwarning("提示", "請輸入專案名稱", parent=self)
            return

        self.result = (
            name,
            self._desc_var.get().strip(),
            self._prod_var.get().strip(),
        )
        self.grab_release()
        self.destroy()

    def _cancel(self) -> None:
        self.result = None
        self.grab_release()
        self.destroy()
