"""User-friendly error dialog with error codes and recovery hints."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
import traceback
from typing import Dict, Tuple

import platform as _platform

from shared.i18n import t

_SYS = _platform.system()
if _SYS == "Darwin":
    _FONT_FAMILY = "Helvetica Neue"
    _MONO_FAMILY = "Menlo"
elif _SYS == "Linux":
    _FONT_FAMILY = "DejaVu Sans"
    _MONO_FAMILY = "DejaVu Sans Mono"
else:
    _FONT_FAMILY = "Segoe UI"
    _MONO_FAMILY = "Consolas"

# Maps exception type names to (code, user_message, recovery_hint).
ERROR_MAP: Dict[str, Tuple[str, str, str]] = {
    "FileNotFoundError": (
        "E-FILE-001",
        "找不到指定的檔案。",
        "請確認檔案路徑是否正確，檔案是否已被移動或刪除。",
    ),
    "PermissionError": (
        "E-FILE-002",
        "沒有權限存取該檔案。",
        "請確認您有足夠的權限，或嘗試以管理員身分執行。",
    ),
    "ImageValidationError": (
        "E-IMG-001",
        "影像資料無效。",
        "請確認輸入影像格式正確，且未損壞。",
    ),
    "cv2.error": (
        "E-CV-001",
        "影像處理引擎發生錯誤。",
        "請確認輸入影像格式和參數是否正確，或嘗試使用不同的參數。",
    ),
    "ValueError": (
        "E-VAL-001",
        "輸入參數值無效。",
        "請檢查輸入的數值範圍是否合理。",
    ),
    "MemoryError": (
        "E-MEM-001",
        "記憶體不足，無法完成操作。",
        "請嘗試使用較小的影像，或關閉其他程式以釋放記憶體。",
    ),
    "RuntimeError": (
        "E-RT-001",
        "執行時發生錯誤。",
        "請查看技術細節以獲取更多資訊。",
    ),
}


class ErrorDialog(tk.Toplevel):
    """User-friendly error dialog with expandable technical details."""

    def __init__(
        self,
        parent: tk.Tk,
        context: str,
        exc: Exception,
    ) -> None:
        super().__init__(parent)
        self.title(t("error.title"))
        self.transient(parent)
        self.grab_set()
        self.resizable(False, False)

        # Lookup error info -- try "module.name" first, then plain "name"
        exc_type = type(exc)
        exc_name = exc_type.__name__
        exc_qualified = f"{exc_type.__module__}.{exc_name}" if exc_type.__module__ else exc_name
        default = ("E-UNKNOWN", t("error.unexpected"), t("error.details"))
        code, user_msg, hint = ERROR_MAP.get(
            exc_qualified, ERROR_MAP.get(exc_name, default),
        )

        bg = "#2b2b2b"
        fg = "#e0e0e0"
        self.configure(bg=bg)

        # Main frame
        main = ttk.Frame(self)
        main.pack(fill=tk.BOTH, expand=True, padx=16, pady=12)

        # Error code + context
        ttk.Label(
            main, text=f"[{code}] {context}", font=(_FONT_FAMILY, 10, "bold"),
        ).pack(anchor=tk.W)

        # User-friendly message
        ttk.Label(main, text=user_msg, wraplength=400).pack(
            anchor=tk.W, pady=(8, 0),
        )

        # Recovery hint
        hint_frame = ttk.Frame(main)
        hint_frame.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(hint_frame, text=t("error.hint_label"), font=(_FONT_FAMILY, 9, "bold")).pack(
            anchor=tk.W,
        )
        ttk.Label(hint_frame, text=hint, wraplength=400).pack(anchor=tk.W)

        # Technical details (collapsed by default)
        self._detail_var = tk.BooleanVar(value=False)
        detail_btn = ttk.Checkbutton(
            main, text=t("error.show_technical"), variable=self._detail_var,
            command=self._toggle_detail,
        )
        detail_btn.pack(anchor=tk.W, pady=(12, 0))

        self._detail_text = tk.Text(
            main, height=6, width=55, bg="#1e1e1e", fg="#cccccc",
            font=(_MONO_FAMILY, 9), wrap=tk.WORD,
        )
        tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
        self._detail_text.insert(tk.END, "".join(tb))
        self._detail_text.configure(state=tk.DISABLED)
        # Hidden initially

        # Buttons
        btn_frame = ttk.Frame(main)
        btn_frame.pack(fill=tk.X, pady=(12, 0))

        ttk.Button(btn_frame, text=t("error.copy"), command=self._copy).pack(
            side=tk.LEFT,
        )
        ttk.Button(btn_frame, text=t("dialog.ok"), command=self.destroy).pack(
            side=tk.RIGHT,
        )

        # Centre on parent
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - self.winfo_width()) // 2
        y = parent.winfo_y() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{x}+{y}")

    def _toggle_detail(self) -> None:
        if self._detail_var.get():
            self._detail_text.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
        else:
            self._detail_text.pack_forget()

    def _copy(self) -> None:
        self.clipboard_clear()
        self.clipboard_append(self._detail_text.get("1.0", tk.END))


def show_error(parent: tk.Tk, context: str, exc: Exception) -> None:
    """Convenience function to show an :class:`ErrorDialog`."""
    ErrorDialog(parent, context, exc)
