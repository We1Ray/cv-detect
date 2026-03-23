"""Results panel — inspection result summary, statistics, and export.

Displays the latest and aggregate inspection results with visual
indicators and one-click export capabilities.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable, Dict, List, Optional

import numpy as np

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
    _FONT_FAMILY = _FONT_FAMILY
    _MONO_FAMILY = "Consolas"


class ResultsPanel(ttk.Frame):
    """Inspection results summary with statistics and export."""

    def __init__(
        self,
        master: tk.Misc,
        on_export_csv: Optional[Callable] = None,
        on_export_pdf: Optional[Callable] = None,
        **kwargs,
    ) -> None:
        super().__init__(master, **kwargs)
        self._on_export_csv = on_export_csv
        self._on_export_pdf = on_export_pdf
        self._results: List[Dict[str, Any]] = []
        self._build_ui()

    def _build_ui(self) -> None:
        # ── Latest Result Section ──
        latest_frame = ttk.LabelFrame(self, text=t("results.latest_result"))
        latest_frame.pack(fill=tk.X, padx=4, pady=(4, 2))

        self._result_indicator = tk.Canvas(
            latest_frame, width=40, height=40,
            bg="#2b2b2b", highlightthickness=0,
        )
        self._result_indicator.pack(side=tk.LEFT, padx=8, pady=4)
        self._draw_idle_indicator()

        info_frame = ttk.Frame(latest_frame)
        info_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4, pady=4)

        self._result_label = tk.Label(
            info_frame, text=t("results.no_results"),
            bg="#2b2b2b", fg="#888888",
            font=(_FONT_FAMILY, 11, "bold"), anchor=tk.W,
        )
        self._result_label.pack(fill=tk.X)

        self._detail_label = tk.Label(
            info_frame, text="",
            bg="#2b2b2b", fg="#aaaaaa",
            font=(_FONT_FAMILY, 8), anchor=tk.W,
        )
        self._detail_label.pack(fill=tk.X)

        # ── Aggregate Statistics ──
        stats_frame = ttk.LabelFrame(self, text=t("results.stats_summary"))
        stats_frame.pack(fill=tk.X, padx=4, pady=2)

        self._stat_vars = {}
        stat_fields = [
            ("total", t("results.total_inspected")),
            ("pass_count", t("results.pass_count")),
            ("fail_count", t("results.fail_count")),
            ("yield_rate", t("results.yield_rate")),
            ("avg_score", t("results.avg_score")),
            ("max_score", t("results.max_score")),
        ]
        for i, (key, label) in enumerate(stat_fields):
            row, col = divmod(i, 2)
            ttk.Label(stats_frame, text=label, font=(_FONT_FAMILY, 8)).grid(
                row=row, column=col * 2, sticky=tk.W, padx=(8, 2), pady=1,
            )
            var = tk.StringVar(value="--")
            self._stat_vars[key] = var
            ttk.Label(stats_frame, textvariable=var, font=(_FONT_FAMILY, 8, "bold")).grid(
                row=row, column=col * 2 + 1, sticky=tk.W, padx=(0, 12), pady=1,
            )

        # ── Results Table ──
        table_frame = ttk.LabelFrame(self, text=t("results.inspection_log"))
        table_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=2)

        columns = ("No", "Result", "Score", "Defects")
        self._results_tree = ttk.Treeview(
            table_frame, columns=columns, show="headings", height=8,
        )
        col_configs = {
            "No": ("#", 40),
            "Result": (t("results.result"), 60),
            "Score": (t("results.score"), 70),
            "Defects": (t("results.defect_count"), 60),
        }
        for col_id, (heading, width) in col_configs.items():
            self._results_tree.heading(col_id, text=heading)
            self._results_tree.column(col_id, width=width, anchor=tk.CENTER)

        tree_scroll = ttk.Scrollbar(
            table_frame, orient=tk.VERTICAL, command=self._results_tree.yview,
        )
        self._results_tree.configure(yscrollcommand=tree_scroll.set)
        self._results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Empty-state placeholder label
        self._empty_label = tk.Label(
            table_frame,
            text=t("results.no_results"),
            bg="#1e1e1e",
            fg="#666666",
            font=(_FONT_FAMILY, 11),
        )
        self._empty_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # Tag styling for pass/fail rows
        self._results_tree.tag_configure("pass", foreground="#4caf50")
        self._results_tree.tag_configure("fail", foreground="#f44336")

        # ── Export Buttons ──
        export_frame = ttk.Frame(self)
        export_frame.pack(fill=tk.X, padx=4, pady=(2, 4))

        ttk.Button(
            export_frame, text=t("results.export_csv"),
            command=self._export_csv,
        ).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)

        ttk.Button(
            export_frame, text=t("results.export_pdf"),
            command=self._export_pdf,
        ).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)

        ttk.Button(
            export_frame, text=t("results.clear_records"),
            command=self._clear_results,
        ).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_result(
        self,
        is_pass: bool,
        score: float,
        defect_count: int = 0,
        image_name: str = "",
        model_type: str = "",
    ) -> None:
        """Add an inspection result and update display."""
        result = {
            "index": len(self._results) + 1,
            "is_pass": is_pass,
            "score": score,
            "defect_count": defect_count,
            "image_name": image_name,
            "model_type": model_type,
        }
        self._results.append(result)

        # Hide empty-state placeholder
        self._empty_label.place_forget()

        # Update latest result indicator
        self._update_latest(result)

        # Add to table
        tag = "pass" if is_pass else "fail"
        label = "PASS" if is_pass else "NG"
        self._results_tree.insert(
            "", 0,  # Insert at top (newest first)
            values=(result["index"], label, f"{score:.4f}", defect_count),
            tags=(tag,),
        )

        # Update statistics
        self._update_stats()

    def clear(self) -> None:
        """Clear all results."""
        self._results.clear()
        for item in self._results_tree.get_children():
            self._results_tree.delete(item)
        self._draw_idle_indicator()
        self._result_label.configure(text=t("results.no_results"), fg="#888888")
        self._detail_label.configure(text="")
        for var in self._stat_vars.values():
            var.set("--")
        # Show empty-state placeholder
        self._empty_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _update_latest(self, result: Dict[str, Any]) -> None:
        """Update the latest result display."""
        is_pass = result["is_pass"]
        score = result["score"]

        if is_pass:
            self._result_label.configure(text="PASS \u2714", fg="#4caf50")
            self._draw_pass_indicator()
        else:
            self._result_label.configure(text="NG \u2718", fg="#f44336")
            self._draw_fail_indicator()

        detail = f"{t('results.score')}: {score:.4f}  |  {t('results.defect_count')}: {result['defect_count']}"
        if result.get("image_name"):
            detail += f"  |  {result['image_name']}"
        self._detail_label.configure(text=detail)

    def _update_stats(self) -> None:
        """Recalculate and display aggregate statistics."""
        n = len(self._results)
        if n == 0:
            return
        passes = sum(1 for r in self._results if r["is_pass"])
        fails = n - passes
        scores = [r["score"] for r in self._results]

        self._stat_vars["total"].set(str(n))
        self._stat_vars["pass_count"].set(str(passes))
        self._stat_vars["fail_count"].set(str(fails))
        yield_rate = (passes / n) * 100
        self._stat_vars["yield_rate"].set(f"{yield_rate:.1f}%")
        self._stat_vars["avg_score"].set(f"{np.mean(scores):.4f}")
        self._stat_vars["max_score"].set(f"{np.max(scores):.4f}")

    def _draw_idle_indicator(self) -> None:
        c = self._result_indicator
        c.delete("all")
        c.create_oval(5, 5, 35, 35, fill="#555555", outline="#777777", width=2)

    def _draw_pass_indicator(self) -> None:
        c = self._result_indicator
        c.delete("all")
        c.create_oval(5, 5, 35, 35, fill="#2e7d32", outline="#4caf50", width=2)
        c.create_text(20, 20, text="\u2714", fill="white", font=(_FONT_FAMILY, 14, "bold"))

    def _draw_fail_indicator(self) -> None:
        c = self._result_indicator
        c.delete("all")
        c.create_oval(5, 5, 35, 35, fill="#c62828", outline="#f44336", width=2)
        c.create_text(20, 20, text="\u2718", fill="white", font=(_FONT_FAMILY, 14, "bold"))

    def _export_csv(self) -> None:
        if not self._results:
            messagebox.showinfo(t("results.info"), t("results.no_export_data"))
            return
        if self._on_export_csv:
            self._on_export_csv(self._results)
            return
        # Default CSV export
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile="inspection_results.csv",
        )
        if not path:
            return
        import csv
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["index", "is_pass", "score", "defect_count", "image_name", "model_type"])
            writer.writeheader()
            for r in self._results:
                row = dict(r)
                row["is_pass"] = "PASS" if row["is_pass"] else "NG"
                writer.writerow(row)

    def _export_pdf(self) -> None:
        if not self._results:
            messagebox.showinfo(t("results.info"), t("results.no_export_data"))
            return
        if self._on_export_pdf:
            self._on_export_pdf(self._results)
        else:
            messagebox.showinfo(t("results.info"), t("results.use_pdf_tool"))

    def _clear_results(self) -> None:
        if self._results and messagebox.askyesno(t("results.confirm"), t("results.confirm_clear")):
            self.clear()
