"""PDF inspection report generator using matplotlib.

Generates professional multi-page PDF reports from inspection results.
Each report may include: cover page, summary statistics, detail pages with
images, measurement tables, SPC control charts, and a signature footer.

Only depends on matplotlib and numpy -- no reportlab / weasyprint required.

Usage::

    gen = PDFReportGenerator(ReportConfig(company_name="TastyByte"))
    gen.add_entry(InspectionEntry(...))
    gen.add_entry(InspectionEntry(...))
    gen.generate("report.pdf")
"""

from __future__ import annotations

import logging
import io
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch
import numpy as np

from shared.op_logger import log_operation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

_CLR_HEADER = '#003366'
_CLR_PASS = '#28a745'
_CLR_FAIL = '#dc3545'
_CLR_TABLE_HEADER = '#f0f0f0'
_CLR_TABLE_ALT = '#fafafa'
_CLR_WARN = '#ffc107'
_CLR_INFO = '#17a2b8'
_CLR_BORDER = '#cccccc'
_CLR_BG_LIGHT = '#f8f9fa'
_CLR_TEXT = '#333333'
_CLR_TEXT_LIGHT = '#666666'
_CLR_TEXT_MUTED = '#999999'

_SOFTWARE_VERSION = 'TastyByte CV-Detect v1.0'


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ReportConfig:
    """Report configuration."""

    company_name: str = "TastyByte"
    report_title: str = "\u6aa2\u6e2c\u5831\u544a"          # 檢測報告
    operator: str = ""
    line_id: str = ""
    product_name: str = ""
    lot_number: str = ""
    logo_path: str = ""          # optional company logo
    page_size: str = "A4"       # "A4" or "Letter"
    language: str = "zh-TW"
    include_spc: bool = True
    include_images: bool = True
    include_measurements: bool = True
    max_defect_images: int = 10
    font_family: str = ""       # auto-detect CJK font


@dataclass
class InspectionEntry:
    """A single inspection result entry for the report."""

    image_path: str = ""
    original_image: Optional[np.ndarray] = None
    result_image: Optional[np.ndarray] = None    # annotated result
    heatmap_image: Optional[np.ndarray] = None
    anomaly_score: float = 0.0
    threshold: float = 0.5
    is_defective: bool = False
    defect_count: int = 0
    defect_area: float = 0.0
    measurements: List[Dict[str, Any]] = field(default_factory=list)
    # e.g. [{"name": "寬度", "value": 10.5, "unit": "mm",
    #         "tolerance_min": 10.0, "tolerance_max": 11.0,
    #         "in_tolerance": True}]
    notes: str = ""
    timestamp: str = ""


# ---------------------------------------------------------------------------
# PDFReportGenerator
# ---------------------------------------------------------------------------

class PDFReportGenerator:
    """Generate professional PDF inspection reports using matplotlib.

    Usage::

        gen = PDFReportGenerator(ReportConfig(company_name="TastyByte"))
        gen.add_entry(InspectionEntry(...))
        gen.add_entry(InspectionEntry(...))
        gen.generate("report.pdf")
    """

    # Font candidates per platform
    _FONT_CANDIDATES: List[str] = [
        # macOS
        'PingFang TC',
        'Heiti TC',
        'STHeiti',
        'Apple LiGothic',
        # Windows
        'Microsoft JhengHei',
        'SimHei',
        'Microsoft YaHei',
        # Linux
        'Noto Sans CJK TC',
        'Noto Sans TC',
        'WenQuanYi Micro Hei',
        'WenQuanYi Zen Hei',
        # Generic fallback
        'Arial Unicode MS',
        'DejaVu Sans',
    ]

    def __init__(self, config: Optional[ReportConfig] = None) -> None:
        self.config = config or ReportConfig()
        self._entries: List[InspectionEntry] = []
        self._spc_data: Optional[Dict[str, Any]] = None
        self._font_prop: Optional[fm.FontProperties] = None
        self._font_prop_bold: Optional[fm.FontProperties] = None
        self._font_prop_small: Optional[fm.FontProperties] = None
        self._setup_font()

    # ------------------------------------------------------------------
    # Font setup
    # ------------------------------------------------------------------

    def _setup_font(self) -> None:
        """Auto-detect and configure a CJK font for matplotlib."""
        if self.config.font_family:
            candidates = [self.config.font_family] + self._FONT_CANDIDATES
        else:
            candidates = list(self._FONT_CANDIDATES)

        available_fonts = {f.name for f in fm.fontManager.ttflist}

        chosen: Optional[str] = None
        for name in candidates:
            if name in available_fonts:
                chosen = name
                break

        if chosen:
            self._font_prop = fm.FontProperties(family=chosen, size=10)
            self._font_prop_bold = fm.FontProperties(
                family=chosen, size=10, weight='bold',
            )
            self._font_prop_small = fm.FontProperties(
                family=chosen, size=8,
            )
            plt.rcParams['font.family'] = chosen
            plt.rcParams['axes.unicode_minus'] = False
            logger.info("Report font set to '%s'", chosen)
        else:
            self._font_prop = fm.FontProperties(size=10)
            self._font_prop_bold = fm.FontProperties(size=10, weight='bold')
            self._font_prop_small = fm.FontProperties(size=8)
            plt.rcParams['axes.unicode_minus'] = False
            logger.warning(
                "No CJK font found; Chinese characters may not render "
                "correctly. Tried: %s",
                ', '.join(candidates[:6]),
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_entry(self, entry: InspectionEntry) -> None:
        """Append a single inspection entry."""
        self._entries.append(entry)

    def add_entries(self, entries: List[InspectionEntry]) -> None:
        """Append multiple inspection entries."""
        self._entries.extend(entries)

    def set_spc_data(
        self,
        spc_metrics: Any,
        trend_data: Any = None,
    ) -> None:
        """Attach SPC data (from ``results_db.SPCMetrics`` / ``TrendData``)."""
        self._spc_data = {'metrics': spc_metrics, 'trend': trend_data}

    @log_operation(logger)
    def generate(self, output_path: str) -> str:
        """Generate the PDF report and return *output_path*."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        total_pages = self._estimate_total_pages()

        with PdfPages(str(out)) as pdf:
            page_num = [0]  # mutable counter shared across renderers

            # Page 1: Cover / Summary
            self._render_cover_page(pdf, page_num, total_pages)

            # Page 2: Summary statistics table
            self._render_summary_page(pdf, page_num, total_pages)

            # Pages 3+: Individual results (with images)
            if self.config.include_images and self._entries:
                self._render_detail_pages(pdf, page_num, total_pages)

            # Measurement summary page
            if (
                self.config.include_measurements
                and any(e.measurements for e in self._entries)
            ):
                self._render_measurement_page(pdf, page_num, total_pages)

            # SPC page
            if self.config.include_spc and self._spc_data:
                self._render_spc_page(pdf, page_num, total_pages)

            # Final page: footer / signature
            self._render_footer_page(pdf, page_num, total_pages)

        logger.info("PDF report saved to %s (%d pages)", out, page_num[0])
        return str(out)

    # ------------------------------------------------------------------
    # Page size helpers
    # ------------------------------------------------------------------

    def _page_size(self) -> Tuple[float, float]:
        """Return ``(width, height)`` in inches."""
        if self.config.page_size == 'Letter':
            return (8.5, 11.0)
        return (8.27, 11.69)  # A4

    def _estimate_total_pages(self) -> int:
        """Rough page-count estimate used for page numbering."""
        pages = 2  # cover + summary
        if self.config.include_images and self._entries:
            # 2 entries per detail page
            pages += max(1, (len(self._entries) + 1) // 2)
        if self.config.include_measurements and any(
            e.measurements for e in self._entries
        ):
            pages += 1
        if self.config.include_spc and self._spc_data:
            pages += 1
        pages += 1  # footer
        return pages

    # ------------------------------------------------------------------
    # Page-level decoration helpers
    # ------------------------------------------------------------------

    def _stamp_header(self, fig: plt.Figure, title: str) -> None:
        """Draw a coloured header bar at the top of *fig*."""
        fig.patches.append(
            FancyBboxPatch(
                (0.0, 0.94), 1.0, 0.06,
                boxstyle='square,pad=0',
                facecolor=_CLR_HEADER,
                edgecolor='none',
                transform=fig.transFigure,
                clip_on=False,
            ),
        )
        fig.text(
            0.5, 0.97, title,
            ha='center', va='center',
            fontsize=14, color='white',
            fontproperties=self._font_prop_bold,
        )
        # Company name in top-left
        fig.text(
            0.03, 0.97,
            self.config.company_name,
            ha='left', va='center',
            fontsize=9, color='#aaccee',
            fontproperties=self._font_prop,
        )

    def _stamp_page_number(
        self,
        fig: plt.Figure,
        page_num: List[int],
        total_pages: int,
    ) -> None:
        """Stamp page number at bottom-centre and increment counter."""
        page_num[0] += 1
        fig.text(
            0.5, 0.012,
            f'\u7b2c {page_num[0]} / {total_pages} \u9801',  # 第 X / Y 頁
            ha='center', va='bottom',
            fontsize=8, color=_CLR_TEXT_LIGHT,
            fontproperties=self._font_prop,
        )
        # Bottom border line
        fig.patches.append(
            FancyBboxPatch(
                (0.05, 0.025), 0.90, 0.001,
                boxstyle='square,pad=0',
                facecolor=_CLR_BORDER,
                edgecolor='none',
                transform=fig.transFigure,
                clip_on=False,
            ),
        )

    # ------------------------------------------------------------------
    # Page 1: Cover page
    # ------------------------------------------------------------------

    def _render_cover_page(
        self,
        pdf: PdfPages,
        page_num: List[int],
        total_pages: int,
    ) -> None:
        """Cover page: company, title, product info, quick summary."""
        w, h = self._page_size()
        fig = plt.figure(figsize=(w, h))

        # Top background block
        fig.patches.append(
            FancyBboxPatch(
                (0.0, 0.55), 1.0, 0.45,
                boxstyle='square,pad=0',
                facecolor=_CLR_HEADER,
                edgecolor='none',
                transform=fig.transFigure,
                clip_on=False,
            ),
        )

        # Optional logo
        if self.config.logo_path and Path(self.config.logo_path).is_file():
            try:
                logo = plt.imread(self.config.logo_path)
                logo_ax = fig.add_axes([0.38, 0.85, 0.24, 0.10])
                logo_ax.imshow(logo)
                logo_ax.axis('off')
            except Exception:
                logger.warning("Failed to load logo: %s", self.config.logo_path)

        # Company name
        fig.text(
            0.5, 0.80,
            self.config.company_name,
            ha='center', va='center',
            fontsize=28, color='white',
            fontproperties=self._font_prop_bold,
        )

        # Report title
        fig.text(
            0.5, 0.72,
            self.config.report_title,
            ha='center', va='center',
            fontsize=22, color='#aaccee',
            fontproperties=self._font_prop_bold,
        )

        # Decorative divider line
        fig.patches.append(
            FancyBboxPatch(
                (0.25, 0.67), 0.50, 0.003,
                boxstyle='square,pad=0',
                facecolor='#ffffff',
                edgecolor='none',
                transform=fig.transFigure,
                clip_on=False,
            ),
        )

        # Date / time
        now = datetime.now()
        fig.text(
            0.5, 0.63,
            now.strftime('%Y \u5e74 %m \u6708 %d \u65e5  %H:%M'),
            ha='center', va='center',
            fontsize=14, color='#ccddee',
            fontproperties=self._font_prop,
        )

        # Product info block (below the coloured area)
        info_items: List[Tuple[str, str]] = []
        if self.config.product_name:
            info_items.append(('\u7522\u54c1\u540d\u7a31', self.config.product_name))
        if self.config.lot_number:
            info_items.append(('\u6279\u865f', self.config.lot_number))
        if self.config.line_id:
            info_items.append(('\u7522\u7dda', self.config.line_id))
        if self.config.operator:
            info_items.append(('\u64cd\u4f5c\u54e1', self.config.operator))

        y_start = 0.48
        for i, (label, value) in enumerate(info_items):
            y = y_start - i * 0.04
            fig.text(
                0.20, y,
                f'{label}\uff1a',
                ha='right', va='center',
                fontsize=12, color=_CLR_TEXT_LIGHT,
                fontproperties=self._font_prop,
            )
            fig.text(
                0.22, y,
                value,
                ha='left', va='center',
                fontsize=12, color=_CLR_TEXT,
                fontproperties=self._font_prop_bold,
            )

        # Quick summary box
        total = len(self._entries)
        n_pass = sum(1 for e in self._entries if not e.is_defective)
        n_fail = total - n_pass
        defect_rate = (n_fail / total * 100) if total > 0 else 0.0

        box_y = 0.21
        fig.patches.append(
            FancyBboxPatch(
                (0.08, box_y - 0.04), 0.84, 0.16,
                boxstyle='round,pad=0.01',
                facecolor=_CLR_BG_LIGHT,
                edgecolor=_CLR_BORDER,
                linewidth=1,
                transform=fig.transFigure,
                clip_on=False,
            ),
        )
        fig.text(
            0.5, box_y + 0.09,
            '\u6aa2\u6e2c\u6458\u8981',  # 檢測摘要
            ha='center', va='center',
            fontsize=13, color=_CLR_HEADER,
            fontproperties=self._font_prop_bold,
        )

        summary_items: List[Tuple[str, str]] = [
            (f'\u7e3d\u6578\uff1a{total}', _CLR_TEXT),
            (f'\u901a\u904e\uff1a{n_pass}', _CLR_PASS),
            (f'\u4e0d\u826f\uff1a{n_fail}', _CLR_FAIL),
            (
                f'\u4e0d\u826f\u7387\uff1a{defect_rate:.1f}%',
                _CLR_FAIL if defect_rate > 0 else _CLR_PASS,
            ),
        ]
        col_width = 0.80 / len(summary_items)
        for j, (txt, clr) in enumerate(summary_items):
            fig.text(
                0.12 + j * col_width, box_y + 0.02,
                txt,
                ha='left', va='center',
                fontsize=11, color=clr,
                fontproperties=self._font_prop_bold,
            )

        self._stamp_page_number(fig, page_num, total_pages)
        pdf.savefig(fig)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Page 2: Summary statistics
    # ------------------------------------------------------------------

    def _render_summary_page(
        self,
        pdf: PdfPages,
        page_num: List[int],
        total_pages: int,
    ) -> None:
        """Summary page: bar chart, pie chart, histogram, table."""
        w, h = self._page_size()
        fig = plt.figure(figsize=(w, h))
        self._stamp_header(fig, '\u7d71\u8a08\u6458\u8981')  # 統計摘要

        gs = gridspec.GridSpec(
            3, 2,
            left=0.10, right=0.92,
            top=0.90, bottom=0.06,
            hspace=0.45, wspace=0.35,
        )

        total = len(self._entries)
        n_pass = sum(1 for e in self._entries if not e.is_defective)
        n_fail = total - n_pass
        scores = [e.anomaly_score for e in self._entries]

        # --- Bar chart: pass vs fail ---
        ax_bar = fig.add_subplot(gs[0, 0])
        bar_labels = ['\u901a\u904e', '\u4e0d\u826f']  # 通過, 不良
        bar_values = [n_pass, n_fail]
        bar_colors = [_CLR_PASS, _CLR_FAIL]
        bars = ax_bar.bar(
            bar_labels, bar_values,
            color=bar_colors, edgecolor='white', width=0.5,
        )
        for bar_item in bars:
            val = int(bar_item.get_height())
            if val > 0:
                ax_bar.text(
                    bar_item.get_x() + bar_item.get_width() / 2,
                    bar_item.get_height() + 0.3,
                    str(val),
                    ha='center', va='bottom', fontsize=10,
                    fontproperties=self._font_prop_bold,
                )
        ax_bar.set_title(
            '\u6aa2\u6e2c\u7d50\u679c\u5206\u4f48',  # 檢測結果分佈
            fontproperties=self._font_prop_bold, fontsize=11,
        )
        ax_bar.set_ylabel(
            '\u6578\u91cf',  # 數量
            fontproperties=self._font_prop,
        )
        for lbl in ax_bar.get_xticklabels():
            lbl.set_fontproperties(self._font_prop)

        # --- Pie chart ---
        ax_pie = fig.add_subplot(gs[0, 1])
        if total > 0:
            pie_data = [
                (n_pass, f'\u901a\u904e ({n_pass})', _CLR_PASS),
                (n_fail, f'\u4e0d\u826f ({n_fail})', _CLR_FAIL),
            ]
            non_zero = [(s, l, c) for s, l, c in pie_data if s > 0]
            if non_zero:
                sz, lb, cl = zip(*non_zero)
                wedges, texts, autotexts = ax_pie.pie(
                    sz, labels=lb, colors=cl,
                    autopct='%1.1f%%', startangle=90,
                    textprops={
                        'fontproperties': self._font_prop, 'fontsize': 9,
                    },
                )
                for at in autotexts:
                    at.set_fontsize(9)
                    at.set_color('white')
                    at.set_fontweight('bold')
            else:
                ax_pie.text(
                    0.5, 0.5, '\u7121\u8cc7\u6599',  # 無資料
                    ha='center', va='center',
                    fontproperties=self._font_prop, fontsize=11,
                    transform=ax_pie.transAxes,
                )
        ax_pie.set_title(
            '\u4e0d\u826f\u7387',  # 不良率
            fontproperties=self._font_prop_bold, fontsize=11,
        )

        # --- Score distribution histogram ---
        ax_hist = fig.add_subplot(gs[1, :])
        if scores:
            n_bins = min(30, max(5, len(scores) // 2))
            ax_hist.hist(
                scores, bins=n_bins,
                color=_CLR_INFO, edgecolor='white', alpha=0.85,
            )
            # Threshold reference line
            if self._entries:
                th = self._entries[0].threshold
                ax_hist.axvline(
                    th, color=_CLR_FAIL, linestyle='--', linewidth=1.5,
                    label=f'\u95be\u503c = {th:.3f}',  # 閾值
                )
                ax_hist.legend(
                    prop=self._font_prop, fontsize=9, loc='upper right',
                )
        ax_hist.set_title(
            '\u7570\u5e38\u5206\u6578\u5206\u4f48',  # 異常分數分佈
            fontproperties=self._font_prop_bold, fontsize=11,
        )
        ax_hist.set_xlabel(
            '\u7570\u5e38\u5206\u6578',  # 異常分數
            fontproperties=self._font_prop,
        )
        ax_hist.set_ylabel(
            '\u6578\u91cf',  # 數量
            fontproperties=self._font_prop,
        )

        # --- Summary table ---
        if self._entries:
            ax_tbl = fig.add_subplot(gs[2, :])
            ax_tbl.axis('off')
            table_data = self._build_summary_table_data()
            col_labels = [
                '\u6a94\u6848',           # 檔案
                '\u5206\u6578',           # 分數
                '\u95be\u503c',           # 閾值
                '\u7d50\u679c',           # 結果
                '\u7f3a\u9677\u6578',     # 缺陷數
                '\u7f3a\u9677\u9762\u7a4d',  # 缺陷面積
            ]
            self._add_table(ax_tbl, table_data, col_labels)

        self._stamp_page_number(fig, page_num, total_pages)
        pdf.savefig(fig)
        plt.close(fig)

    def _build_summary_table_data(self) -> List[List[str]]:
        """Build rows for the summary table (capped at 20 rows)."""
        rows: List[List[str]] = []
        display_entries = self._entries[:20]
        for e in display_entries:
            fname = Path(e.image_path).name if e.image_path else '-'
            if len(fname) > 25:
                fname = fname[:22] + '...'
            verdict = '\u901a\u904e' if not e.is_defective else '\u4e0d\u826f'
            rows.append([
                fname,
                f'{e.anomaly_score:.4f}',
                f'{e.threshold:.4f}',
                verdict,
                str(e.defect_count),
                f'{e.defect_area:.1f}',
            ])
        if len(self._entries) > 20:
            rows.append([
                f'... \u5171 {len(self._entries)} \u7b46',
                '', '', '', '', '',
            ])
        return rows

    # ------------------------------------------------------------------
    # Detail pages (images, 2 entries per page)
    # ------------------------------------------------------------------

    def _render_detail_pages(
        self,
        pdf: PdfPages,
        page_num: List[int],
        total_pages: int,
    ) -> None:
        """Render detail pages with 2 entries per page."""
        entries = self._entries[:self.config.max_defect_images]
        for idx in range(0, len(entries), 2):
            batch = entries[idx:idx + 2]
            self._render_detail_page_pair(
                pdf, batch, idx, page_num, total_pages,
            )

    def _render_detail_page_pair(
        self,
        pdf: PdfPages,
        entries: List[InspectionEntry],
        start_idx: int,
        page_num: List[int],
        total_pages: int,
    ) -> None:
        """Render 1 or 2 entries on a single detail page."""
        w, h = self._page_size()
        fig = plt.figure(figsize=(w, h))
        self._stamp_header(fig, '\u6aa2\u6e2c\u660e\u7d30')  # 檢測明細

        n_entries = len(entries)
        # Each entry uses 3 grid rows: title, images, info
        gs = gridspec.GridSpec(
            n_entries * 3, 2,
            left=0.06, right=0.94,
            top=0.90, bottom=0.05,
            hspace=0.55, wspace=0.12,
        )

        for i, entry in enumerate(entries):
            row_base = i * 3
            entry_num = start_idx + i + 1

            # -- Title row --
            ax_title = fig.add_subplot(gs[row_base, :])
            ax_title.axis('off')
            verdict_str = '\u901a\u904e' if not entry.is_defective else '\u4e0d\u826f'
            verdict_clr = _CLR_PASS if not entry.is_defective else _CLR_FAIL
            fname = (
                Path(entry.image_path).name if entry.image_path
                else f'#{entry_num}'
            )

            ax_title.text(
                0.0, 0.5,
                f'\u6a23\u672c {entry_num}: {fname}',  # 樣本
                transform=ax_title.transAxes,
                ha='left', va='center',
                fontsize=11, color=_CLR_HEADER,
                fontproperties=self._font_prop_bold,
            )
            ax_title.text(
                1.0, 0.5,
                f'[{verdict_str}]',
                transform=ax_title.transAxes,
                ha='right', va='center',
                fontsize=12, color=verdict_clr,
                fontproperties=self._font_prop_bold,
            )

            # -- Image row: original (left) + result/heatmap (right) --
            ax_orig = fig.add_subplot(gs[row_base + 1, 0])
            self._render_image_cell(
                ax_orig, entry.original_image,
                '\u539f\u59cb\u5f71\u50cf',  # 原始影像
            )

            ax_result = fig.add_subplot(gs[row_base + 1, 1])
            disp_img = (
                entry.result_image
                if entry.result_image is not None
                else entry.heatmap_image
            )
            self._render_image_cell(
                ax_result, disp_img,
                '\u6aa2\u6e2c\u7d50\u679c',  # 檢測結果
            )

            # -- Info row --
            ax_info = fig.add_subplot(gs[row_base + 2, :])
            ax_info.axis('off')

            # Metrics line
            metrics_parts = [
                f'\u7570\u5e38\u5206\u6578: {entry.anomaly_score:.4f}',
                f'\u95be\u503c: {entry.threshold:.4f}',
                f'\u7f3a\u9677\u6578: {entry.defect_count}',
                f'\u7f3a\u9677\u9762\u7a4d: {entry.defect_area:.1f}',
            ]
            ax_info.text(
                0.0, 0.80,
                '    '.join(metrics_parts),
                transform=ax_info.transAxes,
                ha='left', va='center',
                fontsize=9, color=_CLR_TEXT,
                fontproperties=self._font_prop,
            )

            # Timestamp if available
            if entry.timestamp:
                ax_info.text(
                    1.0, 0.80,
                    f'\u6642\u9593: {entry.timestamp}',  # 時間
                    transform=ax_info.transAxes,
                    ha='right', va='center',
                    fontsize=8, color=_CLR_TEXT_LIGHT,
                    fontproperties=self._font_prop_small,
                )

            # Inline measurements (up to 5)
            if entry.measurements:
                meas_lines = []
                for m in entry.measurements[:5]:
                    tol_str = ''
                    if 'tolerance_min' in m and 'tolerance_max' in m:
                        tol_str = (
                            f' [{m["tolerance_min"]}'
                            f'~{m["tolerance_max"]}]'
                        )
                    status = (
                        '\u2713' if m.get('in_tolerance', True) else '\u2717'
                    )
                    unit = m.get('unit', '')
                    meas_lines.append(
                        f'  {m.get("name", "")}: '
                        f'{m.get("value", "")} {unit}{tol_str} {status}'
                    )
                ax_info.text(
                    0.0, 0.35,
                    '\n'.join(meas_lines),
                    transform=ax_info.transAxes,
                    ha='left', va='top',
                    fontsize=8, color='#555555',
                    fontproperties=self._font_prop_small,
                    linespacing=1.5,
                )

            # Notes
            if entry.notes:
                ax_info.text(
                    1.0, 0.35,
                    f'\u5099\u8a3b: {entry.notes}',  # 備註
                    transform=ax_info.transAxes,
                    ha='right', va='top',
                    fontsize=8, color='#888888',
                    fontproperties=self._font_prop_small,
                )

        self._stamp_page_number(fig, page_num, total_pages)
        pdf.savefig(fig)
        plt.close(fig)

    def _render_image_cell(
        self,
        ax: plt.Axes,
        image: Optional[np.ndarray],
        title: str,
    ) -> None:
        """Render a single image into *ax* with a title label."""
        ax.set_title(
            title, fontsize=9,
            fontproperties=self._font_prop,
            color='#555555', pad=3,
        )
        if image is not None:
            display_img = image
            if display_img.ndim == 2:
                # Grayscale
                ax.imshow(display_img, cmap='gray', aspect='equal')
            else:
                # Assume BGR from OpenCV; convert to RGB for display
                if display_img.shape[2] == 3:
                    display_img = display_img[..., ::-1]
                ax.imshow(display_img, aspect='equal')
        else:
            ax.text(
                0.5, 0.5,
                '\u7121\u5f71\u50cf',  # 無影像
                transform=ax.transAxes,
                ha='center', va='center',
                fontsize=10, color=_CLR_TEXT_MUTED,
                fontproperties=self._font_prop,
            )
            ax.set_facecolor(_CLR_TABLE_HEADER)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(_CLR_BORDER)

    # ------------------------------------------------------------------
    # Measurement summary page
    # ------------------------------------------------------------------

    def _render_measurement_page(
        self,
        pdf: PdfPages,
        page_num: List[int],
        total_pages: int,
    ) -> None:
        """Measurement summary: statistics table + tolerance chart."""
        w, h = self._page_size()
        fig = plt.figure(figsize=(w, h))
        self._stamp_header(
            fig, '\u91cf\u6e2c\u7d50\u679c\u5f59\u7e3d',  # 量測結果彙總
        )

        # Group measurements by name
        meas_by_name: Dict[str, List[Dict[str, Any]]] = {}
        for entry in self._entries:
            for m in entry.measurements:
                name = m.get('name', 'unknown')
                meas_by_name.setdefault(name, []).append(m)

        gs = gridspec.GridSpec(
            2, 1,
            left=0.08, right=0.94,
            top=0.88, bottom=0.06,
            hspace=0.40,
        )

        # --- Statistics table ---
        ax_tbl = fig.add_subplot(gs[0])
        ax_tbl.axis('off')

        col_labels = [
            '\u91cf\u6e2c\u9805',       # 量測項
            '\u55ae\u4f4d',             # 單位
            '\u6578\u91cf',             # 數量
            '\u5e73\u5747',             # 平均
            '\u6700\u5c0f',             # 最小
            '\u6700\u5927',             # 最大
            '\u6a19\u6e96\u5dee',       # 標準差
            '\u5408\u683c\u7387',       # 合格率
        ]
        table_data: List[List[str]] = []
        tol_data: Dict[str, Dict[str, Any]] = {}

        for name, measurements in meas_by_name.items():
            values = [
                m['value'] for m in measurements
                if 'value' in m and m['value'] is not None
            ]
            if not values:
                continue

            unit = measurements[0].get('unit', '')
            n = len(values)
            arr = np.array(values, dtype=float)
            n_in_tol = sum(
                1 for m in measurements if m.get('in_tolerance', True)
            )
            tol_rate = n_in_tol / n * 100 if n > 0 else 0.0

            table_data.append([
                name, unit, str(n),
                f'{np.mean(arr):.3f}',
                f'{np.min(arr):.3f}',
                f'{np.max(arr):.3f}',
                f'{np.std(arr):.4f}',
                f'{tol_rate:.1f}%',
            ])

            # Collect for tolerance chart
            tol_data[name] = {
                'values': arr,
                'tol_min': measurements[0].get('tolerance_min'),
                'tol_max': measurements[0].get('tolerance_max'),
                'unit': unit,
            }

        if table_data:
            self._add_table(ax_tbl, table_data, col_labels)

        # --- Tolerance visualisation ---
        ax_tol = fig.add_subplot(gs[1])
        if tol_data:
            self._render_tolerance_chart(ax_tol, tol_data)
        else:
            ax_tol.axis('off')
            ax_tol.text(
                0.5, 0.5,
                '\u7121\u516c\u5dee\u8cc7\u6599',  # 無公差資料
                transform=ax_tol.transAxes,
                ha='center', va='center',
                fontsize=11, color=_CLR_TEXT_MUTED,
                fontproperties=self._font_prop,
            )

        self._stamp_page_number(fig, page_num, total_pages)
        pdf.savefig(fig)
        plt.close(fig)

    def _render_tolerance_chart(
        self,
        ax: plt.Axes,
        tol_data: Dict[str, Dict[str, Any]],
    ) -> None:
        """Horizontal box-plot style tolerance chart per measurement."""
        names = list(tol_data.keys())

        for i, name in enumerate(names):
            d = tol_data[name]
            vals = d['values']
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals))

            # Range bar showing mean +/- std
            bar_width = 2 * std_val if std_val > 0 else 0.01
            ax.barh(
                i, bar_width, left=mean_val - std_val,
                height=0.35, color=_CLR_INFO, alpha=0.4,
                edgecolor=_CLR_INFO,
            )

            # Mean marker
            ax.plot(mean_val, i, 'D', color=_CLR_HEADER, markersize=6)

            # Scatter individual values with jitter
            rng = np.random.default_rng(42 + i)
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            colours = []
            for v in vals:
                out_of_tol = False
                if d['tol_min'] is not None and v < d['tol_min']:
                    out_of_tol = True
                if d['tol_max'] is not None and v > d['tol_max']:
                    out_of_tol = True
                colours.append(_CLR_FAIL if out_of_tol else _CLR_PASS)
            ax.scatter(
                vals, i + jitter,
                c=colours, s=14, alpha=0.7, zorder=3, edgecolors='none',
            )

            # Tolerance limit lines
            if d['tol_min'] is not None:
                ax.axvline(
                    d['tol_min'], color=_CLR_FAIL, linestyle=':',
                    linewidth=0.8, alpha=0.6,
                )
            if d['tol_max'] is not None:
                ax.axvline(
                    d['tol_max'], color=_CLR_FAIL, linestyle=':',
                    linewidth=0.8, alpha=0.6,
                )

        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(
            names, fontproperties=self._font_prop, fontsize=9,
        )
        ax.set_title(
            '\u516c\u5dee\u5206\u6790',  # 公差分析
            fontproperties=self._font_prop_bold, fontsize=11,
        )
        ax.set_xlabel(
            '\u91cf\u6e2c\u503c',  # 量測值
            fontproperties=self._font_prop,
        )
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)

    # ------------------------------------------------------------------
    # SPC page
    # ------------------------------------------------------------------

    def _render_spc_page(
        self,
        pdf: PdfPages,
        page_num: List[int],
        total_pages: int,
    ) -> None:
        """SPC analytics: control chart, capability indices, histogram."""
        w, h = self._page_size()
        fig = plt.figure(figsize=(w, h))
        self._stamp_header(
            fig,
            'SPC \u7d71\u8a08\u88fd\u7a0b\u63a7\u5236',  # SPC 統計製程控制
        )

        metrics = self._spc_data.get('metrics')
        trend = self._spc_data.get('trend')

        gs = gridspec.GridSpec(
            3, 2,
            left=0.10, right=0.92,
            top=0.88, bottom=0.06,
            hspace=0.50, wspace=0.30,
        )

        # -- X-bar control chart (top, full width) --
        ax_ctrl = fig.add_subplot(gs[0, :])
        self._render_control_chart(ax_ctrl, metrics, trend)

        # -- Capability indices card (middle-left) --
        ax_cap = fig.add_subplot(gs[1, 0])
        self._render_capability_card(ax_cap, metrics)

        # -- Process capability histogram (middle-right) --
        ax_hist = fig.add_subplot(gs[1, 1])
        self._render_capability_histogram(ax_hist, metrics, trend)

        # -- SPC summary table (bottom, full width) --
        ax_tbl = fig.add_subplot(gs[2, :])
        self._render_spc_summary_table(ax_tbl, metrics)

        self._stamp_page_number(fig, page_num, total_pages)
        pdf.savefig(fig)
        plt.close(fig)

    def _render_control_chart(
        self,
        ax: plt.Axes,
        metrics: Any,
        trend: Any,
    ) -> None:
        """X-bar control chart with UCL / LCL / mean lines."""
        ax.set_title(
            'X-bar \u7ba1\u5236\u5716',  # X-bar 管制圖
            fontproperties=self._font_prop_bold, fontsize=11,
        )

        values: Optional[np.ndarray] = None
        if trend is not None and hasattr(trend, 'values') and trend.values:
            values = np.array(trend.values)
        elif self._entries:
            values = np.array([e.anomaly_score for e in self._entries])

        if values is None or len(values) == 0:
            ax.text(
                0.5, 0.5, '\u7121\u8cc7\u6599',
                transform=ax.transAxes, ha='center', va='center',
                fontproperties=self._font_prop, fontsize=11,
            )
            return

        x = np.arange(len(values))
        mean_val = (
            getattr(metrics, 'mean', None)
            if metrics is not None
            else None
        )
        if mean_val is None:
            mean_val = float(np.mean(values))
        std_val = float(np.std(values)) if len(values) > 1 else 0.0

        ucl = (
            getattr(metrics, 'ucl', None)
            if metrics is not None
            else None
        )
        if ucl is None:
            ucl = mean_val + 3 * std_val

        lcl = (
            getattr(metrics, 'lcl', None)
            if metrics is not None
            else None
        )
        if lcl is None:
            lcl = mean_val - 3 * std_val

        # Classify in-control / out-of-control
        in_ctrl = (values >= lcl) & (values <= ucl)
        out_ctrl = ~in_ctrl

        # Plot data
        ax.plot(
            x, values, '-o', color=_CLR_INFO, markersize=3, linewidth=1,
            zorder=2,
        )
        if np.any(out_ctrl):
            ax.scatter(
                x[out_ctrl], values[out_ctrl],
                color=_CLR_FAIL, s=30, zorder=5,
                label=f'\u7ba1\u5236\u5916 ({int(np.sum(out_ctrl))})',
            )

        # Control limit lines
        ax.axhline(
            ucl, color=_CLR_FAIL, linestyle='--', linewidth=1,
            label=f'UCL = {ucl:.4f}',
        )
        ax.axhline(
            mean_val, color=_CLR_HEADER, linestyle='-', linewidth=1,
            label=f'CL = {mean_val:.4f}',
        )
        ax.axhline(
            lcl, color=_CLR_FAIL, linestyle='--', linewidth=1,
            label=f'LCL = {lcl:.4f}',
        )

        # Moving average overlay
        if (
            trend is not None
            and hasattr(trend, 'moving_avg')
            and trend.moving_avg
        ):
            ma = np.array(trend.moving_avg)
            ax.plot(
                x[:len(ma)], ma,
                color=_CLR_WARN, linewidth=1.2, alpha=0.7,
                label='\u79fb\u52d5\u5e73\u5747',  # 移動平均
            )

        ax.legend(
            prop=self._font_prop, fontsize=7, loc='upper right', ncol=2,
        )
        ax.set_xlabel(
            '\u6a23\u672c\u5e8f\u865f',  # 樣本序號
            fontproperties=self._font_prop,
        )
        ax.set_ylabel(
            '\u7570\u5e38\u5206\u6578',  # 異常分數
            fontproperties=self._font_prop,
        )
        ax.grid(alpha=0.2)

    def _render_capability_card(
        self,
        ax: plt.Axes,
        metrics: Any,
    ) -> None:
        """Render Cp / Cpk / Pp / Ppk as a styled card."""
        ax.axis('off')
        ax.set_title(
            '\u88fd\u7a0b\u80fd\u529b\u6307\u6578',  # 製程能力指數
            fontproperties=self._font_prop_bold, fontsize=11,
        )

        items = [
            ('Cp', getattr(metrics, 'cp', None) if metrics else None),
            ('Cpk', getattr(metrics, 'cpk', None) if metrics else None),
            ('Pp', getattr(metrics, 'pp', None) if metrics else None),
            ('Ppk', getattr(metrics, 'ppk', None) if metrics else None),
        ]

        for i, (label, value) in enumerate(items):
            y_pos = 0.80 - i * 0.20
            ax.text(
                0.10, y_pos, f'{label}:',
                transform=ax.transAxes,
                ha='left', va='center',
                fontsize=12, color=_CLR_HEADER,
                fontproperties=self._font_prop_bold,
            )
            if value is not None:
                if value >= 1.33:
                    clr = _CLR_PASS
                    grade = '\u512a'   # 優
                elif value >= 1.0:
                    clr = _CLR_WARN
                    grade = '\u53ef'   # 可
                else:
                    clr = _CLR_FAIL
                    grade = '\u5dee'   # 差
                ax.text(
                    0.50, y_pos,
                    f'{value:.3f}  ({grade})',
                    transform=ax.transAxes,
                    ha='left', va='center',
                    fontsize=12, color=clr,
                    fontproperties=self._font_prop_bold,
                )
            else:
                ax.text(
                    0.50, y_pos, 'N/A',
                    transform=ax.transAxes,
                    ha='left', va='center',
                    fontsize=12, color=_CLR_TEXT_MUTED,
                    fontproperties=self._font_prop,
                )

    def _render_capability_histogram(
        self,
        ax: plt.Axes,
        metrics: Any,
        trend: Any,
    ) -> None:
        """Process capability histogram with spec limits and normal curve."""
        ax.set_title(
            '\u88fd\u7a0b\u5206\u4f48',  # 製程分佈
            fontproperties=self._font_prop_bold, fontsize=11,
        )

        values: Optional[np.ndarray] = None
        if trend is not None and hasattr(trend, 'values') and trend.values:
            values = np.array(trend.values)
        elif self._entries:
            values = np.array([e.anomaly_score for e in self._entries])

        if values is None or len(values) == 0:
            ax.text(
                0.5, 0.5, '\u7121\u8cc7\u6599',
                transform=ax.transAxes, ha='center', va='center',
                fontproperties=self._font_prop,
            )
            return

        n_bins = min(25, max(5, len(values) // 3))
        ax.hist(
            values, bins=n_bins,
            color=_CLR_INFO, edgecolor='white', alpha=0.8,
            density=True,
        )

        # Overlay normal curve
        mean_val = float(np.mean(values))
        std_val = float(np.std(values))
        if std_val > 0:
            x_range = np.linspace(
                mean_val - 4 * std_val, mean_val + 4 * std_val, 200,
            )
            pdf_vals = (
                np.exp(-0.5 * ((x_range - mean_val) / std_val) ** 2)
                / (std_val * np.sqrt(2 * np.pi))
            )
            ax.plot(
                x_range, pdf_vals, color=_CLR_HEADER, linewidth=1.5,
            )

        # Spec limits
        usl = getattr(metrics, 'usl', None) if metrics else None
        lsl = getattr(metrics, 'lsl', None) if metrics else None
        if usl is not None:
            ax.axvline(
                usl, color=_CLR_FAIL, linestyle='-.',
                linewidth=1.2, label=f'USL={usl:.3f}',
            )
        if lsl is not None:
            ax.axvline(
                lsl, color=_CLR_FAIL, linestyle='-.',
                linewidth=1.2, label=f'LSL={lsl:.3f}',
            )
        if usl is not None or lsl is not None:
            ax.legend(prop=self._font_prop, fontsize=7)

        ax.set_xlabel(
            '\u503c',  # 值
            fontproperties=self._font_prop,
        )
        ax.set_ylabel(
            '\u5bc6\u5ea6',  # 密度
            fontproperties=self._font_prop,
        )

    def _render_spc_summary_table(
        self,
        ax: plt.Axes,
        metrics: Any,
    ) -> None:
        """SPC metrics summary table in 2-column layout."""
        ax.axis('off')

        col_labels = [
            '\u6307\u6a19', '\u503c',    # 指標, 值
            '\u6307\u6a19', '\u503c',    # 指標, 值
        ]

        items: List[Tuple[str, Any]] = [
            ('\u5e73\u5747\u503c (X\u0304)', getattr(metrics, 'mean', None) if metrics else None),
            ('\u6a19\u6e96\u5dee (\u03c3)', getattr(metrics, 'std', None) if metrics else None),
            ('UCL', getattr(metrics, 'ucl', None) if metrics else None),
            ('LCL', getattr(metrics, 'lcl', None) if metrics else None),
            ('USL', getattr(metrics, 'usl', None) if metrics else None),
            ('LSL', getattr(metrics, 'lsl', None) if metrics else None),
            ('Cp', getattr(metrics, 'cp', None) if metrics else None),
            ('Cpk', getattr(metrics, 'cpk', None) if metrics else None),
            ('Pp', getattr(metrics, 'pp', None) if metrics else None),
            ('Ppk', getattr(metrics, 'ppk', None) if metrics else None),
            ('\u6a23\u672c\u6578', getattr(metrics, 'n_samples', None) if metrics else None),
            ('\u7ba1\u5236\u5916\u9ede\u6578', getattr(metrics, 'n_out_of_control', None) if metrics else None),
        ]

        # Pair items into 2-column rows
        rows: List[List[str]] = []
        for r in range(0, len(items), 2):
            row: List[str] = []
            for c in range(2):
                idx = r + c
                if idx < len(items):
                    label, val = items[idx]
                    row.append(label)
                    if isinstance(val, float):
                        row.append(f'{val:.4f}')
                    elif val is not None:
                        row.append(str(val))
                    else:
                        row.append('N/A')
                else:
                    row.extend(['', ''])
            rows.append(row)

        self._add_table(ax, rows, col_labels)

    # ------------------------------------------------------------------
    # Footer / signature page
    # ------------------------------------------------------------------

    def _render_footer_page(
        self,
        pdf: PdfPages,
        page_num: List[int],
        total_pages: int,
    ) -> None:
        """Footer page with signature lines and generation info."""
        w, h = self._page_size()
        fig = plt.figure(figsize=(w, h))
        self._stamp_header(fig, '\u7c3d\u6838\u9801')  # 簽核頁

        # Thin divider below header area
        fig.patches.append(
            FancyBboxPatch(
                (0.10, 0.80), 0.80, 0.002,
                boxstyle='square,pad=0',
                facecolor=_CLR_BORDER,
                edgecolor='none',
                transform=fig.transFigure,
                clip_on=False,
            ),
        )

        # -- Verdict recap --
        total = len(self._entries)
        n_pass = sum(1 for e in self._entries if not e.is_defective)
        n_fail = total - n_pass
        defect_rate = (n_fail / total * 100) if total > 0 else 0.0

        if n_fail == 0:
            verdict_text = '\u5168\u6578\u901a\u904e'  # 全數通過
            verdict_clr = _CLR_PASS
        else:
            verdict_text = (
                f'\u5b58\u5728\u4e0d\u826f '
                f'({n_fail} \u4ef6)'  # 存在不良 (N 件)
            )
            verdict_clr = _CLR_FAIL

        fig.text(
            0.50, 0.74,
            f'\u6aa2\u6e2c\u7d50\u8ad6\uff1a{verdict_text}',  # 檢測結論
            ha='center', va='center',
            fontsize=16, color=verdict_clr,
            fontproperties=self._font_prop_bold,
        )

        recap_parts = [
            f'\u7e3d\u6aa2\u6e2c\u6578: {total}',
            f'\u901a\u904e: {n_pass}',
            f'\u4e0d\u826f: {n_fail}',
            f'\u4e0d\u826f\u7387: {defect_rate:.1f}%',
        ]
        fig.text(
            0.50, 0.69,
            '    '.join(recap_parts),
            ha='center', va='center',
            fontsize=11, color=_CLR_TEXT,
            fontproperties=self._font_prop,
        )

        # -- Signature blocks --
        sig_y = 0.52
        sig_defs = [
            ('\u6aa2\u6e2c\u54e1', 0.15),   # 檢測員
            ('\u5be9\u6838\u54e1', 0.55),   # 審核員
        ]

        for label, x_pos in sig_defs:
            fig.text(
                x_pos, sig_y,
                f'{label}\uff1a',
                ha='left', va='bottom',
                fontsize=12, color=_CLR_TEXT,
                fontproperties=self._font_prop_bold,
            )
            # Signature line
            fig.patches.append(
                FancyBboxPatch(
                    (x_pos + 0.09, sig_y - 0.005), 0.25, 0.001,
                    boxstyle='square,pad=0',
                    facecolor=_CLR_TEXT,
                    edgecolor='none',
                    transform=fig.transFigure,
                    clip_on=False,
                ),
            )
            # Date field
            fig.text(
                x_pos, sig_y - 0.06,
                '\u65e5\u671f\uff1a',  # 日期：
                ha='left', va='bottom',
                fontsize=10, color=_CLR_TEXT_LIGHT,
                fontproperties=self._font_prop,
            )
            fig.patches.append(
                FancyBboxPatch(
                    (x_pos + 0.06, sig_y - 0.065), 0.25, 0.001,
                    boxstyle='square,pad=0',
                    facecolor=_CLR_TEXT_MUTED,
                    edgecolor='none',
                    transform=fig.transFigure,
                    clip_on=False,
                ),
            )

        # -- Generation info box --
        now = datetime.now()
        info_y = 0.25
        fig.patches.append(
            FancyBboxPatch(
                (0.08, info_y - 0.07), 0.84, 0.18,
                boxstyle='round,pad=0.01',
                facecolor=_CLR_BG_LIGHT,
                edgecolor=_CLR_BORDER,
                linewidth=0.5,
                transform=fig.transFigure,
                clip_on=False,
            ),
        )

        gen_info_items = [
            (
                '\u5831\u544a\u7522\u751f\u6642\u9593',
                now.strftime('%Y-%m-%d %H:%M:%S'),
            ),
            (
                '\u7522\u54c1\u540d\u7a31',
                self.config.product_name or 'N/A',
            ),
            ('\u6279\u865f', self.config.lot_number or 'N/A'),
            ('\u7522\u7dda', self.config.line_id or 'N/A'),
            ('\u64cd\u4f5c\u54e1', self.config.operator or 'N/A'),
            ('\u8edf\u9ad4\u7248\u672c', _SOFTWARE_VERSION),
        ]
        for i, (label, value) in enumerate(gen_info_items):
            y = info_y + 0.08 - i * 0.025
            fig.text(
                0.14, y,
                f'{label}\uff1a{value}',
                ha='left', va='center',
                fontsize=8, color=_CLR_TEXT_LIGHT,
                fontproperties=self._font_prop_small,
            )

        # Disclaimer
        fig.text(
            0.50, 0.06,
            '\u672c\u5831\u544a\u7531\u7cfb\u7d71\u81ea\u52d5\u7522\u751f'
            '\uff0c\u50c5\u4f9b\u53c3\u8003\u3002',
            # 本報告由系統自動產生，僅供參考。
            ha='center', va='center',
            fontsize=8, color='#aaaaaa',
            fontproperties=self._font_prop_small,
        )

        self._stamp_page_number(fig, page_num, total_pages)
        pdf.savefig(fig)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Table rendering helper
    # ------------------------------------------------------------------

    def _add_table(
        self,
        ax: plt.Axes,
        data: List[List[str]],
        col_labels: List[str],
        row_colors: Optional[List[str]] = None,
    ) -> None:
        """Render a styled table on an axes that has ``axis('off')``.

        Parameters
        ----------
        ax:
            Target axes (should already have ``ax.axis('off')``).
        data:
            List of rows, each row a list of cell strings.
        col_labels:
            Column header labels.
        row_colors:
            Optional per-row background colour overrides.
        """
        if not data:
            return

        n_cols = len(col_labels)
        # Pad each row to the expected column count
        padded = [
            row + [''] * (n_cols - len(row)) for row in data
        ]

        table = ax.table(
            cellText=padded,
            colLabels=col_labels,
            loc='center',
            cellLoc='center',
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.4)

        # Style header row (row index 0 in the table object)
        for j in range(n_cols):
            cell = table[0, j]
            cell.set_facecolor(_CLR_TABLE_HEADER)
            cell.set_edgecolor(_CLR_BORDER)
            cell.set_text_props(
                fontproperties=self._font_prop_bold,
                fontsize=8,
                color=_CLR_HEADER,
            )

        # Style data rows (row indices 1 .. len(padded))
        for i in range(1, len(padded) + 1):
            for j in range(n_cols):
                cell = table[i, j]
                cell.set_edgecolor(_CLR_BORDER)
                cell.set_text_props(
                    fontproperties=self._font_prop_small,
                    fontsize=8,
                )

                # Alternate row background
                if row_colors and (i - 1) < len(row_colors):
                    cell.set_facecolor(row_colors[i - 1])
                else:
                    cell.set_facecolor(
                        'white' if i % 2 == 1 else _CLR_TABLE_ALT,
                    )

                # Highlight pass / fail verdict text
                text_val = padded[i - 1][j] if j < len(padded[i - 1]) else ''
                if text_val == '\u901a\u904e':  # 通過
                    cell.set_text_props(
                        fontproperties=self._font_prop_bold,
                        fontsize=8, color=_CLR_PASS,
                    )
                elif text_val == '\u4e0d\u826f':  # 不良
                    cell.set_text_props(
                        fontproperties=self._font_prop_bold,
                        fontsize=8, color=_CLR_FAIL,
                    )


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

@log_operation(logger)
def generate_single_report(
    image: np.ndarray,
    result_image: np.ndarray,
    anomaly_score: float,
    threshold: float,
    output_path: str,
    config: Optional[ReportConfig] = None,
    measurements: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Generate a quick single-image inspection report.

    Parameters
    ----------
    image:
        Original input image (numpy array, HWC or HW).
    result_image:
        Annotated result image.
    anomaly_score:
        Anomaly score for the sample.
    threshold:
        Decision threshold.
    output_path:
        Destination PDF path.
    config:
        Optional :class:`ReportConfig`.
    measurements:
        Optional list of measurement dicts.

    Returns
    -------
    str
        The *output_path* upon successful generation.
    """
    gen = PDFReportGenerator(config)
    gen.add_entry(InspectionEntry(
        original_image=image,
        result_image=result_image,
        anomaly_score=anomaly_score,
        threshold=threshold,
        is_defective=anomaly_score > threshold,
        measurements=measurements or [],
        timestamp=datetime.now().isoformat(),
    ))
    return gen.generate(output_path)


@log_operation(logger)
def generate_batch_report(
    entries: List[InspectionEntry],
    output_path: str,
    config: Optional[ReportConfig] = None,
    spc_metrics: Any = None,
    trend_data: Any = None,
) -> str:
    """Generate a batch inspection report.

    Parameters
    ----------
    entries:
        List of :class:`InspectionEntry` objects.
    output_path:
        Destination PDF path.
    config:
        Optional :class:`ReportConfig`.
    spc_metrics:
        Optional SPC metrics (e.g. ``results_db.SPCMetrics``).
    trend_data:
        Optional trend data (e.g. ``results_db.TrendData``).

    Returns
    -------
    str
        The *output_path* upon successful generation.
    """
    gen = PDFReportGenerator(config)
    gen.add_entries(entries)
    if spc_metrics is not None:
        gen.set_spc_data(spc_metrics, trend_data)
    return gen.generate(output_path)
