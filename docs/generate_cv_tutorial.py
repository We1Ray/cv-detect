# -*- coding: utf-8 -*-
"""CV 瑕疵檢測完整教程 PDF 生成器（繁體中文版）"""

import os
from fpdf import FPDF

# ============================================================
# 常數
# ============================================================
PAGE_W = 190          # 可用寬度 (mm)
PAGE_BOTTOM = 270     # 安全底線 (A4=297, margin=20, 留 7mm)
MARGIN = 10

# ============================================================
# PDF 類別
# ============================================================
class P(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=22)
        fd = r"C:\Windows\Fonts"
        for name, reg, bold in [
            ("MSJH", "msjh.ttc", "msjhbd.ttc"),
            ("MSYH", "msyh.ttc", "msyhbd.ttc"),
        ]:
            r_path = os.path.join(fd, reg)
            b_path = os.path.join(fd, bold)
            if os.path.exists(r_path):
                self.add_font(name, "", r_path)
                if os.path.exists(b_path):
                    self.add_font(name, "B", b_path)
                self.F = name
                break
        else:
            self.F = "Helvetica"
        self.ch = 0
        self.sc = 0

    # --- header / footer ---
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font(self.F, "B", 8)
        self.set_text_color(130, 130, 130)
        self.cell(95, 7, "電腦視覺瑕疵檢測教程", new_x="RIGHT", new_y="TOP")
        self.cell(95, 7, f"{self.page_no()}", new_x="LMARGIN", new_y="NEXT", align="R")
        self.set_draw_color(200, 200, 200)
        self.line(MARGIN, 12, MARGIN + PAGE_W, 12)
        self.ln(4)

    def footer(self):
        self.set_y(-14)
        self.set_font(self.F, "", 7)
        self.set_text_color(160, 160, 160)
        self.cell(0, 8, "CV 瑕疵檢測完整教程", align="C")

    # --- 預估高度 ---
    def _est_h(self, txt, w, font, style, size, lh):
        self.set_font(font, style, size)
        lines = self.multi_cell(w, lh, txt, dry_run=True, output="LINES")
        return len(lines) * lh

    def _space_left(self):
        return PAGE_BOTTOM - self.get_y()

    def _ensure_space(self, h):
        if self._space_left() < h:
            self.add_page()

    # --- 章節標題 ---
    def ch_title(self, title):
        self.ch += 1
        self.sc = 0
        self.add_page()
        self.set_font(self.F, "B", 22)
        self.set_text_color(0, 50, 140)
        self.cell(0, 14, f"第 {self.ch} 章", new_x="LMARGIN", new_y="NEXT")
        self.set_font(self.F, "B", 15)
        self.cell(0, 11, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(0, 90, 180)
        self.set_line_width(0.7)
        y = self.get_y() + 1
        self.line(MARGIN, y, MARGIN + PAGE_W, y)
        self.ln(7)

    def sec_title(self, title):
        self.sc += 1
        self._ensure_space(18)
        self.ln(3)
        self.set_font(self.F, "B", 12)
        self.set_text_color(0, 90, 140)
        self.cell(0, 9, f"{self.ch}.{self.sc}  {title}", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(180, 200, 220)
        self.set_line_width(0.3)
        self.line(MARGIN, self.get_y(), 100, self.get_y())
        self.ln(3)

    def sub_sec(self, title):
        self._ensure_space(12)
        self.ln(1)
        self.set_font(self.F, "B", 10)
        self.set_text_color(70, 70, 70)
        self.cell(0, 7, f"  >> {title}", new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    # --- 正文 ---
    def txt(self, text):
        self.set_font(self.F, "", 9.5)
        self.set_text_color(35, 35, 35)
        self.multi_cell(PAGE_W, 5.5, text)
        self.ln(2)

    # --- 項目符號列表 ---
    def blist(self, items):
        self.set_font(self.F, "", 9.5)
        self.set_text_color(35, 35, 35)
        for item in items:
            self._ensure_space(8)
            x = self.get_x()
            self.cell(6, 5.5, "", new_x="RIGHT", new_y="TOP")
            self.cell(4, 5.5, chr(8226), new_x="RIGHT", new_y="TOP")
            self.multi_cell(PAGE_W - 10, 5.5, item)
        self.ln(2)

    # ============================================================
    # 安全 Box 繪製：先預估高度，空間不足先換頁
    # ============================================================
    def _draw_box(self, bg, border_c, title_font_sz, title_color, title_prefix,
                  title, body, body_font_sz=9, body_color=(55,55,55)):
        # 預估總高度
        th = 7  # title height
        bh = self._est_h(body, PAGE_W - 4, self.F, "", body_font_sz, 5) + 2
        total = th + bh + 4
        self._ensure_space(min(total, 80))  # 若內容超大則至少確保 80mm

        y0 = self.get_y()
        page0 = self.page

        # title
        self.set_fill_color(*bg)
        self.set_font(self.F, "B", title_font_sz)
        self.set_text_color(*title_color)
        self.cell(PAGE_W, th, f"  {title_prefix}{title}",
                  new_x="LMARGIN", new_y="NEXT", fill=True)

        # body
        self.set_font(self.F, "", body_font_sz)
        self.set_text_color(*body_color)
        self.multi_cell(PAGE_W, 5, f"  {body}", fill=True)
        self.ln(1)

        # 畫邊框（僅當未跨頁時）
        y1 = self.get_y()
        if self.page == page0:
            self.set_draw_color(*border_c)
            self.set_line_width(0.4)
            self.rect(MARGIN, y0, PAGE_W, y1 - y0)
        else:
            # 跨頁：在當前頁頂部畫一條上邊線表示延續
            self.set_draw_color(*border_c)
            self.set_line_width(0.4)
            self.line(MARGIN, 20, MARGIN + PAGE_W, 20)
            self.line(MARGIN, y1, MARGIN + PAGE_W, y1)
        self.ln(2)

    # --- 公式框 ---
    def formula(self, title, expr, desc=""):
        fh = len(expr.split("\n")) * 5.5 + 8
        dh = self._est_h(desc, PAGE_W - 4, self.F, "", 8.5, 4.5) if desc else 0
        total = fh + dh + 6
        self._ensure_space(min(total, 90))

        y0 = self.get_y()
        page0 = self.page

        self.set_fill_color(243, 247, 255)
        self.set_font(self.F, "B", 9.5)
        self.set_text_color(0, 50, 140)
        self.cell(PAGE_W, 7, f"  公式：{title}",
                  new_x="LMARGIN", new_y="NEXT", fill=True)

        self.set_text_color(170, 0, 0)
        for line in expr.split("\n"):
            if self._is_ascii(line):
                self.set_font("Courier", "", 9)
            else:
                self.set_font(self.F, "", 9)
            self.cell(PAGE_W, 5.5, f"    {line}",
                      new_x="LMARGIN", new_y="NEXT", fill=True)

        if desc:
            self.set_font(self.F, "", 8.5)
            self.set_text_color(80, 80, 80)
            self.multi_cell(PAGE_W, 4.5, f"  {desc}", fill=True)

        y1 = self.get_y()
        self.set_draw_color(0, 90, 180)
        self.set_line_width(0.35)
        if self.page == page0:
            self.rect(MARGIN, y0, PAGE_W, y1 - y0)
        self.ln(3)

    # --- 程式碼框 ---
    def _is_ascii(self, s):
        return all(ord(c) < 128 for c in s)

    def _code_font(self, line):
        if self._is_ascii(line):
            self.set_font("Courier", "", 7.5)
        else:
            self.set_font(self.F, "", 7.5)

    def code(self, src, lang="python"):
        lines = src.strip().split("\n")
        est = (len(lines) + 1) * 4.5 + 6
        self._ensure_space(min(est, 60))

        y0 = self.get_y()
        page0 = self.page
        self.set_fill_color(238, 238, 238)
        self._code_font(lang)
        self.set_text_color(25, 25, 25)
        self.cell(PAGE_W, 4.5, f"  [{lang}]",
                  new_x="LMARGIN", new_y="NEXT", fill=True)

        for line in lines:
            if self._space_left() < 6:
                y1 = self.get_y()
                if self.page == page0:
                    self.set_draw_color(180, 180, 180)
                    self.set_line_width(0.3)
                    self.rect(MARGIN, y0, PAGE_W, y1 - y0)
                self.add_page()
                y0 = self.get_y()
                page0 = self.page
                self.set_fill_color(238, 238, 238)
                self.set_text_color(25, 25, 25)
            self._code_font(line)
            self.cell(PAGE_W, 4.5, f"  {line}",
                      new_x="LMARGIN", new_y="NEXT", fill=True)

        y1 = self.get_y()
        self.set_draw_color(180, 180, 180)
        self.set_line_width(0.3)
        self.rect(MARGIN, y0, PAGE_W, y1 - y0)
        self.ln(3)

    # --- 案例研究框 ---
    def case(self, title, body):
        self._draw_box(
            bg=(255, 247, 228), border_c=(230, 140, 0),
            title_font_sz=9.5, title_color=(170, 90, 0),
            title_prefix="案例研究：", title=title,
            body=body, body_font_sz=9, body_color=(55, 55, 55))

    # --- 提示框 ---
    def tip(self, text):
        self._draw_box(
            bg=(230, 250, 230), border_c=(0, 140, 0),
            title_font_sz=9, title_color=(0, 100, 0),
            title_prefix="", title="提示",
            body=text, body_font_sz=8.5, body_color=(30, 70, 30))

    # --- 注意框 ---
    def warn(self, text):
        self._draw_box(
            bg=(255, 233, 233), border_c=(190, 0, 0),
            title_font_sz=9, title_color=(170, 0, 0),
            title_prefix="", title="注意",
            body=text, body_font_sz=8.5, body_color=(90, 25, 25))

    # --- 表格 ---
    def table(self, headers, rows):
        n = len(headers)
        if n == 4:
            cw = [34, 50, 56, 50]
        elif n == 3:
            cw = [48, 71, 71]
        elif n == 5:
            cw = [30, 35, 35, 45, 45]
        else:
            cw = [PAGE_W // n] * n

        def draw_header():
            self.set_font(self.F, "B", 8)
            self.set_fill_color(0, 90, 170)
            self.set_text_color(255, 255, 255)
            for i, h in enumerate(headers):
                self.cell(cw[i], 7, h, border=1, align="C", fill=True,
                          new_x="RIGHT", new_y="TOP")
            self.ln()

        self._ensure_space(25)
        draw_header()

        self.set_font(self.F, "", 8)
        self.set_text_color(35, 35, 35)
        alt = False
        for row in rows:
            if self._space_left() < 9:
                self.add_page()
                draw_header()
            self.set_fill_color(238, 243, 255) if alt else self.set_fill_color(255, 255, 255)
            for i, ct in enumerate(row):
                self.cell(cw[i], 7, ct, border=1, fill=True,
                          new_x="RIGHT", new_y="TOP")
            self.ln()
            alt = not alt
        self.ln(3)


# ============================================================
# 主程式：組合所有章節並生成 PDF
# ============================================================
def build():
    from _ch1_6 import (write_cover_and_toc, write_ch1, write_ch2,
                         write_ch3, write_ch4, write_ch5, write_ch6)
    from _ch7_13 import (write_ch7, write_ch8, write_ch9, write_ch10,
                          write_ch11, write_ch12, write_ch13, write_ch14)
    from _ch15_19 import (write_ch15, write_ch16, write_ch17,
                           write_ch18, write_ch19)
    from _ch20_25 import (write_ch20, write_ch21, write_ch22,
                           write_ch23, write_ch24, write_ch25)

    pdf = P()
    pdf.alias_nb_pages()

    write_cover_and_toc(pdf)
    write_ch1(pdf)
    write_ch2(pdf)
    write_ch3(pdf)
    write_ch4(pdf)
    write_ch5(pdf)
    write_ch6(pdf)
    write_ch7(pdf)
    write_ch8(pdf)
    write_ch9(pdf)
    write_ch10(pdf)
    write_ch11(pdf)
    write_ch12(pdf)
    write_ch13(pdf)
    write_ch14(pdf)
    write_ch15(pdf)
    write_ch16(pdf)
    write_ch17(pdf)
    write_ch18(pdf)
    write_ch19(pdf)
    write_ch20(pdf)
    write_ch21(pdf)
    write_ch22(pdf)
    write_ch23(pdf)
    write_ch24(pdf)
    write_ch25(pdf)

    out = os.path.join(os.path.dirname(__file__),
                       "CV_Defect_Detection_Tutorial.pdf")
    pdf.output(out)
    print(f"PDF 已生成: {out}")
    print(f"總頁數: {pdf.page_no()}")


if __name__ == "__main__":
    build()
