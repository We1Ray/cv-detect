"""
gui/image_viewer.py - HALCON HDevelop 風格的可互動影像檢視器

功能：
- OpenCV/NumPy 影像顯示（BGR -> RGB -> PhotoImage）
- 滑鼠滾輪縮放（以游標為中心）
- 拖曳平移
- 自動適配視窗（雙擊或按鈕）
- 游標像素座標與數值即時顯示
- 十字游標顯示
- 矩形選取縮放
- 半透明疊加層（瑕疵區域、ROI）
- 格線顯示
"""

from __future__ import annotations

import tkinter as tk
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk


class ImageViewer(tk.Canvas):
    """HALCON HDevelop 風格的可縮放、可平移影像檢視器。"""

    def __init__(self, master: tk.Widget, **kwargs) -> None:
        kwargs.setdefault("bg", "#1e1e1e")
        kwargs.setdefault("highlightthickness", 0)
        super().__init__(master, **kwargs)

        # ── 影像資料 ──
        self._cv_image: Optional[np.ndarray] = None
        self._display_image: Optional[np.ndarray] = None
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._image_id: Optional[int] = None

        # ── 疊加層 ──
        self._overlays: List[Dict] = []

        # ── Region 疊加 ──
        self._region_overlay: Optional[np.ndarray] = None
        self._region_alpha: float = 0.5

        # ── 縮放與平移 ──
        self._scale: float = 1.0
        self._offset_x: float = 0.0
        self._offset_y: float = 0.0

        # ── 拖曳 ──
        self._drag_start_x: int = 0
        self._drag_start_y: int = 0
        self._is_dragging: bool = False

        # ── 矩形選取縮放 ──
        self._rect_zoom_mode: bool = False
        self._rect_start: Optional[Tuple[int, int]] = None
        self._rect_id: Optional[int] = None

        # ── 顯示選項 ──
        self._show_crosshair: bool = False
        self._show_grid: bool = False
        self._crosshair_ids: List[int] = []
        self._grid_ids: List[int] = []

        # ── 回呼 ──
        self._pixel_info_callback: Optional[Callable] = None
        self._zoom_change_callback: Optional[Callable] = None

        # ── 綁定事件 ──
        self.bind("<MouseWheel>", self._on_mousewheel)
        self.bind("<Button-4>", self._on_mousewheel_linux_up)
        self.bind("<Button-5>", self._on_mousewheel_linux_down)
        self.bind("<ButtonPress-1>", self._on_left_press)
        self.bind("<B1-Motion>", self._on_left_motion)
        self.bind("<ButtonRelease-1>", self._on_left_release)
        self.bind("<Motion>", self._on_mouse_move)
        self.bind("<Configure>", self._on_resize)
        self.bind("<Double-Button-1>", self._on_double_click)

        # 防抖重繪
        self._resize_after_id: Optional[str] = None

    # ================================================================== #
    #  公開 API                                                           #
    # ================================================================== #

    def set_image(self, image: np.ndarray) -> None:
        """設定要顯示的影像。

        Args:
            image: OpenCV BGR 或灰階 NumPy 陣列。
        """
        self._cv_image = image.copy()
        self._display_image = self._to_rgb_uint8(image)
        self._overlays.clear()
        self._region_overlay = None
        self.fit_to_window()

    def set_region_overlay(self, region, alpha: float = 0.5) -> None:
        """設定 Region 彩色疊加層。

        Args:
            region: core.region.Region 物件。
            alpha: 透明度 (0.0 ~ 1.0)。
        """
        if region is None or self._cv_image is None:
            self._region_overlay = None
            self._redraw()
            return

        self._region_alpha = alpha
        try:
            from core.region_ops import region_to_display_image
            self._region_overlay = region_to_display_image(region, self._cv_image)
            # Build an RGB display from the overlay
            disp = self._to_rgb_uint8(self._region_overlay)
            self._display_image = disp
        except Exception:
            self._region_overlay = None
        self._redraw()

    def get_image(self) -> Optional[np.ndarray]:
        """取得目前顯示的原始影像。"""
        return self._cv_image

    def clear(self) -> None:
        """清除顯示。"""
        self._cv_image = None
        self._display_image = None
        self._photo = None
        self._overlays.clear()
        self.delete("all")

    def fit_to_window(self) -> None:
        """自動縮放影像以適配 Canvas 大小。"""
        if self._display_image is None:
            return
        canvas_w = self.winfo_width()
        canvas_h = self.winfo_height()
        if canvas_w <= 1 or canvas_h <= 1:
            self.after(50, self.fit_to_window)
            return

        img_h, img_w = self._display_image.shape[:2]
        scale_x = canvas_w / img_w
        scale_y = canvas_h / img_h
        self._scale = min(scale_x, scale_y) * 0.95
        self._offset_x = (canvas_w - img_w * self._scale) / 2
        self._offset_y = (canvas_h - img_h * self._scale) / 2
        self._redraw()
        self._notify_zoom()

    def zoom_in(self) -> None:
        """放大 (以畫布中心為基準)。"""
        cx = self.winfo_width() / 2
        cy = self.winfo_height() / 2
        self._zoom_at(int(cx), int(cy), 1.25)

    def zoom_out(self) -> None:
        """縮小 (以畫布中心為基準)。"""
        cx = self.winfo_width() / 2
        cy = self.winfo_height() / 2
        self._zoom_at(int(cx), int(cy), 1 / 1.25)

    def zoom_100(self) -> None:
        """設定為 100% 縮放。"""
        if self._display_image is None:
            return
        canvas_w = self.winfo_width()
        canvas_h = self.winfo_height()
        img_h, img_w = self._display_image.shape[:2]
        self._scale = 1.0
        self._offset_x = (canvas_w - img_w) / 2
        self._offset_y = (canvas_h - img_h) / 2
        self._redraw()
        self._notify_zoom()

    def get_zoom_percent(self) -> float:
        """取得目前縮放百分比。"""
        return self._scale * 100.0

    def set_crosshair(self, enabled: bool) -> None:
        """啟用或停用十字游標。"""
        self._show_crosshair = enabled
        if not enabled:
            for cid in self._crosshair_ids:
                self.delete(cid)
            self._crosshair_ids.clear()

    def set_grid(self, enabled: bool) -> None:
        """啟用或停用格線。"""
        self._show_grid = enabled
        if enabled:
            self._draw_grid()
        else:
            for gid in self._grid_ids:
                self.delete(gid)
            self._grid_ids.clear()

    def set_rect_zoom_mode(self, enabled: bool) -> None:
        """啟用或停用矩形選取縮放模式。"""
        self._rect_zoom_mode = enabled

    def add_overlay_regions(
        self,
        regions: List[Dict],
        color: str = "#FF0000",
        alpha: float = 0.5,
    ) -> None:
        """新增瑕疵區域疊加層。"""
        self._overlays.append({
            "regions": regions,
            "color": color,
            "alpha": alpha,
        })
        self._redraw()

    def set_pixel_info_callback(self, callback: Callable) -> None:
        """設定像素資訊回呼。

        callback(x: int, y: int, value: str)
        """
        self._pixel_info_callback = callback

    def set_zoom_change_callback(self, callback: Callable) -> None:
        """設定縮放變更回呼。

        callback(zoom_percent: float)
        """
        self._zoom_change_callback = callback

    # ================================================================== #
    #  內部：影像轉換                                                      #
    # ================================================================== #

    @staticmethod
    def _to_rgb_uint8(image: np.ndarray) -> np.ndarray:
        """將任意影像轉為 RGB uint8。"""
        if image.dtype != np.uint8:
            min_v = image.min()
            max_v = image.max()
            if max_v - min_v > 0:
                norm = ((image - min_v) / (max_v - min_v) * 255).astype(np.uint8)
            else:
                norm = np.zeros_like(image, dtype=np.uint8)
        else:
            norm = image.copy()

        if norm.ndim == 2:
            return cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)
        elif norm.ndim == 3 and norm.shape[2] == 3:
            return cv2.cvtColor(norm, cv2.COLOR_BGR2RGB)
        elif norm.ndim == 3 and norm.shape[2] == 4:
            return cv2.cvtColor(norm, cv2.COLOR_BGRA2RGB)
        return norm

    # ================================================================== #
    #  內部：繪製                                                          #
    # ================================================================== #

    def _redraw(self) -> None:
        """根據目前縮放與偏移重新繪製影像。"""
        if self._display_image is None:
            return

        img_h, img_w = self._display_image.shape[:2]
        new_w = max(1, int(img_w * self._scale))
        new_h = max(1, int(img_h * self._scale))

        pil_image = Image.fromarray(self._display_image)
        resample = Image.NEAREST if self._scale > 3.0 else Image.LANCZOS
        pil_image = pil_image.resize((new_w, new_h), resample)

        self._photo = ImageTk.PhotoImage(pil_image)

        self.delete("all")
        self._image_id = self.create_image(
            int(self._offset_x), int(self._offset_y),
            anchor=tk.NW, image=self._photo,
        )

        # 繪製疊加區域矩形
        for overlay_info in self._overlays:
            color = overlay_info["color"]
            for region in overlay_info["regions"]:
                bbox = region.get("bbox")
                if bbox:
                    bx, by, bw, bh = bbox
                    x1 = int(self._offset_x + bx * self._scale)
                    y1 = int(self._offset_y + by * self._scale)
                    x2 = int(self._offset_x + (bx + bw) * self._scale)
                    y2 = int(self._offset_y + (by + bh) * self._scale)
                    self.create_rectangle(
                        x1, y1, x2, y2,
                        outline=color, width=2, dash=(4, 2),
                    )

        if self._show_grid:
            self._draw_grid()

    def _draw_grid(self) -> None:
        """在影像上繪製格線。"""
        for gid in self._grid_ids:
            self.delete(gid)
        self._grid_ids.clear()

        if self._display_image is None or not self._show_grid:
            return

        img_h, img_w = self._display_image.shape[:2]
        step = 50

        for x in range(0, img_w, step):
            cx = int(self._offset_x + x * self._scale)
            cy1 = int(self._offset_y)
            cy2 = int(self._offset_y + img_h * self._scale)
            gid = self.create_line(cx, cy1, cx, cy2, fill="#444444", dash=(2, 4))
            self._grid_ids.append(gid)

        for y in range(0, img_h, step):
            cy = int(self._offset_y + y * self._scale)
            cx1 = int(self._offset_x)
            cx2 = int(self._offset_x + img_w * self._scale)
            gid = self.create_line(cx1, cy, cx2, cy, fill="#444444", dash=(2, 4))
            self._grid_ids.append(gid)

    def _draw_crosshair(self, cx: int, cy: int) -> None:
        """在指定位置繪製十字游標。"""
        for cid in self._crosshair_ids:
            self.delete(cid)
        self._crosshair_ids.clear()

        if not self._show_crosshair:
            return

        canvas_w = self.winfo_width()
        canvas_h = self.winfo_height()
        h_line = self.create_line(0, cy, canvas_w, cy, fill="#00FF00", dash=(3, 3))
        v_line = self.create_line(cx, 0, cx, canvas_h, fill="#00FF00", dash=(3, 3))
        self._crosshair_ids = [h_line, v_line]

    # ================================================================== #
    #  內部：座標轉換                                                      #
    # ================================================================== #

    def _canvas_to_image(self, cx: int, cy: int) -> Tuple[int, int]:
        """Canvas 座標 -> 影像像素座標。"""
        if self._scale == 0:
            return -1, -1
        ix = int((cx - self._offset_x) / self._scale)
        iy = int((cy - self._offset_y) / self._scale)
        return ix, iy

    # ================================================================== #
    #  內部：事件處理                                                      #
    # ================================================================== #

    def _on_mousewheel(self, event: tk.Event) -> None:
        if self._display_image is None:
            return
        factor = 1.15 if event.delta > 0 else 1 / 1.15
        self._zoom_at(event.x, event.y, factor)

    def _on_mousewheel_linux_up(self, event: tk.Event) -> None:
        self._zoom_at(event.x, event.y, 1.15)

    def _on_mousewheel_linux_down(self, event: tk.Event) -> None:
        self._zoom_at(event.x, event.y, 1 / 1.15)

    def _zoom_at(self, cx: int, cy: int, factor: float) -> None:
        new_scale = self._scale * factor
        new_scale = max(0.01, min(new_scale, 80.0))
        self._offset_x = cx - (cx - self._offset_x) * (new_scale / self._scale)
        self._offset_y = cy - (cy - self._offset_y) * (new_scale / self._scale)
        self._scale = new_scale
        self._redraw()
        self._notify_zoom()

    def _on_left_press(self, event: tk.Event) -> None:
        if self._rect_zoom_mode:
            self._rect_start = (event.x, event.y)
            return
        self._drag_start_x = event.x
        self._drag_start_y = event.y
        self._is_dragging = True

    def _on_left_motion(self, event: tk.Event) -> None:
        if self._rect_zoom_mode and self._rect_start is not None:
            if self._rect_id is not None:
                self.delete(self._rect_id)
            sx, sy = self._rect_start
            self._rect_id = self.create_rectangle(
                sx, sy, event.x, event.y,
                outline="#00FF00", width=2, dash=(4, 2),
            )
            return

        if not self._is_dragging:
            return
        dx = event.x - self._drag_start_x
        dy = event.y - self._drag_start_y
        self._offset_x += dx
        self._offset_y += dy
        self._drag_start_x = event.x
        self._drag_start_y = event.y
        self._redraw()

    def _on_left_release(self, event: tk.Event) -> None:
        if self._rect_zoom_mode and self._rect_start is not None:
            if self._rect_id is not None:
                self.delete(self._rect_id)
                self._rect_id = None

            sx, sy = self._rect_start
            ex, ey = event.x, event.y
            self._rect_start = None

            if abs(ex - sx) > 10 and abs(ey - sy) > 10:
                self._zoom_to_rect(sx, sy, ex, ey)
            return

        self._is_dragging = False

    def _zoom_to_rect(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """縮放到選取的矩形區域。"""
        if self._display_image is None:
            return

        ix1, iy1 = self._canvas_to_image(min(x1, x2), min(y1, y2))
        ix2, iy2 = self._canvas_to_image(max(x1, x2), max(y1, y2))

        img_h, img_w = self._display_image.shape[:2]
        ix1 = max(0, ix1)
        iy1 = max(0, iy1)
        ix2 = min(img_w, ix2)
        iy2 = min(img_h, iy2)

        rect_w = ix2 - ix1
        rect_h = iy2 - iy1
        if rect_w <= 0 or rect_h <= 0:
            return

        canvas_w = self.winfo_width()
        canvas_h = self.winfo_height()

        scale_x = canvas_w / rect_w
        scale_y = canvas_h / rect_h
        self._scale = min(scale_x, scale_y) * 0.95

        center_ix = (ix1 + ix2) / 2
        center_iy = (iy1 + iy2) / 2
        self._offset_x = canvas_w / 2 - center_ix * self._scale
        self._offset_y = canvas_h / 2 - center_iy * self._scale

        self._redraw()
        self._notify_zoom()

    def _on_mouse_move(self, event: tk.Event) -> None:
        if self._show_crosshair:
            self._draw_crosshair(event.x, event.y)

        if self._cv_image is None or self._pixel_info_callback is None:
            return

        ix, iy = self._canvas_to_image(event.x, event.y)
        h, w = self._cv_image.shape[:2]

        if 0 <= ix < w and 0 <= iy < h:
            val = self._cv_image[iy, ix]
            if isinstance(val, np.ndarray):
                val_str = str(val.tolist())
            else:
                val_str = str(int(val) if self._cv_image.dtype == np.uint8 else f"{float(val):.2f}")
            self._pixel_info_callback(ix, iy, val_str)

    def _on_resize(self, event: tk.Event) -> None:
        if self._display_image is None:
            return
        if self._resize_after_id is not None:
            self.after_cancel(self._resize_after_id)
        self._resize_after_id = self.after(150, self.fit_to_window)

    def _on_double_click(self, event: tk.Event) -> None:
        self.fit_to_window()

    def _notify_zoom(self) -> None:
        if self._zoom_change_callback:
            self._zoom_change_callback(self.get_zoom_percent())
