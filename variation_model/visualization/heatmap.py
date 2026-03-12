"""
visualization/heatmap.py - 視覺化工具：熱力圖、疊加、組合影像

提供差異熱力圖、瑕疵疊加、模型閾值視覺化、以及結果組合圖生成功能。
"""

from __future__ import annotations

from typing import Dict, Optional

import cv2
import numpy as np

from core.variation_model import VariationModel


def create_difference_heatmap(
    diff_image: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """將差異影像轉換為偽彩色熱力圖。

    Args:
        diff_image: 差異影像（float64，可含正負值）。
        colormap: OpenCV 色彩映射表（預設 JET）。

    Returns:
        BGR 偽彩色熱力圖（uint8）。
    """
    # 取絕對值並正規化至 0-255
    if diff_image.ndim == 3:
        abs_diff = np.mean(np.abs(diff_image), axis=2)
    else:
        abs_diff = np.abs(diff_image)

    max_val = abs_diff.max()
    if max_val > 0:
        normalized = (abs_diff / max_val * 255).astype(np.uint8)
    else:
        normalized = np.zeros(abs_diff.shape, dtype=np.uint8)

    heatmap = cv2.applyColorMap(normalized, colormap)
    return heatmap


def create_defect_overlay(
    original: np.ndarray,
    mask: np.ndarray,
    too_bright_mask: Optional[np.ndarray] = None,
    too_dark_mask: Optional[np.ndarray] = None,
    alpha: float = 0.5,
) -> np.ndarray:
    """將瑕疵遮罩疊加至原始影像上。

    紅色表示過亮瑕疵，藍色表示過暗瑕疵。
    若未分別提供 bright/dark 遮罩，則全部標為紅色。

    Args:
        original: 原始影像（灰階或 BGR）。
        mask: 合併的瑕疵遮罩。
        too_bright_mask: 過亮瑕疵遮罩。
        too_dark_mask: 過暗瑕疵遮罩。
        alpha: 疊加透明度 (0.0 ~ 1.0)。

    Returns:
        BGR 疊加影像（uint8）。
    """
    # 確保原始影像為 BGR
    if original.ndim == 2:
        base = cv2.cvtColor(
            _to_uint8(original), cv2.COLOR_GRAY2BGR
        )
    elif original.dtype != np.uint8:
        base = _to_uint8(original)
    else:
        base = original.copy()

    overlay = base.copy()

    if too_bright_mask is not None and too_dark_mask is not None:
        # 紅色：過亮 (B=0, G=0, R=255)
        overlay[too_bright_mask > 0] = [0, 0, 255]
        # 藍色：過暗 (B=255, G=0, R=0)
        overlay[too_dark_mask > 0] = [255, 0, 0]
    else:
        # 預設紅色
        overlay[mask > 0] = [0, 0, 255]

    result = cv2.addWeighted(base, 1.0 - alpha, overlay, alpha, 0)
    return result


def create_threshold_visualization(model: VariationModel) -> np.ndarray:
    """生成模型閾值 2x2 網格視覺化（均值 / 標準差 / 上界 / 下界）。

    Args:
        model: 已訓練的 VariationModel。

    Returns:
        BGR 組合影像（uint8）。
    """
    imgs = model.get_model_images()

    panels = []
    titles = ["Mean", "Std Dev", "Upper Threshold", "Lower Threshold"]
    keys = ["mean", "std", "upper", "lower"]

    for key, title in zip(keys, titles):
        img = imgs.get(key)
        if img is not None:
            panel = _to_uint8(img)
        else:
            # 建立黑色佔位影像
            if imgs["mean"] is not None:
                h, w = imgs["mean"].shape[:2]
            else:
                h, w = 480, 640
            panel = np.zeros((h, w), dtype=np.uint8)

        # 轉為 BGR 以便加入文字
        if panel.ndim == 2:
            panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)

        # 加入標題文字
        cv2.putText(
            panel, title, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
        )
        panels.append(panel)

    # 2x2 網格
    top = np.hstack([panels[0], panels[1]])
    bottom = np.hstack([panels[2], panels[3]])
    grid = np.vstack([top, bottom])
    return grid


def create_composite_result(
    original: np.ndarray,
    overlay: np.ndarray,
    heatmap: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """生成 2x2 結果組合圖。

    排列：
        原始影像 | 瑕疵疊加
        差異熱力圖 | 二值遮罩

    Args:
        original: 原始/前處理影像。
        overlay: 瑕疵疊加影像。
        heatmap: 差異熱力圖。
        mask: 二值瑕疵遮罩。

    Returns:
        BGR 組合影像（uint8）。
    """
    # 統一轉為 BGR uint8
    orig_bgr = _ensure_bgr_uint8(original)
    overlay_bgr = _ensure_bgr_uint8(overlay)
    heatmap_bgr = _ensure_bgr_uint8(heatmap)
    mask_bgr = _ensure_bgr_uint8(mask)

    # 確保尺寸一致（以第一張為準）
    h, w = orig_bgr.shape[:2]
    overlay_bgr = cv2.resize(overlay_bgr, (w, h))
    heatmap_bgr = cv2.resize(heatmap_bgr, (w, h))
    mask_bgr = cv2.resize(mask_bgr, (w, h))

    # 加入面板標題
    _add_label(orig_bgr, "Original")
    _add_label(overlay_bgr, "Defect Overlay")
    _add_label(heatmap_bgr, "Difference Heatmap")
    _add_label(mask_bgr, "Defect Mask")

    top = np.hstack([orig_bgr, overlay_bgr])
    bottom = np.hstack([heatmap_bgr, mask_bgr])
    composite = np.vstack([top, bottom])
    return composite


# ====================================================================== #
#  內部工具函式                                                           #
# ====================================================================== #

def _to_uint8(image: np.ndarray) -> np.ndarray:
    """將任意影像正規化至 uint8 (0-255)。"""
    if image.dtype == np.uint8:
        return image.copy()
    min_val = image.min()
    max_val = image.max()
    if max_val - min_val > 0:
        normalized = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(image, dtype=np.uint8)
    return normalized


def _ensure_bgr_uint8(image: np.ndarray) -> np.ndarray:
    """確保影像為 BGR uint8。"""
    img = _to_uint8(image)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _add_label(image: np.ndarray, text: str) -> None:
    """在影像左上角加入標題標籤（原地修改）。"""
    cv2.putText(
        image, text, (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
    )
