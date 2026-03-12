"""
core/postprocessor.py - 二值遮罩後處理模組

負責形態學清理與連通區域分析。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import cv2
import numpy as np

from config import Config

logger = logging.getLogger(__name__)


class Postprocessor:
    """瑕疵遮罩後處理器：形態學清理與區域擷取。"""

    def __init__(self, config: Config) -> None:
        self.config = config

    # ------------------------------------------------------------------ #
    #  形態學清理                                                          #
    # ------------------------------------------------------------------ #
    def cleanup(self, mask: np.ndarray) -> np.ndarray:
        """對二值遮罩執行形態學開運算、閉運算，並過濾小面積區域。

        流程：
        1. 開運算（去除小雜訊）
        2. 閉運算（填補小孔洞）
        3. 面積過濾（移除面積 < min_defect_area 的區域）

        Args:
            mask: uint8 二值遮罩（0 或 255）。

        Returns:
            清理後的二值遮罩。
        """
        k = self.config.morph_kernel_size
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))

        # 開運算：消除小型雜訊
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        # 閉運算：填補小型孔洞
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

        # 面積過濾
        cleaned = self._filter_by_area(cleaned, self.config.min_defect_area)

        logger.debug(
            "形態學清理完成: kernel=%d, min_area=%d",
            k,
            self.config.min_defect_area,
        )
        return cleaned

    def _filter_by_area(self, mask: np.ndarray, min_area: int) -> np.ndarray:
        """移除面積小於指定值的連通區域。"""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        filtered = np.zeros_like(mask)
        for i in range(1, num_labels):  # 跳過背景 (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                filtered[labels == i] = 255
        return filtered

    # ------------------------------------------------------------------ #
    #  區域擷取                                                           #
    # ------------------------------------------------------------------ #
    def extract_regions(
        self, mask: np.ndarray, defect_type: str = "unknown"
    ) -> List[Dict[str, Any]]:
        """從二值遮罩擷取每個連通區域的幾何資訊。

        Args:
            mask: uint8 二值遮罩。
            defect_type: 瑕疵類型標籤（如 "too_bright", "too_dark"）。

        Returns:
            區域資訊清單，每筆包含：
            - bbox: (x, y, w, h) 邊界框
            - area: 像素面積
            - centroid: (cx, cy) 重心座標
            - type: 瑕疵類型
        """
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        regions: List[Dict[str, Any]] = []
        for i in range(1, num_labels):
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            area = int(stats[i, cv2.CC_STAT_AREA])
            cx = float(centroids[i, 0])
            cy = float(centroids[i, 1])

            regions.append(
                {
                    "bbox": (x, y, w, h),
                    "area": area,
                    "centroid": (cx, cy),
                    "type": defect_type,
                }
            )

        logger.debug("擷取到 %d 個 %s 區域", len(regions), defect_type)
        return regions
