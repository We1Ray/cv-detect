"""
core/inspector.py - 瑕疵檢測模組

將測試影像與已訓練的 VariationModel 比較，產生檢測結果。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from config import Config
from core.postprocessor import Postprocessor
from core.variation_model import VariationModel

logger = logging.getLogger(__name__)


@dataclass
class InspectionResult:
    """單張影像的瑕疵檢測結果。"""

    # 遮罩影像（uint8, 0/255）
    defect_mask: np.ndarray
    too_bright_mask: np.ndarray
    too_dark_mask: np.ndarray

    # 差異影像（float64，含正負號或絕對值）
    difference_image: np.ndarray

    # 統計摘要
    num_defects: int = 0
    defect_regions: List[Dict[str, Any]] = field(default_factory=list)
    is_defective: bool = False
    score: float = 0.0


class Inspector:
    """瑕疵檢測器：將測試影像與 VariationModel 比較並產生結果。"""

    def __init__(self, model: VariationModel, config: Config) -> None:
        """
        Args:
            model: 已訓練且已 prepare 的 VariationModel。
            config: 專案組態。

        Raises:
            RuntimeError: 模型未訓練或未準備閾值。
        """
        if not model.is_trained:
            raise RuntimeError("Inspector 需要已訓練的模型")

        imgs = model.get_model_images()
        if imgs["upper"] is None or imgs["lower"] is None:
            raise RuntimeError("模型尚未呼叫 prepare() 產生閾值影像")

        self.model = model
        self.config = config
        self.postprocessor = Postprocessor(config)

        self._mean = imgs["mean"]
        self._upper = imgs["upper"]
        self._lower = imgs["lower"]

    # ------------------------------------------------------------------ #
    #  單尺度比較                                                          #
    # ------------------------------------------------------------------ #
    def compare(self, image: np.ndarray) -> InspectionResult:
        """將測試影像與模型閾值比較，產生瑕疵檢測結果。

        Args:
            image: 前處理後的測試影像（與模型同尺寸、同型態）。

        Returns:
            InspectionResult 檢測結果。
        """
        img = image.astype(np.float64)

        # 計算差異
        difference = img - self._mean

        # 過亮（超過上界）與過暗（低於下界）
        bright_raw = (img > self._upper).astype(np.uint8) * 255
        dark_raw = (img < self._lower).astype(np.uint8) * 255

        # 若為多通道，合併各通道結果
        if bright_raw.ndim == 3:
            bright_raw = np.max(bright_raw, axis=2).astype(np.uint8)
            dark_raw = np.max(dark_raw, axis=2).astype(np.uint8)

        # 後處理
        too_bright = self.postprocessor.cleanup(bright_raw)
        too_dark = self.postprocessor.cleanup(dark_raw)

        # 合併遮罩
        defect_mask = cv2.bitwise_or(too_bright, too_dark)

        # 擷取區域
        bright_regions = self.postprocessor.extract_regions(too_bright, "too_bright")
        dark_regions = self.postprocessor.extract_regions(too_dark, "too_dark")
        all_regions = bright_regions + dark_regions

        num_defects = len(all_regions)
        is_defective = num_defects > 0

        # 計算異常分數：瑕疵面積佔比 (0.0 ~ 100.0)
        total_pixels = defect_mask.shape[0] * defect_mask.shape[1]
        defect_pixels = int(np.count_nonzero(defect_mask))
        score = (defect_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0

        logger.info(
            "檢測完成: defects=%d, score=%.4f%%, defective=%s",
            num_defects,
            score,
            is_defective,
        )

        return InspectionResult(
            defect_mask=defect_mask,
            too_bright_mask=too_bright,
            too_dark_mask=too_dark,
            difference_image=difference,
            num_defects=num_defects,
            defect_regions=all_regions,
            is_defective=is_defective,
            score=score,
        )

    # ------------------------------------------------------------------ #
    #  多尺度比較                                                          #
    # ------------------------------------------------------------------ #
    def compare_multiscale(
        self, image: np.ndarray, levels: int = 3
    ) -> InspectionResult:
        """使用高斯金字塔進行多尺度瑕疵偵測。

        在不同解析度下分別比較，再將結果合併回原始尺寸，
        可同時捕捉大範圍與小範圍的瑕疵。

        Args:
            image: 前處理後的測試影像。
            levels: 金字塔層數（含原始尺寸）。

        Returns:
            合併後的 InspectionResult。
        """
        h, w = image.shape[:2]
        combined_bright = np.zeros((h, w), dtype=np.uint8)
        combined_dark = np.zeros((h, w), dtype=np.uint8)

        # 建立測試影像與模型影像的金字塔
        pyr_img = [image.astype(np.float64)]
        pyr_mean = [self._mean.copy()]
        pyr_upper = [self._upper.copy()]
        pyr_lower = [self._lower.copy()]

        for _ in range(levels - 1):
            pyr_img.append(cv2.pyrDown(pyr_img[-1]))
            pyr_mean.append(cv2.pyrDown(pyr_mean[-1]))
            pyr_upper.append(cv2.pyrDown(pyr_upper[-1]))
            pyr_lower.append(cv2.pyrDown(pyr_lower[-1]))

        for lvl in range(levels):
            cur_img = pyr_img[lvl]
            cur_upper = pyr_upper[lvl]
            cur_lower = pyr_lower[lvl]

            bright_raw = (cur_img > cur_upper).astype(np.uint8) * 255
            dark_raw = (cur_img < cur_lower).astype(np.uint8) * 255

            if bright_raw.ndim == 3:
                bright_raw = np.max(bright_raw, axis=2).astype(np.uint8)
                dark_raw = np.max(dark_raw, axis=2).astype(np.uint8)

            # 放大回原始尺寸
            if lvl > 0:
                bright_raw = cv2.resize(bright_raw, (w, h), interpolation=cv2.INTER_NEAREST)
                dark_raw = cv2.resize(dark_raw, (w, h), interpolation=cv2.INTER_NEAREST)

            combined_bright = cv2.bitwise_or(combined_bright, bright_raw)
            combined_dark = cv2.bitwise_or(combined_dark, dark_raw)

        # 後處理
        too_bright = self.postprocessor.cleanup(combined_bright)
        too_dark = self.postprocessor.cleanup(combined_dark)
        defect_mask = cv2.bitwise_or(too_bright, too_dark)

        # 差異影像（使用原始尺度）
        difference = image.astype(np.float64) - self._mean

        # 擷取區域
        bright_regions = self.postprocessor.extract_regions(too_bright, "too_bright")
        dark_regions = self.postprocessor.extract_regions(too_dark, "too_dark")
        all_regions = bright_regions + dark_regions

        num_defects = len(all_regions)
        is_defective = num_defects > 0

        total_pixels = defect_mask.shape[0] * defect_mask.shape[1]
        defect_pixels = int(np.count_nonzero(defect_mask))
        score = (defect_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0

        logger.info(
            "多尺度檢測完成 (levels=%d): defects=%d, score=%.4f%%",
            levels,
            num_defects,
            score,
        )

        return InspectionResult(
            defect_mask=defect_mask,
            too_bright_mask=too_bright,
            too_dark_mask=too_dark,
            difference_image=difference,
            num_defects=num_defects,
            defect_regions=all_regions,
            is_defective=is_defective,
            score=score,
        )
