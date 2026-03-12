"""
core/variation_model.py - 統計變異模型

基於 Welford 線上演算法，以數值穩定方式計算影像集合的逐像素均值與標準差，
進而建立瑕疵偵測的上下界閾值影像。
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class VariationModel:
    """統計變異模型：利用訓練影像的均值與標準差建立瑕疵偵測閾值。

    內部使用 Welford 線上演算法（float64）以確保數值穩定性。
    """

    def __init__(self) -> None:
        # Welford 線上統計量
        self._count: int = 0
        self._mean: Optional[np.ndarray] = None  # float64
        self._m2: Optional[np.ndarray] = None  # float64 (sum of squared diffs)

        # 閾值影像
        self._upper: Optional[np.ndarray] = None  # float64
        self._lower: Optional[np.ndarray] = None  # float64

        # 參考影像（用於對齊）
        self.reference_image: Optional[np.ndarray] = None

        # 訓練參數記錄
        self._abs_threshold: int = 0
        self._var_threshold: float = 0.0

    # ------------------------------------------------------------------ #
    #  屬性                                                               #
    # ------------------------------------------------------------------ #
    @property
    def is_trained(self) -> bool:
        """模型是否已訓練（至少兩張影像才能產生有效標準差）。"""
        return self._count >= 2 and self._mean is not None

    @property
    def count(self) -> int:
        """目前已訓練的影像數量。"""
        return self._count

    @property
    def mean_image(self) -> Optional[np.ndarray]:
        """回傳均值影像（float64）。"""
        return self._mean

    @property
    def std_image(self) -> Optional[np.ndarray]:
        """回傳標準差影像（float64）。"""
        if self._count < 2 or self._m2 is None:
            return None
        return np.sqrt(self._m2 / self._count)

    # ------------------------------------------------------------------ #
    #  訓練                                                               #
    # ------------------------------------------------------------------ #
    def train_incremental(self, image: np.ndarray) -> None:
        """以 Welford 線上演算法遞增更新統計量。

        Args:
            image: 前處理後的單張影像（灰階或彩色皆可）。
        """
        img = image.astype(np.float64)

        if self._mean is None:
            self._mean = np.zeros_like(img, dtype=np.float64)
            self._m2 = np.zeros_like(img, dtype=np.float64)

        self._count += 1
        delta = img - self._mean
        self._mean += delta / self._count
        delta2 = img - self._mean
        self._m2 += delta * delta2

        if self._count % 10 == 0:
            logger.info("已訓練 %d 張影像", self._count)

    def train_batch(self, images: List[np.ndarray]) -> None:
        """批次訓練：依序呼叫 train_incremental。

        Args:
            images: 前處理後的影像列表。
        """
        for img in images:
            self.train_incremental(img)
        logger.info("批次訓練完成，共 %d 張影像", self._count)

    # ------------------------------------------------------------------ #
    #  準備閾值                                                           #
    # ------------------------------------------------------------------ #
    def prepare(self, abs_threshold: int = 10, var_threshold: float = 3.0) -> None:
        """根據均值與標準差計算上下界閾值影像。

        閾值公式：
            upper = mean + max(abs_threshold, var_threshold * std)
            lower = mean - max(abs_threshold, var_threshold * std)

        Args:
            abs_threshold: 絕對閾值（像素值差異下限）。
            var_threshold: 標準差倍數閾值。

        Raises:
            RuntimeError: 模型尚未訓練。
        """
        if not self.is_trained:
            raise RuntimeError("模型尚未訓練，至少需要 2 張影像")

        self._abs_threshold = abs_threshold
        self._var_threshold = var_threshold

        std = self.std_image
        # 取絕對閾值與統計閾值的較大值，以確保最低偵測敏感度
        threshold_map = np.maximum(
            float(abs_threshold),
            var_threshold * std,
        )
        self._upper = self._mean + threshold_map
        self._lower = self._mean - threshold_map

        logger.info(
            "閾值準備完成: abs=%d, var=%.2f, 影像數=%d",
            abs_threshold,
            var_threshold,
            self._count,
        )

    # ------------------------------------------------------------------ #
    #  模型影像輸出                                                       #
    # ------------------------------------------------------------------ #
    def get_model_images(self) -> Dict[str, Optional[np.ndarray]]:
        """回傳模型的關鍵影像字典。

        Returns:
            包含 mean, std, upper, lower 的字典。
        """
        return {
            "mean": self._mean,
            "std": self.std_image,
            "upper": self._upper,
            "lower": self._lower,
        }

    # ------------------------------------------------------------------ #
    #  持久化                                                             #
    # ------------------------------------------------------------------ #
    def save(self, path: Union[str, Path]) -> None:
        """將模型儲存至 .npz 檔案。

        內容包含：mean, m2, count, upper, lower, reference_image,
        以及 abs_threshold, var_threshold, 訓練時間等 metadata。

        Args:
            path: 儲存路徑（副檔名建議 .npz）。

        Raises:
            RuntimeError: 模型尚未訓練。
        """
        if not self.is_trained:
            raise RuntimeError("模型尚未訓練，無法儲存")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data: Dict[str, Any] = {
            "mean": self._mean,
            "m2": self._m2,
            "count": np.array([self._count]),
            "abs_threshold": np.array([self._abs_threshold]),
            "var_threshold": np.array([self._var_threshold]),
            "timestamp": np.array([datetime.now().isoformat()]),
        }

        if self._upper is not None:
            data["upper"] = self._upper
        if self._lower is not None:
            data["lower"] = self._lower
        if self.reference_image is not None:
            data["reference_image"] = self.reference_image

        np.savez_compressed(str(path), **data)
        logger.info("模型已儲存至: %s", path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "VariationModel":
        """從 .npz 檔案載入模型。

        Args:
            path: .npz 檔案路徑。

        Returns:
            還原後的 VariationModel 實例。

        Raises:
            FileNotFoundError: 檔案不存在。
            KeyError: 檔案格式不符。
        """
        path = Path(path)
        if not path.exists():
            # 嘗試補上 .npz 副檔名
            if path.with_suffix(".npz").exists():
                path = path.with_suffix(".npz")
            else:
                raise FileNotFoundError(f"模型檔案不存在: {path}")

        data = np.load(str(path), allow_pickle=True)

        model = cls()
        model._mean = data["mean"]
        model._m2 = data["m2"]
        model._count = int(data["count"][0])

        if "upper" in data:
            model._upper = data["upper"]
        if "lower" in data:
            model._lower = data["lower"]
        if "reference_image" in data:
            model.reference_image = data["reference_image"]
        if "abs_threshold" in data:
            model._abs_threshold = int(data["abs_threshold"][0])
        if "var_threshold" in data:
            model._var_threshold = float(data["var_threshold"][0])

        logger.info(
            "模型已載入: %s (影像數=%d)",
            path,
            model._count,
        )
        return model
