"""
pipeline/vm_trainer.py - 訓練管線

負責完整的模型訓練流程：探索影像 -> 選擇參考 -> 前處理 -> 訓練 -> 準備閾值 -> 儲存。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from dl_anomaly.core.vm_config import VMConfig as Config
from dl_anomaly.core.vm_preprocessor import ImagePreprocessor
from dl_anomaly.core.variation_model import VariationModel

logger = logging.getLogger(__name__)

# 支援的影像副檔名
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


class TrainingPipeline:
    """模型訓練管線：從影像目錄到完成訓練的一站式流程。"""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.preprocessor = ImagePreprocessor(config)
        self.model = VariationModel()

    # ------------------------------------------------------------------ #
    #  影像探索                                                           #
    # ------------------------------------------------------------------ #
    def discover_images(self, directory: Optional[Path] = None) -> List[Path]:
        """在指定目錄中搜尋所有支援的影像檔案。

        Args:
            directory: 搜尋目錄；若為 None 則使用組態 train_image_dir。

        Returns:
            依檔名排序的影像路徑清單。

        Raises:
            FileNotFoundError: 目錄不存在。
            ValueError: 未找到任何影像。
        """
        search_dir = directory or self.config.train_image_dir
        search_dir = Path(search_dir)

        if not search_dir.exists():
            raise FileNotFoundError(f"訓練影像目錄不存在: {search_dir}")

        images: List[Path] = []
        for ext in IMAGE_EXTENSIONS:
            images.extend(search_dir.glob(f"*{ext}"))
            images.extend(search_dir.glob(f"*{ext.upper()}"))

        # 去重並排序
        images = sorted(set(images))

        if not images:
            raise ValueError(f"目錄中未找到支援的影像檔案: {search_dir}")

        logger.info("探索到 %d 張訓練影像於: %s", len(images), search_dir)
        return images

    # ------------------------------------------------------------------ #
    #  參考影像選擇                                                       #
    # ------------------------------------------------------------------ #
    def select_reference(self, images: List[Path]) -> np.ndarray:
        """選擇並載入參考影像。

        若組態中指定了 REFERENCE_IMAGE，使用該路徑；
        否則使用清單中的第一張影像。

        Args:
            images: 影像路徑清單。

        Returns:
            前處理後的參考影像。
        """
        ref_path: Optional[Path] = self.config.reference_image

        if ref_path and ref_path.exists():
            logger.info("使用指定參考影像: %s", ref_path)
        else:
            ref_path = images[0]
            logger.info("使用第一張影像作為參考: %s", ref_path)

        # 載入、縮放、灰階、平滑（不對齊，因為這是參考本身）
        ref_image = self.preprocessor.load_image(ref_path)
        ref_image = self.preprocessor.resize(ref_image)
        ref_image = self.preprocessor.to_grayscale(ref_image)
        ref_image = self.preprocessor.smooth(ref_image)
        return ref_image

    # ------------------------------------------------------------------ #
    #  完整訓練流程                                                       #
    # ------------------------------------------------------------------ #
    def run(
        self,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        image_dir: Optional[Path] = None,
    ) -> VariationModel:
        """執行完整訓練管線。

        Args:
            progress_callback: 進度回呼函式，簽名 (current, total, message)。
            image_dir: 訓練影像目錄；若為 None 則使用組態值。

        Returns:
            已訓練且已準備閾值的 VariationModel。
        """

        def _report(current: int, total: int, msg: str) -> None:
            if progress_callback:
                progress_callback(current, total, msg)

        # 1. 探索影像
        _report(0, 0, "探索訓練影像...")
        images = self.discover_images(image_dir)
        total = len(images)

        # 2. 選擇參考影像
        _report(0, total, "選擇參考影像...")
        reference = self.select_reference(images)
        self.model.reference_image = reference

        # 3. 逐張前處理並訓練
        for i, img_path in enumerate(images):
            _report(i + 1, total, f"訓練中: {img_path.name} ({i + 1}/{total})")
            try:
                processed = self.preprocessor.preprocess(img_path, reference=reference)
                self.model.train_incremental(processed)
            except Exception as exc:
                logger.warning("處理影像 %s 時發生錯誤，已跳過: %s", img_path.name, exc)
                continue

        if not self.model.is_trained:
            raise RuntimeError("訓練失敗：有效影像不足（至少需要 2 張）")

        # 4. 準備閾值
        _report(total, total, "計算閾值影像...")
        self.model.prepare(
            abs_threshold=self.config.abs_threshold,
            var_threshold=self.config.var_threshold,
        )

        # 5. 儲存模型
        _report(total, total, "儲存模型...")
        model_dir = Path(self.config.model_save_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "variation_model.npz"
        self.model.save(model_path)

        _report(total, total, f"訓練完成！共 {self.model.count} 張影像")
        logger.info("訓練管線完成: %d 張影像, 模型儲存至 %s", self.model.count, model_path)

        return self.model
