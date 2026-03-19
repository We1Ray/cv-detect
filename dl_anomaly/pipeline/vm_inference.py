"""
pipeline/vm_inference.py - 推論管線

負責單張或批次瑕疵檢測，以及結果報告生成。
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from dl_anomaly.core.vm_config import VMConfig as Config
from dl_anomaly.core.vm_inspector import InspectionResult, Inspector
from dl_anomaly.core.vm_preprocessor import ImagePreprocessor
from dl_anomaly.core.variation_model import VariationModel
from dl_anomaly.visualization.vm_heatmap import create_composite_result, create_defect_overlay, create_difference_heatmap
from dl_anomaly.visualization.vm_report import save_result_image

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


class InferencePipeline:
    """推論管線：執行瑕疵檢測並輸出結果報告。"""

    def __init__(self, model: VariationModel, config: Config) -> None:
        """
        Args:
            model: 已訓練並已 prepare 的 VariationModel。
            config: 專案組態。
        """
        self.model = model
        self.config = config
        self.preprocessor = ImagePreprocessor(config)
        self.inspector = Inspector(model, config)

    # ------------------------------------------------------------------ #
    #  單張檢測                                                           #
    # ------------------------------------------------------------------ #
    def inspect_single(
        self, image_path: Union[str, Path]
    ) -> Tuple[InspectionResult, np.ndarray]:
        """檢測單張影像。

        Args:
            image_path: 測試影像路徑。

        Returns:
            (InspectionResult, 前處理後的影像) 元組。
        """
        image_path = Path(image_path)
        reference = self.model.reference_image
        processed = self.preprocessor.preprocess(image_path, reference=reference)

        if self.config.enable_multiscale:
            result = self.inspector.compare_multiscale(
                processed, levels=self.config.scale_levels
            )
        else:
            result = self.inspector.compare(processed)

        logger.info("單張檢測完成: %s -> %s", image_path.name,
                     "DEFECTIVE" if result.is_defective else "PASS")
        return result, processed

    # ------------------------------------------------------------------ #
    #  批次檢測                                                           #
    # ------------------------------------------------------------------ #
    def inspect_batch(
        self,
        image_dir: Union[str, Path],
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[Tuple[InspectionResult, Path, np.ndarray]]:
        """批次檢測目錄中的所有影像。

        Args:
            image_dir: 測試影像目錄。
            progress_callback: 進度回呼 (current, total, message)。

        Returns:
            (InspectionResult, 影像路徑, 前處理影像) 元組清單。
        """
        image_dir = Path(image_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"測試影像目錄不存在: {image_dir}")

        # 搜尋影像
        image_paths: List[Path] = []
        for ext in IMAGE_EXTENSIONS:
            image_paths.extend(image_dir.glob(f"*{ext}"))
            image_paths.extend(image_dir.glob(f"*{ext.upper()}"))
        image_paths = sorted(set(image_paths))

        if not image_paths:
            raise ValueError(f"目錄中未找到影像: {image_dir}")

        total = len(image_paths)
        results: List[Tuple[InspectionResult, Path, np.ndarray]] = []

        for i, img_path in enumerate(image_paths):
            if progress_callback:
                progress_callback(i + 1, total, f"檢測中: {img_path.name}")
            try:
                result, processed = self.inspect_single(img_path)
                results.append((result, img_path, processed))
            except Exception as exc:
                logger.warning("檢測 %s 失敗: %s", img_path.name, exc)
                continue

        if progress_callback:
            progress_callback(total, total, f"批次檢測完成: {len(results)}/{total}")

        return results

    # ------------------------------------------------------------------ #
    #  結果報告                                                           #
    # ------------------------------------------------------------------ #
    def generate_report(
        self,
        results: List[Tuple[InspectionResult, Path, np.ndarray]],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Path:
        """生成檢測結果報告：儲存結果影像與 CSV 摘要。

        Args:
            results: inspect_batch 的回傳結果。
            output_dir: 輸出目錄；若為 None 則使用組態 results_dir。

        Returns:
            報告輸出目錄路徑。
        """
        out = Path(output_dir or self.config.results_dir)
        out.mkdir(parents=True, exist_ok=True)

        # 建立帶時間戳的子目錄
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = out / f"report_{timestamp}"
        report_dir.mkdir(parents=True, exist_ok=True)

        # CSV 摘要
        csv_path = report_dir / "summary.csv"
        csv_rows: List[Dict[str, str]] = []

        for result, img_path, processed in results:
            # 儲存結果影像
            try:
                save_result_image(result, processed, img_path, report_dir)
            except Exception as exc:
                logger.warning("儲存結果影像失敗 (%s): %s", img_path.name, exc)

            csv_rows.append(
                {
                    "filename": img_path.name,
                    "result": "DEFECTIVE" if result.is_defective else "PASS",
                    "score": f"{result.score:.4f}",
                    "num_defects": str(result.num_defects),
                    "bright_defects": str(
                        sum(1 for r in result.defect_regions if r["type"] == "too_bright")
                    ),
                    "dark_defects": str(
                        sum(1 for r in result.defect_regions if r["type"] == "too_dark")
                    ),
                }
            )

        # 寫入 CSV
        if csv_rows:
            fieldnames = list(csv_rows[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_rows)

        logger.info("報告已儲存至: %s", report_dir)
        return report_dir
