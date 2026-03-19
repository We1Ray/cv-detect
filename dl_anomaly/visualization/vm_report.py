"""
visualization/vm_report.py - 報告生成工具

產生批次檢測結果的統計摘要與影像輸出。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np

from dl_anomaly.core.vm_inspector import InspectionResult
from dl_anomaly.visualization.vm_heatmap import (
    create_composite_result,
    create_defect_overlay,
    create_difference_heatmap,
)

logger = logging.getLogger(__name__)


def generate_summary_stats(
    results: List[Tuple[InspectionResult, Path, np.ndarray]],
) -> Dict[str, Any]:
    """根據批次檢測結果計算統計摘要。

    Args:
        results: (InspectionResult, 影像路徑, 前處理影像) 元組清單。

    Returns:
        統計摘要字典，包含：
        - total: 總數
        - pass_count: 合格數
        - fail_count: 不合格數
        - pass_rate: 合格率 (%)
        - avg_score: 平均異常分數
        - max_score: 最大異常分數
        - min_score: 最小異常分數
        - total_defects: 瑕疵總數
        - avg_defects_per_image: 每張影像平均瑕疵數
        - bright_defect_count: 過亮瑕疵總數
        - dark_defect_count: 過暗瑕疵總數
    """
    total = len(results)
    if total == 0:
        return {
            "total": 0,
            "pass_count": 0,
            "fail_count": 0,
            "pass_rate": 0.0,
            "avg_score": 0.0,
            "max_score": 0.0,
            "min_score": 0.0,
            "total_defects": 0,
            "avg_defects_per_image": 0.0,
            "bright_defect_count": 0,
            "dark_defect_count": 0,
        }

    pass_count = sum(1 for r, _, _ in results if not r.is_defective)
    fail_count = total - pass_count
    scores = [r.score for r, _, _ in results]
    total_defects = sum(r.num_defects for r, _, _ in results)

    bright_count = sum(
        sum(1 for reg in r.defect_regions if reg["type"] == "too_bright")
        for r, _, _ in results
    )
    dark_count = sum(
        sum(1 for reg in r.defect_regions if reg["type"] == "too_dark")
        for r, _, _ in results
    )

    return {
        "total": total,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "pass_rate": (pass_count / total) * 100.0,
        "avg_score": float(np.mean(scores)),
        "max_score": float(np.max(scores)),
        "min_score": float(np.min(scores)),
        "total_defects": total_defects,
        "avg_defects_per_image": total_defects / total,
        "bright_defect_count": bright_count,
        "dark_defect_count": dark_count,
    }


def save_result_image(
    result: InspectionResult,
    processed_image: np.ndarray,
    original_path: Union[str, Path],
    output_dir: Union[str, Path],
) -> Path:
    """將檢測結果儲存為組合影像檔案。

    Args:
        result: 檢測結果。
        processed_image: 前處理後的影像。
        original_path: 原始影像檔案路徑（用於取得檔名）。
        output_dir: 輸出目錄。

    Returns:
        儲存的檔案路徑。
    """
    original_path = Path(original_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 生成子影像
    overlay = create_defect_overlay(
        processed_image,
        result.defect_mask,
        result.too_bright_mask,
        result.too_dark_mask,
        alpha=0.5,
    )
    heatmap = create_difference_heatmap(result.difference_image)
    composite = create_composite_result(
        processed_image, overlay, heatmap, result.defect_mask
    )

    # 決定輸出檔名
    status = "DEFECTIVE" if result.is_defective else "PASS"
    stem = original_path.stem
    out_name = f"{stem}_{status}.png"
    out_path = output_dir / out_name

    success = cv2.imwrite(str(out_path), composite)
    if not success:
        logger.warning("cv2.imwrite 寫入失敗: %s", out_path)
    else:
        logger.info("結果影像已儲存: %s", out_path)
    return out_path
