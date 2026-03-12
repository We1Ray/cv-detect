"""Summary statistics and result-image persistence.

Provides helpers to aggregate a list of :class:`InspectionResult` objects
into a statistics dictionary and to save composite result images to disk.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Any

import cv2
import numpy as np

from dl_anomaly.pipeline.inference import InspectionResult
from dl_anomaly.visualization.heatmap import (
    create_composite_result,
    create_error_heatmap,
)

logger = logging.getLogger(__name__)


def generate_summary_stats(results: List[InspectionResult]) -> Dict[str, Any]:
    """Compute aggregate statistics from a batch of inspection results.

    Returns a dict with keys such as ``total``, ``defective``, ``pass``,
    ``defect_rate``, ``mean_score``, ``max_score``, ``min_score``, etc.
    """
    if not results:
        return {"total": 0}

    scores = [r.anomaly_score for r in results]
    defective = [r for r in results if r.is_defective]
    total_regions = sum(len(r.defect_regions) for r in defective)

    return {
        "total": len(results),
        "defective": len(defective),
        "pass": len(results) - len(defective),
        "defect_rate": len(defective) / len(results) if results else 0.0,
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "max_score": float(np.max(scores)),
        "min_score": float(np.min(scores)),
        "total_defect_regions": total_regions,
    }


def save_result_image(
    result: InspectionResult,
    original_path: str | Path,
    output_dir: str | Path,
) -> Path:
    """Save a 2x2 composite visualisation to *output_dir*.

    The output file is named after the original image with a ``_result``
    suffix.

    Returns the path of the saved file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    heatmap = create_error_heatmap(result.error_map)
    composite = create_composite_result(
        result.original,
        result.reconstruction,
        heatmap,
        result.defect_mask,
    )

    stem = Path(original_path).stem
    out_path = output_dir / f"{stem}_result.png"

    # Convert RGB -> BGR for cv2.imwrite
    cv2.imwrite(str(out_path), cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
    logger.info("Result saved: %s", out_path)
    return out_path
