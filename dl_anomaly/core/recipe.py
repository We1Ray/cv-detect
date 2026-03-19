"""Recipe system for saving and replaying processing pipelines.

A *Recipe* is a JSON-serialisable list of processing steps (with operation
metadata) that can be applied to any image to reproduce the same pipeline.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Recipe:
    """Ordered collection of processing steps that can be saved / loaded."""

    def __init__(self, steps: Optional[List[Dict[str, Any]]] = None) -> None:
        self.version = 1
        self.steps: List[Dict[str, Any]] = steps or []

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Write the recipe to a JSON file."""
        data = {"version": self.version, "steps": self.steps}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "Recipe":
        """Read a recipe from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        recipe = cls(steps=data.get("steps", []))
        recipe.version = data.get("version", 1)
        return recipe

    # ------------------------------------------------------------------
    # Build from live pipeline
    # ------------------------------------------------------------------

    @classmethod
    def from_pipeline(cls, panel) -> "Recipe":
        """Extract a Recipe from the current PipelinePanel state.

        Steps whose ``op_meta`` is ``None`` or whose category is
        ``"source"`` are skipped.
        """
        recipe = cls()
        for step in panel.get_all_steps():
            meta = step.op_meta
            if meta is None:
                continue
            if meta.get("category") == "source":
                continue
            entry = {
                "name": step.name,
                "category": meta.get("category", ""),
                "op": meta.get("op", ""),
                "params": meta.get("params", {}),
            }
            recipe.steps.append(entry)
        return recipe


# ======================================================================
# Replay engine
# ======================================================================

def replay_recipe(
    recipe: Recipe,
    image: np.ndarray,
) -> List[Tuple[str, np.ndarray, Any]]:
    """Execute every step in *recipe* against *image*.

    Returns a list of ``(name, result_array, region_or_None)`` tuples,
    one per step.
    """
    results: List[Tuple[str, np.ndarray, Any]] = []
    current_image = image.copy()
    current_region = None

    for step in recipe.steps:
        cat = step.get("category", "")
        op = step.get("op", "")
        params = step.get("params", {})
        name = step.get("name", op)

        try:
            if cat in ("halcon", "vision"):
                current_image = _replay_vision(op, params, current_image)
                results.append((name, current_image, None))

            elif cat == "quick_op":
                current_image = _replay_quick_op(op, params, current_image)
                results.append((name, current_image, None))

            elif cat == "dialog_op":
                current_image = _replay_dialog_op(op, params, current_image)
                results.append((name, current_image, None))

            elif cat == "threshold":
                region = _replay_threshold(op, params, current_image)
                current_region = region
                display = region.to_binary_mask()
                results.append((name, display, region))

            elif cat == "region":
                region = _replay_region(op, params, current_image, current_region)
                current_region = region
                if op == "connection":
                    n = max(region.num_regions, 1)
                    vis = np.zeros(region.labels.shape, dtype=np.uint8)
                    for i in range(1, n + 1):
                        gray_val = int(55 + 200 * i / n)
                        vis[region.labels == i] = gray_val
                    display = vis
                else:
                    display = region.to_binary_mask()
                results.append((name, display, region))

            else:
                logger.warning("Unknown recipe category %r, skipping", cat)

        except Exception:
            logger.exception("Failed to replay step %r", name)

    return results


# ------------------------------------------------------------------
# Category handlers
# ------------------------------------------------------------------

def _replay_vision(op: str, params: dict, img: np.ndarray) -> np.ndarray:
    from dl_anomaly.core import vision_ops as hops

    _dispatch = {
        "rgb_to_gray": lambda: hops.rgb_to_gray(img),
        "rgb_to_hsv": lambda: hops.rgb_to_hsv(img),
        "rgb_to_hls": lambda: hops.rgb_to_hls(img),
        "histogram_eq_halcon": lambda: hops.histogram_eq(img),  # legacy name
        "histogram_eq": lambda: hops.histogram_eq(img),
        "invert_image": lambda: hops.invert_image(img),
        "illuminate": lambda: hops.illuminate(img, 41, 1.0),
        "abs_image": lambda: hops.abs_image(img),
        "laplace_filter": lambda: hops.laplace_filter(img),
        "sobel_filter": lambda: hops.sobel_filter(img, "both"),
        "prewitt_filter": lambda: hops.prewitt_filter(img),
        "zero_crossing": lambda: hops.zero_crossing(img),
        "rotate_90": lambda: hops.rotate_image(img, 90, "constant"),
        "rotate_180": lambda: hops.rotate_image(img, 180, "constant"),
        "rotate_270": lambda: hops.rotate_image(img, 270, "constant"),
        "mirror_h": lambda: hops.mirror_image(img, "horizontal"),
        "mirror_v": lambda: hops.mirror_image(img, "vertical"),
        "zoom_50": lambda: hops.zoom_image(img, 0.5, 0.5),
        "zoom_200": lambda: hops.zoom_image(img, 2.0, 2.0),
        "entropy_image": lambda: hops.entropy_image(img, 5),
        "deviation_image": lambda: hops.deviation_image(img, 5),
        "local_min": lambda: hops.local_min(img, 5),
        "local_max": lambda: hops.local_max(img, 5),
        "mean_image": lambda: hops.mean_image(img, params.get("ksize", 5)),
        "median_image": lambda: hops.median_image(img, params.get("ksize", 5)),
        "gauss_filter": lambda: hops.gauss_filter(img, params.get("sigma", 1.5)),
        "gauss_blur": lambda: hops.gauss_blur(img, params.get("ksize", 5)),
        "bilateral_filter": lambda: hops.bilateral_filter(
            img, params.get("d", 9),
            params.get("sigma_color", 75), params.get("sigma_space", 75)),
        "sharpen_image": lambda: hops.sharpen_image(img, params.get("amount", 0.5)),
        "emphasize": lambda: hops.emphasize(
            img, params.get("ksize", 7), params.get("factor", 1.5)),
        "scale_image": lambda: hops.scale_image(
            img, params.get("mult", 1.0), params.get("add", 0)),
        "bright_up": lambda: hops.scale_image(img, 1.0, 30),
        "bright_down": lambda: hops.scale_image(img, 1.0, -30),
        "contrast_up": lambda: hops.scale_image(img, 1.3, 0),
    }

    func = _dispatch.get(op)
    if func is not None:
        return func()

    raise ValueError(f"Unknown vision op: {op}")


def _replay_quick_op(op: str, params: dict, img: np.ndarray) -> np.ndarray:
    if op == "grayscale":
        if img.ndim == 3:
            return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img
    elif op == "blur":
        from scipy.ndimage import gaussian_filter
        sigma = params.get("sigma", 4.0)
        if img.ndim == 2:
            blurred = gaussian_filter(img.astype(np.float64), sigma=sigma)
        else:
            blurred = np.stack(
                [gaussian_filter(img[:, :, c].astype(np.float64), sigma=sigma)
                 for c in range(img.shape[2])],
                axis=2,
            )
        return np.clip(blurred, 0, 255).astype(np.uint8)
    elif op == "edge":
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
        return cv2.Canny(gray, 50, 150)
    elif op == "histeq":
        if img.ndim == 2:
            return cv2.equalizeHist(img)
        ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    elif op == "invert":
        return 255 - img
    raise ValueError(f"Unknown quick_op: {op}")


def _replay_dialog_op(op: str, params: dict, img: np.ndarray) -> np.ndarray:
    from dl_anomaly.core import vision_ops as hops

    _map = {
        "均值濾波": lambda: hops.mean_image(img, params.get("ksize", 5)),
        "中值濾波": lambda: hops.median_image(img, params.get("ksize", 5)),
        "高斯模糊": lambda: (
            hops.gauss_blur(img, params.get("ksize", 5)) if params.get("sigma", 0) == 0
            else hops.gauss_filter(img, params["sigma"])
        ),
        "雙邊濾波": lambda: hops.bilateral_filter(
            img, params.get("d", 9),
            params.get("sigma_color", 75), params.get("sigma_space", 75)),
        "銳化": lambda: hops.sharpen_image(img, params.get("amount", 0.5)),
        "強調": lambda: hops.emphasize(
            img, params.get("ksize", 7), params.get("factor", 1.5)),
        "Canny 邊緣": lambda: hops.edges_canny(
            img, params.get("low", 50), params.get("high", 150),
            params.get("sigma", 1.0)),
        "灰度侵蝕": lambda: hops.gray_erosion(img, params.get("ksize", 5)),
        "灰度膨脹": lambda: hops.gray_dilation(img, params.get("ksize", 5)),
        "灰度開運算": lambda: hops.gray_opening(img, params.get("ksize", 5)),
        "灰度閉運算": lambda: hops.gray_closing(img, params.get("ksize", 5)),
        "Top-hat": lambda: hops.top_hat(img, params.get("ksize", 9)),
        "Bottom-hat": lambda: hops.bottom_hat(img, params.get("ksize", 9)),
        "亮度/對比度調整": lambda: hops.scale_image(
            img, params.get("mult", 1.0), params.get("add", 0)),
        "對數變換": lambda: hops.log_image(img, params.get("base", "e")),
        "指數變換": lambda: hops.exp_image(img, params.get("base", "e")),
        "Gamma 校正": lambda: hops.gamma_image(img, params.get("gamma", 1.0)),
        "FFT 頻譜": lambda: hops.fft_image(img),
        "低通濾波": lambda: hops.freq_filter(img, "lowpass", params.get("cutoff", 30)),
        "高通濾波": lambda: hops.freq_filter(img, "highpass", params.get("cutoff", 30)),
        "高斯導數": lambda: hops.derivative_gauss(
            img, params.get("sigma", 1.0), params.get("component", "x")),
    }

    func = _map.get(op)
    if func is not None:
        return func()
    raise ValueError(f"Unknown dialog_op: {op}")


def _replay_threshold(op: str, params: dict, img: np.ndarray):
    from dl_anomaly.core.region_ops import binary_threshold

    if op == "otsu":
        return binary_threshold(img, method="otsu")
    elif op == "adaptive":
        return binary_threshold(img, method="adaptive")
    elif op == "manual":
        min_val = params.get("min_val", 0)
        max_val = params.get("max_val", 255)
        return binary_threshold(img, method="manual", min_val=min_val, max_val=max_val)
    raise ValueError(f"Unknown threshold op: {op}")


def _replay_region(op: str, params: dict, img: np.ndarray, current_region):
    if current_region is None:
        from dl_anomaly.core.region_ops import binary_threshold
        current_region = binary_threshold(img, method="otsu")

    if op == "connection":
        from dl_anomaly.core.region_ops import connection
        return connection(current_region)

    elif op == "fill_up":
        from dl_anomaly.core.region import Region
        from dl_anomaly.core.region_ops import compute_region_properties
        min_area = params.get("min_area", 0)
        labels_in = current_region.labels
        filled_labels = np.zeros_like(labels_in)
        new_id = 1
        for lbl in range(1, current_region.num_regions + 1):
            comp = ((labels_in == lbl) * 255).astype(np.uint8)
            if not np.any(comp):
                continue
            h, w = comp.shape
            border = np.full((h + 2, w + 2), 255, dtype=np.uint8)
            border[1:-1, 1:-1] = cv2.bitwise_not(comp)
            cv2.floodFill(border, None, (0, 0), 0)
            holes = border[1:-1, 1:-1]
            if min_area > 0:
                hole_n, hole_labels = cv2.connectedComponents(holes, connectivity=8)
                for hi in range(1, hole_n):
                    hole_mask = (hole_labels == hi)
                    if hole_mask.sum() >= min_area:
                        comp[hole_mask] = 255
            else:
                comp = comp | holes
            filled_labels[comp > 0] = new_id
            new_id += 1
        num, labels_out = cv2.connectedComponents(
            (filled_labels > 0).astype(np.uint8) * 255, connectivity=8)
        labels_out = labels_out.astype(np.int32)
        props = compute_region_properties(labels_out, current_region.source_image)
        return Region(labels=labels_out, num_regions=num - 1, properties=props,
                      source_image=current_region.source_image,
                      source_shape=current_region.source_shape)

    elif op == "select_shape":
        from dl_anomaly.core.region_ops import select_shape
        conditions = params.get("conditions", [])
        result_region = current_region
        for cond in conditions:
            feat = cond["feature"]
            mn = cond["min_val"]
            mx = cond["max_val"]
            result_region = select_shape(result_region, feat, mn, mx)
        return result_region

    raise ValueError(f"Unknown region op: {op}")
