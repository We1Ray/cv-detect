"""
core/inspection_flow.py - Chained inspection pipeline orchestrator.

Enables users to build multi-step inspection flows such as:

    Locate -> Detect -> Measure -> Classify -> Judge

Each step is independently configurable and results chain forward through a
shared context dictionary.  The flow engine handles timing, error recovery,
serialisation, and batch execution.

Typical usage::

    flow = InspectionFlow("PCB Inspection")
    flow.add_step(LocateStep("Locate IC", config={...}))
    flow.add_step(DetectStep("Defect Detection", config={...}))
    flow.add_step(MeasureStep("Dimension Check", config={...}))
    flow.add_step(JudgeStep("Pass/Fail", config={...}))

    result = flow.execute(image)
    print(result.overall_pass)
"""

from __future__ import annotations

import ast
import json
import logging
import math
import os
import re
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import cv2
import numpy as np

from shared.op_logger import log_operation

logger = logging.getLogger(__name__)

# Leaked thread tracking for custom step exec-with-timeout.
_LEAKED_THREAD_COUNT = 0
_MAX_LEAKED_THREADS = 5


# ====================================================================== #
#  Result Data Classes                                                     #
# ====================================================================== #


@dataclass
class StepResult:
    """Result of a single flow step.

    Attributes
    ----------
    step_name:
        Human-readable name for this step.
    step_type:
        Categorical type: ``"locate"``, ``"detect"``, ``"measure"``,
        ``"classify"``, ``"judge"``, or ``"custom"``.
    success:
        Whether the step completed without error.
    data:
        Step-specific output data (contents depend on step type).
    image:
        Optional visualisation image produced by the step.
    elapsed_ms:
        Wall-clock execution time in milliseconds.
    message:
        Human-readable status or error message.
    """

    step_name: str
    step_type: str
    success: bool
    data: Dict[str, Any]
    image: Optional[np.ndarray] = None
    elapsed_ms: float = 0.0
    message: str = ""


@dataclass
class FlowResult:
    """Complete result of an inspection flow execution.

    Attributes
    ----------
    flow_name:
        Name of the flow that was executed.
    overall_pass:
        Final pass/fail verdict.  ``True`` when all judge steps pass
        (or when no judge step is present and no step failed).
    steps:
        Ordered list of :class:`StepResult` for every executed step.
    total_time_ms:
        Total wall-clock time for the entire flow in milliseconds.
    timestamp:
        ISO-8601 timestamp of when the flow completed.
    source_image_path:
        Path to the input image (empty string for in-memory images).
    summary:
        Aggregated summary data collected from all steps.
    """

    flow_name: str
    overall_pass: bool
    steps: List[StepResult]
    total_time_ms: float
    timestamp: str
    source_image_path: str = ""
    summary: Dict[str, Any] = field(default_factory=dict)


# ====================================================================== #
#  Abstract Base Step                                                      #
# ====================================================================== #


class FlowStep(ABC):
    """Base class for all inspection flow steps.

    Parameters
    ----------
    name:
        Human-readable identifier for this step instance.
    step_type:
        Categorical type string.
    config:
        Step-specific configuration dictionary.
    """

    def __init__(
        self,
        name: str,
        step_type: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name = name
        self.step_type = step_type
        self.config: Dict[str, Any] = config or {}
        self.enabled: bool = True

    @abstractmethod
    def execute(
        self, image: np.ndarray, context: Dict[str, Any]
    ) -> StepResult:
        """Execute this step.

        Parameters
        ----------
        image:
            Input image (BGR uint8 or grayscale).
        context:
            Accumulated data from all previous steps, keyed by step type.

        Returns
        -------
        StepResult
        """
        ...

    # ------------------------------------------------------------------ #
    #  Serialisation                                                       #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        """Serialise this step to a JSON-compatible dictionary."""
        return {
            "name": self.name,
            "step_type": self.step_type,
            "config": self.config,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FlowStep":
        """Reconstruct a step from a serialised dictionary.

        Uses the global :data:`STEP_REGISTRY` to resolve the concrete class.
        """
        step_type = d["step_type"]
        step_cls = STEP_REGISTRY.get(step_type)
        if step_cls is None:
            raise ValueError(
                f"Unknown step_type '{step_type}'. "
                f"Registered types: {list(STEP_REGISTRY.keys())}"
            )
        instance = step_cls(name=d["name"], config=d.get("config", {}))
        instance.enabled = d.get("enabled", True)
        return instance

    def __repr__(self) -> str:
        state = "enabled" if self.enabled else "disabled"
        return f"<{self.__class__.__name__} name={self.name!r} [{state}]>"


# ====================================================================== #
#  Concrete Steps                                                          #
# ====================================================================== #


class LocateStep(FlowStep):
    """Shape matching / template locating step.

    Config keys
    -----------
    template_path : str
        Path to the template image file.
    min_score : float
        Minimum match score (default ``0.5``).
    num_matches : int
        Maximum number of matches to find (default ``1``).
    angle_start : float
        Start of the angle search range in degrees (default ``0``).
    angle_extent : float
        Extent of the angle search range in degrees (default ``360``).
    scale_min : float
        Minimum scale factor (default ``1.0``).
    scale_max : float
        Maximum scale factor (default ``1.0``).
    roi_padding : int
        Extra pixels around each match for downstream ROIs (default ``10``).
    min_contrast : int
        Minimum edge contrast for shape model creation (default ``30``).
    angle_step : float
        Angle search step in degrees (default ``1.0``).
    max_contour_points : int
        Maximum contour points for the shape model (default ``2000``).
    greediness : float
        Search greediness ``0.0``--``1.0`` (default ``0.9``).

    Output data
    -----------
    matches : List[dict]
        Each entry: ``{x, y, angle, score, scale}``.
    roi_list : List[tuple]
        ``(x, y, w, h)`` bounding boxes for downstream steps.
    num_matches : int
        Number of matches found.
    """

    def __init__(
        self, name: str = "Locate", config: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(name, "locate", config)

    def execute(
        self, image: np.ndarray, context: Dict[str, Any]
    ) -> StepResult:
        t0 = time.perf_counter()

        template_path = self.config.get("template_path", "")
        if not template_path:
            return StepResult(
                step_name=self.name,
                step_type=self.step_type,
                success=False,
                data={},
                elapsed_ms=_elapsed_ms(t0),
                message="template_path not configured.",
            )

        template_path_obj = Path(template_path)
        if not template_path_obj.exists():
            return StepResult(
                step_name=self.name,
                step_type=self.step_type,
                success=False,
                data={},
                elapsed_ms=_elapsed_ms(t0),
                message=f"Template file not found: {template_path}",
            )

        # Lazy import to avoid circular dependencies and heavy startup cost.
        from dl_anomaly.core.shape_matching import (
            create_shape_model,
            draw_shape_matches,
            find_shape_model,
        )

        min_score = float(self.config.get("min_score", 0.5))
        num_matches = int(self.config.get("num_matches", 1))
        angle_start_deg = float(self.config.get("angle_start", 0.0))
        angle_extent_deg = float(self.config.get("angle_extent", 360.0))
        scale_min = float(self.config.get("scale_min", 1.0))
        scale_max = float(self.config.get("scale_max", 1.0))
        roi_padding = int(self.config.get("roi_padding", 10))
        min_contrast = int(self.config.get("min_contrast", 30))
        angle_step_deg = float(self.config.get("angle_step", 1.0))
        max_contour_points = int(self.config.get("max_contour_points", 2000))
        greediness = float(self.config.get("greediness", 0.9))

        angle_start = math.radians(angle_start_deg)
        angle_extent = math.radians(angle_extent_deg)
        angle_step = math.radians(angle_step_deg)

        template = cv2.imread(str(template_path_obj), cv2.IMREAD_UNCHANGED)
        if template is None:
            return StepResult(
                step_name=self.name,
                step_type=self.step_type,
                success=False,
                data={},
                elapsed_ms=_elapsed_ms(t0),
                message=f"Failed to load template: {template_path}",
            )

        try:
            model = create_shape_model(
                template,
                num_levels=int(self.config.get("num_levels", 4)),
                angle_start=angle_start,
                angle_extent=angle_extent,
                scale_min=scale_min,
                scale_max=scale_max,
                min_contrast=min_contrast,
                angle_step=angle_step,
                max_contour_points=max_contour_points,
            )
            matches = find_shape_model(
                image,
                model,
                min_score=min_score,
                num_matches=num_matches,
                greediness=greediness,
            )
        except Exception as exc:
            logger.exception("LocateStep failed during shape matching.")
            return StepResult(
                step_name=self.name,
                step_type=self.step_type,
                success=False,
                data={},
                elapsed_ms=_elapsed_ms(t0),
                message=f"Shape matching error: {exc}",
            )

        h, w = image.shape[:2]
        bw = model.bounding_box[2]
        bh = model.bounding_box[3]

        match_dicts: List[Dict[str, Any]] = []
        roi_list: List[Tuple[int, int, int, int]] = []

        for m in matches:
            match_dicts.append({
                "x": float(m.col),
                "y": float(m.row),
                "angle": float(math.degrees(m.angle)),
                "score": float(m.score),
                "scale": float(m.scale),
            })
            # Compute bounding ROI centred on the match.
            half_w = int((bw * m.scale) / 2.0) + roi_padding
            half_h = int((bh * m.scale) / 2.0) + roi_padding
            rx = max(0, int(m.col) - half_w)
            ry = max(0, int(m.row) - half_h)
            rw = min(w - rx, half_w * 2)
            rh = min(h - ry, half_h * 2)
            roi_list.append((rx, ry, rw, rh))

        # Visualisation.
        vis = None
        try:
            vis = draw_shape_matches(image, matches, model)
        except Exception:
            logger.debug("LocateStep: visualisation failed.", exc_info=True)

        data: Dict[str, Any] = {
            "matches": match_dicts,
            "roi_list": roi_list,
            "num_matches": len(matches),
        }

        elapsed = _elapsed_ms(t0)
        msg = f"Found {len(matches)} match(es)."
        if matches:
            msg += f" Best score: {matches[0].score:.3f}"

        logger.info("LocateStep [%s]: %s (%.1f ms)", self.name, msg, elapsed)
        return StepResult(
            step_name=self.name,
            step_type=self.step_type,
            success=True,
            data=data,
            image=vis,
            elapsed_ms=elapsed,
            message=msg,
        )


class DetectStep(FlowStep):
    """Anomaly detection step supporting multiple detection methods.

    Config keys
    -----------
    method : str
        ``"autoencoder"``, ``"patchcore"``, ``"difference"``, ``"fft"``,
        ``"color"``, ``"blob"``, ``"teacher_student"``, ``"normalizing_flow"``,
        or ``"unet"`` (default ``"autoencoder"``).
    checkpoint_path : str
        Path to the trained model checkpoint (required for DL methods:
        autoencoder, patchcore, teacher_student, normalizing_flow, unet).
    threshold : float
        Anomaly score threshold (default ``0.5``).
    roi_from_previous : bool
        When ``True``, crop to ROIs produced by a preceding locate step
        (default ``False``).
    min_defect_area : int
        Minimum contiguous defect area in pixels (default ``20``).

    Method-specific config keys
    ---------------------------
    **difference** method:
        reference_image_path : str
            Path to the reference (golden) image.
        registration_method : str
            ``"ecc"``, ``"orb"``, ``"phase_correlation"``, or ``"none"``
            (default ``"ecc"``).
        blur_sigma : float
            Gaussian blur sigma before differencing (default ``2.0``).
        morph_kernel_size : int
            Morphological kernel size for mask cleanup (default ``5``).

    **fft** method:
        filter_type : str
            ``"notch"``, ``"bandpass"``, or ``"gaussian_hp"``
            (default ``"gaussian_hp"``).
        cutoff_low : float
            Low-frequency cutoff (default ``30``).
        cutoff_high : float
            High-frequency cutoff (default ``100``).
        threshold_sigma : float
            Number of standard deviations above mean for thresholding
            (default ``3.0``).

    **color** method:
        reference_color_lab : list
            Reference colour as ``[L, a, b]``.
        delta_e_method : str
            ``"CIE76"`` or ``"CIEDE2000"`` (default ``"CIE76"``).
        delta_e_tolerance : float
            Maximum acceptable Delta-E value (default ``10.0``).

    **blob** method:
        blob_threshold_method : str
            ``"otsu"``, ``"fixed"``, or ``"adaptive"``
            (default ``"otsu"``).
        blob_threshold_value : int
            Fixed threshold value when method is ``"fixed"``
            (default ``128``).
        min_area : int
            Minimum blob area in pixels (default ``min_defect_area``).
        max_area : int
            Maximum blob area in pixels (default ``999999``).
        min_circularity : float
            Minimum circularity filter ``0.0``--``1.0``
            (default ``0.0``, disabled).

    **teacher_student** method:
        checkpoint_path : str
            Path to the teacher-student model checkpoint.
        threshold : float
            Anomaly score threshold.

    **normalizing_flow** method:
        checkpoint_path : str
            Path to the normalizing-flow model checkpoint.
        threshold : float
            Anomaly score threshold.

    **unet** method:
        checkpoint_path : str
            Path to the U-Net segmentation model checkpoint.
        threshold : float
            Anomaly score threshold.

    Output data
    -----------
    anomaly_score : float
        Maximum anomaly score across the image (or across all ROIs).
    is_defective : bool
        Whether the score exceeds the configured threshold.
    defect_mask : np.ndarray
        Binary defect mask (uint8, 0 or 255).
    defect_regions : List[dict]
        Each: ``{x, y, w, h, area}``.
    roi_scores : List[dict]
        Per-ROI scores when ``roi_from_previous`` is active.
    """

    def __init__(
        self,
        name: str = "Detect",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, "detect", config)

    def execute(
        self, image: np.ndarray, context: Dict[str, Any]
    ) -> StepResult:
        t0 = time.perf_counter()

        method = self.config.get("method", "autoencoder")
        threshold = float(self.config.get("threshold", 0.5))
        roi_from_previous = bool(self.config.get("roi_from_previous", False))
        min_defect_area = int(self.config.get("min_defect_area", 20))

        # Methods that require a trained checkpoint file.
        _DL_METHODS = {
            "autoencoder", "patchcore", "teacher_student",
            "normalizing_flow", "unet",
        }

        checkpoint_path = self.config.get("checkpoint_path", "")
        if method in _DL_METHODS:
            if not checkpoint_path:
                return StepResult(
                    step_name=self.name,
                    step_type=self.step_type,
                    success=False,
                    data={},
                    elapsed_ms=_elapsed_ms(t0),
                    message="checkpoint_path not configured.",
                )
            cp = Path(checkpoint_path)
            if not cp.exists():
                return StepResult(
                    step_name=self.name,
                    step_type=self.step_type,
                    success=False,
                    data={},
                    elapsed_ms=_elapsed_ms(t0),
                    message=f"Checkpoint not found: {checkpoint_path}",
                )
        else:
            cp = None

        # Determine ROIs from locate context.
        rois: List[Tuple[int, int, int, int]] = []
        if roi_from_previous and "locate" in context:
            rois = context["locate"].get("roi_list", [])

        try:
            if method == "patchcore":
                result_data = self._run_patchcore(
                    image, cp, threshold, rois, min_defect_area
                )
            elif method == "teacher_student":
                result_data = self._run_teacher_student(
                    image, cp, threshold, rois, min_defect_area
                )
            elif method == "normalizing_flow":
                result_data = self._run_normalizing_flow(
                    image, cp, threshold, rois, min_defect_area
                )
            elif method == "unet":
                result_data = self._run_unet(
                    image, cp, threshold, rois, min_defect_area
                )
            elif method == "difference":
                result_data = self._run_difference(
                    image, self.config, rois, min_defect_area
                )
            elif method == "fft":
                result_data = self._run_fft(
                    image, self.config, rois, min_defect_area
                )
            elif method == "color":
                result_data = self._run_color(
                    image, self.config, rois, min_defect_area
                )
            elif method == "blob":
                result_data = self._run_blob(
                    image, self.config, min_defect_area
                )
            else:
                result_data = self._run_autoencoder(
                    image, cp, threshold, rois, min_defect_area
                )
        except Exception as exc:
            logger.exception("DetectStep failed.")
            return StepResult(
                step_name=self.name,
                step_type=self.step_type,
                success=False,
                data={},
                elapsed_ms=_elapsed_ms(t0),
                message=f"Detection error: {exc}",
            )

        # Some non-DL methods may already set is_defective; otherwise derive.
        if "is_defective" not in result_data:
            is_defective = result_data["anomaly_score"] > threshold
            result_data["is_defective"] = is_defective
        else:
            is_defective = result_data["is_defective"]
        result_data["threshold"] = threshold

        # Pop visualisation image out of data before returning.
        vis_image = result_data.pop("_vis_image", None)

        elapsed = _elapsed_ms(t0)
        n_regions = len(result_data.get("defect_regions", []))
        msg = (
            f"Score: {result_data['anomaly_score']:.4f}, "
            f"defective: {is_defective}, "
            f"regions: {n_regions}"
        )
        logger.info("DetectStep [%s]: %s (%.1f ms)", self.name, msg, elapsed)

        return StepResult(
            step_name=self.name,
            step_type=self.step_type,
            success=True,
            data=result_data,
            image=vis_image,
            elapsed_ms=elapsed,
            message=msg,
        )

    # ------------------------------------------------------------------ #
    #  Internal detection runners                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _run_autoencoder(
        image: np.ndarray,
        checkpoint_path: Path,
        threshold: float,
        rois: List[Tuple[int, int, int, int]],
        min_defect_area: int,
    ) -> Dict[str, Any]:
        """Run autoencoder-based anomaly detection."""
        from dl_anomaly.pipeline.inference import InferencePipeline

        pipeline = InferencePipeline(checkpoint_path)

        if rois:
            return DetectStep._run_on_rois(
                image,
                rois,
                threshold,
                min_defect_area,
                scorer_fn=lambda img: DetectStep._score_via_temp_file(
                    img, pipeline.inspect_single
                ),
            )

        result = DetectStep._inspect_array(image, pipeline.inspect_single)
        defect_regions = _extract_defect_regions(
            result.defect_mask, min_defect_area
        )
        return {
            "anomaly_score": float(result.anomaly_score),
            "defect_mask": result.defect_mask,
            "defect_regions": defect_regions,
            "error_map": result.error_map,
            "roi_scores": [],
        }

    @staticmethod
    def _run_patchcore(
        image: np.ndarray,
        checkpoint_path: Path,
        threshold: float,
        rois: List[Tuple[int, int, int, int]],
        min_defect_area: int,
    ) -> Dict[str, Any]:
        """Run PatchCore-based anomaly detection."""
        from dl_anomaly.core.patchcore import PatchCoreInference, PatchCoreModel

        model = PatchCoreModel.load(checkpoint_path)
        inference = PatchCoreInference(model)

        if rois:
            return DetectStep._run_on_rois(
                image,
                rois,
                threshold,
                min_defect_area,
                scorer_fn=lambda img: DetectStep._score_via_temp_file(
                    img, inference.inspect_single
                ),
            )

        result = DetectStep._inspect_array(image, inference.inspect_single)
        defect_regions = _extract_defect_regions(
            result.defect_mask, min_defect_area
        )
        return {
            "anomaly_score": float(result.anomaly_score),
            "defect_mask": result.defect_mask,
            "defect_regions": defect_regions,
            "error_map": result.error_map,
            "roi_scores": [],
        }

    @staticmethod
    def _inspect_array(image: np.ndarray, inspect_fn: Callable) -> Any:
        """Write an array to a temp file and run an inspect_single function."""
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
            cv2.imwrite(tmp_path, image)
        try:
            return inspect_fn(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    @staticmethod
    def _score_via_temp_file(
        image: np.ndarray, inspect_fn: Callable
    ) -> Tuple[float, np.ndarray]:
        """Score a numpy array, returning ``(score, defect_mask)``."""
        result = DetectStep._inspect_array(image, inspect_fn)
        return result.anomaly_score, result.defect_mask

    @staticmethod
    def _run_on_rois(
        image: np.ndarray,
        rois: List[Tuple[int, int, int, int]],
        threshold: float,
        min_defect_area: int,
        scorer_fn: Callable[[np.ndarray], Tuple[float, np.ndarray]],
    ) -> Dict[str, Any]:
        """Run detection independently on each ROI and aggregate results."""
        h, w = image.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        max_score = 0.0
        all_regions: List[Dict[str, Any]] = []
        roi_scores: List[Dict[str, Any]] = []

        for idx, (rx, ry, rw, rh) in enumerate(rois):
            # Clamp ROI to image bounds.
            rx = max(0, rx)
            ry = max(0, ry)
            rw = min(rw, w - rx)
            rh = min(rh, h - ry)
            if rw <= 0 or rh <= 0:
                continue

            crop = image[ry : ry + rh, rx : rx + rw].copy()
            score, mask = scorer_fn(crop)
            max_score = max(max_score, score)

            roi_scores.append({
                "roi_idx": idx,
                "roi": (rx, ry, rw, rh),
                "score": float(score),
                "is_defective": score > threshold,
            })

            # Resize mask to ROI dimensions if needed.
            if mask.shape[:2] != (rh, rw):
                mask = cv2.resize(
                    mask, (rw, rh), interpolation=cv2.INTER_NEAREST
                )

            combined_mask[ry : ry + rh, rx : rx + rw] = np.maximum(
                combined_mask[ry : ry + rh, rx : rx + rw], mask
            )

            # Extract regions with coordinate offsets.
            regions = _extract_defect_regions(mask, min_defect_area)
            for r in regions:
                r["x"] += rx
                r["y"] += ry
                r["roi_idx"] = idx
            all_regions.extend(regions)

        return {
            "anomaly_score": float(max_score),
            "defect_mask": combined_mask,
            "defect_regions": all_regions,
            "roi_scores": roi_scores,
        }

    # ------------------------------------------------------------------ #
    #  Teacher-Student detection runner                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _run_teacher_student(
        image: np.ndarray,
        checkpoint_path: Path,
        threshold: float,
        rois: List[Tuple[int, int, int, int]],
        min_defect_area: int,
    ) -> Dict[str, Any]:
        """Run teacher-student knowledge-distillation anomaly detection."""
        from shared.core.teacher_student import TeacherStudentInference, load_model

        model = load_model(checkpoint_path)
        inference = TeacherStudentInference(model)

        if rois:
            return DetectStep._run_on_rois(
                image,
                rois,
                threshold,
                min_defect_area,
                scorer_fn=lambda img: inference.score_image(img),
            )

        score, error_map = inference.score_image(image)
        defect_mask = (
            (error_map > threshold * error_map.max()).astype(np.uint8) * 255
            if error_map.max() > 0
            else np.zeros_like(error_map, dtype=np.uint8)
        )
        defect_regions = _extract_defect_regions(defect_mask, min_defect_area)
        return {
            "anomaly_score": float(score),
            "defect_mask": defect_mask,
            "defect_regions": defect_regions,
            "error_map": error_map,
            "roi_scores": [],
        }

    # ------------------------------------------------------------------ #
    #  Normalizing-Flow detection runner                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _run_normalizing_flow(
        image: np.ndarray,
        checkpoint_path: Path,
        threshold: float,
        rois: List[Tuple[int, int, int, int]],
        min_defect_area: int,
    ) -> Dict[str, Any]:
        """Run normalizing-flow-based anomaly detection."""
        from shared.core.normalizing_flow import NormFlowInference, load_model

        model = load_model(checkpoint_path)
        inference = NormFlowInference(model)

        if rois:
            return DetectStep._run_on_rois(
                image,
                rois,
                threshold,
                min_defect_area,
                scorer_fn=lambda img: inference.score_image(img),
            )

        score, error_map = inference.score_image(image)
        defect_mask = (
            (error_map > threshold * error_map.max()).astype(np.uint8) * 255
            if error_map.max() > 0
            else np.zeros_like(error_map, dtype=np.uint8)
        )
        defect_regions = _extract_defect_regions(defect_mask, min_defect_area)
        return {
            "anomaly_score": float(score),
            "defect_mask": defect_mask,
            "defect_regions": defect_regions,
            "error_map": error_map,
            "roi_scores": [],
        }

    # ------------------------------------------------------------------ #
    #  U-Net segmentation detection runner                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _run_unet(
        image: np.ndarray,
        checkpoint_path: Path,
        threshold: float,
        rois: List[Tuple[int, int, int, int]],
        min_defect_area: int,
    ) -> Dict[str, Any]:
        """Run U-Net segmentation-based defect detection."""
        from shared.core.unet_segment import UNetInference, load_model

        model = load_model(checkpoint_path)
        inference = UNetInference(model)

        if rois:
            return DetectStep._run_on_rois(
                image,
                rois,
                threshold,
                min_defect_area,
                scorer_fn=lambda img: (
                    inference.score_image(img)
                    if hasattr(inference, "score_image")
                    else (
                        inference.segment(img)[0],
                        (inference.segment(img)[1] * 255).astype(np.uint8),
                    )
                ),
            )

        score, seg_mask = inference.segment(image)
        defect_mask = ((seg_mask > threshold) * 255).astype(np.uint8)
        defect_regions = _extract_defect_regions(defect_mask, min_defect_area)
        return {
            "anomaly_score": float(score),
            "defect_mask": defect_mask,
            "defect_regions": defect_regions,
            "error_map": seg_mask.astype(np.float32),
            "roi_scores": [],
        }

    # ------------------------------------------------------------------ #
    #  Image-difference detection runner                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _run_difference(
        image: np.ndarray,
        config: Dict[str, Any],
        rois: List[Tuple[int, int, int, int]],
        min_defect_area: int,
    ) -> Dict[str, Any]:
        """Run reference-image difference detection."""
        from shared.core.image_difference import ImageDifferencer

        ref_path = config.get("reference_image_path", "")
        if not ref_path:
            raise ValueError(
                "reference_image_path not configured for difference method."
            )
        ref_img = cv2.imread(str(ref_path))
        if ref_img is None:
            raise FileNotFoundError(
                f"Cannot read reference image: {ref_path}"
            )

        differ = ImageDifferencer(
            registration_method=config.get("registration_method", "ecc"),
            threshold=float(config.get("threshold", 30.0)),
            blur_sigma=float(config.get("blur_sigma", 2.0)),
            morph_kernel_size=int(config.get("morph_kernel_size", 5)),
            min_defect_area=min_defect_area,
        )
        differ.set_reference(ref_img)
        result = differ.detect(image)
        return {
            "anomaly_score": result["anomaly_score"],
            "defect_mask": result["defect_mask"],
            "defect_regions": result["defect_regions"],
            "error_map": result["error_map"],
            "roi_scores": [],
            "is_defective": result["is_defective"],
            "alignment_quality": result.get("alignment_quality", 0.0),
        }

    # ------------------------------------------------------------------ #
    #  FFT / frequency-domain detection runner                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _run_fft(
        image: np.ndarray,
        config: Dict[str, Any],
        rois: List[Tuple[int, int, int, int]],
        min_defect_area: int,
    ) -> Dict[str, Any]:
        """Run FFT-based periodic-pattern anomaly detection."""
        from shared.core.frequency import (
            apply_frequency_filter,
            create_bandpass_filter,
            create_gaussian_filter,
            remove_periodic_pattern,
        )

        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if image.ndim == 3
            else image
        )
        filter_type = config.get("filter_type", "gaussian_hp")
        cutoff_low = float(config.get("cutoff_low", 30))
        cutoff_high = float(config.get("cutoff_high", 100))
        threshold_sigma = float(config.get("threshold_sigma", 3.0))

        try:
            filtered = remove_periodic_pattern(
                gray,
                min_distance=int(config.get("min_distance", 10)),
                num_peaks=int(config.get("num_peaks", 5)),
                radius=int(config.get("notch_radius", 8)),
            )
        except Exception:
            # Build a frequency filter mask and apply it.
            h, w = gray.shape[:2]
            if filter_type == "bandpass":
                fmask = create_bandpass_filter(
                    (h, w), low_cutoff=cutoff_low, high_cutoff=cutoff_high
                )
            else:
                # Default: Gaussian high-pass
                fmask = 1.0 - create_gaussian_filter(
                    (h, w), sigma=cutoff_low, filter_type="lowpass"
                )
            filtered = apply_frequency_filter(gray, fmask)

        if filtered.shape == gray.shape:
            diff = cv2.absdiff(gray, filtered)
        else:
            diff = np.abs(
                gray.astype(np.float32) - filtered.astype(np.float32)
            ).astype(np.uint8)

        mean_val = float(np.mean(diff))
        std_val = float(np.std(diff))
        thresh_val = mean_val + threshold_sigma * std_val
        _, defect_mask = cv2.threshold(
            diff, thresh_val, 255, cv2.THRESH_BINARY
        )
        defect_mask = defect_mask.astype(np.uint8)

        score = float(np.max(diff)) / 255.0 if np.max(diff) > 0 else 0.0
        defect_regions = _extract_defect_regions(defect_mask, min_defect_area)
        threshold = float(config.get("threshold", 0.5))
        return {
            "anomaly_score": score,
            "is_defective": score > threshold,
            "defect_mask": defect_mask,
            "defect_regions": defect_regions,
            "error_map": diff.astype(np.float32) / 255.0,
            "roi_scores": [],
        }

    # ------------------------------------------------------------------ #
    #  Colour Delta-E detection runner                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _run_color(
        image: np.ndarray,
        config: Dict[str, Any],
        rois: List[Tuple[int, int, int, int]],
        min_defect_area: int,
    ) -> Dict[str, Any]:
        """Run colour-space Delta-E anomaly detection."""
        from shared.core.color_inspect import compute_delta_e_map

        ref_lab = config.get("reference_color_lab")
        if ref_lab is None:
            raise ValueError(
                "reference_color_lab not configured for color method."
            )
        ref_lab = tuple(ref_lab)
        method = config.get("delta_e_method", "CIE76")
        tolerance = float(config.get("delta_e_tolerance", 10.0))

        if image.ndim == 2:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_bgr = image

        # compute_delta_e_map expects BGR uint8 image + (L, a, b) tuple
        delta_e_map = compute_delta_e_map(
            image_bgr, ref_lab, method=method
        )

        defect_mask = ((delta_e_map > tolerance) * 255).astype(np.uint8)
        score = float(np.mean(delta_e_map)) / 100.0
        defect_regions = _extract_defect_regions(defect_mask, min_defect_area)
        threshold = float(config.get("threshold", 0.5))
        return {
            "anomaly_score": score,
            "is_defective": score > threshold,
            "defect_mask": defect_mask,
            "defect_regions": defect_regions,
            "error_map": delta_e_map.astype(np.float32),
            "roi_scores": [],
        }

    # ------------------------------------------------------------------ #
    #  Blob / connected-component detection runner                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _run_blob(
        image: np.ndarray,
        config: Dict[str, Any],
        min_defect_area: int,
    ) -> Dict[str, Any]:
        """Run blob (connected-component) defect detection."""
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if image.ndim == 3
            else image
        )
        blob_method = config.get("blob_threshold_method", "otsu")
        blob_thresh_val = int(config.get("blob_threshold_value", 128))
        min_area = int(config.get("min_area", min_defect_area))
        max_area = int(config.get("max_area", 999999))
        min_circularity = float(config.get("min_circularity", 0.0))

        if blob_method == "otsu":
            _, binary = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
        elif blob_method == "adaptive":
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 21, 5,
            )
        else:
            _, binary = cv2.threshold(
                gray, blob_thresh_val, 255, cv2.THRESH_BINARY_INV
            )

        num_labels, labels, stats, centroids = (
            cv2.connectedComponentsWithStats(binary, connectivity=8)
        )
        defect_regions: List[Dict[str, Any]] = []
        defect_mask = np.zeros_like(gray, dtype=np.uint8)

        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < min_area or area > max_area:
                continue
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])

            if min_circularity > 0:
                perimeter_mask = (labels == i).astype(np.uint8)
                contours, _ = cv2.findContours(
                    perimeter_mask, cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                if contours:
                    perimeter = cv2.arcLength(contours[0], True)
                    circularity = (
                        (4 * math.pi * area) / (perimeter * perimeter)
                        if perimeter > 0
                        else 0
                    )
                    if circularity < min_circularity:
                        continue

            defect_regions.append(
                {"x": x, "y": y, "w": w, "h": h, "area": area}
            )
            defect_mask[labels == i] = 255

        total_defect_area = int(np.sum(defect_mask > 0))
        score = total_defect_area / max(gray.size, 1)
        threshold = float(config.get("threshold", 0.5))
        return {
            "anomaly_score": score,
            "is_defective": len(defect_regions) > 0,
            "defect_mask": defect_mask,
            "defect_regions": defect_regions,
            "error_map": binary.astype(np.float32) / 255.0,
            "roi_scores": [],
        }


class MeasureStep(FlowStep):
    """Metrology measurement step.

    Config keys
    -----------
    measurements : List[dict]
        Each measurement spec::

            {
                "name": str,
                "type": "distance" | "angle" | "area" | "diameter",
                "points": list | "auto",
                "unit": "px" | "mm",
                "tolerance_min": float,
                "tolerance_max": float,
            }

        When ``points`` is ``"auto"``, reference points are derived from
        defect regions in the context.

    calibration_path : str
        Path to a JSON calibration file containing ``pixels_per_mm`` and
        optionally ``pixels_per_mm_x``, ``pixels_per_mm_y``,
        ``origin_px``, ``rotation_deg``.

    Output data
    -----------
    measurements : List[dict]
        Each: ``{name, value, unit, in_tolerance, tolerance_min,
        tolerance_max, type}``.
    all_in_tolerance : bool
        ``True`` when every measurement is within its tolerance.
    """

    def __init__(
        self,
        name: str = "Measure",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, "measure", config)

    def execute(
        self, image: np.ndarray, context: Dict[str, Any]
    ) -> StepResult:
        t0 = time.perf_counter()

        specs: List[Dict[str, Any]] = self.config.get("measurements", [])
        if not specs:
            return StepResult(
                step_name=self.name,
                step_type=self.step_type,
                success=True,
                data={"measurements": [], "all_in_tolerance": True},
                elapsed_ms=_elapsed_ms(t0),
                message="No measurement specs configured.",
            )

        # Optionally load calibration mapping for mm-unit measurements.
        world_mapping = self._load_world_mapping()

        results: List[Dict[str, Any]] = []
        all_ok = True

        for spec in specs:
            meas_name = spec.get("name", "unnamed")
            meas_type = spec.get("type", "distance")
            points = spec.get("points", "auto")
            unit = spec.get("unit", "px")
            tol_min = spec.get("tolerance_min")
            tol_max = spec.get("tolerance_max")

            try:
                value = self._compute_measurement(
                    image, context, meas_type, points, unit, world_mapping
                )
            except Exception as exc:
                logger.warning(
                    "Measurement '%s' failed: %s", meas_name, exc
                )
                results.append({
                    "name": meas_name,
                    "type": meas_type,
                    "value": None,
                    "unit": unit,
                    "in_tolerance": False,
                    "tolerance_min": tol_min,
                    "tolerance_max": tol_max,
                    "error": str(exc),
                })
                all_ok = False
                continue

            in_tol = _check_tolerance(value, tol_min, tol_max)
            if not in_tol:
                all_ok = False

            results.append({
                "name": meas_name,
                "type": meas_type,
                "value": float(value),
                "unit": unit,
                "in_tolerance": in_tol,
                "tolerance_min": tol_min,
                "tolerance_max": tol_max,
            })

        data: Dict[str, Any] = {
            "measurements": results,
            "all_in_tolerance": all_ok,
        }

        elapsed = _elapsed_ms(t0)
        n_ok = sum(1 for r in results if r.get("in_tolerance", False))
        msg = f"{n_ok}/{len(results)} measurements in tolerance."
        logger.info("MeasureStep [%s]: %s (%.1f ms)", self.name, msg, elapsed)

        return StepResult(
            step_name=self.name,
            step_type=self.step_type,
            success=True,
            data=data,
            elapsed_ms=elapsed,
            message=msg,
        )

    # ------------------------------------------------------------------ #
    #  Calibration loading                                                 #
    # ------------------------------------------------------------------ #

    def _load_world_mapping(self) -> Any:
        """Load a WorldMapping from the calibration_path config, if set."""
        calib_path = self.config.get("calibration_path", "")
        if not calib_path or not Path(calib_path).exists():
            return None

        try:
            from dl_anomaly.core.calibration import WorldMapping

            calib_data = _load_json(calib_path)
            if "pixels_per_mm" not in calib_data:
                logger.warning(
                    "Calibration file does not contain pixels_per_mm; "
                    "falling back to pixel units."
                )
                return None

            return WorldMapping(
                pixels_per_mm_x=calib_data.get(
                    "pixels_per_mm_x", calib_data["pixels_per_mm"]
                ),
                pixels_per_mm_y=calib_data.get(
                    "pixels_per_mm_y", calib_data["pixels_per_mm"]
                ),
                pixels_per_mm=calib_data["pixels_per_mm"],
                origin_px=tuple(calib_data.get("origin_px", (0.0, 0.0))),
                rotation_deg=calib_data.get("rotation_deg", 0.0),
                method=calib_data.get("method", "known_distance"),
            )
        except Exception:
            logger.exception(
                "Failed to load calibration from %s", calib_path
            )
            return None

    # ------------------------------------------------------------------ #
    #  Measurement computation                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_measurement(
        image: np.ndarray,
        context: Dict[str, Any],
        meas_type: str,
        points: Any,
        unit: str,
        world_mapping: Any,
    ) -> float:
        """Compute a single measurement value.

        Parameters
        ----------
        image:
            Input image.
        context:
            Accumulated context from previous steps.
        meas_type:
            One of ``"distance"``, ``"angle"``, ``"area"``, ``"diameter"``.
        points:
            Point specification -- a list of ``[x, y]`` pairs or
            ``"auto"`` to derive from defect regions.
        unit:
            ``"px"`` or ``"mm"``.
        world_mapping:
            Optional ``WorldMapping`` for px-to-mm conversion.

        Returns
        -------
        float
            The computed measurement value.
        """
        if meas_type == "distance":
            pts = MeasureStep._resolve_points(points, context, need=2)
            dx = pts[1][0] - pts[0][0]
            dy = pts[1][1] - pts[0][1]
            dist_px = math.sqrt(dx * dx + dy * dy)
            if unit == "mm" and world_mapping is not None:
                return dist_px / world_mapping.pixels_per_mm
            return dist_px

        if meas_type == "angle":
            pts = MeasureStep._resolve_points(points, context, need=3)
            # Angle at pts[1] between rays to pts[0] and pts[2].
            v1 = (pts[0][0] - pts[1][0], pts[0][1] - pts[1][1])
            v2 = (pts[2][0] - pts[1][0], pts[2][1] - pts[1][1])
            dot = v1[0] * v2[0] + v1[1] * v2[1]
            mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
            mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
            if mag1 < 1e-9 or mag2 < 1e-9:
                return 0.0
            cos_val = max(-1.0, min(1.0, dot / (mag1 * mag2)))
            return math.degrees(math.acos(cos_val))

        if meas_type == "area":
            mask = None
            if "detect" in context:
                mask = context["detect"].get("defect_mask")

            if mask is not None:
                area_px = float(cv2.countNonZero(mask))
            else:
                gray = image
                if gray.ndim == 3:
                    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
                _, bin_mask = cv2.threshold(
                    gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                area_px = float(cv2.countNonZero(bin_mask))

            if unit == "mm" and world_mapping is not None:
                return area_px / (
                    world_mapping.pixels_per_mm_x
                    * world_mapping.pixels_per_mm_y
                )
            return area_px

        if meas_type == "diameter":
            mask = None
            if "detect" in context:
                mask = context["detect"].get("defect_mask")

            if mask is None:
                raise ValueError(
                    "diameter measurement requires a detect step with "
                    "defect_mask in context."
                )

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                return 0.0

            largest = max(contours, key=cv2.contourArea)
            (_, _), radius = cv2.minEnclosingCircle(largest)
            diameter_px = radius * 2.0

            if unit == "mm" and world_mapping is not None:
                return diameter_px / world_mapping.pixels_per_mm
            return diameter_px

        raise ValueError(f"Unknown measurement type: {meas_type!r}")

    @staticmethod
    def _resolve_points(
        points: Any,
        context: Dict[str, Any],
        need: int,
    ) -> List[Tuple[float, float]]:
        """Resolve point specifications into concrete coordinates.

        ``points`` can be:
        - ``"auto"``: derive from defect region centroids in context.
        - A list of ``[x, y]`` pairs.
        """
        if isinstance(points, str) and points == "auto":
            regions: List[Dict[str, Any]] = []
            if "detect" in context:
                regions = context["detect"].get("defect_regions", [])
            if len(regions) < need:
                raise ValueError(
                    f"auto-points need at least {need} defect region(s); "
                    f"found {len(regions)}."
                )
            pts: List[Tuple[float, float]] = []
            for r in regions[:need]:
                cx = r.get("x", 0) + r.get("w", 0) / 2.0
                cy = r.get("y", 0) + r.get("h", 0) / 2.0
                pts.append((cx, cy))
            return pts

        if isinstance(points, (list, tuple)):
            if len(points) < need:
                raise ValueError(
                    f"Need at least {need} point(s); got {len(points)}."
                )
            return [(float(p[0]), float(p[1])) for p in points[:need]]

        raise ValueError(f"Invalid points specification: {points!r}")


class ClassifyStep(FlowStep):
    """Rule-based defect classification step.

    Config keys
    -----------
    rules : List[dict]
        Classification rules.  Each rule::

            {
                "name": str,           # class label
                "area_min": float,
                "area_max": float,
                "aspect_min": float,
                "aspect_max": float,
                "color_range": dict,   # {h_min, h_max, s_min, s_max,
                                       #  v_min, v_max}
            }

    method : str
        ``"rule_based"`` (default), ``"color"``, ``"size"``, ``"shape"``.

    Output data
    -----------
    classifications : List[dict]
        Each: ``{region_idx, class_name, confidence}``.
    class_summary : dict
        Count per class name.
    """

    def __init__(
        self,
        name: str = "Classify",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, "classify", config)

    def execute(
        self, image: np.ndarray, context: Dict[str, Any]
    ) -> StepResult:
        t0 = time.perf_counter()

        rules: List[Dict[str, Any]] = self.config.get("rules", [])
        if not rules:
            return StepResult(
                step_name=self.name,
                step_type=self.step_type,
                success=True,
                data={"classifications": [], "class_summary": {}},
                elapsed_ms=_elapsed_ms(t0),
                message="No classification rules configured.",
            )

        # Retrieve defect regions from detect context.
        regions: List[Dict[str, Any]] = []
        if "detect" in context:
            regions = context["detect"].get("defect_regions", [])

        # Prepare HSV image for colour-based rules.
        hsv: Optional[np.ndarray] = None
        if image.ndim == 3 and image.shape[2] >= 3:
            bgr = image[:, :, :3] if image.shape[2] == 4 else image
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        classifications: List[Dict[str, Any]] = []

        for r_idx, region in enumerate(regions):
            rx = region.get("x", 0)
            ry = region.get("y", 0)
            rw = region.get("w", 0)
            rh = region.get("h", 0)
            area = region.get("area", rw * rh)
            aspect = (rw / rh) if rh > 0 else 0.0

            best_class = "unknown"
            best_conf = 0.0

            for rule in rules:
                match_score = 0.0
                checks = 0

                # Area filter.
                if "area_min" in rule or "area_max" in rule:
                    checks += 1
                    a_min = rule.get("area_min", 0)
                    a_max = rule.get("area_max", float("inf"))
                    if a_min <= area <= a_max:
                        match_score += 1.0

                # Aspect ratio filter.
                if "aspect_min" in rule or "aspect_max" in rule:
                    checks += 1
                    asp_min = rule.get("aspect_min", 0)
                    asp_max = rule.get("aspect_max", float("inf"))
                    if asp_min <= aspect <= asp_max:
                        match_score += 1.0

                # Colour range filter.
                if "color_range" in rule and hsv is not None:
                    checks += 1
                    cr = rule["color_range"]
                    roi_hsv = hsv[ry : ry + rh, rx : rx + rw]
                    if roi_hsv.size > 0:
                        mean_h = float(np.mean(roi_hsv[:, :, 0]))
                        mean_s = float(np.mean(roi_hsv[:, :, 1]))
                        mean_v = float(np.mean(roi_hsv[:, :, 2]))
                        h_ok = cr.get("h_min", 0) <= mean_h <= cr.get(
                            "h_max", 180
                        )
                        s_ok = cr.get("s_min", 0) <= mean_s <= cr.get(
                            "s_max", 255
                        )
                        v_ok = cr.get("v_min", 0) <= mean_v <= cr.get(
                            "v_max", 255
                        )
                        if h_ok and s_ok and v_ok:
                            match_score += 1.0

                confidence = (match_score / checks) if checks > 0 else 0.0
                if confidence > best_conf:
                    best_conf = confidence
                    best_class = rule.get("name", "unnamed")

            classifications.append({
                "region_idx": r_idx,
                "class_name": best_class,
                "confidence": round(best_conf, 4),
            })

        # Build class summary.
        class_summary: Dict[str, int] = {}
        for c in classifications:
            cn = c["class_name"]
            class_summary[cn] = class_summary.get(cn, 0) + 1

        data: Dict[str, Any] = {
            "classifications": classifications,
            "class_summary": class_summary,
        }

        elapsed = _elapsed_ms(t0)
        msg = f"Classified {len(classifications)} region(s): {class_summary}"
        logger.info(
            "ClassifyStep [%s]: %s (%.1f ms)", self.name, msg, elapsed
        )

        return StepResult(
            step_name=self.name,
            step_type=self.step_type,
            success=True,
            data=data,
            elapsed_ms=elapsed,
            message=msg,
        )


class JudgeStep(FlowStep):
    """Final pass/fail judgment step.

    Config keys
    -----------
    rules : List[dict]
        Judgment rules.  Each rule::

            {
                "field": str,       # dot-path into context, e.g.
                                    # "detect.anomaly_score"
                "operator": str,    # "lt", "le", "gt", "ge", "eq", "ne",
                                    # "in_range"
                "value": Any,       # threshold or [min, max] for in_range
                "weight": float,    # for weighted mode (default 1.0)
            }

    logic : str
        ``"all_pass"`` (default), ``"any_pass"``, or ``"weighted"``.
    pass_threshold : float
        For ``weighted`` mode, the minimum weighted score to pass
        (default ``0.5``).

    Output data
    -----------
    overall_pass : bool
    rule_results : List[dict]
        Each: ``{rule, passed, actual_value}``.
    score : float
        Weighted score in ``[0, 1]``.
    """

    def __init__(
        self,
        name: str = "Judge",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, "judge", config)

    def execute(
        self, image: np.ndarray, context: Dict[str, Any]
    ) -> StepResult:
        t0 = time.perf_counter()

        rules: List[Dict[str, Any]] = self.config.get("rules", [])
        logic = self.config.get("logic", "all_pass")
        pass_threshold = float(self.config.get("pass_threshold", 0.5))

        if not rules:
            return StepResult(
                step_name=self.name,
                step_type=self.step_type,
                success=True,
                data={
                    "overall_pass": True,
                    "rule_results": [],
                    "score": 1.0,
                },
                elapsed_ms=_elapsed_ms(t0),
                message="No judgment rules configured; defaulting to PASS.",
            )

        rule_results: List[Dict[str, Any]] = []
        total_weight = 0.0
        weighted_sum = 0.0

        for rule_spec in rules:
            field_path = rule_spec.get("field", "")
            operator = rule_spec.get("operator", "lt")
            target_value = rule_spec.get("value")
            weight = float(rule_spec.get("weight", 1.0))

            actual = _resolve_field_path(context, field_path)
            passed = _evaluate_rule(actual, operator, target_value)

            rule_results.append({
                "rule": rule_spec,
                "passed": passed,
                "actual_value": _safe_serialize(actual),
            })

            total_weight += weight
            if passed:
                weighted_sum += weight

        score = (weighted_sum / total_weight) if total_weight > 0 else 0.0

        if logic == "all_pass":
            overall_pass = all(r["passed"] for r in rule_results)
        elif logic == "any_pass":
            overall_pass = any(r["passed"] for r in rule_results)
        elif logic == "weighted":
            overall_pass = score >= pass_threshold
        else:
            logger.warning(
                "Unknown logic mode '%s'; falling back to all_pass.", logic
            )
            overall_pass = all(r["passed"] for r in rule_results)

        data: Dict[str, Any] = {
            "overall_pass": overall_pass,
            "rule_results": rule_results,
            "score": round(score, 4),
        }

        elapsed = _elapsed_ms(t0)
        verdict = "PASS" if overall_pass else "FAIL"
        n_passed = sum(1 for r in rule_results if r["passed"])
        msg = (
            f"{verdict} ({n_passed}/{len(rule_results)} rules passed, "
            f"score={score:.3f})"
        )
        logger.info("JudgeStep [%s]: %s (%.1f ms)", self.name, msg, elapsed)

        return StepResult(
            step_name=self.name,
            step_type=self.step_type,
            success=True,
            data=data,
            elapsed_ms=elapsed,
            message=msg,
        )


class CustomStep(FlowStep):
    """User-defined Python function step.

    Config keys
    -----------
    function_code : str
        Python source code defining a function
        ``process(image, context) -> dict``.  The returned dictionary
        becomes the step's output data.

    Output data
    -----------
    Whatever the user function returns.
    """

    def __init__(
        self,
        name: str = "Custom",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, "custom", config)

    def execute(
        self, image: np.ndarray, context: Dict[str, Any]
    ) -> StepResult:
        t0 = time.perf_counter()

        function_code = self.config.get("function_code", "")
        if not function_code:
            return StepResult(
                step_name=self.name,
                step_type=self.step_type,
                success=False,
                data={},
                elapsed_ms=_elapsed_ms(t0),
                message="No function_code configured.",
            )

        # -------------------------------------------------------------- #
        #  AST-based sandboxing: validate the code before execution.     #
        # -------------------------------------------------------------- #
        try:
            tree = ast.parse(function_code, filename=f"<custom:{self.name}>")
        except SyntaxError as exc:
            return StepResult(
                step_name=self.name,
                step_type=self.step_type,
                success=False,
                data={},
                elapsed_ms=_elapsed_ms(t0),
                message=f"Code syntax error: {exc}",
            )

        # Node types that are never allowed in user-supplied code.
        _DISALLOWED_NODES: Tuple[type, ...] = (
            ast.Import,
            ast.ImportFrom,
            ast.Global,
            ast.Nonlocal,
        )
        # Built-in names that must not be called.
        _DANGEROUS_BUILTINS = frozenset({
            "exec", "eval", "compile", "__import__", "globals",
            "locals", "getattr", "setattr", "delattr", "vars",
            "breakpoint", "open", "input", "memoryview",
        })

        for node in ast.walk(tree):
            if isinstance(node, _DISALLOWED_NODES):
                return StepResult(
                    step_name=self.name,
                    step_type=self.step_type,
                    success=False,
                    data={},
                    elapsed_ms=_elapsed_ms(t0),
                    message=(
                        "Security violation: import statements are "
                        "not allowed in custom step code."
                    ),
                )
            # Block access to dunder attributes (e.g. __class__, __builtins__).
            if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
                return StepResult(
                    step_name=self.name,
                    step_type=self.step_type,
                    success=False,
                    data={},
                    elapsed_ms=_elapsed_ms(t0),
                    message=(
                        f"Security violation: access to dunder attribute "
                        f"'{node.attr}' is not allowed."
                    ),
                )
            # Block string constants containing dunder patterns (prevents indirect access).
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and re.search(r'__\w+__', node.value):
                return StepResult(
                    step_name=self.name,
                    step_type=self.step_type,
                    success=False,
                    data={},
                    elapsed_ms=_elapsed_ms(t0),
                    message=(
                        "Security violation: string constants containing "
                        "dunder patterns are not allowed."
                    ),
                )
            # Block calls to dangerous built-in functions.
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in _DANGEROUS_BUILTINS
            ):
                return StepResult(
                    step_name=self.name,
                    step_type=self.step_type,
                    success=False,
                    data={},
                    elapsed_ms=_elapsed_ms(t0),
                    message=(
                        f"Security violation: call to '{node.func.id}' "
                        f"is not allowed."
                    ),
                )

        # Restricted namespace: __builtins__ set to empty dict so that
        # default builtins (like __import__) are inaccessible at runtime.
        _SAFE_BUILTINS = {
            "abs": abs, "all": all, "any": any, "bool": bool,
            "dict": dict, "enumerate": enumerate, "filter": filter,
            "float": float, "int": int, "isinstance": isinstance,
            "len": len, "list": list, "map": map, "max": max,
            "min": min, "print": print, "range": range, "round": round,
            "set": set, "sorted": sorted, "str": str, "sum": sum,
            "tuple": tuple, "zip": zip,
        }
        namespace: Dict[str, Any] = {
            "__builtins__": _SAFE_BUILTINS,
            "np": np,
            "cv2": cv2,
            "math": math,
        }

        def _exec_with_timeout(code_obj: Any, ns: Dict[str, Any], timeout_sec: int = 5) -> None:
            """Execute code with a timeout to prevent infinite loops."""
            global _LEAKED_THREAD_COUNT
            if _LEAKED_THREAD_COUNT >= _MAX_LEAKED_THREADS:
                raise RuntimeError(
                    f"Too many timed-out custom step threads ({_LEAKED_THREAD_COUNT}). "
                    "Restart the application to recover."
                )

            exc_holder: list = [None]

            def _target() -> None:
                try:
                    exec(code_obj, ns)  # noqa: S102
                except Exception as e:
                    exc_holder[0] = e

            thread = threading.Thread(target=_target, daemon=True)
            thread.start()
            thread.join(timeout=timeout_sec)
            if thread.is_alive():
                _LEAKED_THREAD_COUNT += 1
                logger.warning(
                    "Custom step execution timed out. Leaked thread count: %d/%d",
                    _LEAKED_THREAD_COUNT, _MAX_LEAKED_THREADS,
                )
                raise TimeoutError("Custom step execution timed out.")
            if exc_holder[0] is not None:
                raise exc_holder[0]

        try:
            code_obj = compile(tree, f"<custom:{self.name}>", "exec")
            _exec_with_timeout(code_obj, namespace)
        except TimeoutError as exc:
            return StepResult(
                step_name=self.name,
                step_type=self.step_type,
                success=False,
                data={},
                elapsed_ms=_elapsed_ms(t0),
                message=f"Execution timeout: {exc}",
            )
        except Exception as exc:
            return StepResult(
                step_name=self.name,
                step_type=self.step_type,
                success=False,
                data={},
                elapsed_ms=_elapsed_ms(t0),
                message=f"Code compilation error: {exc}",
            )

        process_fn = namespace.get("process")
        if process_fn is None or not callable(process_fn):
            return StepResult(
                step_name=self.name,
                step_type=self.step_type,
                success=False,
                data={},
                elapsed_ms=_elapsed_ms(t0),
                message=(
                    "function_code must define a callable "
                    "'process(image, context)'."
                ),
            )

        try:
            result = process_fn(image, context)
        except Exception as exc:
            logger.exception("CustomStep [%s] execution failed.", self.name)
            return StepResult(
                step_name=self.name,
                step_type=self.step_type,
                success=False,
                data={},
                elapsed_ms=_elapsed_ms(t0),
                message=f"Execution error: {exc}",
            )

        if not isinstance(result, dict):
            result = {"output": result}

        elapsed = _elapsed_ms(t0)
        logger.info(
            "CustomStep [%s]: completed (%.1f ms)", self.name, elapsed
        )

        return StepResult(
            step_name=self.name,
            step_type=self.step_type,
            success=True,
            data=result,
            elapsed_ms=elapsed,
            message="Custom step completed.",
        )


# ====================================================================== #
#  Step Registry                                                           #
# ====================================================================== #

STEP_REGISTRY: Dict[str, Type[FlowStep]] = {
    "locate": LocateStep,
    "detect": DetectStep,
    "measure": MeasureStep,
    "classify": ClassifyStep,
    "judge": JudgeStep,
    "custom": CustomStep,
}


def create_step(
    step_type: str,
    name: str,
    config: Optional[Dict[str, Any]] = None,
) -> FlowStep:
    """Create a step instance by type name.

    Parameters
    ----------
    step_type:
        One of ``"locate"``, ``"detect"``, ``"measure"``, ``"classify"``,
        ``"judge"``, ``"custom"``.
    name:
        Human-readable name for the step.
    config:
        Step-specific configuration dictionary.

    Returns
    -------
    FlowStep

    Raises
    ------
    ValueError
        If *step_type* is not in :data:`STEP_REGISTRY`.
    """
    step_cls = STEP_REGISTRY.get(step_type)
    if step_cls is None:
        raise ValueError(
            f"Unknown step_type '{step_type}'. "
            f"Available: {list(STEP_REGISTRY.keys())}"
        )
    return step_cls(name=name, config=config or {})


# ====================================================================== #
#  Flow Engine                                                             #
# ====================================================================== #


class InspectionFlow:
    """Orchestrates a sequence of :class:`FlowStep` instances.

    Usage::

        flow = InspectionFlow("PCB Inspection")
        flow.add_step(LocateStep("Locate IC", config={...}))
        flow.add_step(DetectStep("Defect Detection", config={...}))
        flow.add_step(MeasureStep("Dimension Check", config={...}))
        flow.add_step(JudgeStep("Pass/Fail", config={...}))

        result = flow.execute(image)
        print(result.overall_pass)

    Parameters
    ----------
    name:
        Human-readable name for this flow.
    stop_on_failure:
        When ``True`` (default), abort the flow on the first failed step.
        When ``False``, continue executing subsequent steps even after a
        failure.
    """

    def __init__(
        self,
        name: str = "Inspection Flow",
        stop_on_failure: bool = True,
    ) -> None:
        self.name = name
        self.stop_on_failure = stop_on_failure
        self._steps: List[FlowStep] = []
        self._on_step_complete: Optional[
            Callable[[int, StepResult], None]
        ] = None

    # ------------------------------------------------------------------ #
    #  Step management                                                     #
    # ------------------------------------------------------------------ #

    def add_step(self, step: FlowStep) -> None:
        """Append a step to the end of the flow."""
        self._steps.append(step)

    def insert_step(self, index: int, step: FlowStep) -> None:
        """Insert a step at the given index."""
        self._steps.insert(index, step)

    def remove_step(self, index: int) -> None:
        """Remove the step at the given index.

        Raises
        ------
        IndexError
            If *index* is out of range.
        """
        if not 0 <= index < len(self._steps):
            raise IndexError(
                f"Step index {index} out of range "
                f"(0..{len(self._steps) - 1})."
            )
        self._steps.pop(index)

    def move_step(self, from_idx: int, to_idx: int) -> None:
        """Move a step from one position to another.

        Raises
        ------
        IndexError
            If either index is out of range.
        """
        n = len(self._steps)
        if not 0 <= from_idx < n:
            raise IndexError(f"from_idx {from_idx} out of range.")
        if not 0 <= to_idx < n:
            raise IndexError(f"to_idx {to_idx} out of range.")
        step = self._steps.pop(from_idx)
        self._steps.insert(to_idx, step)

    def get_steps(self) -> List[FlowStep]:
        """Return a shallow copy of the step list."""
        return list(self._steps)

    def __len__(self) -> int:
        return len(self._steps)

    def __repr__(self) -> str:
        return (
            f"<InspectionFlow name={self.name!r} "
            f"steps={len(self._steps)}>"
        )

    # ------------------------------------------------------------------ #
    #  Execution                                                           #
    # ------------------------------------------------------------------ #

    @log_operation(logger)
    def execute(
        self,
        image: np.ndarray,
        on_step_complete: Optional[
            Callable[[int, StepResult], None]
        ] = None,
    ) -> FlowResult:
        """Execute all enabled steps sequentially.

        Each step receives the original input image and a *context*
        dictionary that accumulates output data from all previous steps.
        Data is merged into ``context[step.step_type]``.

        Parameters
        ----------
        image:
            Input image (BGR uint8 or grayscale).
        on_step_complete:
            Optional callback invoked after each step finishes, receiving
            ``(step_index, step_result)``.

        Returns
        -------
        FlowResult
        """
        flow_t0 = time.perf_counter()
        context: Dict[str, Any] = {}
        step_results: List[StepResult] = []
        aborted = False

        for idx, step in enumerate(self._steps):
            if not step.enabled:
                logger.debug(
                    "Skipping disabled step %d: %s", idx, step.name
                )
                continue

            logger.info(
                "Flow [%s] executing step %d/%d: %s (%s)",
                self.name,
                idx + 1,
                len(self._steps),
                step.name,
                step.step_type,
            )

            try:
                result = step.execute(image, context)
            except Exception as exc:
                logger.exception(
                    "Unhandled exception in step %d (%s).", idx, step.name
                )
                result = StepResult(
                    step_name=step.name,
                    step_type=step.step_type,
                    success=False,
                    data={},
                    elapsed_ms=_elapsed_ms(flow_t0),
                    message=f"Unhandled exception: {exc}",
                )

            step_results.append(result)

            # Merge step data into context.
            if result.success and result.data:
                context[step.step_type] = result.data

            # Invoke callback.
            if on_step_complete is not None:
                try:
                    on_step_complete(idx, result)
                except Exception:
                    logger.exception(
                        "on_step_complete callback failed for step %d.", idx
                    )

            # Check for abort.
            if not result.success and self.stop_on_failure:
                logger.warning(
                    "Flow [%s] aborting after step %d (%s) failure: %s",
                    self.name,
                    idx,
                    step.name,
                    result.message,
                )
                aborted = True
                break

        # Determine overall pass/fail.
        overall_pass = self._determine_overall_pass(step_results, aborted)

        total_ms = _elapsed_ms(flow_t0)
        ts = datetime.now(timezone.utc).isoformat()

        # Build summary.
        summary = self._build_summary(step_results, context)

        flow_result = FlowResult(
            flow_name=self.name,
            overall_pass=overall_pass,
            steps=step_results,
            total_time_ms=total_ms,
            timestamp=ts,
            summary=summary,
        )

        verdict = "PASS" if overall_pass else "FAIL"
        logger.info(
            "Flow [%s] completed: %s (%d steps, %.1f ms)",
            self.name,
            verdict,
            len(step_results),
            total_ms,
        )

        return flow_result

    def execute_batch(
        self,
        images: List[Tuple[str, np.ndarray]],
        on_result: Optional[Callable[[int, FlowResult], None]] = None,
    ) -> List[FlowResult]:
        """Execute the flow on multiple images sequentially.

        Parameters
        ----------
        images:
            List of ``(image_path_or_id, image_array)`` tuples.
        on_result:
            Optional callback invoked after each image, receiving
            ``(image_index, flow_result)``.

        Returns
        -------
        List[FlowResult]
        """
        results: List[FlowResult] = []

        for idx, (img_id, img) in enumerate(images):
            logger.info(
                "Flow [%s] batch %d/%d: %s",
                self.name,
                idx + 1,
                len(images),
                img_id,
            )

            try:
                result = self.execute(img)
                result.source_image_path = img_id
            except Exception as exc:
                logger.exception(
                    "Flow batch failed on image %d (%s).", idx, img_id
                )
                result = FlowResult(
                    flow_name=self.name,
                    overall_pass=False,
                    steps=[],
                    total_time_ms=0.0,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    source_image_path=img_id,
                    summary={"error": str(exc)},
                )

            results.append(result)

            if on_result is not None:
                try:
                    on_result(idx, result)
                except Exception:
                    logger.exception(
                        "on_result callback failed for image %d.", idx
                    )

        return results

    # ------------------------------------------------------------------ #
    #  Persistence                                                         #
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        """Save the flow definition to a JSON file.

        Only the flow structure and step configurations are saved, not
        runtime state or trained model weights.

        Parameters
        ----------
        path:
            Destination file path (will be created/overwritten).
        """
        data = {
            "flow_name": self.name,
            "stop_on_failure": self.stop_on_failure,
            "steps": [s.to_dict() for s in self._steps],
            "version": "1.0",
        }
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        logger.info("Flow saved to %s (%d steps).", path, len(self._steps))

    @classmethod
    def load(cls, path: str) -> "InspectionFlow":
        """Load a flow definition from a JSON file.

        Parameters
        ----------
        path:
            Source file path.

        Returns
        -------
        InspectionFlow

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Flow file not found: {path}")

        with open(p, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        flow = cls(
            name=data.get("flow_name", "Loaded Flow"),
            stop_on_failure=data.get("stop_on_failure", True),
        )

        for step_dict in data.get("steps", []):
            step = FlowStep.from_dict(step_dict)
            flow.add_step(step)

        logger.info(
            "Flow loaded from %s: '%s' with %d steps.",
            path,
            flow.name,
            len(flow._steps),
        )
        return flow

    # ------------------------------------------------------------------ #
    #  Validation                                                          #
    # ------------------------------------------------------------------ #

    def validate(self) -> List[str]:
        """Check the flow configuration for potential issues.

        Returns
        -------
        List[str]
            Warning messages.  An empty list means no issues detected.
        """
        warnings: List[str] = []

        if not self._steps:
            warnings.append("Flow has no steps configured.")
            return warnings

        step_types_seen: List[str] = []

        for idx, step in enumerate(self._steps):
            prefix = f"Step {idx} ({step.name!r}, {step.step_type})"

            if not step.enabled:
                warnings.append(f"{prefix}: step is disabled.")

            # Check locate step.
            if step.step_type == "locate":
                tp = step.config.get("template_path", "")
                if not tp:
                    warnings.append(
                        f"{prefix}: template_path is not configured."
                    )
                elif not Path(tp).exists():
                    warnings.append(
                        f"{prefix}: template file does not exist: {tp}"
                    )

            # Check detect step.
            if step.step_type == "detect":
                cp = step.config.get("checkpoint_path", "")
                if not cp:
                    warnings.append(
                        f"{prefix}: checkpoint_path is not configured."
                    )
                elif not Path(cp).exists():
                    warnings.append(
                        f"{prefix}: checkpoint file does not exist: {cp}"
                    )
                if (
                    step.config.get("roi_from_previous")
                    and "locate" not in step_types_seen
                ):
                    warnings.append(
                        f"{prefix}: roi_from_previous=True but no locate "
                        f"step precedes it."
                    )

            # Check measure step.
            if step.step_type == "measure":
                specs = step.config.get("measurements", [])
                if not specs:
                    warnings.append(
                        f"{prefix}: no measurement specs configured."
                    )
                for m_idx, spec in enumerate(specs):
                    pts = spec.get("points", "auto")
                    if pts == "auto" and "detect" not in step_types_seen:
                        warnings.append(
                            f"{prefix}: measurement {m_idx} uses 'auto' "
                            f"points but no detect step precedes it."
                        )

            # Check judge step.
            if step.step_type == "judge":
                judge_rules = step.config.get("rules", [])
                if not judge_rules:
                    warnings.append(
                        f"{prefix}: no judgment rules configured."
                    )

            # Check custom step.
            if step.step_type == "custom":
                code = step.config.get("function_code", "")
                if not code:
                    warnings.append(f"{prefix}: function_code is empty.")
                elif "def process" not in code:
                    warnings.append(
                        f"{prefix}: function_code does not define "
                        f"'process(image, context)'."
                    )

            step_types_seen.append(step.step_type)

        # Warn if there is no judge step.
        if "judge" not in step_types_seen:
            warnings.append(
                "Flow has no judge step; overall_pass will default to "
                "True unless a step fails."
            )

        return warnings

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _determine_overall_pass(
        step_results: List[StepResult],
        aborted: bool,
    ) -> bool:
        """Determine the final pass/fail verdict.

        Uses the judge step's ``overall_pass`` if available; otherwise
        returns ``True`` only if all steps succeeded and the flow was not
        aborted.
        """
        if aborted:
            return False

        # Look for the last judge step result.
        for result in reversed(step_results):
            if result.step_type == "judge" and result.success:
                return bool(result.data.get("overall_pass", False))

        # No judge step -- pass if all steps succeeded.
        return all(r.success for r in step_results)

    @staticmethod
    def _build_summary(
        step_results: List[StepResult],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Aggregate a summary from all step results."""
        summary: Dict[str, Any] = {
            "steps_executed": len(step_results),
            "steps_succeeded": sum(1 for r in step_results if r.success),
            "steps_failed": sum(
                1 for r in step_results if not r.success
            ),
        }

        # Pull key metrics from context.
        if "detect" in context:
            summary["anomaly_score"] = context["detect"].get(
                "anomaly_score"
            )
            summary["is_defective"] = context["detect"].get("is_defective")
            summary["num_defect_regions"] = len(
                context["detect"].get("defect_regions", [])
            )

        if "measure" in context:
            summary["all_in_tolerance"] = context["measure"].get(
                "all_in_tolerance"
            )

        if "judge" in context:
            summary["judgment_score"] = context["judge"].get("score")

        if "classify" in context:
            summary["class_summary"] = context["classify"].get(
                "class_summary"
            )

        if "locate" in context:
            summary["num_matches"] = context["locate"].get("num_matches")

        return summary


# ====================================================================== #
#  Factory Functions                                                       #
# ====================================================================== #


def create_simple_inspect_flow(
    checkpoint_path: str,
    threshold: float = 0.5,
    method: str = "autoencoder",
) -> InspectionFlow:
    """Create a simple detect -> judge flow.

    Parameters
    ----------
    checkpoint_path:
        Path to the trained model checkpoint.
    threshold:
        Anomaly score threshold.
    method:
        ``"autoencoder"`` or ``"patchcore"``.

    Returns
    -------
    InspectionFlow
    """
    flow = InspectionFlow("Simple Inspection")

    flow.add_step(
        DetectStep(
            "Defect Detection",
            config={
                "method": method,
                "checkpoint_path": checkpoint_path,
                "threshold": threshold,
            },
        )
    )

    flow.add_step(
        JudgeStep(
            "Pass/Fail",
            config={
                "rules": [
                    {
                        "field": "detect.is_defective",
                        "operator": "eq",
                        "value": False,
                    },
                ],
                "logic": "all_pass",
            },
        )
    )

    return flow


def create_locate_and_inspect_flow(
    template_path: str,
    checkpoint_path: str,
    threshold: float = 0.5,
    method: str = "autoencoder",
) -> InspectionFlow:
    """Create a locate -> detect -> judge flow.

    Parameters
    ----------
    template_path:
        Path to the template image.
    checkpoint_path:
        Path to the trained model checkpoint.
    threshold:
        Anomaly score threshold.
    method:
        ``"autoencoder"`` or ``"patchcore"``.

    Returns
    -------
    InspectionFlow
    """
    flow = InspectionFlow("Locate & Inspect")

    flow.add_step(
        LocateStep(
            "Locate Component",
            config={
                "template_path": template_path,
                "min_score": 0.5,
                "num_matches": 1,
            },
        )
    )

    flow.add_step(
        DetectStep(
            "Defect Detection",
            config={
                "method": method,
                "checkpoint_path": checkpoint_path,
                "threshold": threshold,
                "roi_from_previous": True,
            },
        )
    )

    flow.add_step(
        JudgeStep(
            "Pass/Fail",
            config={
                "rules": [
                    {
                        "field": "detect.is_defective",
                        "operator": "eq",
                        "value": False,
                    },
                ],
                "logic": "all_pass",
            },
        )
    )

    return flow


def create_full_inspection_flow(
    template_path: str,
    checkpoint_path: str,
    calibration_path: Optional[str] = None,
    measurements: Optional[List[Dict[str, Any]]] = None,
    threshold: float = 0.5,
    method: str = "autoencoder",
) -> InspectionFlow:
    """Create a full locate -> detect -> measure -> judge flow.

    Parameters
    ----------
    template_path:
        Path to the template image.
    checkpoint_path:
        Path to the trained model checkpoint.
    calibration_path:
        Optional path to a calibration file for mm-unit measurements.
    measurements:
        List of measurement specs (see :class:`MeasureStep` for format).
    threshold:
        Anomaly score threshold.
    method:
        ``"autoencoder"`` or ``"patchcore"``.

    Returns
    -------
    InspectionFlow
    """
    flow = InspectionFlow("Full Inspection")

    flow.add_step(
        LocateStep(
            "Locate Component",
            config={
                "template_path": template_path,
                "min_score": 0.5,
                "num_matches": 1,
            },
        )
    )

    flow.add_step(
        DetectStep(
            "Defect Detection",
            config={
                "method": method,
                "checkpoint_path": checkpoint_path,
                "threshold": threshold,
                "roi_from_previous": True,
            },
        )
    )

    measure_config: Dict[str, Any] = {
        "measurements": measurements or [],
    }
    if calibration_path:
        measure_config["calibration_path"] = calibration_path

    flow.add_step(MeasureStep("Dimension Check", config=measure_config))

    flow.add_step(
        JudgeStep(
            "Final Judgment",
            config={
                "rules": [
                    {
                        "field": "detect.is_defective",
                        "operator": "eq",
                        "value": False,
                        "weight": 2.0,
                    },
                    {
                        "field": "measure.all_in_tolerance",
                        "operator": "eq",
                        "value": True,
                        "weight": 1.0,
                    },
                ],
                "logic": "all_pass",
            },
        )
    )

    return flow


# ====================================================================== #
#  Internal Utility Functions                                              #
# ====================================================================== #


def _elapsed_ms(t0: float) -> float:
    """Compute elapsed milliseconds since *t0*."""
    return (time.perf_counter() - t0) * 1000.0


def _extract_defect_regions(
    mask: np.ndarray,
    min_area: int = 20,
) -> List[Dict[str, Any]]:
    """Run connected-component analysis and return region descriptors.

    Parameters
    ----------
    mask:
        Binary mask (uint8, 0 or 255).
    min_area:
        Minimum connected-component area to include.

    Returns
    -------
    List[dict]
        Each: ``{x, y, w, h, area}``.
    """
    if mask is None or mask.size == 0:
        return []

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    regions: List[Dict[str, Any]] = []
    for i in range(1, num_labels):  # skip background
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        regions.append({
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
            "area": int(area),
        })

    return regions


def _resolve_field_path(context: Dict[str, Any], field_path: str) -> Any:
    """Resolve a dot-path like ``"detect.anomaly_score"`` into a value.

    Supports bracket indexing for lists, e.g.
    ``"measure.measurements[0].value"``.

    Returns ``None`` if the path cannot be resolved.
    """
    if not field_path:
        return None

    # Split on dots while preserving bracket notation.
    # Example: "measure.measurements[0].value"
    #   -> ["measure", "measurements[0]", "value"]
    parts = field_path.split(".")
    current: Any = context

    for part in parts:
        if current is None:
            return None

        # Check for bracket indexing: "measurements[0]"
        bracket_match = re.match(r"^(\w+)\[(\d+)\]$", part)
        if bracket_match:
            key = bracket_match.group(1)
            index = int(bracket_match.group(2))

            if isinstance(current, dict):
                current = current.get(key)
            else:
                current = getattr(current, key, None)

            if current is None:
                return None

            if isinstance(current, (list, tuple)):
                if 0 <= index < len(current):
                    current = current[index]
                else:
                    return None
            else:
                return None
        else:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                current = getattr(current, part, None)

    return current


def _evaluate_rule(actual: Any, operator: str, target: Any) -> bool:
    """Evaluate a comparison rule.

    Supported operators: ``"lt"``, ``"le"``, ``"gt"``, ``"ge"``,
    ``"eq"``, ``"ne"``, ``"in_range"`` (target is ``[min, max]``).

    Returns ``False`` if evaluation is not possible (e.g. ``actual`` is
    ``None``).
    """
    if actual is None:
        return False

    try:
        if operator == "lt":
            return float(actual) < float(target)
        if operator == "le":
            return float(actual) <= float(target)
        if operator == "gt":
            return float(actual) > float(target)
        if operator == "ge":
            return float(actual) >= float(target)
        if operator == "eq":
            # Handle bool/numeric comparison gracefully.
            if isinstance(target, bool) or isinstance(actual, bool):
                return bool(actual) == bool(target)
            return actual == target
        if operator == "ne":
            if isinstance(target, bool) or isinstance(actual, bool):
                return bool(actual) != bool(target)
            return actual != target
        if operator == "in_range":
            if isinstance(target, (list, tuple)) and len(target) == 2:
                return (
                    float(target[0]) <= float(actual) <= float(target[1])
                )
            return False
    except (ValueError, TypeError):
        logger.debug(
            "Rule evaluation failed: actual=%r op=%s target=%r",
            actual,
            operator,
            target,
        )
        return False

    logger.warning("Unknown operator: %r", operator)
    return False


def _check_tolerance(
    value: float,
    tol_min: Optional[float],
    tol_max: Optional[float],
) -> bool:
    """Check whether *value* falls within the tolerance range.

    Returns ``True`` if within tolerance or if no tolerances are set.
    """
    if tol_min is not None and value < tol_min:
        return False
    if tol_max is not None and value > tol_max:
        return False
    return True


def _safe_serialize(value: Any) -> Any:
    """Convert a value to a JSON-safe type for logging and result storage."""
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.ndarray):
        return f"<ndarray shape={value.shape} dtype={value.dtype}>"
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, (list, tuple)):
        return [_safe_serialize(v) for v in value]
    if isinstance(value, dict):
        return {k: _safe_serialize(v) for k, v in value.items()}
    return str(value)


def _load_json(path: str) -> Dict[str, Any]:
    """Load a JSON file and return its contents as a dictionary."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
