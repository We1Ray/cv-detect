"""ONNX model loading, export, and inference engine.

Provides utilities for exporting PyTorch models (e.g. AnomalyAutoencoder) to
ONNX format and running inference via onnxruntime.  This enables deployment
of trained models without requiring a full PyTorch installation at runtime.

The module gracefully degrades when ``onnxruntime`` is not installed --
functions raise :class:`ImportError` with installation instructions instead
of crashing at module import time.
"""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import cv2
import numpy as np
import torch

from shared.op_logger import log_operation

logger = logging.getLogger(__name__)


# ======================================================================
# onnxruntime availability helpers
# ======================================================================

def check_onnxruntime_available() -> bool:
    """Return *True* if ``onnxruntime`` (CPU or GPU) is importable."""
    try:
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


def get_available_providers() -> List[str]:
    """Return the list of execution providers available in the current onnxruntime build."""
    _require_onnxruntime()
    import onnxruntime as ort
    return ort.get_available_providers()


def _require_onnxruntime() -> None:
    """Raise a helpful :class:`ImportError` if onnxruntime is missing."""
    if not check_onnxruntime_available():
        raise ImportError(
            "onnxruntime is required but not installed. "
            "Install it with:  pip install onnxruntime  "
            "or for GPU support:  pip install onnxruntime-gpu"
        )


# ======================================================================
# Data class
# ======================================================================

@dataclass
class OnnxModelInfo:
    """Metadata container describing an ONNX model file."""

    path: str
    input_names: List[str]
    output_names: List[str]
    input_shapes: Dict[str, List[int]]
    output_shapes: Dict[str, List[int]]
    model_type: str  # "autoencoder", "classifier", "segmentation", "custom"
    metadata: Dict[str, str] = field(default_factory=dict)
    opset_version: int = 14


# ======================================================================
# Export
# ======================================================================

@log_operation(logger)
def export_to_onnx(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    output_path: Union[str, Path],
    model_type: str = "autoencoder",
    metadata: Optional[Dict[str, str]] = None,
    opset_version: int = 14,
) -> OnnxModelInfo:
    """Export a PyTorch model to ONNX and verify the result.

    Parameters
    ----------
    model:
        A PyTorch ``nn.Module`` in eval mode.
    dummy_input:
        A tensor with the expected input shape (including batch dim).
    output_path:
        Destination ``.onnx`` file path.
    model_type:
        Label stored in model metadata.
    metadata:
        Extra key-value pairs to embed in the ONNX model properties.
    opset_version:
        ONNX opset version (default 14).

    Returns
    -------
    OnnxModelInfo
    """
    import onnx

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()

    input_names = ["input"]
    output_names = ["output"]
    dynamic_axes = {
        "input": {0: "batch_size"},
        "output": {0: "batch_size"},
    }

    logger.info("Exporting model to ONNX: %s  (opset=%d)", output_path, opset_version)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
    )

    # Embed metadata ---------------------------------------------------
    onnx_model = onnx.load(str(output_path))

    all_metadata: Dict[str, str] = {
        "model_type": model_type,
        "image_size": str(dummy_input.shape[-1]),
        "input_channels": str(dummy_input.shape[1]),
        "creation_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "opset_version": str(opset_version),
    }
    if metadata:
        all_metadata.update(metadata)

    for key, value in all_metadata.items():
        entry = onnx_model.metadata_props.add()
        entry.key = key
        entry.value = value

    onnx.save(onnx_model, str(output_path))
    logger.info("ONNX metadata written: %s", list(all_metadata.keys()))

    # Verify export with a test inference ------------------------------
    _require_onnxruntime()
    import onnxruntime as ort

    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    test_out = session.run(None, {"input": dummy_input.cpu().numpy()})
    logger.info(
        "Export verification passed -- output shape: %s", test_out[0].shape,
    )

    # Build info -------------------------------------------------------
    input_shapes: Dict[str, List[int]] = {}
    for inp in session.get_inputs():
        input_shapes[inp.name] = [d if isinstance(d, int) else -1 for d in inp.shape]

    output_shapes: Dict[str, List[int]] = {}
    for out in session.get_outputs():
        output_shapes[out.name] = [d if isinstance(d, int) else -1 for d in out.shape]

    return OnnxModelInfo(
        path=str(output_path),
        input_names=input_names,
        output_names=output_names,
        input_shapes=input_shapes,
        output_shapes=output_shapes,
        model_type=model_type,
        metadata=all_metadata,
        opset_version=opset_version,
    )


# ======================================================================
# Inference engine
# ======================================================================

class OnnxInferenceEngine:
    """Low-level ONNX inference wrapper around ``onnxruntime.InferenceSession``.

    Automatically selects the best available execution provider (CUDA when
    present, otherwise CPU).

    Parameters
    ----------
    model_path:
        Path to an ``.onnx`` file.
    device:
        ``"cpu"`` or ``"cuda"``.  When ``"cuda"`` is requested but the CUDA
        provider is not available, a warning is emitted and the engine falls
        back to CPU.
    """

    def __init__(self, model_path: Union[str, Path], device: str = "cpu") -> None:
        _require_onnxruntime()
        import onnxruntime as ort

        self._model_path = str(model_path)
        if not Path(self._model_path).exists():
            raise FileNotFoundError(f"ONNX model not found: {self._model_path}")

        # Provider selection -----------------------------------------------
        # 支援 CUDA (NVIDIA GPU)、CoreML (Apple Silicon)、CPU
        available = ort.get_available_providers()
        if device == "cuda" and "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif device == "mps" and "CoreMLExecutionProvider" in available:
            providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
            logger.info("使用 CoreMLExecutionProvider (Apple Silicon)")
        else:
            if device == "cuda":
                logger.warning(
                    "CUDAExecutionProvider 不可用，回退至 CPUExecutionProvider。"
                )
            elif device == "mps":
                logger.warning(
                    "CoreMLExecutionProvider 不可用，回退至 CPUExecutionProvider。"
                )
            providers = ["CPUExecutionProvider"]

        self._session = ort.InferenceSession(self._model_path, providers=providers)
        active_provider = self._session.get_providers()[0]
        logger.info("ONNX session loaded: %s  (provider=%s)", self._model_path, active_provider)

        # Parse I/O metadata -----------------------------------------------
        self._input_names = [inp.name for inp in self._session.get_inputs()]
        self._output_names = [out.name for out in self._session.get_outputs()]

        self._input_shapes: Dict[str, List[int]] = {}
        for inp in self._session.get_inputs():
            self._input_shapes[inp.name] = [
                d if isinstance(d, int) else -1 for d in inp.shape
            ]

        self._output_shapes: Dict[str, List[int]] = {}
        for out in self._session.get_outputs():
            self._output_shapes[out.name] = [
                d if isinstance(d, int) else -1 for d in out.shape
            ]

        # Parse model metadata ---------------------------------------------
        self._metadata: Dict[str, str] = {}
        model_meta = self._session.get_modelmeta()
        if model_meta.custom_metadata_map:
            self._metadata = dict(model_meta.custom_metadata_map)

        self._model_type = self._metadata.get("model_type", "custom")
        self._opset_version = int(self._metadata.get("opset_version", "0"))

        logger.info(
            "Model info -- type=%s, inputs=%s, outputs=%s, metadata_keys=%s",
            self._model_type,
            self._input_names,
            self._output_names,
            list(self._metadata.keys()),
        )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_info(self) -> OnnxModelInfo:
        """Return an :class:`OnnxModelInfo` describing the loaded model."""
        return OnnxModelInfo(
            path=self._model_path,
            input_names=list(self._input_names),
            output_names=list(self._output_names),
            input_shapes=dict(self._input_shapes),
            output_shapes=dict(self._output_shapes),
            model_type=self._model_type,
            metadata=dict(self._metadata),
            opset_version=self._opset_version,
        )

    @log_operation(logger)
    def run(self, input_array: np.ndarray) -> np.ndarray:
        """Run inference on a single numpy array.

        If the array is missing the batch dimension it is added automatically.

        Returns
        -------
        np.ndarray
            The first output tensor from the ONNX model.
        """
        arr = input_array
        expected_ndim = len(self._input_shapes[self._input_names[0]])

        if arr.ndim == expected_ndim - 1:
            arr = np.expand_dims(arr, axis=0)
        elif arr.ndim != expected_ndim:
            raise ValueError(
                f"Input has {arr.ndim} dimensions but model expects "
                f"{expected_ndim} (or {expected_ndim - 1} without batch dim)."
            )

        arr = np.asarray(arr, dtype=np.float32)
        feeds = {self._input_names[0]: arr}
        outputs = self._session.run(self._output_names, feeds)
        return outputs[0]

    @log_operation(logger)
    def run_batch(self, images: List[np.ndarray], batch_size: int = 16) -> List[np.ndarray]:
        """Run inference on a list of arrays, batching for efficiency.

        Each element in *images* should have shape ``(C, H, W)`` (no batch
        dim).  Returns a list of output arrays with the same ordering.
        """
        results: List[np.ndarray] = []

        for start in range(0, len(images), batch_size):
            chunk = images[start : start + batch_size]
            batch = np.stack(chunk, axis=0).astype(np.float32)
            feeds = {self._input_names[0]: batch}
            outputs = self._session.run(self._output_names, feeds)
            for i in range(outputs[0].shape[0]):
                results.append(outputs[0][i])

        return results

    def warmup(self, n: int = 3) -> None:
        """Run *n* dummy inferences to warm up the execution engine."""
        shape = list(self._input_shapes[self._input_names[0]])
        # Replace dynamic dims (-1) with sensible defaults
        shape = [max(d, 1) for d in shape]
        dummy = np.random.randn(*shape).astype(np.float32)

        for i in range(n):
            self._session.run(self._output_names, {self._input_names[0]: dummy})
        logger.info("Warmup complete (%d iterations).", n)


# ======================================================================
# High-level anomaly detector (ONNX-backed)
# ======================================================================

class OnnxAnomalyDetector:
    """Drop-in replacement for :class:`InferencePipeline` that runs via ONNX.

    Provides the same ``inspect_single`` / ``inspect_batch`` interface so
    downstream code (GUI, REST API) can switch backends transparently.

    Parameters
    ----------
    model_path:
        Path to an ONNX model exported with :func:`export_to_onnx`.
    image_size:
        Spatial resolution the model was trained on.
    grayscale:
        Whether the model expects single-channel input.
    device:
        ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        image_size: int = 256,
        grayscale: bool = False,
        device: str = "cpu",
    ) -> None:
        from dl_anomaly.core.anomaly_scorer import AnomalyScorer
        from dl_anomaly.core.preprocessor import ImagePreprocessor

        self._engine = OnnxInferenceEngine(model_path, device=device)
        self._preprocessor = ImagePreprocessor(image_size, grayscale)
        self._scorer = AnomalyScorer(device=device)
        self._image_size = image_size
        self._grayscale = grayscale

        # Try to read ssim_weight from model metadata; default to 0.5
        meta = self._engine.get_info().metadata
        self._ssim_weight = float(meta.get("ssim_weight", "0.5"))

        logger.info(
            "OnnxAnomalyDetector ready -- model=%s, size=%d, grayscale=%s",
            model_path,
            image_size,
            grayscale,
        )

    # ------------------------------------------------------------------
    # Threshold
    # ------------------------------------------------------------------

    def set_threshold(self, value: float) -> None:
        """Manually set the anomaly threshold."""
        self._scorer.threshold = value
        logger.info("Anomaly threshold set to %.6f (manual).", value)

    @log_operation(logger)
    def fit_threshold(
        self,
        good_image_dir: Union[str, Path],
        percentile: float = 95.0,
    ) -> float:
        """Compute the anomaly threshold from a directory of known-good images.

        Processes every image, collects per-image anomaly scores, and sets
        the threshold at the given *percentile*.
        """
        good_dir = Path(good_image_dir)
        extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        paths = sorted(
            p for p in good_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in extensions
        )
        if not paths:
            raise FileNotFoundError(f"No images found in {good_dir}")

        scores: List[float] = []
        for p in paths:
            try:
                tensor = self._preprocessor.load_and_preprocess(p)
                inp = tensor.unsqueeze(0).numpy()
                recon = self._engine.run(inp)
                recon_tensor = torch.from_numpy(recon.squeeze(0))

                orig_np = self._preprocessor.inverse_normalize(tensor)
                recon_np = self._preprocessor.inverse_normalize(recon_tensor)

                error_map = self._scorer.compute_combined_error(
                    orig_np, recon_np, self._ssim_weight,
                )
                scores.append(self._scorer.compute_image_score(error_map))
            except Exception:
                logger.exception("Skipping %s during threshold fitting.", p)

        if not scores:
            raise RuntimeError("All images failed during threshold fitting.")

        threshold = self._scorer.fit_threshold(scores, percentile)
        logger.info(
            "Threshold fitted on %d images: %.6f (percentile=%.1f)",
            len(scores),
            threshold,
            percentile,
        )
        return threshold

    # ------------------------------------------------------------------
    # Single-image inspection
    # ------------------------------------------------------------------

    @log_operation(logger)
    def inspect_single(self, image_path: Union[str, Path]) -> Any:
        """Run the full inspection pipeline on one image.

        Returns
        -------
        InspectionResult
            Same dataclass used by :class:`InferencePipeline`.
        """
        from dl_anomaly.pipeline.inference import InferencePipeline, InspectionResult

        image_path = Path(image_path)

        # Preprocess
        tensor = self._preprocessor.load_and_preprocess(image_path)
        inp = tensor.unsqueeze(0).numpy()

        # Inference
        recon = self._engine.run(inp)
        recon_tensor = torch.from_numpy(recon.squeeze(0))

        # Convert to uint8 numpy
        original_np = self._preprocessor.inverse_normalize(tensor)
        recon_np = self._preprocessor.inverse_normalize(recon_tensor)

        # Error map and scoring
        error_map = self._scorer.compute_combined_error(
            original_np, recon_np, self._ssim_weight,
        )
        smoothed = self._scorer.create_anomaly_map(error_map, gaussian_sigma=4.0)
        score = self._scorer.compute_image_score(error_map)

        is_defective = False
        if self._scorer.threshold is not None:
            is_defective = self._scorer.classify(score)

        mask = InferencePipeline._create_defect_mask(smoothed)
        regions = InferencePipeline._extract_defect_regions(mask)

        return InspectionResult(
            original=original_np,
            reconstruction=recon_np,
            error_map=smoothed,
            defect_mask=mask,
            anomaly_score=score,
            is_defective=is_defective,
            defect_regions=regions,
        )

    # ------------------------------------------------------------------
    # Batch inspection
    # ------------------------------------------------------------------

    def inspect_batch(
        self,
        image_dir: Union[str, Path],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[Any]:
        """Inspect all images in *image_dir*.

        Parameters
        ----------
        progress_callback:
            Called with ``(current_index, total_count)`` after each image.
        """
        image_dir = Path(image_dir)
        extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        paths = sorted(
            p for p in image_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in extensions
        )

        if not paths:
            logger.warning("No images found in %s", image_dir)
            return []

        results: List[Any] = []
        for idx, p in enumerate(paths):
            try:
                result = self.inspect_single(p)
                results.append(result)
            except Exception:
                logger.exception("Failed to inspect %s", p)

            if progress_callback is not None:
                progress_callback(idx + 1, len(paths))

        return results
