"""Multi-threaded processing pipeline for throughput optimization.

Implements a 3-stage pipeline (preprocess -> inference -> postprocess) with
concurrent execution.  Each stage can use multiple worker threads and stages
are connected via bounded queues so that back-pressure is applied
automatically when a slower stage cannot keep up.

Typical usage::

    pipeline = create_inspection_pipeline(
        model_type="autoencoder",
        checkpoint_path="model.onnx",
        num_preprocess_workers=2,
        num_postprocess_workers=2,
    )
    results = pipeline.process_batch(image_paths)
"""

from __future__ import annotations

import logging
import queue
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
)

from shared.op_logger import log_operation

logger = logging.getLogger(__name__)

# Sentinel value pushed into queues to signal shutdown.
_SENTINEL = object()


# ======================================================================
# Data classes
# ======================================================================


@dataclass
class PipelineStage:
    """Description of a single processing stage.

    Parameters
    ----------
    name:
        Human-readable stage name (used in logs and timing reports).
    func:
        Callable ``(input) -> output``.  Must be thread-safe when
        *num_workers* > 1.
    num_workers:
        Number of parallel worker threads for this stage.
    timeout:
        Maximum seconds a single invocation of *func* is allowed to take
        before the item is considered failed.
    """

    name: str
    func: Callable
    num_workers: int = 1
    timeout: float = 30.0


@dataclass
class PipelineResult:
    """Outcome of processing a single item through the full pipeline.

    Attributes
    ----------
    input_path:
        Identifier / path of the input that was processed.
    output:
        Result produced by the final stage (``None`` on failure).
    timings:
        Mapping of ``stage_name -> elapsed_seconds`` for each stage.
    total_time:
        Wall-clock time from pipeline entry to completion.
    success:
        Whether all stages completed without error.
    error:
        Error message if *success* is ``False``.
    """

    input_path: str
    output: Any
    timings: Dict[str, float]
    total_time: float
    success: bool
    error: Optional[str] = None


@dataclass
class PipelineStats:
    """Aggregate statistics for a pipeline run.

    Attributes
    ----------
    total_processed:
        Number of items that completed all stages (success or failure).
    total_failed:
        Number of items that failed at any stage.
    avg_time:
        Average wall-clock time per item in seconds.
    throughput:
        Items processed per second.
    stage_timings:
        Average time per stage in seconds.
    start_time:
        ``time.monotonic()`` value when processing began.
    elapsed:
        Total elapsed seconds from start to finish.
    """

    total_processed: int
    total_failed: int
    avg_time: float
    throughput: float
    stage_timings: Dict[str, float]
    start_time: float
    elapsed: float


# ======================================================================
# ThreadSafeAccumulator
# ======================================================================


class ThreadSafeAccumulator:
    """Thread-safe result accumulator with ordering.

    Workers may finish in any order; this class stores results keyed by
    their original index so that the caller can retrieve them in the
    correct input order.
    """

    def __init__(self) -> None:
        self._results: Dict[int, Any] = {}
        self._lock = threading.Lock()

    def add(self, index: int, result: Any) -> None:
        """Store *result* at position *index*."""
        with self._lock:
            self._results[index] = result

    def get_ordered(self) -> List[Any]:
        """Return results sorted by index."""
        with self._lock:
            return [self._results[k] for k in sorted(self._results)]

    def count(self) -> int:
        """Return the number of results stored so far."""
        with self._lock:
            return len(self._results)


# ======================================================================
# ParallelPipeline
# ======================================================================


class ParallelPipeline:
    """Multi-stage pipeline with per-stage thread pools.

    Parameters
    ----------
    stages:
        Ordered list of :class:`PipelineStage` objects.
    max_queue_size:
        Maximum number of items buffered between consecutive stages.
    """

    def __init__(
        self,
        stages: List[PipelineStage],
        max_queue_size: int = 10,
    ) -> None:
        if not stages:
            raise ValueError("At least one PipelineStage is required.")

        self._stages = stages
        self._max_queue_size = max_queue_size

        # Stop event shared by all workers.
        self._stop_event = threading.Event()

        # Runtime statistics (protected by lock).
        self._stats_lock = threading.Lock()
        self._total_processed: int = 0
        self._total_failed: int = 0
        self._item_times: List[float] = []
        self._stage_times: Dict[str, List[float]] = {
            s.name: [] for s in stages
        }
        self._start_time: float = 0.0

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    @log_operation(logger)
    def process_batch(
        self,
        inputs: List[Any],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[PipelineResult]:
        """Process a list of inputs through all pipeline stages.

        Parameters
        ----------
        inputs:
            Items to process (e.g. file paths).
        progress_callback:
            Optional ``(current, total)`` callback invoked each time an
            item finishes all stages.

        Returns
        -------
        List[PipelineResult]
            One result per input, in the same order as *inputs*.
        """
        if not inputs:
            return []

        self._reset_stats()
        total = len(inputs)
        accumulator = ThreadSafeAccumulator()
        completed = threading.Event()
        completed_count = _AtomicCounter()

        # Build inter-stage queues.
        queues = self._build_queues()

        # Start stage workers.
        executors, futures = self._start_workers(queues)

        # Feed inputs into the first queue.
        for idx, item in enumerate(inputs):
            if self._stop_event.is_set():
                break
            queues[0].put(_WorkItem(index=idx, payload=item, timings={}, t0=time.monotonic()))

        # Push sentinels into the first queue (one per worker of stage 0).
        for _ in range(self._stages[0].num_workers):
            queues[0].put(_SENTINEL)

        # Collect results from the last queue.
        last_q = queues[-1]
        while True:
            try:
                item = last_q.get(timeout=0.5)
            except queue.Empty:
                if self._stop_event.is_set():
                    break
                continue

            if item is _SENTINEL:
                # We expect one sentinel per worker of the last stage.
                if completed_count.increment() >= self._stages[-1].num_workers:
                    break
                continue

            result = self._work_item_to_result(item)
            accumulator.add(item.index, result)
            self._record_result(result)

            if progress_callback is not None:
                try:
                    progress_callback(accumulator.count(), total)
                except Exception:
                    logger.warning("progress_callback raised an exception", exc_info=True)

        # Shut down executors.
        self._shutdown_executors(executors)

        results = accumulator.get_ordered()

        # Fill in results for items that never made it through.
        result_indices = {r.input_path for r in results}
        for idx, inp in enumerate(inputs):
            inp_key = str(inp)
            if inp_key not in result_indices and accumulator.count() < total:
                results.append(
                    PipelineResult(
                        input_path=inp_key,
                        output=None,
                        timings={},
                        total_time=0.0,
                        success=False,
                        error="Item was not processed (pipeline stopped).",
                    )
                )

        return results

    def process_stream(
        self,
        input_iterator: Iterable[Any],
        on_result: Optional[Callable[[PipelineResult], None]] = None,
    ) -> PipelineStats:
        """Process items as they arrive from *input_iterator*.

        Parameters
        ----------
        input_iterator:
            Iterable that yields items to process.
        on_result:
            Optional callback invoked with each :class:`PipelineResult`.

        Returns
        -------
        PipelineStats
            Aggregate statistics for the run.
        """
        self._reset_stats()

        queues = self._build_queues()
        executors, _ = self._start_workers(queues)

        completed_count = _AtomicCounter()
        items_fed = _AtomicCounter()

        # Feed items in a separate thread so we can read results
        # concurrently.
        def _feeder() -> None:
            for idx, item in enumerate(input_iterator):
                if self._stop_event.is_set():
                    break
                queues[0].put(
                    _WorkItem(index=idx, payload=item, timings={}, t0=time.monotonic())
                )
                items_fed.increment()
            # Signal end-of-input.
            for _ in range(self._stages[0].num_workers):
                queues[0].put(_SENTINEL)

        feeder_thread = threading.Thread(target=_feeder, daemon=True)
        feeder_thread.start()

        # Collect results from the last queue.
        last_q = queues[-1]
        sentinel_count = _AtomicCounter()
        while True:
            try:
                item = last_q.get(timeout=0.5)
            except queue.Empty:
                if self._stop_event.is_set():
                    break
                continue

            if item is _SENTINEL:
                if sentinel_count.increment() >= self._stages[-1].num_workers:
                    break
                continue

            result = self._work_item_to_result(item)
            self._record_result(result)

            if on_result is not None:
                try:
                    on_result(result)
                except Exception:
                    logger.warning("on_result callback raised an exception", exc_info=True)

        feeder_thread.join(timeout=5.0)
        self._shutdown_executors(executors)

        return self.get_stats()

    def get_stats(self) -> PipelineStats:
        """Return current aggregate processing statistics."""
        with self._stats_lock:
            elapsed = time.monotonic() - self._start_time if self._start_time else 0.0
            total = self._total_processed
            avg = statistics.mean(self._item_times) if self._item_times else 0.0
            throughput = total / elapsed if elapsed > 0 else 0.0
            stage_avgs: Dict[str, float] = {}
            for name, times in self._stage_times.items():
                stage_avgs[name] = statistics.mean(times) if times else 0.0

            return PipelineStats(
                total_processed=total,
                total_failed=self._total_failed,
                avg_time=avg,
                throughput=throughput,
                stage_timings=stage_avgs,
                start_time=self._start_time,
                elapsed=elapsed,
            )

    def stop(self) -> None:
        """Signal all workers to stop and drain queues."""
        logger.info("Pipeline stop requested.")
        self._stop_event.set()

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _reset_stats(self) -> None:
        self._stop_event.clear()
        with self._stats_lock:
            self._total_processed = 0
            self._total_failed = 0
            self._item_times = []
            self._stage_times = {s.name: [] for s in self._stages}
            self._start_time = time.monotonic()

    def _build_queues(self) -> List[queue.Queue]:
        """Create N+1 queues for N stages."""
        return [queue.Queue(maxsize=self._max_queue_size) for _ in range(len(self._stages) + 1)]

    def _start_workers(
        self,
        queues: List[queue.Queue],
    ) -> Tuple[List[ThreadPoolExecutor], List[Any]]:
        executors: List[ThreadPoolExecutor] = []
        futures: List[Any] = []

        for stage_idx, stage in enumerate(self._stages):
            in_q = queues[stage_idx]
            out_q = queues[stage_idx + 1]
            executor = ThreadPoolExecutor(
                max_workers=stage.num_workers,
                thread_name_prefix=f"pipeline-{stage.name}",
            )
            executors.append(executor)
            for _ in range(stage.num_workers):
                fut = executor.submit(
                    self._stage_worker,
                    stage=stage,
                    in_q=in_q,
                    out_q=out_q,
                    is_last=(stage_idx == len(self._stages) - 1),
                    next_stage_workers=(
                        self._stages[stage_idx + 1].num_workers
                        if stage_idx + 1 < len(self._stages)
                        else 0
                    ),
                )
                futures.append(fut)

        return executors, futures

    def _stage_worker(
        self,
        stage: PipelineStage,
        in_q: queue.Queue,
        out_q: queue.Queue,
        is_last: bool,
        next_stage_workers: int,
    ) -> None:
        """Worker loop for a single pipeline stage."""
        sentinel_seen = False
        while not self._stop_event.is_set():
            try:
                item = in_q.get(timeout=0.5)
            except queue.Empty:
                continue

            if item is _SENTINEL:
                sentinel_seen = True
                # Forward sentinel to the next stage.  For multi-worker
                # stages, the *first* worker that sees a sentinel
                # propagates it.  We always forward exactly one sentinel
                # per worker so that the downstream stage receives the
                # correct number.
                out_q.put(_SENTINEL)
                break

            if not isinstance(item, _WorkItem):
                logger.error("Unexpected item type in queue: %s", type(item))
                continue

            # Skip items that already failed in a previous stage.
            if item.error is not None:
                out_q.put(item)
                continue

            t_stage_start = time.monotonic()
            try:
                result_payload = stage.func(item.payload)
                elapsed = time.monotonic() - t_stage_start

                if elapsed > stage.timeout:
                    logger.warning(
                        "Stage '%s' exceeded timeout (%.1fs > %.1fs) for item %s",
                        stage.name,
                        elapsed,
                        stage.timeout,
                        item.index,
                    )

                item.timings[stage.name] = elapsed
                item.payload = result_payload

            except Exception as exc:
                elapsed = time.monotonic() - t_stage_start
                item.timings[stage.name] = elapsed
                item.error = f"Stage '{stage.name}' failed: {exc}"
                item.payload = None
                logger.error(
                    "Stage '%s' error on item %d: %s",
                    stage.name,
                    item.index,
                    exc,
                    exc_info=True,
                )

            out_q.put(item)

        # If we exit the loop due to stop_event without seeing a
        # sentinel, still push one so downstream workers can terminate.
        if not sentinel_seen:
            out_q.put(_SENTINEL)

    @staticmethod
    def _work_item_to_result(item: _WorkItem) -> PipelineResult:
        total_time = time.monotonic() - item.t0
        success = item.error is None
        return PipelineResult(
            input_path=str(item.original_input),
            output=item.payload if success else None,
            timings=dict(item.timings),
            total_time=total_time,
            success=success,
            error=item.error,
        )

    def _record_result(self, result: PipelineResult) -> None:
        with self._stats_lock:
            self._total_processed += 1
            if not result.success:
                self._total_failed += 1
            self._item_times.append(result.total_time)
            for stage_name, t in result.timings.items():
                if stage_name in self._stage_times:
                    self._stage_times[stage_name].append(t)

    @staticmethod
    def _shutdown_executors(executors: List[ThreadPoolExecutor]) -> None:
        for ex in executors:
            ex.shutdown(wait=True)


# ======================================================================
# Internal work item
# ======================================================================


@dataclass
class _WorkItem:
    """Internal wrapper that travels through the pipeline queues."""

    index: int
    payload: Any
    timings: Dict[str, float]
    t0: float
    error: Optional[str] = None
    original_input: Any = None

    def __post_init__(self) -> None:
        if self.original_input is None:
            self.original_input = self.payload


# ======================================================================
# Atomic counter helper
# ======================================================================


class _AtomicCounter:
    """Simple thread-safe counter."""

    def __init__(self, initial: int = 0) -> None:
        self._value = initial
        self._lock = threading.Lock()

    def increment(self) -> int:
        """Increment and return the new value."""
        with self._lock:
            self._value += 1
            return self._value

    @property
    def value(self) -> int:
        with self._lock:
            return self._value


# ======================================================================
# Pre-built pipeline factories
# ======================================================================


def create_inspection_pipeline(
    model_type: str = "autoencoder",
    checkpoint_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    num_preprocess_workers: int = 2,
    num_postprocess_workers: int = 2,
) -> ParallelPipeline:
    """Create a 3-stage inspection pipeline.

    Stages
    ------
    1. **preprocess** -- load image, resize, normalise (N workers).
    2. **inference** -- run model forward pass (1 worker, sequential for
       GPU access).
    3. **postprocess** -- error map computation, thresholding, region
       extraction (N workers).

    Parameters
    ----------
    model_type:
        Model architecture identifier (``"autoencoder"`` or
        ``"patchcore"``).
    checkpoint_path:
        Path to model weights.
    config:
        Optional configuration dict forwarded to the model loader.
    num_preprocess_workers:
        Thread count for the preprocess stage.
    num_postprocess_workers:
        Thread count for the postprocess stage.
    """
    import cv2
    import numpy as np

    cfg = config or {}
    image_size: int = cfg.get("image_size", 256)
    threshold: float = cfg.get("threshold", 0.5)
    mean = np.array(cfg.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32)
    std = np.array(cfg.get("std", [0.229, 0.224, 0.225]), dtype=np.float32)

    # ------------------------------------------------------------------
    # Stage 1: preprocess
    # ------------------------------------------------------------------
    def preprocess(input_path: str) -> Dict[str, Any]:
        img = cv2.imread(input_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {input_path}")
        original_shape = img.shape[:2]
        img_resized = cv2.resize(img, (image_size, image_size))
        img_norm = (img_resized.astype(np.float32) / 255.0 - mean) / std
        # HWC -> CHW
        img_chw = np.transpose(img_norm, (2, 0, 1))
        return {
            "input_path": input_path,
            "tensor": img_chw[np.newaxis, ...],  # add batch dim
            "original_shape": original_shape,
            "original_image": img,
        }

    # ------------------------------------------------------------------
    # Stage 2: inference
    # ------------------------------------------------------------------
    _model = _load_model_for_pipeline(model_type, checkpoint_path, cfg)

    def inference(data: Dict[str, Any]) -> Dict[str, Any]:
        tensor = data["tensor"]
        output = _model(tensor)
        data["model_output"] = output
        return data

    # ------------------------------------------------------------------
    # Stage 3: postprocess
    # ------------------------------------------------------------------
    def postprocess(data: Dict[str, Any]) -> Dict[str, Any]:
        model_output = data["model_output"]
        original_shape = data["original_shape"]

        # Compute error map (absolute difference for autoencoders).
        if isinstance(model_output, np.ndarray) and model_output.ndim == 4:
            # Reconstruct CHW -> HWC
            recon = np.transpose(model_output[0], (1, 2, 0))
            inp = np.transpose(data["tensor"][0], (1, 2, 0))
            error_map = np.mean(np.abs(inp - recon), axis=-1)
        elif isinstance(model_output, dict) and "anomaly_map" in model_output:
            error_map = model_output["anomaly_map"]
        else:
            error_map = np.zeros((image_size, image_size), dtype=np.float32)

        # Resize error map to original shape.
        error_map_resized = cv2.resize(
            error_map, (original_shape[1], original_shape[0])
        )

        # Threshold to binary mask.
        binary_mask = (error_map_resized > threshold).astype(np.uint8) * 255

        # Find defect regions via contours.
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        regions = [cv2.boundingRect(c) for c in contours]

        data["error_map"] = error_map_resized
        data["binary_mask"] = binary_mask
        data["regions"] = regions
        data["is_defective"] = len(regions) > 0
        return data

    stages = [
        PipelineStage(name="preprocess", func=preprocess, num_workers=num_preprocess_workers),
        PipelineStage(name="inference", func=inference, num_workers=1),
        PipelineStage(name="postprocess", func=postprocess, num_workers=num_postprocess_workers),
    ]

    logger.info(
        "Created inspection pipeline: model_type=%s, preprocess_workers=%d, postprocess_workers=%d",
        model_type,
        num_preprocess_workers,
        num_postprocess_workers,
    )

    return ParallelPipeline(stages=stages)


def create_batch_processor(
    process_func: Callable,
    num_workers: int = 4,
    timeout: float = 60.0,
) -> ParallelPipeline:
    """Create a single-stage parallel processing pipeline.

    This is a convenience wrapper for simple parallel map operations
    such as batch filtering, transformation, or format conversion.

    Parameters
    ----------
    process_func:
        Function to apply to each input item.
    num_workers:
        Number of parallel workers.
    timeout:
        Maximum seconds per item.
    """
    stage = PipelineStage(
        name="process",
        func=process_func,
        num_workers=num_workers,
        timeout=timeout,
    )
    return ParallelPipeline(stages=[stage])


# ======================================================================
# Utility functions
# ======================================================================


def benchmark_pipeline(
    pipeline: ParallelPipeline,
    test_inputs: List[Any],
    warmup: int = 3,
) -> Dict[str, Any]:
    """Benchmark a pipeline's throughput.

    Parameters
    ----------
    pipeline:
        The pipeline to benchmark.
    test_inputs:
        Sample inputs to process.
    warmup:
        Number of warmup iterations before measurement.

    Returns
    -------
    dict
        ``{"avg_time", "throughput", "stage_timings", "total_items",
        "warmup_items", "elapsed"}``
    """
    # Warmup
    warmup_items = test_inputs[:warmup] if len(test_inputs) >= warmup else test_inputs
    if warmup_items:
        logger.info("Running %d warmup iterations ...", len(warmup_items))
        pipeline.process_batch(warmup_items)

    # Measure
    logger.info("Benchmarking with %d items ...", len(test_inputs))
    t0 = time.monotonic()
    results = pipeline.process_batch(test_inputs)
    elapsed = time.monotonic() - t0

    successful = [r for r in results if r.success]
    stage_names = list(results[0].timings.keys()) if results else []
    stage_avgs: Dict[str, float] = {}
    for name in stage_names:
        times = [r.timings.get(name, 0.0) for r in successful]
        stage_avgs[name] = statistics.mean(times) if times else 0.0

    item_times = [r.total_time for r in successful]
    avg_time = statistics.mean(item_times) if item_times else 0.0
    throughput = len(successful) / elapsed if elapsed > 0 else 0.0

    summary = {
        "avg_time": avg_time,
        "throughput": throughput,
        "stage_timings": stage_avgs,
        "total_items": len(test_inputs),
        "successful_items": len(successful),
        "warmup_items": len(warmup_items),
        "elapsed": elapsed,
    }
    logger.info("Benchmark result: %.2f items/sec, avg %.3fs per item", throughput, avg_time)
    return summary


def estimate_optimal_workers(
    stage_func: Callable,
    sample_input: Any,
    max_workers: int = 8,
) -> int:
    """Auto-tune the number of workers by benchmarking with increasing parallelism.

    Runs *stage_func* with 1 .. *max_workers* threads and returns the
    worker count that achieved the best throughput.

    Parameters
    ----------
    stage_func:
        The function to benchmark.
    sample_input:
        A representative input for *stage_func*.
    max_workers:
        Upper bound on workers to test.

    Returns
    -------
    int
        Optimal number of workers.
    """
    best_workers = 1
    best_throughput = 0.0
    repetitions = 10

    for n_workers in range(1, max_workers + 1):
        pipeline = create_batch_processor(
            process_func=stage_func,
            num_workers=n_workers,
        )
        inputs = [sample_input] * repetitions
        t0 = time.monotonic()
        pipeline.process_batch(inputs)
        elapsed = time.monotonic() - t0
        throughput = repetitions / elapsed if elapsed > 0 else 0.0

        logger.info(
            "Workers=%d  throughput=%.2f items/sec  elapsed=%.3fs",
            n_workers,
            throughput,
            elapsed,
        )

        if throughput > best_throughput:
            best_throughput = throughput
            best_workers = n_workers

    logger.info(
        "Optimal worker count: %d (%.2f items/sec)", best_workers, best_throughput
    )
    return best_workers


# ======================================================================
# Internal helpers
# ======================================================================


def _load_model_for_pipeline(
    model_type: str,
    checkpoint_path: Optional[str],
    config: Dict[str, Any],
) -> Callable:
    """Load or create a model callable for the inference stage.

    Returns a callable that accepts a numpy array (NCHW, float32) and
    returns the model output.  When no checkpoint is provided a dummy
    identity function is returned so that the pipeline can still be
    tested end-to-end.
    """
    import numpy as np

    if checkpoint_path is None:
        logger.warning(
            "No checkpoint_path provided -- using identity model (pass-through)."
        )
        return lambda x: x

    # Try ONNX runtime first.
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(checkpoint_path)
        input_name = session.get_inputs()[0].name

        def _onnx_infer(tensor: np.ndarray) -> np.ndarray:
            return session.run(None, {input_name: tensor.astype(np.float32)})[0]

        logger.info("Loaded ONNX model from %s", checkpoint_path)
        return _onnx_infer

    except ImportError:
        logger.warning(
            "onnxruntime is not installed. Install it with: "
            "pip install onnxruntime  (or onnxruntime-gpu for CUDA support). "
            "Falling back to PyTorch."
        )
    except Exception as exc:
        logger.debug("ONNX loading failed (%s), trying PyTorch ...", exc)

    # Fall back to PyTorch.
    try:
        import torch

        if model_type == "autoencoder":
            from dl_anomaly.core.autoencoder import AnomalyAutoencoder  # noqa: E501

            model = AnomalyAutoencoder.load(checkpoint_path, config)
        elif model_type == "patchcore":
            from shared.core.patchcore import PatchCoreModel

            model = PatchCoreModel.load(checkpoint_path, config)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        @torch.no_grad()
        def _torch_infer(tensor: np.ndarray) -> Any:
            t = torch.from_numpy(tensor).float().to(device)
            output = model(t)
            if isinstance(output, torch.Tensor):
                return output.cpu().numpy()
            return output

        logger.info(
            "Loaded PyTorch %s model from %s (device=%s)",
            model_type,
            checkpoint_path,
            device,
        )
        return _torch_infer

    except Exception as exc:
        logger.error("Failed to load model: %s", exc, exc_info=True)
        raise RuntimeError(
            f"Cannot load model from '{checkpoint_path}': {exc}"
        ) from exc
