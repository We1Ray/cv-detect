"""Operator-level parallel execution for image processing.

Provides fine-grained parallelism at the operator level -- processing
multiple images simultaneously or splitting a single large image into
tiles for parallel computation.  Complements the coarser-grained
:mod:`shared.core.parallel_pipeline` which parallelises across pipeline
stages.

Key components:
    - :class:`ParallelExecutor` -- configurable parallel map / pipeline
    - :class:`ROIParallel` -- split a single image by ROIs, process, merge
    - Convenience functions: :func:`par_threshold`, :func:`par_filter`

Usage::

    executor = ParallelExecutor(max_workers=8, pool_type="thread")
    results = executor.map(my_func, images)

    # Tiled processing of a large image
    output = executor.tiled_apply(
        func=heavy_filter,
        image=large_image,
        tile_size=(512, 512),
        overlap=32,
    )
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from concurrent.futures import (
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Default to number of CPU cores (logical).
_DEFAULT_WORKERS = min(os.cpu_count() or 4, 16)


# =========================================================================
# Data classes
# =========================================================================
@dataclass
class TileSpec:
    """Specification of a single tile within a larger image.

    Attributes
    ----------
    y_start, y_end : int
        Row slice (inclusive start, exclusive end).
    x_start, x_end : int
        Column slice (inclusive start, exclusive end).
    y_out_start, y_out_end : int
        Row range in the output where the non-overlapping portion maps to.
    x_out_start, x_out_end : int
        Column range in the output for the non-overlapping portion.
    """

    y_start: int
    y_end: int
    x_start: int
    x_end: int
    y_out_start: int
    y_out_end: int
    x_out_start: int
    x_out_end: int


@dataclass
class ParallelStats:
    """Statistics for a parallel execution run.

    Attributes
    ----------
    total_items : int
        Number of items processed.
    elapsed_s : float
        Total wall-clock time in seconds.
    throughput : float
        Items per second.
    errors : int
        Number of items that raised exceptions.
    """

    total_items: int = 0
    elapsed_s: float = 0.0
    throughput: float = 0.0
    errors: int = 0


# =========================================================================
# ParallelExecutor
# =========================================================================
class ParallelExecutor:
    """Configurable parallel executor for image processing operators.

    Parameters
    ----------
    max_workers : int, optional
        Maximum number of parallel workers.  Defaults to the number of
        CPU cores (capped at 16).
    pool_type : str
        ``"thread"`` for I/O-bound work (:class:`ThreadPoolExecutor`),
        ``"process"`` for CPU-bound work (:class:`ProcessPoolExecutor`).
    """

    def __init__(
        self,
        max_workers: int = _DEFAULT_WORKERS,
        pool_type: str = "thread",
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1.")
        if pool_type not in ("thread", "process"):
            raise ValueError("pool_type must be 'thread' or 'process'.")

        self._max_workers = max_workers
        self._pool_type = pool_type
        self._stats = ParallelStats()
        # Shared pool for par_start / par_join operations.
        pool_cls = ThreadPoolExecutor if pool_type == "thread" else ProcessPoolExecutor
        self._pool = pool_cls(max_workers=max_workers)

    @property
    def max_workers(self) -> int:
        """Return the configured worker count."""
        return self._max_workers

    @property
    def last_stats(self) -> ParallelStats:
        """Return statistics from the most recent :meth:`map` call."""
        return self._stats

    # ------------------------------------------------------------------
    # Core parallel operations
    # ------------------------------------------------------------------

    def map(
        self,
        func: Callable[..., Any],
        images: Sequence[Any],
        *extra_args: Any,
        **extra_kwargs: Any,
    ) -> List[Any]:
        """Apply *func* to each element of *images* in parallel.

        Parameters
        ----------
        func : Callable
            Processing function.  Called as ``func(image, *extra_args,
            **extra_kwargs)``.
        images : Sequence
            Input items (typically numpy arrays).
        *extra_args, **extra_kwargs
            Additional arguments forwarded to every *func* call.

        Returns
        -------
        List[Any]
            Results in the same order as *images*.
        """
        n = len(images)
        if n == 0:
            self._stats = ParallelStats()
            return []

        results: List[Any] = [None] * n
        errors = 0
        t0 = time.monotonic()

        pool_cls = self._get_pool_class()
        workers = min(self._max_workers, n)

        with pool_cls(max_workers=workers) as pool:
            future_to_idx: Dict[Future, int] = {}
            for idx, img in enumerate(images):
                fut = pool.submit(func, img, *extra_args, **extra_kwargs)
                future_to_idx[fut] = idx

            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                try:
                    results[idx] = fut.result()
                except Exception as exc:
                    errors += 1
                    results[idx] = None
                    logger.error(
                        "Parallel map item %d failed: %s", idx, exc, exc_info=True
                    )

        elapsed = time.monotonic() - t0
        self._stats = ParallelStats(
            total_items=n,
            elapsed_s=elapsed,
            throughput=n / elapsed if elapsed > 0 else 0.0,
            errors=errors,
        )
        logger.debug(
            "map: %d items in %.3fs (%.1f items/s, %d errors)",
            n, elapsed, self._stats.throughput, errors,
        )
        return results

    def par_start(self, func: Callable[..., Any], *args: Any) -> Future:
        """Submit an async operation to the shared pool.

        The caller is responsible for calling :meth:`par_join` or
        ``future.result()`` to retrieve the outcome.

        Parameters
        ----------
        func : Callable
            Function to execute.
        *args
            Positional arguments for *func*.

        Returns
        -------
        Future
            A :class:`~concurrent.futures.Future` representing the
            pending result.
        """
        return self._pool.submit(func, *args)

    @staticmethod
    def par_join(
        futures: Sequence[Future],
        timeout: Optional[float] = None,
    ) -> List[Any]:
        """Wait for all futures to complete and return their results.

        Parameters
        ----------
        futures : Sequence[Future]
            Futures returned by :meth:`par_start`.
        timeout : float, optional
            Maximum seconds to wait per future.

        Returns
        -------
        List[Any]
            Results in the same order as *futures*.
        """
        results: List[Any] = []
        for f in futures:
            results.append(f.result(timeout=timeout))
        return results

    # ------------------------------------------------------------------
    # Tiled processing
    # ------------------------------------------------------------------

    def tiled_apply(
        self,
        func: Callable[[np.ndarray], np.ndarray],
        image: np.ndarray,
        tile_size: Tuple[int, int] = (512, 512),
        overlap: int = 0,
    ) -> np.ndarray:
        """Split *image* into tiles, process in parallel, and stitch back.

        Parameters
        ----------
        func : Callable
            Processing function ``(tile) -> processed_tile``.  The output
            tile must have the same spatial dimensions as the input tile.
        image : np.ndarray
            Input image (H x W or H x W x C).
        tile_size : Tuple[int, int]
            ``(tile_height, tile_width)`` in pixels.
        overlap : int
            Number of overlapping pixels on each side.  Overlap regions
            are discarded during stitching to avoid boundary artefacts.

        Returns
        -------
        np.ndarray
            Processed image with the same shape as the input.
        """
        h, w = image.shape[:2]
        th, tw = tile_size
        if th <= 0 or tw <= 0:
            raise ValueError("tile_size must be positive.")
        if overlap < 0:
            raise ValueError("overlap must be non-negative.")

        tiles = self._split_tiles(h, w, th, tw, overlap)
        logger.debug(
            "tiled_apply: %d tiles (%dx%d, overlap=%d) for %dx%d image",
            len(tiles), th, tw, overlap, h, w,
        )

        # Extract tile images.
        tile_images = [image[t.y_start:t.y_end, t.x_start:t.x_end] for t in tiles]

        # Process tiles in parallel.
        processed = self.map(func, tile_images)

        # Stitch results.
        output = np.empty_like(image)
        for tile_spec, tile_result in zip(tiles, processed):
            if tile_result is None:
                continue
            # Compute the region within the processed tile to extract
            # (strip overlap).
            inner_y_start = tile_spec.y_out_start - tile_spec.y_start
            inner_y_end = inner_y_start + (tile_spec.y_out_end - tile_spec.y_out_start)
            inner_x_start = tile_spec.x_out_start - tile_spec.x_start
            inner_x_end = inner_x_start + (tile_spec.x_out_end - tile_spec.x_out_start)

            output[
                tile_spec.y_out_start:tile_spec.y_out_end,
                tile_spec.x_out_start:tile_spec.x_out_end,
            ] = tile_result[inner_y_start:inner_y_end, inner_x_start:inner_x_end]

        return output

    # ------------------------------------------------------------------
    # Multi-stage pipeline
    # ------------------------------------------------------------------

    def pipeline(
        self,
        stages: List[Callable[[Any], Any]],
        image: Any,
    ) -> Any:
        """Execute a multi-stage pipeline with stage-level parallelism.

        Each stage receives the output of the previous stage.  When there
        is only a single image, parallelism comes from overlapping the
        execution of different images across stages (ping-pong).  For a
        single image the stages are executed sequentially.

        Parameters
        ----------
        stages : List[Callable]
            Ordered list of processing functions.
        image : Any
            Initial input.

        Returns
        -------
        Any
            Output of the final stage.
        """
        result = image
        for idx, stage_fn in enumerate(stages):
            try:
                result = stage_fn(result)
            except Exception as exc:
                logger.error("Pipeline stage %d failed: %s", idx, exc, exc_info=True)
                raise
        return result

    def pipeline_batch(
        self,
        stages: List[Callable[[Any], Any]],
        images: Sequence[Any],
    ) -> List[Any]:
        """Execute a multi-stage pipeline on a batch of images.

        Each image is processed through all stages independently, with
        images parallelised at each stage boundary.

        Parameters
        ----------
        stages : List[Callable]
            Ordered processing stages.
        images : Sequence
            Input images / items.

        Returns
        -------
        List[Any]
            Results in the same order as *images*.
        """
        current = list(images)
        for idx, stage_fn in enumerate(stages):
            current = self.map(stage_fn, current)
            logger.debug("Pipeline stage %d/%d complete.", idx + 1, len(stages))
        return current

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_pool_class(self) -> type:
        """Return the appropriate executor class."""
        if self._pool_type == "process":
            return ProcessPoolExecutor
        return ThreadPoolExecutor

    @staticmethod
    def _split_tiles(
        h: int, w: int, th: int, tw: int, overlap: int,
    ) -> List[TileSpec]:
        """Compute tile specifications for a given image size."""
        tiles: List[TileSpec] = []
        step_y = th - 2 * overlap if overlap > 0 else th
        step_x = tw - 2 * overlap if overlap > 0 else tw
        step_y = max(step_y, 1)
        step_x = max(step_x, 1)

        for y in range(0, h, step_y):
            for x in range(0, w, step_x):
                y_start = max(y - overlap, 0)
                x_start = max(x - overlap, 0)
                y_end = min(y + th - overlap, h) if overlap > 0 else min(y + th, h)
                x_end = min(x + tw - overlap, w) if overlap > 0 else min(x + tw, w)

                # Extend to include overlap on the far side.
                y_end_with_overlap = min(y_end + overlap, h)
                x_end_with_overlap = min(x_end + overlap, w)

                tiles.append(
                    TileSpec(
                        y_start=y_start,
                        y_end=y_end_with_overlap,
                        x_start=x_start,
                        x_end=x_end_with_overlap,
                        y_out_start=y_start + (overlap if y > 0 else 0),
                        y_out_end=y_end,
                        x_out_start=x_start + (overlap if x > 0 else 0),
                        x_out_end=x_end,
                    )
                )

        return tiles


# =========================================================================
# Convenience functions
# =========================================================================

def par_threshold(
    images: Sequence[np.ndarray],
    threshold: float = 128.0,
    max_val: float = 255.0,
    method: int = 0,
    max_workers: int = _DEFAULT_WORKERS,
) -> List[np.ndarray]:
    """Apply binary threshold to multiple images in parallel.

    Parameters
    ----------
    images : Sequence[np.ndarray]
        Input images (grayscale).
    threshold : float
        Threshold value.
    max_val : float
        Value assigned to pixels above threshold.
    method : int
        OpenCV threshold type (default ``cv2.THRESH_BINARY`` = 0).
    max_workers : int
        Number of parallel workers.

    Returns
    -------
    List[np.ndarray]
        Thresholded images.
    """
    import cv2

    def _thresh(img: np.ndarray) -> np.ndarray:
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, result = cv2.threshold(gray, threshold, max_val, method)
        return result

    executor = ParallelExecutor(max_workers=max_workers, pool_type="thread")
    return executor.map(_thresh, images)


def par_filter(
    images: Sequence[np.ndarray],
    filter_func: Callable[[np.ndarray], np.ndarray],
    max_workers: int = _DEFAULT_WORKERS,
    pool_type: str = "thread",
) -> List[np.ndarray]:
    """Apply an arbitrary filter function to multiple images in parallel.

    Parameters
    ----------
    images : Sequence[np.ndarray]
        Input images.
    filter_func : Callable
        Function ``(image) -> filtered_image``.
    max_workers : int
        Number of parallel workers.
    pool_type : str
        ``"thread"`` or ``"process"``.

    Returns
    -------
    List[np.ndarray]
        Filtered images.
    """
    executor = ParallelExecutor(max_workers=max_workers, pool_type=pool_type)
    return executor.map(filter_func, images)


# =========================================================================
# ROIParallel
# =========================================================================
class ROIParallel:
    """Split a single large image by ROIs, process in parallel, and merge.

    Parameters
    ----------
    max_workers : int
        Number of parallel workers.
    pool_type : str
        ``"thread"`` or ``"process"``.

    Usage::

        roi_proc = ROIParallel(max_workers=4)
        rois = [(0, 0, 200, 200), (200, 0, 400, 200), ...]
        result = roi_proc.process(image, rois, my_func)
    """

    def __init__(
        self,
        max_workers: int = _DEFAULT_WORKERS,
        pool_type: str = "thread",
    ) -> None:
        self._executor = ParallelExecutor(
            max_workers=max_workers, pool_type=pool_type
        )

    def process(
        self,
        image: np.ndarray,
        rois: Sequence[Tuple[int, int, int, int]],
        func: Callable[[np.ndarray], np.ndarray],
        merge: bool = True,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Process ROI regions in parallel.

        Parameters
        ----------
        image : np.ndarray
            Source image.
        rois : Sequence[Tuple[int, int, int, int]]
            List of ``(x, y, width, height)`` ROI rectangles.
        func : Callable
            Processing function applied to each ROI crop.
        merge : bool
            If True, write processed ROI crops back into a copy of the
            original image and return it.  If False, return a list of
            individual ROI results.

        Returns
        -------
        np.ndarray or List[np.ndarray]
            Merged image or list of processed ROI crops.
        """
        if not rois:
            return image.copy() if merge else []

        # Extract ROI crops.
        crops: List[np.ndarray] = []
        for x, y, w, h in rois:
            x = max(x, 0)
            y = max(y, 0)
            x2 = min(x + w, image.shape[1])
            y2 = min(y + h, image.shape[0])
            crops.append(image[y:y2, x:x2].copy())

        # Process in parallel.
        processed = self._executor.map(func, crops)

        if not merge:
            return processed

        # Merge back.
        output = image.copy()
        for (x, y, w, h), result in zip(rois, processed):
            if result is None:
                continue
            x = max(x, 0)
            y = max(y, 0)
            x2 = min(x + w, output.shape[1])
            y2 = min(y + h, output.shape[0])
            rh, rw = result.shape[:2]
            # Handle potential size mismatch from func.
            copy_h = min(rh, y2 - y)
            copy_w = min(rw, x2 - x)
            output[y:y + copy_h, x:x + copy_w] = result[:copy_h, :copy_w]

        return output

    def process_with_results(
        self,
        image: np.ndarray,
        rois: Sequence[Tuple[int, int, int, int]],
        func: Callable[[np.ndarray], Tuple[np.ndarray, Any]],
    ) -> Tuple[np.ndarray, List[Any]]:
        """Process ROIs and collect both images and metadata.

        Parameters
        ----------
        image : np.ndarray
            Source image.
        rois : Sequence[Tuple[int, int, int, int]]
            ROI rectangles.
        func : Callable
            Function returning ``(processed_crop, metadata)``.

        Returns
        -------
        Tuple[np.ndarray, List[Any]]
            ``(merged_image, list_of_metadata)``
        """
        crops: List[np.ndarray] = []
        for x, y, w, h in rois:
            x, y = max(x, 0), max(y, 0)
            x2 = min(x + w, image.shape[1])
            y2 = min(y + h, image.shape[0])
            crops.append(image[y:y2, x:x2].copy())

        raw_results = self._executor.map(func, crops)

        output = image.copy()
        metadata_list: List[Any] = []

        for (x, y, w, h), raw in zip(rois, raw_results):
            if raw is None:
                metadata_list.append(None)
                continue
            processed_crop, meta = raw
            metadata_list.append(meta)

            x, y = max(x, 0), max(y, 0)
            x2 = min(x + w, output.shape[1])
            y2 = min(y + h, output.shape[0])
            rh, rw = processed_crop.shape[:2]
            copy_h = min(rh, y2 - y)
            copy_w = min(rw, x2 - x)
            output[y:y + copy_h, x:x + copy_w] = processed_crop[:copy_h, :copy_w]

        return output, metadata_list
