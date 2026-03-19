"""Tests for shared.core.parallel_pipeline -- multi-threaded processing pipeline."""

from __future__ import annotations

import time

import numpy as np
import pytest

from shared.core.parallel_pipeline import (
    ParallelPipeline,
    PipelineResult,
    PipelineStage,
    PipelineStats,
    ThreadSafeAccumulator,
    create_batch_processor,
    create_inspection_pipeline,
)


# ------------------------------------------------------------------
# PipelineStage
# ------------------------------------------------------------------


class TestPipelineStage:
    """Tests for the PipelineStage dataclass."""

    def test_basic_creation(self) -> None:
        """Should create a PipelineStage with the given name and function."""
        stage = PipelineStage(name="test", func=lambda x: x)
        assert stage.name == "test"
        assert stage.num_workers == 1
        assert stage.timeout == 30.0

    def test_custom_workers_and_timeout(self) -> None:
        """Should accept custom num_workers and timeout."""
        stage = PipelineStage(name="heavy", func=lambda x: x, num_workers=4, timeout=60.0)
        assert stage.num_workers == 4
        assert stage.timeout == 60.0

    def test_func_is_callable(self) -> None:
        """The func attribute should be callable."""
        fn = lambda x: x * 2
        stage = PipelineStage(name="double", func=fn)
        assert callable(stage.func)
        assert stage.func(5) == 10


# ------------------------------------------------------------------
# ThreadSafeAccumulator
# ------------------------------------------------------------------


class TestThreadSafeAccumulator:
    """Tests for the ThreadSafeAccumulator helper class."""

    def test_add_and_get_ordered(self) -> None:
        """Items should be returned in index order regardless of insertion order."""
        acc = ThreadSafeAccumulator()
        acc.add(2, "c")
        acc.add(0, "a")
        acc.add(1, "b")
        assert acc.get_ordered() == ["a", "b", "c"]

    def test_count(self) -> None:
        """Count should reflect the number of items added."""
        acc = ThreadSafeAccumulator()
        assert acc.count() == 0
        acc.add(0, "x")
        acc.add(1, "y")
        assert acc.count() == 2

    def test_empty_get_ordered(self) -> None:
        """Empty accumulator should return an empty list."""
        acc = ThreadSafeAccumulator()
        assert acc.get_ordered() == []


# ------------------------------------------------------------------
# ParallelPipeline
# ------------------------------------------------------------------


class TestParallelPipeline:
    """Tests for the ParallelPipeline class."""

    def test_requires_at_least_one_stage(self) -> None:
        """Creating a pipeline with no stages should raise ValueError."""
        with pytest.raises(ValueError, match="At least one"):
            ParallelPipeline(stages=[])

    def test_single_stage_identity(self) -> None:
        """A single-stage identity pipeline should pass data through."""
        stage = PipelineStage(name="identity", func=lambda x: x)
        pipeline = ParallelPipeline(stages=[stage])
        results = pipeline.process_batch(["a", "b", "c"])
        assert len(results) == 3
        assert all(isinstance(r, PipelineResult) for r in results)
        outputs = {r.output for r in results if r.success}
        assert outputs == {"a", "b", "c"}

    def test_multi_stage_pipeline(self) -> None:
        """A two-stage pipeline should chain transformations."""
        stage1 = PipelineStage(name="double", func=lambda x: x * 2)
        stage2 = PipelineStage(name="add_ten", func=lambda x: x + 10)
        pipeline = ParallelPipeline(stages=[stage1, stage2])
        results = pipeline.process_batch([1, 2, 3])
        outputs = sorted(r.output for r in results if r.success)
        assert outputs == [12, 14, 16]  # (1*2+10, 2*2+10, 3*2+10)

    def test_process_batch_returns_results_for_all_inputs(self) -> None:
        """process_batch should return one result per input."""
        stage = PipelineStage(name="square", func=lambda x: x ** 2)
        pipeline = ParallelPipeline(stages=[stage])
        inputs = list(range(10))
        results = pipeline.process_batch(inputs)
        assert len(results) == 10

    def test_result_has_timings(self) -> None:
        """Each PipelineResult should have timing information."""
        stage = PipelineStage(name="noop", func=lambda x: x)
        pipeline = ParallelPipeline(stages=[stage])
        results = pipeline.process_batch(["test"])
        r = results[0]
        assert "noop" in r.timings
        assert r.total_time > 0

    def test_error_handling(self) -> None:
        """A stage that raises should mark the result as failed."""

        def failing_func(x):
            if x == "bad":
                raise ValueError("intentional error")
            return x

        stage = PipelineStage(name="maybe_fail", func=failing_func)
        pipeline = ParallelPipeline(stages=[stage])
        results = pipeline.process_batch(["good", "bad", "good"])
        successes = [r for r in results if r.success]
        failures = [r for r in results if not r.success]
        assert len(successes) == 2
        assert len(failures) == 1
        assert failures[0].error is not None

    def test_empty_batch(self) -> None:
        """An empty input list should return an empty result list."""
        stage = PipelineStage(name="noop", func=lambda x: x)
        pipeline = ParallelPipeline(stages=[stage])
        results = pipeline.process_batch([])
        assert results == []

    def test_multi_worker_stage(self) -> None:
        """A stage with multiple workers should process items correctly."""
        stage = PipelineStage(name="double", func=lambda x: x * 2, num_workers=3)
        pipeline = ParallelPipeline(stages=[stage])
        results = pipeline.process_batch(list(range(20)))
        outputs = sorted(r.output for r in results if r.success)
        expected = sorted(x * 2 for x in range(20))
        assert outputs == expected

    def test_get_stats(self) -> None:
        """get_stats should return a PipelineStats object after processing."""
        stage = PipelineStage(name="noop", func=lambda x: x)
        pipeline = ParallelPipeline(stages=[stage])
        pipeline.process_batch(["a", "b", "c"])
        stats = pipeline.get_stats()
        assert isinstance(stats, PipelineStats)
        assert stats.total_processed == 3
        assert stats.total_failed == 0

    def test_stop_event(self) -> None:
        """Calling stop() should set the internal stop event."""
        stage = PipelineStage(name="noop", func=lambda x: x)
        pipeline = ParallelPipeline(stages=[stage])
        pipeline.stop()
        assert pipeline._stop_event.is_set()


# ------------------------------------------------------------------
# create_batch_processor
# ------------------------------------------------------------------


class TestCreateBatchProcessor:
    """Tests for the create_batch_processor factory."""

    def test_creates_single_stage_pipeline(self) -> None:
        """Should create a pipeline with exactly one stage."""
        pipeline = create_batch_processor(lambda x: x + 1, num_workers=2)
        assert isinstance(pipeline, ParallelPipeline)
        assert len(pipeline._stages) == 1
        assert pipeline._stages[0].name == "process"

    def test_processes_items(self) -> None:
        """The created pipeline should process items correctly."""
        pipeline = create_batch_processor(lambda x: x.upper(), num_workers=2)
        results = pipeline.process_batch(["hello", "world"])
        outputs = {r.output for r in results if r.success}
        assert outputs == {"HELLO", "WORLD"}


# ------------------------------------------------------------------
# create_inspection_pipeline
# ------------------------------------------------------------------


class TestCreateInspectionPipeline:
    """Tests for the create_inspection_pipeline factory."""

    def test_creates_three_stage_pipeline(self) -> None:
        """Should create a pipeline with three stages (preprocess, inference, postprocess)."""
        pipeline = create_inspection_pipeline(checkpoint_path=None)
        assert isinstance(pipeline, ParallelPipeline)
        assert len(pipeline._stages) == 3

    def test_stage_names(self) -> None:
        """The three stages should be named preprocess, inference, postprocess."""
        pipeline = create_inspection_pipeline(checkpoint_path=None)
        names = [s.name for s in pipeline._stages]
        assert names == ["preprocess", "inference", "postprocess"]

    def test_inference_is_single_worker(self) -> None:
        """Inference stage should have exactly 1 worker (GPU serialization)."""
        pipeline = create_inspection_pipeline(checkpoint_path=None)
        inference_stage = pipeline._stages[1]
        assert inference_stage.num_workers == 1

    def test_custom_worker_counts(self) -> None:
        """Custom worker counts should be reflected in the stages."""
        pipeline = create_inspection_pipeline(
            checkpoint_path=None,
            num_preprocess_workers=4,
            num_postprocess_workers=3,
        )
        assert pipeline._stages[0].num_workers == 4
        assert pipeline._stages[2].num_workers == 3
