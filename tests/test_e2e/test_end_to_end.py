"""End-to-end integration tests for the cv-detect project.

Covers full training/inspection pipelines for AutoEncoder, VariationModel,
InspectionFlow, PipelineModel, and SPC database workflows.  All tests
use synthetic images, tiny model sizes, and tmp_path for isolation.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import sys
from pathlib import Path as _Path

import cv2
import numpy as np
import pytest
import torch

# Ensure project root and sub-packages are importable
_PROJECT_ROOT = _Path(__file__).resolve().parent.parent.parent
for _p in (str(_PROJECT_ROOT), str(_PROJECT_ROOT / "dl_anomaly"), str(_PROJECT_ROOT / "variation_model")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dl_anomaly.config import Config as AEConfig
from dl_anomaly.core.inspection_flow import (
    DetectStep,
    FlowResult,
    InspectionFlow,
    JudgeStep,
    LocateStep,
    MeasureStep,
)
from dl_anomaly.pipeline.inference import InferencePipeline, InspectionResult
from dl_anomaly.pipeline.trainer import TrainingPipeline
from shared.core.pipeline_model import PipelineModel
from shared.core.results_db import InspectionRecord, ResultsDatabase

try:
    from variation_model.core.variation_model import VariationModel
    from variation_model.core.inspector import Inspector as VMInspector
    from variation_model.config import Config as VMConfig
    _HAS_VM = True
except ImportError:
    _HAS_VM = False


# ====================================================================== #
#  Shared helpers                                                          #
# ====================================================================== #


def _make_synthetic_images(
    n: int,
    size: int,
    channels: int = 3,
    seed: int = 42,
) -> list[np.ndarray]:
    """Generate *n* synthetic uint8 images of shape (size, size, channels)."""
    rng = np.random.RandomState(seed)
    return [
        rng.randint(80, 180, (size, size, channels), dtype=np.uint8)
        for _ in range(n)
    ]


def _write_images(images: list[np.ndarray], directory: Path) -> list[Path]:
    """Write images to *directory* as PNG files and return their paths."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for idx, img in enumerate(images):
        p = directory / f"img_{idx:04d}.png"
        cv2.imwrite(str(p), img)
        paths.append(p)
    return paths


# ====================================================================== #
#  TestAutoEncoderE2E                                                      #
# ====================================================================== #


class TestAutoEncoderE2E:
    """Train an autoencoder on synthetic data, inspect, and checkpoint."""

    @pytest.fixture
    def ae_config(self, tmp_path: Path) -> AEConfig:
        train_dir = tmp_path / "train"
        test_dir = tmp_path / "test"

        # Write 10 synthetic training images.
        images = _make_synthetic_images(10, 64)
        _write_images(images, train_dir)

        # Write 1 test image.
        test_img = _make_synthetic_images(1, 64, seed=99)
        _write_images(test_img, test_dir)

        return AEConfig(
            train_image_dir=train_dir,
            test_image_dir=test_dir,
            checkpoint_dir=tmp_path / "checkpoints",
            results_dir=tmp_path / "results",
            image_size=64,
            grayscale=False,
            latent_dim=32,
            base_channels=16,
            num_encoder_blocks=2,
            batch_size=4,
            learning_rate=0.001,
            num_epochs=2,
            early_stopping_patience=2,
            device="cpu",
        )

    def test_train_and_inspect(self, ae_config: AEConfig) -> None:
        # Train.
        pipeline = TrainingPipeline(ae_config)
        history = pipeline.run()

        assert "checkpoint_path" in history
        assert Path(history["checkpoint_path"]).exists()

        # Inspect.
        test_images = list(ae_config.test_image_dir.glob("*.png"))
        assert len(test_images) >= 1

        inf = InferencePipeline(history["checkpoint_path"])
        result = inf.inspect_single(str(test_images[0]))

        assert isinstance(result, InspectionResult)
        assert result.original is not None
        assert result.reconstruction is not None
        assert result.error_map is not None
        assert result.defect_mask is not None
        assert isinstance(result.anomaly_score, float)
        assert isinstance(result.is_defective, bool)

    def test_checkpoint_save_load(self, ae_config: AEConfig) -> None:
        # Train.
        pipeline = TrainingPipeline(ae_config)
        history = pipeline.run()
        ckpt_path = history["checkpoint_path"]

        # Inspect with original checkpoint.
        test_images = list(ae_config.test_image_dir.glob("*.png"))
        inf1 = InferencePipeline(ckpt_path)
        result1 = inf1.inspect_single(str(test_images[0]))

        # Load checkpoint into a new pipeline and re-inspect.
        inf2 = InferencePipeline(ckpt_path)
        result2 = inf2.inspect_single(str(test_images[0]))

        # Scores should be identical from the same checkpoint.
        assert abs(result1.anomaly_score - result2.anomaly_score) < 1e-5


# ====================================================================== #
#  TestVariationModelE2E                                                   #
# ====================================================================== #


@pytest.mark.skipif(not _HAS_VM, reason="variation_model import unavailable")
class TestVariationModelE2E:
    """Train a VariationModel on synthetic images and inspect."""

    def test_train_and_inspect(self, tmp_path: Path) -> None:
        # Create 5 similar grayscale images.
        rng = np.random.RandomState(42)
        base = rng.randint(100, 150, (64, 64), dtype=np.uint8)
        images = [
            np.clip(base + rng.randint(-5, 5, base.shape), 0, 255).astype(
                np.uint8
            )
            for _ in range(5)
        ]

        # Train variation model.
        model = VariationModel()
        for img in images:
            model.train_incremental(img)

        assert model.is_trained
        assert model.count == 5

        # Prepare thresholds.
        model.prepare(abs_threshold=10, var_threshold=3.0)

        # Create a config for the inspector.
        vm_config = VMConfig(
            train_image_dir=tmp_path / "train",
            test_image_dir=tmp_path / "test",
            model_save_dir=tmp_path / "models",
            results_dir=tmp_path / "results",
            target_width=64,
            target_height=64,
            grayscale=True,
        )

        # Inspect with a modified image (inject a bright spot).
        test_img = base.copy()
        test_img[20:30, 20:30] = 255  # bright defect patch

        inspector = VMInspector(model, vm_config)
        result = inspector.compare(test_img)

        assert result.defect_mask is not None
        assert result.difference_image is not None
        assert isinstance(result.is_defective, bool)
        assert isinstance(result.score, float)


# ====================================================================== #
#  TestInspectionFlowE2E                                                   #
# ====================================================================== #


class TestInspectionFlowE2E:
    """Run a full InspectionFlow pipeline on synthetic data."""

    def test_full_pipeline(self) -> None:
        image = np.random.RandomState(42).randint(
            50, 200, (256, 256, 3), dtype=np.uint8
        )

        flow = InspectionFlow("E2E Pipeline", stop_on_failure=False)

        # LocateStep will fail (no template), but we continue.
        flow.add_step(LocateStep("Locate"))

        # DetectStep with blob method (no checkpoint needed).
        flow.add_step(
            DetectStep(
                "Detect",
                config={"method": "blob", "threshold": 0.5},
            )
        )

        # MeasureStep with no specs (passes trivially).
        flow.add_step(MeasureStep("Measure", config={}))

        # JudgeStep checking that anomaly_score < very large number.
        flow.add_step(
            JudgeStep(
                "Judge",
                config={
                    "rules": [
                        {
                            "field": "detect.anomaly_score",
                            "operator": "lt",
                            "value": 999.0,
                        },
                    ],
                    "logic": "all_pass",
                },
            )
        )

        result = flow.execute(image)

        assert isinstance(result, FlowResult)
        assert isinstance(result.overall_pass, bool)
        # All 4 steps should have results (locate fails but continues).
        assert len(result.steps) == 4
        # Each step has a result.
        for sr in result.steps:
            assert sr.step_name in ("Locate", "Detect", "Measure", "Judge")
        assert result.total_time_ms > 0
        assert result.timestamp != ""


# ====================================================================== #
#  TestPipelineModelE2E                                                    #
# ====================================================================== #


class TestPipelineModelE2E:
    """Build, save, load, and verify a PipelineModel."""

    def test_build_save_load_execute(self, tmp_path: Path) -> None:
        # Build a flow with mock steps.
        flow = InspectionFlow("Model Flow", stop_on_failure=False)
        flow.add_step(
            DetectStep(
                "Detect",
                config={"method": "blob", "threshold": 0.5},
            )
        )
        flow.add_step(
            JudgeStep(
                "Judge",
                config={
                    "rules": [
                        {
                            "field": "detect.anomaly_score",
                            "operator": "lt",
                            "value": 999.0,
                        },
                    ],
                    "logic": "all_pass",
                },
            )
        )

        # Build PipelineModel.
        model = PipelineModel.build(
            name="E2E Test Model",
            author="Test Suite",
            description="Integration test model",
            version="1.0.0",
            flow=flow,
        )

        # Save.
        model_path = str(tmp_path / "e2e_test.cpmodel")
        model.save(model_path)
        assert Path(model_path).exists()

        # Load and verify metadata.
        with PipelineModel.load(model_path) as loaded:
            assert loaded.metadata["name"] == "E2E Test Model"
            assert loaded.metadata["author"] == "Test Suite"
            assert loaded.metadata["version"] == "1.0.0"
            assert loaded.metadata["description"] == "Integration test model"


# ====================================================================== #
#  TestSPCDatabaseE2E                                                      #
# ====================================================================== #


class TestSPCDatabaseE2E:
    """Insert inspection records and compute SPC metrics."""

    def test_inspect_and_store(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "test_results.db")
        db = ResultsDatabase(db_path)

        # Insert multiple records simulating inspection results.
        rng = np.random.RandomState(42)
        scores = rng.uniform(0.1, 0.8, size=20)

        for i, score in enumerate(scores):
            record = InspectionRecord(
                timestamp=datetime(2026, 3, 19, 10, 0, i).isoformat(),
                image_path=f"/images/img_{i:04d}.png",
                model_type="autoencoder",
                anomaly_score=float(score),
                threshold=0.5,
                is_defective=bool(score > 0.5),
                defect_count=int(score > 0.5),
                total_defect_area=int(score * 1000) if score > 0.5 else 0,
                max_defect_area=int(score * 500) if score > 0.5 else 0,
                batch_id="batch_001",
                line_id="line_A",
            )
            db.insert_record(record)

        # Compute SPC metrics.
        metrics = db.compute_spc_metrics(field="anomaly_score")

        assert metrics.n_samples == 20
        assert metrics.mean > 0
        assert metrics.std > 0
        assert metrics.ucl > metrics.mean
        assert metrics.lcl < metrics.mean

        # Verify we can query back records.
        records = db.query_records(limit=5)
        assert len(records) == 5
        assert all(isinstance(r, InspectionRecord) for r in records)
