"""Tests for dl_anomaly.core.inspection_flow.

Covers StepResult, FlowResult, all concrete step types, InspectionFlow
orchestration, serialisation round-trips, and the STEP_REGISTRY.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dl_anomaly.core.inspection_flow import (
    ClassifyStep,
    CustomStep,
    DetectStep,
    FlowResult,
    FlowStep,
    InspectionFlow,
    JudgeStep,
    LocateStep,
    MeasureStep,
    STEP_REGISTRY,
    StepResult,
)


# ====================================================================== #
#  Fixtures                                                                #
# ====================================================================== #


@pytest.fixture
def sample_image() -> np.ndarray:
    """256x256x3 BGR uint8 test image with gradient pattern."""
    rng = np.random.RandomState(42)
    return rng.randint(50, 200, size=(256, 256, 3), dtype=np.uint8)


# ====================================================================== #
#  TestStepResult                                                          #
# ====================================================================== #


class TestStepResult:
    """Validate StepResult dataclass defaults and custom values."""

    def test_default_values(self) -> None:
        result = StepResult(
            step_name="s1",
            step_type="detect",
            success=True,
            data={},
        )
        assert result.step_name == "s1"
        assert result.step_type == "detect"
        assert result.success is True
        assert result.data == {}
        assert result.image is None
        assert result.elapsed_ms == 0.0
        assert result.message == ""

    def test_custom_values(self) -> None:
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        result = StepResult(
            step_name="custom_step",
            step_type="custom",
            success=False,
            data={"key": "value"},
            image=img,
            elapsed_ms=42.5,
            message="something failed",
        )
        assert result.step_name == "custom_step"
        assert result.step_type == "custom"
        assert result.success is False
        assert result.data == {"key": "value"}
        assert result.image is not None
        assert result.image.shape == (10, 10, 3)
        assert result.elapsed_ms == 42.5
        assert result.message == "something failed"


# ====================================================================== #
#  TestFlowResult                                                          #
# ====================================================================== #


class TestFlowResult:
    """Validate FlowResult dataclass."""

    def test_overall_pass_when_all_pass(self) -> None:
        steps = [
            StepResult("a", "detect", True, {}),
            StepResult("b", "judge", True, {"overall_pass": True}),
        ]
        fr = FlowResult(
            flow_name="test",
            overall_pass=True,
            steps=steps,
            total_time_ms=10.0,
            timestamp="2026-01-01T00:00:00Z",
        )
        assert fr.overall_pass is True
        assert len(fr.steps) == 2

    def test_has_timestamp(self) -> None:
        fr = FlowResult(
            flow_name="ts_test",
            overall_pass=False,
            steps=[],
            total_time_ms=0.0,
            timestamp="2026-03-19T12:00:00+00:00",
        )
        assert "2026" in fr.timestamp
        assert fr.summary == {}


# ====================================================================== #
#  TestConcreteSteps                                                       #
# ====================================================================== #


class TestConcreteSteps:
    """Execute each concrete step type with a synthetic image."""

    def test_locate_step_execute(self, sample_image: np.ndarray) -> None:
        """LocateStep without a template_path returns success=False."""
        step = LocateStep("Locate", config={})
        result = step.execute(sample_image, {})
        assert isinstance(result, StepResult)
        assert result.step_type == "locate"
        # No template configured, so success should be False.
        assert result.success is False
        assert "template_path" in result.message

    def test_detect_step_execute(self, sample_image: np.ndarray) -> None:
        """DetectStep with blob method runs without a checkpoint."""
        step = DetectStep(
            "Detect",
            config={"method": "blob", "threshold": 0.5},
        )
        result = step.execute(sample_image, {})
        assert isinstance(result, StepResult)
        assert result.step_type == "detect"
        assert result.success is True
        assert "anomaly_score" in result.data
        assert "defect_mask" in result.data
        assert "defect_regions" in result.data

    def test_measure_step_execute(self, sample_image: np.ndarray) -> None:
        """MeasureStep with no specs returns empty measurements."""
        step = MeasureStep("Measure", config={})
        result = step.execute(sample_image, {})
        assert result.success is True
        assert result.data["measurements"] == []
        assert result.data["all_in_tolerance"] is True

    def test_classify_step_execute(self, sample_image: np.ndarray) -> None:
        """ClassifyStep with no rules returns empty classifications."""
        step = ClassifyStep("Classify", config={})
        result = step.execute(sample_image, {})
        assert result.success is True
        assert result.data["classifications"] == []
        assert result.data["class_summary"] == {}

    def test_judge_step_pass(self, sample_image: np.ndarray) -> None:
        """JudgeStep passes when anomaly_score is below threshold."""
        context = {
            "detect": {"anomaly_score": 0.1, "is_defective": False},
        }
        step = JudgeStep(
            "Judge",
            config={
                "rules": [
                    {
                        "field": "detect.anomaly_score",
                        "operator": "lt",
                        "value": 0.5,
                    },
                ],
                "logic": "all_pass",
            },
        )
        result = step.execute(sample_image, context)
        assert result.success is True
        assert result.data["overall_pass"] is True
        assert result.data["score"] == 1.0

    def test_judge_step_fail(self, sample_image: np.ndarray) -> None:
        """JudgeStep fails when anomaly_score exceeds threshold."""
        context = {
            "detect": {"anomaly_score": 0.9, "is_defective": True},
        }
        step = JudgeStep(
            "Judge",
            config={
                "rules": [
                    {
                        "field": "detect.anomaly_score",
                        "operator": "lt",
                        "value": 0.5,
                    },
                ],
                "logic": "all_pass",
            },
        )
        result = step.execute(sample_image, context)
        assert result.success is True
        assert result.data["overall_pass"] is False
        assert result.data["score"] == 0.0

    def test_custom_step_execute(self, sample_image: np.ndarray) -> None:
        """CustomStep executes user-defined Python code."""
        code = (
            "def process(image, context):\n"
            "    h, w = image.shape[:2]\n"
            "    return {'height': h, 'width': w}\n"
        )
        step = CustomStep("Custom", config={"function_code": code})
        result = step.execute(sample_image, {})
        assert result.success is True
        assert result.data["height"] == 256
        assert result.data["width"] == 256


# ====================================================================== #
#  TestInspectionFlow                                                      #
# ====================================================================== #


class TestInspectionFlow:
    """Validate InspectionFlow step management and execution."""

    def test_add_step(self) -> None:
        flow = InspectionFlow("Test")
        flow.add_step(DetectStep("D1"))
        flow.add_step(JudgeStep("J1"))
        assert len(flow) == 2

    def test_remove_step(self) -> None:
        flow = InspectionFlow("Test")
        flow.add_step(DetectStep("D1"))
        flow.add_step(JudgeStep("J1"))
        flow.remove_step(0)
        assert len(flow) == 1
        assert flow.get_steps()[0].name == "J1"

    def test_execute_empty_flow(self, sample_image: np.ndarray) -> None:
        flow = InspectionFlow("Empty")
        result = flow.execute(sample_image)
        assert isinstance(result, FlowResult)
        # No steps, no failures, no judge -> overall_pass is True.
        assert result.overall_pass is True
        assert result.steps == []

    def test_execute_single_step(self, sample_image: np.ndarray) -> None:
        """Single blob-detect step should succeed."""
        flow = InspectionFlow("Single")
        flow.add_step(
            DetectStep("D", config={"method": "blob", "threshold": 0.5})
        )
        result = flow.execute(sample_image)
        assert isinstance(result, FlowResult)
        assert len(result.steps) == 1
        assert result.steps[0].success is True

    def test_execute_multi_step(self, sample_image: np.ndarray) -> None:
        """Locate -> Detect -> Judge pipeline runs end-to-end.

        LocateStep will fail (no template), but stop_on_failure=False
        allows the flow to continue.
        """
        flow = InspectionFlow("Multi", stop_on_failure=False)
        flow.add_step(LocateStep("Locate"))
        flow.add_step(
            DetectStep("Detect", config={"method": "blob", "threshold": 0.5})
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
        result = flow.execute(sample_image)
        assert isinstance(result, FlowResult)
        # 3 steps executed (locate fails but flow continues).
        assert len(result.steps) == 3
        # Judge should pass because anomaly_score < 999.
        assert result.overall_pass is True

    def test_disabled_step_skipped(self, sample_image: np.ndarray) -> None:
        flow = InspectionFlow("Disabled")
        step = DetectStep(
            "D", config={"method": "blob", "threshold": 0.5}
        )
        step.enabled = False
        flow.add_step(step)
        result = flow.execute(sample_image)
        # Disabled steps are not added to step_results.
        assert len(result.steps) == 0
        assert result.overall_pass is True

    def test_flow_timing(self, sample_image: np.ndarray) -> None:
        flow = InspectionFlow("Timing")
        flow.add_step(
            DetectStep("D", config={"method": "blob", "threshold": 0.5})
        )
        result = flow.execute(sample_image)
        assert result.total_time_ms > 0


# ====================================================================== #
#  TestFlowSerialization                                                   #
# ====================================================================== #


class TestFlowSerialization:
    """Validate to_dict / from_dict and save / load round-trips."""

    def test_to_dict_from_dict_roundtrip(self) -> None:
        flow = InspectionFlow("Roundtrip", stop_on_failure=False)
        flow.add_step(
            DetectStep("D", config={"method": "blob", "threshold": 0.3})
        )
        flow.add_step(
            JudgeStep(
                "J",
                config={
                    "rules": [
                        {
                            "field": "detect.anomaly_score",
                            "operator": "lt",
                            "value": 0.5,
                        },
                    ],
                    "logic": "all_pass",
                },
            )
        )

        # Serialize.
        data = {
            "flow_name": flow.name,
            "stop_on_failure": flow.stop_on_failure,
            "steps": [s.to_dict() for s in flow.get_steps()],
            "version": "1.0",
        }

        # Deserialize.
        loaded = InspectionFlow(
            name=data["flow_name"],
            stop_on_failure=data["stop_on_failure"],
        )
        for step_dict in data["steps"]:
            loaded.add_step(FlowStep.from_dict(step_dict))

        assert loaded.name == "Roundtrip"
        assert loaded.stop_on_failure is False
        assert len(loaded) == 2
        steps = loaded.get_steps()
        assert steps[0].step_type == "detect"
        assert steps[0].config["method"] == "blob"
        assert steps[1].step_type == "judge"

    def test_save_load_file_roundtrip(self, tmp_path: Path) -> None:
        flow = InspectionFlow("FileSave")
        flow.add_step(
            DetectStep("D", config={"method": "blob", "threshold": 0.2})
        )
        flow.add_step(MeasureStep("M", config={}))

        filepath = str(tmp_path / "test_flow.json")
        flow.save(filepath)
        assert Path(filepath).exists()

        loaded = InspectionFlow.load(filepath)
        assert loaded.name == "FileSave"
        assert len(loaded) == 2
        assert loaded.get_steps()[0].step_type == "detect"
        assert loaded.get_steps()[1].step_type == "measure"

    def test_unknown_step_type_raises(self) -> None:
        bad_dict = {
            "name": "Bad",
            "step_type": "nonexistent_type",
            "config": {},
            "enabled": True,
        }
        with pytest.raises(ValueError, match="Unknown step_type"):
            FlowStep.from_dict(bad_dict)


# ====================================================================== #
#  TestStepRegistry                                                        #
# ====================================================================== #


class TestStepRegistry:
    """Validate the global STEP_REGISTRY mapping."""

    def test_all_types_registered(self) -> None:
        expected = {"locate", "detect", "measure", "classify", "judge", "custom"}
        assert set(STEP_REGISTRY.keys()) == expected

    def test_from_dict_resolves_correct_class(self) -> None:
        mapping = {
            "locate": LocateStep,
            "detect": DetectStep,
            "measure": MeasureStep,
            "classify": ClassifyStep,
            "judge": JudgeStep,
            "custom": CustomStep,
        }
        for step_type, expected_cls in mapping.items():
            d = {
                "name": f"test_{step_type}",
                "step_type": step_type,
                "config": {},
                "enabled": True,
            }
            step = FlowStep.from_dict(d)
            assert isinstance(step, expected_cls)
            assert step.name == f"test_{step_type}"
            assert step.step_type == step_type
            assert step.enabled is True
