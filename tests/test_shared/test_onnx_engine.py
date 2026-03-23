"""Tests for the ONNX inference engine module."""
import pytest
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# The onnx_engine module imports torch at the top level, so guard it.
torch = pytest.importorskip("torch")

from shared.core.onnx_engine import (
    OnnxModelInfo,
    check_onnxruntime_available,
)


# ======================================================================
# onnxruntime availability helpers
# ======================================================================

class TestOnnxruntimeAvailability:
    def test_check_returns_bool(self):
        result = check_onnxruntime_available()
        assert isinstance(result, bool)

    def test_get_available_providers_when_installed(self):
        if not check_onnxruntime_available():
            pytest.skip("onnxruntime not installed")
        from shared.core.onnx_engine import get_available_providers
        providers = get_available_providers()
        assert isinstance(providers, list)
        assert "CPUExecutionProvider" in providers


# ======================================================================
# OnnxModelInfo data class
# ======================================================================

class TestOnnxModelInfo:
    def test_create_info(self):
        info = OnnxModelInfo(
            path="/tmp/model.onnx",
            input_names=["input"],
            output_names=["output"],
            input_shapes={"input": [1, 3, 256, 256]},
            output_shapes={"output": [1, 3, 256, 256]},
            model_type="autoencoder",
            metadata={"key": "value"},
            opset_version=14,
        )
        assert info.path == "/tmp/model.onnx"
        assert info.model_type == "autoencoder"
        assert info.opset_version == 14
        assert info.input_names == ["input"]
        assert info.metadata["key"] == "value"

    def test_default_metadata_is_empty(self):
        info = OnnxModelInfo(
            path="model.onnx",
            input_names=["input"],
            output_names=["output"],
            input_shapes={},
            output_shapes={},
            model_type="custom",
        )
        assert info.metadata == {}
        assert info.opset_version == 14


# ======================================================================
# OnnxInferenceEngine
# ======================================================================

class TestOnnxInferenceEngine:
    def test_init_nonexistent_model_raises(self):
        """Initializing with a non-existent path should raise FileNotFoundError."""
        if not check_onnxruntime_available():
            pytest.skip("onnxruntime not installed")
        from shared.core.onnx_engine import OnnxInferenceEngine
        with pytest.raises(FileNotFoundError, match="ONNX model not found"):
            OnnxInferenceEngine("/nonexistent/path/model.onnx")

    def test_init_without_onnxruntime(self, monkeypatch):
        """If onnxruntime is not available, should raise ImportError."""
        from shared.core import onnx_engine
        monkeypatch.setattr(onnx_engine, "check_onnxruntime_available", lambda: False)
        with pytest.raises(ImportError, match="onnxruntime is required"):
            onnx_engine.OnnxInferenceEngine("/tmp/fake.onnx")

    def test_class_has_expected_methods(self):
        """Verify the public API surface of OnnxInferenceEngine."""
        from shared.core.onnx_engine import OnnxInferenceEngine
        assert hasattr(OnnxInferenceEngine, "run")
        assert hasattr(OnnxInferenceEngine, "run_batch")
        assert hasattr(OnnxInferenceEngine, "warmup")
        assert hasattr(OnnxInferenceEngine, "get_info")
        assert callable(getattr(OnnxInferenceEngine, "run"))


class TestOnnxInferenceEngineWithRealModel:
    """Tests that require onnxruntime and a real exported model.

    These tests export a small PyTorch model to ONNX at fixture time.
    """

    @pytest.fixture
    def onnx_model_path(self, tmp_path):
        """Export a tiny PyTorch model to ONNX and return the path."""
        if not check_onnxruntime_available():
            pytest.skip("onnxruntime not installed")
        try:
            import torch
            import torch.nn as nn
            import onnx  # noqa: F401
        except ImportError:
            pytest.skip("torch or onnx not installed")

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 1, 3, padding=1)

            def forward(self, x):
                return self.conv(x)

        model = TinyModel()
        model.eval()
        dummy = torch.randn(1, 1, 16, 16)
        path = tmp_path / "tiny.onnx"
        torch.onnx.export(
            model, dummy, str(path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )
        return path

    def test_load_and_run(self, onnx_model_path):
        from shared.core.onnx_engine import OnnxInferenceEngine
        engine = OnnxInferenceEngine(str(onnx_model_path), device="cpu")
        inp = np.random.randn(1, 1, 16, 16).astype(np.float32)
        out = engine.run(inp)
        assert isinstance(out, np.ndarray)
        assert out.shape == (1, 1, 16, 16)

    def test_run_auto_adds_batch_dim(self, onnx_model_path):
        from shared.core.onnx_engine import OnnxInferenceEngine
        engine = OnnxInferenceEngine(str(onnx_model_path), device="cpu")
        # Pass without batch dim
        inp = np.random.randn(1, 16, 16).astype(np.float32)
        out = engine.run(inp)
        assert out.shape == (1, 1, 16, 16)

    def test_run_wrong_ndim_raises(self, onnx_model_path):
        from shared.core.onnx_engine import OnnxInferenceEngine
        engine = OnnxInferenceEngine(str(onnx_model_path), device="cpu")
        inp = np.random.randn(16, 16).astype(np.float32)  # 2D, expects 4D
        with pytest.raises(ValueError, match="dimensions"):
            engine.run(inp)

    def test_run_batch(self, onnx_model_path):
        from shared.core.onnx_engine import OnnxInferenceEngine
        engine = OnnxInferenceEngine(str(onnx_model_path), device="cpu")
        images = [np.random.randn(1, 16, 16).astype(np.float32) for _ in range(5)]
        results = engine.run_batch(images, batch_size=2)
        assert len(results) == 5
        for r in results:
            assert r.shape == (1, 16, 16)

    def test_get_info(self, onnx_model_path):
        from shared.core.onnx_engine import OnnxInferenceEngine
        engine = OnnxInferenceEngine(str(onnx_model_path), device="cpu")
        info = engine.get_info()
        assert isinstance(info, OnnxModelInfo)
        assert info.input_names == ["input"]
        assert info.output_names == ["output"]

    def test_warmup_runs(self, onnx_model_path):
        from shared.core.onnx_engine import OnnxInferenceEngine
        engine = OnnxInferenceEngine(str(onnx_model_path), device="cpu")
        # Should not raise
        engine.warmup(n=2)


# ======================================================================
# OnnxAnomalyDetector
# ======================================================================

class TestOnnxAnomalyDetector:
    def test_class_has_expected_methods(self):
        from shared.core.onnx_engine import OnnxAnomalyDetector
        assert hasattr(OnnxAnomalyDetector, "inspect_single")
        assert hasattr(OnnxAnomalyDetector, "inspect_batch")
        assert hasattr(OnnxAnomalyDetector, "set_threshold")
        assert hasattr(OnnxAnomalyDetector, "fit_threshold")
