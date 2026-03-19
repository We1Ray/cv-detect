"""Tests for shared.core.patchcore -- PatchCore anomaly detection module."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

from shared.core.patchcore import (  # noqa: E402
    FeatureExtractor,
    PatchCoreModel,
    PatchCoreTrainer,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


@pytest.fixture
def small_batch() -> torch.Tensor:
    """Synthetic image batch: (2, 3, 64, 64)."""
    rng = torch.Generator().manual_seed(0)
    return torch.rand(2, 3, 64, 64, generator=rng)


@pytest.fixture
def extractor_resnet18() -> FeatureExtractor:
    """FeatureExtractor with resnet18 and layers (layer2, layer3)."""
    ext = FeatureExtractor(
        backbone_name="resnet18",
        layers=("layer2", "layer3"),
        device="cpu",
    )
    yield ext
    ext.remove_hooks()


# ======================================================================
# TestFeatureExtractor
# ======================================================================


class TestFeatureExtractor:
    """Tests for the FeatureExtractor class."""

    def test_init_resnet18(self, extractor_resnet18: FeatureExtractor) -> None:
        """Creating an extractor with resnet18 should succeed."""
        assert extractor_resnet18.backbone_name == "resnet18"
        assert extractor_resnet18.layers == ("layer2", "layer3")
        assert len(extractor_resnet18._hooks) == 2

    def test_invalid_backbone(self) -> None:
        """An unknown backbone name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown backbone"):
            FeatureExtractor(backbone_name="nonexistent_net")

    def test_invalid_layer(self) -> None:
        """A layer name not present in the backbone should raise ValueError."""
        with pytest.raises(ValueError, match="not found in backbone"):
            FeatureExtractor(
                backbone_name="resnet18",
                layers=("layer2", "nonexistent_layer"),
            )

    def test_extract_shape(
        self,
        extractor_resnet18: FeatureExtractor,
        small_batch: torch.Tensor,
    ) -> None:
        """extract() should return a dict keyed by layer names."""
        features = extractor_resnet18.extract(small_batch)
        assert isinstance(features, dict)
        assert set(features.keys()) == {"layer2", "layer3"}
        for name, feat in features.items():
            assert feat.dim() == 4, f"{name} should be (B, C, H, W)"
            assert feat.shape[0] == 2, f"{name} batch dim should match input"

    def test_feature_dim(
        self,
        extractor_resnet18: FeatureExtractor,
        small_batch: torch.Tensor,
    ) -> None:
        """After extract(), get_feature_dim() should return a positive int."""
        extractor_resnet18.extract(small_batch)
        dim = extractor_resnet18.get_feature_dim()
        assert isinstance(dim, int)
        assert dim > 0

    def test_feature_dim_before_extract(
        self,
        extractor_resnet18: FeatureExtractor,
    ) -> None:
        """get_feature_dim() before any extract() should raise RuntimeError."""
        with pytest.raises(RuntimeError, match="call extract"):
            extractor_resnet18.get_feature_dim()

    def test_remove_hooks(
        self,
        extractor_resnet18: FeatureExtractor,
    ) -> None:
        """After remove_hooks(), the hook list should be empty."""
        assert len(extractor_resnet18._hooks) > 0
        extractor_resnet18.remove_hooks()
        assert len(extractor_resnet18._hooks) == 0


# ======================================================================
# TestPatchCoreModel
# ======================================================================


class TestPatchCoreModel:
    """Tests for the PatchCoreModel dataclass and its persistence."""

    def _make_model(self) -> PatchCoreModel:
        """Create a minimal PatchCoreModel with synthetic data."""
        rng = np.random.default_rng(42)
        return PatchCoreModel(
            memory_bank=rng.standard_normal((50, 128)).astype(np.float16),
            backbone_name="resnet18",
            layers=("layer2", "layer3"),
            image_size=64,
            feature_dim=128,
            coreset_ratio=0.1,
            threshold=1.234,
            config={"batch_size": 4},
        )

    def test_model_attributes(self) -> None:
        """All fields should be set correctly after construction."""
        model = self._make_model()
        assert model.memory_bank.shape == (50, 128)
        assert model.backbone_name == "resnet18"
        assert model.layers == ("layer2", "layer3")
        assert model.image_size == 64
        assert model.feature_dim == 128
        assert model.coreset_ratio == 0.1
        assert model.threshold == pytest.approx(1.234)
        assert model.config == {"batch_size": 4}

    def test_save_load_roundtrip(self, tmp_path) -> None:
        """Saving and loading should preserve all fields."""
        model = self._make_model()
        path = tmp_path / "model.npz"
        model.save(path)

        loaded = PatchCoreModel.load(path)
        assert loaded.memory_bank.shape == model.memory_bank.shape
        np.testing.assert_array_almost_equal(
            loaded.memory_bank, model.memory_bank, decimal=3,
        )
        assert loaded.backbone_name == model.backbone_name
        assert loaded.layers == model.layers
        assert loaded.image_size == model.image_size
        assert loaded.feature_dim == model.feature_dim
        assert loaded.coreset_ratio == model.coreset_ratio
        assert loaded.threshold == pytest.approx(model.threshold)

    def test_load_nonexistent(self, tmp_path) -> None:
        """Loading from a nonexistent path should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            PatchCoreModel.load(tmp_path / "does_not_exist.npz")


# ======================================================================
# TestCoresetSelection
# ======================================================================


class TestCoresetSelection:
    """Tests for PatchCoreTrainer._coreset_selection (static method)."""

    def test_coreset_reduces_size(self) -> None:
        """With ratio=0.1 on 1000 features, the result should be ~100."""
        rng = np.random.default_rng(0)
        features = rng.standard_normal((1000, 64)).astype(np.float32)
        coreset = PatchCoreTrainer._coreset_selection(features, ratio=0.1)
        assert coreset.shape == (100, 64)

    def test_coreset_ratio_one(self) -> None:
        """With ratio >= 1.0, all features should be returned."""
        rng = np.random.default_rng(0)
        features = rng.standard_normal((50, 32)).astype(np.float32)
        coreset = PatchCoreTrainer._coreset_selection(features, ratio=1.0)
        assert coreset.shape == features.shape

    def test_coreset_ratio_tiny(self) -> None:
        """A very small ratio should still return at least 1 feature."""
        rng = np.random.default_rng(0)
        features = rng.standard_normal((100, 16)).astype(np.float32)
        coreset = PatchCoreTrainer._coreset_selection(features, ratio=0.001)
        assert coreset.shape[0] >= 1
        assert coreset.shape[1] == 16

    def test_coreset_deterministic(self) -> None:
        """Same input should produce the same coreset (seed=42 internally)."""
        rng = np.random.default_rng(99)
        features = rng.standard_normal((200, 32)).astype(np.float32)
        coreset_a = PatchCoreTrainer._coreset_selection(features, ratio=0.1)
        coreset_b = PatchCoreTrainer._coreset_selection(features, ratio=0.1)
        np.testing.assert_array_equal(coreset_a, coreset_b)
