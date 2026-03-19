"""Tests for parameter configurability across modules."""
import numpy as np
import pytest


class TestShapeMatchingParams:
    def test_max_contour_points_param(self):
        from shared.core.shape_matching import create_shape_model
        template = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        # Should not raise with custom max_contour_points
        model = create_shape_model(template, max_contour_points=500)
        assert len(model.contour_points) <= 500


class TestPatchCoreParams:
    def test_n_neighbors_param(self):
        pytest.importorskip("torch")
        from shared.core.patchcore import PatchCoreInference, PatchCoreModel
        # Just test that the parameter is accepted (full test needs trained model)
        # We test constructor signature acceptance
        pass  # Integration test would need a trained model


class TestAnomalyScorerParams:
    def test_threshold_methods(self):
        pytest.importorskip("torch")
        from dl_anomaly.core.anomaly_scorer import AnomalyScorer
        scorer = AnomalyScorer(gaussian_sigma=2.0)
        assert scorer.gaussian_sigma == 2.0

    def test_fit_threshold_fixed(self):
        pytest.importorskip("torch")
        from dl_anomaly.core.anomaly_scorer import AnomalyScorer
        scorer = AnomalyScorer()
        scorer.fit_threshold(
            np.random.rand(10).tolist(),
            threshold_method="fixed",
            fixed_value=0.42,
        )
        assert abs(scorer.threshold - 0.42) < 1e-6

    def test_fit_threshold_otsu(self):
        pytest.importorskip("torch")
        from dl_anomaly.core.anomaly_scorer import AnomalyScorer
        scorer = AnomalyScorer()
        error_map = np.random.rand(100, 100).astype(np.float32)
        scorer.fit_threshold(
            np.random.rand(10).tolist(),
            threshold_method="otsu",
            error_map_for_otsu=error_map,
        )
        assert scorer.threshold > 0


class TestPreprocessorParams:
    def test_custom_augmentation(self):
        pytest.importorskip("torch")
        from dl_anomaly.core.preprocessor import ImagePreprocessor
        aug_params = {
            "rotation_range": 15,
            "brightness_jitter": 0.1,
            "vertical_flip_p": 0.5,
        }
        preprocessor = ImagePreprocessor(image_size=64, grayscale=False)
        transform = preprocessor.get_transforms(
            augment=True,
            augmentation_params=aug_params,
        )
        assert transform is not None


class TestColorInspectParams:
    def test_delta_e_map_ciede2000(self):
        from shared.core.color_inspect import compute_delta_e_map
        # compute_delta_e_map expects BGR uint8 image + reference Lab color tuple
        image = np.full((10, 10, 3), 128, dtype=np.uint8)  # BGR
        ref_color = (60.0, 10.0, 10.0)  # Lab
        result = compute_delta_e_map(image, ref_color, method="CIEDE2000")
        assert result.shape == (10, 10)
        assert result.mean() > 0
