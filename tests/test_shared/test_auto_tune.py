"""Tests for shared.core.auto_tune."""
import numpy as np
import pytest
from shared.core.auto_tune import AutoTuner, TuneResult


class TestAutoTuner:
    def test_perfect_separation(self, tmp_path):
        """OK scores all below threshold, NG all above."""
        import cv2
        ok_dir = tmp_path / "ok"
        ng_dir = tmp_path / "ng"
        ok_dir.mkdir()
        ng_dir.mkdir()

        # Create images (scorer will use mean brightness)
        for i in range(5):
            dark = np.full((32, 32, 3), 50, dtype=np.uint8)
            cv2.imwrite(str(ok_dir / f"ok_{i}.png"), dark)
            bright = np.full((32, 32, 3), 200, dtype=np.uint8)
            cv2.imwrite(str(ng_dir / f"ng_{i}.png"), bright)

        # Scorer: mean pixel value / 255
        scorer = lambda img: float(np.mean(img)) / 255.0
        tuner = AutoTuner(scorer)
        result = tuner.tune(ok_dir, ng_dir, n_thresholds=50)
        assert isinstance(result, TuneResult)
        assert result.best_score > 0.9  # should achieve near-perfect F1
        assert result.precision > 0.9
        assert result.recall > 0.9

    def test_roc_auc(self):
        labels = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        auc = AutoTuner._compute_roc_auc(labels, scores)
        assert auc > 0.9

    def test_metrics_computation(self):
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.4, 0.6, 0.9])
        metrics = AutoTuner._compute_metrics(labels, scores, 0.5)
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0

    def test_different_metrics(self, tmp_path):
        import cv2
        ok_dir = tmp_path / "ok"
        ng_dir = tmp_path / "ng"
        ok_dir.mkdir()
        ng_dir.mkdir()
        for i in range(3):
            cv2.imwrite(str(ok_dir / f"ok_{i}.png"), np.full((32, 32, 3), 50, dtype=np.uint8))
            cv2.imwrite(str(ng_dir / f"ng_{i}.png"), np.full((32, 32, 3), 200, dtype=np.uint8))

        scorer = lambda img: float(np.mean(img)) / 255.0
        tuner = AutoTuner(scorer)
        for metric in ["f1", "precision", "recall", "balanced_accuracy", "youden_j"]:
            result = tuner.tune(ok_dir, ng_dir, n_thresholds=20, metric=metric)
            assert result.optimal_threshold > 0
