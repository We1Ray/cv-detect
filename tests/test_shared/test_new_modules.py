"""Tests for the 18 new Halcon-equivalent core modules.

All tests are designed to run with numpy, cv2 (optional via importorskip),
scipy, and scikit-learn available.  Heavy dependencies (torch, ultralytics,
harvesters) are never required -- mocks and skip decorators are used.
"""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")


# ============================================================
# 1. Photometric Stereo
# ============================================================


class TestPhotometricStereo:
    """Tests for shared.core.photometric_stereo."""

    def test_light_direction_from_angle(self):
        from shared.core.photometric_stereo import LightDirection

        ld = LightDirection.from_angle(0, 45)
        arr = ld.to_array()
        assert arr.shape == (3,)
        assert abs(np.linalg.norm(arr) - 1.0) < 1e-6

    def test_light_direction_from_angle_90_elevation(self):
        from shared.core.photometric_stereo import LightDirection

        ld = LightDirection.from_angle(0, 90)
        arr = ld.to_array()
        # At 90-degree elevation the light points straight up (z ~= 1).
        assert abs(arr[2] - 1.0) < 1e-6

    def test_standard_4_light(self):
        from shared.core.photometric_stereo import PhotometricStereo

        lights = PhotometricStereo.create_standard_4_light()
        assert len(lights) == 4

    def test_standard_8_light(self):
        from shared.core.photometric_stereo import PhotometricStereo

        lights = PhotometricStereo.create_standard_8_light()
        assert len(lights) == 8

    def test_compute_basic(self):
        from shared.core.photometric_stereo import LightDirection, PhotometricStereo

        ps = PhotometricStereo(method="least_squares")
        lights = PhotometricStereo.create_standard_4_light(elevation_deg=60)
        h, w = 32, 32
        images = []
        for ld in lights:
            arr = ld.to_array()
            # Flat surface normal = (0, 0, 1), uniform albedo.
            intensity = max(arr[2], 0) * 200 + 20
            images.append(np.full((h, w), intensity, dtype=np.uint8))
        result = ps.compute(images, lights)
        assert result.normal_map.shape == (h, w, 3)
        assert result.albedo_map.shape == (h, w)
        assert result.gradient_x.shape == (h, w)

    def test_compute_requires_3_images(self):
        from shared.core.photometric_stereo import LightDirection, PhotometricStereo

        ps = PhotometricStereo()
        with pytest.raises(ValueError, match="at least 3"):
            ps.compute(
                [np.zeros((10, 10))],
                [LightDirection(0, 0, 1)],
            )

    def test_compute_mismatch_raises(self):
        from shared.core.photometric_stereo import LightDirection, PhotometricStereo

        ps = PhotometricStereo()
        images = [np.zeros((10, 10), dtype=np.uint8)] * 4
        lights = [LightDirection.from_angle(i * 90, 45) for i in range(3)]
        with pytest.raises(ValueError, match="must match"):
            ps.compute(images, lights)

    def test_normal_to_rgb(self):
        from shared.core.photometric_stereo import PhotometricResult

        normal = np.zeros((5, 5, 3), dtype=np.float32)
        normal[:, :, 2] = 1.0  # pointing up
        result = PhotometricResult(
            normal_map=normal,
            albedo_map=np.ones((5, 5), dtype=np.float32),
            gradient_x=np.zeros((5, 5), dtype=np.float32),
            gradient_y=np.zeros((5, 5), dtype=np.float32),
        )
        rgb = result.normal_to_rgb()
        assert rgb.shape == (5, 5, 3)
        assert rgb.dtype == np.uint8

    def test_gradient_magnitude(self):
        from shared.core.photometric_stereo import PhotometricResult

        gx = np.full((5, 5), 3.0, dtype=np.float32)
        gy = np.full((5, 5), 4.0, dtype=np.float32)
        result = PhotometricResult(
            normal_map=np.zeros((5, 5, 3), dtype=np.float32),
            albedo_map=np.ones((5, 5), dtype=np.float32),
            gradient_x=gx,
            gradient_y=gy,
        )
        mag = result.gradient_magnitude()
        np.testing.assert_allclose(mag, 5.0, atol=0.01)


# ============================================================
# 2. Image Sequence
# ============================================================


class TestImageSequence:
    """Tests for shared.core.image_sequence."""

    def test_accumulator_mean(self):
        from shared.core.image_sequence import ImageAccumulator

        acc = ImageAccumulator()
        img1 = np.full((10, 10), 100, dtype=np.uint8)
        img2 = np.full((10, 10), 200, dtype=np.uint8)
        acc.add(img1)
        acc.add(img2)
        stats = acc.get_stats()
        assert stats.count == 2
        np.testing.assert_allclose(stats.mean, 150.0, atol=0.1)

    def test_accumulator_std(self):
        from shared.core.image_sequence import ImageAccumulator

        acc = ImageAccumulator()
        for v in [100, 200]:
            acc.add(np.full((5, 5), v, dtype=np.uint8))
        stats = acc.get_stats()
        # Welford with ddof=1: std([100, 200]) ~= 70.71
        expected_std = np.std([100, 200], ddof=1)
        np.testing.assert_allclose(stats.std[0, 0], expected_std, atol=0.5)

    def test_accumulator_empty_raises(self):
        from shared.core.image_sequence import ImageAccumulator

        acc = ImageAccumulator()
        with pytest.raises(ValueError, match="No images"):
            acc.get_stats()

    def test_accumulator_reset(self):
        from shared.core.image_sequence import ImageAccumulator

        acc = ImageAccumulator()
        acc.add(np.full((3, 3), 50, dtype=np.uint8))
        acc.reset()
        assert acc.count == 0

    def test_mean_image_func(self):
        from shared.core.image_sequence import mean_image

        imgs = [np.full((10, 10), v, dtype=np.uint8) for v in [50, 150]]
        result = mean_image(imgs)
        assert result.dtype == np.uint8
        assert result[0, 0] == 100

    def test_temporal_filter_ema(self):
        from shared.core.image_sequence import TemporalFilter

        tf = TemporalFilter(method="ema", alpha=0.5)
        f1 = np.full((5, 5), 100, dtype=np.uint8)
        f2 = np.full((5, 5), 200, dtype=np.uint8)
        out1 = tf.filter(f1)
        assert out1[0, 0] == 100  # first frame returned as-is
        out2 = tf.filter(f2)
        assert 100 < out2[0, 0] < 200

    def test_background_model_running_avg(self):
        from shared.core.image_sequence import BackgroundModel

        bg = BackgroundModel(method="running_avg", learning_rate=0.5)
        frame = np.full((20, 20), 128, dtype=np.uint8)
        mask = bg.update(frame)
        assert mask.shape == (20, 20)

    def test_temporal_denoise_median(self):
        from shared.core.image_sequence import temporal_denoise

        imgs = [np.full((10, 10), v, dtype=np.uint8) for v in [90, 100, 110]]
        result = temporal_denoise(imgs, method="median")
        assert result[0, 0] == 100


# ============================================================
# 3. Classifier
# ============================================================


class TestClassifier:
    """Tests for shared.core.classifier (requires scikit-learn)."""

    sklearn = pytest.importorskip("sklearn")

    def test_train_and_predict_knn(self):
        from shared.core.classifier import ClassifierInference, ClassifierTrainer

        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
        y = np.array([0, 1, 1, 0])
        trainer = ClassifierTrainer(
            model_type="knn", class_names=["A", "B"], n_neighbors=1
        )
        model = trainer.train(X, y)
        inf = ClassifierInference(model)
        result = inf.predict(np.array([0, 0], dtype=np.float32))
        assert result.class_name == "A"

    def test_cross_validate(self):
        from shared.core.classifier import ClassifierTrainer

        np.random.seed(42)
        X = np.random.randn(20, 3).astype(np.float32)
        y = (X[:, 0] > 0).astype(int)
        trainer = ClassifierTrainer(
            model_type="random_forest", class_names=["neg", "pos"]
        )
        cv = trainer.cross_validate(X, y, cv=3)
        assert "accuracy_mean" in cv
        assert 0.0 <= cv["accuracy_mean"] <= 1.0

    def test_model_save_load(self, tmp_path):
        from shared.core.classifier import ClassifierModel, ClassifierTrainer

        X = np.array([[0], [1], [2], [3]], dtype=np.float32)
        y = np.array([0, 0, 1, 1])
        trainer = ClassifierTrainer(
            model_type="knn", class_names=["A", "B"], n_neighbors=1
        )
        model = trainer.train(X, y)
        path = tmp_path / "model.pkl"
        model.save(path)
        loaded = ClassifierModel.load(path)
        assert loaded.model_type == "knn"
        assert loaded.class_names == ["A", "B"]

    def test_predict_batch(self):
        from shared.core.classifier import ClassifierInference, ClassifierTrainer

        X = np.array([[0], [1], [2], [3]], dtype=np.float32)
        y = np.array([0, 0, 1, 1])
        trainer = ClassifierTrainer(
            model_type="knn", class_names=["lo", "hi"], n_neighbors=1
        )
        model = trainer.train(X, y)
        inf = ClassifierInference(model)
        batch_results = inf.predict_batch(
            np.array([[0], [3]], dtype=np.float32)
        )
        assert len(batch_results) == 2
        assert batch_results[0].class_name == "lo"
        assert batch_results[1].class_name == "hi"


# ============================================================
# 4. Defect Grading
# ============================================================


class TestDefectGrading:
    """Tests for shared.core.defect_grading."""

    def test_grade_from_score_pass(self):
        from shared.core.defect_grading import DefectGrader, GradingConfig

        config = GradingConfig.default_config()
        grader = DefectGrader(config)
        result = grader.grade_from_score(0.95, defect_count=0, area_ratio=0.0)
        assert result.grade.name == "PASS"

    def test_grade_from_score_reject(self):
        from shared.core.defect_grading import DefectGrader, GradingConfig

        config = GradingConfig.default_config()
        grader = DefectGrader(config)
        result = grader.grade_from_score(0.05, defect_count=10, area_ratio=0.5)
        # Score 0.05 >= threshold 0.0 -> REJECT
        assert result.grade.name == "REJECT"

    def test_grade_from_rules(self):
        from shared.core.defect_grading import (
            DefectGrader,
            GradeLevel,
            GradeRule,
            GradingConfig,
        )

        rules = [
            GradeRule(
                field_name="anomaly_score",
                op=">",
                threshold=0.8,
                grade=GradeLevel.CRITICAL,
            ),
            GradeRule(
                field_name="defect_count",
                op=">=",
                threshold=3,
                grade=GradeLevel.MAJOR,
            ),
        ]
        config = GradingConfig(rules=rules, use_worst_grade=True)
        grader = DefectGrader(config)
        features = {"anomaly_score": 0.9, "defect_count": 5, "defect_area_ratio": 0.1}
        result = grader.grade_from_rules(features)
        assert result.grade == GradeLevel.CRITICAL

    def test_pareto_analysis(self):
        from shared.core.defect_grading import DefectGrader

        defects = [
            {"defect_type": "scratch"},
            {"defect_type": "dent"},
            {"defect_type": "scratch"},
            {"defect_type": "scratch"},
            {"defect_type": "stain"},
            {"defect_type": "dent"},
        ]
        pareto = DefectGrader.pareto_analysis(defects)
        assert pareto[0].defect_type == "scratch"
        assert pareto[0].count == 3

    def test_grading_config_json_roundtrip(self, tmp_path):
        from shared.core.defect_grading import GradingConfig

        config = GradingConfig.default_config()
        path = tmp_path / "grading.json"
        config.to_json(path)
        loaded = GradingConfig.from_json(path)
        assert loaded.score_thresholds == config.score_thresholds

    def test_grade_level_ordering(self):
        from shared.core.defect_grading import GradeLevel

        assert GradeLevel.PASS < GradeLevel.MINOR
        assert GradeLevel.MINOR < GradeLevel.CRITICAL
        assert GradeLevel.CRITICAL <= GradeLevel.REJECT


# ============================================================
# 5. Blob Analysis
# ============================================================


class TestBlobAnalysis:
    """Tests for shared.core.blob_analysis."""

    def test_extract_features(self):
        from shared.core.blob_analysis import extract_blob_features

        labels = np.zeros((50, 50), dtype=np.int32)
        labels[10:20, 10:20] = 1  # 10x10 square blob
        labels[30:40, 25:35] = 2  # 10x10 square blob
        gray = np.random.randint(50, 200, (50, 50), dtype=np.uint8)
        features = extract_blob_features(labels, gray)
        assert len(features) == 2
        assert features[0].area > 0
        assert features[0].hu_moments is not None
        assert len(features[0].hu_moments) == 7

    def test_feret_diameters(self):
        from shared.core.blob_analysis import compute_feret_diameters

        contour = np.array(
            [[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32
        ).reshape(-1, 1, 2)
        min_d, max_d, angle = compute_feret_diameters(contour)
        assert min_d > 0
        assert max_d >= min_d

    def test_euler_number_solid(self):
        from shared.core.blob_analysis import compute_euler_number

        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:40, 10:40] = 255  # solid square, no holes
        euler = compute_euler_number(mask)
        assert euler == 1  # 1 component, 0 holes

    def test_euler_number_with_hole(self):
        from shared.core.blob_analysis import compute_euler_number

        mask = np.zeros((60, 60), dtype=np.uint8)
        mask[5:55, 5:55] = 255
        mask[20:40, 20:40] = 0  # hole
        euler = compute_euler_number(mask)
        assert euler == 0  # 1 component - 1 hole = 0

    def test_select_blobs(self):
        from shared.core.blob_analysis import extract_blob_features, select_blobs

        labels = np.zeros((100, 100), dtype=np.int32)
        labels[10:20, 10:20] = 1   # 100 px area
        labels[40:80, 40:80] = 2   # 1600 px area
        features = extract_blob_features(labels)
        big = select_blobs(features, {"area": (">", 500)})
        assert len(big) == 1
        assert big[0].index == 2


# ============================================================
# 6. Vision 3D
# ============================================================


class TestVision3D:
    """Tests for shared.core.vision_3d."""

    def test_depth_to_point_cloud(self):
        from shared.core.vision_3d import depth_to_point_cloud

        depth = np.ones((10, 10), dtype=np.float32) * 5.0
        fx, fy, cx, cy = 100.0, 100.0, 5.0, 5.0
        cloud = depth_to_point_cloud(depth, fx, fy, cx, cy)
        assert cloud.xyz.shape[1] == 3
        assert cloud.xyz.shape[0] > 0

    def test_fit_plane_ransac(self):
        from shared.core.vision_3d import PointCloud, fit_plane_ransac

        # Perfect flat plane at z=0 with slight noise.
        rng = np.random.default_rng(42)
        x = rng.uniform(-1, 1, 200)
        y = rng.uniform(-1, 1, 200)
        z = np.zeros(200) + rng.normal(0, 0.001, 200)
        xyz = np.column_stack([x, y, z]).astype(np.float64)
        cloud = PointCloud(xyz=xyz)
        plane = fit_plane_ransac(cloud)
        # Normal should be close to (0, 0, 1) or (0, 0, -1).
        assert abs(abs(plane.normal[2]) - 1.0) < 0.1

    def test_ply_save_load(self, tmp_path):
        from shared.core.vision_3d import PointCloud, load_ply, save_ply

        xyz = np.random.randn(50, 3).astype(np.float64)
        cloud = PointCloud(xyz=xyz)
        path = tmp_path / "test.ply"
        save_ply(cloud, path)
        loaded = load_ply(path)
        np.testing.assert_allclose(loaded.xyz, xyz, atol=1e-4)

    def test_point_cloud_centroid(self):
        from shared.core.vision_3d import PointCloud

        xyz = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0]], dtype=np.float64)
        cloud = PointCloud(xyz=xyz)
        centroid = cloud.centroid
        np.testing.assert_allclose(centroid, [2 / 3, 2 / 3, 0], atol=1e-6)

    def test_point_cloud_bounding_box(self):
        from shared.core.vision_3d import PointCloud

        xyz = np.array([[0, 0, 0], [10, 5, 3]], dtype=np.float64)
        cloud = PointCloud(xyz=xyz)
        min_c, max_c = cloud.bounding_box()
        np.testing.assert_array_equal(min_c, [0, 0, 0])
        np.testing.assert_array_equal(max_c, [10, 5, 3])


# ============================================================
# 7. XLD Contour
# ============================================================


class TestXLDContour:
    """Tests for shared.core.xld_contour."""

    def test_xld_contour_creation(self):
        from shared.core.xld_contour import XLDContour

        pts = np.array([[0.5, 0.5], [1.5, 1.5], [2.5, 0.5]], dtype=np.float64)
        c = XLDContour(points=pts)
        assert len(c) == 3
        assert not c.is_closed

    def test_length_xld(self):
        from shared.core.xld_contour import XLDContour, length_xld

        pts = np.array([[0, 0], [3, 0], [3, 4]], dtype=np.float64)
        c = XLDContour(points=pts)
        length = length_xld(c)
        # seg1 = 3, seg2 = sqrt((3-3)^2 + (4-0)^2) = 4. Total = 7.
        assert abs(length - 7.0) < 0.01

    def test_smooth_contours_xld(self):
        from shared.core.xld_contour import XLDContour, XLDContourSet, smooth_contours_xld

        pts = np.array(
            [[0, 0], [1, 2], [2, 0], [3, 2], [4, 0]], dtype=np.float64
        )
        c = XLDContour(points=pts)
        cs = XLDContourSet(contours=[c])
        smoothed_set = smooth_contours_xld(cs, sigma=1.0)
        assert len(smoothed_set) == 1
        assert len(smoothed_set[0]) == len(c)

    def test_fit_circle(self):
        from shared.core.xld_contour import XLDContour, fit_circle_contour_xld

        angles = np.linspace(0, 2 * np.pi, 36, endpoint=False)
        pts = np.column_stack(
            [50 + 10 * np.cos(angles), 50 + 10 * np.sin(angles)]
        )
        c = XLDContour(points=pts)
        cx, cy, radius, residual = fit_circle_contour_xld(c)
        assert abs(cx - 50) < 1.0
        assert abs(cy - 50) < 1.0
        assert abs(radius - 10) < 1.0

    def test_area_center_xld(self):
        from shared.core.xld_contour import XLDContour, area_center_xld

        # Unit square (0,0)->(1,0)->(1,1)->(0,1) counter-clockwise.
        pts = np.array(
            [[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64
        )
        c = XLDContour(points=pts, is_closed=True)
        area, cx, cy = area_center_xld(c)
        assert abs(abs(area) - 1.0) < 0.01
        assert abs(cx - 0.5) < 0.1
        assert abs(cy - 0.5) < 0.1

    def test_circularity_xld(self):
        from shared.core.xld_contour import XLDContour, circularity_xld

        # Circle contour should have circularity close to 1.0.
        angles = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        pts = np.column_stack([np.cos(angles), np.sin(angles)]) * 50 + 100
        c = XLDContour(points=pts, is_closed=True)
        circ = circularity_xld(c)
        assert 0.9 < circ < 1.1  # discrete sampling may exceed 1.0 slightly


# ============================================================
# 8. Hand-Eye Calibration
# ============================================================


class TestHandEyeCalibration:
    """Tests for shared.core.hand_eye_calibration."""

    def test_euler_roundtrip(self):
        from shared.core.hand_eye_calibration import (
            euler_to_rotation_matrix,
            rotation_matrix_to_euler,
        )

        roll, pitch, yaw = 0.1, 0.2, 0.3
        R = euler_to_rotation_matrix(roll, pitch, yaw)
        r2, p2, y2 = rotation_matrix_to_euler(R)
        np.testing.assert_allclose([roll, pitch, yaw], [r2, p2, y2], atol=1e-10)

    def test_quaternion_roundtrip(self):
        from shared.core.hand_eye_calibration import (
            euler_to_rotation_matrix,
            quaternion_to_rotation_matrix,
            rotation_matrix_to_quaternion,
        )

        R = euler_to_rotation_matrix(0.5, -0.3, 1.2)
        q = rotation_matrix_to_quaternion(R)
        R2 = quaternion_to_rotation_matrix(*q)
        np.testing.assert_allclose(R, R2, atol=1e-10)

    def test_chain_transforms(self):
        from shared.core.hand_eye_calibration import chain_transforms, invert_transform

        T1 = np.eye(4)
        T1[:3, 3] = [1, 0, 0]
        T2 = np.eye(4)
        T2[:3, 3] = [0, 2, 0]
        chained = chain_transforms(T1, T2)
        np.testing.assert_allclose(chained[:3, 3], [1, 2, 0])

    def test_invert_transform(self):
        from shared.core.hand_eye_calibration import invert_transform

        T = np.eye(4)
        T[:3, 3] = [1, 0, 0]
        inv = invert_transform(T)
        np.testing.assert_allclose(inv[:3, 3], [-1, 0, 0])

    def test_rotation_matrix_is_orthonormal(self):
        from shared.core.hand_eye_calibration import euler_to_rotation_matrix

        R = euler_to_rotation_matrix(0.5, -0.3, 1.2)
        # R^T @ R should be identity.
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
        # det(R) should be +1.
        assert abs(np.linalg.det(R) - 1.0) < 1e-12


# ============================================================
# 9. Parallel Ops
# ============================================================


class TestParallelOps:
    """Tests for shared.core.parallel_ops."""

    def test_map_basic(self):
        from shared.core.parallel_ops import ParallelExecutor

        executor = ParallelExecutor(max_workers=2)
        images = [np.full((10, 10), v, dtype=np.uint8) for v in range(5)]
        results = executor.map(lambda img: float(img.mean()), images)
        assert len(results) == 5
        for i, r in enumerate(results):
            assert abs(r - i) < 0.01

    def test_par_start_join(self):
        from shared.core.parallel_ops import ParallelExecutor

        executor = ParallelExecutor(max_workers=2)
        futures = [executor.par_start(lambda x: x * 2, i) for i in range(3)]
        results = executor.par_join(futures)
        assert results == [0, 2, 4]

    def test_map_empty(self):
        from shared.core.parallel_ops import ParallelExecutor

        executor = ParallelExecutor(max_workers=2)
        results = executor.map(lambda x: x, [])
        assert results == []

    def test_last_stats(self):
        from shared.core.parallel_ops import ParallelExecutor

        executor = ParallelExecutor(max_workers=2)
        executor.map(lambda x: x, [1, 2, 3])
        stats = executor.last_stats
        assert stats.total_items == 3
        assert stats.errors == 0


# ============================================================
# 10. Metrology Advanced
# ============================================================


class TestMetrologyAdvanced:
    """Tests for shared.core.metrology_advanced."""

    def test_measure_parallelism(self):
        from shared.core.metrology_advanced import measure_parallelism

        line1 = (0, 0, 10, 0)
        line2 = (0, 5, 10, 5)
        result = measure_parallelism(line1, line2)
        assert abs(result.value) < 0.01  # perfectly parallel
        assert result.tolerance_type == "parallelism"

    def test_measure_perpendicularity(self):
        from shared.core.metrology_advanced import measure_perpendicularity

        line1 = (0, 0, 10, 0)
        line2 = (5, 0, 5, 10)
        result = measure_perpendicularity(line1, line2)
        assert abs(result.value) < 0.01  # perfectly perpendicular

    def test_measure_roundness(self):
        from shared.core.metrology_advanced import measure_roundness

        angles = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        pts = np.column_stack([10 * np.cos(angles), 10 * np.sin(angles)])
        result = measure_roundness(pts)
        assert result.value < 0.5  # very round

    def test_measure_straightness(self):
        from shared.core.metrology_advanced import measure_straightness

        pts = np.column_stack([np.linspace(0, 10, 50), np.zeros(50)])
        result = measure_straightness(pts)
        assert result.value < 0.01  # perfectly straight

    def test_measure_concentricity(self):
        from shared.core.metrology_advanced import measure_concentricity

        result = measure_concentricity((10, 10, 5), (10, 10, 3))
        assert abs(result.value) < 0.01  # perfectly concentric

    def test_measure_symmetry(self):
        from shared.core.metrology_advanced import measure_symmetry

        # Symmetric rectangle about the x-axis.
        pts = np.array(
            [[-5, -2], [5, -2], [5, 2], [-5, 2]], dtype=np.float64
        )
        result = measure_symmetry(pts, axis_point=(0, 0), axis_direction=(1, 0))
        assert result.value < 0.5  # near-symmetric

    def test_gdt_result_pass_check(self):
        from shared.core.metrology_advanced import GDT_Result

        r = GDT_Result(tolerance_type="test", value=0.5, tolerance=1.0)
        assert r.is_pass is True
        r2 = GDT_Result(tolerance_type="test", value=2.0, tolerance=1.0)
        assert r2.is_pass is False


# ============================================================
# 11. Deformable Matching
# ============================================================


class TestDeformableMatching:
    """Tests for shared.core.deformable_matching."""

    def test_create_model(self):
        from shared.core.deformable_matching import create_deformable_model

        template = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        cv2.rectangle(template, (20, 20), (80, 80), 255, 2)
        cv2.circle(template, (50, 50), 20, 200, 2)
        cv2.line(template, (10, 50), (90, 50), 128, 2)
        cv2.line(template, (50, 10), (50, 90), 128, 2)
        model = create_deformable_model(template, feature_type="orb")
        assert model.template is not None
        assert len(model.keypoints) > 0

    def test_model_has_descriptors(self):
        from shared.core.deformable_matching import create_deformable_model

        template = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(template, (10, 10), (90, 90), 255, 3)
        cv2.line(template, (10, 50), (90, 50), 200, 2)
        cv2.line(template, (50, 10), (50, 90), 200, 2)
        cv2.circle(template, (30, 30), 15, 180, 2)
        cv2.circle(template, (70, 70), 15, 180, 2)
        model = create_deformable_model(template, feature_type="orb")
        assert model.descriptors is not None
        assert model.descriptors.shape[0] > 0

    def test_deformable_model_dataclass(self):
        from shared.core.deformable_matching import DeformableModel, FeatureType

        model = DeformableModel(
            template=np.zeros((10, 10), dtype=np.uint8),
            keypoints=np.array([[5.0, 5.0]]),
            descriptors=np.zeros((1, 32), dtype=np.uint8),
            feature_type=FeatureType.ORB,
        )
        assert model.feature_type == FeatureType.ORB


# ============================================================
# 12. Stereo 3D
# ============================================================


class TestStereo3D:
    """Tests for shared.core.stereo_3d."""

    def test_focus_measure(self):
        from shared.core.stereo_3d import focus_measure

        sharp = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        blurry = cv2.GaussianBlur(sharp, (15, 15), 5)
        score_sharp = focus_measure(sharp, method="laplacian_variance")
        score_blurry = focus_measure(blurry, method="laplacian_variance")
        assert score_sharp > score_blurry

    def test_find_best_focus(self):
        from shared.core.stereo_3d import find_best_focus

        base = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        images = [cv2.GaussianBlur(base, (k, k), 0) for k in [15, 7, 3, 7, 15]]
        best_idx, best_pos, best_score = find_best_focus(images)
        assert best_idx == 2  # least blurry (kernel size 3)

    def test_disparity_to_depth(self):
        from shared.core.stereo_3d import disparity_to_depth

        disp = np.full((10, 10), 16.0, dtype=np.float32)
        depth = disparity_to_depth(disp, baseline=0.1, focal_length=500)
        expected = 0.1 * 500 / 16.0
        np.testing.assert_allclose(depth, expected, atol=0.01)

    def test_disparity_to_depth_invalid(self):
        from shared.core.stereo_3d import disparity_to_depth

        disp = np.array([[0.0, 10.0], [5.0, -1.0]], dtype=np.float32)
        depth = disparity_to_depth(disp, baseline=1.0, focal_length=100.0)
        assert depth[0, 0] == 0.0  # invalid disparity -> 0
        assert depth[1, 1] == 0.0  # negative disparity -> 0
        assert depth[0, 1] > 0     # valid disparity
        assert depth[1, 0] > 0     # valid disparity


# ============================================================
# 13. OCR Trainer
# ============================================================


class TestOCRTrainer:
    """Tests for shared.core.ocr_trainer."""

    def test_character_extractor(self):
        from shared.core.ocr_trainer import CharacterExtractor

        # Create image with characters drawn on it.
        img = np.zeros((50, 200), dtype=np.uint8)
        cv2.putText(img, "ABC", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2)
        extractor = CharacterExtractor()
        chars = extractor.extract_characters(img)
        assert len(chars) >= 1

    def test_character_extractor_sorted_ltr(self):
        from shared.core.ocr_trainer import CharacterExtractor

        img = np.zeros((60, 300), dtype=np.uint8)
        cv2.putText(img, "XY", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 255, 3)
        extractor = CharacterExtractor()
        chars = extractor.extract_characters(img)
        if len(chars) >= 2:
            # Should be sorted left-to-right.
            assert chars[0].bbox[0] <= chars[1].bbox[0]

    def test_ocr_training_config_defaults(self):
        from shared.core.ocr_trainer import BackendType, OCRTrainingConfig

        config = OCRTrainingConfig()
        assert config.image_size == (32, 32)
        assert config.backend == BackendType.HOG_KNN


# ============================================================
# 14. Motion Interface
# ============================================================


class TestMotionInterface:
    """Tests for shared.core.motion_interface."""

    def test_simulated_motion(self):
        from shared.core.motion_interface import (
            AxisPosition,
            MotionConfig,
            SimulatedMotionInterface,
        )

        config = MotionConfig(protocol="simulated", host="localhost", port=0)
        sim = SimulatedMotionInterface(config)
        sim.connect()
        assert sim.is_connected
        pos = sim.get_position()
        assert isinstance(pos, AxisPosition)
        sim.move_to(AxisPosition(x=10, y=20, z=0), speed=100)
        sim.wait_motion_complete(timeout=5.0)
        new_pos = sim.get_position()
        assert abs(new_pos.x - 10) < 0.1
        assert abs(new_pos.y - 20) < 0.1
        sim.disconnect()
        assert not sim.is_connected

    def test_axis_position_distance(self):
        from shared.core.motion_interface import AxisPosition

        a = AxisPosition(x=0, y=0, z=0)
        b = AxisPosition(x=3, y=4, z=0)
        assert abs(a.distance_to(b) - 5.0) < 0.01

    def test_axis_position_to_from_array(self):
        from shared.core.motion_interface import AxisPosition

        p = AxisPosition(x=1, y=2, z=3, rx=4, ry=5, rz=6)
        arr = p.to_array()
        assert arr.shape == (6,)
        p2 = AxisPosition.from_array(arr)
        assert abs(p2.x - 1.0) < 1e-10
        assert abs(p2.rz - 6.0) < 1e-10


# ============================================================
# 15. Instance Segmentation
# ============================================================


class TestInstanceSegmentation:
    """Tests for shared.core.instance_segmentation (utility functions only)."""

    def test_masks_to_labels(self):
        from shared.core.instance_segmentation import masks_to_labels

        m1 = np.zeros((10, 10), dtype=np.uint8)
        m1[2:5, 2:5] = 255
        m2 = np.zeros((10, 10), dtype=np.uint8)
        m2[6:9, 6:9] = 255
        labels = masks_to_labels([m1, m2])
        assert labels[3, 3] == 1
        assert labels[7, 7] == 2
        assert labels[0, 0] == 0

    def test_compute_iou_identical(self):
        from shared.core.instance_segmentation import compute_iou

        m1 = np.zeros((10, 10), dtype=np.uint8)
        m1[2:8, 2:8] = 1
        assert abs(compute_iou(m1, m1.copy()) - 1.0) < 0.01

    def test_compute_iou_disjoint(self):
        from shared.core.instance_segmentation import compute_iou

        m1 = np.zeros((10, 10), dtype=np.uint8)
        m1[0:3, 0:3] = 1
        m2 = np.zeros((10, 10), dtype=np.uint8)
        m2[7:10, 7:10] = 1
        assert compute_iou(m1, m2) == 0.0

    def test_compute_iou_empty(self):
        from shared.core.instance_segmentation import compute_iou

        m1 = np.zeros((10, 10), dtype=np.uint8)
        m2 = np.zeros((10, 10), dtype=np.uint8)
        assert compute_iou(m1, m2) == 0.0

    def test_segmentation_result_filter(self):
        from shared.core.instance_segmentation import SegmentationResult

        sr = SegmentationResult(
            masks=[
                np.zeros((10, 10), dtype=np.uint8),
                np.zeros((10, 10), dtype=np.uint8),
            ],
            class_ids=[0, 1],
            class_names=["cat", "dog"],
            scores=[0.9, 0.3],
        )
        filtered = sr.filter_by_score(0.5)
        assert filtered.count == 1
        assert filtered.class_names[0] == "cat"


# ============================================================
# 16. Pyramid ROI
# ============================================================


class TestPyramidROI:
    """Tests for shared.core.pyramid_roi."""

    def test_gen_pyramid_rois(self):
        from shared.core.pyramid_roi import gen_pyramid_rois

        rois = gen_pyramid_rois((640, 480), levels=4, roi=(100, 100, 200, 200))
        assert 0 in rois.level_rois
        assert 3 in rois.level_rois
        # Level 0 should match original (possibly clamped).
        assert rois.level_rois[0] == (100, 100, 200, 200)

    def test_gen_pyramid_rois_scaling(self):
        from shared.core.pyramid_roi import gen_pyramid_rois

        rois = gen_pyramid_rois((640, 480), levels=3, roi=(0, 0, 640, 480))
        # Level 1 should be roughly half.
        r1 = rois.level_rois[1]
        assert r1[2] <= 320 + 1  # width at level 1
        assert r1[3] <= 240 + 1  # height at level 1

    def test_gen_pyramid_rois_single_level(self):
        from shared.core.pyramid_roi import gen_pyramid_rois

        rois = gen_pyramid_rois((100, 100), levels=1, roi=(10, 10, 50, 50))
        assert rois.num_levels == 1

    def test_pyramid_roi_to_dict(self):
        from shared.core.pyramid_roi import PyramidROI

        pr = PyramidROI(level_rois={0: (0, 0, 100, 100), 1: (0, 0, 50, 50)})
        d = pr.to_dict()
        assert d["num_levels"] == 2


# ============================================================
# 17. Genicam Interface (stub/mock tests)
# ============================================================


class TestGenicamInterface:
    """Tests for shared.core.genicam_interface (stub mode)."""

    def test_device_dataclass(self):
        from shared.core.genicam_interface import GenICamDevice

        dev = GenICamDevice(serial="123", model="Test", vendor="TestVendor")
        assert dev.serial == "123"
        assert dev.model == "Test"
        assert dev.ip == ""

    def test_manager_stub_mode(self):
        """When harvesters is not installed, manager should work in stub mode."""
        from shared.core.genicam_interface import GenICamManager

        mgr = GenICamManager()
        devices = mgr.discover_devices()
        assert isinstance(devices, list)

    def test_feature_type_enum(self):
        from shared.core.genicam_interface import FeatureType

        assert FeatureType.INTEGER.value == "Integer"
        assert FeatureType.FLOAT.value == "Float"

    def test_feature_info_dataclass(self):
        from shared.core.genicam_interface import FeatureInfo, FeatureType

        info = FeatureInfo(
            name="ExposureTime",
            feature_type=FeatureType.FLOAT,
            value=5000.0,
            writable=True,
            description="Camera exposure time",
        )
        assert info.name == "ExposureTime"
        assert info.value == 5000.0


# ============================================================
# 18. IO Interface (Digital I/O abstraction)
# ============================================================


class TestIOInterface:
    """Tests for shared.core.io_interface (if present)."""

    def test_import_io_interface(self):
        """Verify the io_interface module can be imported."""
        import shared.core.io_interface  # noqa: F401

    def test_io_interface_dataclasses(self):
        """Verify basic data classes exist and can be instantiated."""
        try:
            from shared.core.io_interface import IOConfig

            config = IOConfig()
            assert config is not None
        except (ImportError, AttributeError):
            pytest.skip("IOConfig not available in io_interface")
