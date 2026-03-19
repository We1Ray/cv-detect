"""Tests for shared.core.calibration -- camera calibration and measurement."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from shared.core.calibration import (
    CalibrationResult,
    WorldMapping,
    calibrate_from_known_distance,
    load_calibration,
    measure_area_mm2,
    measure_distance_mm,
    measure_length_mm,
    pixel_to_world,
    save_calibration,
    world_to_pixel,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def simple_mapping() -> WorldMapping:
    """Isotropic 10 px/mm mapping with no rotation and origin at (0,0)."""
    return WorldMapping(
        pixels_per_mm_x=10.0,
        pixels_per_mm_y=10.0,
        pixels_per_mm=10.0,
        origin_px=(0.0, 0.0),
        rotation_deg=0.0,
        method="known_distance",
    )


@pytest.fixture
def rotated_mapping() -> WorldMapping:
    """Isotropic 10 px/mm mapping rotated 90 degrees."""
    return WorldMapping(
        pixels_per_mm_x=10.0,
        pixels_per_mm_y=10.0,
        pixels_per_mm=10.0,
        origin_px=(0.0, 0.0),
        rotation_deg=90.0,
        method="known_distance",
    )


@pytest.fixture
def sample_calibration_result() -> CalibrationResult:
    """Minimal CalibrationResult for persistence tests."""
    return CalibrationResult(
        camera_matrix=np.eye(3, dtype=np.float64),
        dist_coeffs=np.zeros(5, dtype=np.float64),
        rvecs=[np.zeros(3, dtype=np.float64)],
        tvecs=[np.array([1.0, 2.0, 3.0], dtype=np.float64)],
        rms_error=0.25,
        image_size=(640, 480),
        num_images=1,
        pattern_size=(9, 6),
        square_size=25.0,
    )


# ======================================================================
# TestKnownDistanceCalibration
# ======================================================================


class TestKnownDistanceCalibration:
    """Tests for calibrate_from_known_distance."""

    def test_basic_mapping(self) -> None:
        """100 pixels over 10 mm should give 10 px/mm."""
        mapping = calibrate_from_known_distance(100.0, 10.0)
        assert mapping.pixels_per_mm == pytest.approx(10.0)

    def test_invalid_pixel_distance(self) -> None:
        """pixel_distance <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="pixel_distance"):
            calibrate_from_known_distance(0.0, 10.0)
        with pytest.raises(ValueError, match="pixel_distance"):
            calibrate_from_known_distance(-5.0, 10.0)

    def test_invalid_real_distance(self) -> None:
        """real_distance_mm <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="real_distance_mm"):
            calibrate_from_known_distance(100.0, 0.0)
        with pytest.raises(ValueError, match="real_distance_mm"):
            calibrate_from_known_distance(100.0, -1.0)

    def test_invalid_axis(self) -> None:
        """An unsupported axis string should raise ValueError."""
        with pytest.raises(ValueError, match="axis"):
            calibrate_from_known_distance(100.0, 10.0, axis="z")

    def test_mapping_attributes(self) -> None:
        """Returned mapping should have correct pixels_per_mm and method."""
        mapping = calibrate_from_known_distance(200.0, 50.0, axis="x")
        assert mapping.pixels_per_mm == pytest.approx(4.0)
        assert mapping.pixels_per_mm_x == pytest.approx(4.0)
        assert mapping.pixels_per_mm_y == pytest.approx(4.0)
        assert mapping.method == "known_distance"
        assert mapping.origin_px == (0.0, 0.0)
        assert mapping.rotation_deg == 0.0


# ======================================================================
# TestCoordinateConversion
# ======================================================================


class TestCoordinateConversion:
    """Tests for pixel_to_world and world_to_pixel."""

    def test_pixel_to_world_origin(self, simple_mapping: WorldMapping) -> None:
        """Pixel at the origin should map to world (0, 0)."""
        wx, wy = pixel_to_world(0.0, 0.0, simple_mapping)
        assert wx == pytest.approx(0.0)
        assert wy == pytest.approx(0.0)

    def test_pixel_to_world_simple(self, simple_mapping: WorldMapping) -> None:
        """100 pixels at 10 px/mm should be 10 mm."""
        wx, wy = pixel_to_world(100.0, 0.0, simple_mapping)
        assert wx == pytest.approx(10.0)
        assert wy == pytest.approx(0.0)

    def test_world_to_pixel_roundtrip(self, simple_mapping: WorldMapping) -> None:
        """pixel_to_world then world_to_pixel should return the original."""
        orig_px, orig_py = 73.5, 42.1
        wx, wy = pixel_to_world(orig_px, orig_py, simple_mapping)
        px, py = world_to_pixel(wx, wy, simple_mapping)
        assert px == pytest.approx(orig_px, abs=1e-6)
        assert py == pytest.approx(orig_py, abs=1e-6)

    def test_rotated_mapping(self, rotated_mapping: WorldMapping) -> None:
        """With 90-degree rotation, axes should effectively swap."""
        # Moving 100 px along pixel-X with a 90-deg rotation:
        # rotation_deg=90 means world axes are rotated 90 deg relative to image.
        # In pixel_to_world, theta = -rotation_deg = -90 deg.
        # dx=100, dy=0 -> rx = 100*cos(-90) - 0*sin(-90) = 0
        #                 ry = 100*sin(-90) + 0*cos(-90) = -100
        # world_x = rx / ppm_x = 0, world_y = ry / ppm_y = -10
        wx, wy = pixel_to_world(100.0, 0.0, rotated_mapping)
        assert wx == pytest.approx(0.0, abs=1e-6)
        assert wy == pytest.approx(-10.0, abs=1e-6)

    @pytest.mark.parametrize(
        "px, py",
        [(0.0, 0.0), (50.0, 50.0), (123.4, 567.8), (-10.0, -20.0)],
    )
    def test_roundtrip_parametrized(
        self, simple_mapping: WorldMapping, px: float, py: float,
    ) -> None:
        """Round-trip conversion should preserve coordinates for various inputs."""
        wx, wy = pixel_to_world(px, py, simple_mapping)
        rpx, rpy = world_to_pixel(wx, wy, simple_mapping)
        assert rpx == pytest.approx(px, abs=1e-6)
        assert rpy == pytest.approx(py, abs=1e-6)


# ======================================================================
# TestMeasurements
# ======================================================================


class TestMeasurements:
    """Tests for measure_distance_mm, measure_area_mm2, measure_length_mm."""

    def test_distance_mm(self, simple_mapping: WorldMapping) -> None:
        """Two points 100 px apart at 10 px/mm should be 10 mm."""
        dist = measure_distance_mm(0.0, 0.0, 100.0, 0.0, simple_mapping)
        assert dist == pytest.approx(10.0)

    def test_distance_mm_diagonal(self, simple_mapping: WorldMapping) -> None:
        """Diagonal distance should follow Pythagorean theorem."""
        dist = measure_distance_mm(0.0, 0.0, 30.0, 40.0, simple_mapping)
        # 30px = 3mm, 40px = 4mm -> sqrt(9+16) = 5mm
        assert dist == pytest.approx(5.0)

    def test_area_mm2_mask(self, simple_mapping: WorldMapping) -> None:
        """A 100x100 binary mask at 10 px/mm should be 100 mm^2."""
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[50:150, 50:150] = 255
        area = measure_area_mm2(mask, simple_mapping)
        # 100*100 = 10000 non-zero pixels, ppm_x * ppm_y = 100
        assert area == pytest.approx(100.0)

    def test_area_mm2_contour(self, simple_mapping: WorldMapping) -> None:
        """A rectangular contour should give correct area."""
        # Rectangle contour: 100px x 50px
        contour = np.array(
            [[0, 0], [100, 0], [100, 50], [0, 50]], dtype=np.int32,
        )
        area = measure_area_mm2(contour, simple_mapping)
        # 5000 px^2 / 100 px^2/mm^2 = 50 mm^2
        assert area == pytest.approx(50.0)

    def test_length_mm_contour(self, simple_mapping: WorldMapping) -> None:
        """A straight-line contour of 100px at 10 px/mm should be 10 mm."""
        contour = np.array([[0, 0], [100, 0]], dtype=np.int32)
        length = measure_length_mm(contour, simple_mapping)
        assert length == pytest.approx(10.0)

    def test_length_mm_closed_contour(self, simple_mapping: WorldMapping) -> None:
        """Arc length of a known contour should convert correctly."""
        # A simple right-angle path: 0,0 -> 30,0 -> 30,40
        contour = np.array([[0, 0], [30, 0], [30, 40]], dtype=np.int32)
        length = measure_length_mm(contour, simple_mapping)
        # 30 + 40 = 70 px, /10 = 7 mm
        assert length == pytest.approx(7.0)


# ======================================================================
# TestCalibrationPersistence
# ======================================================================


class TestCalibrationPersistence:
    """Tests for save_calibration and load_calibration."""

    def test_save_load_roundtrip(
        self,
        tmp_path,
        sample_calibration_result: CalibrationResult,
    ) -> None:
        """Saving and loading should preserve all calibration fields."""
        path = tmp_path / "calibration.json"
        save_calibration(sample_calibration_result, path)
        loaded = load_calibration(path)

        np.testing.assert_array_almost_equal(
            loaded.camera_matrix, sample_calibration_result.camera_matrix,
        )
        np.testing.assert_array_almost_equal(
            loaded.dist_coeffs, sample_calibration_result.dist_coeffs,
        )
        assert loaded.rms_error == pytest.approx(
            sample_calibration_result.rms_error,
        )
        assert loaded.image_size == sample_calibration_result.image_size
        assert loaded.num_images == sample_calibration_result.num_images
        assert loaded.pattern_size == sample_calibration_result.pattern_size
        assert loaded.square_size == pytest.approx(
            sample_calibration_result.square_size,
        )
        assert len(loaded.rvecs) == len(sample_calibration_result.rvecs)
        assert len(loaded.tvecs) == len(sample_calibration_result.tvecs)
        for rv_loaded, rv_orig in zip(
            loaded.rvecs, sample_calibration_result.rvecs,
        ):
            np.testing.assert_array_almost_equal(rv_loaded, rv_orig)
        for tv_loaded, tv_orig in zip(
            loaded.tvecs, sample_calibration_result.tvecs,
        ):
            np.testing.assert_array_almost_equal(tv_loaded, tv_orig)

    def test_load_nonexistent(self, tmp_path) -> None:
        """Loading from a nonexistent path should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_calibration(tmp_path / "no_such_file.json")
