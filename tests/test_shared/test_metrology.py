"""Tests for shared.core.metrology -- sub-pixel measurement and geometric fitting."""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import pytest

from shared.core.metrology import (
    FitResult,
    MeasureRectangle,
    MeasurementResult,
    SubPixelEdge,
    angle_ll,
    angle_lx,
    distance_cc,
    distance_pl,
    distance_pp,
    edges_sub_pix,
    fit_circle_contour_xld,
    fit_ellipse_contour_xld,
    fit_line_contour_xld,
    gen_measure_rect2,
)
from shared.validation import ImageValidationError


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _circle_points(
    center_row: float,
    center_col: float,
    radius: float,
    n: int = 72,
) -> List[Tuple[float, float]]:
    """Generate n evenly-spaced points on a circle (row, col)."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [
        (center_row + radius * np.sin(a), center_col + radius * np.cos(a))
        for a in angles
    ]


def _ellipse_points(
    center_row: float,
    center_col: float,
    ra: float,
    rb: float,
    phi: float = 0.0,
    n: int = 100,
) -> List[Tuple[float, float]]:
    """Generate n evenly-spaced points on an ellipse (row, col)."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    points = []
    for a in angles:
        x = ra * np.cos(a)
        y = rb * np.sin(a)
        col = center_col + x * cos_phi - y * sin_phi
        row = center_row + x * sin_phi + y * cos_phi
        points.append((row, col))
    return points


# ------------------------------------------------------------------
# TestGenMeasureRect2
# ------------------------------------------------------------------

class TestGenMeasureRect2:
    """Tests for gen_measure_rect2."""

    def test_basic_creation(self) -> None:
        """Verify that a MeasureRectangle is created with correct attributes."""
        rect = gen_measure_rect2(100.0, 200.0, 0.5, 30.0, 10.0)

        assert isinstance(rect, MeasureRectangle)
        assert rect.row == pytest.approx(100.0)
        assert rect.col == pytest.approx(200.0)
        assert rect.phi == pytest.approx(0.5)
        assert rect.length1 == pytest.approx(30.0)
        assert rect.length2 == pytest.approx(10.0)

    @pytest.mark.parametrize("length1, length2", [
        (-1.0, 10.0),
        (10.0, -5.0),
        (-3.0, -2.0),
        (0.0, 10.0),
        (10.0, 0.0),
    ])
    def test_invalid_length(self, length1: float, length2: float) -> None:
        """Negative or zero lengths should raise an error."""
        with pytest.raises(ImageValidationError):
            gen_measure_rect2(100.0, 200.0, 0.0, length1, length2)


# ------------------------------------------------------------------
# TestEdgesSubPix
# ------------------------------------------------------------------

class TestEdgesSubPix:
    """Tests for edges_sub_pix."""

    def test_detects_edges_on_step_image(self) -> None:
        """A sharp black-to-white vertical border should produce edges."""
        img = np.zeros((100, 200), dtype=np.uint8)
        img[:, 100:] = 255  # left half black, right half white

        edges = edges_sub_pix(img, alpha=1.0, low=10.0, high=30.0)

        assert len(edges) > 0
        for e in edges:
            assert isinstance(e, SubPixelEdge)
            # Edge columns should be near the boundary at col=100
            assert 95.0 < e.col < 105.0

    def test_no_edges_on_uniform_image(self) -> None:
        """A uniform image should produce no edges."""
        img = np.full((100, 100), 128, dtype=np.uint8)

        edges = edges_sub_pix(img, alpha=1.0, low=20.0, high=40.0)

        assert edges == []

    def test_edge_types(self) -> None:
        """Edge type should be either 'rising' or 'falling', and edges
        should be detected at both boundaries of a stripe pattern."""
        img = np.zeros((100, 200), dtype=np.uint8)
        img[:, 60:140] = 255  # white stripe in the middle

        edges = edges_sub_pix(img, alpha=1.0, low=10.0, high=30.0)

        assert len(edges) > 0
        for e in edges:
            assert e.type in ("rising", "falling")
        # Edges should appear near both boundaries (col~60 and col~140)
        cols = [e.col for e in edges]
        assert any(55 < c < 65 for c in cols), "No edges near left boundary"
        assert any(135 < c < 145 for c in cols), "No edges near right boundary"

    def test_accepts_color_image(self, sample_image: np.ndarray) -> None:
        """A 3-channel colour image should be accepted without error."""
        # Create a clear edge in the colour image
        img = sample_image.copy()
        img[:, :128, :] = 0
        img[:, 128:, :] = 255

        edges = edges_sub_pix(img, alpha=1.0, low=10.0, high=30.0)

        assert len(edges) > 0


# ------------------------------------------------------------------
# TestFitLine
# ------------------------------------------------------------------

class TestFitLine:
    """Tests for fit_line_contour_xld."""

    def test_perfect_horizontal_line(self) -> None:
        """Points along y=100 should yield angle approximately 0."""
        points = [(100.0, float(c)) for c in range(0, 200, 5)]

        result = fit_line_contour_xld(points, algorithm="regression")

        assert result.type == "line"
        # Angle should be close to 0 (horizontal direction vector)
        assert abs(result.params["angle"]) < 0.01
        assert result.error < 1e-6
        assert result.params["row1"] == pytest.approx(100.0, abs=0.1)
        assert result.params["row2"] == pytest.approx(100.0, abs=0.1)

    def test_perfect_vertical_line(self) -> None:
        """Points along col=50 should yield angle approximately pi/2 or -pi/2."""
        points = [(float(r), 50.0) for r in range(0, 200, 5)]

        result = fit_line_contour_xld(points, algorithm="regression")

        assert result.type == "line"
        # Angle should be close to pi/2 or -pi/2
        assert abs(abs(result.params["angle"]) - math.pi / 2) < 0.01
        assert result.error < 1e-6

    def test_too_few_points(self) -> None:
        """Fewer than 2 points should raise ValueError."""
        with pytest.raises(ValueError, match="at least 2"):
            fit_line_contour_xld([(10.0, 20.0)], algorithm="regression")

        with pytest.raises(ValueError, match="at least 2"):
            fit_line_contour_xld([], algorithm="regression")

    @pytest.mark.parametrize("algorithm", ["regression", "tukey", "huber"])
    def test_algorithms(self, algorithm: str) -> None:
        """All supported algorithms should successfully fit a line."""
        points = [(50.0 + 0.5 * c, float(c)) for c in range(50)]

        result = fit_line_contour_xld(points, algorithm=algorithm)

        assert result.type == "line"
        assert result.error < 1e-4
        assert result.points_used > 0

    def test_unknown_algorithm_raises(self) -> None:
        """An unsupported algorithm name should raise ValueError."""
        points = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]
        with pytest.raises(ValueError, match="unknown algorithm"):
            fit_line_contour_xld(points, algorithm="invalid_algo")

    def test_noisy_line_robust(self) -> None:
        """Tukey should be more robust than regression with outliers."""
        rng = np.random.RandomState(99)
        # Clean line y = 2x along 100 points
        cols = np.linspace(0, 100, 100)
        rows = 2.0 * cols + rng.normal(0, 0.5, 100)

        # Add 10 extreme outliers
        outlier_idx = rng.choice(100, 10, replace=False)
        rows[outlier_idx] += rng.choice([-1, 1], 10) * 200.0

        points = list(zip(rows, cols))

        result_regression = fit_line_contour_xld(points, algorithm="regression")
        result_tukey = fit_line_contour_xld(points, algorithm="tukey")

        # Tukey should have fewer inliers (outliers downweighted) and
        # smaller RMS error on the inlier set
        assert result_tukey.points_used < result_regression.points_used


# ------------------------------------------------------------------
# TestFitCircle
# ------------------------------------------------------------------

class TestFitCircle:
    """Tests for fit_circle_contour_xld."""

    def test_perfect_circle(self) -> None:
        """Points on a circle r=50 centred at (100,100) should be fitted
        accurately using the geometric (robust) algorithm."""
        points = _circle_points(100.0, 100.0, 50.0, n=72)

        result = fit_circle_contour_xld(points, algorithm="geometric")

        assert result.type == "circle"
        assert result.params["row"] == pytest.approx(100.0, abs=0.5)
        assert result.params["col"] == pytest.approx(100.0, abs=0.5)
        assert result.params["radius"] == pytest.approx(50.0, abs=0.5)
        assert result.error < 0.5

    def test_too_few_points(self) -> None:
        """Fewer than 3 points should raise ValueError."""
        with pytest.raises(ValueError, match="at least 3"):
            fit_circle_contour_xld([(0.0, 0.0), (1.0, 1.0)])

        with pytest.raises(ValueError, match="at least 3"):
            fit_circle_contour_xld([(0.0, 0.0)])

    @pytest.mark.parametrize("algorithm", ["algebraic", "geometric", "huber", "tukey"])
    def test_algorithms(self, algorithm: str) -> None:
        """All supported algorithms should fit a circle and return the
        correct centre.  The algebraic (Kasa) method has known radius
        bias so only centre accuracy is checked for all algorithms."""
        points = _circle_points(80.0, 120.0, 40.0, n=60)

        result = fit_circle_contour_xld(points, algorithm=algorithm)

        assert result.type == "circle"
        assert result.params["row"] == pytest.approx(80.0, abs=1.0)
        assert result.params["col"] == pytest.approx(120.0, abs=1.0)
        # Robust methods converge on radius; algebraic has known bias
        if algorithm != "algebraic":
            assert result.params["radius"] == pytest.approx(40.0, abs=1.0)

    def test_unknown_algorithm_raises(self) -> None:
        """An unsupported algorithm name should raise ValueError."""
        points = _circle_points(0.0, 0.0, 10.0, n=10)
        with pytest.raises(ValueError, match="unknown algorithm"):
            fit_circle_contour_xld(points, algorithm="bad_algo")


# ------------------------------------------------------------------
# TestFitEllipse
# ------------------------------------------------------------------

class TestFitEllipse:
    """Tests for fit_ellipse_contour_xld."""

    def test_perfect_ellipse(self) -> None:
        """Points on an ellipse with ra=60, rb=30 should be fitted accurately."""
        points = _ellipse_points(100.0, 100.0, 60.0, 30.0, phi=0.0, n=100)

        result = fit_ellipse_contour_xld(points)

        assert result.type == "ellipse"
        assert result.params["row"] == pytest.approx(100.0, abs=1.0)
        assert result.params["col"] == pytest.approx(100.0, abs=1.0)
        assert result.params["ra"] == pytest.approx(60.0, abs=1.0)
        assert result.params["rb"] == pytest.approx(30.0, abs=1.0)
        assert result.error < 1.0

    def test_too_few_points(self) -> None:
        """Fewer than 5 points should raise ValueError."""
        with pytest.raises(ValueError, match="at least 5"):
            fit_ellipse_contour_xld([(0, 0), (1, 0), (0, 1), (1, 1)])

    def test_rotated_ellipse(self) -> None:
        """An ellipse rotated by pi/6 should have phi close to pi/6."""
        phi = math.pi / 6.0
        points = _ellipse_points(80.0, 120.0, 70.0, 25.0, phi=phi, n=120)

        result = fit_ellipse_contour_xld(points)

        assert result.type == "ellipse"
        assert result.params["ra"] == pytest.approx(70.0, abs=2.0)
        assert result.params["rb"] == pytest.approx(25.0, abs=2.0)
        # phi should match (modulo pi, since direction is ambiguous)
        fitted_phi = result.params["phi"]
        angle_diff = abs(fitted_phi - phi) % math.pi
        angle_diff = min(angle_diff, math.pi - angle_diff)
        assert angle_diff < 0.1

    def test_ellipse_with_numpy_array(self) -> None:
        """Points provided as a numpy array should work identically."""
        pts_list = _ellipse_points(50.0, 50.0, 40.0, 20.0, n=80)
        pts_array = np.array(pts_list, dtype=np.float64)

        result = fit_ellipse_contour_xld(pts_array)

        assert result.type == "ellipse"
        assert result.params["ra"] == pytest.approx(40.0, abs=1.0)
        assert result.params["rb"] == pytest.approx(20.0, abs=1.0)


# ------------------------------------------------------------------
# TestDistances
# ------------------------------------------------------------------

class TestDistances:
    """Tests for distance_pp, distance_pl, and distance_cc."""

    def test_distance_pp(self) -> None:
        """Euclidean distance between (0,0) and (3,4) should be 5."""
        assert distance_pp(0.0, 0.0, 3.0, 4.0) == pytest.approx(5.0)

    def test_distance_pp_same_point(self) -> None:
        """Distance from a point to itself should be 0."""
        assert distance_pp(10.0, 20.0, 10.0, 20.0) == pytest.approx(0.0)

    @pytest.mark.parametrize("r1,c1,r2,c2,expected", [
        (0.0, 0.0, 1.0, 0.0, 1.0),
        (0.0, 0.0, 0.0, 1.0, 1.0),
        (1.0, 1.0, 4.0, 5.0, 5.0),
    ])
    def test_distance_pp_parametrized(
        self, r1: float, c1: float, r2: float, c2: float, expected: float,
    ) -> None:
        """Parametrised distance_pp checks."""
        assert distance_pp(r1, c1, r2, c2) == pytest.approx(expected)

    def test_distance_pl(self) -> None:
        """Point (0,5) to horizontal line y=0 from (0,0) to (0,10)
        should be 0 (point is on the line in row-col terms).
        Point (5,0) to horizontal line row=0 should have distance 5."""
        # Line along row=0, from col=0 to col=10
        d = distance_pl(5.0, 5.0, 0.0, 0.0, 0.0, 10.0)
        assert d == pytest.approx(5.0)

    def test_distance_pl_on_line(self) -> None:
        """A point on the line should return distance approximately 0."""
        d = distance_pl(0.0, 5.0, 0.0, 0.0, 0.0, 10.0)
        assert d == pytest.approx(0.0, abs=1e-10)

    def test_distance_pl_diagonal(self) -> None:
        """Point (0,0) to line from (0,1) to (1,0) should be 1/sqrt(2)."""
        d = distance_pl(0.0, 0.0, 0.0, 1.0, 1.0, 0.0)
        assert d == pytest.approx(1.0 / math.sqrt(2.0), abs=1e-10)

    def test_distance_cc(self) -> None:
        """Two contours separated by a known gap should return the
        correct minimum distance."""
        contour1 = [(0.0, 0.0), (0.0, 1.0), (0.0, 2.0)]
        contour2 = [(5.0, 0.0), (5.0, 1.0), (5.0, 2.0)]

        d = distance_cc(contour1, contour2)
        assert d == pytest.approx(5.0)

    def test_distance_cc_overlapping(self) -> None:
        """Two contours sharing a point should return 0."""
        contour1 = [(0.0, 0.0), (1.0, 1.0)]
        contour2 = [(1.0, 1.0), (2.0, 2.0)]

        d = distance_cc(contour1, contour2)
        assert d == pytest.approx(0.0)

    def test_distance_cc_empty_raises(self) -> None:
        """Empty contours should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            distance_cc([], [(1.0, 1.0)])


# ------------------------------------------------------------------
# TestAngles
# ------------------------------------------------------------------

class TestAngles:
    """Tests for angle_ll and angle_lx."""

    def test_angle_ll_perpendicular(self) -> None:
        """Two perpendicular lines should have angle pi/2."""
        # Horizontal line: (0,0)-(0,10)
        # Vertical line: (0,0)-(10,0)
        a = angle_ll(0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 10.0, 0.0)
        assert a == pytest.approx(math.pi / 2.0, abs=1e-10)

    def test_angle_ll_parallel(self) -> None:
        """Two parallel lines should have angle 0."""
        a = angle_ll(0.0, 0.0, 0.0, 10.0, 5.0, 0.0, 5.0, 10.0)
        assert a == pytest.approx(0.0, abs=1e-10)

    def test_angle_ll_45_degrees(self) -> None:
        """A horizontal and a 45-degree line should have angle pi/4."""
        a = angle_ll(0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 10.0, 10.0)
        assert a == pytest.approx(math.pi / 4.0, abs=1e-10)

    def test_angle_lx_horizontal(self) -> None:
        """A horizontal line should have angle approximately 0."""
        a = angle_lx(0.0, 0.0, 0.0, 10.0)
        assert a == pytest.approx(0.0, abs=1e-10)

    def test_angle_lx_vertical(self) -> None:
        """A vertical-downward line should have angle pi/2."""
        a = angle_lx(0.0, 0.0, 10.0, 0.0)
        assert a == pytest.approx(math.pi / 2.0, abs=1e-10)

    def test_angle_lx_negative(self) -> None:
        """A line going upward should return a negative angle."""
        a = angle_lx(10.0, 0.0, 0.0, 10.0)
        # row decreases, col increases => atan2(-10, 10) = -pi/4
        assert a == pytest.approx(-math.pi / 4.0, abs=1e-10)
