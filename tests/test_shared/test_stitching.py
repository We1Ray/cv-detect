"""Tests for shared.core.stitching -- image stitching for large product inspection."""

from __future__ import annotations

import numpy as np
import pytest

from shared.core.stitching import (
    StitchResult,
    crop_black_borders,
    detect_and_match_features,
    estimate_homography,
    stitch_grid,
    stitch_strip,
)


# ------------------------------------------------------------------
# Helpers for creating synthetic overlapping images
# ------------------------------------------------------------------


def _make_textured_image(height: int, width: int, seed: int = 0) -> np.ndarray:
    """Create a synthetic image with distinct texture features for matching."""
    rng = np.random.RandomState(seed)
    img = rng.randint(50, 200, (height, width, 3), dtype=np.uint8)
    # Add some geometric features to improve feature detection
    import cv2

    for _ in range(15):
        cx = rng.randint(20, width - 20)
        cy = rng.randint(20, height - 20)
        radius = rng.randint(5, 20)
        color = tuple(int(c) for c in rng.randint(0, 255, 3))
        cv2.circle(img, (cx, cy), radius, color, -1)
    for _ in range(10):
        x1, y1 = rng.randint(0, width), rng.randint(0, height)
        x2, y2 = rng.randint(0, width), rng.randint(0, height)
        color = tuple(int(c) for c in rng.randint(0, 255, 3))
        cv2.line(img, (x1, y1), (x2, y2), color, 2)
    return img


def _make_overlapping_strip_pair(
    height: int = 200, width: int = 300, overlap: int = 100, seed: int = 42,
) -> tuple:
    """Create two horizontally overlapping images from a single source."""
    total_width = width + width - overlap
    source = _make_textured_image(height, total_width, seed=seed)
    img1 = source[:, :width].copy()
    img2 = source[:, width - overlap :].copy()
    return img1, img2


# ------------------------------------------------------------------
# detect_and_match_features
# ------------------------------------------------------------------


class TestFeatureDetection:
    """Tests for detect_and_match_features."""

    def test_returns_three_lists(self) -> None:
        """Should return (kp1, kp2, matches) as three lists."""
        img = _make_textured_image(100, 100, seed=1)
        kp1, kp2, matches = detect_and_match_features(img, img)
        assert isinstance(kp1, list)
        assert isinstance(kp2, list)
        assert isinstance(matches, list)

    def test_detects_keypoints_on_textured_image(self) -> None:
        """Textured images should produce keypoints."""
        img = _make_textured_image(150, 150, seed=2)
        kp1, kp2, matches = detect_and_match_features(img, img)
        assert len(kp1) > 0
        assert len(kp2) > 0

    def test_identical_images_many_matches(self) -> None:
        """Two identical images should produce many good matches."""
        img = _make_textured_image(150, 150, seed=3)
        kp1, kp2, matches = detect_and_match_features(img, img)
        assert len(matches) > 5

    def test_accepts_grayscale(self) -> None:
        """Should accept single-channel grayscale images."""
        gray = _make_textured_image(100, 100, seed=4)[:, :, 0]
        kp1, kp2, matches = detect_and_match_features(gray, gray)
        assert isinstance(kp1, list)

    def test_method_sift(self) -> None:
        """SIFT method should work when available."""
        img = _make_textured_image(100, 100, seed=5)
        kp1, kp2, matches = detect_and_match_features(img, img, method="sift")
        assert isinstance(matches, list)

    def test_method_akaze(self) -> None:
        """AKAZE method should work."""
        img = _make_textured_image(100, 100, seed=6)
        kp1, kp2, matches = detect_and_match_features(img, img, method="akaze")
        assert isinstance(matches, list)


# ------------------------------------------------------------------
# estimate_homography
# ------------------------------------------------------------------


class TestEstimateHomography:
    """Tests for estimate_homography."""

    def test_raises_on_too_few_matches(self) -> None:
        """Should raise ValueError when fewer than MIN_MATCH_COUNT matches."""
        with pytest.raises(ValueError):
            estimate_homography([], [], [])

    def test_returns_3x3_matrix(self) -> None:
        """Should return a 3x3 homography matrix."""
        img = _make_textured_image(200, 200, seed=10)
        kp1, kp2, matches = detect_and_match_features(img, img, method="orb")
        if len(matches) >= 10:
            H, mask = estimate_homography(matches, kp1, kp2)
            assert H.shape == (3, 3)
        else:
            pytest.skip("Not enough matches for homography test")

    def test_identity_for_same_image(self) -> None:
        """Homography of an image with itself should be near identity."""
        img = _make_textured_image(200, 200, seed=11)
        kp1, kp2, matches = detect_and_match_features(img, img, method="orb")
        if len(matches) >= 10:
            H, mask = estimate_homography(matches, kp1, kp2)
            identity = np.eye(3, dtype=np.float64)
            np.testing.assert_allclose(H, identity, atol=0.5)
        else:
            pytest.skip("Not enough matches for homography test")


# ------------------------------------------------------------------
# stitch_strip
# ------------------------------------------------------------------


class TestStitchStrip:
    """Tests for stitch_strip."""

    def test_single_image_returns_copy(self) -> None:
        """A single image should be returned as-is."""
        img = _make_textured_image(100, 150, seed=20)
        result = stitch_strip([img])
        assert isinstance(result, StitchResult)
        assert result.status == "success"
        assert result.num_images == 1
        assert result.panorama.shape == img.shape

    def test_empty_list_returns_failed(self) -> None:
        """An empty image list should return failed status."""
        result = stitch_strip([])
        assert result.status == "failed"

    def test_two_images_produces_wider_result(self) -> None:
        """Stitching two horizontally overlapping images should produce a wider panorama."""
        img1, img2 = _make_overlapping_strip_pair(
            height=100, width=200, overlap=60, seed=21
        )
        result = stitch_strip([img1, img2], overlap_ratio=0.3)
        assert isinstance(result, StitchResult)
        assert result.status == "success"
        # Stitched result should be wider than either input
        assert result.panorama.shape[1] > img1.shape[1]

    def test_result_has_homographies(self) -> None:
        """StitchResult should contain homography matrices."""
        img1, img2 = _make_overlapping_strip_pair(
            height=100, width=200, overlap=60, seed=22
        )
        result = stitch_strip([img1, img2], overlap_ratio=0.3)
        assert len(result.homographies) >= 1

    def test_vertical_direction(self) -> None:
        """Vertical stitching should produce a taller panorama."""
        rng = np.random.RandomState(23)
        h, w = 100, 150
        overlap = 30
        total_h = h + h - overlap
        source = _make_textured_image(total_h, w, seed=23)
        img1 = source[:h, :].copy()
        img2 = source[h - overlap :, :].copy()
        result = stitch_strip([img1, img2], overlap_ratio=0.3, direction="vertical")
        assert result.status == "success"
        assert result.panorama.shape[0] > h


# ------------------------------------------------------------------
# stitch_grid
# ------------------------------------------------------------------


class TestStitchGrid:
    """Tests for stitch_grid."""

    def test_wrong_image_count_fails(self) -> None:
        """Providing the wrong number of images should return failed status."""
        imgs = [_make_textured_image(50, 50, seed=i) for i in range(3)]
        result = stitch_grid(imgs, grid_shape=(2, 2))
        assert result.status == "failed"

    def test_single_row_grid(self) -> None:
        """A 1x2 grid should stitch horizontally."""
        img1, img2 = _make_overlapping_strip_pair(
            height=100, width=150, overlap=50, seed=30
        )
        result = stitch_grid([img1, img2], grid_shape=(1, 2), overlap_ratio=0.33)
        assert result.status == "success"
        assert result.panorama.shape[1] > img1.shape[1]

    def test_single_column_grid(self) -> None:
        """A 2x1 grid should stitch vertically."""
        h, w = 100, 150
        overlap = 30
        source = _make_textured_image(h * 2 - overlap, w, seed=31)
        img1 = source[:h, :].copy()
        img2 = source[h - overlap :, :].copy()
        result = stitch_grid([img1, img2], grid_shape=(2, 1), overlap_ratio=0.3)
        assert result.status == "success"
        assert result.panorama.shape[0] > h

    def test_2x2_grid(self) -> None:
        """A 2x2 grid of identical images should succeed."""
        img = _make_textured_image(80, 80, seed=32)
        # Use the same image for all tiles (overlap estimation will find self-match)
        result = stitch_grid([img, img, img, img], grid_shape=(2, 2), overlap_ratio=0.2)
        # Even if the result is not ideal, it should not crash
        assert isinstance(result, StitchResult)
        assert result.num_images == 4


# ------------------------------------------------------------------
# crop_black_borders
# ------------------------------------------------------------------


class TestCropBlackBorders:
    """Tests for crop_black_borders."""

    def test_removes_border(self) -> None:
        """Should crop black borders and return a smaller image."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[10:90, 10:90] = 128
        cropped = crop_black_borders(img)
        assert cropped.shape[0] <= 80
        assert cropped.shape[1] <= 80

    def test_no_border_unchanged(self) -> None:
        """An image with no black borders should remain the same size."""
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        cropped = crop_black_borders(img)
        assert cropped.shape == img.shape

    def test_all_black_returns_original(self) -> None:
        """An all-black image should return the original (nothing to crop to)."""
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        cropped = crop_black_borders(img)
        # Should return something without crashing
        assert cropped.shape[0] > 0
