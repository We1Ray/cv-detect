"""Tests for shared.core.image_difference."""
import numpy as np
import pytest
from shared.core.image_difference import ImageDifferencer, DifferenceResult


class TestImageDifferencer:
    def test_init_defaults(self):
        d = ImageDifferencer()
        assert d.registration_method == "ecc"
        assert d.threshold == 30.0

    def test_set_reference(self):
        d = ImageDifferencer()
        ref = np.zeros((100, 100, 3), dtype=np.uint8)
        d.set_reference(ref)
        assert d.has_reference

    def test_identical_images_low_score(self):
        d = ImageDifferencer(registration_method="none")
        img = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
        d.set_reference(img)
        result = d.compute_difference(img.copy())
        assert isinstance(result, DifferenceResult)
        assert result.score < 0.1  # nearly identical

    def test_different_images_high_score(self):
        d = ImageDifferencer(registration_method="none", threshold=10)
        ref = np.full((100, 100), 128, dtype=np.uint8)
        target = ref.copy()
        target[30:60, 30:60] = 255  # big defect
        d.set_reference(ref)
        result = d.compute_difference(target)
        assert result.score > 0.1

    def test_detect_interface(self):
        d = ImageDifferencer(registration_method="none")
        ref = np.full((100, 100, 3), 128, dtype=np.uint8)
        d.set_reference(ref)
        target = ref.copy()
        target[40:60, 40:60] = [255, 0, 0]
        result = d.detect(target)
        assert "anomaly_score" in result
        assert "defect_mask" in result
        assert "defect_regions" in result
        assert "is_defective" in result

    def test_registration_methods(self):
        for method in ["ecc", "orb", "phase_correlation", "none"]:
            d = ImageDifferencer(registration_method=method)
            ref = np.random.randint(50, 200, (100, 100), dtype=np.uint8)
            d.set_reference(ref)
            result = d.compute_difference(ref.copy())
            assert isinstance(result, DifferenceResult)

    def test_grayscale_input(self):
        d = ImageDifferencer(registration_method="none")
        ref = np.full((80, 80), 100, dtype=np.uint8)
        d.set_reference(ref)
        result = d.compute_difference(ref.copy())
        assert result.score < 0.1
