"""Tests for dl_anomaly.core.preprocessor -- ImagePreprocessor class."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from dl_anomaly.core.preprocessor import ImagePreprocessor


# ------------------------------------------------------------------
# load_and_preprocess
# ------------------------------------------------------------------

class TestLoadAndPreprocess:
    """Tests for loading an image file and returning a normalised tensor."""

    def test_rgb_output_shape(self, tmp_path: Path) -> None:
        """RGB preprocessing should produce a (3, H, W) tensor."""
        img_path = tmp_path / "test_rgb.png"
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        img.save(str(img_path))

        pp = ImagePreprocessor(image_size=64, grayscale=False)
        tensor = pp.load_and_preprocess(img_path)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 64, 64)

    def test_grayscale_output_shape(self, tmp_path: Path) -> None:
        """Grayscale preprocessing should produce a (1, H, W) tensor."""
        img_path = tmp_path / "test_gray.png"
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        img.save(str(img_path))

        pp = ImagePreprocessor(image_size=64, grayscale=True)
        tensor = pp.load_and_preprocess(img_path)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 64, 64)

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        """Attempting to load a nonexistent file should raise FileNotFoundError."""
        pp = ImagePreprocessor(image_size=64, grayscale=False)
        with pytest.raises(FileNotFoundError):
            pp.load_and_preprocess(tmp_path / "nonexistent.png")

    def test_output_is_float_tensor(self, tmp_path: Path) -> None:
        """Output tensor should be float32 (after ToTensor + Normalize)."""
        img_path = tmp_path / "test.png"
        Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(str(img_path))

        pp = ImagePreprocessor(image_size=32, grayscale=False)
        tensor = pp.load_and_preprocess(img_path)
        assert tensor.dtype == torch.float32


# ------------------------------------------------------------------
# inverse_normalize round-trip
# ------------------------------------------------------------------

class TestInverseNormalize:
    """Tests for the inverse_normalize method."""

    def test_round_trip_rgb(self, tmp_path: Path) -> None:
        """Normalize then inverse_normalize should approximately recover the original image."""
        # Create a known image and save it
        original = np.random.randint(30, 220, (64, 64, 3), dtype=np.uint8)
        img_path = tmp_path / "roundtrip.png"
        Image.fromarray(original).save(str(img_path))

        pp = ImagePreprocessor(image_size=64, grayscale=False)
        tensor = pp.load_and_preprocess(img_path)
        recovered = pp.inverse_normalize(tensor)

        assert recovered.dtype == np.uint8
        assert recovered.shape == (64, 64, 3)
        # Allow tolerance for quantization and resize interpolation
        np.testing.assert_allclose(
            recovered.astype(np.float32),
            original.astype(np.float32),
            atol=10.0,
        )

    def test_round_trip_grayscale(self, tmp_path: Path) -> None:
        """Normalize then inverse_normalize for grayscale should recover a (H, W) array."""
        original = np.random.randint(30, 220, (64, 64), dtype=np.uint8)
        img_path = tmp_path / "roundtrip_gray.png"
        Image.fromarray(original, mode="L").save(str(img_path))

        pp = ImagePreprocessor(image_size=64, grayscale=True)
        tensor = pp.load_and_preprocess(img_path)
        recovered = pp.inverse_normalize(tensor)

        assert recovered.dtype == np.uint8
        assert recovered.ndim == 2
        assert recovered.shape == (64, 64)

    def test_inverse_normalize_clamps_to_uint8(self) -> None:
        """Even with extreme tensor values, output should be valid uint8."""
        pp = ImagePreprocessor(image_size=32, grayscale=False)
        extreme = torch.randn(3, 32, 32) * 10.0  # far outside [0, 1]
        result = pp.inverse_normalize(extreme)

        assert result.dtype == np.uint8
        assert result.min() >= 0
        assert result.max() <= 255

    def test_get_transforms_caching(self) -> None:
        """Calling get_transforms twice should return the same cached object."""
        pp = ImagePreprocessor(image_size=64, grayscale=False)
        t1 = pp.get_transforms(augment=False)
        t2 = pp.get_transforms(augment=False)
        assert t1 is t2

        t3 = pp.get_transforms(augment=True)
        t4 = pp.get_transforms(augment=True)
        assert t3 is t4
        assert t1 is not t3
