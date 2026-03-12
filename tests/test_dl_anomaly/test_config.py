"""Tests for dl_anomaly.config -- Config dataclass and helper functions."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from dl_anomaly.config import Config, _parse_bool, _select_device


# ------------------------------------------------------------------
# _select_device
# ------------------------------------------------------------------

class TestSelectDevice:
    """Tests for the _select_device auto-detection helper."""

    def test_cpu_always_available(self) -> None:
        """Requesting 'cpu' should always return 'cpu'."""
        assert _select_device("cpu") == "cpu"

    def test_auto_returns_string(self) -> None:
        """'auto' should resolve to one of the known device strings."""
        result = _select_device("auto")
        assert result in ("cpu", "cuda", "mps")

    def test_cuda_fallback_when_unavailable(self) -> None:
        """Requesting 'cuda' on a machine without CUDA should fall back gracefully."""
        result = _select_device("cuda")
        if torch.cuda.is_available():
            assert result == "cuda"
        else:
            # Falls back to auto-detection
            assert result in ("cpu", "mps")

    def test_whitespace_stripped(self) -> None:
        """Leading/trailing whitespace in the requested device should be ignored."""
        assert _select_device("  cpu  ") == "cpu"

    def test_case_insensitive(self) -> None:
        """Device names should be matched case-insensitively."""
        assert _select_device("CPU") == "cpu"


# ------------------------------------------------------------------
# _parse_bool
# ------------------------------------------------------------------

class TestParseBool:
    """Tests for the _parse_bool helper."""

    @pytest.mark.parametrize("val", ["true", "True", "TRUE", "1", "yes", "Yes"])
    def test_truthy_values(self, val: str) -> None:
        """Values like 'true', '1', 'yes' should parse as True."""
        assert _parse_bool(val) is True

    @pytest.mark.parametrize("val", ["false", "0", "no", "anything", ""])
    def test_falsy_values(self, val: str) -> None:
        """Values not in the truthy set should parse as False."""
        assert _parse_bool(val) is False


# ------------------------------------------------------------------
# Config defaults
# ------------------------------------------------------------------

class TestConfigDefaults:
    """Tests for Config default values and post-init behaviour."""

    def test_default_image_size(self, sample_config) -> None:
        """Config created by the fixture should have image_size=64."""
        assert sample_config.image_size == 64

    def test_in_channels_rgb(self, sample_config) -> None:
        """When grayscale=False, in_channels should be 3."""
        assert sample_config.in_channels == 3

    def test_in_channels_grayscale(self, tmp_path: Path) -> None:
        """When grayscale=True, in_channels should be 1."""
        cfg = Config(
            checkpoint_dir=tmp_path / "ckpt",
            results_dir=tmp_path / "res",
            grayscale=True,
            device="cpu",
        )
        assert cfg.in_channels == 1

    def test_directories_created(self, sample_config) -> None:
        """__post_init__ should create checkpoint_dir and results_dir."""
        assert sample_config.checkpoint_dir.is_dir()
        assert sample_config.results_dir.is_dir()

    def test_device_is_cpu(self, sample_config) -> None:
        """The fixture explicitly sets device='cpu'."""
        assert sample_config.device == "cpu"


# ------------------------------------------------------------------
# Config serialisation round-trip
# ------------------------------------------------------------------

class TestConfigSerialization:
    """Tests for to_dict / from_dict round-trip."""

    def test_round_trip(self, sample_config) -> None:
        """Config -> dict -> Config should preserve all field values."""
        d = sample_config.to_dict()
        restored = Config.from_dict(d)

        assert restored.image_size == sample_config.image_size
        assert restored.latent_dim == sample_config.latent_dim
        assert restored.grayscale == sample_config.grayscale
        assert restored.device == sample_config.device
        assert restored.in_channels == sample_config.in_channels

    def test_to_dict_paths_are_strings(self, sample_config) -> None:
        """Path fields should be serialised as plain strings in to_dict."""
        d = sample_config.to_dict()
        assert isinstance(d["checkpoint_dir"], str)
        assert isinstance(d["results_dir"], str)
