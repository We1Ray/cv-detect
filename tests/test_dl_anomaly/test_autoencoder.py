"""Tests for the convolutional autoencoder model."""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

torch = pytest.importorskip("torch")

from dl_anomaly.core.autoencoder import (
    AnomalyAutoencoder,
    ConvBlock,
    Decoder,
    Encoder,
)


# ======================================================================
# ConvBlock
# ======================================================================

class TestConvBlock:
    def test_forward_shape(self):
        block = ConvBlock(in_channels=3, out_channels=16, residual=True)
        x = torch.randn(2, 3, 32, 32)
        out = block(x)
        assert out.shape == (2, 16, 32, 32)

    def test_same_channels_residual(self):
        block = ConvBlock(in_channels=16, out_channels=16, residual=True)
        x = torch.randn(1, 16, 8, 8)
        out = block(x)
        assert out.shape == (1, 16, 8, 8)

    def test_no_residual(self):
        block = ConvBlock(in_channels=3, out_channels=16, residual=False)
        x = torch.randn(1, 3, 16, 16)
        out = block(x)
        assert out.shape == (1, 16, 16, 16)


# ======================================================================
# Encoder
# ======================================================================

class TestEncoder:
    def test_output_shape(self):
        enc = Encoder(in_channels=3, base_channels=16, num_blocks=3, latent_dim=64)
        x = torch.randn(2, 3, 64, 64)
        z = enc(x)
        assert z.shape == (2, 64)

    def test_grayscale_input(self):
        enc = Encoder(in_channels=1, base_channels=16, num_blocks=2, latent_dim=32)
        x = torch.randn(1, 1, 64, 64)
        z = enc(x)
        assert z.shape == (1, 32)

    def test_final_channels_attribute(self):
        enc = Encoder(in_channels=3, base_channels=16, num_blocks=3, latent_dim=64)
        # After 3 blocks: 16 -> 32 -> 64, final_channels should be 64
        assert enc.final_channels == 64


# ======================================================================
# Decoder
# ======================================================================

class TestDecoder:
    def test_output_shape_matches_image_size(self):
        dec = Decoder(
            latent_dim=64,
            base_channels=16,
            num_blocks=3,
            out_channels=3,
            image_size=64,
        )
        z = torch.randn(2, 64)
        out = dec(z)
        assert out.shape == (2, 3, 64, 64)

    def test_output_range_sigmoid(self):
        dec = Decoder(
            latent_dim=32,
            base_channels=16,
            num_blocks=2,
            out_channels=3,
            image_size=64,
        )
        z = torch.randn(1, 32)
        out = dec(z)
        # Sigmoid output should be in [0, 1]
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_grayscale_output(self):
        dec = Decoder(
            latent_dim=32,
            base_channels=16,
            num_blocks=2,
            out_channels=1,
            image_size=64,
        )
        z = torch.randn(1, 32)
        out = dec(z)
        assert out.shape == (1, 1, 64, 64)


# ======================================================================
# AnomalyAutoencoder (full model)
# ======================================================================

class TestAnomalyAutoencoder:
    def test_default_instantiation(self):
        model = AnomalyAutoencoder()
        assert model.in_channels == 3
        assert model.latent_dim == 128
        assert model.base_channels == 32
        assert model.num_blocks == 4
        assert model.image_size == 256

    def test_custom_config(self):
        model = AnomalyAutoencoder(
            in_channels=1,
            latent_dim=64,
            base_channels=16,
            num_blocks=3,
            image_size=128,
        )
        assert model.in_channels == 1
        assert model.latent_dim == 64
        assert model.image_size == 128

    def test_forward_shape_default(self):
        model = AnomalyAutoencoder(
            in_channels=3,
            latent_dim=64,
            base_channels=16,
            num_blocks=3,
            image_size=64,
        )
        model.eval()
        x = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out.shape == x.shape

    def test_forward_shape_grayscale(self):
        model = AnomalyAutoencoder(
            in_channels=1,
            latent_dim=32,
            base_channels=16,
            num_blocks=2,
            image_size=64,
        )
        model.eval()
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out.shape == x.shape

    def test_encoder_decoder_components_exist(self):
        model = AnomalyAutoencoder()
        assert hasattr(model, "encoder")
        assert hasattr(model, "decoder")
        assert isinstance(model.encoder, Encoder)
        assert isinstance(model.decoder, Decoder)

    def test_get_latent(self):
        model = AnomalyAutoencoder(
            in_channels=3,
            latent_dim=64,
            base_channels=16,
            num_blocks=3,
            image_size=64,
        )
        model.eval()
        x = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            z = model.get_latent(x)
        assert z.shape == (2, 64)

    def test_latent_dimension_correct(self):
        for ldim in [32, 64, 128, 256]:
            model = AnomalyAutoencoder(
                latent_dim=ldim,
                base_channels=16,
                num_blocks=2,
                image_size=64,
            )
            model.eval()
            x = torch.randn(1, 3, 64, 64)
            with torch.no_grad():
                z = model.get_latent(x)
            assert z.shape[1] == ldim

    def test_parameter_count_reasonable(self):
        """A small model should have a non-trivial but bounded param count."""
        model = AnomalyAutoencoder(
            in_channels=3,
            latent_dim=64,
            base_channels=16,
            num_blocks=3,
            image_size=64,
        )
        total_params = sum(p.numel() for p in model.parameters())
        # Should have meaningful parameters (at least 10K)
        assert total_params > 10_000
        # But a small model should not exceed 50M params
        assert total_params < 50_000_000

    def test_output_in_valid_range(self):
        """Output should be bounded in [0, 1] due to final Sigmoid."""
        model = AnomalyAutoencoder(
            in_channels=3,
            latent_dim=32,
            base_channels=16,
            num_blocks=2,
            image_size=64,
        )
        model.eval()
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_model_trainable(self):
        """Model should be differentiable and produce gradients."""
        model = AnomalyAutoencoder(
            in_channels=3,
            latent_dim=32,
            base_channels=16,
            num_blocks=2,
            image_size=64,
        )
        model.train()
        x = torch.randn(2, 3, 64, 64)
        out = model(x)
        loss = torch.nn.functional.mse_loss(out, x)
        loss.backward()
        # Check that at least some parameters have gradients
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad

    def test_different_image_sizes(self):
        """Model should work with various square input sizes."""
        for size in [32, 64, 128]:
            model = AnomalyAutoencoder(
                in_channels=3,
                latent_dim=32,
                base_channels=16,
                num_blocks=2,
                image_size=size,
            )
            model.eval()
            x = torch.randn(1, 3, size, size)
            with torch.no_grad():
                out = model(x)
            assert out.shape == (1, 3, size, size), f"Failed for image_size={size}"
