"""Tests for shared.core.unet_segment."""
import pytest
import numpy as np

pytest.importorskip("torch")

from shared.core.unet_segment import UNet, UNetModel, UNetTrainer, UNetInference, save_model, load_model
import torch


class TestUNetArchitecture:
    def test_forward_shape(self):
        model = UNet(in_channels=3, num_classes=1, base_channels=16, depth=3)
        x = torch.randn(2, 3, 64, 64)
        out = model(x)
        assert out.shape == (2, 1, 64, 64)

    def test_reconstruction_mode_shape(self):
        model = UNet(in_channels=3, num_classes=3, base_channels=16, depth=3)
        x = torch.randn(2, 3, 64, 64)
        out = model(x)
        assert out.shape == (2, 3, 64, 64)


class TestUNetTrainInfer:
    def test_reconstruction_train(self, tmp_path):
        ok_dir = tmp_path / "ok"
        ok_dir.mkdir()
        import cv2
        for i in range(4):
            img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            cv2.imwrite(str(ok_dir / f"ok_{i}.png"), img)

        trainer = UNetTrainer(
            mode="reconstruction", base_channels=8, depth=2,
            image_size=64, device="cpu",
        )
        model = trainer.train(ok_dir, epochs=1, batch_size=2)
        assert isinstance(model, UNetModel)
        assert model.mode == "reconstruction"

        inference = UNetInference(model, device="cpu")
        score, mask = inference.segment(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        assert isinstance(score, float)
        assert isinstance(mask, np.ndarray)

    def test_save_load(self, tmp_path):
        ok_dir = tmp_path / "ok"
        ok_dir.mkdir()
        import cv2
        for i in range(4):
            img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            cv2.imwrite(str(ok_dir / f"ok_{i}.png"), img)

        trainer = UNetTrainer(mode="reconstruction", base_channels=8, depth=2, image_size=64, device="cpu")
        model = trainer.train(ok_dir, epochs=1, batch_size=2)

        path = tmp_path / "unet.pth"
        save_model(model, path)
        loaded = load_model(path, device="cpu")
        assert loaded.mode == model.mode
        assert loaded.depth == model.depth
