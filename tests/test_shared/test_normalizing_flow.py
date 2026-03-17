"""Tests for shared.core.normalizing_flow."""
import pytest
import numpy as np

pytest.importorskip("torch")

from shared.core.normalizing_flow import (
    NormFlowModel,
    NormFlowTrainer,
    NormFlowInference,
    save_model,
    load_model,
    AffineCouplingBlock,
    NormalizingFlow2D,
)
import torch


class TestFlowArchitecture:
    def test_coupling_block_shape(self):
        block = AffineCouplingBlock(channels=16, hidden_channels=32)
        x = torch.randn(2, 16, 8, 8)
        y, log_det = block(x)
        assert y.shape == x.shape
        assert log_det.shape == (2,)

    def test_flow_forward_inverse(self):
        flow = NormalizingFlow2D(channels=16, n_blocks=2, hidden_channels=32)
        x = torch.randn(2, 16, 8, 8)
        z, log_det = flow(x)
        assert z.shape == x.shape

    def test_flow_log_prob(self):
        flow = NormalizingFlow2D(channels=16, n_blocks=2, hidden_channels=32)
        x = torch.randn(2, 16, 8, 8)
        lp = flow.log_prob(x)
        assert lp.shape == (2,)


class TestNormFlowTrainInfer:
    def test_train_and_infer(self, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        import cv2
        for i in range(4):
            img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            cv2.imwrite(str(img_dir / f"img_{i}.png"), img)

        trainer = NormFlowTrainer(
            backbone_name="resnet18", layers=("layer1",),
            image_size=64, n_flow_blocks=2, hidden_channels=32, device="cpu",
        )
        model = trainer.train(img_dir, epochs=1, batch_size=2)
        assert isinstance(model, NormFlowModel)

        inference = NormFlowInference(model, device="cpu")
        test_img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        score, amap = inference.score_image(test_img)
        assert isinstance(score, float)
        assert isinstance(amap, np.ndarray)

    def test_save_load(self, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        import cv2
        for i in range(4):
            img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            cv2.imwrite(str(img_dir / f"img_{i}.png"), img)

        trainer = NormFlowTrainer(
            backbone_name="resnet18", layers=("layer1",),
            image_size=64, n_flow_blocks=2, hidden_channels=32, device="cpu",
        )
        model = trainer.train(img_dir, epochs=1, batch_size=2)

        path = tmp_path / "nf_model.pt"
        save_model(model, path)
        loaded = load_model(path, device="cpu")
        assert loaded.backbone_name == model.backbone_name
