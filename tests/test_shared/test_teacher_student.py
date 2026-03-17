"""Tests for shared.core.teacher_student."""
import pytest
import numpy as np

pytest.importorskip("torch")

from shared.core.teacher_student import (
    TeacherStudentModel,
    TeacherStudentTrainer,
    TeacherStudentInference,
    save_model,
    load_model,
)


class TestTeacherStudent:
    def test_trainer_init(self):
        trainer = TeacherStudentTrainer(backbone_name="resnet18", image_size=64)
        assert trainer is not None

    def test_inference_score_shape(self, tmp_path):
        """Test that score_image returns correct types."""
        # Create a minimal trained model by training 1 epoch on synthetic data
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        import cv2
        for i in range(4):
            img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            cv2.imwrite(str(img_dir / f"img_{i}.png"), img)

        trainer = TeacherStudentTrainer(
            backbone_name="resnet18", layers=("layer1", "layer2"),
            image_size=64, device="cpu",
        )
        model = trainer.train(img_dir, epochs=1, lr=0.01, batch_size=2)
        assert isinstance(model, TeacherStudentModel)

        inference = TeacherStudentInference(model, device="cpu")
        test_img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        score, amap = inference.score_image(test_img)
        assert isinstance(score, float)
        assert isinstance(amap, np.ndarray)

    def test_save_load_roundtrip(self, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        import cv2
        for i in range(4):
            img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            cv2.imwrite(str(img_dir / f"img_{i}.png"), img)

        trainer = TeacherStudentTrainer(
            backbone_name="resnet18", layers=("layer1",),
            image_size=64, device="cpu",
        )
        model = trainer.train(img_dir, epochs=1, batch_size=2)

        model_path = tmp_path / "ts_model.pt"
        save_model(model, model_path)
        loaded = load_model(model_path, device="cpu")
        assert loaded.backbone_name == model.backbone_name
        assert loaded.layers == model.layers

    def test_detect_interface(self, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        import cv2
        for i in range(4):
            img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            cv2.imwrite(str(img_dir / f"img_{i}.png"), img)

        trainer = TeacherStudentTrainer(
            backbone_name="resnet18", layers=("layer1",),
            image_size=64, device="cpu",
        )
        model = trainer.train(img_dir, epochs=1, batch_size=2)
        inference = TeacherStudentInference(model, device="cpu")
        result = inference.detect(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        assert "anomaly_score" in result
        assert "defect_mask" in result
