"""Tests for shared.core.pipeline_model."""

from __future__ import annotations

import json
import os
import tempfile
import zipfile

import numpy as np
import pytest

from shared.core.pipeline_model import (
    FORMAT_VERSION,
    FLOW_FILE,
    MANIFEST_FILE,
    RECIPE_FILE,
    PipelineModel,
    PipelineModelRegistry,
    _absolutize_paths,
    _collect_embedded_files,
    _relativize_paths,
)


# ====================================================================== #
#  Fixtures                                                                #
# ====================================================================== #


@pytest.fixture
def sample_image():
    """256x256 RGB uint8 test image."""
    return np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)


@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path


@pytest.fixture
def fake_checkpoint(tmp_dir):
    """Create a fake checkpoint file."""
    path = tmp_dir / "model.pt"
    path.write_bytes(b"fake_model_data_12345")
    return str(path)


@pytest.fixture
def fake_template(tmp_dir):
    """Create a fake template image."""
    path = tmp_dir / "template.png"
    path.write_bytes(b"fake_png_data")
    return str(path)


@pytest.fixture
def fake_calibration(tmp_dir):
    """Create a fake calibration file."""
    path = tmp_dir / "calib.json"
    path.write_text('{"pixels_per_mm": 10.0}', encoding="utf-8")
    return str(path)


@pytest.fixture
def sample_flow_dict(fake_checkpoint, fake_template):
    """A sample flow dict with path-valued config keys."""
    return {
        "flow_name": "Test Flow",
        "stop_on_failure": True,
        "version": "1.0",
        "steps": [
            {
                "name": "Locate",
                "step_type": "locate",
                "config": {"template_path": fake_template, "min_score": 0.5},
                "enabled": True,
            },
            {
                "name": "Detect",
                "step_type": "detect",
                "config": {
                    "checkpoint_path": fake_checkpoint,
                    "threshold": 0.5,
                },
                "enabled": True,
            },
            {
                "name": "Judge",
                "step_type": "judge",
                "config": {"rules": [], "logic": "all_pass"},
                "enabled": True,
            },
        ],
    }


# ====================================================================== #
#  Path helper tests                                                       #
# ====================================================================== #


class TestCollectEmbeddedFiles:
    def test_finds_checkpoint_and_template(
        self, sample_flow_dict, fake_checkpoint, fake_template
    ):
        result = _collect_embedded_files(sample_flow_dict)
        assert len(result) == 2

        # Step 0 has template_path
        assert result[0] == (0, "template_path", fake_template)
        # Step 1 has checkpoint_path
        assert result[1] == (1, "checkpoint_path", fake_checkpoint)

    def test_skips_nonexistent_files(self):
        flow_dict = {
            "steps": [
                {
                    "config": {
                        "checkpoint_path": "/nonexistent/model.pt",
                    }
                }
            ]
        }
        result = _collect_embedded_files(flow_dict)
        assert result == []

    def test_skips_empty_values(self):
        flow_dict = {
            "steps": [{"config": {"checkpoint_path": ""}}]
        }
        result = _collect_embedded_files(flow_dict)
        assert result == []

    def test_empty_steps(self):
        assert _collect_embedded_files({"steps": []}) == []
        assert _collect_embedded_files({}) == []


class TestRelativizePaths:
    def test_replaces_known_paths(self, sample_flow_dict, fake_checkpoint, fake_template):
        mapping = {
            fake_checkpoint: "models/model.pt",
            fake_template: "templates/template.png",
        }
        result = _relativize_paths(sample_flow_dict, mapping)

        assert result["steps"][0]["config"]["template_path"] == "templates/template.png"
        assert result["steps"][1]["config"]["checkpoint_path"] == "models/model.pt"

    def test_does_not_mutate_original(self, sample_flow_dict, fake_checkpoint):
        original_path = sample_flow_dict["steps"][1]["config"]["checkpoint_path"]
        mapping = {fake_checkpoint: "models/model.pt"}
        _relativize_paths(sample_flow_dict, mapping)

        # Original should be unchanged
        assert sample_flow_dict["steps"][1]["config"]["checkpoint_path"] == original_path

    def test_leaves_unmapped_paths_alone(self, sample_flow_dict):
        result = _relativize_paths(sample_flow_dict, {})
        assert result["steps"][1]["config"]["checkpoint_path"] == sample_flow_dict["steps"][1]["config"]["checkpoint_path"]


class TestAbsolutizePaths:
    def test_rewrites_relative_to_absolute(self):
        flow_dict = {
            "steps": [
                {
                    "config": {
                        "checkpoint_path": "models/model.pt",
                        "template_path": "templates/tpl.png",
                    }
                }
            ]
        }
        result = _absolutize_paths(flow_dict, "/tmp/extract")
        assert result["steps"][0]["config"]["checkpoint_path"] == os.path.join(
            "/tmp/extract", "models/model.pt"
        )
        assert result["steps"][0]["config"]["template_path"] == os.path.join(
            "/tmp/extract", "templates/tpl.png"
        )

    def test_does_not_mutate_original(self):
        flow_dict = {
            "steps": [{"config": {"checkpoint_path": "models/m.pt"}}]
        }
        _absolutize_paths(flow_dict, "/tmp/x")
        assert flow_dict["steps"][0]["config"]["checkpoint_path"] == "models/m.pt"

    def test_ignores_non_archive_paths(self):
        flow_dict = {
            "steps": [{"config": {"checkpoint_path": "/abs/path/model.pt"}}]
        }
        result = _absolutize_paths(flow_dict, "/tmp/x")
        # /abs doesn't start with "models", so unchanged
        assert result["steps"][0]["config"]["checkpoint_path"] == "/abs/path/model.pt"


# ====================================================================== #
#  PipelineModel tests                                                     #
# ====================================================================== #


class TestPipelineModelBuild:
    def test_build_basic(self):
        model = PipelineModel.build(
            name="Test Model",
            author="Tester",
            description="A test model",
        )
        assert model.metadata["name"] == "Test Model"
        assert model.metadata["author"] == "Tester"
        assert model.metadata["format_version"] == FORMAT_VERSION
        assert "created_at" in model.metadata
        assert model.recipe is None
        assert model.flow is None

    def test_build_with_custom_pipeline_config(self):
        cfg = {"num_preprocess_workers": 4, "num_postprocess_workers": 4}
        model = PipelineModel.build(name="X", pipeline_config=cfg)
        assert model.pipeline_config["num_preprocess_workers"] == 4

    def test_default_pipeline_config(self):
        model = PipelineModel()
        assert model.pipeline_config["num_preprocess_workers"] == 2
        assert model.pipeline_config["max_queue_size"] == 10


class TestPipelineModelInfo:
    def test_info_without_flow(self):
        model = PipelineModel.build(name="X")
        info = model.info()
        assert info["name"] == "X"
        assert info["has_flow"] is False
        assert info["has_recipe"] is False

    def test_info_with_mocked_flow(self):
        class FakeFlow:
            def __len__(self):
                return 3

            def get_steps(self):
                class FakeStep:
                    def __init__(self, n):
                        self.name = n
                return [FakeStep("A"), FakeStep("B"), FakeStep("C")]

        model = PipelineModel.build(name="Y", flow=FakeFlow())
        info = model.info()
        assert info["has_flow"] is True
        assert info["num_steps"] == 3
        assert info["step_names"] == ["A", "B", "C"]


class TestPipelineModelRepr:
    def test_repr_no_flow(self):
        model = PipelineModel.build(name="Test")
        r = repr(model)
        assert "Test" in r
        assert "steps=0" in r

    def test_repr_with_flow(self):
        class FakeFlow:
            def __len__(self):
                return 2

        model = PipelineModel.build(name="ABC", flow=FakeFlow())
        r = repr(model)
        assert "ABC" in r
        assert "steps=2" in r


class TestPipelineModelValidate:
    def test_validate_no_flow(self):
        model = PipelineModel.build(name="Empty")
        warnings = model.validate()
        assert any("No inspection flow" in w for w in warnings)

    def test_validate_bad_pipeline_config(self):
        model = PipelineModel.build(name="Bad Config")
        model.pipeline_config["num_preprocess_workers"] = -1

        class FakeFlow:
            def validate(self):
                return []

            def get_steps(self):
                return []

        model.flow = FakeFlow()
        warnings = model.validate()
        assert any("num_preprocess_workers" in w for w in warnings)


class TestPipelineModelExecute:
    def test_execute_without_flow_raises(self, sample_image):
        model = PipelineModel.build(name="No Flow")
        with pytest.raises(RuntimeError, match="No inspection flow"):
            model.execute(sample_image)

    def test_execute_batch_without_flow_raises(self, sample_image):
        model = PipelineModel.build(name="No Flow")
        with pytest.raises(RuntimeError, match="No inspection flow"):
            model.execute_batch([("img1", sample_image)])


class TestPipelineModelSaveLoad:
    def test_save_and_load_manifest_only(self, tmp_dir):
        """Save/load a model with no recipe and no flow."""
        model = PipelineModel.build(
            name="Manifest Only",
            author="Test",
            description="Minimal model",
        )
        path = str(tmp_dir / "test.cpmodel")
        model.save(path)

        assert os.path.isfile(path)

        # Verify it's a valid ZIP with manifest
        with zipfile.ZipFile(path, "r") as zf:
            names = zf.namelist()
            assert MANIFEST_FILE in names
            assert FLOW_FILE not in names
            assert RECIPE_FILE not in names

            manifest = json.loads(zf.read(MANIFEST_FILE))
            assert manifest["metadata"]["name"] == "Manifest Only"
            assert manifest["format_version"] == FORMAT_VERSION

    def test_save_embeds_files(self, tmp_dir, fake_checkpoint, fake_template):
        """Save a model with a flow that references files — verify they are embedded."""
        # We can't easily create a real InspectionFlow without heavy deps,
        # so we test via the internal _flow_to_dict + save logic using a
        # mock flow.
        class MockStep:
            def __init__(self, name, step_type, config):
                self.name = name
                self.step_type = step_type
                self.config = config
                self.enabled = True

            def to_dict(self):
                return {
                    "name": self.name,
                    "step_type": self.step_type,
                    "config": self.config,
                    "enabled": self.enabled,
                }

        class MockFlow:
            name = "Test"
            stop_on_failure = True

            def __init__(self, steps):
                self._steps = steps

            def get_steps(self):
                return self._steps

            def __len__(self):
                return len(self._steps)

        steps = [
            MockStep("Locate", "locate", {"template_path": fake_template}),
            MockStep("Detect", "detect", {"checkpoint_path": fake_checkpoint}),
        ]
        flow = MockFlow(steps)

        model = PipelineModel.build(name="Embed Test", flow=flow)
        path = str(tmp_dir / "embed.cpmodel")
        model.save(path)

        with zipfile.ZipFile(path, "r") as zf:
            names = zf.namelist()
            assert FLOW_FILE in names
            assert any(n.startswith("models/") for n in names)
            assert any(n.startswith("templates/") for n in names)

            # Verify embedded file content
            for name in names:
                if name.startswith("models/"):
                    assert zf.read(name) == b"fake_model_data_12345"
                elif name.startswith("templates/"):
                    assert zf.read(name) == b"fake_png_data"

            # Verify flow.json has relativized paths
            flow_data = json.loads(zf.read(FLOW_FILE))
            for step in flow_data["steps"]:
                cp = step["config"].get("checkpoint_path", "")
                tp = step["config"].get("template_path", "")
                if cp:
                    assert cp.startswith("models/")
                if tp:
                    assert tp.startswith("templates/")

    def test_roundtrip_manifest_metadata(self, tmp_dir):
        """Save and reload: metadata should be preserved."""
        model = PipelineModel.build(
            name="Roundtrip",
            author="Author",
            version="2.0.0",
            description="Test roundtrip",
            target_product="Widget",
        )
        path = str(tmp_dir / "rt.cpmodel")
        model.save(path)

        loaded = PipelineModel.load(path)
        try:
            assert loaded.metadata["name"] == "Roundtrip"
            assert loaded.metadata["author"] == "Author"
            assert loaded.metadata["version"] == "2.0.0"
            assert loaded.metadata["target_product"] == "Widget"
        finally:
            loaded.close()


class TestPipelineModelContextManager:
    def test_context_manager_cleanup(self, tmp_dir):
        model = PipelineModel.build(name="CM Test")
        path = str(tmp_dir / "cm.cpmodel")
        model.save(path)

        with PipelineModel.load(path) as loaded:
            extract_dir = loaded._extract_dir
            assert os.path.isdir(extract_dir)

        # After exiting context, temp dir should be cleaned up
        assert not os.path.exists(extract_dir)

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            PipelineModel.load("/nonexistent/path.cpmodel")


# ====================================================================== #
#  PipelineModelRegistry tests                                             #
# ====================================================================== #


class TestPipelineModelRegistry:
    def test_empty_registry(self, tmp_dir):
        registry = PipelineModelRegistry(str(tmp_dir / "registry"))
        models = registry.list_models()
        assert models == []

    def test_add_and_list(self, tmp_dir):
        registry = PipelineModelRegistry(str(tmp_dir / "registry"))
        model = PipelineModel.build(name="Model A", author="Tester")
        path = registry.add_model(model)

        assert os.path.isfile(path)
        assert path.endswith(".cpmodel")

        models = registry.list_models()
        assert len(models) == 1
        assert models[0]["metadata"]["name"] == "Model A"

    def test_add_duplicate_raises(self, tmp_dir):
        registry = PipelineModelRegistry(str(tmp_dir / "registry"))
        model = PipelineModel.build(name="Dup")
        registry.add_model(model)

        with pytest.raises(FileExistsError):
            registry.add_model(model)

    def test_add_duplicate_overwrite(self, tmp_dir):
        registry = PipelineModelRegistry(str(tmp_dir / "registry"))
        model = PipelineModel.build(name="Over")
        registry.add_model(model)
        # Should not raise
        registry.add_model(model, overwrite=True)

    def test_delete_model(self, tmp_dir):
        registry = PipelineModelRegistry(str(tmp_dir / "registry"))
        model = PipelineModel.build(name="Del")
        registry.add_model(model)

        registry.delete_model("Del.cpmodel")
        models = registry.list_models()
        assert len(models) == 0

    def test_delete_nonexistent_raises(self, tmp_dir):
        registry = PipelineModelRegistry(str(tmp_dir / "registry"))
        with pytest.raises(FileNotFoundError):
            registry.delete_model("nope.cpmodel")

    def test_find_model_case_insensitive(self, tmp_dir):
        registry = PipelineModelRegistry(str(tmp_dir / "registry"))
        model = PipelineModel.build(name="My Model")
        registry.add_model(model)

        found = registry.find_model("my model")
        assert found is not None
        assert found["metadata"]["name"] == "My Model"

    def test_find_model_not_found(self, tmp_dir):
        registry = PipelineModelRegistry(str(tmp_dir / "registry"))
        assert registry.find_model("nonexistent") is None

    def test_scan_refreshes_cache(self, tmp_dir):
        reg_dir = tmp_dir / "registry"
        registry = PipelineModelRegistry(str(reg_dir))

        model = PipelineModel.build(name="Scan Test")
        registry.add_model(model)

        # Create a new registry instance and scan
        registry2 = PipelineModelRegistry(str(reg_dir))
        registry2.scan()
        models = registry2.list_models()
        assert len(models) == 1
        assert models[0]["metadata"]["name"] == "Scan Test"

    def test_file_size_tracked(self, tmp_dir):
        registry = PipelineModelRegistry(str(tmp_dir / "registry"))
        model = PipelineModel.build(name="Size")
        registry.add_model(model)

        models = registry.list_models()
        assert "file_size_mb" in models[0]
        assert models[0]["file_size_mb"] >= 0

    def test_repr(self, tmp_dir):
        registry = PipelineModelRegistry(str(tmp_dir / "registry"))
        r = repr(registry)
        assert "PipelineModelRegistry" in r

    def test_custom_filename(self, tmp_dir):
        registry = PipelineModelRegistry(str(tmp_dir / "registry"))
        model = PipelineModel.build(name="Custom Name")
        path = registry.add_model(model, filename="custom.cpmodel")
        assert path.endswith("custom.cpmodel")
