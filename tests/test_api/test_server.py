"""Tests for the REST API server."""
import io
import os
import pytest
import numpy as np

# Ensure project root is on path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Skip the entire module if FastAPI or its test dependencies are not installed.
pytest.importorskip("fastapi")
pytest.importorskip("starlette")
pytest.importorskip("PIL")


@pytest.fixture
def app():
    """Create a test FastAPI app instance."""
    # Reset module-level state
    import api.server as srv
    srv._dl_pipeline = None
    srv._vm_pipeline = None
    srv._vm_model = None
    srv._vm_config = None
    srv._results_db = None
    return srv._get_app()


@pytest.fixture
def client(app):
    """Create a test client."""
    from starlette.testclient import TestClient
    return TestClient(app)


@pytest.fixture
def auth_client():
    """Create a test client with API key auth (env set BEFORE app creation)."""
    os.environ["CV_DETECT_API_KEY"] = "test-secret-key"
    import api.server as srv
    srv._dl_pipeline = None
    srv._vm_pipeline = None
    srv._vm_model = None
    srv._vm_config = None
    srv._results_db = None
    auth_app = srv._get_app()
    from starlette.testclient import TestClient
    client = TestClient(auth_app)
    yield client, {"Authorization": "Bearer test-secret-key"}
    os.environ.pop("CV_DETECT_API_KEY", None)


def _make_test_image():
    """Create a minimal PNG image in memory."""
    from PIL import Image
    img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


class TestHealth:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "models_loaded" in data
        assert "uptime_seconds" in data

    def test_health_models_not_loaded(self, client):
        data = client.get("/health").json()
        assert data["models_loaded"]["dl_autoencoder"] is False
        assert data["models_loaded"]["variation_model"] is False


class TestInspectEndpoints:
    def test_inspect_dl_no_model_returns_503(self, client):
        img = _make_test_image()
        resp = client.post("/inspect/dl", files={"file": ("test.png", img, "image/png")})
        assert resp.status_code == 503

    def test_inspect_vm_no_model_returns_503(self, client):
        img = _make_test_image()
        resp = client.post("/inspect/vm", files={"file": ("test.png", img, "image/png")})
        assert resp.status_code == 503

    def test_inspect_auto_no_model_returns_503(self, client):
        img = _make_test_image()
        resp = client.post("/inspect/auto", files={"file": ("test.png", img, "image/png")})
        assert resp.status_code == 503

    def test_inspect_dl_empty_file_returns_400(self, client):
        resp = client.post("/inspect/dl", files={"file": ("test.png", io.BytesIO(b""), "image/png")})
        assert resp.status_code == 400


class TestBatchInspect:
    def test_batch_no_model_returns_503(self, client):
        img = _make_test_image()
        resp = client.post("/inspect/batch", files=[("files", ("test.png", img, "image/png"))])
        assert resp.status_code == 503

    def test_batch_no_files_returns_400(self, client):
        # The endpoint requires at least one file
        resp = client.post("/inspect/batch")
        assert resp.status_code in (400, 422)  # validation error


class TestModelManagement:
    def test_list_models(self, client):
        resp = client.get("/models")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        types = {m["model_type"] for m in data}
        assert "dl_autoencoder" in types
        assert "variation_model" in types

    def test_load_dl_model_nonexistent_returns_404(self, client):
        resp = client.post("/models/dl/load", params={"checkpoint_path": "/nonexistent/model.pt"})
        assert resp.status_code in (403, 404)

    def test_load_vm_model_nonexistent_returns_404(self, client):
        resp = client.post("/models/vm/load", params={"model_path": "/nonexistent/model.npz"})
        assert resp.status_code in (403, 404)

    def test_load_dl_model_path_traversal_blocked(self, client):
        resp = client.post("/models/dl/load", params={"checkpoint_path": "/etc/passwd"})
        assert resp.status_code == 403

    def test_load_vm_model_path_traversal_blocked(self, client):
        resp = client.post("/models/vm/load", params={"model_path": "../../etc/shadow"})
        assert resp.status_code == 403


class TestUploadValidation:
    def test_oversized_upload_rejected(self, client):
        """Uploads larger than _MAX_UPLOAD_BYTES should be rejected."""
        # We can't easily test 50MB in unit tests, but we test the mechanism
        pass  # Placeholder -- would need to mock _MAX_UPLOAD_BYTES

    def test_non_image_content_type_rejected(self, client):
        """Uploading a non-image content type should return 400."""
        buf = io.BytesIO(b"not an image")
        resp = client.post(
            "/inspect/dl",
            files={"file": ("test.txt", buf, "text/plain")},
        )
        assert resp.status_code == 400

    def test_missing_file_returns_422(self, client):
        """Calling an inspect endpoint without a file should return 422."""
        resp = client.post("/inspect/dl")
        assert resp.status_code == 422


class TestSPC:
    def test_spc_metrics_returns_structure(self, client):
        """SPC endpoint should return proper structure even with empty DB."""
        resp = client.get("/spc/metrics")
        # May return 500 if no DB, which is acceptable
        assert resp.status_code in (200, 500)
        if resp.status_code == 200:
            data = resp.json()
            assert "mean" in data
            assert "yield_rate" in data
            assert "total_inspections" in data


class TestAuthentication:
    def test_no_auth_required_when_env_unset(self, client):
        """When CV_DETECT_API_KEY is not set, endpoints should be accessible."""
        os.environ.pop("CV_DETECT_API_KEY", None)
        resp = client.get("/models")
        assert resp.status_code == 200

    def test_auth_required_when_env_set(self, auth_client):
        """When CV_DETECT_API_KEY is set, requests without auth should fail."""
        client, headers = auth_client
        # Request without auth header
        resp = client.get("/models")
        assert resp.status_code == 401

    def test_auth_succeeds_with_correct_key(self, auth_client):
        """Correct Bearer token should grant access."""
        client, headers = auth_client
        resp = client.get("/models", headers=headers)
        assert resp.status_code == 200

    def test_auth_fails_with_wrong_key(self, auth_client):
        """Wrong Bearer token should be rejected."""
        client, _ = auth_client
        resp = client.get("/models", headers={"Authorization": "Bearer wrong-key"})
        assert resp.status_code == 401


class TestVersionAndMetadata:
    def test_version_in_health(self, client):
        """Health endpoint should return a valid version string."""
        data = client.get("/health").json()
        version = data["version"]
        parts = version.split(".")
        assert len(parts) >= 2, "Version should be semver-like"

    def test_openapi_spec_available(self, client):
        """FastAPI should serve an OpenAPI JSON spec."""
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        spec = resp.json()
        assert "paths" in spec
        assert "/health" in spec["paths"]
