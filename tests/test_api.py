"""Test per l'API FastAPI (backend/main.py): /predict, /mesh, /report.

Usa un volume reale del dataset di test (data/test/test) per esercitare la
pipeline completa — preprocessing, modello, marching cubes, reportlab —
invece di mockare i singoli passi.
"""

import pytest
from fastapi.testclient import TestClient

from brain_age.config import MRI_TEST_DIR

import main as backend_main

client = TestClient(backend_main.app)


def _sample_mri_path():
    candidates = sorted(MRI_TEST_DIR.glob("*.nii"))
    if not candidates:
        pytest.skip(f"Nessun file MRI di test trovato in {MRI_TEST_DIR}")
    return candidates[0]


@pytest.fixture(scope="module")
def sample_mri_bytes():
    path = _sample_mri_path()
    return path.name, path.read_bytes()


class TestRoot:
    def test_health_check(self):
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestPredict:
    def test_predict_with_cnn(self, sample_mri_bytes):
        if backend_main.cnn_model is None:
            pytest.skip("Modello CNN non disponibile sul server di test")
        filename, contents = sample_mri_bytes
        response = client.post(
            "/predict",
            params={"model": "cnn"},
            files={"file": (filename, contents, "application/octet-stream")},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["model"] == "cnn"
        assert isinstance(body["predicted_age"], float)

    def test_predict_with_ensemble(self, sample_mri_bytes):
        if backend_main.classical_model is None:
            pytest.skip("Modello Ensemble non disponibile sul server di test")
        filename, contents = sample_mri_bytes
        response = client.post(
            "/predict",
            params={"model": "ensemble"},
            files={"file": (filename, contents, "application/octet-stream")},
        )
        assert response.status_code == 200
        assert response.json()["model"] == "ensemble"

    def test_predict_invalid_model_name(self, sample_mri_bytes):
        filename, contents = sample_mri_bytes
        response = client.post(
            "/predict",
            params={"model": "not-a-model"},
            files={"file": (filename, contents, "application/octet-stream")},
        )
        assert response.status_code == 400


class TestMesh:
    def test_mesh_returns_glb(self, sample_mri_bytes):
        filename, contents = sample_mri_bytes
        response = client.post(
            "/mesh",
            files={"file": (filename, contents, "application/octet-stream")},
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "model/gltf-binary"
        assert len(response.content) > 0
        assert response.content[:4] == b"glTF"  # magic header GLB

    def test_mesh_with_gradcam_overlay(self, sample_mri_bytes):
        if backend_main.cnn_model is None:
            pytest.skip("Grad-CAM richiede il modello CNN, non disponibile")
        filename, contents = sample_mri_bytes
        response = client.post(
            "/mesh",
            params={"overlay": "gradcam"},
            files={"file": (filename, contents, "application/octet-stream")},
        )
        assert response.status_code == 200
        assert response.content[:4] == b"glTF"


class TestReport:
    def test_report_returns_pdf(self, sample_mri_bytes):
        filename, contents = sample_mri_bytes
        response = client.post(
            "/report",
            data={"model": "cnn", "predicted_age": "11.2", "chrono_age": "15"},
            files={"file": (filename, contents, "application/octet-stream")},
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/pdf"
        assert response.content[:4] == b"%PDF"

    def test_report_invalid_model(self, sample_mri_bytes):
        filename, contents = sample_mri_bytes
        response = client.post(
            "/report",
            data={"model": "not-a-model", "predicted_age": "11.2", "chrono_age": "15"},
            files={"file": (filename, contents, "application/octet-stream")},
        )
        assert response.status_code == 400
