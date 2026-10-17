"""API 合同测试不下载模型、不训练；使用可计数的模型适配器。"""

from __future__ import annotations

import base64
import io

from fastapi.testclient import TestClient
from PIL import Image

from app.api import create_app
from tests.test_inference_service import png_bytes


def test_health_catalog_and_prediction_share_threshold_contract(service):
    predictor, counts = service
    with TestClient(create_app(predictor)) as client:
        assert client.get("/health").json()["device"] == "cpu"
        assert client.get("/models").json()["models"][0]["model_id"] == "bottle"
        response = client.post("/predict", data={"model_id": "bottle", "include_images": "true"},
                               files={"file": ("test.png", png_bytes(), "image/png")})
        assert response.status_code == 200
        record = response.json()
        assert record["is_anomaly"] is False  # score == threshold，统一为 >
        assert record["threshold_source"] == "checkpoint"
        assert record["comparison"] == ">"
        image = Image.open(io.BytesIO(base64.b64decode(record["overlay_png_base64"])))
        assert image.size == (16, 16)
        response = client.post("/predict", data={"model_id": "bottle", "threshold": "0.49"},
                               files={"file": ("test.png", png_bytes(), "image/png")})
        assert response.json()["is_anomaly"] is True
        assert counts["loads"] == 1


def test_api_errors_are_actionable_and_do_not_expose_model_paths(service):
    predictor, _ = service
    with TestClient(create_app(predictor)) as client:
        assert client.post("/predict", data={"model_id": "unknown"},
                           files={"file": ("test.png", png_bytes())}).status_code == 404
        assert client.post("/predict", data={"model_id": "bottle"},
                           files={"file": ("fake.png", b"not an image")}).status_code == 422
        for threshold in ("nan", "1.1", "-1"):
            response = client.post("/predict", data={"model_id": "bottle", "threshold": threshold},
                                   files={"file": ("test.png", png_bytes())})
            assert response.status_code == 422
        assert client.post("/predict", data={"model_id": "bottle"}).status_code == 422


def test_model_failure_returns_service_unavailable(service):
    predictor, _ = service

    def fail(spec, device):
        raise RuntimeError("private/checkpoint/path")

    predictor._loader = fail
    with TestClient(create_app(predictor)) as client:
        response = client.post("/predict", data={"model_id": "bottle"},
                               files={"file": ("test.png", png_bytes())})
        assert response.status_code == 503
        assert "private" not in response.text
