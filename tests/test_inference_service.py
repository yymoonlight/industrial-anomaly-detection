"""无训练测试：模型复用、并发、版本失效、输入边界和统一业务判定。"""

from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from src.inference.service import (
    MAX_UPLOAD_BYTES,
    ImageTooLargeError,
    InvalidImageError,
    ModelNotFoundError,
    decode_image,
)


def png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), (80, 90, 100)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_concurrent_requests_load_once_and_threshold_changes_do_not_infer(service):
    predictor, counts = service
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: predictor.predict_bytes("bottle", png_bytes()), range(4)))
    assert counts == {"loads": 1, "predictions": 4}
    prediction = results[0]
    assert prediction.record("normal.png")["is_anomaly"] is False
    assert prediction.record("normal.png", 0.49)["is_anomaly"] is True
    assert prediction.record("normal.png", 0.5)["is_anomaly"] is False
    assert counts["predictions"] == 4


def test_checkpoint_and_config_changes_invalidate_resident_model(service):
    predictor, counts = service
    predictor.predict_bytes("bottle", png_bytes())
    (predictor.results_dir / "bottle/model.ckpt").write_bytes(b"new checkpoint bytes")
    predictor.predict_bytes("bottle", png_bytes())
    config = predictor.results_dir / "bottle/config.yaml"
    config.write_text(config.read_text(encoding="utf-8") + "experiment:\n  seed: 5\n", encoding="utf-8")
    predictor.predict_bytes("bottle", png_bytes())
    assert counts["loads"] == 3


def test_invalid_inputs_do_not_load_model(service, monkeypatch):
    predictor, counts = service
    for content in (b"", b"not an image"):
        with pytest.raises(InvalidImageError):
            predictor.predict_bytes("bottle", content)
    with pytest.raises(ImageTooLargeError):
        predictor.predict_bytes("bottle", b"x" * (MAX_UPLOAD_BYTES + 1))
    monkeypatch.setattr("src.inference.service.MAX_IMAGE_PIXELS", 100)
    with pytest.raises(ImageTooLargeError):
        predictor.predict_bytes("bottle", png_bytes())
    assert counts["loads"] == 0


def test_unknown_model_and_invalid_threshold(service):
    predictor, _ = service
    with pytest.raises(ModelNotFoundError):
        predictor.predict_bytes("../../anything", png_bytes())
    prediction = predictor.predict_bytes("bottle", png_bytes())
    for threshold in (-0.1, 1.1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            prediction.record("test.png", threshold)
    record = prediction.record("C:\\private\\test.png")
    assert record["filename"] == "test.png"
    assert record["model_version"] == "test-weights"
    assert record["calibration_status"] == "legacy_exploratory"
    assert "rgb" not in record


def test_grayscale_is_converted_to_rgb():
    buffer = io.BytesIO()
    Image.new("L", (10, 10)).save(buffer, format="PNG")
    assert decode_image(buffer.getvalue()).shape == (10, 10, 3)


def test_backend_preserves_evaluation_transform_instead_of_export_transform():
    import torch

    from src.inference.service import TorchBackend

    calls = []

    def transform(tensor):
        calls.append("evaluation_transform")
        return tensor + 0.25

    def forward(tensor):
        assert tensor.mean().item() == pytest.approx(0.25)
        return SimpleNamespace(pred_score=torch.tensor([0.5]), pred_label=torch.tensor([False]),
                               anomaly_map=torch.zeros((1, 1, 4, 4)))

    backend = TorchBackend.__new__(TorchBackend)
    backend.device = "cpu"
    backend.model = SimpleNamespace(pre_processor=SimpleNamespace(transform=transform), model=forward,
                                    post_processor=lambda output: output)
    score, label, anomaly_map = backend.predict(np.zeros((4, 4, 3), dtype=np.uint8))
    assert calls == ["evaluation_transform"]
    assert score == 0.5 and label is False and anomaly_map.shape == (4, 4)
