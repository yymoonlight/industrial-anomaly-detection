"""真实执行页面脚本：批次导出及修改阈值不触发重复推理。"""

from pathlib import Path

import streamlit as st
from streamlit.testing.v1 import AppTest

from src.inference import service as service_module
from tests.test_inference_service import png_bytes


def test_batch_page_reuses_predictions_when_threshold_changes(service, monkeypatch):
    predictor, counts = service

    class Upload:
        name = "sample.png"

        def getvalue(self):
            return png_bytes()

    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: [Upload()])
    monkeypatch.setattr(service_module, "Predictor", lambda *args, **kwargs: predictor)
    script = Path(__file__).resolve().parents[1] / "app/streamlit_app.py"
    app = AppTest.from_file(str(script)).run(timeout=20)
    assert not app.exception
    next(button for button in app.button if button.label == "开始检测").click().run()
    assert not app.exception
    assert counts["predictions"] == 1
    assert len(app.get("download_button")) == 3
    app.checkbox[0].check().run()
    app.number_input[0].set_value(0.49).run()
    assert not app.exception
    assert counts == {"loads": 1, "predictions": 1}
    assert app.metric[1].value == "1"
