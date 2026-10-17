"""本地推理 API：python -m uvicorn app.api:app --host 127.0.0.1 --port 8000。"""

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from src.inference.service import (
    MAX_UPLOAD_BYTES,
    ImageTooLargeError,
    InvalidImageError,
    ModelNotFoundError,
    Predictor,
    image_png,
    visualization,
)
from src.utils.config import PROJECT_ROOT

LOGGER = logging.getLogger(__name__)


class PredictionResponse(BaseModel):
    prediction_id: str
    created_at: str
    filename: str
    model_id: str
    model_version: str
    model: str
    category: str
    image_sha256: str
    anomaly_score: float
    threshold: float
    threshold_source: Literal["checkpoint", "manual"]
    comparison: Literal[">"]
    is_anomaly: bool
    label: Literal["Defect", "Normal"]
    framework_label: Literal["Defect", "Normal"]
    inference_ms: float
    image_width: int
    image_height: int
    calibration_status: Literal["legacy_exploratory"]
    heatmap_png_base64: str | None = None
    overlay_png_base64: str | None = None


def create_app(predictor: Predictor | None = None) -> FastAPI:
    service = predictor or Predictor(
        Path(os.environ.get("IAD_RESULTS_DIR", str(PROJECT_ROOT / "results"))),
        device=os.environ.get("IAD_DEVICE", "cpu"),
    )
    application = FastAPI(title="工业视觉异常检测 API", version="0.2.0")
    application.state.predictor = service

    @application.get("/health")
    def health():
        return {"status": "ok", "device": service.device, "models_available": len(service.catalog())}

    @application.get("/models")
    def models():
        return {"models": service.catalog()}

    @application.post("/predict", response_model=PredictionResponse, response_model_exclude_none=True)
    async def predict(
        file: UploadFile = File(...),
        model_id: str = Form(...),
        threshold: float | None = Form(None, ge=0, le=1),
        include_images: bool = Form(False),
    ):
        try:
            content = await file.read(MAX_UPLOAD_BYTES + 1)
            prediction = await run_in_threadpool(service.predict_bytes, model_id, content)
            response = prediction.record(file.filename or "image", threshold)
            if include_images:
                heat, overlay = await run_in_threadpool(visualization, prediction.rgb, prediction.anomaly_map)
                response["heatmap_png_base64"] = base64.b64encode(image_png(heat)).decode("ascii")
                response["overlay_png_base64"] = base64.b64encode(image_png(overlay)).decode("ascii")
            return response
        except ImageTooLargeError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except (InvalidImageError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ModelNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            LOGGER.exception("推理失败，model_id=%s", model_id)
            raise HTTPException(status_code=503, detail="模型暂时不可用，请检查服务日志") from exc
        finally:
            await file.close()

    return application


app = create_app()
