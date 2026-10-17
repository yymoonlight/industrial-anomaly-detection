"""CLI、网页和 API 共用的常驻推理服务；默认 CPU，不启动训练。"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import threading
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from src.utils.config import find_latest_checkpoint, load_config

LOGGER = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 8_000_000
IMAGE_FORMATS = {"PNG", "JPEG", "BMP", "WEBP"}


class InvalidImageError(ValueError):
    """图片内容、尺寸或格式无效。"""


class ImageTooLargeError(InvalidImageError):
    """图片超过体积或像素限制。"""


class ModelNotFoundError(LookupError):
    """未找到指定的已训练模型。"""


def decode_image(content: bytes) -> np.ndarray:
    """按真实文件内容校验图片，返回 RGB；不创建临时图片路径。"""
    if len(content) > MAX_UPLOAD_BYTES:
        raise ImageTooLargeError("图片不能超过 10 MiB")
    if not content:
        raise InvalidImageError("图片内容为空")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as image:
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise ImageTooLargeError("图片不能超过 800 万像素")
                if image.format not in IMAGE_FORMATS or getattr(image, "n_frames", 1) != 1:
                    raise InvalidImageError("请使用单帧 PNG、JPEG、BMP 或 WEBP 图片")
                return np.array(ImageOps.exif_transpose(image).convert("RGB"))
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ImageTooLargeError("图片像素过大") from exc
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise InvalidImageError("文件不是完整、可读取的图片") from exc


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    config_path: Path
    checkpoint: Path
    architecture: str
    category: str

    @property
    def cache_key(self) -> tuple:
        stat = self.checkpoint.stat()
        config_hash = hashlib.sha256(self.config_path.read_bytes()).hexdigest()
        return str(self.checkpoint.resolve()), stat.st_size, stat.st_mtime_ns, config_hash


def discover_models(results_dir: str | Path) -> dict[str, ModelSpec]:
    """只发现实验根目录中的 config+checkpoint，不加载模型。"""
    root = Path(results_dir)
    if not root.is_dir():
        return {}
    items = {}
    for folder in sorted(root.iterdir()):
        config_path = folder / "config.yaml"
        if not folder.is_dir() or not config_path.is_file():
            continue
        try:
            checkpoint = find_latest_checkpoint(folder)
            if checkpoint is None:
                continue
            cfg = load_config(config_path)
            architecture = cfg["model"]["name"].lower()
            if architecture not in {"patchcore", "efficientad"}:
                continue
            items[folder.name] = ModelSpec(folder.name, config_path, checkpoint, architecture,
                                           cfg["data"].get("category", cfg["data"].get("name", "")))
        except (OSError, KeyError, ValueError):
            LOGGER.warning("忽略无效模型目录: %s", folder.name, exc_info=True)
    return items


def image_png(rgb: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(rgb.astype(np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def visualization(rgb: np.ndarray, anomaly_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """沿用图像内相对强度上色，返回 RGB 热力图与叠加图。"""
    import cv2

    from src.inference.heatmap import build_heatmap

    heat = build_heatmap(anomaly_map, (rgb.shape[1], rgb.shape[0]))
    heat_rgb = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return heat_rgb, cv2.addWeighted(rgb, 0.55, heat_rgb, 0.45, 0)


def validate_threshold(threshold: float) -> float:
    value = float(threshold)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("阈值必须是 0 到 1 之间的有限数值")
    return value


@dataclass(frozen=True)
class Prediction:
    prediction_id: str
    created_at: str
    model_id: str
    model_version: str
    architecture: str
    category: str
    image_sha256: str
    rgb: np.ndarray
    anomaly_map: np.ndarray
    score: float
    framework_label: bool
    default_threshold: float
    inference_ms: float

    def record(self, filename: str, threshold: float | None = None) -> dict[str, Any]:
        """滑块只调用纯判定；CLI/UI/API 统一使用与框架相同的严格大于。"""
        value = self.default_threshold if threshold is None else validate_threshold(threshold)
        is_anomaly = self.score > value
        return {
            "prediction_id": self.prediction_id, "created_at": self.created_at,
            "filename": filename.replace("\\", "/").rsplit("/", 1)[-1][:200],
            "model_id": self.model_id, "model_version": self.model_version,
            "model": self.architecture, "category": self.category,
            "image_sha256": self.image_sha256, "anomaly_score": self.score,
            "threshold": value, "threshold_source": "checkpoint" if threshold is None else "manual",
            "comparison": ">", "is_anomaly": is_anomaly, "label": "Defect" if is_anomaly else "Normal",
            "framework_label": "Defect" if self.framework_label else "Normal",
            "inference_ms": round(self.inference_ms, 3),
            "image_width": self.rgb.shape[1], "image_height": self.rgb.shape[0],
            "calibration_status": "legacy_exploratory",
        }


class TorchBackend:
    """一次加载 checkpoint，复用原评估的预处理/后处理，无 Trainer 重建。"""

    def __init__(self, spec: ModelSpec, device: str):
        import inspect

        import torch
        from anomalib.models import EfficientAd, Patchcore

        if device not in {"cpu", "cuda"}:
            raise ValueError("device 仅支持 cpu 或 cuda；应用默认 cpu")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用，请使用 cpu")
        cls = Patchcore if spec.architecture == "patchcore" else EfficientAd
        # 权重来自管理员维护的模型目录；不允许通过 API 上传或指定 checkpoint。
        checkpoint = torch.load(str(spec.checkpoint), map_location="cpu", weights_only=False)
        parameters = inspect.signature(cls.__init__).parameters
        kwargs = {k: v for k, v in checkpoint["hyper_parameters"].items() if k in parameters}
        kwargs.update(evaluator=False, visualizer=False)
        if spec.architecture == "patchcore":
            kwargs["pre_trained"] = False
        self.model = cls(**kwargs)
        # 训练指标对象可能保存 CUDA 设备；部署无需评价器。仅剔除这两类键，模型键严格校验。
        state = {k: v for k, v in checkpoint["state_dict"].items()
                 if not k.startswith(("evaluator.", "visualizer."))}
        self.model.load_state_dict(state, strict=True)
        self.model.to(device).eval()
        self.device = device
        self.default_threshold = float(self.model.post_processor.normalized_image_threshold.item())
        with spec.checkpoint.open("rb") as handle:
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        self.version = digest.hexdigest()[:16]

    def predict(self, rgb: np.ndarray) -> tuple[float, bool, np.ndarray]:
        import torch
        from torchvision.transforms.v2.functional import to_dtype, to_image

        tensor = to_dtype(to_image(rgb), torch.float32, scale=True).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            # Anomalib.forward 使用导出变换（关闭 resize 抗锯齿）；部署沿用评估路径的 transform。
            transform = self.model.pre_processor.transform if self.model.pre_processor else None
            if transform is not None:
                tensor = transform(tensor)
            output = self.model.model(tensor)
            if self.model.post_processor:
                output = self.model.post_processor(output)
        score = float(output.pred_score.detach().cpu().reshape(-1)[0])
        label = bool(output.pred_label.detach().cpu().reshape(-1)[0])
        anomaly_map = output.anomaly_map.detach().cpu().numpy().squeeze()
        return score, label, anomaly_map


class Predictor:
    """每进程最多常驻一个模型，锁保护切换/推理，防止并发重复加载和显存堆积。"""

    def __init__(self, results_dir: str | Path, device: str = "cpu", loader=None, models=None):
        self.results_dir = Path(results_dir)
        self.device = device
        self._loader = loader or TorchBackend
        self._models = models
        self._backend = None
        self._key = None
        self._lock = threading.Lock()

    def catalog(self) -> list[dict[str, str]]:
        return [{"model_id": s.model_id, "model": s.architecture, "category": s.category}
                for s in self._specs().values()]

    def _specs(self) -> dict[str, ModelSpec]:
        return self._models if self._models is not None else discover_models(self.results_dir)

    def predict_bytes(self, model_id: str, content: bytes) -> Prediction:
        rgb = decode_image(content)  # 无效上传在加载权重前拒绝
        spec = self._specs().get(model_id)
        if spec is None:
            raise ModelNotFoundError("未找到已训练模型，请刷新模型列表")
        with self._lock:
            key = spec.cache_key
            if key != self._key:
                # 切换前释放旧实例，避免同时持有多个大模型；载入失败后不复用旧模型。
                self._backend, self._key = None, None
                self._backend = self._loader(spec, self.device)
                self._key = key
            start = time.perf_counter()
            score, framework_label, anomaly_map = self._backend.predict(rgb)
            elapsed = (time.perf_counter() - start) * 1000
            version = self._backend.version
            threshold = validate_threshold(self._backend.default_threshold)
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise RuntimeError("模型返回了无效的归一化分数")
        if anomaly_map.ndim != 2 or not np.isfinite(anomaly_map).all():
            raise RuntimeError("模型返回了无效的异常图")
        result = Prediction(uuid4().hex, datetime.now(timezone.utc).isoformat(), model_id, version,
                            spec.architecture, spec.category, hashlib.sha256(content).hexdigest(),
                            rgb, anomaly_map, score, framework_label, threshold, elapsed)
        LOGGER.info(json.dumps({"event": "prediction", "prediction_id": result.prediction_id,
                                "model_id": model_id, "model_version": version,
                                "image_sha256": result.image_sha256, "inference_ms": round(elapsed, 3)}))
        return result
