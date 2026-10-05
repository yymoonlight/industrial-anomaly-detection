"""模型工厂：按配置构造 PatchCore / EfficientAD，并挂上评估器。"""

from __future__ import annotations

from typing import Any

from anomalib.models import EfficientAd, Patchcore
from anomalib.models.image.efficient_ad.torch_model import EfficientAdModelSize

from src.evaluation.metrics import build_evaluator

# EfficientAD 模型规模：str 枚举只能按 value('small'/'medium') 或按 name 索引，
# 不能用 EfficientAdModelSize('S')（会 ValueError），这里统一用 name 映射。
EFFICIENTAD_SIZES: dict[str, EfficientAdModelSize] = {
    "s": EfficientAdModelSize.S,
    "m": EfficientAdModelSize.M,
}


def build_model(cfg: dict[str, Any]):
    """根据配置构造 anomalib 模型实例。"""
    m = cfg["model"]
    name = m["name"].lower()
    evaluator = build_evaluator(cfg)

    if name == "patchcore":
        return Patchcore(
            backbone=m.get("backbone", "wide_resnet50_2"),
            layers=tuple(m.get("layers", ["layer2", "layer3"])),
            pre_trained=m.get("pre_trained", True),
            coreset_sampling_ratio=m.get("coreset_sampling_ratio", 0.1),
            num_neighbors=m.get("num_neighbors", 9),
            evaluator=evaluator,
        )

    if name == "efficientad":
        size = m.get("model_size", "s").lower()
        if size not in EFFICIENTAD_SIZES:
            raise ValueError(f"EfficientAD model_size 仅支持 s / m，收到: {size}")
        return EfficientAd(
            imagenet_dir=m.get("imagenet_dir", "data/imagenette"),
            model_size=EFFICIENTAD_SIZES[size],
            teacher_out_channels=m.get("teacher_out_channels", 384),
            lr=m.get("lr", 1e-4),
            weight_decay=m.get("weight_decay", 1e-5),
            evaluator=evaluator,
        )

    raise ValueError(f"不支持的模型: {name}（可选 patchcore / efficientad）")
