"""异常热力图绘制：把 anomaly_map 叠加到原图，输出三联图。"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")  # 无显示环境（服务器）下保存图片
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def _normalize(anomaly_map: np.ndarray) -> np.ndarray:
    """归一化到 0~1。"""
    amin, amax = float(anomaly_map.min()), float(anomaly_map.max())
    if amax - amin < 1e-12:
        return np.zeros_like(anomaly_map)
    return (anomaly_map - amin) / (amax - amin)


def build_heatmap(anomaly_map: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    """把异常图 resize 到原图尺寸并上色，返回 BGR 彩色热力图。"""
    norm = _normalize(anomaly_map)
    resized = cv2.resize(norm, size_wh, interpolation=cv2.INTER_CUBIC)
    heat = cv2.applyColorMap((resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return heat


def save_triptych(
    image_path: str | Path,
    anomaly_map: np.ndarray,
    out_path: str | Path,
    score: float | None = None,
    label: str | None = None,
) -> None:
    """保存「原图 | 热力图 | 叠加图」三联图。

    Args:
        image_path: 原始图片路径。
        anomaly_map: 二维异常分数图（H, W），任意尺度（会 resize 到原图）。
        out_path: 输出图片路径。
        score: 图像级 anomaly score，用于标题。
        label: Normal / Defect，用于标题。
    """
    original = cv2.imread(str(image_path))
    if original is None:  # 兜底：用 PIL 读取（兼容部分中文路径）
        pil = Image.open(image_path).convert("RGB")
        original = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    h, w = original.shape[:2]

    heat = build_heatmap(np.asarray(anomaly_map, dtype=np.float32), (w, h))
    overlay = cv2.addWeighted(original, 0.55, heat, 0.45, 0)

    title = f"{label or ''}  score={score:.4f}" if score is not None else (label or "")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, img_bgr, name in zip(
        axes,
        [original, heat, overlay],
        ["Original", "Heatmap", "Overlay"],
    ):
        ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle(f"{Path(image_path).name}  |  {title}")
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
