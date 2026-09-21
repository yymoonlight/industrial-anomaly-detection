"""评估指标构造：Image/Pixel AUROC、AUPRO、Image/Pixel F1。

指标基于 anomalib 2.x 的 AnomalibMetric，fields 指定从 ImageBatch 中取的字段：
- pred_score / gt_label：图像级异常分数与标签
- anomaly_map / gt_mask：像素级异常热力图与真值 mask
- pred_label / pred_mask：阈值化后的图像/像素预测（阈值由 post-processor 自适应给出）
"""

from __future__ import annotations

from typing import Any

from anomalib.metrics import AUPRO, AUROC, Evaluator, F1Score

METRIC_BUILDERS = {
    "image_auroc": lambda: AUROC(fields=["pred_score", "gt_label"], prefix="image_"),
    "pixel_auroc": lambda: AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_"),
    "aupro": lambda: AUPRO(fields=["anomaly_map", "gt_mask"]),
    "image_f1": lambda: F1Score(fields=["pred_label", "gt_label"], prefix="image_"),
    "pixel_f1": lambda: F1Score(fields=["pred_mask", "gt_mask"], prefix="pixel_"),
}


def build_evaluator(cfg: dict[str, Any]) -> Evaluator:
    """按配置 evaluation.metrics 构造 Evaluator。"""
    names = cfg.get("evaluation", {}).get("metrics", ["image_auroc", "pixel_auroc"])
    metrics = []
    for name in names:
        if name not in METRIC_BUILDERS:
            raise KeyError(f"未知指标: {name}，可选 {list(METRIC_BUILDERS)}")
        metrics.append(METRIC_BUILDERS[name]())
    return Evaluator(test_metrics=metrics)
