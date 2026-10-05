"""报告生成快速测试：用伪造 experiments.csv 验证透视表与图表产出。"""

from __future__ import annotations

import csv

from src.experiments.report import generate_reports

KEYS = ["timestamp", "experiment", "model", "category", "backbone", "coreset_ratio",
        "image_AUROC", "pixel_AUROC", "AUPRO", "image_F1", "pixel_F1",
        "inference_ms_per_image", "fps", "ckpt"]


def _row(experiment: str, model: str, category: str, backbone: str, ratio: str,
         auroc: str, ms: str) -> dict:
    return {"experiment": experiment, "model": model, "category": category,
            "backbone": backbone, "coreset_ratio": ratio,
            "image_AUROC": auroc, "pixel_AUROC": "0.97", "AUPRO": "0.9",
            "image_F1": "0.95", "pixel_F1": "0.9",
            "inference_ms_per_image": ms, "fps": "20", "ckpt": "x"}


def test_generate_reports(tmp_path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    rows = [
        _row("stage3_patchcore_bottle", "patchcore", "bottle", "wide_resnet50_2", "0.1", "0.99", "50"),
        _row("stage3_efficientad_bottle", "efficientad", "bottle", "", "", "0.98", "10"),
        _row("ablation_coreset_sampling_ratio_0.01", "patchcore", "bottle",
             "wide_resnet50_2", "0.01", "0.95", "20"),
        _row("ablation_backbone_resnet18", "patchcore", "bottle", "resnet18", "0.1", "0.90", "15"),
    ]
    with (results / "experiments.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=KEYS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in KEYS})

    generate_reports(results)

    assert (results / "tables" / "compare_image_auroc.csv").is_file()
    assert (results / "figures" / "accuracy_vs_speed.png").is_file()
    assert (results / "figures" / "image_auroc_by_category.png").is_file()
    assert (results / "figures" / "coreset_trend.png").is_file()
    assert (results / "figures" / "backbone_comparison.png").is_file()
