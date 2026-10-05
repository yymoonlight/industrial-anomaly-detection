"""对比矩阵快速测试：配置展开、参数填充与已完成组合解析。"""

from __future__ import annotations

import csv

from src.experiments.matrix import (
    build_comparison_config,
    build_comparison_configs,
    completed_pairs,
)


def matrix_cfg() -> dict:
    return {
        "matrix": {"categories": ["bottle", "cable"], "models": ["patchcore", "efficientad"]},
        "data_root": "data/x", "imagenet_dir": "data/imagenette", "seed": 1,
        "eval_batch_size": 4, "num_workers": 0, "heatmap_count": 2,
        "train": {
            "patchcore": {"max_epochs": 1, "train_batch_size": 2},
            "efficientad": {"max_epochs": 5, "train_batch_size": 1, "model_size": "s"},
        },
    }


def test_expand_count_and_order() -> None:
    names = [c["experiment"]["name"] for c in build_comparison_configs(matrix_cfg())]
    assert names == [
        "stage3_patchcore_bottle", "stage3_patchcore_cable",
        "stage3_efficientad_bottle", "stage3_efficientad_cable",
    ]


def test_patchcore_params() -> None:
    cfg = build_comparison_config(matrix_cfg(), "patchcore", "bottle")
    assert cfg["model"]["coreset_sampling_ratio"] == 0.1
    assert cfg["engine"]["max_epochs"] == 1
    assert cfg["data"]["category"] == "bottle"


def test_efficientad_params() -> None:
    cfg = build_comparison_config(matrix_cfg(), "efficientad", "cable")
    assert cfg["model"]["imagenet_dir"] == "data/imagenette"
    assert cfg["data"]["train_batch_size"] == 1
    assert cfg["engine"]["max_epochs"] == 5


def test_completed_pairs(tmp_path) -> None:
    csv_path = tmp_path / "experiments.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment"])
        writer.writerow(["stage3_patchcore_bottle"])
        writer.writerow(["other_experiment"])
    assert completed_pairs(csv_path) == {("patchcore", "bottle")}
