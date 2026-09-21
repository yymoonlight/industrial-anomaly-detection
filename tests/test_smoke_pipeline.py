"""端到端冒烟（CPU，标记 slow）：合成数据上跑通 train -> evaluate -> 单图推理。

覆盖：特征库构建、checkpoint 保存、AUROC/AUPRO/F1 计算、
scores.csv、三联热力图、单图 data_path 推理（Streamlit Demo 的技术路径）。

运行：
    python -m pytest tests/test_smoke_pipeline.py -v -s
"""

import csv
from pathlib import Path

import pytest
import yaml

from src.evaluation import evaluate as evaluate_mod
from src.inference.predict import predict_single
from src.train import train as train_mod


def _write_config(case_dir: Path) -> tuple[Path, dict]:
    cfg = {
        "experiment": {"name": "smoke_patchcore", "seed": 42},
        "data": {
            "dataset": "folder",
            "name": "synthetic",
            "root": str(case_dir),
            "normal_dir": "good",
            "abnormal_dir": "abnormal",
            "normal_test_dir": "good_test",
            "mask_dir": "mask",
            "train_batch_size": 2,
            "eval_batch_size": 2,
            "num_workers": 0,
        },
        "model": {
            "name": "patchcore",
            "backbone": "wide_resnet50_2",
            "layers": ["layer2", "layer3"],
            "pre_trained": True,
            "coreset_sampling_ratio": 0.1,
            "num_neighbors": 9,
        },
        "engine": {
            "max_epochs": 1,
            "accelerator": "cpu",
            "devices": 1,
            "default_root_dir": str(case_dir / "results"),
        },
        "evaluation": {
            "metrics": ["image_auroc", "pixel_auroc", "aupro", "image_f1", "pixel_f1"],
            "save_heatmaps": True,
            "heatmap_count": 4,
            "predictions_csv": True,
        },
    }
    cfg_path = case_dir / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return cfg_path, cfg


@pytest.mark.slow
def test_train_evaluate_predict(synthetic_dataset, monkeypatch):
    case_dir = synthetic_dataset
    cfg_path, cfg = _write_config(case_dir)
    # 避免冒烟结果污染项目真实 results/experiments.csv
    monkeypatch.setattr(evaluate_mod, "PROJECT_ROOT", case_dir)

    # 1) 训练 / 特征库构建
    ckpt = train_mod.train(str(cfg_path))
    assert ckpt is not None
    assert Path(ckpt).is_file()

    # 2) 评估（指标 + 热力图 + scores.csv + experiments.csv）
    metrics = evaluate_mod.evaluate(str(cfg_path))
    for key in ("image_AUROC", "pixel_AUROC", "AUPRO", "image_F1Score", "pixel_F1Score"):
        assert key in metrics, f"缺少指标 {key}"
        assert 0.0 <= metrics[key] <= 1.0

    out_dir = case_dir / "results" / "smoke_patchcore"
    assert (out_dir / "metrics.json").is_file()
    assert (out_dir / "config.yaml").is_file()
    heatmaps = list((out_dir / "predictions" / "heatmaps").glob("*.png"))
    assert len(heatmaps) == 4
    assert (out_dir / "predictions" / "scores.csv").is_file()

    # experiments.csv 表头与一行记录
    with (case_dir / "results" / "experiments.csv").open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["model"] == "patchcore"

    # 3) 单图推理（Demo 路径）
    img = sorted((case_dir / "abnormal").glob("*.png"))[0]
    result = predict_single(cfg, ckpt, img, case_dir / "single_out")
    assert result["label"] in ("Normal", "Defect")
    assert isinstance(result["pred_score"], float)
    assert list((case_dir / "single_out").glob("*.png"))
