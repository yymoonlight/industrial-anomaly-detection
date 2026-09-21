"""模型评估统一入口：计算指标、保存 metrics.json、追加 experiments.csv，并生成热力图。

用法：
    python -m src.evaluation.evaluate --config configs/patchcore_bottle.yaml
"""

from __future__ import annotations

import argparse
import json

from anomalib.engine import Engine

from src.data.datamodule import build_datamodule
from src.inference.predict import predict_testset
from src.models.factory import build_model
from src.utils.config import (
    PROJECT_ROOT,
    append_experiment_row,
    experiment_dir,
    find_latest_checkpoint,
    load_config,
    set_seed,
)


def evaluate(config_path: str, ckpt_path: str | None = None) -> dict:
    """加载权重完成一次完整评估，落盘指标与热力图。"""
    cfg = load_config(config_path)
    set_seed(cfg["experiment"].get("seed", 42))
    out_dir = experiment_dir(cfg)

    datamodule = build_datamodule(cfg)
    model = build_model(cfg)
    e = cfg["engine"]
    engine = Engine(
        default_root_dir=str(out_dir),
        logger=False,
        accelerator=e.get("accelerator", "auto"),
        devices=e.get("devices", 1),
    )

    ckpt = ckpt_path or find_latest_checkpoint(out_dir)
    if ckpt is None:
        raise FileNotFoundError("未找到 checkpoint，请先运行 src.train.train")

    print(f"[evaluate] 使用权重: {ckpt}")
    test_results = engine.test(model=model, datamodule=datamodule, ckpt_path=str(ckpt))
    metrics = test_results[0] if test_results else {}

    # 保存指标 JSON
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[evaluate] 指标: {json.dumps(metrics, indent=2)}")

    # 测试集推理：热力图 + 分数 + 速度
    timing = {}
    eval_cfg = cfg.get("evaluation", {})
    if eval_cfg.get("save_heatmaps", True):
        timing = predict_testset(
            cfg,
            ckpt,
            out_dir / "predictions",
            heatmap_count=eval_cfg.get("heatmap_count", 20),
            save_csv=eval_cfg.get("predictions_csv", True),
        )

    # 追加到 results/experiments.csv
    row = {
        "experiment": cfg["experiment"]["name"],
        "model": cfg["model"]["name"],
        "category": cfg["data"].get("category", cfg["data"].get("name", "")),
        "backbone": cfg["model"].get("backbone", ""),
        "coreset_ratio": cfg["model"].get("coreset_sampling_ratio", ""),
        "image_AUROC": metrics.get("image_AUROC", ""),
        "pixel_AUROC": metrics.get("pixel_AUROC", ""),
        "AUPRO": metrics.get("AUPRO", ""),
        "image_F1": metrics.get("image_F1Score", ""),
        "pixel_F1": metrics.get("pixel_F1Score", ""),
        "inference_ms_per_image": timing.get("inference_time_ms_per_image", ""),
        "fps": timing.get("fps", ""),
        "ckpt": str(ckpt),
    }
    append_experiment_row(PROJECT_ROOT / "results" / "experiments.csv", row)
    print("[evaluate] 已追加到 results/experiments.csv")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="模型评估入口")
    parser.add_argument("--config", required=True, help="YAML 配置文件路径")
    parser.add_argument("--ckpt", default=None, help="checkpoint 路径，默认取实验目录最新权重")
    args = parser.parse_args()
    evaluate(args.config, args.ckpt)


if __name__ == "__main__":
    main()
