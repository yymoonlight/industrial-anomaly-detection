"""阶段三：模型 × 类别 对比矩阵的批量编排。

职责：
- 把矩阵配置展开为每个 (model, category) 的完整 cfg（纯逻辑，便于单测）；
- 串行执行 train -> evaluate（单卡，避免显存竞争）；
- 断点续跑：已在 experiments.csv 中的组合跳过；
- 容错：单个组合失败记录到 results/logs/stage3_failures.csv，不影响后续。

用法：
    python -m src.experiments.matrix --config configs/stage3_matrix.yaml
"""

from __future__ import annotations

import argparse
import csv
from typing import Any

from src.evaluation.evaluate import run_evaluation
from src.train.train import run_training
from src.utils.config import PROJECT_ROOT, load_config

METRICS = ["image_auroc", "pixel_auroc", "aupro", "image_f1", "pixel_f1"]


def build_comparison_config(matrix_cfg: dict[str, Any], model_name: str, category: str) -> dict[str, Any]:
    """为单个 (模型, 类别) 组合生成完整配置。"""
    train_spec = matrix_cfg.get("train", {}).get(model_name, {})
    cfg: dict[str, Any] = {
        "experiment": {"name": f"stage3_{model_name}_{category}", "seed": matrix_cfg.get("seed", 42)},
        "data": {
            "dataset": "mvtec",
            "root": matrix_cfg.get("data_root", "data/mvtec_anomaly_detection"),
            "category": category,
            "train_batch_size": train_spec.get("train_batch_size", 32),
            "eval_batch_size": matrix_cfg.get("eval_batch_size", 32),
            "num_workers": matrix_cfg.get("num_workers", 8),
        },
        "model": {"name": model_name},
        "engine": {
            "max_epochs": train_spec.get("max_epochs", 1),
            "accelerator": "auto",
            "devices": 1,
            "default_root_dir": "results",
        },
        "evaluation": {
            "metrics": METRICS,
            "save_heatmaps": True,
            "heatmap_count": matrix_cfg.get("heatmap_count", 10),
            "predictions_csv": True,
        },
    }

    if model_name == "patchcore":
        cfg["model"].update(
            backbone=train_spec.get("backbone", "wide_resnet50_2"),
            layers=train_spec.get("layers", ["layer2", "layer3"]),
            pre_trained=True,
            coreset_sampling_ratio=train_spec.get("coreset_sampling_ratio", 0.1),
            num_neighbors=train_spec.get("num_neighbors", 9),
        )
    elif model_name == "efficientad":
        cfg["model"].update(
            model_size=train_spec.get("model_size", "s"),
            imagenet_dir=matrix_cfg.get("imagenet_dir", "data/imagenette"),
            teacher_out_channels=train_spec.get("teacher_out_channels", 384),
            lr=train_spec.get("lr", 1e-4),
            weight_decay=train_spec.get("weight_decay", 1e-5),
        )
    else:
        raise ValueError(f"矩阵中不支持的模型: {model_name}")
    return cfg


def build_comparison_configs(matrix_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """展开整个矩阵，顺序为模型优先、类别其次。"""
    models = matrix_cfg["matrix"]["models"]
    categories = matrix_cfg["matrix"]["categories"]
    return [build_comparison_config(matrix_cfg, m, c) for m in models for c in categories]


def completed_pairs(csv_path) -> set[tuple[str, str]]:
    """从 experiments.csv 解析已完成的 stage3 (model, category)。"""
    csv_path = PROJECT_ROOT / csv_path if not hasattr(csv_path, "is_file") else csv_path
    if not csv_path.is_file():
        return set()
    pairs: set[tuple[str, str]] = set()
    with csv_path.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            name = row.get("experiment", "")
            if not name.startswith("stage3_"):
                continue
            rest = name[len("stage3_"):]
            for model_name in ("patchcore", "efficientad"):
                if rest.startswith(model_name + "_"):
                    pairs.add((model_name, rest[len(model_name) + 1:]))
    return pairs


def run_comparison(matrix_config_path: str, resume: bool | None = None) -> list[dict[str, str]]:
    """执行整个对比矩阵，返回失败组合列表。"""
    matrix_cfg = load_config(matrix_config_path)
    do_resume = matrix_cfg.get("resume", True) if resume is None else resume
    csv_path = PROJECT_ROOT / "results" / "experiments.csv"
    done = completed_pairs(csv_path) if do_resume else set()

    failures: list[dict[str, str]] = []
    for cfg in build_comparison_configs(matrix_cfg):
        model_name = cfg["model"]["name"]
        category = cfg["data"]["category"]
        if (model_name, category) in done:
            print(f"[matrix] 跳过已完成: {model_name}/{category}")
            continue
        print(f"\n[matrix] ===== {model_name} / {category} =====")
        try:
            ckpt = run_training(cfg)
            if not ckpt:
                raise RuntimeError("训练后未产出 checkpoint")
            run_evaluation(cfg, ckpt)
        except Exception as exc:  # noqa: BLE001 - 批量任务需容错并继续
            print(f"[matrix] 失败 {model_name}/{category}: {exc}")
            failures.append({"model": model_name, "category": category, "error": str(exc)})

    if failures:
        log_path = PROJECT_ROOT / "results" / "logs" / "stage3_failures.csv"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["model", "category", "error"])
            writer.writeheader()
            writer.writerows(failures)
        print(f"[matrix] {len(failures)} 个组合失败，明细见 {log_path}")

    print(f"[matrix] 对比矩阵执行结束，成功 {len(build_comparison_configs(matrix_cfg)) - len(failures)} 组，"
          f"失败 {len(failures)} 组")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="多模型多类别对比矩阵")
    parser.add_argument("--config", default="configs/stage3_matrix.yaml", help="矩阵配置文件")
    parser.add_argument("--no-resume", action="store_true", help="不跳过已完成组合")
    args = parser.parse_args()
    run_comparison(args.config, resume=not args.no_resume)


if __name__ == "__main__":
    main()
