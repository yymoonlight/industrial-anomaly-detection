"""阶段四：PatchCore 控制变量消融的批量编排。

把 baseline 与逐维度扫描展开为配置列表（每次只变一个维度），串行执行
train -> evaluate，支持断点续跑与容错。

用法：
    python -m src.experiments.ablation --config configs/stage4_ablation.yaml
"""

from __future__ import annotations

import argparse
import csv
from typing import Any

from src.evaluation.evaluate import run_evaluation
from src.train.train import run_training
from src.utils.config import PROJECT_ROOT, load_config

METRICS = ["image_auroc", "pixel_auroc", "aupro", "image_f1", "pixel_f1"]


def _value_tag(value: Any) -> str:
    """把一个扫描值转成可用于实验名的标签。"""
    if isinstance(value, (list, tuple)):
        return "_".join(str(v) for v in value)
    return str(value)


def _cfg_from_params(abl_cfg: dict[str, Any], params: dict[str, Any], tag: str) -> dict[str, Any]:
    base = abl_cfg["base"]
    return {
        "experiment": {"name": f"ablation_{tag}", "seed": base.get("seed", 42)},
        "data": {
            "dataset": "mvtec",
            "root": base.get("data_root", "data/mvtec_anomaly_detection"),
            "category": base["category"],
            "train_batch_size": 32,
            "eval_batch_size": base.get("eval_batch_size", 32),
            "num_workers": base.get("num_workers", 8),
        },
        "model": {"name": "patchcore", "pre_trained": True, **params},
        "engine": {"max_epochs": 1, "accelerator": "auto", "devices": 1, "default_root_dir": "results"},
        "evaluation": {
            "metrics": METRICS,
            "save_heatmaps": True,
            "heatmap_count": abl_cfg.get("heatmap_count", 5),
            "predictions_csv": True,
        },
    }


def _fingerprint(params: dict[str, Any]):
    """可哈希的参数指纹（list / tuple 转 tuple）。"""
    return tuple(sorted(
        (k, tuple(v) if isinstance(v, (list, tuple)) else v)
        for k, v in params.items()
    ))


def build_ablation_configs(abl_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """展开 baseline + 逐维度扫描，按参数指纹去重。"""
    baseline_params = abl_cfg["base"]["baseline"]
    configs: list[dict[str, Any]] = [_cfg_from_params(abl_cfg, dict(baseline_params), "baseline")]
    seen = {_fingerprint(baseline_params)}

    for dim, values in abl_cfg.get("sweep", {}).items():
        for value in values:
            params = dict(baseline_params)
            params[dim] = list(value) if isinstance(value, (list, tuple)) else value
            fingerprint = _fingerprint(params)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            configs.append(_cfg_from_params(abl_cfg, params, f"{dim}_{_value_tag(value)}"))
    return configs


def completed_experiments(csv_path) -> set[str]:
    if not csv_path.is_file():
        return set()
    with csv_path.open(encoding="utf-8-sig") as f:
        return {row.get("experiment", "") for row in csv.DictReader(f)}


def run_ablation(ablation_config_path: str, resume: bool | None = None) -> list[dict[str, str]]:
    abl_cfg = load_config(ablation_config_path)
    do_resume = abl_cfg["base"].get("resume", True) if resume is None else resume
    csv_path = PROJECT_ROOT / "results" / "experiments.csv"
    done = completed_experiments(csv_path) if do_resume else set()

    failures: list[dict[str, str]] = []
    configs = build_ablation_configs(abl_cfg)
    for cfg in configs:
        name = cfg["experiment"]["name"]
        if name in done:
            print(f"[ablation] 跳过已完成: {name}")
            continue
        print(f"\n[ablation] ===== {name} =====")
        try:
            ckpt = run_training(cfg)
            if not ckpt:
                raise RuntimeError("训练后未产出 checkpoint")
            run_evaluation(cfg, ckpt)
        except Exception as exc:  # noqa: BLE001
            print(f"[ablation] 失败 {name}: {exc}")
            failures.append({"experiment": name, "error": str(exc)})

    if failures:
        log_path = PROJECT_ROOT / "results" / "logs" / "stage4_failures.csv"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["experiment", "error"])
            writer.writeheader()
            writer.writerows(failures)
        print(f"[ablation] {len(failures)} 组失败，明细见 {log_path}")

    print(f"[ablation] 消融执行结束，共 {len(configs) - len(failures)} 组成功，{len(failures)} 组失败")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="PatchCore 控制变量消融")
    parser.add_argument("--config", default="configs/stage4_ablation.yaml")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    run_ablation(args.config, resume=not args.no_resume)


if __name__ == "__main__":
    main()
