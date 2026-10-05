"""训练 / 特征库构建统一入口。

用法：
    python -m src.train.train --config configs/patchcore_bottle.yaml

PatchCore 为特征库方法，1 个 epoch 即可完成 memory bank 构建与 coreset 子采样；
EfficientAD 为蒸馏训练方法，所需迭代次数由配置中的 max_epochs 控制（第三阶段）。

批量编排（matrix/ablation）直接调用 run_training(cfg)，无需落临时 YAML。
"""

from __future__ import annotations

import argparse

import yaml
from anomalib.engine import Engine

from src.data.datamodule import build_datamodule
from src.models.factory import build_model
from src.utils.config import (
    experiment_dir,
    find_latest_checkpoint,
    load_config,
    set_seed,
)


def run_training(cfg: dict) -> str | None:
    """按配置字典完成一次训练 / 特征库构建，返回 checkpoint 路径。"""
    set_seed(cfg["experiment"].get("seed", 42))
    out_dir = experiment_dir(cfg)

    # 保存本次配置快照（无论 cfg 来自文件还是批量生成），保证可复现
    with (out_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    datamodule = build_datamodule(cfg)
    model = build_model(cfg)

    e = cfg["engine"]
    engine = Engine(
        default_root_dir=str(out_dir),
        logger=False,
        max_epochs=e.get("max_epochs", 1),
        accelerator=e.get("accelerator", "auto"),
        devices=e.get("devices", 1),
    )
    engine.fit(model=model, datamodule=datamodule)

    ckpt = find_latest_checkpoint(out_dir)
    if ckpt is None:
        print("[train] 警告：未在输出目录找到 checkpoint 文件")
    else:
        print(f"[train] 训练完成，checkpoint: {ckpt}")
    return str(ckpt) if ckpt else None


def train(config_path: str) -> str | None:
    """从配置文件加载并训练。"""
    cfg = load_config(config_path)
    ckpt = run_training(cfg)
    return str(ckpt) if ckpt else None


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 / 特征库构建入口")
    parser.add_argument("--config", required=True, help="YAML 配置文件路径")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
