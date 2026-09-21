"""配置与通用辅助函数。

约定：
- 所有 YAML 配置放在 configs/ 下；
- 配置中的相对路径一律相对于项目根目录（本文件上两级）解析；
- 实验产物统一落在 <default_root_dir>/<experiment.name>/ 下。
"""

from __future__ import annotations

import csv
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# 项目根目录：src/utils/config.py -> 上两级
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(config_path: str | Path) -> dict[str, Any]:
    """加载 YAML 配置文件。"""
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"配置文件内容格式不正确: {config_path}")
    cfg["_config_path"] = str(config_path)
    return cfg


def resolve_path(path: str | Path) -> Path:
    """将配置中的相对路径解析为相对于项目根目录的绝对路径。"""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def set_seed(seed: int) -> None:
    """固定 Python / NumPy / PyTorch 随机种子，保证可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    try:
        from lightning.pytorch import seed_everything

        seed_everything(seed, workers=True)
    except ImportError:  # pragma: no cover - lightning 为必装依赖，仅作兜底
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def experiment_dir(cfg: dict[str, Any]) -> Path:
    """本次实验的输出目录：<default_root_dir>/<experiment.name>。"""
    root = resolve_path(cfg["engine"]["default_root_dir"])
    out = root / cfg["experiment"]["name"]
    out.mkdir(parents=True, exist_ok=True)
    return out


def find_latest_checkpoint(search_root: str | Path) -> Path | None:
    """查找权重文件，优先 anomalib 维护的 latest 目录，否则取修改时间最新者。

    权重实际路径形如：
        <root>/<Model>/<Dataset>/<category>/latest/weights/lightning/model.ckpt
    """
    root = resolve_path(search_root)
    ckpts = sorted(root.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not ckpts:
        return None
    latest = [p for p in ckpts if "latest" in p.parts]
    return latest[0] if latest else ckpts[0]


def append_experiment_row(csv_path: str | Path, row: dict[str, Any]) -> None:
    """把一次实验的指标追加到 experiments.csv（不存在则写表头）。"""
    csv_path = resolve_path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    row = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **row}
    fieldnames = list(row.keys())
    write_header = not csv_path.is_file()
    with csv_path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
