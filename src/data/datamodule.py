"""数据模块工厂：按配置构造 anomalib 数据模块。

支持：
- mvtec：MVTec AD 官方目录结构（train/good、test/<缺陷类型>、ground_truth）
- folder：anomalib Folder 通用结构（normal/abnormal/normal_test/mask），
          用于自定义数据与本地合成冒烟测试
"""

from __future__ import annotations

from typing import Any

from anomalib.data import Folder, MVTecAD

from src.utils.config import resolve_path


def build_datamodule(cfg: dict[str, Any]):
    """根据配置构造 anomalib 数据模块。"""
    d = cfg["data"]
    dataset = d.get("dataset", "mvtec").lower()

    common = {
        "train_batch_size": d.get("train_batch_size", 32),
        "eval_batch_size": d.get("eval_batch_size", 32),
        "num_workers": d.get("num_workers", 4),
    }

    if dataset == "mvtec":
        root = resolve_path(d["root"])
        return MVTecAD(root=root, category=d["category"], **common)

    if dataset == "folder":
        root = resolve_path(d["root"])
        return Folder(
            name=d.get("name", "folder"),
            root=root,
            normal_dir=d["normal_dir"],
            abnormal_dir=d.get("abnormal_dir"),
            normal_test_dir=d.get("normal_test_dir"),
            mask_dir=d.get("mask_dir"),
            **common,
        )

    raise ValueError(f"不支持的 dataset 类型: {dataset}（可选 mvtec / folder）")
