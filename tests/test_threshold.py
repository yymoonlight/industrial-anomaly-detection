"""阈值标定快速测试：用构造数据验证最优 F1、约束选点与异常处理。"""

from __future__ import annotations

import numpy as np
import pytest

from src.experiments.threshold import calibrate_threshold


def test_perfect_separation() -> None:
    scores = np.array([0.1, 0.2, 0.15, 0.8, 0.9, 0.85])
    labels = np.array([0, 0, 0, 1, 1, 1])
    result = calibrate_threshold(scores, labels)
    assert 0.2 < result["best_f1"]["threshold"] <= 0.8
    assert result["best_f1"]["f1"] == pytest.approx(1.0)
    assert result["n_normal"] == 3
    assert result["n_defect"] == 3
    assert len(result["sweep"]) >= 6


def test_constrained_points_reachable() -> None:
    scores = np.concatenate([np.linspace(0.1, 0.4, 20), np.linspace(0.6, 0.95, 20)])
    labels = np.array([0] * 20 + [1] * 20)
    result = calibrate_threshold(scores, labels, recall_target=0.95, precision_target=0.95)
    assert result["recall_constrained"]["reachable"]
    assert result["recall_constrained"]["recall"] >= 0.95
    assert result["precision_constrained"]["reachable"]
    assert result["precision_constrained"]["precision"] >= 0.95


def test_constraint_unreachable_flagged() -> None:
    # 每个分数都成对包含正常与缺陷，任何阈值下精度最高约 0.5，达不到 0.99
    scores = np.tile(np.linspace(0, 1, 20), 2)
    labels = np.array([0] * 20 + [1] * 20)
    result = calibrate_threshold(scores, labels, precision_target=0.99)
    assert result["precision_constrained"]["reachable"] is False


def test_single_class_raises() -> None:
    with pytest.raises(ValueError):
        calibrate_threshold(np.array([0.1, 0.2]), np.array([0, 0]))
