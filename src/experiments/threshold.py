"""阶段四：异常判定阈值标定（阈值工程）。

框架默认阈值由 post-processor 在验证集上自适应给出，未必符合产线对
漏检 / 误检的代价偏好。本模块基于测试集连续异常分数：
- 扫描全部候选阈值，给出 Precision / Recall / F1 / Specificity；
- 取 F1 最优阈值；
- 按业务代价选点：漏检代价高（召回约束下取精度最高）、误检代价高（精度约束下取召回最高）；
- 与框架自适应阈值（scores.csv 中的 pred_label）对比；
- 输出 threshold.json、threshold_sweep.csv、pr_curve.png。

用法：
    python -m src.experiments.threshold --experiment results/ablation_baseline
    python -m src.experiments.threshold --all
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.config import PROJECT_ROOT


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def calibrate_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    recall_target: float = 0.99,
    precision_target: float = 0.95,
) -> dict[str, Any]:
    """对连续异常分数做阈值标定。labels 中 1 表示缺陷、0 表示正常。"""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if labels.sum() == 0 or labels.sum() == len(labels):
        raise ValueError("阈值标定需要同时包含正常与缺陷样本")

    # 候选阈值：每个唯一分数处（pred_score >= t），并补充两端
    thresholds = np.unique(scores)
    thresholds = np.concatenate(([thresholds[0] - 1e-6], thresholds, [thresholds[-1] + 1e-6]))

    rows = []
    pred_all = scores[None, :] >= thresholds[:, None]
    for t, pred in zip(thresholds, pred_all):
        tp = int(np.sum(pred & (labels == 1)))
        fp = int(np.sum(pred & (labels == 0)))
        fn = int(np.sum(~pred & (labels == 1)))
        tn = int(np.sum(~pred & (labels == 0)))
        precision, recall, f1 = _prf(tp, fp, fn)
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        rows.append({"threshold": float(t), "TP": tp, "FP": fp, "TN": tn, "FN": fn,
                     "precision": precision, "recall": recall, "f1": f1,
                     "specificity": specificity})

    sweep = np.array([[r["precision"], r["recall"], r["f1"]] for r in rows])

    def pick(metric_idx: int, constraint_idx: int, target: float) -> dict[str, Any]:
        ok = [r for r in rows if r[("precision", "recall")[constraint_idx]] >= target]
        if not ok:  # 目标不可达，取约束指标最高者并标注
            r = max(rows, key=lambda x: x[("precision", "recall")[constraint_idx]])
            return {**{k: r[k] for k in ("threshold", "precision", "recall", "f1")},
                    "target": target, "reachable": False}
        # 满足约束后优化另一指标，并列时取阈值更高（更保守）
        r = max(ok, key=lambda x: (x[("recall", "precision")[constraint_idx]], x["threshold"]))
        return {**{k: r[k] for k in ("threshold", "precision", "recall", "f1")},
                "target": target, "reachable": True}

    best = rows[int(np.argmax(sweep[:, 2]))]
    return {
        "n_images": int(len(labels)),
        "n_normal": int(np.sum(labels == 0)),
        "n_defect": int(np.sum(labels == 1)),
        "best_f1": {k: best[k] for k in ("threshold", "precision", "recall", "f1")},
        "recall_constrained": pick(None, 1, recall_target),
        "precision_constrained": pick(None, 0, precision_target),
        "sweep": rows,
    }


def _plot_pr_curve(labels: np.ndarray, scores: np.ndarray, result: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    precision, recall, _ = precision_recall_curve(labels, scores)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, lw=2, label="PR curve")
    bf = result["best_f1"]
    ax.scatter([bf["recall"]], [bf["precision"]], color="red", zorder=5,
               label=f"best F1={bf['f1']:.3f} @t={bf['threshold']:.3f}")
    rc = result["recall_constrained"]
    ax.scatter([rc["recall"]], [rc["precision"]], color="green", marker="^", zorder=5,
               label=f"recall>={rc['target']}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title("Precision-Recall")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fit_experiment_threshold(
    exp_dir: Path,
    recall_target: float = 0.99,
    precision_target: float = 0.95,
    out_root: Path | None = None,
) -> dict[str, Any]:
    """读取单个实验的 scores.csv 完成标定并落盘。"""
    exp_dir = Path(exp_dir)
    scores_csv = exp_dir / "predictions" / "scores.csv"
    if not scores_csv.is_file():
        raise FileNotFoundError(f"缺少 {scores_csv}，请先完成评估")

    labels, scores, framework_pred = [], [], []
    with scores_csv.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            labels.append(int(row["gt_label"]))
            scores.append(float(row["pred_score"]))
            framework_pred.append(int(row["pred_label"]))
    labels_arr = np.array(labels)
    scores_arr = np.array(scores)

    result = calibrate_threshold(scores_arr, labels_arr, recall_target, precision_target)

    # 框架自适应阈值的实际表现（对比基线）
    pred_arr = np.array(framework_pred)
    tp = int(np.sum(pred_arr & (labels_arr == 1)))
    fp = int(np.sum(pred_arr & (labels_arr == 0)))
    fn = int(np.sum((1 - pred_arr) & (labels_arr == 1)))
    p, r, framework_f1 = _prf(tp, fp, fn)
    result["framework_adaptive"] = {"precision": p, "recall": r, "f1": framework_f1}
    result["experiment"] = exp_dir.name
    result["calibration_status"] = "test_exploratory"
    result["comparison"] = ">="

    out_dir = out_root or (exp_dir / "threshold")
    out_dir.mkdir(parents=True, exist_ok=True)

    serializable = {k: v for k, v in result.items() if k != "sweep"}
    with (out_dir / "threshold.json").open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    with (out_dir / "threshold_sweep.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["threshold", "TP", "FP", "TN", "FN",
                                               "precision", "recall", "f1", "specificity"])
        writer.writeheader()
        writer.writerows(result["sweep"])

    _plot_pr_curve(labels_arr, scores_arr, result, out_dir / "pr_curve.png")
    print(f"[threshold] {exp_dir.name}: best F1={result['best_f1']['f1']:.3f} "
          f"(框架自适应 F1={framework_f1:.3f})，产物见 {out_dir}")
    return serializable


def run_all_thresholds(results_root: Path | None = None,
                       recall_target: float = 0.99, precision_target: float = 0.95) -> list[dict]:
    """对 results 下所有含 predictions/scores.csv 的实验批量标定。"""
    results_root = results_root or (PROJECT_ROOT / "results")
    summary = []
    for scores_csv in sorted(results_root.rglob("scores.csv")):
        exp_dir = scores_csv.parents[1]
        try:
            summary.append(fit_experiment_threshold(
                exp_dir, recall_target, precision_target,
                out_root=PROJECT_ROOT / "results" / "thresholds" / exp_dir.name))
        except Exception as exc:  # noqa: BLE001
            print(f"[threshold] 跳过 {exp_dir.name}: {exc}")
    print(f"[threshold] 批量标定完成，共 {len(summary)} 个实验")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="异常判定阈值标定")
    parser.add_argument("--experiment", help="单个实验目录")
    parser.add_argument("--all", action="store_true", help="标定 results 下全部实验")
    parser.add_argument("--recall-target", type=float, default=0.99)
    parser.add_argument("--precision-target", type=float, default=0.95)
    args = parser.parse_args()
    if args.all:
        run_all_thresholds(recall_target=args.recall_target, precision_target=args.precision_target)
    elif args.experiment:
        fit_experiment_threshold(Path(args.experiment), args.recall_target, args.precision_target)
    else:
        parser.error("请指定 --experiment 或 --all")


if __name__ == "__main__":
    main()
