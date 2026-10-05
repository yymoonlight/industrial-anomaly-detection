"""阶段四：实验结果汇总报告。

读取 results/experiments.csv，生成：
- 模型对比透视表（按类别，PatchCore vs EfficientAD）；
- 精度-速度权衡散点（image_AUROC vs 单图耗时）；
- 类别对比分组柱状图；
- coreset ratio、backbone 消融趋势图。

用法：
    python -m src.experiments.report
产物：results/tables/*.csv、results/figures/*.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from src.utils.config import PROJECT_ROOT

NUMERIC = ["image_AUROC", "pixel_AUROC", "AUPRO", "image_F1", "pixel_F1",
           "inference_ms_per_image", "fps"]


def load_experiments(csv_path: Path) -> list[dict[str, Any]]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"未找到 {csv_path}，请先运行实验")
    rows = []
    with csv_path.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            for key in NUMERIC:
                value = row.get(key, "")
                try:
                    row[key] = float(value) if value not in ("", None) else None
                except ValueError:
                    row[key] = None
            rows.append(row)
    return rows


def _save_pivot(rows: list[dict], value_key: str, out_path: Path) -> None:
    table: dict[str, dict[str, float]] = {}
    for r in rows:
        if r["experiment"].startswith("stage3_"):
            table.setdefault(r["category"], {})[r["model"]] = r.get(value_key)
    if not table:
        return
    models = sorted({m for v in table.values() for m in v})
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["category", *models])
        for category in sorted(table):
            writer.writerow([category, *[table[category].get(m, "") for m in models]])


def _new_axes():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _plot_scatter(rows: list[dict], out_path: Path) -> None:
    data = [r for r in rows if r["experiment"].startswith("stage3_")
            and r.get("inference_ms_per_image") and r.get("image_AUROC")]
    if not data:
        return
    plt = _new_axes()
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"patchcore": "#1f77b4", "efficientad": "#d62728"}
    for model in sorted({r["model"] for r in data}):
        sub = [r for r in data if r["model"] == model]
        ax.scatter([r["inference_ms_per_image"] for r in sub],
                   [r["image_AUROC"] for r in sub],
                   s=60, alpha=0.8, color=colors.get(model), label=model)
        for r in sub:
            ax.annotate(r["category"], (r["inference_ms_per_image"], r["image_AUROC"]),
                        fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Inference time per image (ms, lower is better)")
    ax.set_ylabel("Image AUROC (higher is better)")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("Accuracy vs Speed")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_category_bar(rows: list[dict], out_path: Path) -> None:
    table: dict[str, dict[str, float]] = {}
    for r in rows:
        if r["experiment"].startswith("stage3_") and r.get("image_AUROC"):
            table.setdefault(r["category"], {})[r["model"]] = r["image_AUROC"]
    if not table:
        return
    plt = _new_axes()
    categories = sorted(table)
    models = sorted({m for v in table.values() for m in v})
    x = range(len(categories))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, model in enumerate(models):
        offsets = [p + (i - (len(models) - 1) / 2) * width for p in x]
        ax.bar(offsets, [table[c].get(model, 0) for c in categories], width, label=model)
    ax.set_xticks(list(x))
    ax.set_xticklabels(categories, rotation=30, ha="right")
    ax.set_ylabel("Image AUROC")
    ax.set_ylim(0.5, 1.01)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    ax.set_title("Image AUROC by category")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_coreset_trend(rows: list[dict], out_path: Path) -> None:
    sub = [r for r in rows if r["experiment"].startswith("ablation_coreset_sampling_ratio_")
           and r.get("coreset_ratio") is not None]
    if not sub:
        return
    sub.sort(key=lambda r: r["coreset_ratio"])
    plt = _new_axes()
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot([r["coreset_ratio"] for r in sub], [r["image_AUROC"] for r in sub],
             "o-", color="#1f77b4", label="image_AUROC")
    ax1.set_xlabel("Coreset sampling ratio")
    ax1.set_ylabel("Image AUROC", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot([r["coreset_ratio"] for r in sub],
             [r["inference_ms_per_image"] for r in sub], "s--", color="#d62728",
             label="ms/image")
    ax2.set_ylabel("Inference time (ms)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    fig.suptitle("Coreset ratio: accuracy vs speed")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_backbone_bar(rows: list[dict], out_path: Path) -> None:
    sub = [r for r in rows if r["experiment"].startswith("ablation_backbone_")
           and r.get("image_AUROC")]
    if not sub:
        return
    sub.sort(key=lambda r: r["image_AUROC"])
    plt = _new_axes()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar([r["backbone"] for r in sub], [r["image_AUROC"] for r in sub],
           color="#2ca02c", alpha=0.8)
    for r in sub:
        ax.text(r["backbone"], r["image_AUROC"] + 0.002, f"{r['image_AUROC']:.3f}",
                ha="center", fontsize=8)
    ax.set_ylabel("Image AUROC")
    ax.set_ylim(min(r["image_AUROC"] for r in sub) - 0.03, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Backbone comparison (bottle)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def generate_reports(results_dir: Path | None = None) -> None:
    results_dir = results_dir or (PROJECT_ROOT / "results")
    rows = load_experiments(results_dir / "experiments.csv")

    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    for key, filename in (("image_AUROC", "compare_image_auroc.csv"),
                          ("pixel_AUROC", "compare_pixel_auroc.csv"),
                          ("inference_ms_per_image", "compare_speed.csv")):
        _save_pivot(rows, key, tables_dir / filename)

    _plot_scatter(rows, figures_dir / "accuracy_vs_speed.png")
    _plot_category_bar(rows, figures_dir / "image_auroc_by_category.png")
    _plot_coreset_trend(rows, figures_dir / "coreset_trend.png")
    _plot_backbone_bar(rows, figures_dir / "backbone_comparison.png")
    print(f"[report] 报告已生成：{tables_dir}、{figures_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="实验结果汇总报告")
    parser.add_argument("--results-dir", default=None)
    args = parser.parse_args()
    generate_reports(Path(args.results_dir) if args.results_dir else None)


if __name__ == "__main__":
    main()
