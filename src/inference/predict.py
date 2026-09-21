"""统一推理入口：测试集批量预测（含热力图）与单图/目录推理。

用法：
    # 对整个测试集预测并保存热力图、scores.csv
    python -m src.inference.predict --config configs/patchcore_bottle.yaml --input test

    # 单张图片 / 某个目录
    python -m src.inference.predict --config configs/patchcore_bottle.yaml \
        --input data/mvtec_anomaly_detection/bottle/test/broken_large/000.png
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

from anomalib.engine import Engine

from src.data.datamodule import build_datamodule
from src.inference.heatmap import save_triptych
from src.models.factory import build_model
from src.utils.config import experiment_dir, find_latest_checkpoint, load_config, resolve_path, set_seed


def _build_engine(cfg: dict[str, Any]) -> Engine:
    """构造推理用 Engine；运行时目录放系统临时目录，避免在结果目录留下权重空壳。

    注意：不能传 enable_checkpointing=False，anomalib 模型自带 ModelCheckpoint
    callback，显式关闭会触发 MisconfigurationException；predict 阶段本身不会保存权重。
    """
    import tempfile

    e = cfg["engine"]
    runtime_dir = Path(tempfile.gettempdir()) / "iad_runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    return Engine(
        default_root_dir=str(runtime_dir),
        logger=False,
        accelerator=e.get("accelerator", "auto"),
        devices=e.get("devices", 1),
    )


def predict_testset(
    cfg: dict[str, Any],
    ckpt: str | Path,
    output_dir: str | Path,
    heatmap_count: int = 20,
    save_csv: bool = True,
) -> dict[str, Any]:
    """对测试集批量推理，保存每张图分数与前 N 张三联热力图（异常样本优先）。"""
    out_dir = Path(output_dir)
    heat_dir = out_dir / "heatmaps"
    out_dir.mkdir(parents=True, exist_ok=True)

    datamodule = build_datamodule(cfg)
    model = build_model(cfg)
    engine = _build_engine(cfg)

    start = time.perf_counter()
    predictions = engine.predict(
        model=model,
        datamodule=datamodule,
        ckpt_path=str(ckpt),
        return_predictions=True,
    )
    elapsed = time.perf_counter() - start

    records = []
    for batch in predictions:
        scores = batch.pred_score.detach().cpu()
        pred_labels = batch.pred_label.detach().cpu()
        gt_labels = batch.gt_label.detach().cpu()
        maps = batch.anomaly_map.detach().cpu()  # (B, H, W)
        for i in range(scores.shape[0]):
            anomaly_map = maps[i].numpy()
            if anomaly_map.ndim == 3:  # 兼容 (1, H, W)
                anomaly_map = anomaly_map[0]
            records.append(
                {
                    "image_path": batch.image_path[i],
                    "gt_label": int(gt_labels[i].item()),
                    "pred_label": int(pred_labels[i].item()),
                    "pred_score": float(scores[i].item()),
                    "anomaly_map": anomaly_map,
                }
            )

    # 异常样本优先，取前 heatmap_count 张保存热力图
    records_sorted = sorted(records, key=lambda r: (r["pred_label"], r["pred_score"]), reverse=True)
    for idx, rec in enumerate(records_sorted[:heatmap_count]):
        label = "Defect" if rec["pred_label"] == 1 else "Normal"
        save_triptych(
            rec["image_path"],
            rec["anomaly_map"],
            heat_dir / f"{idx:02d}_{label}_{rec['pred_score']:.4f}.png",
            score=rec["pred_score"],
            label=label,
        )

    if save_csv:
        with (out_dir / "scores.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["image_path", "gt_label", "pred_label", "pred_score"])
            for rec in records:
                writer.writerow([rec["image_path"], rec["gt_label"], rec["pred_label"], f"{rec['pred_score']:.6f}"])

    n = max(len(records), 1)
    summary = {
        "num_images": len(records),
        "inference_time_s": round(elapsed, 3),
        "inference_time_ms_per_image": round(1000 * elapsed / n, 2),
        "fps": round(n / elapsed, 3) if elapsed > 0 else None,
    }
    print(f"[predict] 测试集 {len(records)} 张，平均 {summary['inference_time_ms_per_image']} ms/张，"
          f"热力图已保存到 {heat_dir}")
    return summary


def predict_single(
    cfg: dict[str, Any],
    ckpt: str | Path,
    image_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """对单张图片推理，返回分数、判定并保存三联热力图。"""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_path = resolve_path(image_path)

    model = build_model(cfg)
    engine = _build_engine(cfg)
    predictions = engine.predict(
        model=model,
        data_path=str(image_path),
        ckpt_path=str(ckpt),
        return_predictions=True,
    )

    batch = predictions[0]
    score = float(batch.pred_score.detach().cpu()[0].item())
    pred_label = int(batch.pred_label.detach().cpu()[0].item())
    anomaly_map = batch.anomaly_map.detach().cpu()[0].numpy()
    if anomaly_map.ndim == 3:  # 兼容 (1, H, W)
        anomaly_map = anomaly_map[0]
    label = "Defect" if pred_label == 1 else "Normal"

    save_triptych(
        image_path,
        anomaly_map,
        out_dir / f"{image_path.stem}_{label}.png",
        score=score,
        label=label,
    )
    result = {"image_path": str(image_path), "pred_label": pred_label, "label": label, "pred_score": score}
    print(f"[predict] {image_path.name}: score={score:.4f} -> {label}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="统一推理入口")
    parser.add_argument("--config", required=True, help="YAML 配置文件路径")
    parser.add_argument("--ckpt", default=None, help="checkpoint 路径，默认取实验目录下最新权重")
    parser.add_argument("--input", default="test", help="test（测试集）/ 单张图片路径 / 图片目录")
    parser.add_argument("--output", default=None, help="输出目录，默认实验目录下 predictions")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["experiment"].get("seed", 42))
    out_dir = experiment_dir(cfg)
    ckpt = args.ckpt or find_latest_checkpoint(out_dir)
    if ckpt is None:
        raise FileNotFoundError("未找到 checkpoint，请先运行训练，或通过 --ckpt 指定权重路径")
    pred_dir = Path(args.output) if args.output else out_dir / "predictions"

    if args.input == "test":
        eval_cfg = cfg.get("evaluation", {})
        predict_testset(
            cfg,
            ckpt,
            pred_dir,
            heatmap_count=eval_cfg.get("heatmap_count", 20),
            save_csv=eval_cfg.get("predictions_csv", True),
        )
    else:
        input_path = resolve_path(args.input)
        if input_path.is_dir():
            exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
            images = sorted(p for p in input_path.rglob("*") if p.suffix.lower() in exts)
            for img in images:
                predict_single(cfg, ckpt, img, pred_dir)
        elif input_path.is_file():
            predict_single(cfg, ckpt, input_path, pred_dir)
        else:
            raise FileNotFoundError(f"输入路径不存在: {input_path}")


if __name__ == "__main__":
    main()
