"""仅做 CPU 推理一致性验收，不训练；对比常驻服务与原 Engine 路径。"""

from __future__ import annotations

import argparse
import copy
import json

from src.inference.predict import _build_engine, build_predictor
from src.models.factory import build_model
from src.utils.config import experiment_dir, find_latest_checkpoint, load_config, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="已有权重的 CPU 推理一致性验收，不训练")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt")
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--output", default="results/application_cpu_smoke.json")
    args = parser.parse_args()
    import torch

    torch.set_num_threads(4)
    cfg = load_config(args.config)
    ckpt = resolve_path(args.ckpt) if args.ckpt else find_latest_checkpoint(experiment_dir(cfg))
    if ckpt is None:
        raise FileNotFoundError("需要已有 checkpoint；此脚本不会训练或下载模型")
    service = build_predictor(cfg, ckpt, device="cpu")
    reference_cfg = copy.deepcopy(cfg)
    reference_cfg["engine"]["accelerator"] = "cpu"
    reference_cfg["data"]["num_workers"] = 0
    if cfg["model"]["name"].lower() == "patchcore":
        reference_cfg["model"]["pre_trained"] = False  # 完整状态由已有 checkpoint 恢复
    rows = []
    for image in args.images:
        path = resolve_path(image)
        result = service.predict_bytes(cfg["experiment"]["name"], path.read_bytes())
        reference = _build_engine(reference_cfg).predict(
            model=build_model(reference_cfg), data_path=str(path), ckpt_path=str(ckpt), return_predictions=True,
        )[0]
        reference_score = float(reference.pred_score.detach().cpu().reshape(-1)[0])
        reference_label = bool(reference.pred_label.detach().cpu().reshape(-1)[0])
        delta = result.score - reference_score
        if abs(delta) > 1e-6 or result.framework_label != reference_label:
            raise AssertionError(f"{path.name} 推理路径不一致，score delta={delta}")
        rows.append({**result.record(path.name), "reference_score": reference_score, "score_delta": delta})
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"device": "cpu", "passed": len(rows), "records": rows},
                                  indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[verify] CPU 推理一致性验收通过 {len(rows)} 张；记录见 {output}")


if __name__ == "__main__":
    main()
