"""EfficientAD CPU 端到端冒烟（slow）：合成数据 + 迷你 imagenette，1 epoch。"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.evaluation.evaluate import run_evaluation
from src.train.train import run_training


@pytest.mark.slow
def test_efficientad_cpu_one_epoch(synthetic_dataset) -> None:
    case = synthetic_dataset
    rng = np.random.default_rng(1)
    mini = case / "mini_imagenette"
    for cls in ("a", "b"):
        for i in range(2):
            path = mini / f"class_{cls}" / f"{i}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rng.integers(0, 255, (128, 128, 3), dtype=np.int16).astype("uint8")).save(path)

    cfg = {
        "experiment": {"name": "ead_smoke", "seed": 42},
        "data": {"dataset": "folder", "name": "smoke", "root": str(case),
                 "normal_dir": "good", "abnormal_dir": "abnormal",
                 "normal_test_dir": "good_test", "mask_dir": "mask",
                 "train_batch_size": 1, "eval_batch_size": 2, "num_workers": 0},
        "model": {"name": "efficientad", "model_size": "s", "imagenet_dir": str(mini),
                  "teacher_out_channels": 384, "lr": 1e-4, "weight_decay": 1e-5},
        "engine": {"max_epochs": 1, "accelerator": "cpu", "devices": 1,
                   "default_root_dir": str(case / "results")},
        "evaluation": {"metrics": ["image_auroc", "pixel_auroc", "aupro", "image_f1", "pixel_f1"],
                       "save_heatmaps": True, "heatmap_count": 4, "predictions_csv": True},
    }

    ckpt = run_training(cfg)
    assert ckpt
    metrics = run_evaluation(cfg, ckpt)
    assert "image_AUROC" in metrics
    assert "pixel_AUROC" in metrics
