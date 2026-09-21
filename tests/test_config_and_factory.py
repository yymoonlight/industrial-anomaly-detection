"""快速检查：配置可加载、数据模块可构造、指标可实例化（不下载权重、不访问数据）。"""

from pathlib import Path

from src.data.datamodule import build_datamodule
from src.evaluation.metrics import METRIC_BUILDERS, build_evaluator
from src.utils.config import PROJECT_ROOT, load_config

CONFIG_PATH = Path("configs/patchcore_bottle.yaml")


def test_real_config_loads():
    cfg = load_config(CONFIG_PATH)
    assert cfg["model"]["name"] == "patchcore"
    assert cfg["data"]["category"] == "bottle"
    assert "image_auroc" in cfg["evaluation"]["metrics"]
    assert "aupro" in cfg["evaluation"]["metrics"]


def test_mvtec_datamodule_constructs():
    cfg = load_config(CONFIG_PATH)
    dm = build_datamodule(cfg)
    assert dm.category == "bottle"


def test_all_metrics_build():
    cfg = {"evaluation": {"metrics": list(METRIC_BUILDERS.keys())}}
    evaluator = build_evaluator(cfg)
    assert evaluator is not None


def test_folder_datamodule_constructs(synthetic_dataset):
    cfg = {
        "data": {
            "dataset": "folder",
            "name": "synthetic",
            "root": str(synthetic_dataset),
            "normal_dir": "good",
            "abnormal_dir": "abnormal",
            "normal_test_dir": "good_test",
            "mask_dir": "mask",
            "train_batch_size": 2,
            "eval_batch_size": 2,
            "num_workers": 0,
        }
    }
    dm = build_datamodule(cfg)
    dm.setup()
    assert len(dm.train_data) > 0


def test_project_root_exists():
    assert (PROJECT_ROOT / "configs").is_dir()
