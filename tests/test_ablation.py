"""消融配置快速测试：baseline 优先、单因子覆盖与去重。"""

from __future__ import annotations

from src.experiments.ablation import build_ablation_configs


def ablation_cfg() -> dict:
    return {
        "base": {
            "data_root": "data/x", "category": "bottle", "seed": 1,
            "eval_batch_size": 4, "num_workers": 0, "resume": True,
            "baseline": {
                "backbone": "wide_resnet50_2", "layers": ["layer2", "layer3"],
                "coreset_sampling_ratio": 0.1, "num_neighbors": 9,
            },
        },
        "heatmap_count": 2,
        "sweep": {
            "backbone": ["resnet18", "wide_resnet50_2"],
            "coreset_sampling_ratio": [0.01, 0.1],
        },
    }


def test_baseline_first_and_dedup() -> None:
    names = [c["experiment"]["name"] for c in build_ablation_configs(ablation_cfg())]
    assert names[0] == "ablation_baseline"
    assert "ablation_backbone_resnet18" in names
    assert "ablation_backbone_wide_resnet50_2" not in names
    assert "ablation_coreset_sampling_ratio_0.01" in names
    assert "ablation_coreset_sampling_ratio_0.1" not in names


def test_single_factor_override_keeps_others() -> None:
    cfgs = build_ablation_configs(ablation_cfg())
    resnet = next(c for c in cfgs if c["experiment"]["name"] == "ablation_backbone_resnet18")
    assert resnet["model"]["backbone"] == "resnet18"
    assert resnet["model"]["coreset_sampling_ratio"] == 0.1
    assert resnet["model"]["num_neighbors"] == 9
