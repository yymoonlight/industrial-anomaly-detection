"""脱离 Anomalib 的简化 PatchCore，实现用于与框架主线对照。

仅依赖 timm（特征提取）、scikit-learn（最近邻）、numpy / opencv：
1. 预训练 WideResNet50，取 layer2 / layer3 特征，layer2 局部平均池化对齐 layer3 后拼接；
2. 收集全部正常图的 patch 特征构成 memory bank；
3. 贪心 coreset 子采样（numpy 实现，限制候选规模以适配 CPU）；
4. 最近邻距离作为 patch 异常分，图像分取最大值，双线性上采样得到热力图；
5. 在测试集上计算 Image / Pixel AUROC 并保存三联图。

用法：
    python baselines/minimal_patchcore.py --layout folder --root <folder 数据目录>
    python baselines/minimal_patchcore.py --layout mvtec --root <MVTec 单类目录，如 .../bottle>
"""

# 需先设置环境变量与 sys.path 再 import 业务模块，故豁免 E402
# ruff: noqa: E402

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from timm import create_model
from torchvision import transforms

IMAGE_SIZE = 256
TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


def build_feature_extractor(device: str):
    model = create_model("wide_resnet50_2", pretrained=True).to(device).eval()
    features: dict[str, torch.Tensor] = {}

    def make_hook(name: str):
        def hook(_module, _inputs, output):
            features[name] = output
        return hook

    model.layer2.register_forward_hook(make_hook("layer2"))
    model.layer3.register_forward_hook(make_hook("layer3"))
    return model, features


@torch.no_grad()
def extract_patches(image: Image.Image, model, features, device) -> tuple[np.ndarray, int]:
    """返回单张图的 patch 特征 (P, d) 与 layer3 网格边长。"""
    tensor = TRANSFORM(image).unsqueeze(0).to(device)
    model(tensor)
    l2 = F.avg_pool2d(features["layer2"], kernel_size=2, stride=2)  # 对齐 layer3 网格
    l3 = features["layer3"]
    combined = torch.cat([l2, l3], dim=1)  # (1, d, h3, w3)
    grid = combined.shape[-1]
    patches = combined.flatten(2).squeeze(0).transpose(0, 1)  # (h3*w3, d)
    return patches.cpu().numpy(), grid


def greedy_coreset(pool: np.ndarray, ratio: float, max_candidates: int = 4096,
                   seed: int = 42) -> np.ndarray:
    """贪心 coreset：迭代选取距已选集合最小距离最大的点。"""
    rng = np.random.default_rng(seed)
    if len(pool) > max_candidates:
        pool = pool[rng.choice(len(pool), max_candidates, replace=False)]
    target = max(1, int(len(pool) * ratio))

    center_distance = np.sum((pool - pool.mean(0, keepdims=True)) ** 2, axis=1)
    first = int(np.argmax(center_distance))
    selected = [first]
    min_distance = np.sum((pool - pool[first]) ** 2, axis=1)
    for _ in range(target - 1):
        nxt = int(np.argmax(min_distance))
        if nxt in selected:
            break
        selected.append(nxt)
        min_distance = np.minimum(min_distance, np.sum((pool - pool[nxt]) ** 2, axis=1))
    return pool[selected]


def gather_samples(root: Path, layout: str):
    """返回训练正常图列表，以及测试 (路径, 标签, mask路径或None) 列表。"""
    if layout == "folder":
        train = sorted((root / "good").glob("*.png"))
        test = [(p, 0, None) for p in sorted((root / "good_test").glob("*.png"))]
        for p in sorted((root / "abnormal").glob("*.png")):
            mask = root / "mask" / p.name
            test.append((p, 1, mask if mask.is_file() else None))
        return train, test

    # MVTec 单类布局
    train = sorted((root / "train" / "good").glob("*.png"))
    test = [(p, 0, None) for p in sorted((root / "test" / "good").glob("*.png"))]
    for defect_dir in sorted((root / "test").iterdir()):
        if defect_dir.name == "good" or not defect_dir.is_dir():
            continue
        for p in sorted(defect_dir.glob("*.png")):
            mask = root / "ground_truth" / defect_dir.name / f"{p.stem}_mask.png"
            test.append((p, 1, mask if mask.is_file() else None))
    return train, test


def save_triptych(path: Path, image: Image.Image, heatmap: np.ndarray, score: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    resized = np.array(image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE)))
    colored = cv2.applyColorMap(cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                                cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(resized, 0.55, colored, 0.45, 0)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, img, title in zip(axes, (resized, colored, overlay),
                              ("Original", "Heatmap", "Overlay")):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"score={score:.4f}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="最小 PatchCore 实现")
    parser.add_argument("--root", required=True)
    parser.add_argument("--layout", choices=["folder", "mvtec"], default="folder")
    parser.add_argument("--ratio", type=float, default=0.1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="results/minimal_patchcore")
    args = parser.parse_args()

    root = Path(args.root)
    train_paths, test_samples = gather_samples(root, args.layout)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, features = build_feature_extractor(args.device)

    print(f"构建 memory bank：{len(train_paths)} 张正常图")
    bank, grid = [], None
    for path in train_paths:
        patches, grid = extract_patches(Image.open(path).convert("RGB"), model, features, args.device)
        bank.append(patches)
    bank = np.concatenate(bank, axis=0)
    print(f"memory bank: {bank.shape}，coreset ratio={args.ratio}")
    coreset = greedy_coreset(bank, args.ratio)

    neighbor = NearestNeighbors(n_neighbors=1).fit(coreset)

    labels, image_scores = [], []
    pixel_scores, pixel_masks = [], []
    for i, (path, label, mask_path) in enumerate(test_samples):
        image = Image.open(path).convert("RGB")
        patches, _ = extract_patches(image, model, features, args.device)
        distances, _ = neighbor.kneighbors(patches)
        patch_map = distances.reshape(grid, grid)
        heatmap = cv2.resize(patch_map, (IMAGE_SIZE, IMAGE_SIZE))
        heatmap = cv2.GaussianBlur(heatmap, (5, 5), 0)
        score = float(patch_map.max())

        labels.append(label)
        image_scores.append(score)
        if mask_path is not None:
            mask = np.array(Image.open(mask_path).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))) > 0
            pixel_masks.append(mask.flatten())
            pixel_scores.append(heatmap.flatten())
        if i < 8:
            save_triptych(out_dir / f"{i}_{path.stem}.png", image, heatmap, score)

    image_auroc = roc_auc_score(labels, image_scores)
    print(f"Image AUROC: {image_auroc:.4f}")
    if pixel_masks:
        pixel_auroc = roc_auc_score(np.concatenate(pixel_masks), np.concatenate(pixel_scores))
        print(f"Pixel AUROC: {pixel_auroc:.4f}")
    print(f"三联图已保存到 {out_dir}")


if __name__ == "__main__":
    main()
