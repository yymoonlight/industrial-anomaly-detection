#!/usr/bin/env bash
# 阶段三：PatchCore / EfficientAD × 多类别对比矩阵（自动断点续跑、单组失败不影响后续）
# 前置：已准备数据软链（setup_data_symlink.sh）与 imagenette（setup_imagenette.sh）
# 用法：bash scripts/run_stage3.sh
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
mkdir -p results/logs

python -m src.experiments.matrix --config configs/stage3_matrix.yaml 2>&1 | tee results/logs/stage3.log
echo "阶段三完成，见 results/stage3_*/ 与 results/experiments.csv"
