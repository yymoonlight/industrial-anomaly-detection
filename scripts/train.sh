#!/usr/bin/env bash
# 在 AutoDL 上运行训练 / 特征库构建。
# 用法：bash scripts/train.sh [配置文件]
set -e

# 国内网络下通过 HF 镜像下载预训练权重（已设置 HF_ENDPOINT 或使用学术加速时不覆盖）
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

CONFIG=${1:-configs/patchcore_bottle.yaml}
python -m src.train.train --config "$CONFIG"
