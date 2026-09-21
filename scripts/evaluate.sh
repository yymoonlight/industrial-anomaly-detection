#!/usr/bin/env bash
# 在 AutoDL 上运行评估（指标 + 热力图 + 速度 + experiments.csv）。
# 用法：bash scripts/evaluate.sh [配置文件]
set -e

# 国内网络下通过 HF 镜像下载预训练权重（已设置 HF_ENDPOINT 或使用学术加速时不覆盖）
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

CONFIG=${1:-configs/patchcore_bottle.yaml}
python -m src.evaluation.evaluate --config "$CONFIG"
