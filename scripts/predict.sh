#!/usr/bin/env bash
# 推理：默认对测试集预测；也可传入单张图片或目录。
# 用法：
#   bash scripts/predict.sh                                   # 测试集
#   bash scripts/predict.sh configs/xxx.yaml path/to/img.png  # 单图
set -e

# 国内网络下通过 HF 镜像下载预训练权重（已设置 HF_ENDPOINT 或使用学术加速时不覆盖）
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

CONFIG=${1:-configs/patchcore_bottle.yaml}
INPUT=${2:-test}
python -m src.inference.predict --config "$CONFIG" --input "$INPUT"
