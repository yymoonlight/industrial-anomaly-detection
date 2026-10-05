#!/usr/bin/env bash
# 阶段一：bottle 单类别 PatchCore baseline（训练 + 评估，含日志）
# 用法：bash scripts/run_stage1.sh
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
mkdir -p results/logs

python -m src.train.train --config configs/patchcore_bottle.yaml 2>&1 | tee results/logs/stage1_train.log
python -m src.evaluation.evaluate --config configs/patchcore_bottle.yaml 2>&1 | tee results/logs/stage1_eval.log
echo "阶段一完成，见 results/patchcore_bottle/ 与 results/experiments.csv"
