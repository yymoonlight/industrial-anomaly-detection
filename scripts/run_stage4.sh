#!/usr/bin/env bash
# 阶段四：PatchCore 消融 + 阈值标定 + 汇总报告
# 用法：bash scripts/run_stage4.sh
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
mkdir -p results/logs

python -m src.experiments.ablation --config configs/stage4_ablation.yaml 2>&1 | tee results/logs/stage4_ablation.log
python -m src.experiments.threshold --all 2>&1 | tee results/logs/stage4_threshold.log
python -m src.experiments.report 2>&1 | tee results/logs/stage4_report.log
echo "阶段四完成，见 results/ablation_*/、results/thresholds/、results/tables/、results/figures/"
