#!/usr/bin/env bash
# ============================================================
# 下载并准备 imagenette（EfficientAD 训练中 penalty 项需要），
# 并软链到项目 data/imagenette（指向 imagenette2/train，使其可被 ImageFolder 读取）。
#
# 用法：
#   bash scripts/setup_imagenette.sh [数据目录] [项目目录]
# 默认：
#   数据目录 /root/autodl-tmp/data，项目目录 /root/industrial-anomaly-detection
# ============================================================
set -e

DATA_DIR=${1:-/root/autodl-tmp/data}
PROJECT_DIR=${2:-/root/industrial-anomaly-detection}

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

if [ ! -d imagenette2 ]; then
  echo "下载 imagenette2.tgz（约 1.5 GB，AutoDL 可先 source /etc/network_turbo 加速）..."
  wget --progress=bar:force:noscroll https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz
  tar -xzf imagenette2.tgz
fi

# ImageFolder 要求目录下直接是类别子目录（n01440764 等），因此指向 train
mkdir -p "$PROJECT_DIR/data"
ln -sfn "$DATA_DIR/imagenette2/train" "$PROJECT_DIR/data/imagenette"

echo "imagenette 已就绪，类别示例："
ls "$PROJECT_DIR/data/imagenette" | head
