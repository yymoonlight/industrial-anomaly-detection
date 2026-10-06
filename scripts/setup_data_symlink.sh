#!/usr/bin/env bash
# 把数据盘上的 MVTec AD 解压目录软链到项目 data/ 下，使配置中的相对路径生效。
# 用法：bash scripts/setup_data_symlink.sh [数据真实路径]
set -e

# 整包 tar 解压后各类别直接位于该目录下（bottle、cable 等）
SRC=${1:-/root/autodl-tmp/data}

if [ ! -d "$SRC/bottle" ]; then
  echo "[警告] 未找到 $SRC/bottle，请确认数据集已解压（$SRC 下应直接包含 bottle、cable 等类别目录）"
fi

mkdir -p data
ln -sfn "$SRC" data/mvtec_anomaly_detection
echo "软链已建立："
ls -l data/
