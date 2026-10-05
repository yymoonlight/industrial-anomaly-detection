#!/usr/bin/env bash
# 阶段五：启动 Streamlit Demo（监听 8501，AutoDL 需在「自定义服务」中开放该端口）
# 用法：bash scripts/run_demo.sh
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
