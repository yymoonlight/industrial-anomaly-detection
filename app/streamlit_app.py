"""阶段五：工业视觉缺陷检测 Streamlit Demo。

功能：
- 自动发现 results/ 下已训练好的实验（含 config.yaml 与 .ckpt）并可切换；
- 上传图片，显示原图、异常热力图、叠加图与 anomaly score；
- 给出 Normal / Defect 判定，支持阈值滑块（默认读取阶段四标定的 best_f1 阈值）。

运行（在项目根目录）：
    streamlit run app/streamlit_app.py
服务器访问：先 bash scripts/run_demo.sh，再用 AutoDL 自定义服务端口（8501）转发。
"""

# 本脚本需先把项目根加入 sys.path 再 import 业务模块，故豁免 E402
# ruff: noqa: E402

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import streamlit as st
from anomalib.engine import Engine
from PIL import Image

from src.models.factory import build_model
from src.utils.config import load_config

st.set_page_config(page_title="工业视觉缺陷检测", layout="wide")


@st.cache_data(show_spinner=False)
def discover_models(results_dir: str) -> list[tuple[str, str, str]]:
    """返回 (实验名, config 路径, ckpt 路径) 列表。"""
    items = []
    for config_path in Path(results_dir).rglob("config.yaml"):
        exp_dir = config_path.parent
        ckpts = list(exp_dir.rglob("*.ckpt"))
        if ckpts:
            ckpt = sorted(ckpts, key=lambda p: p.stat().st_mtime)[-1]
            items.append((exp_dir.name, str(config_path), str(ckpt)))
    return sorted(items)


def colorize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """把单通道异常分数归一化并映射为 JET 伪彩图。"""
    normalized = (heatmap - heatmap.min()) / (np.ptp(heatmap) + 1e-8)
    colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def default_threshold(exp_dir: Path) -> float | None:
    """优先读取阶段四标定的 best_f1 阈值。"""
    for path in (exp_dir / "threshold" / "threshold.json",
                 ROOT / "results" / "thresholds" / exp_dir.name / "threshold.json"):
        if path.is_file():
            with path.open(encoding="utf-8") as f:
                return float(json.load(f)["best_f1"]["threshold"])
    return None


st.title("工业视觉缺陷检测与异常定位 Demo")
st.caption("MVTec AD · PatchCore / EfficientAD · 仅用正常样本训练")

with st.sidebar:
    st.header("设置")
    available = discover_models(str(ROOT / "results"))
    if not available:
        st.warning("results/ 下未发现已训练模型，请先在服务器完成训练与评估。")
        st.stop()
    choice = st.selectbox("选择模型实验", [name for name, _, _ in available], index=0)
    name, config_path, ckpt_path = next(item for item in available if item[0] == choice)
    exp_dir = Path(config_path).parent

uploaded = st.file_uploader("上传待检测图片", type=["png", "jpg", "jpeg", "bmp"])
if uploaded is None:
    st.info("请上传一张图片开始检测。")
    st.stop()

# 存到英文临时路径，避免中文路径问题
with tempfile.TemporaryDirectory() as tmp:
    image_path = Path(tmp) / "input.png"
    image_path.write_bytes(uploaded.getvalue())

    cfg = load_config(config_path)
    model = build_model(cfg)
    runtime = Path(tempfile.gettempdir()) / "iad_runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    engine = Engine(default_root_dir=str(runtime), logger=False, accelerator="auto")

    with st.spinner("推理中..."):
        predictions = engine.predict(
            model=model,
            data_path=str(image_path),
            ckpt_path=ckpt_path,
            return_predictions=True,
        )
        result = predictions[0]

    original = np.array(Image.open(uploaded).convert("RGB"))
    heatmap = result.anomaly_map.detach().cpu().numpy().squeeze()
    score = float(result.pred_score.detach().cpu()[0])
    framework_label = int(result.pred_label.detach().cpu()[0])

    heatmap_color = colorize_heatmap(heatmap)
    heatmap_resized = cv2.resize(heatmap_color, (original.shape[1], original.shape[0]))
    overlay = cv2.addWeighted(original, 0.55, heatmap_resized, 0.45, 0)

    initial = default_threshold(exp_dir)
    threshold = st.slider(
        "判定阈值（可调，默认取阶段四标定值）",
        min_value=0.0,
        max_value=float(max(score * 1.5, 1.0)),
        value=float(initial if initial is not None else max(score * 0.5, 0.5)),
        step=0.01,
    )
    is_defect = score >= threshold

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("原图")
        st.image(original, use_container_width=True)
    with col2:
        st.subheader("异常热力图")
        st.image(heatmap_resized, use_container_width=True)
    with col3:
        st.subheader("叠加定位")
        st.image(overlay, use_container_width=True)

    st.divider()
    metric_col, verdict_col = st.columns(2)
    metric_col.metric("Anomaly Score", f"{score:.4f}")
    if is_defect:
        verdict_col.error("判定结果：Defect（检出缺陷）")
    else:
        verdict_col.success("判定结果：Normal（正常）")
    st.caption(f"框架自适应阈值判定：{'Defect' if framework_label else 'Normal'}；"
               f"当前使用阈值 {threshold:.4f}")
