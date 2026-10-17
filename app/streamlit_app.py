"""本地 CPU 检测台；常驻模型、会话内预测复用、批量判定和报告导出。"""

# ruff: noqa: E402
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.inference.service import InvalidImageError, Predictor, image_png, visualization

st.set_page_config(page_title="工业视觉缺陷检测", layout="wide")


@st.cache_resource(show_spinner=False)
def get_predictor(results_dir: str) -> Predictor:
    return Predictor(results_dir, device="cpu")


def main() -> None:
    st.title("工业视觉缺陷检测与异常定位")
    st.caption("选择对应产品类别的模型，上传图片，查看定位并导出检测记录。")
    service = get_predictor(os.environ.get("IAD_RESULTS_DIR", str(ROOT / "results")))
    catalog = service.catalog()
    if not catalog:
        st.info("尚未发现已训练模型。请将已有模型的实验配置和权重放入 results，再刷新页面。")
        return
    by_id = {item["model_id"]: item for item in catalog}
    with st.sidebar:
        st.header("检测设置")
        ids = list(by_id)
        preferred = next((i for i, name in enumerate(ids) if name == "stage3_efficientad_bottle"), 0)
        choice = st.selectbox("模型实验", ids, index=preferred)
        info = by_id[choice]
        st.write(f"产品类别：{info['category']} · 模型：{info['model']}")
        st.caption("模型仅针对所选类别验证；上传其他类别可能得到不可靠结果。")
        manual = st.checkbox("手动调整判定阈值", value=False)
        threshold = st.number_input("阈值", min_value=0.0, max_value=1.0, value=0.5,
                                    step=0.0001, format="%.4f", disabled=not manual)
        st.caption("阈值沿用已有模型标定，当前用于演示；独立测试验证待补充。")
        if st.button("清除本次检测记录"):
            st.session_state.pop("detections", None)
    uploads = st.file_uploader("上传同一产品类别的图片（最多 10 张，每张不超过 10 MiB）",
                               type=["png", "jpg", "jpeg", "bmp", "webp"], accept_multiple_files=True)
    if len(uploads) > 10:
        st.warning("请将单次检测控制在 10 张以内。")
        return
    if st.button("开始检测", type="primary", disabled=not uploads):
        detections, failures = [], []
        # 只保留本次会话的当前批次；改阈值不会再调用模型。
        for uploaded in uploads:
            try:
                with st.spinner(f"正在检测 {uploaded.name}..."):
                    prediction = service.predict_bytes(choice, uploaded.getvalue())
                detections.append((uploaded.name, prediction))
            except InvalidImageError as exc:
                failures.append(f"{uploaded.name}：{exc}")
            except Exception:
                failures.append(f"{uploaded.name}：推理失败，请检查模型与运行日志。")
        st.session_state["detections"] = {"model_id": choice, "items": detections, "failures": failures,
                                          "inputs": [hashlib.sha256(u.getvalue()).hexdigest() for u in uploads]}
    batch = st.session_state.get("detections")
    current_inputs = [hashlib.sha256(u.getvalue()).hexdigest() for u in uploads]
    if not batch or batch["model_id"] != choice or batch["inputs"] != current_inputs:
        st.info("选择图片后点击“开始检测”。切换图片或模型后需重新检测。")
        return
    for failure in batch["failures"]:
        st.warning(failure)
    items = batch["items"]
    if not items:
        return
    records = [prediction.record(name, threshold if manual else None) for name, prediction in items]
    a, b, c = st.columns(3)
    a.metric("已检测", len(records))
    b.metric("检出异常", sum(r["is_anomaly"] for r in records))
    c.metric("判为正常", sum(not r["is_anomaly"] for r in records))
    st.dataframe([{k: r[k] for k in ("filename", "label", "anomaly_score", "threshold", "threshold_source")}
                  for r in records], hide_index=True, use_container_width=True)
    st.caption("Anomaly Score 是归一化异常分数，不是缺陷概率。调整阈值只更新判定，无需重新推理。")
    selected = st.selectbox("查看定位详情", range(len(items)), format_func=lambda i: f"{i + 1}. {items[i][0]}")
    name, prediction = items[selected]
    heat, overlay = visualization(prediction.rgb, prediction.anomaly_map)
    for column, title, image in zip(st.columns(3), ("原图", "相对异常热力图", "叠加定位"),
                                     (prediction.rgb, heat, overlay)):
        column.subheader(title)
        column.image(image, use_container_width=True)
    st.caption("热力图按当前图片内相对强度着色；正常图也可能出现红色区域，请结合分数与判定。")
    detail = records[selected]
    st.write(f"判定：**{detail['label']}** · 分数 {detail['anomaly_score']:.6f} · 阈值 {detail['threshold']:.4f}")
    st.caption(f"模型版本 {prediction.model_version} · 检测记录 {prediction.prediction_id}")
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
    left, middle, right = st.columns(3)
    left.download_button("下载批次 CSV", csv_buffer.getvalue().encode("utf-8-sig"),
                         "detections.csv", "text/csv")
    middle.download_button("下载检测记录 JSON", json.dumps(records, indent=2, ensure_ascii=False),
                           "detections.json", "application/json")
    right.download_button("下载当前定位图", image_png(overlay), "overlay.png", "image/png")


if __name__ == "__main__":
    main()
