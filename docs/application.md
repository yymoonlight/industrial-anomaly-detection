# 本地 CPU 应用使用与设计

这一层复用已有 checkpoint，不启动训练。目标是让相同图片经 CLI、页面、API 得到一致的分数和业务判定，并能说明结果来自哪个模型、哪个阈值。

## 启动

在项目根目录、现有 `iad-cpu` 环境运行：

```powershell
python -m pip install -r requirements.txt
# 页面：直接调用统一 Predictor，默认 CPU
python -m streamlit run app/streamlit_app.py --server.address 127.0.0.1
# API：另开一个终端，同样默认 CPU
python -m uvicorn app.api:app --host 127.0.0.1 --port 8000 --log-config configs/logging.yaml
```

页面与 API 是两个独立进程，各自常驻一个模型。页面不依赖 API 在线，两者通过相同的 Python 服务层共享行为；当前没有 HTTP 转发链路。打开页面 `http://127.0.0.1:8501`，接口文档 `http://127.0.0.1:8000/docs`。

`results/<实验名>/config.yaml` 与同目录下的 `.ckpt` 必须同时存在。GitHub 不包含权重，新克隆无法直接完成真实推理；可将已有服务器实验目录复制回来，无需重新训练。必要时设置 `IAD_RESULTS_DIR` 为已有 results 的绝对路径。应用只加载该目录中管理员维护的模型，不接受通过请求传入权重路径。

## 页面操作

1. 选择产品类别对应的模型，上传最多 10 张图片。
2. 点击“开始检测”，查看批次异常数量、逐图判定与定位图。
3. 可手动调整阈值；此时只重新计算 `score > threshold`，不重新加载模型或推理。
4. 下载批次 CSV、JSON 或当前叠加 PNG。图片和检测结果只保留在当前会话，不自动写入服务器历史库。

切换图片或模型后需重新检测，避免把旧结果误认为新输入结果。单图限制 10 MiB、800 万像素；按实际内容验证 PNG/JPEG/BMP/WEBP，灰度和透明通道统一转换为 RGB。

## API

```powershell
curl.exe http://127.0.0.1:8000/health
curl.exe http://127.0.0.1:8000/models
curl.exe -X POST http://127.0.0.1:8000/predict `
  -F "model_id=stage3_efficientad_bottle" `
  -F "file=@samples/bottle_broken_l/002.png" `
  -F "include_images=true"
```

`samples/` 为本地自备演示图，不随 Git 提交；替换为实际图片路径。可选 `threshold=0.51` 指定人工阈值。`include_images` 默认 false，打开时额外返回 PNG 的 base64 字符串；否则返回结构化判定和追溯字段。

主要响应字段：`prediction_id`、UTC `created_at`、`model_id`、权重 SHA256 前 16 位 `model_version`、`image_sha256`、`anomaly_score`、`threshold`、`threshold_source`、`comparison`、`is_anomaly`、`label`、`inference_ms`。错误输入返回 422、体积/像素超限返回 413、模型不存在返回 404、模型加载/计算失败返回 503。

健康检查表示进程可响应，并返回可发现模型数量；不表示每个 checkpoint 都已加载验证。首次请求加载权重，此后相同配置和权重复用；每进程最多保留一个模型，切换模型需要重新加载，避免大量模型占满内存。锁保护模型切换和推理，适用于本地单机展示，尚未进行高并发容量测试。

## CLI

```powershell
python -m src.inference.predict --config configs/efficientad_bottle.yaml `
  --ckpt results/stage3_efficientad_bottle/EfficientAd/MVTecAD/bottle/v0/weights/lightning/model.ckpt `
  --input samples/bottle_broken_l --device cpu
```

目录内多张图复用一个模型；输出三联图及逐图 JSON。`--threshold` 为可选人工阈值。`--input test` 保留原实验评估路径和配置，其批量计时不能与应用的 `inference_ms` 直接比较。

## 阈值与结果含义

- 三个应用入口统一使用严格大于 `>`，与框架默认业务比较符号一致。分数不是缺陷概率。
- 默认阈值读取 checkpoint 中的归一化阈值；不自动读取旧 `best_f1` 文件。既有 checkpoint 的标定协议仍包含测试集复用，因此响应标记 `legacy_exploratory`。应用改造不会使旧成绩自动变成独立测试成绩。
- 旧阈值扫描明确标记为 `test_exploratory`，继续使用其原有 `>=` 计算口径用于探索；不会被应用悄悄用作工作阈值。独立 calibration/test 的正式标定留待后续。
- 热力图按单图内相对强度着色，正常图也可能有红区；它不表示跨图片可比较的绝对缺陷概率。
- `inference_ms` 是模型加载后的一次预处理、前向与后处理耗时（含输出回传 CPU），不含上传、图片解码、权重加载、哈希和绘图；没有 warmup/重复统计，不是性能基准或延迟保证。

## 推理服务设计

`src/inference/service.py` 负责文件校验、模型发现、模型生命周期、原评估预处理、业务判定及追溯记录。页面负责交互与导出，FastAPI 负责 HTTP 输入输出，CLI 负责本地文件批处理。

训练时保存的评价器可能带 CUDA 设备状态。应用按 checkpoint 中的结构参数构建模型，剔除仅供训练评价/可视化的状态，严格加载其余权重，避免 CPU 部署误用 GPU 评价器。预处理沿用原评估中的抗锯齿 resize，而不是默认导出路径中关闭抗锯齿的变换；真实样本一致性验收会核对这一点。

这些是模型应用工程上的实现价值；PatchCore/EfficientAD 主体来自 Anomalib，不声称原创算法。也不将页面、API 或导出功能等同于工业现场验证。

## CPU 验收

```powershell
python -m pytest tests/ -q -m "not slow"
python -m ruff check .
```

新测试覆盖并发时单次加载、权重/配置变化后的缓存失效、损坏/超限输入、阈值边界、接口错误与 PNG 返回、页面调整阈值不重复推理、阈值产物保存后正常返回。接口合同测试使用轻量适配器；真实 checkpoint 的 CPU 验证记录与合同测试分开陈述。

已有权重可按下列方式对比新服务与原 Engine 路径（只做 CPU 推理，不训练）：

```powershell
python -m scripts.verify_inference_cpu --config results/stage3_efficientad_bottle/config.yaml `
  --images samples/bottle_broken_l/002.png samples/bottle_good/000.png
```

默认输出 `results/application_cpu_smoke.json`；用 `--output` 指定其他位置。比较的是同环境下的两条 CPU 路径，不要求 CPU 和历史 GPU 分数逐位相同。
