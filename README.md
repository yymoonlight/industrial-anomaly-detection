# Industrial Anomaly Detection — 工业视觉缺陷检测与异常定位系统

基于 **MVTec AD** 数据集，使用 **PatchCore** 与 **EfficientAD** 实现工业图像的无监督缺陷检测与像素级异常定位，覆盖「数据处理 → 模型训练/特征库构建 → 多类别实验 → 评估优化 → 异常热力图 → 推理脚本 → Streamlit Demo」的完整 AI 工程流程。

> 状态：阶段一 Baseline 代码完成，已在 CPU 合成数据上端到端冒烟通过；真实 bottle 类别 GPU 实验待运行
> 框架：[Anomalib v2.6.2](https://github.com/open-edge-platform/anomalib)（PyTorch / Lightning）

## 目录结构

```
industrial-anomaly-detection/
├── configs/        # YAML 配置（数据、模型、引擎、指标），不硬编码超参
├── data/           # MVTec AD 数据集（不入库）
├── src/
│   ├── data/       # 数据模块工厂（MVTecAD / Folder）
│   ├── models/     # PatchCore / EfficientAD 模型工厂
│   ├── train/      # 训练 / 特征库构建统一入口
│   ├── evaluation/ # 指标（Image/Pixel AUROC、AUPRO、F1）与汇总
│   └── inference/  # 批量/单图推理、anomaly score、三联热力图
├── app/            # Streamlit Demo（阶段五）
├── scripts/        # 服务器运行脚本（train/evaluate/predict、数据软链）
├── results/        # 实验指标、热力图、日志（不入库）
├── tests/          # 快速测试 + CPU 端到端冒烟测试
├── requirements.txt
├── pyproject.toml  # ruff / pytest 配置
├── README.md
└── .gitignore
```

## 数据集

- MVTec AD：15 个类别（10 物体 + 5 纹理），5,354 张高分辨率图像（3,629 张训练正常图 + 1,725 张测试图），缺陷测试图带像素级 mask；
- 下载直链与存储占用见 `项目任务书.md` 附录 B；
- 解压后目录结构：`data/mvtec_anomaly_detection/<category>/{train/good, test/<缺陷类型>, ground_truth/<缺陷类型>}`；
- 许可证 **CC BY-NC-SA 4.0**，仅限非商业学术用途；数据集、权重、实验产物均不提交 Git。

## 环境搭建

要求 **Python 3.10**（本地 Windows CPU 与远程 GPU 服务器保持一致）。

**本地（Windows，无 GPU，仅开发、静态检查与轻量测试）：**

```powershell
conda create -n iad-cpu python=3.10 -y
conda activate iad-cpu
pip install "anomalib[cpu]==2.6.2"
pip install -r requirements.txt
```

**GPU 服务器（AutoDL，RTX 3090/4090，CUDA 12.6 路线）：**

```bash
conda create -n iad python=3.10 -y
conda activate iad
pip install "anomalib[cu126]==2.6.2"   # 会自动安装匹配的 torch 2.6+/torchvision
pip install -r requirements.txt
```

> 国内网络说明：预训练 backbone 权重从 HuggingFace 拉取。`src/__init__.py` 与 `scripts/*.sh`
> 已默认把 `HF_ENDPOINT` 指向 `https://hf-mirror.com`；AutoDL 上也可先执行
> `source /etc/network_turbo` 开启学术加速。
> Windows 用户目录若含非 ASCII 字符（如中文用户名），权重缓存会自动重定向到项目内
> `.cache/huggingface`（纯英文路径），数据与输出路径也请避免中文/空格。

## 使用方式（统一命令入口）

所有超参集中在 `configs/*.yaml`，三个入口均可通过 `--config` 切换实验：

```bash
# 1. 训练 / 构建特征库（PatchCore 只需 1 个 epoch）
python -m src.train.train --config configs/patchcore_bottle.yaml
#   或 bash scripts/train.sh configs/patchcore_bottle.yaml

# 2. 评估：Image/Pixel AUROC、AUPRO、F1 + 测试集热力图 + scores.csv，
#    并把本次实验追加到 results/experiments.csv
python -m src.evaluation.evaluate --config configs/patchcore_bottle.yaml
#   或 bash scripts/evaluate.sh configs/patchcore_bottle.yaml

# 3. 推理：整个测试集 / 单张图片 / 某个目录
python -m src.inference.predict --config configs/patchcore_bottle.yaml --input test
python -m src.inference.predict --config configs/patchcore_bottle.yaml \
    --input data/mvtec_anomaly_detection/bottle/test/broken_large/000.png
#   或 bash scripts/predict.sh configs/patchcore_bottle.yaml test
```

每次实验产物保存在 `results/<experiment_name>/` 下：

- `metrics.json`：五项指标；
- `config.yaml`：本次实验配置快照（可复现）；
- `predictions/scores.csv`：每张图的路径、真值、预测标签、anomaly score；
- `predictions/heatmaps/`：「原图 | JET 热力图 | 叠加图」三联图（异常样本优先）；
- 权重 checkpoint：`<root>/Patchcore/<Dataset>/<category>/latest/weights/lightning/model.ckpt`；
- 跨实验汇总：`results/experiments.csv`（含精度与 ms/张、FPS）。

### 测试与静态检查（本地 CPU）

```powershell
# 快速测试（目录结构、配置、数据模块、指标）
python -m pytest tests/ -v -m "not slow"
# 含端到端冒烟（合成数据跑通 train→evaluate→单图推理，首次会下载 backbone 权重）
python -m pytest tests/ -v
# 静态检查
python -m ruff check src tests
```

## 路线图

- [x] 阶段一：bottle 单类别 PatchCore Baseline 代码（指标 + 热力图 + 日志 + CPU 冒烟）
- [ ] 阶段一：AutoDL GPU 上跑出 bottle 正式结果
- [x] 阶段二：统一入口与配置化封装（train/evaluate/predict + configs）
- [ ] 阶段三：PatchCore vs EfficientAD 多类别对比（精度 / 速度 / FPS）
- [ ] 阶段四：工程优化实验（backbone、feature layer、coreset ratio、memory bank、阈值）
- [ ] 阶段五：Streamlit Demo（上传图 → 原图/热力图/score/Normal·Defect）
- [ ] 收尾：README 完善与 GitHub 整理

## 参考实现

- [open-edge-platform/anomalib](https://github.com/open-edge-platform/anomalib)：主力框架（Apache-2.0）
- [amazon-science/patchcore-inspection](https://github.com/amazon-science/patchcore-inspection)：PatchCore 官方实现，Roth et al., CVPR 2022（arXiv:2106.08265）
- [nelson1425/EfficientAD](https://github.com/nelson1425/EfficientAD)：EfficientAD 复现，Batzner et al., WACV 2024（arXiv:2303.14535）
- [MVTec AD Dataset](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)
