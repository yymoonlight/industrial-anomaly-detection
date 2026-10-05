# Industrial Anomaly Detection — 工业视觉缺陷检测与异常定位系统

基于 **MVTec AD** 数据集，使用 **PatchCore** 与 **EfficientAD** 实现工业图像的无监督缺陷检测与像素级异常定位，覆盖「数据处理 → 模型训练/特征库构建 → 多类别实验 → 评估优化 → 阈值工程 → 异常热力图 → 推理脚本 → Streamlit Demo」的完整 AI 工程流程；并附带一份**不依赖框架的手写最小 PatchCore**，用于验证对算法原理的理解。

> 状态：阶段一~六代码全部完成，已在 CPU 合成数据上通过 35 项快速测试、EfficientAD 与最小 PatchCore 端到端冒烟、Streamlit headless 启动；GPU 正式实验待在 AutoDL 运行
> 框架：[Anomalib v2.6.2](https://github.com/open-edge-platform/anomalib)（PyTorch / Lightning）

## 目录结构

```
industrial-anomaly-detection/
├── configs/        # YAML 配置（数据、模型、引擎、指标、批量矩阵、消融），不硬编码超参
├── data/           # MVTec AD 数据集与 imagenette（不入库）
├── src/
│   ├── data/       # 数据模块工厂（MVTecAD / Folder）
│   ├── models/     # PatchCore / EfficientAD 模型工厂
│   ├── train/      # 训练 / 特征库构建统一入口
│   ├── evaluation/ # 指标（Image/Pixel AUROC、AUPRO、F1）与汇总
│   ├── inference/  # 批量/单图推理、anomaly score、三联热力图
│   └── experiments/# 阶段三矩阵对比、阶段四消融/阈值标定/报告生成
├── app/            # Streamlit Demo
├── baselines/      # 手写最小 PatchCore（仅 timm + sklearn，讲清原理）
├── scripts/        # 服务器运行脚本（各阶段一键脚本、数据/imagenette 准备）
├── results/        # 实验指标、热力图、阈值、报告、日志（不入库）
├── tests/          # 快速测试 + CPU 端到端冒烟测试
├── requirements.txt
├── pyproject.toml  # ruff / pytest 配置
├── README.md
└── .gitignore
```

## 数据集

- MVTec AD：15 个类别（10 物体 + 5 纹理），5,354 张高分辨率图像（3,629 张训练正常图 + 1,725 张测试图），缺陷测试图带像素级 mask；
- EfficientAD 训练还需 **imagenette**（ImageNet 子集，约 1.5 GB），用于惩罚学生网络在自然图像上的输出；
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
pip install "anomalib[cu126]==2.6.2"   # 自动安装匹配的 torch 2.6+/torchvision
pip install -r requirements.txt
```

> 国内网络说明：预训练 backbone 权重从 HuggingFace 拉取。`src/__init__.py` 与 `scripts/*.sh`
> 已默认把 `HF_ENDPOINT` 指向 `https://hf-mirror.com`；AutoDL 上建议先执行
> `source /etc/network_turbo` 开启学术加速（EfficientAD 的 teacher 权重与 imagenette 均在境外）。
> Windows 用户目录若含非 ASCII 字符（如中文用户名），权重缓存会自动重定向到项目内
> `.cache/huggingface`（纯英文路径），数据与输出路径也请避免中文/空格。

## 使用方式（统一命令入口）

所有超参集中在 `configs/*.yaml`，三个基础入口均可通过 `--config` 切换实验：

```bash
# 1. 训练 / 构建特征库（PatchCore 只需 1 个 epoch；EfficientAD 约 100 epoch）
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

### 批量实验（阶段三 / 四 / 五 / 六）

```bash
# 准备 imagenette（EfficientAD 依赖，只需一次）
bash scripts/setup_imagenette.sh

# 阶段三：7 类别（bottle/cable/metal_nut/hazelnut/grid/leather/carpet）× 2 模型批量对比
#   自动展开配置、串行 train+evaluate，支持断点续跑（读 experiments.csv 跳过已完成），
#   单组失败会记录到 results/logs/stage3_failures.csv 后继续
bash scripts/run_stage3.sh

# 阶段四：bottle 上的工程消融 + 阈值标定 + 报告
#   消融维度：backbone / feature layers / coreset ratio / num_neighbors（控制变量、去重、断点续跑）
#   阈值：扫阈值画 PR 曲线，选最优 F1，并给召回≥0.99 / 精度≥0.95 约束下的工作点
bash scripts/run_stage4.sh
# 报告：精度-速度散点、类别柱状图、coreset/backbone 趋势图、透视表
python -m src.experiments.report

# 阶段五：Streamlit Demo（上传图 → 原图/热力图/叠加/score/Normal·Defect，可选模型与阈值）
bash scripts/run_demo.sh        # 服务器端，再用 SSH 端口转发到本地浏览器

# 阶段六：手写最小 PatchCore（timm 取 layer2/3 特征 → greedy coreset → 1-NN → 上采样）
python baselines/minimal_patchcore.py --layout mvtec \
    --root data/mvtec_anomaly_detection/bottle --out results/minimal_bottle
```

### 实验产物

每次实验保存在 `results/<experiment_name>/` 下：

- `metrics.json`：五项指标（Image/Pixel AUROC、AUPRO、Image/Pixel F1）；
- `config.yaml`：本次实验配置快照（可复现）；
- `predictions/scores.csv`：每张图的路径、真值、预测标签、anomaly score；
- `predictions/heatmaps/`：「原图 | JET 热力图 | 叠加图」三联图（异常样本优先）；
- 权重 checkpoint：`<root>/<Model>/<Dataset>/<category>/latest/weights/lightning/model.ckpt`。

跨实验汇总与报告：

- `results/experiments.csv`：所有实验的精度、ms/张、FPS、关键超参；
- `results/thresholds/<experiment>/threshold.json`：最优 F1 与召回/精度约束工作点；
- `results/tables/`、`results/figures/`：透视表与对比图表；
- `results/logs/`：各阶段运行日志与失败清单。

### 测试与静态检查（本地 CPU）

```powershell
# 快速测试（目录结构、配置、工厂、矩阵、消融、阈值、报告）
python -m pytest tests/ -v -m "not slow"
# 含端到端冒烟（合成数据跑通 train→evaluate→推理，首次会下载 backbone 权重）
python -m pytest tests/ -v
# 静态检查
python -m ruff check .
```

## 路线图

- [x] 阶段一：bottle 单类别 PatchCore Baseline（指标 + 热力图 + 日志 + CPU 冒烟）
- [x] 阶段二：统一入口与配置化封装（train/evaluate/predict + configs）
- [x] 阶段三：多类别 × PatchCore/EfficientAD 批量对比代码（精度 / 速度 / FPS，断点续跑）
- [x] 阶段四：工程消融 + 阈值工程 + 报告图表代码
- [x] 阶段五：Streamlit Demo（上传图 → 原图/热力图/score/Normal·Defect）
- [x] 阶段六：手写最小 PatchCore（验证对算法原理的理解，CPU 冒烟通过）
- [ ] AutoDL GPU 上依次跑出阶段一 / 三 / 四正式结果
- [ ] 收尾：README 结果回填与 GitHub 整理

## 参考实现

- [open-edge-platform/anomalib](https://github.com/open-edge-platform/anomalib)：主力框架（Apache-2.0）
- [amazon-science/patchcore-inspection](https://github.com/amazon-science/patchcore-inspection)：PatchCore 官方实现，Roth et al., CVPR 2022（arXiv:2106.08265）
- [nelson1425/EfficientAD](https://github.com/nelson1425/EfficientAD)：EfficientAD 复现，Batzner et al., WACV 2024（arXiv:2303.14535）
- [MVTec AD Dataset](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)
