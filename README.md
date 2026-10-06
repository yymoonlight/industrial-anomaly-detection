# Industrial Anomaly Detection — 工业视觉缺陷检测与异常定位系统

基于 **MVTec AD** 数据集，使用 **PatchCore** 与 **EfficientAD** 实现工业图像的无监督缺陷检测与像素级异常定位，覆盖「数据处理 → 模型训练/特征库构建 → 多类别实验 → 评估优化 → 阈值工程 → 异常热力图 → 推理脚本 → Streamlit Demo」的完整 AI 工程流程；并附带一份**不依赖框架的手写最小 PatchCore**，用于验证对算法原理的理解。

> 状态：阶段一~六全部完成。GPU 正式实验（阶段一 bottle baseline、阶段三 7 类×2 模型、阶段四消融与阈值标定、阶段六手写 PatchCore）已在 AutoDL RTX 3080 Ti 上跑完；Streamlit Demo 已在本地 Windows CPU 跑通并验证缺陷/正常样本判定。
> 框架：[Anomalib v2.6.2](https://github.com/open-edge-platform/anomalib)（PyTorch / Lightning）

## 目录结构

```
industrial-anomaly-detection/
├── configs/        # YAML 配置（数据、模型、引擎、指标、批量矩阵、消融），不硬编码超参
├── data/           # MVTec AD 数据集与 imagenette（不入库）
├── src/
│   ├── data/       # 数据模块工厂（MVTecAD / Folder，含 Jupyter 残留自动清理）
│   ├── models/     # PatchCore / EfficientAD 模型工厂
│   ├── train/      # 训练 / 特征库构建统一入口
│   ├── evaluation/ # 指标（Image/Pixel AUROC、AUPRO、F1）与汇总
│   ├── inference/  # 批量/单图推理、anomaly score、三联热力图
│   └── experiments/# 阶段三矩阵对比、阶段四消融/阈值标定/报告生成
├── app/            # Streamlit Demo
├── baselines/      # 手写最小 PatchCore（仅 timm + sklearn，讲清原理）
├── scripts/        # 服务器运行脚本（各阶段一键脚本、数据/imagenette 准备）
├── docs/           # 展示用分析产物：消融报告、experiments.csv、对比 tables/figures、Demo 截图
├── results/        # 实验指标、热力图、阈值、权重、日志（不入库）
├── tests/          # 快速测试 + CPU 端到端冒烟测试
├── 项目任务书.md    # 完整项目规划、参考实现、数据集下载直链与存储占用
├── requirements.txt
├── pyproject.toml  # ruff / pytest 配置
├── README.md
└── .gitignore
```

## 数据集

- **MVTec AD**：15 个类别（10 物体 + 5 纹理），5,354 张高分辨率图像（3,629 张训练正常图 + 1,725 张测试图），缺陷测试图带像素级 mask；
- **整包下载直链**（约 4.90 GiB / 5.26 GB，`mvtec_anomaly_detection.tar.xz`）：
  <https://www.mydrive.ch/shares/150996/b52ecdcbf521176e9db9c731f2304b27/download/420938113-1629960298/mvtec_anomaly_detection.tar.xz>
- 单类别下载直链与各类别存储占用见 `项目任务书.md` 附录 B；
- **EfficientAD** 训练还需 **imagenette**（ImageNet 子集，约 1.5 GB），用于惩罚学生网络在自然图像上的输出；
- 解压后目录结构：`data/mvtec_anomaly_detection/<category>/{train/good, test/<缺陷类型>, ground_truth/<缺陷类型>}`；
- 许可证 **CC BY-NC-SA 4.0**，仅限非商业学术用途；数据集、权重、完整实验产物均不提交 Git。

## 环境搭建

要求 **Python 3.10**（本地 Windows CPU 与远程 GPU 服务器保持一致）。

**本地（Windows，无 GPU，仅开发、静态检查与轻量测试）：**

```powershell
conda create -n iad-cpu python=3.10 -y
conda activate iad-cpu
pip install "anomalib[cpu]==2.6.2"
pip install -r requirements.txt
```

**GPU 服务器（AutoDL，RTX 3080 Ti / 3090 / 4090，CUDA 12.6 路线）：**

```bash
conda create -n iad python=3.10 -y
conda activate iad
pip install "anomalib[cu126]==2.6.2"   # 自动安装匹配的 torch 2.6+/torchvision
pip install -r requirements.txt
```

> 国内网络说明：预训练 backbone 权重从 HuggingFace 拉取。`src/__init__.py` 与 `scripts/*.sh`
> 已默认把 `HF_ENDPOINT` 指向 `https://hf-mirror.com`；AutoDL 上建议先执行
> `source /etc/network_turbo` 开启学术加速（EfficientAD 的 teacher 权重与 imagenette 均在境外）。
> Windows 用户目录若含非 ASCII 字符（如中文用户名），权重缓存与临时目录会自动重定向到项目内
> 纯英文路径，规避 anomalib 的路径校验问题。

## 使用方式（统一命令入口）

所有超参集中在 `configs/*.yaml`，三个基础入口均可通过 `--config` 切换实验：

```bash
# 1. 训练 / 构建特征库（PatchCore 只需 1 个 epoch；EfficientAD 约 100 epoch）
python -m src.train.train --config configs/patchcore_bottle.yaml

# 2. 评估：Image/Pixel AUROC、AUPRO、F1 + 测试集热力图 + scores.csv，
#    并把本次实验追加到 results/experiments.csv
python -m src.evaluation.evaluate --config configs/patchcore_bottle.yaml

# 3. 推理：整个测试集 / 单张图片 / 某个目录
python -m src.inference.predict --config configs/patchcore_bottle.yaml --input test
python -m src.inference.predict --config configs/patchcore_bottle.yaml \
    --input data/mvtec_anomaly_detection/bottle/test/broken_large/000.png
```

### 批量实验（阶段三 / 四 / 五 / 六）

```bash
# 准备 imagenette（EfficientAD 依赖，只需一次）
bash scripts/setup_imagenette.sh

# 阶段三：7 类别（bottle/cable/metal_nut/hazelnut/grid/leather/carpet）× 2 模型批量对比
#   自动展开配置、串行 train+evaluate，支持断点续跑（读 experiments.csv 跳过已完成）
bash scripts/run_stage3.sh

# 阶段四：bottle 上的工程消融 + 阈值标定 + 报告
#   消融维度：backbone / feature layers / coreset ratio / num_neighbors（控制变量、去重、断点续跑）
#   阈值：扫阈值画 PR 曲线，选最优 F1，并给召回≥0.99 / 精度≥0.95 约束下的工作点
bash scripts/run_stage4.sh
# 报告：精度-速度散点、类别柱状图、coreset/backbone 趋势图、透视表
python -m src.experiments.report

# 阶段五：Streamlit Demo（上传图 → 原图/热力图/叠加/score/Normal·Defect，可选模型与阈值）
streamlit run app/streamlit_app.py

# 阶段六：手写最小 PatchCore（timm 取 layer2/3 特征 → greedy coreset → 1-NN → 上采样）
python baselines/minimal_patchcore.py --layout mvtec \
    --root data/mvtec_anomaly_detection/bottle --out results/minimal_bottle
```

## 实验结果速览

完整 25 组实验记录见 [`docs/experiments.csv`](docs/experiments.csv)，对比表格见 [`docs/tables/`](docs/tables/)，消融实验方法论与结论见 [`docs/消融实验说明报告.md`](docs/消融实验说明报告.md)。

**阶段三（7 类 × PatchCore / EfficientAD）关键对比（metal_nut 为例）：**

| 模型 | Image AUROC | Pixel AUROC | AUPRO | Image F1 | Pixel F1 | 单图耗时 |
|---|---|---|---|---|---|---|
| PatchCore (WR50) | 0.9985 | 0.9871 | 0.9423 | 0.9891 | 0.8416 | 141.6 ms |
| EfficientAD (s) | 0.9853 | 0.9748 | 0.9127 | 0.9674 | 0.8087 | 133.3 ms |

精度-速度权衡散点图：

![精度-速度散点](docs/figures/accuracy_vs_speed.png)

各类别 Image AUROC 对比：

![各类别 Image AUROC](docs/figures/image_auroc_by_category.png)

**阶段四消融（bottle，PatchCore）核心结论：**
- 骨干网络越强，像素级定位越准（WR50 > resnet50 > resnet18），但推理略慢；
- 简单物体上**仅用 layer2 定位效果最好**（AUPRO 0.9632），三层拼接因显存不足在 12GB 卡上不可行；
- coreset 采样比例 0.01~0.25 对精度几乎无影响，**0.01 即可把记忆库缩到 1/10、明显省显存提速**；
- 近邻数 1/3/9 结果一致，本场景取 1 即可。

backbone 与 coreset 趋势：

![backbone 对比](docs/figures/backbone_comparison.png) ![coreset 趋势](docs/figures/coreset_trend.png)

## Streamlit Demo

上传图片后显示原图、异常热力图、叠加定位图、Anomaly Score 与 Normal/Defect 判定；阈值滑块默认取阶段四标定的 best F1 阈值，可现场拖动演示精确率-召回率权衡。

| 缺陷样本（score 0.503 → Defect） | 正常样本（score 0.499 → Normal） |
|---|---|
| ![缺陷样本](docs/demo_screenshots/bottle_defect.png) | ![正常样本](docs/demo_screenshots/bottle_normal.png) |

本地运行（Windows CPU，推荐 EfficientAD，PatchCore CPU 推理较慢）：

```powershell
streamlit run app/streamlit_app.py
```

## 测试与静态检查（本地 CPU）

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
- [x] 阶段三：多类别 × PatchCore/EfficientAD 批量对比（7 类×2 模型，精度 / 速度 / FPS）
- [x] 阶段四：工程消融 + 阈值工程 + 报告图表（backbone/layers/coreset/neighbors，PR 曲线与工作点）
- [x] 阶段五：Streamlit Demo（上传图 → 原图/热力图/score/Normal·Defect，阈值可调）
- [x] 阶段六：手写最小 PatchCore（验证对算法原理的理解，CPU 冒烟通过）
- [x] AutoDL GPU 正式实验（阶段一 / 三 / 四）
- [x] README 结果回填与 GitHub 整理

## 参考实现

- [open-edge-platform/anomalib](https://github.com/open-edge-platform/anomalib)：主力框架（Apache-2.0）
- [amazon-science/patchcore-inspection](https://github.com/amazon-science/patchcore-inspection)：PatchCore 官方实现，Roth et al., CVPR 2022（arXiv:2106.08265）
- [nelson1425/EfficientAD](https://github.com/nelson1425/EfficientAD)：EfficientAD 复现，Batzner et al., WACV 2024（arXiv:2303.14535）
- [MVTec AD Dataset](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)
