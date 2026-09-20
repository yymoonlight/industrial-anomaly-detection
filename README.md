# Industrial Anomaly Detection — 工业视觉缺陷检测与异常定位系统

基于 **MVTec AD** 数据集，使用 **PatchCore** 与 **EfficientAD** 实现工业图像的无监督缺陷检测与像素级异常定位，覆盖「数据处理 → 模型训练/特征库构建 → 多类别实验 → 评估优化 → 异常热力图 → 推理脚本 → Streamlit Demo」的完整 AI 工程流程。

> 状态：🚧 第一阶段 Baseline 搭建中
> 框架：[Anomalib v2.6.2](https://github.com/open-edge-platform/anomalib)（PyTorch / Lightning）

## 目录结构

```
industrial-anomaly-detection/
├── configs/        # 配置文件（模型、数据、实验参数）
├── data/           # MVTec AD 数据集（不入库）
├── src/
│   ├── data/       # 数据接口与预处理
│   ├── models/     # PatchCore / EfficientAD 模型封装
│   ├── train/      # 训练 / 特征库构建统一入口
│   ├── evaluation/ # 指标（Image/Pixel AUROC、AUPRO、F1）与阈值评估
│   └── inference/  # 推理、anomaly score 与热力图
├── app/            # Streamlit Demo
├── scripts/        # 服务器运行脚本
├── results/        # 实验指标、日志、图表（不入库）
├── tests/          # 轻量测试
├── requirements.txt
├── README.md
└── .gitignore
```

## 数据集

- MVTec AD：15 个类别（10 物体 + 5 纹理），5,354 张高分辨率图像（3,629 张训练正常图 + 1,725 张测试图），缺陷测试图带像素级 mask；
- 下载与存储说明见 `项目任务书.md` 附录 B；
- 解压后放置为 `data/mvtec_anomaly_detection/<category>/{train,test,ground_truth}`；
- 许可证 **CC BY-NC-SA 4.0**，仅限非商业学术用途；数据集不提交 Git。

## 环境搭建

要求 **Python 3.10**（本地 Windows CPU 与远程 GPU 服务器保持一致）。

**本地（Windows，无 GPU，仅语法检查与轻量测试）：**

```powershell
conda create -n iad-cpu python=3.10 -y
conda activate iad-cpu
pip install "anomalib[cpu]==2.6.2"
pip install -r requirements.txt
python -m pytest tests/ -v
```

**GPU 服务器（AutoDL，RTX 3090/4090，CUDA 12.6 路线）：**

```bash
conda create -n iad python=3.10 -y
conda activate iad
pip install "anomalib[cu126]==2.6.2"
pip install -r requirements.txt
```

## 使用方式

待第一阶段 Baseline 跑通后补充（统一命令入口 `src/train/train.py`、`src/evaluation/evaluate.py`、`src/inference/predict.py`）。

## 路线图

- [ ] 阶段一：bottle 单类别 PatchCore Baseline（指标 + 热力图 + 日志）
- [ ] 阶段二：统一入口与配置化封装
- [ ] 阶段三：PatchCore vs EfficientAD 多类别对比（精度 / 速度 / FPS）
- [ ] 阶段四：工程优化实验（backbone、feature layer、coreset ratio、memory bank、阈值）
- [ ] 阶段五：predict.py + Streamlit Demo
- [ ] 收尾：README 完善与 GitHub 整理

## 参考实现

- [open-edge-platform/anomalib](https://github.com/open-edge-platform/anomalib)：主力框架（Apache-2.0）
- [amazon-science/patchcore-inspection](https://github.com/amazon-science/patchcore-inspection)：PatchCore 官方实现，Roth et al., CVPR 2022（arXiv:2106.08265）
- [nelson1425/EfficientAD](https://github.com/nelson1425/EfficientAD)：EfficientAD 复现，Batzner et al., WACV 2024（arXiv:2303.14535）
- [MVTec AD Dataset](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)
