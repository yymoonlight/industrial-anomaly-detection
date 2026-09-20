"""工业视觉缺陷检测与异常定位系统（industrial-anomaly-detection）。

子模块职责：
- data: MVTec AD 数据接口与预处理
- models: PatchCore / EfficientAD 模型封装
- train: 训练 / 特征库构建统一入口
- evaluation: 指标计算与阈值评估
- inference: 推理、anomaly score 与异常热力图
"""

__version__ = "0.1.0"
