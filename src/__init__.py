"""工业视觉缺陷检测与异常定位系统（industrial-anomaly-detection）。

子模块职责：
- data: MVTec AD 数据接口与预处理
- models: PatchCore / EfficientAD 模型封装
- train: 训练 / 特征库构建统一入口
- evaluation: 指标计算与阈值评估
- inference: 推理、anomaly score 与异常热力图

注意：本模块在导入早期（torch/timm 之前）统一设置运行环境变量：
- HF_ENDPOINT：国内网络下走 HuggingFace 镜像下载预训练权重；
- HF_HOME：Windows 用户目录含非 ASCII 字符（如中文用户名）时，
  把权重缓存重定向到项目内纯英文路径，规避 anomalib/timm 的路径校验问题。
已显式设置过的环境变量不会被覆盖。
"""

import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

if os.name == "nt":
    try:
        str(Path.home()).encode("ascii")
    except UnicodeEncodeError:
        os.environ.setdefault("HF_HOME", str(_PROJECT_ROOT / ".cache" / "huggingface"))

        # 中文用户名会让系统临时目录（如 %LOCALAPPDATA%\Temp）含非 ASCII 字符，
        # anomalib 校验上传/临时图片路径时会报 "Path contains non-printable characters"。
        # 把临时目录统一重定向到项目内纯英文路径，并直接覆盖 tempfile.tempdir。
        import tempfile

        _tmp_root = _PROJECT_ROOT / ".cache" / "tmp"
        _tmp_root.mkdir(parents=True, exist_ok=True)
        for _var in ("TMPDIR", "TEMP", "TMP"):
            os.environ.setdefault(_var, str(_tmp_root))
        tempfile.tempdir = str(_tmp_root)

    # 兼容：在 Windows 加载 Linux 训练的 checkpoint 时，权重内 pickle 的 PosixPath
    # 无法在本机实例化（NotImplementedError），临时映射到 WindowsPath。
    # 这些路径仅用于 checkpoint 内记录，不影响权重加载与推理结果。
    import pathlib

    pathlib.PosixPath = pathlib.WindowsPath

__version__ = "0.1.0"
