"""pytest 全局配置与合成数据 fixture。

注意：pytest 默认临时目录在 C:\\Users\\<用户名>\\AppData\\Local\\Temp 下，
若 Windows 用户名含非 ASCII 字符会触发 anomalib 的路径校验错误，
因此合成数据统一放在项目内纯英文目录 .test_tmp/ 下并在用例结束后清理。
"""

import os
import shutil
import uuid

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import pytest
from PIL import Image

from src.utils.config import PROJECT_ROOT

TEST_TMP = PROJECT_ROOT / ".test_tmp"


@pytest.fixture
def synthetic_dataset():
    """在项目内英文路径生成一套符合 anomalib Folder 约定的合成数据。

    结构：
        good/        训练正常图（8 张）
        good_test/   测试正常图（4 张）
        abnormal/    测试异常图（4 张）
        mask/        异常像素级 mask（4 张）
    """
    case_dir = TEST_TMP / f"case_{uuid.uuid4().hex[:8]}"
    case_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    size = 128

    def save(folder, idx, arr):
        p = case_dir / folder / f"{idx:03d}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr.astype("uint8")).save(p)

    try:
        for i in range(8):
            save("good", i, rng.integers(0, 255, (size, size, 3), dtype=np.int16))
        for i in range(4):
            save("good_test", i, rng.integers(0, 255, (size, size, 3), dtype=np.int16))
        for i in range(4):
            arr = rng.integers(0, 255, (size, size, 3), dtype=np.int16).copy()
            mask = np.zeros((size, size), dtype=np.uint8)
            arr[30:60, 30:60] = np.clip(arr[30:60, 30:60] + 120, 0, 255)
            mask[30:60, 30:60] = 255
            save("abnormal", i, arr)
            save("mask", i, mask)
        yield case_dir
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)
