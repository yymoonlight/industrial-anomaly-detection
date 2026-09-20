"""项目骨架冒烟测试：不依赖 GPU / anomalib，仅校验目录与包结构完整。

在项目根目录运行：
    python -m pytest tests/ -v
"""

import importlib
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EXPECTED_DIRS = [
    "configs",
    "data",
    "src",
    os.path.join("src", "data"),
    os.path.join("src", "models"),
    os.path.join("src", "train"),
    os.path.join("src", "evaluation"),
    os.path.join("src", "inference"),
    "app",
    "scripts",
    "results",
    "tests",
]

EXPECTED_PACKAGES = [
    "src",
    "src.data",
    "src.models",
    "src.train",
    "src.evaluation",
    "src.inference",
]


@pytest.mark.parametrize("rel_dir", EXPECTED_DIRS)
def test_directory_exists(rel_dir):
    assert os.path.isdir(os.path.join(ROOT, rel_dir)), f"缺少目录: {rel_dir}"


@pytest.mark.parametrize("pkg", EXPECTED_PACKAGES)
def test_package_importable(pkg):
    assert importlib.import_module(pkg) is not None


def test_root_files_exist():
    for name in ("requirements.txt", "README.md", ".gitignore"):
        assert os.path.isfile(os.path.join(ROOT, name)), f"缺少文件: {name}"
