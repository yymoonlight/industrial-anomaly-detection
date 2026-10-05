"""最小 PatchCore CPU 端到端冒烟（slow）：合成数据，产出三联图。"""

from __future__ import annotations

import sys

import pytest


@pytest.mark.slow
def test_minimal_patchcore_cpu(synthetic_dataset) -> None:
    case = synthetic_dataset
    out = case / "minimal"

    import baselines.minimal_patchcore as mp

    old_argv = sys.argv
    sys.argv = ["minimal_patchcore", "--layout", "folder", "--root", str(case),
                "--out", str(out), "--device", "cpu"]
    try:
        mp.main()
    finally:
        sys.argv = old_argv

    assert list(out.glob("*.png"))
