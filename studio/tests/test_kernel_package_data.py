from fnmatch import fnmatchcase
from importlib.resources import files
from pathlib import Path
import re

import pytest


@pytest.mark.parametrize("backend, name", [
    ("metal", "tq_decode.h"),
    ("metal", "tq_matmul.h"),
    ("cuda", "tq_ops.cu"),
    ("cuda", "tq_indexed.cu"),
    ("cuda", "tq_routed_views.cu"),
])
def test_kernel_source_resources_are_installed(backend, name):
    resource = files("kernels").joinpath(f"{backend}/{name}")
    assert resource.is_file()
    assert resource.read_text(encoding="utf-8").strip()


def test_cuda_tq_local_include_dependencies_are_installed():
    tomllib = pytest.importorskip("tomllib")
    project = Path(__file__).resolve().parents[1] / "pyproject.toml"
    patterns = tomllib.loads(project.read_text(encoding="utf-8"))["tool"]["setuptools"]["package-data"]["kernels.cuda"]
    root = files("kernels").joinpath("cuda")
    pending, seen = ["tq_ops.cu", "tq_indexed.cu", "tq_routed_views.cu"], set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        assert any(fnmatchcase(name, pattern) for pattern in patterns), name
        resource = root.joinpath(name)
        assert resource.is_file(), name
        pending.extend(re.findall(r'^\s*#\s*include\s*"([^"]+)"',
            resource.read_text(encoding="utf-8"), re.MULTILINE))
    assert "tq_decode.cuh" in seen
    assert "tq_prefill_gemm.cuh" in seen
    assert "tq_prefill_scale.cuh" in seen
    assert "prefill_scale.cuh" in seen
    assert "routed_prefill_f16.cuh" in seen
    assert "tq_projection.cuh" in seen
    assert "tq_routed_views.cuh" in seen
    assert "tq_matmul_core.cuh" in seen
    assert "tq_indexed.cuh" in seen


def test_wheel_declares_metal_kernel_resources():
    tomllib = pytest.importorskip("tomllib")
    project = Path(__file__).resolve().parents[1] / "pyproject.toml"
    packages = tomllib.loads(project.read_text(encoding="utf-8"))["tool"]["setuptools"]["package-data"]
    patterns = packages.get("kernels.metal", [])
    for name in ("tq_decode.h", "tq_matmul.h"):
        assert any(fnmatchcase(name, pattern) for pattern in patterns), name
    cuda_patterns = packages.get("kernels.cuda", [])
    for name in ("tq_ops.cu", "tq_indexed.cu", "tq_routed_views.cu", "tq_projection.cuh", "gr_bf16_projection.cuh"):
        assert any(fnmatchcase(name, pattern) for pattern in cuda_patterns), name
