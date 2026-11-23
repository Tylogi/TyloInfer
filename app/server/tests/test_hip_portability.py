from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from kernels.hip._build import stage_sources

ROOT = Path(__file__).resolve().parents[3]


def hashes(root):
    return {p.relative_to(root): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


def test_staging_is_content_addressed_and_excludes_cuda(tmp_path):
    root = tmp_path / "repository"
    kernel = root / "app/server/kernels/hip"
    kernel.mkdir(parents=True)
    (kernel / "fixed.cuh").write_text("#pragma once\n#define WIDTH 8\n")
    cuda = root / "app/server/kernels/cuda"
    cuda.mkdir()
    (cuda / "fixed.cuh").write_text("#pragma once\n#define WIDTH 32\n")
    original = hashes(root)
    cache = tmp_path / "cache"
    with ThreadPoolExecutor(max_workers=4) as executor:
        stages = list(executor.map(lambda _: stage_sources(root, cache), range(4)))
    assert len(set(stages)) == 1
    stage = stages[0]
    assert not (stage / "app/server/kernels/cuda").exists()
    target = stage / "app/server/kernels/hip/fixed.cuh"
    stamp = target.stat().st_mtime_ns
    assert stage_sources(root, cache) == stage
    assert target.stat().st_mtime_ns == stamp
    assert hashes(root) == original
    (kernel / "fixed.cuh").write_text("#pragma once\n#define WIDTH 16\n")
    updated = stage_sources(root, cache)
    assert updated != stage
    assert (updated / "app/server/kernels/hip/fixed.cuh").read_text().endswith("WIDTH 16\n")
    assert target.read_text().endswith("WIDTH 8\n")


def test_real_hipify_does_not_replace_staged_or_repository_hip_headers(tmp_path):
    pytest.importorskip("torch")
    from torch.utils.hipify import hipify_python
    from kernels.hip._ext import _SOURCES

    stage = stage_sources(ROOT, tmp_path / "cache")
    original = hashes(ROOT / "app/server/kernels/hip")
    original_cuda = hashes(ROOT / "app/server/kernels/cuda")
    staged = hashes(stage / "app/server/kernels/hip")
    build = tmp_path / "build"
    build.mkdir()
    hipify_python.hipify(project_directory=str(build), output_directory=str(build),
                        header_include_dirs=[str(stage)],
                        extra_files=[str(stage / "app/server/kernels/hip" / Path(source).name) for source in _SOURCES],
                        show_progress=False, is_pytorch_extension=True)
    after = hashes(stage / "app/server/kernels/hip")
    assert all(after[path] == digest for path, digest in staged.items())
    assert hashes(ROOT / "app/server/kernels/hip") == original
    assert hashes(ROOT / "app/server/kernels/cuda") == original_cuda


def test_loader_only_passes_staged_inputs_to_torch(tmp_path, monkeypatch):
    from kernels.hip import _ext

    captured = {}
    def load(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(version=SimpleNamespace(hip="test")))
    monkeypatch.setitem(sys.modules, "torch.utils.cpp_extension", SimpleNamespace(
        load=load, _get_build_directory=lambda *args, **kwargs: str(tmp_path / "cache")))
    monkeypatch.setattr(_ext, "_module", None)
    module = _ext.ext()
    assert _ext.ext() is module
    for path in captured["sources"] + captured["extra_include_paths"]:
        value = Path(path)
        assert value.exists() and value.is_relative_to(tmp_path)
    assert captured["build_directory"] == str(tmp_path / "cache")
