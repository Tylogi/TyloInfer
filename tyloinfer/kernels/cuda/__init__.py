"""TQ CUDA kernel sources and native operator bindings."""

from __future__ import annotations

from pathlib import Path

from ._sources import source_root


def tq_source_directory() -> Path:
    package = Path(__file__).resolve().parent
    return source_root(package) / "csrc/engine/backends/cuda/kernels/projection/tq"
