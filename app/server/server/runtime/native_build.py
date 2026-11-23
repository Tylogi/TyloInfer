"""Locate the native inference runtime this server was built for."""

from __future__ import annotations

import json
import os
import platform
import shutil
from dataclasses import dataclass
from pathlib import Path


class BuildError(RuntimeError):
    """Raised when the native runtime cannot be located."""


@dataclass(frozen=True)
class ManagedBuild:
    backend: str
    build_dir: Path
    executable: Path


_MANIFEST_VERSION = 1


def repository_root() -> Path:
    root = Path(__file__).resolve().parents[4]
    if not (root / "csrc" / "CMakeLists.txt").is_file():
        raise BuildError(
            "the native runtime sources are not installed; start the server from a TyloInfer "
            "source checkout or pass --running-executable"
        )
    return root


def build_manifest_path() -> Path:
    """Return the native build manifest written by a CMake build of this checkout."""

    return repository_root() / "build" / "mfq-runtime.json"


def _empty_manifest() -> dict[str, object]:
    return {"version": _MANIFEST_VERSION, "builds": {}}


def _read_manifest_document(path: Path) -> dict[str, object]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return _empty_manifest()
    if not isinstance(document, dict) or document.get("version") != _MANIFEST_VERSION:
        return _empty_manifest()
    if not isinstance(document.get("builds"), dict):
        return _empty_manifest()
    return document


def load_managed_build(backend: str) -> ManagedBuild | None:
    """Load the recorded build for this checkout and host, if its executable is present."""

    root = repository_root()
    document = _read_manifest_document(build_manifest_path())
    builds = document["builds"]
    assert isinstance(builds, dict)
    record = builds.get(backend)
    if not isinstance(record, dict):
        return None
    if record.get("system") != platform.system():
        return None
    if record.get("machine") != platform.machine().lower():
        return None
    required_strings = ("source_dir", "build_dir", "executable")
    if any(not isinstance(record.get(name), str) for name in required_strings):
        return None
    source_dir = Path(record["source_dir"]).expanduser().resolve()
    if source_dir != (root / "csrc").resolve():
        return None
    executable = Path(record["executable"]).expanduser().resolve()
    if not executable.is_file():
        return None
    return ManagedBuild(
        backend=backend,
        build_dir=Path(record["build_dir"]).expanduser().resolve(),
        executable=executable,
    )


def _cuda_compiler() -> Path | None:
    configured = os.environ.get("CUDACXX")
    if configured and Path(configured).is_file():
        return Path(configured)
    found = shutil.which("nvcc")
    if found:
        return Path(found)
    for variable in ("CUDA_HOME", "CUDA_PATH"):
        root = os.environ.get(variable)
        if not root:
            continue
        candidate = Path(root) / "bin" / ("nvcc.exe" if os.name == "nt" else "nvcc")
        if candidate.is_file():
            return candidate
    return None


def _hip_compiler() -> Path | None:
    found = shutil.which("hipcc")
    if found:
        return Path(found)
    for variable in ("ROCM_PATH", "ROCM_HOME"):
        root = os.environ.get(variable)
        if root:
            candidate = Path(root) / "bin" / "hipcc"
            if candidate.is_file():
                return candidate
    candidate = Path("/opt/rocm") / "bin" / "hipcc"
    if candidate.is_file():
        return candidate
    return None


def detect_backend(requested: str = "auto") -> str:
    system = platform.system()
    machine = platform.machine().lower()
    if requested == "metal":
        if system != "Darwin" or machine not in {"arm64", "aarch64"}:
            raise BuildError("the Metal runtime requires Apple silicon and macOS")
        return "metal"
    if requested == "cuda":
        if system not in {"Linux", "Windows"}:
            raise BuildError("the CUDA runtime is supported on Linux and Windows")
        if _cuda_compiler() is None:
            raise BuildError("CUDA was selected but nvcc was not found")
        return "cuda"
    if requested == "hip":
        if system != "Linux":
            raise BuildError("the HIP runtime is supported on Linux")
        if _hip_compiler() is None:
            raise BuildError("HIP was selected but hipcc was not found")
        return "hip"
    if requested != "auto":
        raise BuildError(f"unsupported inference backend: {requested}")
    if system == "Darwin" and machine in {"arm64", "aarch64"}:
        return "metal"
    if system in {"Linux", "Windows"} and _cuda_compiler() is not None:
        return "cuda"
    if system == "Linux" and _hip_compiler() is not None:
        return "hip"
    raise BuildError(
        "no supported inference accelerator was detected; TyloInfer requires Apple silicon/Metal "
        "on macOS, an NVIDIA CUDA toolkit, or an AMD ROCm/HIP toolchain on Linux"
    )
