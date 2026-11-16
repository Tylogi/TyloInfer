"""Assemble the Python distribution and build its wheel."""

import argparse
import re
import runpy
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


def cuda_files(repository: Path) -> tuple[Path, ...]:
    """Validate and return the quoted-include closure of the CUDA JIT manifest."""
    repository = repository.resolve()
    package = repository / 'tyloinfer/kernels/cuda'
    manifest = runpy.run_path(str(package / '_sources.py'))
    roots = manifest['include_roots'](repository)
    pending = list(manifest['sources'](repository))
    entrypoints = set(pending)
    visited: set[Path] = set()
    kernel_root = repository / 'csrc/engine/backends/cuda/kernels'
    binding_root = repository / 'csrc/bindings/cuda'
    native_root = repository / 'csrc/engine/backends/cuda/native'
    encoding_root = repository / 'csrc/engine/weights/encoding'
    ggml_root = repository / 'csrc/components/ggml'
    include = re.compile(r'^\s*#\s*include\s*"([^"\n]+)"', re.MULTILINE)
    while pending:
        source = pending.pop().resolve()
        if source in visited:
            continue
        if not source.is_file():
            raise RuntimeError(f'CUDA JIT source is missing: {source}')
        allowed = (
            source in entrypoints
            or (source.is_relative_to(kernel_root) and source.suffix in {'.h', '.cuh', '.inc'})
            or (source.is_relative_to(binding_root) and source.suffix in {'.h', '.cuh'})
            or source == native_root / 'env.h'
            or (source.is_relative_to(encoding_root) and source.suffix == '.h')
            or (source.is_relative_to(ggml_root) and source.suffix in {'.h', '.cuh', '.inc'})
        )
        if not allowed:
            raise RuntimeError(f'CUDA JIT depends on non-portable source: {source}')
        visited.add(source)
        for name in include.findall(source.read_text(encoding='utf-8')):
            # These mutually exclusive vendor headers are outside the CUDA JIT.
            if source == ggml_root / 'src/ggml-cuda/common.cuh' and name in {'vendors/hip.h', 'vendors/musa.h'}:
                continue
            candidates = (source.parent / name, *(root / name for root in roots))
            target = next((path.resolve() for path in candidates if path.is_file()), None)
            if target is None:
                raise RuntimeError(f'CUDA JIT include {name!r} from {source} cannot be resolved')
            pending.append(target)
    return tuple(sorted(visited))


def collect_cuda_sources(repository: Path, project: Path, files: tuple[Path, ...]) -> None:
    """Stage the validated CUDA sources with their canonical csrc paths."""
    repository = repository.resolve()
    for source in files:
        destination = project / 'tyloinfer/kernels/cuda/_src' / source.relative_to(repository)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def collect_tq_sources(repository: Path, project: Path) -> None:
    kernels = Path('csrc/engine/backends/cuda/kernels/projection/tq')
    resources = project / 'tyloinfer/kernels/cuda/_src' / kernels
    shutil.copytree(repository / kernels, resources, dirs_exist_ok=True)


def build_wheel(output_dir: Path) -> Path:
    """Build a wheel after staging the shared CUDA source closure."""
    repository = Path(__file__).resolve().parents[2]
    files = cuda_files(repository)
    with TemporaryDirectory(prefix="tyloinfer-wheel-") as temporary:
        project = Path(temporary) / "project"
        project.mkdir()
        for name in ("pyproject.toml", "README.md", "LICENSE", "NOTICE"):
            shutil.copy2(repository / name, project / name)
        shutil.copytree(repository / "LICENSES", project / "LICENSES")
        shutil.copytree(
            repository / "tyloinfer", project / "tyloinfer",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )

        collect_tq_sources(repository, project)
        collect_cuda_sources(repository, project, files)
        resources = project / 'tyloinfer/kernels/cuda/_src/csrc/engine/backends/cuda/kernels/projection/tq'
        included = {
            Path('tyloinfer/kernels/cuda/_src') / source.relative_to(repository)
            for source in files
        }
        included.update(source.relative_to(project) for source in resources.iterdir())
        (project / 'MANIFEST.in').write_text(
            ''.join(f'include {path.as_posix()}\n' for path in sorted(included)), encoding='utf-8')

        artifacts = Path(temporary) / "dist"
        subprocess.run(
            ["uv", "build", "--wheel", str(project), "--out-dir", str(artifacts)],
            check=True, stdout=sys.stderr,
        )
        wheel, = artifacts.glob("*.whl")
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        destination = output_dir / wheel.name
        shutil.copy2(wheel, destination)
        return destination


def main() -> None:
    """Build the distribution in the requested output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent / "dist",
    )
    args = parser.parse_args()
    print(build_wheel(args.output_dir))


if __name__ == "__main__":
    main()
