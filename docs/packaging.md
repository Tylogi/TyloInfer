# Self-contained releases

TyloInfer Studio desktop releases support macOS and Windows. One script,
`packaging/build_release.py`, builds either target and writes artifacts to
`packaging/dist/`.

The packages include the server, Python environment, native runtime, and
required runtime libraries. `python -m server.serve` or TyloInfer manages the native
sidecars.

## macOS

The macOS release packages the TyloInfer service (`tyloinfer-server`), the
private native `mfq-decode-metal` sidecar and Studio into an Apple Silicon DMG.
It includes:

- the frozen service process, started by Studio for local mode;
- the private native `mfq-decode-metal` sidecar;
- `libmlx`, `libjaccl`, `mlx.metallib`, and the AVFoundation video bridge;
- TyloInfer and its production Web UI, without PyTorch.

Build from a clean checkout on Apple Silicon:

```shell
uv run --no-project packaging/build_release.py --target macos
```

The script builds from the canonical `csrc` source tree, validates the
native runtime, the Rust desktop app, Web UI, architectures, and
code signatures, then copies the DMG to `packaging/dist/`.

## Windows

The Windows release packages the TyloInfer service (`tyloinfer-server.exe`),
Studio, and the native CUDA `mfq-runtime` sidecar
into a single x64 NSIS installer. The package
excludes PyTorch and the training, calibration, quantization, and MiniCPM-o
dependency groups. The CUDA, cuBLAS, MSVC, and optional OpenMP runtime DLLs are
bundled with the sidecar, so end users do not need Python, the CUDA Toolkit, or a
native build environment.

Build on 64-bit Windows:

```shell
uv run --no-project packaging/build_release.py --target windows
```

The build host needs Visual Studio Build Tools with the C++ x64 workload, a CUDA
Toolkit, `uv`, CMake, Ninja, Rust, Node.js/npm, and NSIS. The release script declares its supported CUDA architectures. Override release inputs when needed, and pass `--clean` to rebuild the native sidecar from scratch:

```shell
TYLOINFER_RELEASE_CUDA_ARCHITECTURES=89 TYLOINFER_RELEASE_JOBS=2 \
  uv run --no-project packaging/build_release.py --target windows --clean
```

The installer supports 64-bit Windows with a compatible NVIDIA GPU and driver.
It bundles CUDA runtime DLLs from the build machine, so users do not need the
CUDA Toolkit.

## Runtime behavior

Studio starts the bundled `tyloinfer-server` binary locally without a model. The catalog and native
file picker can load, unload, or replace GGUF and TQ models (`.mfq` is also
recognized). Architecture and tensor-format coverage still follows the
[runtime support matrix](runtime/support.md). Studio records external
paths without copying files; selecting one shard registers the full sibling
shard family.
