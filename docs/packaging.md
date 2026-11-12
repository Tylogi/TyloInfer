# Self-contained releases

TyloInfer Studio desktop releases support macOS and Windows. Build scripts live in
`packaging/` and write artifacts to `packaging/dist/`.

The packages include the server, Python environment, native runtime, and
required runtime libraries. `tyloinfer serve` or TyloInfer manages the native
sidecars.

## macOS

The macOS release packages the TyloInfer service CLI and Studio into an
Apple Silicon DMG. The frozen CLI is named `tyloinfer-cli` inside the bundle;
source checkouts use the public `tyloinfer` command. It includes:

- the unified service CLI with service and model management;
- the private native `mfq-decode-metal` sidecar;
- `libmlx`, `libjaccl`, `mlx.metallib`, and the AVFoundation video bridge;
- TyloInfer and its production Web UI, without PyTorch.

Build from a clean checkout on Apple Silicon:

```shell
cd packaging
./build_release_mac.sh
```

The script builds from the canonical `cpp_runtime` source tree, validates the
native runtime, CLI subcommands, Rust desktop app, Web UI, architectures, and
code signatures, then copies the DMG to `packaging/dist/`.

## Windows

The Windows release packages the service CLI (`tyloinfer-cli.exe`), Studio,
and the native CUDA `mfq-runtime` sidecar into a single x64 NSIS installer. Service and model management are included. The package
excludes PyTorch and the training, calibration, quantization, and MiniCPM-o
dependency groups. The CUDA, cuBLAS, MSVC, and optional OpenMP runtime DLLs are
bundled with the sidecar, so end users do not need Python, the CUDA Toolkit, or a
native build environment.

Build from PowerShell 7 or newer on 64-bit Windows:

```powershell
cd packaging
.\build_release_windows.ps1
```

The build host needs Visual Studio Build Tools with the C++ x64 workload, a CUDA
Toolkit, `uv`, CMake, Ninja, Rust, Node.js/npm, and NSIS. The release script declares its supported CUDA architectures. Override release inputs when needed:

```powershell
.\build_release_windows.ps1 -CudaArchitectures 89 -Jobs 2
```

The installer supports 64-bit Windows with a compatible NVIDIA GPU and driver.
It bundles CUDA runtime DLLs from the build machine, so users do not need the
CUDA Toolkit.

## Runtime behavior

Studio starts the bundled CLI's `serve` subcommand locally without a model. The catalog and native
file picker can load, unload, or replace GGUF and TQ models (`.mfq` is also
recognized). Architecture and tensor-format coverage still follows the
[runtime support matrix](runtime/support.md). Studio records external
paths without copying files; selecting one shard registers the full sibling
shard family.
