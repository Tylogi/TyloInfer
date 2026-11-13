# `tyloinfer build`

`tyloinfer build` compiles the native inference runtime for the current machine. It
detects Metal, CUDA or the early HIP backend, invokes CMake, and records the resulting executable so
that [`tyloinfer serve`](serve.md) can find it without a binary path.

The `tyloinfer` command must come from a TyloInfer source checkout containing
`cpp_runtime/CMakeLists.txt`. TyloInfer derives the repository root from the installed
Python package, not from the current working directory. In an editable
development environment, run the CLI through uv from that checkout.

## Quick start

Sync the dependency set for the current host once:

```shell
cd /path/to/TyloInfer
uv sync --extra metal
```

On a CUDA host, the native build itself needs no Python accelerator package:

```shell
uv sync
```

Build with automatic backend detection:

```shell
uv run tyloinfer build
```

Automatic backend selection uses:

- Metal on Apple silicon macOS;
- CUDA on Linux or Windows when `nvcc` is available;
- HIP on Linux when `hipcc` is available and CUDA was not selected.

HIP remains an early backend with limited model coverage; see the
[support matrix](../runtime/support.md#amd-gpus-and-rocm-hip).

Backend detection does not check every compiler dependency. Requirements are
listed below. Pass the backend explicitly when debugging a build:

```shell
uv run tyloinfer build --backend metal
uv run tyloinfer build --backend cuda
```

## Native format profiles

The default build includes shared floating-point/FP8/MX operators and TQ.
Legacy MFQ inference operators are kept under each backend's `ops/mfq/` and
excluded from the ordinary build. Use a separate build directory for a
compatibility runtime:

```shell
uv run tyloinfer build --build-dir build/native-compat -- -DTYLOINFER_ENABLE_MFQ_COMPAT=ON
```

Compatibility adds legacy NINT/NVQ/MFE/SQ inference; it does not add offline
quantization or calibration. CMake and release packaging default to `OFF`;
packaging accepts the same `TYLOINFER_ENABLE_MFQ_COMPAT=ON` environment setting.
ROCm's existing model/format coverage is unchanged; this switch does not add a
TQ HIP implementation.

## Output and build manifest

The default CMake build directory is `<repo>/build/cpp_runtime`. Executable
paths:

| Backend | Executable |
| --- | --- |
| Metal | `<repo>/build/cpp_runtime/metal/mfq-decode-metal` |
| CUDA | `<repo>/build/cpp_runtime/mfq-runtime`, `mfq-diagnostics` |
| CUDA on Windows | Usually `<repo>\build\cpp_runtime\mfq-runtime.exe` (plus `mfq-diagnostics.exe`); multi-configuration generators may add `Release\` |

`mfq-runtime` runs inference and service mode; CUDA checks and benchmarks use
`mfq-diagnostics`:

```shell
mfq-diagnostics --check-deepseek-v41
```

TyloInfer searches the build tree after compilation, so generators may place the
executable in a configuration subdirectory.

A successful build writes `<repo>/build/mfq-runtime.json`. This location is
fixed relative to the source checkout even when `--build-dir` points elsewhere.
The manifest stores the backend, source directory, build directory, build type,
generator, forwarded CMake arguments, target, and the actual absolute
executable path. It is internal CLI state under the ignored `build/` directory;
users do not add the native executable to `PATH`.

To inspect a generator-dependent path, read the `executable` field for the
selected backend in the manifest:

```shell
uv run -m json.tool build/mfq-runtime.json
```

`tyloinfer serve` reads this manifest automatically. If the executable exists, it is
used directly. If it was deleted, TyloInfer rebuilds it in the recorded directory with
the recorded configuration.

## Custom build directory

Use `--build-dir` when the build tree belongs on another disk:

```shell
cd /path/to/TyloInfer
uv run tyloinfer build --build-dir /mnt/fast-build/tyloinfer
```

The manifest records the absolute build directory and executable path. A later
`tyloinfer serve` command finds this custom build without another option:

```shell
uv run tyloinfer serve --model /models/model.tq
```

Both commands must use the `tyloinfer` installation associated with the same source
checkout. The current working directory may change between them.

## Forward CMake configuration arguments

Arguments after `--` are passed to the CMake configure command. The separator
is required so the arguments are not parsed as `tyloinfer build` options.

```shell
uv run tyloinfer build --backend cuda -- \
  -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DGGML_CCACHE=OFF
```

The build manifest saves these arguments and reuses them when rebuilding a
missing executable.

## Command options

| Option | Meaning | Default |
| --- | --- | --- |
| `--backend {auto,cuda,hip,metal}` | Select or detect the accelerator backend. | `auto` |
| `--build-dir PATH` | Set the CMake build directory. | `<repo>/build/cpp_runtime` |
| `--build-type TYPE` | Set `CMAKE_BUILD_TYPE` and the build configuration. | `Release` |
| `-j N`, `--jobs N` | Set parallel build jobs. | Host CPU count |
| `--generator NAME` | Select a CMake generator. | Ninja when available, otherwise CMake default |
| `--dry-run` | Print configure and build commands without executing them. | Off |
| `-- CMAKE_ARG...` | Forward remaining arguments to CMake configuration. | None |

`--dry-run` does not compile an executable or update the manifest.

The public command is `tyloinfer`, not `mfq`. Native executable names such as
`mfq-runtime` and `mfq-decode-metal`, and the manifest `mfq-runtime.json`, are
retained internal names; do not rename them when following these instructions.

## Backend requirements

### Metal

- Apple silicon macOS;
- CMake 3.26 or newer;
- a C++ toolchain;
- the `metal` optional dependency, which provides MLX and its native assets.

For a source checkout managed by uv:

```shell
uv sync --extra metal
uv run tyloinfer build
```

### HIP (experimental)

- Linux with a ROCm/HIP toolchain and `hipcc`;
- CMake 3.26 or newer and a C++ toolchain.

Use `uv run tyloinfer build --backend hip`. This is not a claim of CUDA feature
parity; consult the [known gaps](../runtime/support.md#amd-gpus-and-rocm-hip)
before choosing a model.

### CUDA

- Linux or Windows;
- CMake 3.26 or newer;
- a C++/CUDA toolchain with `nvcc`;
- CUDA Toolkit 12 or newer, including `nvcc`, cuBLAS, and the CUDA runtime.

The default CUDA runtime needs no Python, PyTorch, or LibTorch. Migration A/B
builds may set `-DTYLOINFER_BUILD_TORCH_REFERENCE_RUNTIME=ON`; that reference target
has separate Python and LibTorch requirements.

TyloInfer checks `CUDACXX`, `PATH`, `CUDA_HOME`, and `CUDA_PATH` when locating
`nvcc`.

## Troubleshooting

### No supported accelerator was detected

Confirm the host is Apple silicon, make `nvcc` visible through one of the
CUDA locations above, or provide a supported Linux ROCm/HIP toolchain. `--backend` validates the requested backend; it does not
bypass platform and toolchain checks.

### CMake completed without producing the executable

Check the printed configure and build commands. TyloInfer searches custom generator
layouts by executable name; if the search still fails, the target was not
produced.

### Inspect the exact commands

```shell
uv run tyloinfer build --dry-run --backend cuda -- \
  -DCMAKE_CUDA_ARCHITECTURES=90
```

Next: [`tyloinfer serve`](serve.md).
