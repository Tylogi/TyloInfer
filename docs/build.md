# Build the C++ runtime

CMake builds the native inference runtime in `csrc/` and selects Metal, CUDA or
HIP from the host toolchain. Run the result with
[`python -m serve`](serve.md) and `--running-executable`.

## Quick start

Install CMake 3.26 or newer and a C++ toolchain.

```shell
cd /path/to/TyloInfer

# NVIDIA (Linux or Windows) or AMD (Linux with hipcc): CMake detects the backend
cmake -S csrc -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target mfq-runtime --parallel

uv run --project studio python -m serve --model-dir ./models \
  --running-executable build/mfq-runtime
```

On Apple silicon the target is `mfq-decode-metal` and the binary lands in
`build/metal/`. MLX must be located explicitly because CMake searches
`<repo>/.venv`, while uv keeps the package in the project environment:

```shell
uv sync --project studio --extra metal
cmake -S csrc -B build -DCMAKE_BUILD_TYPE=Release \
  -DTYLOINFER_MLX_ROOT="$(uv run --project studio python -c 'import mlx, os; print(os.path.dirname(mlx.__file__))')"
cmake --build build --target mfq-decode-metal --parallel

uv run --project studio python -m serve --model-dir ./models \
  --running-executable build/metal/mfq-decode-metal
```

## Backend requirements

### Metal

- Apple silicon macOS;
- CMake 3.26 or newer and a C++ toolchain;
- the `metal` optional dependency, which provides MLX and its native assets.

### CUDA

- Linux or Windows;
- CMake 3.26 or newer and a C++/CUDA toolchain with `nvcc`;
- CUDA Toolkit 12 or newer, including `nvcc`, cuBLAS, and the CUDA runtime.

The default CUDA runtime needs no Python, PyTorch, or LibTorch. Migration A/B
builds may set `-DTYLOINFER_BUILD_TORCH_REFERENCE_RUNTIME=ON`; that reference
target has separate Python and LibTorch requirements.

### HIP (experimental)

- Linux with a ROCm/HIP toolchain and `hipcc`;
- CMake 3.26 or newer and a C++ toolchain.

This is not a claim of CUDA feature parity; consult the
[known gaps](runtime/support.md#amd-gpus-and-rocm-hip) before choosing a
model.

## Build options

| Option | Meaning | Default |
| --- | --- | --- |
| `-DTYLOINFER_BUILD_CUDA_RUNTIME=ON/OFF` | Build the CUDA runtime. | `ON` off macOS when `nvcc` is found |
| `-DTYLOINFER_BUILD_METAL_RUNTIME=ON/OFF` | Build the MLX/Metal runtime. | `ON` on Apple |
| `-DTYLOINFER_BUILD_HIP_RUNTIME=ON/OFF` | Build the ROCm/HIP runtime. | `ON` when `hipcc` is found and CUDA was not selected |
| `-DTYLOINFER_MLX_ROOT=PATH` | MLX package root with `include/`, `lib/libmlx.dylib`, and `lib/mlx.metallib`. | first `mlx` under `<repo>/.venv` |
| `-DTYLOINFER_BUILD_PRECOMPILED_METAL=ON` | Build the bundled Metal libraries for packaging. | `OFF` |
| `-DTYLOINFER_ENABLE_MFQ_COMPAT=ON` | Include legacy MFQ inference operators. | `OFF` |
| `-DTYLOINFER_BUILD_RUNTIME_COMMUNICATION=ON/OFF` | Build the runtime transport component. | `ON` off macOS |
| `-DBUILD_TESTING=OFF` | Skip the native test executables. | `ON` |

CUDA architectures follow `CMAKE_CUDA_ARCHITECTURES`, for example
`-DCMAKE_CUDA_ARCHITECTURES=90`.

## Native format profiles

The default build includes shared floating-point/FP8/MX operators and TQ.
Legacy MFQ inference operators are kept under each backend's `ops/mfq/` and
excluded from the ordinary build. Use a separate build directory for a
compatibility runtime:

```shell
cmake -S csrc -B build/native-compat -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DTYLOINFER_ENABLE_MFQ_COMPAT=ON
```

Compatibility adds legacy NINT/NVQ/MFE/SQ inference; it does not add offline
quantization or calibration. ROCm's existing model/format coverage is unchanged;
this switch does not add a TQ HIP implementation.

## Output

| Backend | Executable |
| --- | --- |
| Metal | `<build>/metal/mfq-decode-metal` |
| CUDA, HIP | `<build>/mfq-runtime` |
| CUDA on Windows | usually `<build>\mfq-runtime.exe`; multi-configuration generators may add `Release\` |

`mfq-runtime` runs inference and service mode. Multi-configuration generators may
place the executable in a configuration subdirectory, so pass the path you built
to `python -m serve --running-executable`.

`python -m serve` also reads `<repo>/build/mfq-runtime.json`, a managed-build
record left by older installations, and uses the executable it names when that
file is present, current, and points at an existing binary. Nothing in this
repository writes that file: after a plain CMake build, pass
`--running-executable`.

## Testing

The native tests are CTest cases. They are configured by default
(`BUILD_TESTING=ON`), which adds more than 140 test executables to a full build;
build only what you need:

```shell
cmake --build build --target mfq-file-range-reader-test --parallel
ctest --test-dir build -R file-range-reader --output-on-failure
ctest --test-dir build --output-on-failure        # everything that was built
```

| Location | Runs on |
| --- | --- |
| `csrc/tests/` | any host; no GPU, model weights, or accelerator backend |
| `csrc/backends/cuda/tests/` | only when the CUDA runtime is enabled, and needs an NVIDIA GPU |
| `csrc/backends/metal/tests/` | only on Apple silicon |

`ctest --test-dir build -N` lists the cases without running them. The Python,
browser, and plugin suites are listed in the
[README](../README.md#run-the-tests).

## Troubleshooting

### No supported accelerator was detected

CMake enables a backend only when its compiler is available. Make `nvcc` visible
through `PATH` or `CUDACXX`, or provide a supported Linux ROCm/HIP toolchain, or
build on Apple silicon.

### CMake completed without producing the executable

Check the configured backend in the CMake output
(`TYLOINFER_BUILD_*`), then the generator-specific path under the build
directory. `cmake --build build --target <name>` builds one target; configure
output lists the backend that was selected.
