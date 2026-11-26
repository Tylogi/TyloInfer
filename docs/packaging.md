# Self-contained releases

TyloInfer Studio and Agent desktop releases support macOS and Windows. One script,
`packaging/build.py`, selects the product with `--app studio` (default) or
`--app agent`. Studio artifacts go to `packaging/dist/`; Agent artifacts go to
`packaging/dist/agent/`.

Studio packages include the server, Python environment, native runtime, and
required runtime libraries. Agent is a separate DSH desktop shell with bundled
Node.js, the TyloInfer plugin, and the same local Core service and native runtime.

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
uv run --no-project packaging/build.py --target macos
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
uv run --no-project packaging/build.py --target windows
```

Prepare Visual Studio Build Tools with the C++ x64 workload, a CUDA Toolkit,
`uv`, CMake, Ninja, Rust, Node.js/pnpm, and NSIS (or Tauri's cached NSIS).
Run from an **x64 Native Tools Command Prompt** or an x64-configured Developer
PowerShell. The caller owns toolchain setup: the build script does not run
`VsDevCmd.bat`, patch toolchain PATH entries, or guess an installed CUDA version.

For Studio, set `CUDA_PATH` to the intended Toolkit; `nvcc` on PATH must resolve
to that Toolkit's `bin/nvcc.exe`. The VS terminal must provide `cl`, the SDK/linker
environment and `VCToolsRedistDir` with x64 redistributables. Missing prerequisites
fail before building. For example, check the prepared terminal with:

```powershell
Get-Command cl, nvcc, uv, cmake, ninja, cargo, pnpm.cmd
$env:CUDA_PATH
$env:VCToolsRedistDir
```

The script still owns build-local isolation: its release venv, temporary files,
Cargo output, and compiler/signing options. It only changes its private child
process environment. C/C++ overrides use `CFLAGS`/`CXXFLAGS`; CUDA overrides,
including custom NVCC host options, use `CUDAFLAGS` rather than implicitly
copying C++ flags into NVCC.

Override release inputs when needed, and pass `--clean` to rebuild the native
sidecar from scratch:

```powershell
$env:TYLOINFER_RELEASE_CUDA_ARCHITECTURES = '89'
$env:TYLOINFER_RELEASE_JOBS = '2'
uv run --no-project packaging/build.py --target windows --clean
```

Installer collection requires one matching installer in the bundle directory;
it reports ambiguous stale outputs rather than selecting a file by timestamp.

The installer supports 64-bit Windows with a compatible NVIDIA GPU and driver.
It bundles CUDA runtime DLLs from the build machine, so users do not need the
CUDA Toolkit.

## Agent

Build on the target OS (Windows x64 or Apple Silicon):

```shell
uv run --no-project packaging/build.py --app agent --target windows
# On macOS, use --target macos.
```

Agent needs the Core build prerequisites above, plus the official Node.js 24
archive (24.9 or newer, with its LICENSE). Its separate Cargo output and runtime
staging do not replace Studio artifacts.

Windows Agent requires Windows 10 1803 or newer (or Windows 11). The build
precompresses Core and Agent resources into a ZIP before handing them to NSIS,
avoiding its 2 GiB uncompressed solid-block limit. The installer extracts the ZIP
with Windows' native tar.exe and removes the temporary archive. Disk space checks
include both the temporary ZIP and its expanded payload. End users do not
need Python, Node.js, CUDA Toolkit, or a development environment. Uninstall removes
only installed payload paths and keeps the user profile outside the installation.

The validated Agent stage is promoted with bounded retries for temporary Windows
file locks. A failed promotion restores the previous runtime and retains the
validated stage; if rollback also fails, the error reports both recovery paths.
Do not remove a reported backup until the prior runtime has been recovered.

Run packaging regressions without building CUDA or Rust:

```shell
node --test app/src-tauri/agent/runtime-stage.test.mjs
python -m unittest discover -s packaging/tests -v
```

The Tauri build hook deploys the locked production dependency graph, checks the
bundled manifests, and launches the actual profile with a temporary `DSH_HOME`.
The smoke test exchanges the launch token for a browser cookie, verifies the DSH
shell and TyloInfer client plugin, and checks graceful shutdown. It runs outside
the installation directory with a minimal PATH and never uses the user's profile.
Run it against a staged or installed runtime with:

```shell
node app/src-tauri/agent/smoke.mjs "path/to/agent-runtime"
```

Startup failures report the location of `dsh/agent-startup.log` under Agent's
application data directory. On Windows, the desktop launcher converts safe
verbatim paths (`\\?\...`) to ordinary paths before passing them to Node;
Node 24 cannot reliably resolve verbatim script-entry paths. The Rust host
integration test uses a canonicalized runtime path to cover this distinction
from launching Node directly in a terminal. After staging, run the desktop-host
regression explicitly (it is ignored by the ordinary Cargo test suite):

```powershell
# Run from app/src-tauri on Windows.
$env:CARGO_TARGET_DIR = 'target/agent'
$env:TAURI_CONFIG = '{"identifier":"com.tylogi.tyloinfer.agent","build":{"frontendDist":"agent/ui"},"app":{"windows":[]}}'
cargo test --release --target x86_64-pc-windows-msvc --features agent agent::host::tests -- --include-ignored
```

Both the desktop host and the release smoke test require readiness within 30
seconds. On macOS the smoke test runs after runtime code signing.

## Runtime behavior

Studio starts the bundled `tyloinfer-server` binary locally without a model. The catalog and native
file picker can load, unload, or replace GGUF and TQ models (`.mfq` is also
recognized). Architecture and tensor-format coverage still follows the
[runtime support matrix](runtime/support.md). Studio records external
paths without copying files; selecting one shard registers the full sibling
shard family.
