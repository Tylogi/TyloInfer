#!/usr/bin/env python3
"""Build self-contained Studio or Agent desktop releases on the target OS."""
#
#   uv run --no-project packaging/build.py --app studio --target macos
#   uv run --no-project packaging/build.py --app agent --target windows
#
# Overrides (environment): TYLOINFER_RELEASE_PYTHON, TYLOINFER_RELEASE_VENV,
# TYLOINFER_RELEASE_OUTPUT_DIR, TYLOINFER_RELEASE_JOBS,
# TYLOINFER_RELEASE_CUDA_ARCHITECTURES, TYLOINFER_RELEASE_SIGNING_IDENTITY,
# TYLOINFER_ENABLE_MFQ_COMPAT.
#
# Steps are shared; platform values sit in new_context() and the platform steps
# are selected by context.windows.

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass, field
import ctypes
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import zipfile

from agent_payload import build_agent_payload

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT = SCRIPT_DIR.parent
APP = PROJECT / "app"
TAURI = APP / "src-tauri"
PYTHON_PROJECT = APP / "server"
SERVER_ENTRY = PYTHON_PROJECT / "serve.py"
SERVER_NAME = "tyloinfer-server"


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


def run(command: Sequence[str | Path], *, cwd: Path | None = None,
        env: dict[str, str] | None = None, capture: bool = False) -> str:
    """Stream build commands; capture only queries, retaining diagnostics on failure."""
    arguments = [str(item) for item in command]
    display = subprocess.list2cmdline(arguments) if os.name == "nt" else shlex.join(arguments)
    try:
        result = subprocess.run(arguments, cwd=cwd, env=env,
                                capture_output=capture, text=True)
    except OSError as error:
        fail(f"could not run {display}: {error}")
    if result.returncode != 0:
        details = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part) if capture else ""
        fail(f"command failed ({result.returncode}): {display}"
             + (f"\n{details}" if details else ""))
    return result.stdout if capture else ""


def remove_directory(directory: Path) -> None:
    """A missing staging directory is harmless; a failed cleanup is not."""
    try:
        shutil.rmtree(directory)
    except FileNotFoundError:
        if directory.exists():
            raise


def _flags(flag: str, values: tuple[str, ...]) -> list[str]:
    return [item for value in values for item in (flag, value)]


# Shared by both targets. Platform-only entries live in new_context(): the macOS
# server bundles the MLX runtime (mlx_lm pulls in transformers), the Windows
# server has no MLX and leaves transformers and safetensors out.
PYINSTALLER_ARGS = [
    *_flags("--collect-data", ("_gyro", "kernels", "server")),
    *_flags("--collect-all", (
        "av", "fastapi", "httpx", "PIL", "pydantic", "pypdf", "pyarrow", "pandas",
        "sympy", "regex", "loguru", "antlr4", "yaml", "huggingface_hub",
        "modelscope_hub", "uvicorn", "websockets")),
    *_flags("--copy-metadata", ("huggingface-hub", "modelscope-hub")),
    *_flags("--exclude-module", (
        "torch", "scipy", "tiktoken", "minicpmo_utils", "stepaudio2_minicpmo",
        "torchaudio", "onnxruntime", "s3tokenizer", "hyperpyyaml", "librosa")),
]


@dataclass
class Context:
    windows: bool
    triple: str
    architecture: str
    clean: bool
    venv: Path
    output: Path
    build: Path
    native_build: Path
    pyinstaller_build: Path
    spec: Path
    sidecars: Path
    resources: Path
    stage_directories: tuple[str, ...]
    native_target: str
    jobs: str
    uv_extras: tuple[str, ...]
    pyinstaller_extras: list[str]
    product: str = "studio"
    deployment_target: str = "26.2"
    signing_identity: str = "-"
    cuda_architectures: str = ""
    environment: dict[str, str] = field(default_factory=os.environ.copy)

    @property
    def bin_directory(self) -> str:
        return "Scripts" if self.windows else "bin"

    @property
    def executable(self) -> str:
        return ".exe" if self.windows else ""

    def in_venv(self, name: str) -> Path:
        return self.venv / self.bin_directory / f"{name}{self.executable}"

    @property
    def server(self) -> Path:
        return self.resources / SERVER_NAME / f"{SERVER_NAME}{self.executable}"

    @property
    def sidecar(self) -> Path:
        return self.sidecars / f"{self.native_target}-{self.triple}{self.executable}"

    @property
    def cargo_target(self) -> Path:
        configured = self.environment.get("CARGO_TARGET_DIR")
        if not configured:
            return TAURI / "target"
        return TAURI / configured

    @property
    def frameworks(self) -> Path:
        return SCRIPT_DIR / "Frameworks"

    @property
    def windows_runtime(self) -> Path:
        return SCRIPT_DIR / "windows-runtime"

    @property
    def cuda_root(self) -> Path:
        return configured_cuda_root(self.environment)


# --- preflight ------------------------------------------------------------

def add_to_path(environment: dict[str, str], directory: str) -> None:
    if directory and os.path.isdir(directory):
        parts = environment.get("PATH", "").split(os.pathsep)
        if directory not in parts:
            environment["PATH"] = directory + os.pathsep + environment.get("PATH", "")


def require_commands(environment: dict[str, str], names: tuple[str, ...]) -> None:
    for name in names:
        if shutil.which(name, path=environment.get("PATH", "")) is None:
            fail(f"{name} is required on PATH; prepare the build toolchain before running this script"
                 + (" (use an x64 Visual Studio Developer PowerShell on Windows)" if os.name == "nt" else ""))


def configured_cuda_root(environment: dict[str, str]) -> Path:
    configured = environment.get("CUDA_PATH")
    if not configured or not (Path(configured) / "bin" / "nvcc.exe").is_file():
        fail("set CUDA_PATH to the CUDA Toolkit to package (must contain bin/nvcc.exe)")
    return Path(configured)


def total_physical_bytes() -> int:
    if not hasattr(ctypes, "windll"):
        return 0

    class Status(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    status = Status()
    status.dwLength = ctypes.sizeof(Status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return 0
    return status.ullTotalPhys


def windows_jobs(environment: dict[str, str]) -> str:
    if environment.get("TYLOINFER_RELEASE_JOBS"):
        return environment["TYLOINFER_RELEASE_JOBS"]
    # The largest native CUDA translation units need several GiB each, so bound
    # parallelism by installed memory.
    installed = total_physical_bytes()
    if not installed:
        return str(os.cpu_count() or 1)
    return str(max(1, min(installed // (5 * 1024 ** 3), os.cpu_count() or 1)))


def preflight(context: Context) -> None:
    environment = context.environment
    node = ("node",) if context.product == "agent" else ()
    if not context.windows:
        if sys.platform != "darwin" or platform.machine() != "arm64":
            fail("this release target requires native Apple Silicon macOS")
        require_commands(environment, ("uv", "cmake", "ninja", "cargo", "pnpm", "codesign",
                                      "hdiutil", "install_name_tool", "lipo", "xcrun", *node))
        for shader in ("metal", "metallib"):
            run(["xcrun", "--find", shader], env=environment, capture=True)
        return

    if sys.platform != "win32" or sys.maxsize <= 2 ** 31:
        fail("this release target must be built on 64-bit Windows")
    require_commands(environment, ("uv", "cmake", "ninja", "cargo", "pnpm.cmd", "cl", "nvcc", *node))
    nvcc = context.cuda_root / "bin" / "nvcc.exe"
    configured_nvcc = shutil.which("nvcc", path=environment.get("PATH", ""))
    if not configured_nvcc or Path(configured_nvcc).resolve() != nvcc.resolve():
        fail("nvcc on PATH must match CUDA_PATH; select one CUDA Toolkit in the build terminal")
    redist = environment.get("VCTOOLSREDISTDIR")
    if not redist or not (Path(redist) / "x64").is_dir():
        fail("VCToolsRedistDir must point to the MSVC redistributables; use an x64 VS developer terminal")


def clean(context: Context) -> None:
    for name in context.stage_directories:
        remove_directory(SCRIPT_DIR / name)


def initialize_staging(context: Context) -> None:
    """Prepare staging and temporary paths for both clean and incremental builds."""
    for directory in (context.native_build, context.pyinstaller_build, context.spec,
                      context.sidecars, context.resources, context.output):
        directory.mkdir(parents=True, exist_ok=True)
    if context.windows:
        context.windows_runtime.mkdir(parents=True, exist_ok=True)
        temporary = context.build / "tmp"
        temporary.mkdir(parents=True, exist_ok=True)
        context.environment["TEMP"] = str(temporary)
        context.environment["TMP"] = str(temporary)
        context.environment["PYINSTALLER_CONFIG_DIR"] = str(context.build / "pyinstaller-cache")
    else:
        # Frameworks is in stage_directories; stage_macos_runtime only writes into it.
        context.frameworks.mkdir(parents=True, exist_ok=True)


# --- keep build-host paths out of the binaries ----------------------------

def short_path(path: str) -> str:
    """Windows 8.3 alias of a path, which native generators can emit."""
    buffer = ctypes.create_unicode_buffer(32768)
    length = ctypes.windll.kernel32.GetShortPathNameW(path, buffer, 32768)
    return buffer.value if 0 < length < 32768 and buffer.value != path else ""


def release_path_sources(context: Context) -> list[tuple[str, str]]:
    if context.windows:
        sources = [(str(PROJECT), "/mfq-src"),
                   (context.environment.get("USERPROFILE", ""), "/mfq-build/home")]
        configured = context.environment.get("CARGO_TARGET_DIR")
        if configured:
            sources.append((str(context.cargo_target), "/mfq-build/target"))
        return sources
    home = context.environment.get("HOME", "")
    return [
        (home, "/mfq-build/home"),
        (run([str(context.in_venv("python")), "-c", "import sys; print(sys.base_prefix)"],
             env=context.environment, capture=True).strip(), "/mfq-build/python"),
        (str(context.venv), "/mfq-build/venv"),
        (context.environment.get("CARGO_HOME") or os.path.join(home, ".cargo"), "/mfq-build/cargo"),
        (context.environment.get("RUSTUP_HOME") or os.path.join(home, ".rustup"), "/mfq-build/rustup"),
        (str(context.cargo_target), "/mfq-build/target"),
        (str(PROJECT), "/mfq-src"),
    ]


def cuda_host_option(flag: str) -> str:
    # NVCC parses a comma-separated list, then forwards to the host compiler.
    # Preserve an inner pair of quotes for that second command-line parse.
    if "," in flag:
        fail("CUDA release path mappings cannot contain commas")
    return "--compiler-options=" + subprocess.list2cmdline([f'"{flag}"'])


def configure_release_paths(context: Context) -> None:
    environment = context.environment
    native: list[str] = []
    rust: list[str] = []
    if context.windows:
        native.append("/experimental:deterministic")
        for source, destination in release_path_sources(context):
            if not source:
                continue
            spellings = [source]
            if short := short_path(source):
                spellings.append(short)
            for path in spellings:
                for spelling in (path, path.replace("\\", "/")):
                    native.append(f"/pathmap:{spelling}={destination}")
                    rust.append(f"--remap-path-prefix={spelling}={destination}")
        rust += ["-C", "link-arg=/PDBALTPATH:tyloinfer.pdb"]
    else:
        for source, destination in release_path_sources(context):
            if not source:
                continue
            resolved = str(Path(source).resolve())
            if resolved == os.sep:
                fail("refusing to remap the filesystem root")
            for spelling in (source.rstrip("/\\"), resolved):
                native.append(f"-ffile-prefix-map={shlex.quote(spelling)}={destination}")
                rust.append(f"--remap-path-prefix={spelling}={destination}")
                if spelling == resolved:
                    break

    inherited = environment.get("CARGO_ENCODED_RUSTFLAGS")
    rust = (inherited.split("\x1f") if inherited else
            environment.get("RUSTFLAGS", "").split()) + rust
    environment["CARGO_ENCODED_RUSTFLAGS"] = "\x1f".join(rust)
    if context.windows:
        # Serialize the same structured options for NVCC independently. Arbitrary
        # user CXXFLAGS belong to C++; CUDA host overrides belong in CUDAFLAGS.
        environment["CUDAFLAGS"] = " ".join([
            environment.get("CUDAFLAGS", ""), *(cuda_host_option(flag) for flag in native),
        ]).strip()
        native = [f'"{flag}"' for flag in native]
    # OBJCXXFLAGS falls back to CXXFLAGS before either one gains release flags.
    objective = environment.get("OBJCXXFLAGS") or environment.get("CXXFLAGS", "")
    for name, base in (("CFLAGS", environment.get("CFLAGS", "")),
                       ("CXXFLAGS", environment.get("CXXFLAGS", "")),
                       ("OBJCXXFLAGS", objective)):
        environment[name] = " ".join([base, *native]).strip()
    environment["CC_SHELL_ESCAPED_FLAGS"] = "1"


def reject_in_use_environment(context: Context) -> None:
    """Never wipe a release environment another build is still holding open."""
    python = context.in_venv("python")
    if not python.is_file():
        return
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                           wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    create_file.restype = wintypes.HANDLE
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    close_handle.restype = wintypes.BOOL
    try:
        handle = create_file(
            str(python), 0x40000000, 0, None, 3, 0x80, None)  # open existing, no sharing
    except OSError:
        fail(f"release Python environment is in use or not writable: {context.venv}")
    if handle in (ctypes.c_void_p(-1).value, None):
        error = ctypes.get_last_error()
        fail(f"release Python environment is in use or not writable: {context.venv} "
             f"(Win32 error {error})")
    close_handle(handle)


# --- release environment --------------------------------------------------

def create_release_environment(context: Context) -> None:
    environment = context.environment
    environment["VIRTUAL_ENV"] = str(context.venv)
    add_to_path(environment, str(context.venv / context.bin_directory))
    if environment.get("TYLOINFER_RELEASE_REUSE_VENV") == "1":
        for name in ("python", "pyinstaller"):
            if not os.access(context.in_venv(name), os.X_OK if not context.windows else os.R_OK):
                fail(f"reused release environment has no {name}")
        return
    if context.windows:
        reject_in_use_environment(context)
    run(["uv", "venv", "--clear", "--python",
         environment.get("TYLOINFER_RELEASE_PYTHON", "3.12"), context.venv],
        cwd=PROJECT, env=environment)
    run(["uv", "sync", "--project", PYTHON_PROJECT, "--locked", "--active",
         "--no-default-groups", "--no-editable", "--group", "release",
         *context.uv_extras], cwd=PROJECT, env=environment)
    for name in ("python", "pyinstaller"):
        if not context.in_venv(name).is_file():
            fail(f"uv did not install {name} into the release environment")


def stage_licenses(context: Context) -> None:
    directory = context.resources / "licenses"
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(PROJECT / "LICENSE", directory / "TyloInfer-Apache-2.0.txt")
    shutil.copyfile(PROJECT / "NOTICE", directory / "NOTICE.txt")
    for notice in sorted((PROJECT / "LICENSES").glob("*.txt")):
        shutil.copyfile(notice, directory / notice.name)


# --- native sidecar -------------------------------------------------------

def mlx_root(context: Context) -> Path:
    root = Path(run([str(context.in_venv("python")), "-c",
                     "import mlx; print(next(iter(mlx.__path__)))"],
                    env=context.environment, capture=True).strip())
    for name in ("lib/libmlx.dylib", "lib/libjaccl.dylib", "lib/mlx.metallib"):
        if not (root / name).is_file():
            fail(f"missing MLX runtime file: {root / name}")
    return root


def build_native(context: Context, root: Path | None = None) -> None:
    environment = context.environment
    defines = ["-DTYLOINFER_BUILD_RUNTIME_COMMUNICATION=ON",
               f"-DTYLOINFER_ENABLE_MFQ_COMPAT={environment.get('TYLOINFER_ENABLE_MFQ_COMPAT', 'OFF')}"]
    if context.windows:
        configure = ["-DCMAKE_C_FLAGS=" + environment["CFLAGS"],
                     "-DCMAKE_CXX_FLAGS=" + environment["CXXFLAGS"],
                     f"-DCMAKE_CUDA_FLAGS={environment.get('CUDAFLAGS', '')}",
                     f"-DCMAKE_CUDA_COMPILER={context.cuda_root / 'bin' / 'nvcc.exe'}",
                     f"-DCMAKE_CUDA_ARCHITECTURES={context.cuda_architectures}",
                     f"-DTYLOINFER_CUDA_ARCHITECTURES={context.cuda_architectures}",
                     "-DTYLOINFER_BUILD_METAL_RUNTIME=OFF",
                     "-DTYLOINFER_BUILD_TORCH_REFERENCE_RUNTIME=OFF"]
        extra_build = ["--config", "Release"]
    else:
        configure = ["-DCMAKE_C_FLAGS=" + environment["CFLAGS"],
                     "-DCMAKE_CXX_FLAGS=" + environment["CXXFLAGS"],
                     "-DCMAKE_OBJCXX_FLAGS=" + environment["OBJCXXFLAGS"],
                     "-DCMAKE_OSX_ARCHITECTURES=arm64",
                     f"-DCMAKE_OSX_DEPLOYMENT_TARGET={context.deployment_target}",
                     "-DTYLOINFER_BUILD_METAL_RUNTIME=ON",
                     "-DTYLOINFER_BUILD_PRECOMPILED_METAL=ON",
                     f"-DTYLOINFER_MLX_ROOT={root}",
                     "-DTYLOINFER_MLX_METALLIB_DEFAULT:STRING="]
        extra_build = []
    run(["cmake", "-S", PROJECT / "csrc", "-B", context.native_build, "-G", "Ninja",
         "-DCMAKE_BUILD_TYPE=Release", *configure, *defines], env=environment)
    run(["cmake", "--build", context.native_build, "--target", context.native_target,
         *extra_build, "--parallel", context.jobs], env=environment)


def stage_native(context: Context) -> Path:
    """Ninja and the CUDA CMake target place the executable at the build root."""
    executable = context.native_build / f"{context.native_target}{context.executable}"
    if not executable.is_file():
        fail(f"native build did not create {executable}")
    shutil.copyfile(executable, context.sidecar)
    return executable


def copy_files(directory: Path, destination: Path, *patterns: str) -> None:
    if not directory.is_dir():
        return
    for pattern in patterns:
        for source in directory.glob(pattern):
            shutil.copyfile(source, destination / source.name)


def stage_windows_runtime(context: Context, native_output: Path) -> None:
    destination = context.windows_runtime
    remove_directory(destination)
    destination.mkdir(parents=True)
    copy_files(native_output.parent, destination, "*.dll")

    root = context.cuda_root
    runtime = next((directory for directory in (root / "bin" / "x64", root / "bin")
                    if any(directory.glob("cudart64_*.dll"))), None)
    if runtime is None:
        fail(f"the CUDA Toolkit runtime DLL directory is missing from {root}")
    for pattern in ("cudart64_*.dll", "cublas64_*.dll", "cublasLt64_*.dll"):
        copy_files(runtime, destination, pattern)
        if not any(destination.glob(pattern)):
            fail(f"the CUDA Toolkit did not provide the native runtime DLL matching {pattern}")

    redist = context.environment.get("VCTOOLSREDISTDIR")
    if not redist:
        fail("the Visual Studio environment did not provide VCToolsRedistDir")
    x64 = Path(redist) / "x64"
    if not x64.is_dir():
        fail(f"the Visual C++ x64 redistributable directory is missing from {redist}")
    crt = [path for path in x64.iterdir() if re.fullmatch(r"Microsoft\.VC\d+\.CRT", path.name)]
    if not crt:
        fail(f"the Visual C++ x64 CRT redistributable is missing from {x64}")
    copy_files(crt[0], destination, "*.dll")
    # CMake may enable OpenMP for mfq-runtime when the VS workload provides it.
    for directory in x64.iterdir():
        if re.fullmatch(r"Microsoft\.VC\d+\.OpenMP", directory.name):
            copy_files(directory, destination, "*.dll")
    if not any(destination.glob("*.dll")):
        fail("no native runtime DLLs were staged")


# --- macOS runtime --------------------------------------------------------

def self_test(sidecar: Path, metallib: Path, environment: dict[str, str],
              frameworks: Path | None = None) -> None:
    environment = {**environment, "TYLOINFER_MLX_METALLIB": str(metallib)}
    if frameworks is not None:
        environment["DYLD_LIBRARY_PATH"] = str(frameworks)
    run([sidecar, "--self-test-metal"], env=environment)


def stage_macos_runtime(context: Context, root: Path) -> None:
    environment = context.environment
    sidecar = context.sidecar
    video = context.native_build / "metal/libmfq_avfoundation_video.dylib"
    if not os.access(context.native_build / "metal/mfq-decode-metal", os.X_OK):
        fail("native build did not create mfq-decode-metal")
    if not video.is_file():
        fail("native build did not create the AVFoundation video library")
    shutil.copyfile(context.native_build / "metal/mfq-decode-metal", sidecar)
    sidecar.chmod(0o755)
    for name in ("libmlx.dylib", "libjaccl.dylib"):
        target = context.frameworks / name
        shutil.copyfile(root / "lib" / name, target)
        target.chmod(0o644)
    shutil.copyfile(video, context.frameworks / video.name)
    (context.frameworks / video.name).chmod(0o755)
    shutil.copyfile(root / "lib/mlx.metallib", context.resources / "mlx.metallib")
    manifest = context.native_build / "metal/lib/mfq-metal/manifest.json"
    if not manifest.is_file() or manifest.stat().st_size == 0:
        fail("native build did not create the precompiled MFQ Metal libraries")
    metal = context.resources / "mfq-metal"
    remove_directory(metal)
    shutil.copytree(context.native_build / "metal/lib/mfq-metal", metal)

    run(["install_name_tool", "-add_rpath", "@executable_path/../Frameworks", sidecar],
        env=environment)
    for signed in (context.frameworks / "libjaccl.dylib", context.frameworks / "libmlx.dylib",
                   context.frameworks / video.name, sidecar):
        run(["codesign", "--force", "--sign", context.signing_identity, signed], env=environment)
        run(["codesign", "--verify", "--strict", "--verbose=2", signed], env=environment)
    architectures = run(["lipo", "-archs", sidecar], env=environment, capture=True).split()
    if context.architecture not in architectures:
        fail(f"{sidecar} is not {context.architecture}")
    self_test(sidecar, context.resources / "mlx.metallib", environment, context.frameworks)


# --- frozen server, Studio, artifacts -----------------------------------

def build_server(context: Context) -> None:
    environment = context.environment
    pyinstaller = [context.in_venv("pyinstaller"),
                   "--noconfirm", "--clean", "--onedir", "--name", SERVER_NAME,
                   "--paths", PROJECT, *PYINSTALLER_ARGS, *context.pyinstaller_extras,
                   "--distpath", context.resources,
                   "--workpath", context.pyinstaller_build,
                   "--specpath", context.spec]
    if not context.windows and context.signing_identity != "-":
        pyinstaller += ["--codesign-identity", context.signing_identity]
    run(pyinstaller + [SERVER_ENTRY], env=environment)

    if not context.server.is_file():
        fail(f"PyInstaller did not create {context.server}")
    if not context.windows:
        # PyInstaller hoists libmlx.dylib into _internal, but leaves mlx.metallib
        # under mlx/lib/. libmlx loads its default metallib from its own directory;
        # without this move the server traps right after "Application startup complete".
        internal = context.resources / SERVER_NAME / "_internal"
        nested = internal / "mlx" / "lib" / "mlx.metallib"
        if not nested.is_file():
            fail("PyInstaller did not collect the MLX default metallib")
        nested.replace(internal / "mlx.metallib")
        run(["codesign", "--verify", "--strict", "--verbose=2", context.server], env=environment)
    run([context.server, "--version"], env=environment)
    print(f"built {context.server}", flush=True)


# --- Studio ---------------------------------------------------------------


def tauri_environment(context: Context) -> dict[str, str]:
    environment = {**context.environment, "CI": "true"}
    if not context.windows:
        environment["APPLE_SIGNING_IDENTITY"] = context.signing_identity
    return environment


def build_studio(context: Context) -> None:
    environment = tauri_environment(context)
    command = ["run", "tauri:studio", "build",
               "--config", f"src-tauri/tauri.release-{'windows' if context.windows else 'macos'}.conf.json"]
    if not context.windows and context.signing_identity != "-":
        command += ["--config", '{"bundle":{"macOS":{"hardenedRuntime":true}}}']
    # The app/ workspace is pnpm-managed (app/pnpm-lock.yaml); npm ci cannot install from it.
    pnpm = ["pnpm.cmd" if context.windows else "pnpm", "--dir", str(APP)]
    run(pnpm + ["install", "--frozen-lockfile"], env=environment)
    run(pnpm + ["run", "check:studio"], env=environment)
    run(pnpm + command + ["--target", context.triple], env=environment)
    if context.windows:
        # The PyInstaller workpath is rebuilt by --clean. Reclaim it before Tauri
        # copies the multi-GiB sidecars and runtime resources into NSIS staging.
        remove_directory(context.pyinstaller_build)


def build_agent(context: Context) -> None:
    # The existing Tauri hook stages Node, DSH, the plugin, licenses and signatures.
    if context.clean:
        remove_directory(context.cargo_target)
        remove_directory(TAURI / "agent/runtime")
    environment = tauri_environment(context)
    platform_config = TAURI / f"tauri.release-{'windows' if context.windows else 'macos'}.conf.json"
    core_bundle = json.loads(platform_config.read_text())
    del core_bundle["bundle"]["resources"]["../web/dist"]
    if not context.windows and context.signing_identity != "-":
        core_bundle["bundle"]["macOS"]["hardenedRuntime"] = True
    if context.windows:
        # The hook prepares this archive after validating the deployed Node graph.
        environment["TYLOINFER_STAGE_AGENT_ARCHIVE"] = "1"
        core_bundle["bundle"]["resources"] = {
            "agent/runtime/": None,
            "../web/dist": None,
            "../../packaging/build/agent-payload.zip": "agent-payload.zip",
        }
        core_bundle["bundle"]["externalBin"] = []
        core_bundle["bundle"]["windows"]["nsis"].update({
            "compression": "none",
            "installerHooks": "../../packaging/windows_agent_hooks.nsh",
        })
    pnpm = ["pnpm.cmd" if context.windows else "pnpm", "--dir", str(APP)]
    run(pnpm + ["install", "--frozen-lockfile"], env=environment)
    run(pnpm + ["exec", "tauri", "build", "--config", "src-tauri/tauri.agent.conf.json",
                "--config", json.dumps(core_bundle), "--target", context.triple],
        env=environment)


def select_artifact(directory: Path, pattern: str) -> Path:
    candidates = sorted(path for path in directory.glob(pattern) if path.is_file())
    if not candidates:
        fail(f"no build artifact matching {pattern} in {directory}")
    if len(candidates) != 1:
        fail(f"ambiguous build artifacts in {directory}; remove stale outputs: "
             + ", ".join(path.name for path in candidates))
    return candidates[0]


def collect_installer(context: Context) -> Path:
    context.output.mkdir(parents=True, exist_ok=True)
    bundle = context.cargo_target / context.triple / "release/bundle"
    source = select_artifact(bundle / ("nsis" if context.windows else "dmg"),
                             "*.exe" if context.windows else "*.dmg")
    release = context.output / source.name
    shutil.copyfile(source, release)
    if context.windows:
        print(f"copied {release}", flush=True)
        return release
    run(["hdiutil", "verify", release], env=context.environment)
    mount = Path(tempfile.mkdtemp(prefix="mfq-release-verify."))
    try:
        run(["hdiutil", "attach", "-readonly", "-nobrowse", "-mountpoint", mount, release],
            env=context.environment, capture=True)
        try:
            app = mount / ("TyloInfer Agent.app" if context.product == "agent" else "TyloInfer.app")
            run(["codesign", "--verify", "--deep", "--strict", "--verbose=2", app],
                env=context.environment)
            if context.product == "studio":
                self_test(app / "Contents/MacOS/mfq-decode-metal",
                          app / "Contents/Resources/mlx.metallib", context.environment)
                run([app / "Contents/Resources" / SERVER_NAME / SERVER_NAME, "--version"],
                    env=context.environment, capture=True)
        finally:
            run(["hdiutil", "detach", mount], env=context.environment, capture=True)
    finally:
        remove_directory(mount)
    return release


# --- Windows single-file app ---------------------------------------------

def portable_payload(project: Path, executable: Path) -> dict[str, Path]:
    tauri = project / "app" / "src-tauri"
    bundle = json.loads((tauri / "tauri.release-windows.conf.json").read_text())["bundle"]
    files: dict[str, Path] = {}
    names: set[str] = set()

    def add(source: Path, destination: str) -> None:
        path = PurePosixPath(destination.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts or ":" in str(path):
            raise ValueError(f"Unsafe archive path: {destination}")
        # Compare whole paths: resources mapped to the archive root have a bare
        # file name, which would otherwise collide with same-named files deeper
        # in the tree (windows-runtime/VCRUNTIME140.dll vs _internal/...).
        key = str(path).casefold()
        if key in names:
            raise ValueError(f"Duplicate archive path: {path}")
        if not source.is_file():
            raise FileNotFoundError(source)
        files[str(path)] = source
        names.add(key)

    add(executable, "tyloinfer.exe")
    for source, destination in bundle["resources"].items():
        root = (tauri / source).resolve()
        if root.is_dir():
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    add(path, str(PurePosixPath(destination) / path.relative_to(root).as_posix()))
        else:
            add(root, destination)
    for source in bundle["externalBin"]:
        add((tauri / f"{source}-x86_64-pc-windows-msvc.exe").resolve(),
            Path(source).name + ".exe")
    for required in ("mfq-runtime.exe", "tyloinfer-server/tyloinfer-server.exe", "web-ui/index.html"):
        if required not in files:
            raise FileNotFoundError(f"Missing portable resource: {required}")
    return files


def nsis_string(value: str) -> str:
    return value.replace("$", "$$").replace('"', '$\\"').replace("\r", "$\\r").replace("\n", "$\\n")


def build_single_file_app(project: Path, executable: Path, output: Path,
                          makensis: Path, *, environment: dict[str, str] | None = None) -> Path:
    files = portable_payload(project, executable)
    tauri = project / "app" / "src-tauri"
    packaging = project / "packaging"
    version = json.loads((tauri / "tauri.conf.json").read_text())["version"]
    digest = hashlib.sha256()
    instructions = []
    directory = None
    for name, path in sorted(files.items()):
        digest.update(name.encode() + b"\0")
        with path.open("rb") as stream:
            digest.update(hashlib.file_digest(stream, "sha256").digest())
        target = PurePosixPath(name)
        parent = str(target.parent).replace("/", "\\")
        if parent != directory:
            instructions.append(f'  SetOutPath "$Stage\\{nsis_string(parent)}"')
            directory = parent
        instructions += [f'  File "/oname={nsis_string(target.name)}" "{nsis_string(str(path.resolve()))}"',
                         "  IfErrors failed"]
    output.mkdir(parents=True, exist_ok=True)
    destination = output / f"TyloInfer_{version}_x64.exe"
    partial = destination.with_suffix(".exe.partial")
    script = packaging / "build/standalone/launcher.nsi"
    script.parent.mkdir(parents=True, exist_ok=True)
    # The Studio updater re-launches this launcher with /WAITPID= and reads
    # TYLOINFER_STUDIO_LAUNCHER, so packaging/windows_standalone.nsi is required.
    template = (packaging / "windows_standalone.nsi").read_text(encoding="utf-8")
    for key, value in {
        "OUTPUT": nsis_string(str(partial.resolve())), "VERSION": version,
        "ICON": nsis_string(str(tauri / "icons/icon.ico")),
        "PAYLOAD_ID": f"{version}-{digest.hexdigest()[:16]}",
        "PAYLOAD_FILES": "\n".join(instructions),
    }.items():
        template = template.replace(f"@{key}@", value)
    script.write_text(template, encoding="utf-8")
    print(f"Packing {len(files)} files into a standalone executable", flush=True)
    try:
        run([makensis, "-INPUTCHARSET", "UTF8", "-V2", script], env=environment)
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)
    with destination.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    destination.with_suffix(".exe.sha256").write_text(f"{checksum}  {destination.name}\n",
                                                     encoding="utf-8")
    print(f"Built {destination} ({destination.stat().st_size / 1024**2:.1f} MiB)", flush=True)
    archive = output / f"TyloInfer_{version}_x64-portable.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as package:
        package.write(destination, "TyloInfer.exe")
    with archive.open("rb") as stream:
        zip_checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    archive.with_suffix(".zip.sha256").write_text(f"{zip_checksum}  {archive.name}\n",
                                                  encoding="utf-8")
    print(f"Built {archive} (one executable)", flush=True)
    return destination


def build_portable(context: Context) -> None:
    executable = context.cargo_target / context.triple / "release/tyloinfer.exe"
    makensis = context.cargo_target / ".tauri/NSIS/makensis.exe"
    if not makensis.is_file():
        found = shutil.which("makensis", path=context.environment.get("PATH", ""))
        if not found:
            fail("makensis is required to build the single-file app")
        makensis = Path(found)
    build_single_file_app(PROJECT, executable, context.output, makensis,
                          environment=context.environment)


# --- orchestration --------------------------------------------------------

def build_release(context: Context) -> None:
    preflight(context)
    if context.clean:
        clean(context)
    initialize_staging(context)
    create_release_environment(context)
    configure_release_paths(context)
    stage_licenses(context)
    if context.windows:
        build_native(context)
        native_output = stage_native(context)
        stage_windows_runtime(context, native_output)
    else:
        root = mlx_root(context)
        build_native(context, root)
        stage_macos_runtime(context, root)
    build_server(context)
    if context.product == "agent":
        build_agent(context)
    else:
        build_studio(context)
    print(f"built {collect_installer(context)}", flush=True)
    if context.windows and context.product == "studio":
        build_portable(context)


def new_context(target: str, clean_requested: bool, product: str = "studio", *,
                environment: dict[str, str] | None = None) -> Context:
    windows = target == "windows"
    # The caller prepares the toolchain; keep all build-local changes private.
    environment = dict(os.environ if environment is None else environment)
    if os.name == "nt":
        environment = {name.upper(): value for name, value in environment.items()}
    output = Path(environment.get("TYLOINFER_RELEASE_OUTPUT_DIR") or SCRIPT_DIR / "dist")
    if product == "agent":
        # Separate Cargo output prevents Studio and Agent bundles from mixing.
        cargo_target = TAURI / environment.get("CARGO_TARGET_DIR", "target")
        environment["CARGO_TARGET_DIR"] = str((cargo_target / "agent").resolve())
    return Context(
        windows=windows,
        triple="x86_64-pc-windows-msvc" if windows else "aarch64-apple-darwin",
        architecture="amd64" if windows else "arm64",
        product=product,
        clean=clean_requested or (product == "studio" and not windows),
        venv=Path(environment.get("TYLOINFER_RELEASE_VENV") or SCRIPT_DIR / ".venv"),
        output=output,
        build=SCRIPT_DIR / "build",
        native_build=SCRIPT_DIR / "build/runtime",
        pyinstaller_build=SCRIPT_DIR / "build/pyinstaller",
        spec=SCRIPT_DIR / "build/spec",
        sidecars=SCRIPT_DIR / "sidecars",
        resources=SCRIPT_DIR / "Resources",
        # Published artifacts are not staging files; keep both products on clean builds.
        stage_directories=("build", "sidecars", "Resources", "windows-runtime") if windows
        else ("build", "sidecars", "Frameworks", "Resources"),
        native_target="mfq-runtime" if windows else "mfq-decode-metal",
        jobs=windows_jobs(environment) if windows else (
            environment.get("TYLOINFER_RELEASE_JOBS") or str(os.cpu_count() or 1)),
        uv_extras=() if windows else ("--extra", "metal"),
        pyinstaller_extras=([
            *_flags("--collect-all", ("tokenizers",)),
            # No MLX runtime on Windows, so transformers and safetensors stay out.
            *_flags("--exclude-module", ("transformers", "safetensors")),
        ] if windows else [
            "--target-architecture", "arm64",
            # mlx_lm pulls in transformers, which the MLX-free target omits.
            *_flags("--collect-all", ("mlx", "mlx_lm", "transformers", "socksio")),
            "--copy-metadata", "requests",
        ]),
        deployment_target=environment.get("TYLOINFER_RELEASE_MACOS_DEPLOYMENT_TARGET", "26.2"),
        signing_identity=environment.get("TYLOINFER_RELEASE_SIGNING_IDENTITY")
        or environment.get("APPLE_SIGNING_IDENTITY") or "-",
        cuda_architectures=environment.get("TYLOINFER_RELEASE_CUDA_ARCHITECTURES")
        or "80-real;86-real;89",
        environment=environment,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", choices=("studio", "agent"), default="studio",
                        help="desktop app to package (default: studio)")
    parser.add_argument("--target", choices=("macos", "windows"),
                        default={"darwin": "macos", "win32": "windows"}.get(sys.platform),
                        required=sys.platform not in ("darwin", "win32"),
                        help="target platform (defaults to the current macOS/Windows host)")
    parser.add_argument("--clean", action="store_true",
                        help="rebuild Core staging and, for Agent, Cargo/DSH staging from scratch "
                             "(default for Studio on macOS; keeps published artifacts)")
    parser.add_argument("--stage-agent-archive", action="store_true", help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.stage_agent_archive:
        if arguments.app != "agent" or arguments.target != "windows":
            parser.error("--stage-agent-archive requires Windows Agent")
        build_agent_payload(PROJECT, SCRIPT_DIR / "build/agent-payload.zip")
        return 0
    build_release(new_context(arguments.target, arguments.clean, arguments.app))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
