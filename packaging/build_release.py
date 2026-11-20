#!/usr/bin/env python3
# Build the self-contained TyloInfer desktop release for one platform.
#
#   uv run --no-project packaging/build_release.py --target macos
#   uv run --no-project packaging/build_release.py --target windows
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

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT = SCRIPT_DIR.parent
STUDIO = PROJECT / "studio" / "web"
SERVER_PROJECT = PROJECT / "studio" / "server"
SERVER_NAME = "tyloinfer-server"


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


def run(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None,
        quiet: bool = False) -> str:
    result = subprocess.run([str(item) for item in command], cwd=cwd, env=env,
                            capture_output=quiet, text=True)
    if result.returncode != 0:
        fail(f"command failed ({result.returncode}): {' '.join(str(item) for item in command)}")
    return result.stdout if quiet else ""


def _flags(flag: str, values: tuple[str, ...]) -> list[str]:
    return [item for value in values for item in (flag, value)]


# Shared by both targets. Platform-only entries live in new_context(): the macOS
# server bundles the MLX runtime (mlx_lm pulls in transformers), the Windows
# server has no MLX and leaves transformers and safetensors out.
PYINSTALLER_ARGS = [
    "--collect-data", "server",
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
    deployment_target: str = "26.2"
    signing_identity: str = "-"
    cuda_architectures: str = ""
    cuda_root: Path | None = None
    native_output: Path | None = None
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
            return STUDIO / "src-tauri" / "target"
        path = Path(configured)
        return path if path.is_absolute() else STUDIO / path

    @property
    def frameworks(self) -> Path:
        return SCRIPT_DIR / "Frameworks"

    @property
    def windows_runtime(self) -> Path:
        return SCRIPT_DIR / "windows-runtime"


# --- preflight ------------------------------------------------------------

def add_to_path(environment: dict[str, str], directory: str) -> None:
    if directory and os.path.isdir(directory):
        parts = environment.get("PATH", "").split(os.pathsep)
        if directory not in parts:
            environment["PATH"] = directory + os.pathsep + environment.get("PATH", "")


def require_commands(environment: dict[str, str], names: tuple[str, ...]) -> None:
    for name in names:
        if shutil.which(name, path=environment.get("PATH")) is None:
            fail(f"{name} is required; install it and make it available to this script")


def import_visual_studio(environment: dict[str, str]) -> None:
    program_files_x86 = environment.get("ProgramFiles(x86)") or environment.get("ProgramFiles", "")
    vswhere = Path(program_files_x86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not vswhere.is_file():
        fail("Visual Studio Build Tools with the C++ workload are required")

    def versus(*arguments: str) -> str:
        return run([str(vswhere), *arguments], quiet=True).strip()

    installation = versus("-latest", "-products", "*",
                          "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                          "-property", "installationPath").splitlines()[:1]
    if not installation:
        installation = [directory for directory in
                        versus("-all", "-products", "*", "-property", "installationPath").splitlines()
                        if (Path(directory) / "Common7" / "Tools" / "VsDevCmd.bat").is_file()]
    if not installation:
        fail("Visual Studio C++ x64 tools are required")

    development = Path(installation[0].strip()) / "Common7" / "Tools" / "VsDevCmd.bat"
    if not development.is_file():
        fail(f"VsDevCmd.bat is missing from {installation[0]}")
    # shortcut: the console codepage is assumed, which is fine for build-host
    # paths that are plain ASCII; switch to an explicit encoding if not.
    lines = run(["cmd.exe", "/d", "/s", "/c",
                 f'call "{development}" -no_logo -arch=x64 -host_arch=x64 >nul && set'],
                quiet=True)
    for line in lines.splitlines():
        name, separator, value = line.partition("=")
        if separator and name:
            environment[name] = value


def find_cuda_root(environment: dict[str, str]) -> Path:
    configured = environment.get("CUDA_PATH")
    if configured and (Path(configured) / "bin" / "nvcc.exe").is_file():
        return Path(configured)
    parent = Path(environment.get("ProgramFiles", "")) / "NVIDIA GPU Computing Toolkit" / "CUDA"
    if parent.is_dir():
        for candidate in sorted((path for path in parent.iterdir() if path.is_dir()),
                                key=lambda path: path.name, reverse=True):
            if (candidate / "bin" / "nvcc.exe").is_file():
                return candidate
    fail("CUDA Toolkit with nvcc is required")


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
    if not context.windows:
        if sys.platform != "darwin" or platform.machine() != "arm64":
            fail("this release target requires native Apple Silicon macOS")
        require_commands(environment, ("uv", "cmake", "ninja", "cargo", "npm", "codesign",
                                      "hdiutil", "install_name_tool", "lipo", "xcrun"))
        for shader in ("metal", "metallib"):
            if subprocess.run(["xcrun", "--find", shader], capture_output=True).returncode != 0:
                fail("Xcode Metal Toolchain is required to build release shaders")
        return

    if sys.platform != "win32" or sys.maxsize <= 2 ** 31:
        fail("this release target must be built on 64-bit Windows")
    program_files = environment.get("ProgramFiles", "")
    program_files_x86 = environment.get("ProgramFiles(x86)") or program_files
    add_to_path(environment, str(Path(program_files) / "CMake" / "bin"))
    add_to_path(environment, str(Path(environment.get("USERPROFILE", "")) / ".cargo" / "bin"))
    add_to_path(environment, str(Path(program_files_x86) / "NSIS"))
    import_visual_studio(environment)
    context.cuda_root = find_cuda_root(environment)
    environment["CUDA_PATH"] = str(context.cuda_root)
    add_to_path(environment, str(context.cuda_root / "bin"))
    add_to_path(environment, str(context.cuda_root / "lib" / "x64"))
    require_commands(environment, ("uv", "cmake", "ninja", "cargo", "npm.cmd", "cl", "nvcc"))


def clean(context: Context) -> None:
    for name in context.stage_directories:
        shutil.rmtree(SCRIPT_DIR / name, ignore_errors=True)
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
             env=context.environment, quiet=True).strip(), "/mfq-build/python"),
        (str(context.venv), "/mfq-build/venv"),
        (context.environment.get("CARGO_HOME") or os.path.join(home, ".cargo"), "/mfq-build/cargo"),
        (context.environment.get("RUSTUP_HOME") or os.path.join(home, ".rustup"), "/mfq-build/rustup"),
        (str(context.cargo_target), "/mfq-build/target"),
        (str(PROJECT), "/mfq-src"),
    ]


def configure_release_paths(context: Context) -> None:
    environment = context.environment
    native: list[str] = []
    rust: list[str] = []
    if context.windows:
        native.append("/experimental:deterministic")
        for source, destination in release_path_sources(context):
            spellings = [source]
            if short := short_path(source):
                spellings.append(short)
            for path in spellings:
                for spelling in (path, path.replace("\\", "/")):
                    # The MSVC flag stays inside one CMAKE_*_FLAGS string, so it
                    # carries its own quoting; cc-rs is told via the env var.
                    native.append(f'"/pathmap:{spelling}={destination}"')
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
    try:
        handle = ctypes.windll.kernel32.CreateFileW(
            str(python), 0x40000000, 0, None, 3, 0x80, None)  # open existing, no sharing
    except OSError:
        fail(f"release Python environment is in use or not writable: {context.venv}")
    if handle in (-1, None):
        error = ctypes.get_last_error()
        if error in (32, 5):  # sharing violation, access denied
            fail(f"release Python environment is in use or not writable: {context.venv}")
        return
    ctypes.windll.kernel32.CloseHandle(handle)


# --- release environment --------------------------------------------------

def create_release_environment(context: Context) -> None:
    environment = context.environment
    environment["VIRTUAL_ENV"] = str(context.venv)
    add_to_path(environment, str(context.venv / context.bin_directory))
    if os.environ.get("TYLOINFER_RELEASE_REUSE_VENV") == "1":
        for name in ("python", "pyinstaller"):
            if not os.access(context.in_venv(name), os.X_OK if not context.windows else os.R_OK):
                fail(f"reused release environment has no {name}")
        return
    if context.windows:
        reject_in_use_environment(context)
    run(["uv", "venv", "--clear", "--python",
         os.environ.get("TYLOINFER_RELEASE_PYTHON", "3.12"), context.venv],
        cwd=PROJECT, env=environment)
    run(["uv", "sync", "--project", SERVER_PROJECT, "--locked", "--active",
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
                    env=context.environment, quiet=True).strip())
    for name in ("lib/libmlx.dylib", "lib/libjaccl.dylib", "lib/mlx.metallib"):
        if not (root / name).is_file():
            fail(f"missing MLX runtime file: {root / name}")
    return root


def build_native(context: Context, root: Path | None = None) -> None:
    environment = context.environment
    defines = ["-DTYLOINFER_BUILD_RUNTIME_COMMUNICATION=ON",
               f"-DTYLOINFER_ENABLE_MFQ_COMPAT={os.environ.get('TYLOINFER_ENABLE_MFQ_COMPAT', 'OFF')}"]
    if context.windows:
        cuda_flags = " ".join(f"--compiler-options={flag}" for flag in
                              environment["CXXFLAGS"].split())
        configure = ["-DCMAKE_C_FLAGS=" + environment["CFLAGS"],
                     "-DCMAKE_CXX_FLAGS=" + environment["CXXFLAGS"],
                     f"-DCMAKE_CUDA_FLAGS={environment.get('CUDAFLAGS', '')} {cuda_flags}",
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


def copy_newest_native(context: Context) -> None:
    """The CUDA target lands in a generator-dependent subdirectory."""
    built = [path for path in context.native_build.rglob(f"{context.native_target}.exe")]
    if not built:
        fail("native build did not create mfq-runtime.exe")
    context.native_output = max(built, key=lambda path: path.stat().st_mtime)
    shutil.copyfile(context.native_output, context.sidecar)


def copy_files(directory: Path, destination: Path, *patterns: str) -> None:
    if not directory.is_dir():
        return
    for pattern in patterns:
        for source in directory.glob(pattern):
            shutil.copyfile(source, destination / source.name)


def stage_windows_runtime(context: Context) -> None:
    destination = context.windows_runtime
    shutil.rmtree(destination, ignore_errors=True)
    destination.mkdir(parents=True)
    copy_files(context.native_output, destination, "*.dll")

    root = context.cuda_root
    runtime = next((directory for directory in (root / "bin" / "x64", root / "bin")
                    if any(directory.glob("cudart64_*.dll"))), None)
    if runtime is None:
        fail(f"the CUDA Toolkit runtime DLL directory is missing from {root}")
    for pattern in ("cudart64_*.dll", "cublas64_*.dll", "cublasLt64_*.dll"):
        copy_files(runtime, destination, pattern)
        if not any(destination.glob(pattern)):
            fail(f"the CUDA Toolkit did not provide the native runtime DLL matching {pattern}")

    redist = context.environment.get("VCToolsRedistDir")
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
    shutil.copytree(context.native_build / "metal/lib/mfq-metal",
                    context.resources / "mfq-metal")

    run(["install_name_tool", "-add_rpath", "@executable_path/../Frameworks", sidecar],
        env=environment)
    for signed in (context.frameworks / "libjaccl.dylib", context.frameworks / "libmlx.dylib",
                   context.frameworks / video.name, sidecar):
        run(["codesign", "--force", "--sign", context.signing_identity, signed], env=environment)
        run(["codesign", "--verify", "--strict", "--verbose=2", signed], env=environment)
    architectures = run(["lipo", "-archs", sidecar], env=environment, quiet=True).split()
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
    run(pyinstaller + [SERVER_PROJECT / "serve.py"], env=environment)

    if not context.server.is_file():
        fail(f"PyInstaller did not create {context.server}")
    if not context.windows:
        run(["codesign", "--verify", "--strict", "--verbose=2", context.server], env=environment)
    run([context.server, "--version"], env=environment)
    print(f"built {context.server}", flush=True)


# --- Studio ---------------------------------------------------------------


def build_studio(context: Context) -> None:
    environment = {**context.environment, "CI": "true"}
    if not context.windows:
        environment["APPLE_SIGNING_IDENTITY"] = os.environ.get("APPLE_SIGNING_IDENTITY") or \
            context.signing_identity
    command = ["npm", "run", "tauri", "--", "build",
               "--config", f"src-tauri/tauri.release-{'windows' if context.windows else 'macos'}.conf.json"]
    if not context.windows and context.signing_identity != "-":
        command += ["--config", '{"bundle":{"macOS":{"hardenedRuntime":true}}}']
    npm = "npm.cmd" if context.windows else "npm"
    run([npm, "ci"], cwd=STUDIO, env=environment)
    run([npm, "run", "check"], cwd=STUDIO, env=environment)
    run(command + ["--target", context.triple], cwd=STUDIO, env=environment)
    if context.windows:
        # The PyInstaller workpath is rebuilt by --clean. Reclaim it before Tauri
        # copies the multi-GiB sidecars and runtime resources into NSIS staging.
        shutil.rmtree(context.pyinstaller_build, ignore_errors=True)


def collect_installer(context: Context) -> Path:
    bundle = context.cargo_target / context.triple / "release/bundle"
    if context.windows:
        directory = bundle / "nsis"
        created = sorted(directory.glob("*.exe"), key=lambda path: path.stat().st_mtime)
        if not created:
            fail("Tauri did not create an NSIS installer")
        release = context.output / created[-1].name
        shutil.copyfile(created[-1], release)
        print(f"copied {release}", flush=True)
        return release

    directory = bundle / "dmg"
    if not directory.is_dir():
        fail("Tauri did not create a DMG directory")
    created = sorted(directory.glob("*.dmg"))
    if not created:
        fail("Tauri did not create a DMG")
    release = context.output / created[0].name
    shutil.copyfile(created[0], release)
    run(["hdiutil", "verify", release])
    mount = Path(tempfile.mkdtemp(prefix="mfq-release-verify."))
    try:
        run(["hdiutil", "attach", "-readonly", "-nobrowse", "-mountpoint", mount, release],
            quiet=True)
        try:
            app = mount / "TyloInfer.app"
            run(["codesign", "--verify", "--deep", "--strict", "--verbose=2", app])
            self_test(app / "Contents/MacOS/mfq-decode-metal",
                      app / "Contents/Resources/mlx.metallib", context.environment)
            run([app / "Contents/Resources" / SERVER_NAME / SERVER_NAME, "--version"], quiet=True)
        finally:
            run(["hdiutil", "detach", mount], quiet=True)
    finally:
        shutil.rmtree(mount, ignore_errors=True)
    return release


# --- Windows single-file app ---------------------------------------------

def portable_payload(project: Path, executable: Path) -> dict[str, Path]:
    tauri = project / "studio" / "web" / "src-tauri"
    bundle = json.loads((tauri / "tauri.release-windows.conf.json").read_text())["bundle"]
    files: dict[str, Path] = {}
    names: set[str] = set()

    def add(source: Path, destination: str) -> None:
        path = PurePosixPath(destination.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts or ":" in str(path):
            raise ValueError(f"Unsafe archive path: {destination}")
        if path.name.casefold() in names:
            raise ValueError(f"Duplicate archive path: {path}")
        if not source.is_file():
            raise FileNotFoundError(source)
        files[str(path)] = source
        names.add(str(path).casefold())

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
                          makensis: Path) -> Path:
    files = portable_payload(project, executable)
    tauri = project / "studio" / "web" / "src-tauri"
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
        subprocess.run([str(makensis), "-INPUTCHARSET", "UTF8", "-V2", str(script)], check=True)
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
        found = shutil.which("makensis", path=context.environment.get("PATH"))
        if not found:
            fail("makensis is required to build the single-file app")
        makensis = Path(found)
    build_single_file_app(PROJECT, executable, context.output, makensis)


# --- orchestration --------------------------------------------------------

def build_release(context: Context) -> None:
    preflight(context)
    if context.clean:
        clean(context)
    create_release_environment(context)
    configure_release_paths(context)
    stage_licenses(context)
    if context.windows:
        build_native(context)
        copy_newest_native(context)
        stage_windows_runtime(context)
    else:
        root = mlx_root(context)
        build_native(context, root)
        stage_macos_runtime(context, root)
    build_server(context)
    build_studio(context)
    print(f"built {collect_installer(context)}", flush=True)
    if context.windows:
        build_portable(context)


def new_context(target: str, clean_requested: bool) -> Context:
    windows = target == "windows"
    environment = os.environ.copy()
    return Context(
        windows=windows,
        triple="x86_64-pc-windows-msvc" if windows else "aarch64-apple-darwin",
        architecture="amd64" if windows else "arm64",
        clean=clean_requested or not windows,
        venv=Path(os.environ.get("TYLOINFER_RELEASE_VENV") or SCRIPT_DIR / ".venv"),
        output=Path(os.environ.get("TYLOINFER_RELEASE_OUTPUT_DIR") or SCRIPT_DIR / "dist"),
        build=SCRIPT_DIR / "build",
        native_build=SCRIPT_DIR / "build/runtime",
        pyinstaller_build=SCRIPT_DIR / "build/pyinstaller",
        spec=SCRIPT_DIR / "build/spec",
        sidecars=SCRIPT_DIR / "sidecars",
        resources=SCRIPT_DIR / "Resources",
        stage_directories=("build", "sidecars", "Resources", "windows-runtime", "dist") if windows
        else ("build", "sidecars", "Frameworks", "Resources", "dist"),
        native_target="mfq-runtime" if windows else "mfq-decode-metal",
        jobs=windows_jobs(environment) if windows else (
            environment.get("TYLOINFER_RELEASE_JOBS") or str(os.cpu_count() or 1)),
        uv_extras=() if windows else ("--extra", "metal"),
        pyinstaller_extras=([
            *_flags("--collect-all", ("tokenizers",)),
            # No MLX runtime on Windows, so transformers and safetensors stay out.
            *_flags("--exclude-module", ("transformers", "safetensors", "server.runtime")),
        ] if windows else [
            "--target-architecture", "arm64",
            # mlx_lm pulls in transformers, which the MLX-free target omits.
            *_flags("--collect-all", ("mlx", "mlx_lm", "transformers", "socksio")),
            "--copy-metadata", "requests",
        ]),
        deployment_target=os.environ.get("TYLOINFER_RELEASE_MACOS_DEPLOYMENT_TARGET", "26.2"),
        signing_identity=os.environ.get("TYLOINFER_RELEASE_SIGNING_IDENTITY")
        or os.environ.get("APPLE_SIGNING_IDENTITY") or "-",
        cuda_architectures=os.environ.get("TYLOINFER_RELEASE_CUDA_ARCHITECTURES")
        or "80-real;86-real;89",
        environment=environment,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, choices=("macos", "windows"))
    parser.add_argument("--clean", action="store_true",
                        help="also rebuild the native sidecar from scratch (default on macOS)")
    arguments = parser.parse_args()
    build_release(new_context(arguments.target, arguments.clean))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
