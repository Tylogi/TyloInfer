"""The HIP compatibility shim compiled on a plain C++ toolchain.

Structure gate, not a behavior test: it compiles the hip kernel headers and the
csrc accelerator selection. See conftest.py.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]




def cmake_path():
    path = shutil.which("cmake") or str(Path(sys.executable).with_name("cmake"))
    if not Path(path).is_file():
        pytest.skip("CMake is unavailable")
    return path




@pytest.mark.parametrize("cuda,hip,flags,expected", [
    (True, True, [], "ON;OFF"),
    (True, False, [], "ON;OFF"),
    (False, True, [], "OFF;ON"),
    (False, False, [], "OFF;OFF"),
    (True, True, ["-DTYLOINFER_BUILD_HIP_RUNTIME=ON"], "OFF;ON"),
    (True, True, ["-DTYLOINFER_BUILD_CUDA_RUNTIME=OFF"], "OFF;ON"),
    (True, True, ["-DTYLOINFER_BUILD_CUDA_RUNTIME=ON", "-DTYLOINFER_BUILD_HIP_RUNTIME=OFF"], "ON;OFF"),
    (True, True, ["-DTYLOINFER_BUILD_CUDA_RUNTIME=ON", "-DTYLOINFER_BUILD_HIP_RUNTIME=ON"], None),
])
def test_cmake_accelerator_selection(tmp_path, cuda, hip, flags, expected):
    source = (ROOT / "csrc/CMakeLists.txt").read_text()
    options = source.split("include(CheckLanguage)\n", 1)[1].split("\nif(APPLE)\n", 1)[0]
    options = options.replace("check_language(CUDA)", "").replace("check_language(HIP)", "")
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.26)\nproject(backend_options NONE)\nset(APPLE FALSE)\n"
        f"set(CMAKE_CUDA_COMPILER {'mock' if cuda else 'NOTFOUND'})\n"
        f"set(CMAKE_HIP_COMPILER {'mock' if hip else 'NOTFOUND'})\n" + options +
        '\nfile(WRITE "${CMAKE_BINARY_DIR}/selected.txt" "${TYLOINFER_BUILD_CUDA_RUNTIME};${TYLOINFER_BUILD_HIP_RUNTIME}")\n'
    )
    output = tmp_path / "build"
    command = [cmake_path(), "-S", str(tmp_path), "-B", str(output)]
    environment = dict(os.environ)
    # A Python-installed CMake launcher may use another interpreter.
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    result = subprocess.run([*command, *flags], capture_output=True, text=True, env=environment)
    if expected is None:
        assert result.returncode != 0
        assert "Select either TYLOINFER_BUILD_CUDA_RUNTIME or TYLOINFER_BUILD_HIP_RUNTIME" in result.stderr
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert (output / "selected.txt").read_text() == expected
        if flags == ["-DTYLOINFER_BUILD_HIP_RUNTIME=ON"]:
            result = subprocess.run([*command, "-DTYLOINFER_BUILD_CUDA_RUNTIME=ON", "-DTYLOINFER_BUILD_HIP_RUNTIME=OFF"],
                                    capture_output=True, text=True, env=environment)
            assert result.returncode == 0, result.stdout + result.stderr
            assert (output / "selected.txt").read_text() == "ON;OFF"




def test_wave_intrinsics_preserve_subwarps_and_both_half_waves(tmp_path):
    compiler = shutil.which("c++") or shutil.which("clang++")
    if not compiler:
        pytest.skip("C++ compiler is unavailable")
    include = tmp_path / "hip"
    include.mkdir()
    (include / "hip_bf16.h").write_text("")
    (include / "hip_runtime.h").write_text(r"""
#pragma once
#define __device__
#define __forceinline__ inline
struct Dim { unsigned x, y, z; };
inline Dim threadIdx{}, blockDim{8, 8, 4};
inline unsigned warpSize = 64, current_lane;
inline unsigned long long votes;
template<typename T> T __shfl(T v, int s, int w) { return v - current_lane % w + s; }
template<typename T> T __shfl_down(T v, unsigned d, int w) { return current_lane % w + d < unsigned(w) ? v + d : v; }
template<typename T> T __shfl_up(T v, unsigned d, int w) { return current_lane % w >= d ? v - d : v; }
template<typename T> T __shfl_xor(T v, int s, int w) { return v - current_lane % w + ((current_lane % w) ^ s); }
inline unsigned long long __ballot(bool) { return votes; }
""")
    source = tmp_path / "wave.cpp"
    source.write_text(r"""
#include <cassert>
#define __HIP_PLATFORM_AMD__ 1
#include "mfq_hip_compat.h"
int main() {
    assert(__dp4a(int(0x80808080u), int(0x80808080u), 0) == 65536);
    assert(__dp4a(int(0xfffe0180u), int(0x027f80ffu), 50) == -206);
    assert(__dp4a(0x7f7f7f7f, 0x7f7f7f7f, 2147483647) == -2147419133);
    assert(__dp4a(int(0x80808080u), 0x7f7f7f7f, -2147483647 - 1) == 2147418624);
    for (unsigned wave : {32u, 64u}) {
        warpSize = wave;
        for (unsigned thread = 0; thread < 256; ++thread) {
            threadIdx = {thread % 8, thread / 8 % 8, thread / 64};
            current_lane = thread % wave;
            const int v = int(thread);
            assert(__shfl_sync(~0u, v, 1) == int(thread / 32 * 32 + 1));
            assert(__shfl_sync(~0u, float(v), 3, 16) == float(thread / 16 * 16 + 3));
            for (int width : {8, 16, 32}) {
                assert(__shfl_sync(~0u, v, 1, width) == int(thread / width * width + 1));
                assert(__shfl_down_sync(~0u, v, 1, width) == (thread % width + 1 < unsigned(width) ? v + 1 : v));
                assert(__shfl_up_sync(~0u, v, 1, width) == (thread % width ? v - 1 : v));
                assert(__shfl_xor_sync(~0u, v, 1, width) == (v ^ 1));
            }
            for (int pattern = 0; pattern < 4; ++pattern) {
                votes = 0;
                unsigned expected = 0;
                for (unsigned lane = 0; lane < wave; ++lane) {
                    unsigned id = thread / wave * wave + lane;
                    bool p = pattern == 0 ? id % 64 < 48 : pattern == 1 ? id % 64 >= 32 :
                        pattern == 2 ? id % 3 == 0 : id % 64 < 17;
                    if (p) votes |= 1ull << lane;
                    if (p && id / 32 == thread / 32) expected |= 1u << (id % 32);
                }
                assert(__ballot_sync(~0u, true) == expected);
                assert(__ballot_sync(0x55555555u, true) == (expected & 0x55555555u));
            }
        }
    }
}
""")
    executable = tmp_path / "wave"
    result = subprocess.run([compiler, "-std=c++17", "-include", "initializer_list",
                             "-I", str(tmp_path), "-I", str(ROOT / "studio/server/kernels/hip"),
                             str(source), "-o", str(executable)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    subprocess.run([str(executable)], check=True)




def test_host_translation_units_do_not_require_device_builtins(tmp_path):
    compiler = shutil.which("c++") or shutil.which("clang++")
    if not compiler:
        pytest.skip("C++ compiler is unavailable")
    include = tmp_path / "hip"
    include.mkdir()
    (include / "hip_bf16.h").write_text("")
    (include / "hip_runtime.h").write_text("#define __device__\n#define __forceinline__ inline\n")
    source = tmp_path / "host.cpp"
    source.write_text('#define __HIP_PLATFORM_AMD__ 1\n#include "mfq_hip_compat.h"\n'
                      'int main() { return 0; }\n')
    executable = tmp_path / "host"
    result = subprocess.run([compiler, "-std=c++17", "-I", str(tmp_path),
                             "-I", str(ROOT / "studio/server/kernels/hip"), str(source), "-o", str(executable)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    subprocess.run([str(executable)], check=True)
