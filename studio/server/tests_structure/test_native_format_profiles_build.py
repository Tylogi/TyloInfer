"""CMake wiring and include hygiene of the native format operators.

Structure gate, not a behavior test: it reads CMake files and native sources.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]




ROOT = Path(__file__).resolve().parents[3]




def test_ordinary_build_is_explicit_and_compatibility_is_opt_in():
    cmake = (ROOT / 'csrc/CMakeLists.txt').read_text()
    assert 'option(TYLOINFER_ENABLE_MFQ_COMPAT' in cmake
    assert '"Include legacy MFQ inference formats" OFF)' in cmake
    for backend in ('cpu', 'cuda', 'hip', 'metal'):
        source = (ROOT / f'csrc/backends/{backend}/CMakeLists.txt').read_text()
        assert 'if(TYLOINFER_ENABLE_MFQ_COMPAT)' in source
        assert 'ops/common/' in source
        assert 'ops/mfq/' in source




def test_moved_native_operators_keep_resolvable_relative_includes():
    for path in (ROOT / 'csrc/backends').rglob('*'):
        if path.suffix not in {'.h', '.cpp', '.cu', '.hip', '.mm'}:
            continue
        for include in re.findall(r'^#include\s+"(\.\.[^"]+)"', path.read_text(), re.M):
            assert (path.parent / include).is_file(), (path, include)
