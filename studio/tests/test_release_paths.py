"""Release checks cover native files and compressed Python payloads."""

import importlib.util
import marshal
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
import zipfile

import pytest


PACKAGING = Path(__file__).resolve().parents[2] / "packaging"
WINDOWS_BUILD = r"Z:\fixture-build\sample-profile"
POSIX_BUILD = "/fixture-build/sample-profile"


def load_packaging_module(name):
    spec = importlib.util.spec_from_file_location(name, PACKAGING / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


checker_module = load_packaging_module("check_release_paths")


@pytest.mark.parametrize("spelling", [WINDOWS_BUILD, WINDOWS_BUILD.replace("\\", "/"),
                                      WINDOWS_BUILD.replace("\\", "\\\\")])
@pytest.mark.parametrize("encoding", ["utf-8", "utf-16le"])
def test_windows_build_paths_are_detected(spelling, encoding):
    checker = checker_module.PathChecker([WINDOWS_BUILD])
    checker.check_bytes((spelling.upper() + "/source.cpp").encode(encoding), "program.exe")
    assert checker.failures == {"program.exe"}


def test_relative_and_remapped_paths_are_allowed():
    checker = checker_module.PathChecker([WINDOWS_BUILD, POSIX_BUILD])
    checker.check_bytes(b"/mfq-build/home/.cargo/src/crate.rs; studio/web/assets/index.js", "program.exe")
    assert not checker.failures


def test_compressed_zip_inspects_frozen_code(tmp_path):
    code = compile("pass", WINDOWS_BUILD + r"\stdlib\module.py", "exec")
    path = tmp_path / "base_library.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("module.pyc", b"\0" * 16 + marshal.dumps(code))
    checker = checker_module.PathChecker([WINDOWS_BUILD])
    checker.check_file(path)
    assert checker.failures == {"base_library.zip!module.pyc"}


def test_matches_across_chunks_are_detected(tmp_path):
    path = tmp_path / "program.exe"
    path.write_bytes(b"\0" * (4 * 1024 * 1024 - 5) + WINDOWS_BUILD.encode())
    checker = checker_module.PathChecker([WINDOWS_BUILD])
    checker.check_file(path)
    assert checker.failures == {"program.exe"}


def test_sysconfig_hook_remaps_nested_metadata():
    hook = runpy.run_path(str(PACKAGING / "hooks" / "hook-sysconfig.py"))
    code = compile(f"paths = (({POSIX_BUILD + '/include'!r},),)\nanswer = 42", "sysconfigdata.py", "exec")
    cleaned = hook["replace_build_prefix"](code, POSIX_BUILD)
    checker = checker_module.PathChecker([POSIX_BUILD])
    checker.check_code(cleaned, "sysconfigdata.py")
    assert not checker.failures
    namespace = {}
    exec(cleaned, namespace)
    assert namespace["paths"] == (("/mfq-build/python/include",),)
    assert namespace["answer"] == 42


def test_windows_installer_is_scanned_before_copy(tmp_path, monkeypatch, capfd):
    release = load_packaging_module("build_release")
    stage = tmp_path / "stage"
    bundle = stage / "target" / "release" / "bundle" / "nsis"
    bundle.mkdir(parents=True)
    installer = bundle / "setup.exe"
    output = tmp_path / "output"
    output.mkdir()
    context = SimpleNamespace(windows=True, cargo_target=stage, triple="target", output=output,
                              environment={}, in_venv=lambda _: sys.executable)
    monkeypatch.setattr(release, "release_path_sources", lambda _: [(str(stage), "/mfq-build/target")])
    monkeypatch.setattr(release, "short_path", lambda _: "")

    installer.write_bytes((str(stage) + "/before.cpp").encode())
    with pytest.raises(SystemExit, match="release path scan failed"):
        release.collect_installer(context)
    assert not (output / installer.name).exists()
    diagnostic = capfd.readouterr().out
    assert "Private build path in setup.exe" in diagnostic
    assert str(stage) not in diagnostic

    installer.write_bytes(b"/mfq-build/target/after.cpp")
    assert release.collect_installer(context).read_bytes() == installer.read_bytes()
