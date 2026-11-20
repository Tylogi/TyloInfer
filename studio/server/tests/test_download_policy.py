"""Download registration is explicit and never starts provider work."""

from __future__ import annotations

from pathlib import Path

import pytest

from server.cli import _build_parser
from server.server.services.download_policy import apply_download_policy
from server.server.services.tool_jobs import ToolJobHandlers, ToolJobPaths
from server.server.state.catalog import ModelCatalog


def paths(root):
    return ToolJobPaths(work_root=root / "work", python=Path("python"), modelscope=None, huggingface=None,
        runtime=root / "runtime", diagnostics=root / "diagnostics", standalone_cli=True)


def test_disabled_never_probes_or_registers_download_handlers(tmp_path):
    source = paths(tmp_path)
    source = ToolJobPaths(**{**source.__dict__, "internal_huggingface": True, "internal_modelscope": True})

    def forbidden(_name):
        pytest.fail("disabled policy must not probe providers")

    selected = apply_download_policy(source, "disabled", module_available=forbidden)
    assert not selected.internal_huggingface and not selected.internal_modelscope
    assert selected.huggingface is None and selected.modelscope is None
    assert selected.runtime == source.runtime and selected.diagnostics == source.diagnostics
    assert selected.standalone_cli is True
    catalog = ModelCatalog([tmp_path / "models"])
    handlers = ToolJobHandlers(catalog, selected).handlers()
    assert "download.huggingface" not in handlers and "download.modelscope" not in handlers


@pytest.mark.parametrize("mode", ["auto", "enabled"])
@pytest.mark.parametrize("available", ["huggingface_hub", "modelscope_hub"])
def test_installed_internal_provider_is_registered_without_execution(tmp_path, mode, available):
    selected = apply_download_policy(paths(tmp_path), mode, module_available=lambda name: name == available)
    handlers = ToolJobHandlers(ModelCatalog([tmp_path / "models"]), selected).handlers()
    assert ("download.huggingface" in handlers) == (available == "huggingface_hub")
    assert ("download.modelscope" in handlers) == (available == "modelscope_hub")
    assert not (tmp_path / "models").exists()


def test_enabled_fails_closed_when_no_provider_exists(tmp_path):
    with pytest.raises(ValueError, match="no supported Hub SDK"):
        apply_download_policy(paths(tmp_path), "enabled", module_available=lambda _name: False)
    selected = apply_download_policy(paths(tmp_path), "auto", module_available=lambda _name: False)
    assert not selected.internal_huggingface and not selected.internal_modelscope


def test_installed_external_provider_is_preserved_without_launching(tmp_path):
    executable = tmp_path / "fixture-hub"
    executable.write_bytes(b"fixture-not-executed")
    source = paths(tmp_path)
    source = ToolJobPaths(**{**source.__dict__, "huggingface": executable})
    selected = apply_download_policy(source, "enabled", module_available=lambda _name: False)
    assert selected.huggingface == executable
    assert not selected.internal_huggingface
    assert executable.read_bytes() == b"fixture-not-executed"


def test_serve_exposes_policy_and_preserves_default_auto():
    parser = _build_parser()
    assert parser.parse_args(["serve"]).downloads == "auto"
    for mode in ("auto", "disabled", "enabled"):
        assert parser.parse_args(["serve", "--downloads", mode]).downloads == mode
    with pytest.raises(SystemExit):
        parser.parse_args(["serve", "--downloads", "invalid"])
