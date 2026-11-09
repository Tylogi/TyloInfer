import pytest
from pydantic import ValidationError

from tyloinfer.cli import _build_parser
from tyloinfer.server.protocol.models import ModelLoadRequest
from tyloinfer.server.runtime import native
from tyloinfer.server.runtime.native import NativeRuntimeError, apply_native_prefill_mode


def test_prefill_mode_defaults_and_profile_roundtrip():
    assert _build_parser().parse_args(["serve"]).prefill_mode == "precise"
    assert ModelLoadRequest(model="model").prefill_mode == "precise"
    request = ModelLoadRequest(model="model", prefill_mode="fast")
    assert ModelLoadRequest.model_validate_json(request.model_dump_json()) == request
    assert _build_parser().parse_args(["serve", "--prefill-mode", "fast"]).prefill_mode == "fast"
    with pytest.raises(ValidationError):
        ModelLoadRequest(model="model", prefill_mode="unknown")


def test_modes_are_per_worker_and_override_inherited_settings():
    parent = {"TYLOINFER_PREFILL_MODE": "fast", "OTHER_SETTING": "preserved"}
    precise, fast = dict(parent), dict(parent)
    apply_native_prefill_mode(precise, "cuda", "precise")
    apply_native_prefill_mode(fast, "cuda", "fast")
    assert precise["TYLOINFER_PREFILL_MODE"] == "precise"
    assert fast["TYLOINFER_PREFILL_MODE"] == parent["TYLOINFER_PREFILL_MODE"] == "fast"
    assert precise["OTHER_SETTING"] == "preserved"


@pytest.mark.parametrize("mode", ["precise", "fast"])
def test_direct_worker_start_passes_the_selected_mode(tmp_path, monkeypatch, mode):
    executable, model = tmp_path / "worker", tmp_path / "model.gguf"
    executable.touch()
    model.touch()
    runtime = native.NativeRuntime(executable, model, "model", "cuda", prefill_mode=mode)
    monkeypatch.setattr(runtime, "command", lambda port: ["worker"])
    monkeypatch.setattr(runtime, "_wait_until_ready", lambda: None)
    monkeypatch.setattr(native, "native_runtime_environment", lambda *args, **kwargs: {"TYLOINFER_PREFILL_MODE": "fast"})
    captured = {}
    def start(command, **kwargs):
        captured.update(kwargs)
        return object()
    monkeypatch.setattr(native.subprocess, "Popen", start)
    runtime.start()
    assert captured["env"]["TYLOINFER_PREFILL_MODE"] == mode


@pytest.mark.parametrize("backend", ["metal", "hip"])
def test_other_backends_keep_their_native_prefill(backend):
    environment = {"TYLOINFER_PREFILL_MODE": "fast"}
    apply_native_prefill_mode(environment, backend, "precise")
    assert "TYLOINFER_PREFILL_MODE" not in environment
    with pytest.raises(NativeRuntimeError, match="requires CUDA"):
        apply_native_prefill_mode(environment, backend, "fast")
