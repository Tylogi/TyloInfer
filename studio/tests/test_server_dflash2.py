from __future__ import annotations

import asyncio
import json
import struct
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import pytest

from server.protocol.models import ModelLoadRequest, RuntimeInstanceState, SamplingParams, UpdateRuntimeInstanceRequest
from server.runtime.backend import BackendDelta
from server.runtime.runtime_pool import RuntimePool, _Runtime, _RuntimeLoadContext
from server.services.jobs import JobExecutionError
from server.state.catalog import ModelCatalog
from tests.test_server_models import IdleBackend, _model


def _draft(path):
    path.mkdir()
    (path / "config.json").write_text(json.dumps({
        "model_type": "qwen3", "hidden_size": 64, "vocab_size": 128,
        "dflash_config": {"target_layer_ids": [0, 2]},
    }))
    header = json.dumps({
        "projection.weight": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]},
    }).encode()
    (path / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"\0\0")


def test_dflash2_is_a_registered_companion_and_launches_by_id(tmp_path):
    async def run():
        _model(tmp_path / "target.tq")
        _draft(tmp_path / "draft")
        catalog = ModelCatalog([tmp_path])
        target = await catalog.resolve("target")
        target = replace(target, resource=target.resource.model_copy(update={"architecture": "qwen4_exp_text"}),
                         source_config={"text_config": {"hidden_size": 64, "vocab_size": 128, "num_hidden_layers": 3}})
        assert [item.name for item in (await catalog.list()).data] == ["target"]
        draft, = [item for item in (await catalog.list(include_components=True)).data if item.asset_kind == "dflash2"]
        assert draft.complete and not draft.loadable
        pool = RuntimePool(catalog, tmp_path / "runtime", backend="metal")
        selected = await pool._dflash2_artifact(target, draft.id)
        request = ModelLoadRequest(model=target.resource.name, dflash2_artifact_id=draft.id)
        command, _ = pool._launch_configuration(target, request, port=0, dflash2_path=selected.path)
        assert command[command.index("--dflash2-draft") + 1] == str(tmp_path / "draft")
        incompatible = replace(target, source_config={"hidden_size": 32})
        with pytest.raises(JobExecutionError, match="geometry does not match"):
            await pool._dflash2_artifact(incompatible, draft.id)
        await pool.aclose()

    asyncio.run(run())


def test_dflash2_failed_replacement_restores_original_options(tmp_path, monkeypatch):
    async def run():
        _model(tmp_path / "target.tq")
        _draft(tmp_path / "draft")
        catalog = ModelCatalog([tmp_path])
        target = await catalog.resolve("target")
        target = replace(target, resource=target.resource.model_copy(update={"architecture": "qwen4_exp_text"}))
        draft, = [item for item in (await catalog.list(include_components=True)).data if item.asset_kind == "dflash2"]
        pool = RuntimePool(catalog, tmp_path / "runtime", backend="metal")
        instance = _Runtime(id=uuid4(), artifact=target, process=SimpleNamespace(returncode=None),
                            backend=IdleBackend(), port=0, context_size=8192, state=RuntimeInstanceState.READY,
                            pinned=True)
        pool._instances[instance.id] = instance
        pool._load_requests["target"] = ModelLoadRequest(model="target", context_size=8192, pin=True)

        async def retire(item):
            pool._instances.pop(item.id, None)

        attempted = []

        async def load(_context, config):
            attempted.append(config)
            listed, = (await pool.runtime_models())["data"]
            assert listed["id"] == "target" and listed["state"] == "loading"
            if len(attempted) == 1:
                pool._load_events["target"] = asyncio.Event()
                pool._loading_model_names.add("target")
                raise RuntimeError("incompatible worker")
            assert "target" not in pool._loading_model_names
            return {"context_size": 8192}

        monkeypatch.setattr(pool, "_retire_instance", retire)
        monkeypatch.setattr(pool, "_load", load)
        context = _RuntimeLoadContext()
        request = {"model": target.resource.id, "dflash2_artifact_id": draft.id, "replace_instance_id": str(instance.id)}
        with pytest.raises(RuntimeError, match="incompatible worker"):
            await pool.load(context, request)
        first, recovery = attempted
        assert first["dflash2_artifact_id"] == draft.id and recovery["dflash2_artifact_id"] is None
        assert first["context_size"] == recovery["context_size"] == 8192 and first["pin"] and recovery["pin"]
        assert not pool._dflash2_replacements
        await pool.aclose()

    asyncio.run(run())


def test_dflash2_instance_toggle_applies_to_the_next_request(tmp_path):
    class Backend(IdleBackend):
        enabled = None

        async def stream(self, **options):
            self.enabled = options["sampling"].enable_mtp
            yield BackendDelta(content_delta="ok", finish_reason="stop")

    async def run():
        _model(tmp_path / "target.tq")
        catalog = ModelCatalog([tmp_path])
        target = await catalog.resolve("target")
        backend = Backend()
        pool = RuntimePool(catalog, tmp_path / "runtime", backend="metal", automatic_memory_budget=False)
        instance = _Runtime(id=uuid4(), artifact=target, process=SimpleNamespace(returncode=None),
                            backend=backend, port=0, context_size=4096, state=RuntimeInstanceState.READY,
                            request_slots=asyncio.Semaphore(1), dflash2_path=tmp_path / "draft", dflash2_enabled=True)
        pool._instances[instance.id] = instance
        pool._load_requests["target"] = ModelLoadRequest(model="target")
        for enabled in (False, True):
            updated = await pool.update_instance(instance.id, UpdateRuntimeInstanceRequest(dflash2_enabled=enabled))
            assert updated.id == instance.id and updated.dflash2_enabled is enabled
            assert pool._load_requests["target"].dflash2_enabled is enabled
            async for _ in pool.stream(model="target", messages=({"role": "user", "content": "hello"},),
                                       sampling=SamplingParams(enable_mtp=True)):
                pass
            assert backend.enabled is enabled

    asyncio.run(run())
