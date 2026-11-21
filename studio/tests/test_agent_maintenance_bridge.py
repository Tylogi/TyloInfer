"""Exercise maintenance via the real API using only isolated fixtures and native-control doubles."""

from __future__ import annotations

import asyncio
import shutil
import socket
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest
import uvicorn

from tests.test_prefix_cache_management import text_record
from tests.test_prefix_disk_budget import block
from tests.test_server_service import FakeBackend
from server.api import create_app
from server.protocol.models import JobEventType
from server.runtime.runtime_pool import RuntimePool
from server.services.service import ServerService
from server.services.tool_jobs import ToolJobHandlers, ToolJobPaths
from server.state.catalog import ModelCatalog
from server.state.storage import SessionStore

INSTANCE_ID = UUID("11111111-1111-4111-8111-111111111111")


class MaintenanceBackend(FakeBackend):
    def __init__(self):
        super().__init__()
        self.controls = []

    async def clear_runtime_cache(self, instance_id):
        assert instance_id == INSTANCE_ID
        self.controls.append(("clear", instance_id))
        return {"status": "ok", "released_snapshots": 2}

    async def trim_runtime_cache(self, target_bytes, instance_id):
        assert instance_id == INSTANCE_ID
        self.controls.append(("trim", instance_id, target_bytes))
        return {"status": "ok", "released_bytes": 8, "target_bytes": target_bytes}


class BusyFixturePool(RuntimePool):
    async def purge_prefix_cache(self, namespace=None):
        busy = namespace == "c" * 64
        if busy:
            self._loading_model_names.add("busy-fixture")
        try:
            return await super().purge_prefix_cache(namespace)
        finally:
            if busy:
                self._loading_model_names.remove("busy-fixture")


def test_maintenance_bridge_real_api_scope_partial_failure_directory_and_events(tmp_path, monkeypatch):
    node = shutil.which("node")
    if node is None:
        pytest.skip("requires Node.js for the product Host module")
    root = Path(__file__).resolve().parents[2]
    models, downloads, work = (tmp_path / name for name in ("models", "downloads", "work"))
    for directory in (models, downloads, work):
        directory.mkdir()
    cache = tmp_path / "prefix-cache"
    selected = block(cache, "a", "1")
    selected_text = text_record(selected, "a", "1", [41] * 256)
    partial = block(cache, "b", "2")
    blocked_text = text_record(partial, "b", "2", [42] * 256)
    busy = block(cache, "c", "3")
    remaining = block(cache, "d", "4")
    untouched = cache / "keep.txt"
    untouched.write_text("fixture", encoding="utf-8")
    (selected.parent.parent / "identity.txt").write_text(
        "mfq-cache-identity-v1\nnamespace=" + "a" * 64 + "\nshard=0:fixture.tq:100:1\n", encoding="utf-8"
    )
    catalog = ModelCatalog([models], browse_roots=[downloads])

    async def resolve_fixture(_name):
        return SimpleNamespace(path=tmp_path / "fixture-tokenizer")

    monkeypatch.setattr(catalog, "resolve", resolve_fixture)
    monkeypatch.setattr("server.runtime.runtime_pool.decode_prefix_cache_tokens", lambda _path, tokens: " ".join(str(token) for token in tokens))
    unlink = Path.unlink

    def deny_fixture_sidecar(path, *args, **kwargs):
        if path == blocked_text:
            raise PermissionError("fixture sidecar is locked")
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", deny_fixture_sidecar)
    image_releases = []

    def release_fixture_images():
        image_releases.append(True)
        return 7

    monkeypatch.setattr("server.services.service.clear_image_decode_cache", release_fixture_images)
    backend = MaintenanceBackend()
    pool = BusyFixturePool(catalog, tmp_path / "never-started-runtime", runtime_environment={"TYLOINFER_RUNTIME_PREFIX_CACHE_DIR": str(cache)}, startup_loads=())
    store = SessionStore(tmp_path / "core.sqlite3")
    fixture_job = store.create_job("maintenance.fixture", {"fixture": True})
    store.claim_job(fixture_job.id)
    store.append_job_event(fixture_job.id, JobEventType.LOG, message="fixture event", data={"fixture": True})
    store.complete_job(fixture_job.id, {"fixture": True})
    handlers = ToolJobHandlers(catalog, ToolJobPaths(work_root=work, python=Path(sys.executable), modelscope=None, huggingface=None, runtime=None), runtime_manager=pool)
    service = ServerService(store, backend, catalog=catalog, runtime_manager=pool, tool_handlers=handlers)
    ready = threading.Event()

    class ReadyServer(uvicorn.Server):
        async def startup(self, sockets=None):
            await super().startup(sockets=sockets)
            if self.started:
                ready.set()

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    server = ReadyServer(uvicorn.Config(create_app(service), log_level="error", access_log=False))
    thread = threading.Thread(target=lambda: asyncio.run(server.serve(sockets=[listener])), daemon=True)
    thread.start()
    try:
        assert ready.wait(10), "maintenance fixture API did not start"
        subprocess.run([node, "plugins/host/tests/maintenance-api-smoke.mjs", f"http://127.0.0.1:{port}", str(fixture_job.id)], cwd=root, check=True, timeout=30)
        assert backend.controls == [("clear", INSTANCE_ID), ("trim", INSTANCE_ID, 0)]
        assert image_releases == [True]
        assert backend.calls == [] and not pool._instances
        assert not selected.exists() and not selected_text.exists()
        assert not partial.exists() and blocked_text.exists()
        assert not busy.exists() and not remaining.exists()
        assert untouched.read_text(encoding="utf-8") == "fixture"
        assert handlers.download_directories.selected is None
        assert handlers.download_directories.evict_prefix_cache is False
        assert not list(downloads.iterdir())
    finally:
        server.should_exit = True
        thread.join(10)
        listener.close()
        assert not thread.is_alive(), "maintenance fixture API did not stop"
