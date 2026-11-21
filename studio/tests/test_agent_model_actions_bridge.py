"""Exercise fixed model actions against isolated real APIs and controlled runtime/Hub work."""

from __future__ import annotations

import asyncio
import shutil
import socket
import subprocess
import sys
import threading
from pathlib import Path
from uuid import UUID

import pytest
import uvicorn

from tests.test_checkpoint_cache_profile import checkpoint
from tests.test_server_service import FakeBackend
from server.api import create_app
from server.protocol.models import (
    ErrorDetail,
    HubModelFile,
    HubModelInfo,
    RuntimeInstanceResource,
)
from server.runtime.runtime_pool import RuntimePool
from server.services.jobs import JobManager
from server.services.service import ServerService
from server.services.tool_jobs import ToolJobHandlers, ToolJobPaths
from server.state.catalog import ModelCatalog
from server.state.storage import SessionStore

INSTANCE = UUID("11111111-1111-4111-8111-111111111111")


class ControlledRuntime(RuntimePool):
    def __init__(self, catalog, executable):
        super().__init__(catalog, executable, startup_loads=())
        self.policy_updates = []
        self.context_updates = []
        self.context_release = asyncio.Event()

    async def update_instance(self, instance_id, request):
        assert instance_id == INSTANCE
        self.policy_updates.append(request.model_dump(exclude_unset=True))
        return RuntimeInstanceResource(id=INSTANCE, model="fixture", state="ready", devices=[], active_sessions=0,
            queued_requests=0, pinned=request.pinned, idle_ttl_seconds=request.idle_ttl_seconds)

    async def yarn_context_info(self, instance_id):
        assert instance_id == INSTANCE
        return {"supported": False, "native_context": None, "maximum_context": None, "maximum_factor": None,
            "enabled": False, "effective_factor": 1}

    async def configure_context(self, context, payload):
        assert payload["instance_id"] == str(INSTANCE)
        self.context_updates.append(payload)
        await self.context_release.wait()
        return {"fixture": True}


class ControlledHub:
    def __init__(self):
        self.calls = []

    async def readme(self, provider, repo_id, revision):
        self.calls.append(("readme", provider, repo_id, revision))
        return None if revision == "missing" else "# Fixture model"

    async def probe_download_sources(self, sources):
        self.calls.append(("probe", len(sources)))
        return {"data": [{"provider": row.provider, "repo_id": row.repo_id, "bytes_per_second": 0} for row in sources]}

    async def download_files(self, provider, repo_id, revision):
        self.calls.append(("files", provider, repo_id, revision))
        return HubModelInfo(provider=provider, repo_id=repo_id, revision=revision or "main",
            files=[HubModelFile(name="fixture.tq", byte_size=7)])


def test_model_actions_bridge_real_api_with_isolated_deletion_and_controlled_work(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("requires Node.js for the product Host module")
    root = Path(__file__).resolve().parents[2]
    models, work = tmp_path / "models", tmp_path / "work"
    models.mkdir()
    work.mkdir()
    deletable = models / "deletable.tq"
    assert deletable.resolve().is_relative_to(tmp_path.resolve())
    checkpoint(deletable)
    reusable = models / "reuse" / "fixture.tq"
    reusable.parent.mkdir()
    reusable.write_bytes(b"fixture")
    catalog = ModelCatalog([models])
    artifact_id = ModelCatalog._inspect(models, deletable).resource.id
    runtime = ControlledRuntime(catalog, tmp_path / "never-started-runtime")
    backend = FakeBackend()
    hub = ControlledHub()
    store = SessionStore(tmp_path / "core.sqlite3")
    payload = {"repo_id": "example/retried", "destination": "models/retried"}
    failed = store.create_job("download.huggingface", payload)
    store.claim_job(failed.id)
    store.fail_job(failed.id, ErrorDetail(code="fixture_failure", message="Fixture failure"))
    downloads = []

    async def download(context, request):
        downloads.append(request)
        return {"fixture": True}

    jobs = JobManager(store, {"download.huggingface": download, "download.modelscope": download})
    handlers = ToolJobHandlers(catalog, ToolJobPaths(work_root=work, python=Path(sys.executable),
        modelscope=None, huggingface=None, runtime=None))
    service = ServerService(store, backend, catalog=catalog, runtime_manager=runtime, jobs=jobs,
        tool_handlers=handlers, hub_catalog=hub)
    ready = threading.Event()

    class ReadyServer(uvicorn.Server):
        async def startup(self, sockets=None):
            await super().startup(sockets=sockets)
            if self.started:
                ready.set()

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    app = create_app(service)

    @app.post("/fixture/release-context")
    async def release_context():
        runtime.context_release.set()
        return {"released": True}

    server = ReadyServer(uvicorn.Config(app, log_level="error", access_log=False))
    thread = threading.Thread(target=lambda: asyncio.run(server.serve(sockets=[listener])), daemon=True)
    thread.start()
    try:
        assert ready.wait(10), "model action fixture API did not start"
        subprocess.run([node, "plugins/host/tests/model-actions-api-smoke.mjs", f"http://127.0.0.1:{port}",
            artifact_id, str(failed.id)], cwd=root, check=True, timeout=30)
        assert runtime.policy_updates == [{"pinned": True, "idle_ttl_seconds": None}]
        assert len(runtime.context_updates) == 1
        assert runtime.context_updates[0]["context_size"] == 1024
        assert len(downloads) == 1
        assert not deletable.exists()
        assert reusable.read_bytes() == b"fixture"
        assert backend.calls == [] and not runtime._instances
        assert [row[0] for row in hub.calls] == ["readme", "readme", "probe", "files", "files"]
    finally:
        server.should_exit = True
        thread.join(10)
        listener.close()
        assert not thread.is_alive(), "model action fixture API did not stop"
