"""Exercise page routes through the real API and jobs without native inference or network."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.agent_bridge_server import running_agent_bridge
from tests.test_checkpoint_analysis import checkpoint
from tests.test_server_service import FakeBackend
from server.api import create_app
from server.protocol.models import (
    HubModelInfo,
    HubModelSearchResult,
    HubSystemProfile,
    OfficialModelList,
)
from server.runtime.runtime_pool import RuntimePool
from server.services.jobs import JobManager
from server.services.service import ServerService
from server.services.tool_jobs import ToolJobHandlers, ToolJobPaths
from server.state.catalog import ModelCatalog
from server.state.storage import SessionStore


class EmptyBackend(FakeBackend):
    async def runtime_status(self):
        return {"runtime_state": "idle", "model": None, "active_requests": 0}

    async def runtime_models(self):
        return {"object": "list", "data": []}


class ControlledHub:
    def __init__(self):
        self.calls = []

    async def browse(self, provider, query, **options):
        self.calls.append(("search", provider, query, options))
        return HubModelSearchResult(data=[])

    async def info(self, provider, repo_id, revision):
        self.calls.append(("info", provider, repo_id, revision))
        return HubModelInfo(provider=provider, repo_id=repo_id, revision=revision or "main", files=[])

    async def official(self, **options):
        self.calls.append(("official", options))
        return OfficialModelList(system=options["profile"], data=[])


class ControlledSensors:
    async def sample(self, service):
        return {"sampled_at": 1, "gpus": [], "disks": [], "weights": []}


def test_page_bridge_real_api_registration_analysis_policies_and_controlled_downloads(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("requires Node.js for the product Host module")
    root = Path(__file__).resolve().parents[2]
    catalog_root = tmp_path / "catalog"
    source = tmp_path / "source"
    work = tmp_path / "work"
    for directory in (catalog_root, source, work):
        directory.mkdir()
    checkpoint(source / "bridge-metadata.tq")
    catalog = ModelCatalog([catalog_root], browse_roots=[source])
    backend = EmptyBackend()
    runtime = RuntimePool(catalog, str(tmp_path / "never-started-runtime"), backend="cuda", startup_loads=())
    store = SessionStore(tmp_path / "core.sqlite3")
    downloads = []

    async def controlled_download(context, payload):
        downloads.append(payload)
        if payload["repo_id"] == "example/pending":
            await asyncio.Event().wait()
        return {"fixture": True}

    jobs = JobManager(store, {"download.huggingface": controlled_download, "download.modelscope": controlled_download})
    handlers = ToolJobHandlers(catalog, ToolJobPaths(work_root=work, python=Path(sys.executable), modelscope=None, huggingface=None, runtime=None))
    hub = ControlledHub()
    service = ServerService(store, backend, catalog=catalog, runtime_manager=runtime, jobs=jobs, tool_handlers=handlers, hub_catalog=hub)
    service.resource_monitor = ControlledSensors()
    service._hub_system_profile = lambda: HubSystemProfile(platform="fixture", machine="fixture", backend="cpu")
    with running_agent_bridge(create_app(service)) as base_url:
        subprocess.run([node, "plugins/host/tests/page-api-smoke.mjs", base_url], cwd=root, check=True, timeout=30)
        assert [item["repo_id"] for item in downloads] == ["example/success", "example/pending"]
        assert [item[0] for item in hub.calls] == ["search", "info", "official"]
        assert not runtime._instances
        assert backend.calls == []
