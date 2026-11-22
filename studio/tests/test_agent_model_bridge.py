"""Exercise the JS bridge against the real API/job manager and a controlled runtime."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.agent_bridge_server import running_agent_bridge
from tests.test_server_service import FakeBackend
from server.api import create_app
from server.services.service import ServerService
from server.state.storage import SessionStore


class ControlledRuntime:
    def __init__(self):
        self.loads = []
        self.unloads = []

    async def start(self):
        pass

    async def load(self, context, payload):
        self.loads.append(payload)
        if payload["model"] == "bridge-fixture-pending":
            await asyncio.Event().wait()
        return {"instance_id": "22222222-2222-4222-8222-222222222222"}

    async def unload(self, context, payload):
        self.unloads.append(payload)
        return {"unloaded": True}


def test_model_bridge_real_api_tickets_completion_cancel_and_validation(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("requires Node.js for the product Host module")
    runtime = ControlledRuntime()
    backend = FakeBackend()
    service = ServerService(SessionStore(tmp_path / "core.sqlite3"), backend, runtime_manager=runtime)
    with running_agent_bridge(create_app(service)) as base_url:
        root = Path(__file__).resolve().parents[2]
        subprocess.run(
            [node, "plugins/host/tests/core-api-smoke.mjs", base_url],
            cwd=root, check=True, timeout=30,
        )
        assert [item["model"] for item in runtime.loads] == ["bridge-fixture-success", "bridge-fixture-pending"]
        assert runtime.loads[0]["kv_memory"]["qsa_mode"] == "auto"
        assert runtime.loads[0]["kv_memory"]["placement"] == "auto"
        assert len(runtime.unloads) == 1
        assert runtime.unloads[0]["force"] is False
        assert backend.calls == []
