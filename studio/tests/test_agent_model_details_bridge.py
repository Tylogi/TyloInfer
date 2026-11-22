"""Verify model detail GETs through the real API using isolated checkpoint metadata."""

from __future__ import annotations

import asyncio
import shutil
import socket
import subprocess
import threading
from pathlib import Path

import numpy as np
import pytest
import uvicorn

from tests.test_checkpoint_cache_profile import checkpoint
from tests.test_server_service import FakeBackend
from formats.header import FileHeader
from formats.io import MMapTensorStore, save
from server.api import create_app
from server.services.service import ServerService
from server.state.catalog import ModelCatalog
from server.state.storage import SessionStore


def test_model_details_bridge_reads_native_profiles_and_directories_without_loading(tmp_path, monkeypatch):
    node = shutil.which("node")
    if node is None:
        pytest.skip("requires Node.js for the product Host module")
    root = Path(__file__).resolve().parents[2]
    models = tmp_path / "models"
    models.mkdir()
    known = models / "known.tq"
    unknown = models / "unknown.tq"
    checkpoint(known, predictor=True)
    save(unknown, FileHeader(model_arch="unknown"), {"weight": np.ones(4, dtype=np.float16)})
    catalog = ModelCatalog([models])
    known_id = ModelCatalog._inspect(models, known).resource.id
    unknown_id = ModelCatalog._inspect(models, unknown).resource.id
    monkeypatch.setattr(MMapTensorStore, "__getitem__", lambda *_: pytest.fail("model details must not decode tensor payloads"))
    backend = FakeBackend()
    service = ServerService(SessionStore(tmp_path / "core.sqlite3"), backend, catalog=catalog)
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
        assert ready.wait(10), "model detail fixture API did not start"
        subprocess.run(
            [node, "plugins/src/host/tests/model-details-api-smoke.ts", f"http://127.0.0.1:{port}", known_id, unknown_id],
            cwd=root, check=True, timeout=30,
        )
        assert backend.calls == []
        assert known.is_file() and unknown.is_file()
    finally:
        server.should_exit = True
        thread.join(10)
        listener.close()
        assert not thread.is_alive(), "model detail fixture API did not stop"
