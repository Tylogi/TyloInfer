"""Real core route boundaries without Agent imports, model loading, or external network."""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import httpx

from server.tests.test_server_service import FakeBackend
from server.server.api import create_app
from server.server.protocol.models import ErrorDetail
from server.server.services.download_policy import apply_download_policy
from server.server.services.service import ServerService
from server.server.state.catalog import ModelArtifactNotFoundError, ModelCatalog
from server.server.state.storage import SessionStore


def test_companion_reference_requires_complete_registered_gguf_and_refreshes_identity(tmp_path, monkeypatch):
    source = tmp_path / "fixture.gguf"
    source.write_bytes(b"fixture")
    catalog = ModelCatalog([tmp_path])
    reads = []

    async def get(model_id, *, refresh=False):
        reads.append((model_id, refresh))
        if model_id == "0" * 32:
            raise ModelArtifactNotFoundError(model_id)
        return SimpleNamespace(path=source, resource=SimpleNamespace(id=model_id,
            format="hf" if model_id == "b" * 32 else "gguf", complete=model_id != "c" * 32))

    monkeypatch.setattr(catalog, "get", get)

    async def run():
        service = ServerService(SessionStore(tmp_path / "core.sqlite3"), FakeBackend(), catalog=catalog)
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(service)), base_url="http://test") as client:
                result = await client.get(f'/api/v1/models/{"a" * 32}/companion-reference')
                assert result.status_code == 200
                assert result.json() == {"model_id": "a" * 32, "format": "gguf", "path": str(source.resolve())}
                for model_id, expected in (("0" * 32, 404), ("b" * 32, 422), ("c" * 32, 422), ("invalid", 422)):
                    assert (await client.get(f"/api/v1/models/{model_id}/companion-reference")).status_code == expected
                assert (await client.get("/api/v1/agent/capabilities")).status_code == 404
                assert not hasattr(client._transport.app.state, "agent")
            assert reads and all(refresh for _, refresh in reads)
            assert source.read_bytes() == b"fixture"
        finally:
            await service.aclose()

    asyncio.run(run())


def test_native_job_archive_is_atomic_for_terminal_records_and_preserves_files(tmp_path):
    preserved = tmp_path / "weights.tq"
    preserved.write_bytes(b"fixture")
    store = SessionStore(tmp_path / "core.sqlite3")
    done = store.create_job("benchmark.fixture", {})
    store.claim_job(done.id)
    store.complete_job(done.id, {"fixture": True})
    failed = store.create_job("model.load", {})
    store.claim_job(failed.id)
    store.fail_job(failed.id, ErrorDetail(code="fixture", message="Fixture failure"))
    active = store.create_job("fixture.active", {})

    async def run():
        service = ServerService(store, FakeBackend())
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(service)), base_url="http://test") as client:
                assert (await client.delete(f"/api/v1/jobs/{active.id}")).status_code == 409
                assert (await client.delete(f"/api/v1/jobs/{done.id}")).status_code == 204
                assert (await client.get(f"/api/v1/jobs/{done.id}")).status_code == 404
                assert (await client.delete("/api/v1/jobs/completed")).status_code == 204
                assert (await client.get(f"/api/v1/jobs/{failed.id}")).status_code == 404
                assert (await client.get(f"/api/v1/jobs/{active.id}")).status_code == 200
            assert preserved.read_bytes() == b"fixture"
        finally:
            await service.aclose()

    asyncio.run(run())


def test_preview_download_policy_changes_only_registration_in_isolated_state(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location("isolated_preview_factory", root / "references" / "agent-preview-server.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "apply_download_policy", lambda paths, mode: apply_download_policy(paths, mode, module_available=lambda _name: True))

    async def run():
        for mode, expected in (("disabled", set()), ("enabled", {"download.huggingface", "download.modelscope"})):
            service = module.create_preview_service(tmp_path / mode, mode)
            try:
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(service)), base_url="http://test") as client:
                    response = await client.get("/api/v1/jobs/kinds")
                    assert response.status_code == 200
                    kinds = {row["kind"] for row in response.json()["data"] if row["kind"].startswith("download.")}
                    assert kinds == expected
                    assert (await client.get("/api/v1/runtime/instances")).json()["data"] == []
                assert not list((tmp_path / mode / "models").iterdir())
            finally:
                await service.aclose()

    asyncio.run(run())
