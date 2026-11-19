"""Local official catalog reads must never activate network, refresh, or persistence."""

from __future__ import annotations

import asyncio
import json
import time

import httpx
import pytest

from tests.test_server_service import FakeBackend
from tyloinfer.server.api import create_app
from tyloinfer.server.protocol.models import HubModelFile, HubModelInfo, HubSystemProfile
from tyloinfer.server.services import hub as hub_module
from tyloinfer.server.services.hub import HubCatalog
from tyloinfer.server.services.service import ServerService
from tyloinfer.server.state.storage import SessionStore

PROFILE = HubSystemProfile(platform="fixture", machine="fixture", backend="cpu", runtime_memory_budget_bytes=4096)
REPOSITORY = "Tylogi/Fixture-TQ"


def cached_model():
    return HubModelInfo(provider="huggingface", repo_id=REPOSITORY, revision="main",
        files=[HubModelFile(name="Q4/fixture.tq", byte_size=128)])


def seed(catalog):
    model = cached_model()
    key = (model.provider, model.repo_id)
    catalog._discovered_sources.add(key)
    catalog._official_cache[key] = model
    return model


def memory_state(catalog):
    return {
        "models": {key: value.model_dump(mode="json") if value else None for key, value in catalog._official_cache.items()},
        "sources": set(catalog._discovered_sources),
        "cache_time": catalog._official_cache_time,
        "source_times": dict(catalog._source_cache_times),
        "detail_times": dict(catalog._detail_attempt_times),
        "task": catalog._refresh_task,
        "closed": catalog._closed,
    }


def forbid_side_effects(catalog, monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("local snapshot attempted refresh, network, or cache persistence")
    for name in ("_refresh_official_cache", "_start_detail_refresh", "_refresh_mfq_details", "_fetch_info",
                 "_inspect_mfq_files", "_discover_official", "_save_cache", "_load_cache", "info", "official"):
        monkeypatch.setattr(catalog, name, forbidden)
    monkeypatch.setattr(hub_module, "_metadata_client", forbidden)


@pytest.mark.parametrize("cache_state", ["cold", "stale", "fresh"])
def test_local_snapshot_does_not_refresh_or_persist_any_cache_state(tmp_path, monkeypatch, cache_state):
    path = tmp_path / "official.json"
    catalog = HubCatalog(cache_path=path)
    if cache_state != "cold":
        seed(catalog)
    if cache_state == "fresh":
        catalog._official_cache_time = time.monotonic()
    before = memory_state(catalog)
    forbid_side_effects(catalog, monkeypatch)

    async def run():
        try:
            result = await catalog.official_local(profile=PROFILE)
            assert result.system == PROFILE
            assert result.refreshing is False
            assert len(result.data) == (0 if cache_state == "cold" else 1)
            if result.data:
                assert result.data[0].sources[0].repo_id == REPOSITORY
                assert result.data[0].variants[0].files == ["Q4/fixture.tq"]
            assert memory_state(catalog) == before
            assert not path.exists()
        finally:
            await catalog.aclose()

    asyncio.run(run())


def test_local_snapshot_projects_loaded_cache_without_rereading_or_writing_disk(tmp_path, monkeypatch):
    path = tmp_path / "official.json"
    now = time.time()
    path.write_text(json.dumps({"version": 1, "updated_at": now,
        "models": [{"model": cached_model().model_dump(mode="json"), "cached_at": now}]}), encoding="utf-8")
    catalog = HubCatalog(cache_path=path)
    path.write_bytes(b"changed-after-cache-was-loaded")
    before_bytes = path.read_bytes()
    before_time = path.stat().st_mtime_ns
    before = memory_state(catalog)
    forbid_side_effects(catalog, monkeypatch)

    async def run():
        try:
            result = await catalog.official_local(profile=PROFILE)
            assert len(result.data) == 1
            assert result.data[0].sources[0].repo_id == REPOSITORY
            assert memory_state(catalog) == before
            assert path.read_bytes() == before_bytes
            assert path.stat().st_mtime_ns == before_time
        finally:
            await catalog.aclose()

    asyncio.run(run())


def test_local_snapshot_reports_but_never_replaces_or_cancels_an_existing_refresh(monkeypatch):
    async def run():
        catalog = HubCatalog()
        pending = asyncio.create_task(asyncio.Event().wait())
        catalog._refresh_task = pending
        before = memory_state(catalog)
        forbid_side_effects(catalog, monkeypatch)
        try:
            result = await catalog.official_local(profile=PROFILE)
            assert result.refreshing is True
            assert catalog._refresh_task is pending and not pending.done()
            assert memory_state(catalog) == before
        finally:
            await catalog.aclose()
        assert pending.cancelled()

    asyncio.run(run())


def test_explicit_official_refresh_semantics_are_unchanged(monkeypatch):
    async def run():
        catalog = HubCatalog()
        calls = []

        async def refresh(profile):
            assert profile == PROFILE
            calls.append("refresh")

        def detail(profile):
            assert profile == PROFILE
            calls.append("detail")
            return False

        monkeypatch.setattr(catalog, "_refresh_official_cache", refresh)
        monkeypatch.setattr(catalog, "_start_detail_refresh", detail)
        try:
            assert (await catalog.official(profile=PROFILE, refresh=False)).refreshing is True
            await catalog._refresh_task
            catalog._official_cache_time = time.monotonic()
            assert (await catalog.official(profile=PROFILE, refresh=False)).refreshing is False
            assert calls == ["refresh", "detail"]
            assert (await catalog.official(profile=PROFILE, refresh=True)).refreshing is True
            await catalog._refresh_task
            assert calls == ["refresh", "detail", "refresh"]
        finally:
            await catalog.aclose()

    asyncio.run(run())


def test_local_snapshot_real_route_returns_native_data_and_current_profile_without_side_effects(tmp_path, monkeypatch):
    async def run():
        catalog = HubCatalog(cache_path=tmp_path / "official.json")
        seed(catalog)
        before = memory_state(catalog)
        forbid_side_effects(catalog, monkeypatch)
        backend = FakeBackend()
        service = ServerService(SessionStore(tmp_path / "core.sqlite3"), backend, hub_catalog=catalog)
        monkeypatch.setattr(service, "_hub_system_profile", lambda: PROFILE)
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(service)), base_url="http://test") as client:
                for suffix in ("", "?refresh=true"):
                    response = await client.get("/api/v1/hub/official/local" + suffix)
                    assert response.status_code == 200
                    value = response.json()
                    assert value["system"] == PROFILE.model_dump(mode="json")
                    assert value["data"][0]["sources"][0]["repo_id"] == REPOSITORY
                    assert value["refreshing"] is False
                    assert memory_state(catalog) == before
            assert backend.calls == []
            assert not (tmp_path / "official.json").exists()
        finally:
            await service.aclose()

    asyncio.run(run())


def test_local_snapshot_unavailable_provider_never_falls_back_to_network_catalog(tmp_path, monkeypatch):
    class NetworkOnlyHub:
        async def official(self, **_kwargs):
            pytest.fail("local route must never fall back to the network catalog")

        async def aclose(self):
            pass

    async def run():
        service = ServerService(SessionStore(tmp_path / "core.sqlite3"), FakeBackend(), hub_catalog=NetworkOnlyHub())
        monkeypatch.setattr(service, "_hub_system_profile", lambda: pytest.fail("unsupported provider must fail before profile sampling"))
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(service)), base_url="http://test") as client:
                response = await client.get("/api/v1/hub/official/local")
                assert response.status_code == 501
                assert response.json()["error"]["code"] == "local_official_catalog_unavailable"
        finally:
            await service.aclose()

    asyncio.run(run())
