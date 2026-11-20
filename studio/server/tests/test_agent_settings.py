"""Native settings API checks; real profile tests never submit model work."""

from __future__ import annotations

import asyncio
import json
import os
from uuid import uuid4

import httpx
import pytest

from server.tests.legacy_agent_support import AgentService, create_app
from server.tests.test_server_service import FakeBackend, make_service
from server.server.api.auth import ApiKeyManager
from server.server.protocol.models import CreateApiKeyRequest


def test_settings_require_admin_before_starting_a_runtime(tmp_path):
    async def check():
        service = make_service(tmp_path, FakeBackend())
        keys = ApiKeyManager(service.store)
        issued = keys.create(CreateApiKeyRequest(name="reader", scopes=["inference"]))
        app = create_app(service, api_key="test-root", api_keys=keys)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as client:
            assert (await client.get("/api/v1/agent/settings")).status_code == 401
            denied = await client.get(
                "/api/v1/agent/settings", headers={"Authorization": "Bearer " + issued.token}
            )
            assert denied.status_code == 403
            denied = await client.patch(
                "/api/v1/agent/settings/ui-theme",
                json={},
                headers={"Authorization": "Bearer " + issued.token},
            )
            assert denied.status_code == 403
        assert app.state.agent is None

    asyncio.run(check())


def test_settings_missing_runtime_and_invalid_envelopes(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "tyloinfer_legacy_agent.settings.DshRuntime.capabilities",
        lambda: {"available": False},
    )

    async def check():
        app = create_app(make_service(tmp_path, FakeBackend()))
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as client:
            response = await client.get("/api/v1/agent/settings")
            assert response.status_code == 503
            body = {
                "values": {"fontSize": 14},
                "expected_revision": 0,
                "generation": str(uuid4()),
            }
            for invalid in (
                {**body, "expected_revision": True},
                {**body, "unknown": "value"},
                {**body, "values": {"textFontFamily": "x" * 32769}},
            ):
                assert (
                    await client.patch("/api/v1/agent/settings/ui-theme", json=invalid)
                ).status_code == 422
        await app.state.agent.aclose()
        assert not app.state.agent.settings.root.exists()

    asyncio.run(check())


@pytest.mark.skipif(
    os.environ.get("TYLOINFER_TEST_DSH") != "1",
    reason="requires the pinned optional DSH SDK and matching official CLI",
)
def test_real_native_settings_save_validate_conflict_reset_and_restart(tmp_path):
    async def check():
        backend = FakeBackend()
        service = make_service(tmp_path, backend)
        app = create_app(service)
        app.state.agent = AgentService(service)
        agent = app.state.agent
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app), base_url="http://test"
            ) as client:
                response = await client.get("/api/v1/agent/settings")
                assert response.status_code == 200
                assert response.headers["cache-control"] == "no-store"
                view = response.json()
                assert view["persistence"] == "isolated-profile"
                assert view["runtime"] == "dsh" and view["writable"]
                assert not view["hasDocument"]
                assert {row["ns"] for row in view["namespaces"]} == {
                    "ui-theme",
                    "locale",
                    "ui-chat",
                    "ui-conversation",
                    "ui-settings",
                }
                theme = next(row for row in view["namespaces"] if row["ns"] == "ui-theme")
                assert theme["value"]["fontSize"] == 14
                root_schema = theme["schema"]["refs"][str(theme["schema"]["uid"])]
                assert root_schema["type"] == "object"
                original = {
                    "values": {"fontSize": 16},
                    "expected_revision": theme["revision"],
                    "generation": view["generation"],
                }
                response = await client.patch("/api/v1/agent/settings/ui-theme", json=original)
                assert response.status_code == 200
                saved = response.json()
                assert saved["value"]["fontSize"] == 16
                assert saved["applies"] == "live" and saved["revision"] > theme["revision"]
                current = {**original, "expected_revision": saved["revision"]}
                rejected = await client.patch(
                    "/api/v1/agent/settings/ui-theme",
                    json={**current, "values": {"fontSize": 99}},
                )
                assert rejected.status_code == 422
                assert (
                    await client.patch("/api/v1/agent/settings/ui-theme", json=original)
                ).status_code == 409
                assert (
                    await client.patch(
                        "/api/v1/agent/settings/ui-theme",
                        json={**current, "generation": str(uuid4())},
                    )
                ).status_code == 409
                for namespace, values, status in (
                    ("ui-theme", {"terminalFontSize": 15}, 422),
                    ("ui-theme", {"apiKey": "not-a-credential"}, 422),
                    ("ui-conversation", {"busyEnter": "steer"}, 403),
                    ("llm-deepseek", {}, 403),
                ):
                    assert (
                        await client.patch(
                            "/api/v1/agent/settings/" + namespace,
                            json={**current, "values": values},
                        )
                    ).status_code == status
                current_view = (await client.get("/api/v1/agent/settings")).json()
                assert (
                    next(row for row in current_view["namespaces"] if row["ns"] == "ui-theme")[
                        "value"
                    ]["fontSize"]
                    == 16
                )
                locale = next(row for row in current_view["namespaces"] if row["ns"] == "locale")
                selected = await client.patch(
                    "/api/v1/agent/settings/locale",
                    json={
                        "values": {"preference": "zh"},
                        "expected_revision": locale["revision"],
                        "generation": view["generation"],
                    },
                )
                assert selected.status_code == 200
                restored = await client.patch(
                    "/api/v1/agent/settings/locale",
                    json={
                        "mode": "replace",
                        "values": {},
                        "expected_revision": selected.json()["revision"],
                        "generation": view["generation"],
                    },
                )
                assert restored.status_code == 200
                assert "preference" not in restored.json()["value"]
                assert str(tmp_path) not in json.dumps(current_view)
                assert backend.calls == []
        finally:
            await agent.aclose()
        assert agent.settings.harness is None and agent.settings.client is None
        reopened = AgentService(service)
        try:
            snapshot = await reopened.settings.describe()
            assert snapshot["generation"] != view["generation"]
            assert (
                next(row for row in snapshot["namespaces"] if row["ns"] == "ui-theme")["value"][
                    "fontSize"
                ]
                == 16
            )
            assert backend.calls == []
        finally:
            await reopened.aclose()

    asyncio.run(check())
