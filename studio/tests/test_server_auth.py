from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import httpx

from server.api import create_app
from server.api.auth import ApiKeyManager, hash_api_key
from server.protocol.models import CreateApiKeyRequest
from server.state.storage import SessionStore


def test_scoped_keys_rotate_revoke_and_never_persist_plaintext(tmp_path: Path) -> None:
    async def run() -> None:
        database = tmp_path / "server.sqlite3"
        store = SessionStore(database)
        manager = ApiKeyManager(store, "root-secret")
        app = create_app(api_keys=manager)
        transport = httpx.ASGITransport(app=app)
        root = {"Authorization": "Bearer root-secret"}
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            created = await client.post(
                "/api/v1/auth/keys",
                headers=root,
                json={"name": "jobs", "scopes": ["jobs"]},
            )
            assert created.status_code == 201, created.text
            body = created.json()
            token = body["token"]
            key_id = body["key"]["id"]
            assert token.startswith("mfq_")

            job_key = {"Authorization": f"Bearer {token}"}
            assert (await client.get("/api/v1/jobs", headers=job_key)).status_code == 501
            assert (await client.get("/api/v1/sessions", headers=job_key)).status_code == 403
            assert (await client.get("/api/v1/auth/keys", headers=job_key)).status_code == 403
            desktop_denied = await client.get("/api/v1/auth/keys", headers={**job_key, "Origin": "tauri://localhost"})
            assert desktop_denied.status_code == 403
            assert desktop_denied.headers["access-control-allow-origin"] == "tauri://localhost"

            operator = await client.post(
                "/api/v1/auth/keys",
                headers=root,
                json={"name": "operator", "role": "operator"},
            )
            assert operator.status_code == 201
            operator_header = {"Authorization": f"Bearer {operator.json()['token']}"}
            assert (await client.get("/api/v1/jobs", headers=operator_header)).status_code == 501
            assert (
                await client.get("/api/v1/models/directories", headers=operator_header)
            ).status_code == 501
            assert (
                await client.get("/api/v1/auth/keys", headers=operator_header)
            ).status_code == 403

            viewer = await client.post(
                "/api/v1/auth/keys",
                headers=root,
                json={"name": "viewer", "role": "viewer"},
            )
            viewer_header = {"Authorization": f"Bearer {viewer.json()['token']}"}
            assert (
                await client.get("/api/v1/models/directories", headers=viewer_header)
            ).status_code == 403
            assert (await client.post("/api/v1/models/directories/open", headers=viewer_header,
                json={"directory_id": "0" * 32})).status_code == 403

            rotated = await client.post(f"/api/v1/auth/keys/{key_id}/rotate", headers=root)
            assert rotated.status_code == 200
            replacement = rotated.json()["token"]
            assert replacement != token
            assert (await client.get("/api/v1/jobs", headers=job_key)).status_code == 401
            replacement_header = {"Authorization": f"Bearer {replacement}"}
            assert (await client.get("/api/v1/jobs", headers=replacement_header)).status_code == 501

            revoked = await client.post(f"/api/v1/auth/keys/{key_id}/revoke", headers=root)
            assert revoked.status_code == 200
            assert (await client.get("/api/v1/jobs", headers=replacement_header)).status_code == 401

        database_bytes = database.read_bytes()
        assert token.encode() not in database_bytes
        assert replacement.encode() not in database_bytes
        with store._connection() as connection:
            row = connection.execute(
                "SELECT key_hash FROM api_keys WHERE id = ?", (key_id,)
            ).fetchone()
        assert row["key_hash"] == hash_api_key(replacement)

    asyncio.run(run())


def test_scoped_key_lookup_does_not_block_the_event_loop(tmp_path: Path) -> None:
    """A key lookup writes last_used_at; that write must not stall the event loop."""

    async def run() -> None:
        store = SessionStore(tmp_path / "server.sqlite3")
        manager = ApiKeyManager(store, "root-secret")
        token = manager.create(CreateApiKeyRequest(name="slow", scopes=["inference"])).token
        entered, release, returned = threading.Event(), threading.Event(), threading.Event()
        lookup = store.authenticate_api_key

        def slow_lookup(key_hash: str, **kwargs: object):
            result = lookup(key_hash, **kwargs)
            entered.set()
            release.wait(5)
            returned.set()
            return result

        store.authenticate_api_key = slow_lookup
        answered = asyncio.Event()
        app = create_app(api_keys=manager)

        async def health(client: httpx.AsyncClient) -> None:
            assert (await client.get("/health")).status_code == 200
            answered.set()

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            health_task = asyncio.create_task(health(client))
            authed = asyncio.create_task(
                client.get("/v1/models", headers={"Authorization": f"Bearer {token}"})
            )
            assert await asyncio.to_thread(entered.wait, 5)
            # /health has to be served while the credential write is still blocked.
            await asyncio.wait_for(answered.wait(), 1)
            assert not returned.is_set()
            release.set()
            # 501 comes from the contract-only app, so the request passed authentication.
            assert (await authed).status_code == 501
            await health_task

    asyncio.run(run())


def test_store_reopen_preserves_api_keys_and_jobs(tmp_path: Path) -> None:
    """Persisted credentials and jobs remain usable after reopening the database."""
    database = tmp_path / "server.sqlite3"
    store = SessionStore(database)
    secret = ApiKeyManager(store).create(
        CreateApiKeyRequest(name="jobs", scopes=["jobs"])
    )
    job = store.create_job("test.job", {"model": "tiny"})

    reopened = SessionStore(database)
    authenticated = ApiKeyManager(reopened).authenticate(secret.token, include_root=False)
    assert authenticated is not None
    assert authenticated.resource is not None
    assert authenticated.resource.id == secret.key.id
    assert authenticated.scopes == frozenset({"jobs"})
    restored = reopened.get_job(job.id)
    assert restored.kind == "test.job"
    assert restored.payload == {"model": "tiny"}
