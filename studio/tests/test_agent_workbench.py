"""Agent API lifecycle tests and an opt-in real DSH process integration test."""

from __future__ import annotations

import asyncio
import json
import os
from uuid import uuid4

import httpx
import pytest

from tests.legacy_agent_support import AgentService, AgentStore, create_app
from tests.test_server_service import FakeBackend, make_service
from server.runtime.backend import BackendDelta, BackendError, BackendToolCallDelta


class Runtime:
    @staticmethod
    def capabilities():
        return {"available": True, "version": "test", "reason": None}

    async def run(self, *, emit, **_kwargs):
        emit(
            "tool.started",
            {"tool_call_id": "call-1", "name": "local_inference", "status": "running"},
        )
        emit(
            "tool.completed",
            {
                "tool_call_id": "call-1",
                "name": "local_inference",
                "status": "completed",
                "output": "ready",
                "is_error": False,
            },
        )
        emit("text.delta", {"delta": "ready"})

    async def cancel(self):
        return None


class BlockingRuntime(Runtime):
    async def run(self, **_kwargs):
        await asyncio.Event().wait()


def setup(tmp_path, runtime=Runtime):
    service = make_service(tmp_path, FakeBackend())
    app = create_app(service)
    app.state.agent = AgentService(service, runtime_factory=runtime)
    return app


async def wait_for_run(agent, run_id):
    async for _ in agent.events(run_id, 0):
        pass
    return agent.store.run(run_id)


def test_run_replay_and_idempotency(tmp_path):
    async def check():
        app = setup(tmp_path)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as client:
            session = (
                await client.post(
                    "/api/v1/agent/sessions", json={"title": "Task", "model": "model-a"}
                )
            ).json()
            body = {"request_id": str(uuid4()), "input": "Run a local task"}
            path = f"/api/v1/agent/sessions/{session['id']}/runs"
            response = await client.post(path, json=body)
            assert response.status_code == 202
            run = await wait_for_run(app.state.agent, response.json()["id"])
            assert run["status"] == "completed" and run["output"] == "ready"
            duplicate = await client.post(path, json=body)
            assert duplicate.json()["id"] == run["id"]
            conflict = await client.post(path, json={**body, "input": "Different task"})
            assert conflict.status_code == 409
            events = await client.get(f"/api/v1/agent/runs/{run['id']}/events?after=2")
            assert events.status_code == 200
            assert "event: tool.started" in events.text
            assert '"event_id": 1,' not in events.text
            restored = (await client.get(f"/api/v1/agent/sessions/{session['id']}")).json()
            assert restored["runs"][0]["output"] == "ready"
            assert len((await client.get("/api/v1/agent/sessions")).json()["data"]) == 1
        await app.state.agent.aclose()

    asyncio.run(check())


def test_busy_cancel_and_invalid_cursor(tmp_path):
    async def check():
        app = setup(tmp_path, BlockingRuntime)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as client:
            session = (
                await client.post("/api/v1/agent/sessions", json={"model": "model-a"})
            ).json()
            path = f"/api/v1/agent/sessions/{session['id']}/runs"
            run = (
                await client.post(path, json={"request_id": str(uuid4()), "input": "Wait"})
            ).json()
            busy = await client.post(path, json={"request_id": str(uuid4()), "input": "Other task"})
            assert busy.status_code == 409
            rejected = await client.get(f"/api/v1/agent/runs/{run['id']}/events?after=999")
            assert rejected.status_code == 400
            cancelled = await client.post(f"/api/v1/agent/runs/{run['id']}/cancel")
            assert cancelled.json()["status"] == "cancelled"
            assert (await client.post(f"/api/v1/agent/runs/{run['id']}/cancel")).json()[
                "status"
            ] == "cancelled"
        await app.state.agent.aclose()

    asyncio.run(check())


def test_restart_marks_unfinished_runs_without_reexecution(tmp_path):
    store = AgentStore(tmp_path / "agent.sqlite3")
    session = store.create_session("Task", "model-a")
    run, _ = store.begin(session["id"], str(uuid4()), "Wait")
    reopened = AgentStore(store.path)
    reopened.recover()
    assert reopened.run(run["id"])["status"] == "interrupted"
    assert reopened.events(run["id"], 0)[-1]["data"]["status"] == "interrupted"


def test_authentication_and_input_validation(tmp_path):
    async def check():
        app = setup(tmp_path)
        app.state.auth_policy._configuration = (True, "not-a-credential")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as client:
            assert (await client.get("/api/v1/agent/sessions")).status_code == 401
        app.state.auth_policy._configuration = (False, "")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as client:
            invalid = await client.post(
                "/api/v1/agent/sessions", json={"model": " ", "endpoint": "invalid"}
            )
            assert invalid.status_code == 422
        await app.state.agent.aclose()

    asyncio.run(check())


def test_recent_history_keeps_latest_run(tmp_path):
    store = AgentStore(tmp_path / "agent.sqlite3")
    session = store.create_session("Task", "model-a")
    identifiers = []
    for index in range(101):
        run, _ = store.begin(session["id"], str(uuid4()), str(index))
        identifiers.append(run["id"])
        store.event(run["id"], "run.status", {"status": "completed"})
    runs = store.session(session["id"], runs=True)["runs"]
    assert [run["id"] for run in runs] == identifiers[1:]
    assert store.session(session["id"])["last_run_id"] == runs[-1]["id"]


def test_existing_run_survives_runtime_unavailability(tmp_path):
    class Unavailable(Runtime):
        @staticmethod
        def capabilities():
            return {"available": False, "version": None, "reason": "Runtime unavailable"}

    async def check():
        app = setup(tmp_path)
        agent = app.state.agent
        session = agent.store.create_session("Task", "model-a")
        request = str(uuid4())
        run = agent.start(session["id"], request, "Task")
        await wait_for_run(agent, run["id"])
        agent.runtime_factory = Unavailable
        assert agent.start(session["id"], request, "Task")["id"] == run["id"]
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as client:
            response = await client.post(
                f"/api/v1/agent/sessions/{session['id']}/runs",
                json={"request_id": str(uuid4()), "input": "Other task"},
            )
            assert response.status_code == 503
        await agent.aclose()

    asyncio.run(check())


def test_failure_and_event_limit_never_report_success(tmp_path, monkeypatch):
    monkeypatch.setattr("tyloinfer_legacy_agent.service.MAX_EVENTS", 3)

    async def check():
        app = setup(tmp_path)
        agent = app.state.agent
        session = agent.store.create_session("Task", "model-a")
        run = agent.start(session["id"], str(uuid4()), "Task")
        result = await wait_for_run(agent, run["id"])
        assert result["status"] == "failed"
        assert result["error"]["code"] == "agent_execution_failed"
        assert agent.store.events(run["id"], 0)[-1]["data"]["status"] == "failed"
        await agent.aclose()

    asyncio.run(check())


def test_local_tool_refuses_extra_authority_and_invalid_limits(tmp_path):
    from tyloinfer_legacy_agent.dsh import LocalBridge

    async def check():
        backend = FakeBackend()
        bridge = LocalBridge(
            make_service(tmp_path, backend),
            {"model": "model-a"},
            {"id": str(uuid4())},
            lambda *_: None,
        )
        try:
            for arguments in (
                {"prompt": "Task", "url": "invalid"},
                {"prompt": "Task", "max_tokens": True},
                {"prompt": "Task", "max_tokens": 4097},
                {"prompt": " "},
            ):
                with pytest.raises(ValueError):
                    await bridge.tool({"tool_call_id": "call-1", "arguments": arguments})
            async with httpx.AsyncClient(trust_env=False) as client:
                response = await client.post(
                    bridge.url + "/tool", json={"arguments": {"prompt": "Task"}}
                )
                assert response.status_code == 403
            assert backend.calls == []
        finally:
            await bridge.close()

    asyncio.run(check())


def test_scoped_inference_key_cannot_access_agent(tmp_path):
    from server.api.auth import ApiKeyManager
    from server.protocol.models import CreateApiKeyRequest

    async def check():
        service = make_service(tmp_path, FakeBackend())
        keys = ApiKeyManager(service.store)
        issued = keys.create(CreateApiKeyRequest(name="reader", scopes=["inference"]))
        app = create_app(service, api_key="test-root", api_keys=keys)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app),
            base_url="http://test",
            headers={"Authorization": "Bearer " + issued.token},
        ) as client:
            assert (await client.get("/api/v1/agent/sessions")).status_code == 403
        if app.state.agent is not None:
            await app.state.agent.aclose()

    asyncio.run(check())


class ScriptedBackend(FakeBackend):
    async def stream(self, **kwargs):
        self.calls.append(kwargs)
        tools = kwargs.get("tools", ())
        if not tools:
            yield BackendDelta(content_delta="ready", finish_reason="stop")
        elif any(item.get("role") == "tool" for item in kwargs["messages"]):
            yield BackendDelta(content_delta="Local task finished.", finish_reason="stop")
        else:
            assert [tool.function.name for tool in tools] == ["local_inference"]
            yield BackendDelta(
                tool_calls=(
                    BackendToolCallDelta(
                        index=0,
                        call_id="call-local",
                        name="local_inference",
                        arguments_delta=json.dumps({"prompt": "Return ready", "max_tokens": 16}),
                    ),
                ),
                finish_reason="tool_calls",
            )


@pytest.mark.skipif(
    os.environ.get("TYLOINFER_TEST_DSH") != "1",
    reason="requires the pinned optional DSH SDK runtime",
)
def test_real_dsh_process_calls_local_inference_tool(tmp_path):
    async def check():
        backend = ScriptedBackend()
        service = make_service(tmp_path, backend)
        app = create_app(service)
        app.state.agent = AgentService(service)
        agent = app.state.agent
        assert agent.capabilities()["available"]
        session = agent.store.create_session("Local task", "model-a")
        run = agent.start(
            session["id"], str(uuid4()), "Use local_inference, then summarize its result."
        )
        result = await asyncio.wait_for(wait_for_run(agent, run["id"]), timeout=60)
        assert result["status"] == "completed", result
        assert result["output"] == "Local task finished."
        assert len(backend.calls) == 3
        events = agent.store.events(run["id"], 0)
        assert any(
            event["type"] == "tool.completed" and event["data"]["output"] == "ready"
            for event in events
        )
        continued = agent.start(session["id"], str(uuid4()), "Summarize the previous result.")
        assert (await asyncio.wait_for(wait_for_run(agent, continued["id"]), 60))[
            "status"
        ] == "completed"
        assert len(backend.calls) == 4
        assert any(item.get("role") == "tool" for item in backend.calls[-1]["messages"])
        await agent.aclose()

    asyncio.run(check())


@pytest.mark.skipif(
    os.environ.get("TYLOINFER_TEST_DSH") != "1",
    reason="requires the pinned optional DSH SDK runtime",
)
def test_real_dsh_cancel_releases_local_model_work(tmp_path):
    class WaitingBackend(FakeBackend):
        def __init__(self):
            super().__init__()
            self.entered = asyncio.Event()
            self.released = asyncio.Event()

        async def stream(self, **kwargs):
            self.entered.set()
            try:
                yield BackendDelta(content_delta="Waiting")
                await asyncio.Event().wait()
            finally:
                self.released.set()

    async def check():
        backend = WaitingBackend()
        service = make_service(tmp_path, backend)
        app = create_app(service)
        app.state.agent = AgentService(service)
        agent = app.state.agent
        session = agent.store.create_session("Wait", "model-a")
        run = agent.start(session["id"], str(uuid4()), "Wait for cancellation")
        await asyncio.wait_for(backend.entered.wait(), 30)
        result = await asyncio.wait_for(agent.cancel(run["id"]), 20)
        assert result["status"] == "cancelled"
        assert backend.released.is_set()
        assert not agent.tasks and not agent.runtimes
        await agent.aclose()

    asyncio.run(check())


@pytest.mark.skipif(
    os.environ.get("TYLOINFER_TEST_DSH") != "1",
    reason="requires the pinned optional DSH SDK runtime",
)
@pytest.mark.parametrize("failure_stage", ["planner", "tool"])
def test_real_dsh_backend_failure_never_reports_success(tmp_path, failure_stage):
    class FailingBackend(ScriptedBackend):
        async def stream(self, **kwargs):
            if bool(kwargs.get("tools")) == (failure_stage == "planner"):
                raise BackendError("backend_failure", "Internal-only failure detail")
            async for delta in super().stream(**kwargs):
                yield delta

    async def check():
        agent = AgentService(make_service(tmp_path, FailingBackend()))
        try:
            session = agent.store.create_session("Failure", "model-a")
            run = agent.start(session["id"], str(uuid4()), "Perform a bounded local task")
            result = await asyncio.wait_for(wait_for_run(agent, run["id"]), 60)
            assert result["status"] == "failed"
            assert result["error"] == {
                "code": "agent_execution_failed",
                "message": "Agent runtime could not complete the task",
            }
            events = agent.store.events(run["id"], 0)
            assert events[-1]["data"]["status"] == "failed"
            assert "Internal-only failure detail" not in json.dumps([result, events])
            if failure_stage == "tool":
                assert any(
                    event["type"] == "tool.completed" and event["data"]["is_error"]
                    for event in events
                )
            assert not agent.tasks and not agent.runtimes
        finally:
            await agent.aclose()

    asyncio.run(check())


def test_concurrent_first_requests_share_one_agent_service(tmp_path, monkeypatch):
    from tyloinfer_legacy_agent import routes

    created = []

    def create(service):
        instance = AgentService(service, runtime_factory=Runtime)
        created.append(instance)
        return instance

    monkeypatch.setattr(routes, "AgentService", create)

    async def check():
        app = create_app(make_service(tmp_path, FakeBackend()))
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as client:
            responses = await asyncio.gather(
                *(client.get("/api/v1/agent/capabilities") for _ in range(12))
            )
            assert all(response.status_code == 200 for response in responses)
            assert len(created) == 1
        await app.state.agent.aclose()

    asyncio.run(check())


def test_concurrent_cancellation_closes_runtime_once(tmp_path):
    async def check():
        started, cancelling, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

        class CountingRuntime(Runtime):
            calls = 0

            async def run(self, **_kwargs):
                started.set()
                await asyncio.Event().wait()

            async def cancel(self):
                self.calls += 1
                cancelling.set()
                await release.wait()

        app = setup(tmp_path, CountingRuntime)
        agent = app.state.agent
        session = agent.store.create_session("Wait", "model-a")
        run = agent.start(session["id"], str(uuid4()), "Wait")
        await asyncio.wait_for(started.wait(), 5)
        runtime = agent.runtimes[run["id"]]
        first = asyncio.create_task(agent.cancel(run["id"]))
        try:
            await asyncio.wait_for(cancelling.wait(), 5)
            assert (await agent.cancel(run["id"]))["status"] == "cancelling"
            assert runtime.calls == 1
        finally:
            release.set()
            result = await asyncio.wait_for(first, 5)
            await agent.aclose()
        assert result["status"] == "cancelled"
        assert not agent.tasks and not agent.runtimes

    asyncio.run(check())
