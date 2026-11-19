"""Persistent Agent sessions, bounded runs, and replayable execution events."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from tyloinfer.server.services.service import ServiceError

TERMINAL = frozenset({"completed", "cancelled", "failed", "interrupted"})
MAX_OUTPUT_BYTES = 262144
MAX_EVENTS = 10000


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AgentStore:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS agent_sessions (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL, model TEXT NOT NULL,
                    runtime TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    last_run_id TEXT);
                CREATE TABLE IF NOT EXISTS agent_runs (
                    id TEXT PRIMARY KEY, session_id TEXT NOT NULL, request_id TEXT NOT NULL,
                    status TEXT NOT NULL, input TEXT NOT NULL, output TEXT NOT NULL DEFAULT '',
                    error TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    last_event_id INTEGER NOT NULL DEFAULT 0, UNIQUE(session_id, request_id));
                CREATE TABLE IF NOT EXISTS agent_events (
                    run_id TEXT NOT NULL, event_id INTEGER NOT NULL, session_id TEXT NOT NULL,
                    type TEXT NOT NULL, data TEXT NOT NULL, PRIMARY KEY(run_id, event_id));
            """)

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=5)
        try:
            with db:
                db.row_factory = sqlite3.Row
                yield db
        finally:
            db.close()

    @staticmethod
    def resource(row) -> dict[str, Any]:
        result = dict(row)
        if "error" in result:
            result["error"] = json.loads(result["error"]) if result["error"] else None
        return result

    def sessions(self):
        with self.connection() as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT * FROM agent_sessions ORDER BY updated_at DESC LIMIT 100"
                )
            ]

    def session(self, session_id: str, *, runs=False):
        with self.connection() as db:
            row = db.execute("SELECT * FROM agent_sessions WHERE id=?", (session_id,)).fetchone()
            if row is None:
                raise ServiceError(404, "agent_session_not_found", "Agent session not found")
            result = dict(row)
            if runs:
                result["runs"] = list(
                    reversed(
                        [
                            self.resource(item)
                            for item in db.execute(
                                "SELECT * FROM agent_runs WHERE session_id=? ORDER BY created_at DESC, rowid DESC LIMIT 100",
                                (session_id,),
                            )
                        ]
                    )
                )
            return result

    def create_session(self, title: str, model: str):
        identifier, timestamp = str(uuid4()), now()
        with self.connection() as db:
            db.execute(
                "INSERT INTO agent_sessions VALUES (?,?,?,?,?,?,NULL)",
                (identifier, title, model, "dsh", timestamp, timestamp),
            )
        return self.session(identifier)

    def run(self, run_id: str):
        with self.connection() as db:
            row = db.execute("SELECT * FROM agent_runs WHERE id=?", (run_id,)).fetchone()
            if row is None:
                raise ServiceError(404, "agent_run_not_found", "Agent run not found")
            return self.resource(row)

    def existing(self, session_id: str, request_id: str, text: str):
        with self.connection() as db:
            row = db.execute(
                "SELECT * FROM agent_runs WHERE session_id=? AND request_id=?",
                (session_id, request_id),
            ).fetchone()
            if row is not None:
                if row["input"] != text:
                    raise ServiceError(
                        409, "idempotency_conflict", "Request ID already has different input"
                    )
                return self.resource(row)
        return None

    def begin(self, session_id: str, request_id: str, text: str):
        self.session(session_id)
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM agent_runs WHERE session_id=? AND request_id=?",
                (session_id, request_id),
            ).fetchone()
            if row:
                if row["input"] != text:
                    raise ServiceError(
                        409, "idempotency_conflict", "Request ID already has different input"
                    )
                return self.resource(row), False
            if db.execute(
                "SELECT 1 FROM agent_runs WHERE session_id=? AND status NOT IN "
                "('completed','cancelled','failed','interrupted')",
                (session_id,),
            ).fetchone():
                raise ServiceError(
                    409, "agent_session_busy", "Wait for the active Agent run to settle"
                )
            identifier, timestamp = str(uuid4()), now()
            db.execute(
                "INSERT INTO agent_runs (id,session_id,request_id,status,input,created_at,updated_at) "
                "VALUES (?,?,?,'queued',?,?,?)",
                (identifier, session_id, request_id, text, timestamp, timestamp),
            )
            db.execute(
                "UPDATE agent_sessions SET last_run_id=?,updated_at=? WHERE id=?",
                (identifier, timestamp, session_id),
            )
        self.event(identifier, "run.status", {"status": "queued"})
        return self.run(identifier), True

    def event(self, run_id: str, kind: str, data: dict[str, Any]):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM agent_runs WHERE id=?", (run_id,)).fetchone()
            if row is None or row["status"] in TERMINAL:
                return
            number = row["last_event_id"] + 1
            closing = kind == "error" or (kind == "run.status" and data.get("status") in TERMINAL)
            if number > MAX_EVENTS + (2 if closing else 0):
                raise ServiceError(413, "agent_event_limit", "Agent event limit reached")
            output = row["output"] + (data.get("delta", "") if kind == "text.delta" else "")
            if len(output.encode("utf-8")) > MAX_OUTPUT_BYTES:
                raise ServiceError(413, "agent_output_limit", "Agent output limit reached")
            status = data["status"] if kind == "run.status" else row["status"]
            error = json.dumps(data) if kind == "error" else row["error"]
            timestamp = now()
            db.execute(
                "INSERT INTO agent_events VALUES (?,?,?,?,?)",
                (run_id, number, row["session_id"], kind, json.dumps(data, ensure_ascii=False)),
            )
            db.execute(
                "UPDATE agent_runs SET last_event_id=?,status=?,output=?,error=?,updated_at=? WHERE id=?",
                (number, status, output, error, timestamp, run_id),
            )
            db.execute(
                "UPDATE agent_sessions SET updated_at=? WHERE id=?", (timestamp, row["session_id"])
            )

    def events(self, run_id: str, after: int):
        self.run(run_id)
        with self.connection() as db:
            return [
                {**dict(row), "data": json.loads(row["data"])}
                for row in db.execute(
                    "SELECT * FROM agent_events WHERE run_id=? AND event_id>? ORDER BY event_id LIMIT 256",
                    (run_id, after),
                )
            ]

    def recover(self):
        with self.connection() as db:
            pending = [
                row[0]
                for row in db.execute(
                    "SELECT id FROM agent_runs WHERE status NOT IN ('completed','cancelled','failed','interrupted')"
                )
            ]
        for identifier in pending:
            self.event(
                identifier,
                "error",
                {"code": "agent_interrupted", "message": "Server stopped before the run settled"},
            )
            self.event(identifier, "run.status", {"status": "interrupted"})


class AgentService:
    def __init__(self, service, *, runtime_factory=None, max_concurrent_runs=4):
        from .dsh import DshRuntime
        from .settings import NativeSettings

        self.service = service
        self.store = AgentStore(service.store.path.with_name("agent-workspace.sqlite3"))
        self.root = service.store.path.parent / "agent-runtime"
        self.settings = NativeSettings(self.root / "settings")
        self.runtime_factory = runtime_factory or DshRuntime
        self.tasks: dict[str, asyncio.Task] = {}
        self.runtimes: dict[str, Any] = {}
        self.signals: dict[str, asyncio.Event] = {}
        self.slots = asyncio.Semaphore(max_concurrent_runs)
        self.store.recover()
        self.closed = False

    def capabilities(self):
        return {
            **self.runtime_factory.capabilities(),
            "runtime": "dsh",
            "tools": ["local_inference"],
            "permissions": False,
            "multi_agent": False,
            "event_replay": True,
            "persistence": True,
        }

    def start(self, session_id: str, request_id: str, text: str):
        self.store.session(session_id)
        existing = self.store.existing(session_id, request_id, text)
        if existing is not None:
            return existing
        if self.closed:
            raise ServiceError(503, "agent_stopping", "Agent service is stopping")
        if not self.capabilities()["available"]:
            raise ServiceError(503, "agent_runtime_unavailable", self.capabilities()["reason"])
        if len(self.tasks) >= 32:
            raise ServiceError(429, "agent_queue_full", "Agent run queue is full", retryable=True)
        run, created = self.store.begin(session_id, request_id, text)
        if created:
            self.signals[run["id"]] = asyncio.Event()
            task = asyncio.create_task(self._execute(run), name="tyloinfer-agent-run")
            self.tasks[run["id"]] = task
        return run

    def emit(self, run_id, kind, data):
        self.store.event(run_id, kind, data)
        signal = self.signals.get(run_id)
        if signal is not None:
            signal.set()

    async def _execute(self, run):
        identifier = run["id"]
        runtime = None
        try:
            async with self.slots:
                runtime = self.runtime_factory()
                self.runtimes[identifier] = runtime
                self.emit(identifier, "run.status", {"status": "running"})
                session = self.store.session(run["session_id"])
                await runtime.run(
                    session=session,
                    run=run,
                    root=self.root,
                    service=self.service,
                    emit=lambda kind, data: self.emit(identifier, kind, data),
                )
                status = (
                    "cancelled"
                    if self.store.run(identifier)["status"] == "cancelling"
                    else "completed"
                )
                self.emit(identifier, "run.status", {"status": status})
        except asyncio.CancelledError:
            self.emit(identifier, "run.status", {"status": "cancelled"})
        except Exception:
            if self.store.run(identifier)["status"] == "cancelling":
                self.emit(identifier, "run.status", {"status": "cancelled"})
            else:
                self.emit(
                    identifier,
                    "error",
                    {
                        "code": "agent_execution_failed",
                        "message": "Agent runtime could not complete the task",
                    },
                )
                self.emit(identifier, "run.status", {"status": "failed"})
        finally:
            self.runtimes.pop(identifier, None)
            self.tasks.pop(identifier, None)
            signal = self.signals.pop(identifier, None)
            if signal is not None:
                signal.set()

    async def cancel(self, run_id: str):
        run = self.store.run(run_id)
        if run["status"] in TERMINAL or run["status"] == "cancelling":
            return run
        self.emit(run_id, "run.status", {"status": "cancelling"})
        runtime = self.runtimes.get(run_id)
        if runtime is not None:
            await runtime.cancel()
        task = self.tasks.get(run_id)
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if self.store.run(run_id)["status"] not in TERMINAL:
            self.emit(run_id, "run.status", {"status": "cancelled"})
        self.tasks.pop(run_id, None)
        self.runtimes.pop(run_id, None)
        signal = self.signals.pop(run_id, None)
        if signal is not None:
            signal.set()
        return self.store.run(run_id)

    async def events(self, run_id, after):
        signal = self.signals.setdefault(run_id, asyncio.Event())
        while True:
            signal.clear()
            batch = self.store.events(run_id, after)
            for event in batch:
                after = event["event_id"]
                yield event
            if batch:
                continue
            if self.store.run(run_id)["status"] in TERMINAL:
                self.signals.pop(run_id, None)
                return
            try:
                await asyncio.wait_for(signal.wait(), timeout=15)
            except asyncio.TimeoutError:
                yield None

    async def aclose(self):
        self.closed = True
        await asyncio.gather(
            *(self.cancel(key) for key in list(self.tasks)), return_exceptions=True
        )
        await asyncio.gather(*list(self.tasks.values()), return_exceptions=True)
        await self.settings.aclose()
