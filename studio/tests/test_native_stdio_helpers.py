"""Unit tests for real-runtime resource preflight and process cleanup."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from tests import test_native_stdio_e2e as native_e2e
from tests.native_stdio_helpers import (
    MissingNativeRuntimeAssetsError,
    NativeRuntimeAssets,
    NativeStdioSession,
    ProductionStdioSession,
    resolve_native_runtime_assets,
)

_RESOURCE_VARIABLES = (
    "TYLOINFER_STEP_TEST_RUNTIME",
    "TYLOINFER_STEP_TEST_MODEL",
    "TYLOINFER_STEP_TEST_TOKENIZER",
)


def test_native_asset_gate_skips_when_optional(monkeypatch: pytest.MonkeyPatch) -> None:
    for variable in (*_RESOURCE_VARIABLES, "TYLOINFER_REQUIRE_REAL_RUNTIME"):
        monkeypatch.delenv(variable, raising=False)
    with pytest.raises(pytest.skip.Exception, match="missing native test resources"):
        native_e2e.native_runtime_assets.__wrapped__()


def test_native_asset_gate_fails_when_required(monkeypatch: pytest.MonkeyPatch) -> None:
    for variable in _RESOURCE_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("TYLOINFER_REQUIRE_REAL_RUNTIME", "1")
    with pytest.raises(pytest.fail.Exception, match="missing native test resources"):
        native_e2e.native_runtime_assets.__wrapped__()


def test_protocol_extractors_match_native_unary_and_stream_shapes() -> None:
    """Keep the two native stdio result shapes distinct from the C++ protocol."""

    # Mirrors common.cpp runtime_generation_result and stdio.cpp stream completion.
    unary_frames = [
        {
            "id": "fixed-1",
            "type": "event",
            "data": {
                "request_id": "run-1794518295-0",
                "created": 1794518295,
                "model": "mfq-model",
                "output": {"text": "", "reasoning": "Okay,"},
                "finish_reason": "length",
                "usage": {"prompt_tokens": 11, "completion_tokens": 2, "total_tokens": 13},
                "metrics": {"mtp_available": 1.0},
            },
            "v": 1,
        },
        {"id": "fixed-1", "type": "done", "v": 1},
    ]
    stream_frames = [
        {"id": "stream", "type": "event", "data": {"event": "delta", "delta": {"content": "Hi"}}},
        {
            "id": "stream",
            "type": "event",
            "data": {"event": "complete", "finish_reason": "stop", "metrics": {}},
        },
        {"id": "stream", "type": "done", "v": 1},
    ]
    assert native_e2e.unary_result(unary_frames)["output"]["reasoning"] == "Okay,"
    assert native_e2e.stream_completion(stream_frames)["finish_reason"] == "stop"


@pytest.mark.parametrize(("terminal", "late"), [("done", "event"), ("result", "result")])
def test_session_rejects_messages_after_terminal_response(
    tmp_path: Path, terminal: str, late: str
) -> None:
    """A completed request must never produce another response or stream event."""

    session = NativeStdioSession(
        NativeRuntimeAssets(
            runtime=tmp_path / "runtime",
            model=tmp_path / "model",
            tokenizer=tmp_path / "tokenizer",
        ),
        batch_size=0,
        log_path=tmp_path / "runtime.log",
    )
    session._record_frame({"id": "request", "type": terminal})
    with pytest.raises(AssertionError, match="late or duplicate terminal frame"):
        session._record_frame({"id": "request", "type": late, "data": {"output": {}}})


def test_native_asset_preflight_requires_all_explicit_paths(tmp_path: Path) -> None:
    with pytest.raises(MissingNativeRuntimeAssetsError, match="TYLOINFER_STEP_TEST_MODEL"):
        resolve_native_runtime_assets({}, root=tmp_path)


def test_native_asset_preflight_rejects_configured_missing_path(tmp_path: Path) -> None:
    environment = {
        "TYLOINFER_STEP_TEST_RUNTIME": str(tmp_path / "runtime"),
        "TYLOINFER_STEP_TEST_MODEL": str(tmp_path / "model.tq"),
        "TYLOINFER_STEP_TEST_TOKENIZER": str(tmp_path / "tokenizer.gguf"),
    }
    with pytest.raises(FileNotFoundError, match="runtime"):
        resolve_native_runtime_assets(environment, root=tmp_path)


def test_native_asset_preflight_accepts_model_directory_and_resolves_relative_paths(tmp_path: Path) -> None:
    runtime = tmp_path / "bin" / "mfq-runtime"
    tokenizer = tmp_path / "assets" / "tokenizer.gguf"
    model = tmp_path / "assets" / "model"
    runtime.parent.mkdir()
    tokenizer.parent.mkdir()
    model.mkdir()
    runtime.touch()
    tokenizer.touch()
    assets = resolve_native_runtime_assets(
        {
            "TYLOINFER_STEP_TEST_RUNTIME": "bin/mfq-runtime",
            "TYLOINFER_STEP_TEST_MODEL": "assets/model",
            "TYLOINFER_STEP_TEST_TOKENIZER": "assets/tokenizer.gguf",
        },
        root=tmp_path,
    )
    assert assets.runtime == runtime
    assert assets.model == model
    assert assets.tokenizer == tokenizer


class _ProcessThatTimesOut:
    """Minimal process double for the helper's bounded cleanup contract."""

    returncode = None

    def __init__(self) -> None:
        self.terminated = False
        self.killed = False

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    async def wait(self) -> None:
        if not self.killed:
            await asyncio.Future()


class _Stdin:
    """Minimal stdin handle used to verify session-owned cleanup."""

    def __init__(self) -> None:
        self.closed = False

    def is_closing(self) -> bool:
        return self.closed

    def close(self) -> None:
        self.closed = True


async def _real_never_ready_session(tmp_path: Path) -> ProductionStdioSession:
    """Start a live Python child that never emits a native ready frame."""

    log = (tmp_path / "production.log").open("wb")
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import time; time.sleep(60)",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=log,
    )
    session = ProductionStdioSession(
        NativeRuntimeAssets(
            runtime=Path(sys.executable),
            model=tmp_path / "model",
            tokenizer=tmp_path / "tokenizer",
        ),
        batch_size=0,
        log_path=tmp_path / "production.log",
        startup_timeout=0.05,
        client_close_timeout=0.05,
        process_timeout=0.2,
    )
    from server.runtime.client import StdioRuntimeClient

    session.process = process
    session.client = StdioRuntimeClient(process)
    session._log = log
    return session


def test_production_session_real_never_ready_cleanup_is_bounded(tmp_path: Path) -> None:
    async def scenario() -> None:
        session = await _real_never_ready_session(tmp_path)
        assert session.client is not None
        with pytest.raises(asyncio.TimeoutError):
            await session.wait_ready()
        reader_task = session.client._reader_task
        await asyncio.wait_for(session.close(), timeout=3)
        await session.process.wait()  # type: ignore[union-attr]
        assert session.process.returncode is not None  # type: ignore[union-attr]
        assert session.process.stdin.is_closing()  # type: ignore[union-attr]
        assert session._log is not None and session._log.closed
        assert reader_task is not None and reader_task.done()

    asyncio.run(scenario())


def test_production_session_enter_timeout_reaps_real_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A startup failure through ``async with`` must reap the live child."""

    async def scenario() -> None:
        original_spawn = asyncio.create_subprocess_exec

        async def spawn_never_ready(_runtime: str, *args: str, **kwargs: object) -> asyncio.subprocess.Process:
            del args
            return await original_spawn(
                sys.executable,
                "-u",
                "-c",
                "import time; time.sleep(60)",
                **kwargs,
            )

        monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn_never_ready)
        session = ProductionStdioSession(
            NativeRuntimeAssets(
                runtime=tmp_path / "runtime",
                model=tmp_path / "model",
                tokenizer=tmp_path / "tokenizer",
            ),
            batch_size=0,
            log_path=tmp_path / "enter-timeout.log",
            startup_timeout=0.05,
            client_close_timeout=0.05,
            process_timeout=0.2,
        )
        entering = asyncio.create_task(session.__aenter__())
        done, _ = await asyncio.wait({entering}, timeout=3)
        if entering not in done:
            entering.cancel()
            await asyncio.gather(entering, return_exceptions=True)
            pytest.fail("startup cleanup watchdog expired")
        with pytest.raises(asyncio.TimeoutError):
            await entering
        assert session.process is not None
        await session.process.wait()
        assert session.process.returncode is not None
        assert session.process.stdin.is_closing()
        assert session._log is not None and session._log.closed
        assert session.client is not None
        assert session.client._reader_task is not None
        assert session.client._reader_task.done()

    asyncio.run(scenario())


def test_production_session_real_cancelled_close_still_reaps_child(tmp_path: Path) -> None:
    async def scenario() -> None:
        session = await _real_never_ready_session(tmp_path)
        with pytest.raises(asyncio.TimeoutError):
            await session.wait_ready()
        reader_task = session.client._reader_task  # type: ignore[union-attr]
        closing = asyncio.create_task(session.close())
        await asyncio.sleep(0)
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(closing, timeout=3)
        await session.process.wait()  # type: ignore[union-attr]
        assert session.process.returncode is not None  # type: ignore[union-attr]
        assert session.process.stdin.is_closing()  # type: ignore[union-attr]
        assert session._log is not None and session._log.closed
        assert reader_task is not None and reader_task.done()

    asyncio.run(scenario())


def test_session_close_reuses_bounded_process_cleanup(tmp_path: Path) -> None:
    async def scenario() -> None:
        process = _ProcessThatTimesOut()
        process.stdin = _Stdin()
        log = (tmp_path / "runtime.log").open("wb")
        session = NativeStdioSession(
            NativeRuntimeAssets(
                runtime=tmp_path / "runtime",
                model=tmp_path / "model",
                tokenizer=tmp_path / "tokenizer",
            ),
            batch_size=0,
            log_path=tmp_path / "runtime.log",
        )
        session.process = process
        session._log = log
        await session.close()
        assert process.terminated
        assert process.killed
        assert process.stdin.closed
        assert log.closed

    asyncio.run(scenario())


def test_until_event_rejects_cached_terminal_frames(tmp_path: Path) -> None:
    async def scenario() -> None:
        session = NativeStdioSession(
            NativeRuntimeAssets(
                runtime=tmp_path / "runtime",
                model=tmp_path / "model",
                tokenizer=tmp_path / "tokenizer",
            ),
            batch_size=0,
            log_path=tmp_path / "runtime.log",
        )
        session.frames["done"] = [{"id": "done", "type": "done"}]
        with pytest.raises(AssertionError, match="completed without matching active event"):
            await session.until_event("done", lambda data: data.get("event") == "delta")
        with pytest.raises(AssertionError, match="completed without matching event"):
            await session.until_event(
                "done", lambda data: data.get("event") == "delta", require_active=False
            )
        session.frames["error"] = [{"id": "error", "type": "error", "error": {"status_code": 500}}]
        with pytest.raises(AssertionError, match="failed"):
            await session.until_event("error", lambda data: True)

    asyncio.run(scenario())


def test_until_rejects_cached_done_when_waiting_for_result(tmp_path: Path) -> None:
    async def scenario() -> None:
        session = NativeStdioSession(
            NativeRuntimeAssets(
                runtime=tmp_path / "runtime",
                model=tmp_path / "model",
                tokenizer=tmp_path / "tokenizer",
            ),
            batch_size=0,
            log_path=tmp_path / "runtime.log",
        )
        session.frames["request"] = [{"id": "request", "type": "done"}]
        with pytest.raises(AssertionError, match="completed before 'result'"):
            await session.until("request", "result")

    asyncio.run(scenario())


def test_until_event_can_query_cached_history_when_not_active(tmp_path: Path) -> None:
    async def scenario() -> None:
        session = NativeStdioSession(
            NativeRuntimeAssets(
                runtime=tmp_path / "runtime",
                model=tmp_path / "model",
                tokenizer=tmp_path / "tokenizer",
            ),
            batch_size=0,
            log_path=tmp_path / "runtime.log",
        )
        session.frames["request"] = [
            {"id": "request", "type": "event", "data": {"event": "delta"}},
            {"id": "request", "type": "done"},
        ]
        frames = await session.until_event(
            "request", lambda data: data.get("event") == "delta", require_active=False
        )
        assert frames[-1]["type"] == "done"

    asyncio.run(scenario())
