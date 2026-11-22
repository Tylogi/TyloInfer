"""Helpers for opt-in tests that exercise the real native stdio runtime."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import struct
import time
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from server.runtime.client import StdioRuntimeClient


class MissingNativeRuntimeAssetsError(RuntimeError):
    """Raised when the real-runtime test gate is not configured."""


@dataclass(frozen=True)
class NativeRuntimeAssets:
    """Paths required to start a real native stdio runtime."""

    runtime: Path
    model: Path
    tokenizer: Path


class _StdoutWiretap:
    """Observe raw native frames while leaving production client parsing intact."""

    def __init__(
        self,
        reader: Any,
        frames: list[dict[str, Any]],
        changed: asyncio.Event,
    ) -> None:
        self._reader = reader
        self._frames = frames
        self._changed = changed

    async def readline(self) -> bytes:
        line = await self._reader.readline()
        if line:
            self._frames.append(json.loads(line))
            self._changed.set()
        return line


class _StdinWiretap:
    """Observe encoded outbound frames without changing production writes."""

    def __init__(
        self,
        writer: Any,
        frames: list[dict[str, Any]],
        changed: asyncio.Event,
    ) -> None:
        self._writer = writer
        self._frames = frames
        self._changed = changed

    def write(self, data: bytes) -> None:
        self._writer.write(data)
        for line in data.splitlines():
            self._frames.append(json.loads(line))
            self._changed.set()

    async def drain(self) -> None:
        await self._writer.drain()

    def is_closing(self) -> bool:
        return self._writer.is_closing()

    def close(self) -> None:
        self._writer.close()

    async def wait_closed(self) -> None:
        await self._writer.wait_closed()


@dataclass
class ProductionStdioSession:
    """Own a native process through the production ``StdioRuntimeClient``."""

    assets: NativeRuntimeAssets
    batch_size: int
    log_path: Path
    context_size: int = 256
    process: asyncio.subprocess.Process | None = None
    client: StdioRuntimeClient | None = None
    startup_timeout: float = 60.0
    client_close_timeout: float = 5.0
    process_timeout: float = 5.0
    _log: Any = None
    outbound_frames: list[dict[str, Any]] = field(default_factory=list)
    inbound_frames: list[dict[str, Any]] = field(default_factory=list)
    inbound_changed: asyncio.Event = field(default_factory=asyncio.Event)
    outbound_changed: asyncio.Event = field(default_factory=asyncio.Event)

    async def __aenter__(self) -> ProductionStdioSession:
        self._log = self.log_path.open("wb")
        try:
            self.process = await asyncio.create_subprocess_exec(
                str(self.assets.runtime),
                "--model",
                str(self.assets.model),
                "--tokenizer",
                str(self.assets.tokenizer),
                "--transport",
                "stdio",
                "--ctx-size",
                str(self.context_size),
                "--prefill-chunk-size",
                "8",
                "--continuous-batching",
                str(self.batch_size),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=self._log,
                env={**os.environ, "TYLOINFER_RUNTIME_DISABLE_PREFIX_CACHE": "1"},
            )
            if self.process.stdin is not None:
                self.process.stdin = _StdinWiretap(
                    self.process.stdin,
                    self.outbound_frames,
                    self.outbound_changed,
                )
            if self.process.stdout is not None:
                self.process.stdout = _StdoutWiretap(
                    self.process.stdout,
                    self.inbound_frames,
                    self.inbound_changed,
                )
            self.client = StdioRuntimeClient(self.process, pending_event_limit=512)
            await self.wait_ready()
            return self
        except BaseException:
            process = self.process
            if process is not None:
                cleanup = asyncio.create_task(
                    terminate_process(process, timeout=self.process_timeout)
                )
                with suppress(BaseException):
                    await asyncio.shield(cleanup)
            await self._finish_client_close()
            await self._close_stdin()
            if self._log is not None and not self._log.closed:
                self._log.close()
            raise

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the client with a deadline, then always terminate the process."""

        process = self.process
        close_error: BaseException | None = None
        cancelled = False
        try:
            if self.client is not None:
                try:
                    await asyncio.wait_for(
                        self.client.aclose(), timeout=self.client_close_timeout
                    )
                except asyncio.TimeoutError:
                    # The process termination below is the authoritative
                    # cleanup path when shutdown is waiting for a missing ready
                    # or terminal frame.
                    pass
                except asyncio.CancelledError as error:
                    cancelled = True
                    close_error = error
                except BaseException as error:
                    close_error = error
            if process is not None:
                cleanup = asyncio.create_task(
                    terminate_process(process, timeout=self.process_timeout)
                )
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError as error:
                    cancelled = True
                    close_error = close_error or error
                    with suppress(BaseException):
                        await cleanup
                except BaseException as error:
                    close_error = close_error or error
        finally:
            # ``StdioRuntimeClient.aclose`` can be blocked waiting for the
            # ready frame (for example when a child process is alive but
            # never announces readiness).  Its normal shutdown path cannot
            # reach the reader-task cleanup in that state, so the session
            # owns the final cancellation boundary as well.
            await self._finish_client_close()
            await self._close_stdin()
            if self._log is not None and not self._log.closed:
                self._log.close()
        if cancelled:
            raise asyncio.CancelledError from close_error
        if close_error is not None:
            raise close_error

    async def _finish_client_close(self) -> None:
        """Let the production client's public close path reap its reader."""

        if self.client is None:
            return
        with suppress(BaseException):
            await asyncio.wait_for(
                self.client.aclose(),
                timeout=self.client_close_timeout,
            )
        process_stdout = self.process.stdout if self.process is not None else None
        transport = getattr(process_stdout, "_transport", None)
        if transport is not None:
            transport.close()

    async def _close_stdin(self) -> None:
        """Close stdin and finish its transport without extending shutdown."""

        if self.process is None or self.process.stdin is None:
            return
        stdin = self.process.stdin
        if not stdin.is_closing():
            stdin.close()
        wait_closed = getattr(stdin, "wait_closed", None)
        if wait_closed is not None:
            with suppress(BaseException):
                await asyncio.wait_for(wait_closed(), timeout=1)

    async def wait_ready(self, timeout: float | None = None) -> None:
        """Wait for the native ready frame without allowing startup to hang forever."""

        if self.client is None:
            raise RuntimeError("production stdio client has not been created")
        await asyncio.wait_for(
            self.client.health(),
            timeout=self.startup_timeout if timeout is None else timeout,
        )

    async def wait_for_inbound(
        self,
        predicate: Callable[[dict[str, Any]], bool],
        *,
        timeout: float = 30,
        start_index: int = 0,
    ) -> dict[str, Any]:
        """Wait until the stdout wiretap observes a matching native frame."""

        return await self._wait_for_frame(
            self.inbound_frames, self.inbound_changed, predicate, timeout, start_index
        )

    async def wait_for_outbound(
        self,
        predicate: Callable[[dict[str, Any]], bool],
        *,
        timeout: float = 30,
        start_index: int = 0,
    ) -> dict[str, Any]:
        """Wait until the stdin wiretap observes a matching client frame."""

        return await self._wait_for_frame(
            self.outbound_frames, self.outbound_changed, predicate, timeout, start_index
        )

    @staticmethod
    async def _wait_for_frame(
        frames: list[dict[str, Any]],
        changed: asyncio.Event,
        predicate: Callable[[dict[str, Any]], bool],
        timeout: float,
        start_index: int,
    ) -> dict[str, Any]:
        """Wait on a wiretap notification with a bounded deadline."""

        deadline = time.monotonic() + timeout
        while True:
            for frame in frames[start_index:]:
                if predicate(frame):
                    return frame
            changed.clear()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AssertionError("timed out waiting for native stdio frame")
            try:
                await asyncio.wait_for(changed.wait(), timeout=remaining)
            except asyncio.TimeoutError as error:
                raise AssertionError("timed out waiting for native stdio frame") from error


def resolve_native_runtime_assets(
    environment: Mapping[str, str] | None = None,
    *,
    root: Path,
) -> NativeRuntimeAssets:
    """Resolve and validate explicitly configured native runtime assets."""

    values = os.environ if environment is None else environment
    names = {
        "runtime": "TYLOINFER_STEP_TEST_RUNTIME",
        "model": "TYLOINFER_STEP_TEST_MODEL",
        "tokenizer": "TYLOINFER_STEP_TEST_TOKENIZER",
    }
    paths = {
        name: Path(values[variable]).expanduser()
        for name, variable in names.items()
        if values.get(variable)
    }
    paths = {
        name: path if path.is_absolute() else root / path
        for name, path in paths.items()
    }
    for name, path in paths.items():
        if name == "model" and path.is_dir():
            continue
        if not path.is_file():
            raise FileNotFoundError(f"configured native test resource does not exist: {name}={path}")

    missing = [name for name, variable in names.items() if not values.get(variable)]
    if missing:
        raise MissingNativeRuntimeAssetsError(
            "missing native test resources: " + ", ".join(names[name] for name in missing)
        )
    assert len(paths) == len(names)
    return NativeRuntimeAssets(**paths)


def tensor(dtype: str, shape: list[int], values: list[int]) -> dict[str, Any]:
    """Build a small protocol tensor used to exercise request validation."""

    code = "f" if dtype == "float32" else "i"
    return {
        "dtype": dtype,
        "shape": shape,
        "encoding": "base64",
        "data": base64.b64encode(struct.pack("<" + code * len(values), *values)).decode(),
    }


def generation_request(stream: bool, tokens: int = 16) -> dict[str, Any]:
    """Return the deterministic text generation request used by the gate."""

    return {
        "input": {"messages": [{"role": "user", "content": "Count from one to ten."}]},
        "sampling": {"max_new_tokens": tokens, "temperature": 0, "top_k": 1, "enable_mtp": False},
        "stream": stream,
        "include_usage": True,
    }


INDEPENDENT_PROMPT = "Count from one to ten."


def sha256_file(path: Path) -> str:
    """Hash a large fixture resource without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def invalid_media_request() -> dict[str, Any]:
    """Return a malformed media request that must be rejected before inference."""

    request = generation_request(False)
    request["input"]["preformatted_prompt"] = "Hello"
    request["media"] = {
        "pixel_values": tensor("float32", [1, 3, 1, 1], [1, 1, 1]),
        "vision_types": tensor("int32", [1], [1]),
        "image_grid_thw": tensor("int32", [1, 3], [1, 1, 1]),
    }
    return request


class NativeStdioSession:
    """Own a real runtime process and exchange newline-delimited protocol frames."""

    def __init__(self, assets: NativeRuntimeAssets, *, batch_size: int, log_path: Path) -> None:
        self.assets = assets
        self.batch_size = batch_size
        self.log_path = log_path
        self.process: asyncio.subprocess.Process | None = None
        self.frames: dict[str, list[dict[str, Any]]] = {}
        self._terminal_ids: set[str] = set()
        self._log = None

    async def __aenter__(self) -> NativeStdioSession:
        self._log = self.log_path.open("wb")
        try:
            self.process = await asyncio.create_subprocess_exec(
                str(self.assets.runtime),
                "--model",
                str(self.assets.model),
                "--tokenizer",
                str(self.assets.tokenizer),
                "--transport",
                "stdio",
                "--ctx-size",
                "256",
                "--prefill-chunk-size",
                "8",
                "--continuous-batching",
                str(self.batch_size),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=self._log,
                env={**os.environ, "TYLOINFER_RUNTIME_DISABLE_PREFIX_CACHE": "1"},
            )
            return self
        except BaseException:
            self._log.close()
            raise

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        await self.close()

    async def close(self) -> None:
        """Terminate a stuck process and close all owned handles within a bound."""

        process = self.process
        try:
            if process is not None:
                await terminate_process(process, timeout=5)
        finally:
            if process is not None and process.stdin is not None and not process.stdin.is_closing():
                process.stdin.close()
            if self._log is not None and not self._log.closed:
                self._log.close()

    async def shutdown(self, timeout: float = 30) -> None:
        """Request a graceful shutdown, drain stdout, and require a clean exit."""

        assert self.process is not None and self.process.stdout is not None
        await self.send("shutdown", "shutdown")
        await self.until("shutdown", "result", timeout=timeout)
        assert await asyncio.wait_for(self.process.wait(), timeout) == 0
        while line := await self.process.stdout.readline():
            frame = json.loads(line)
            self._record_frame(frame)

    async def receive(self, timeout: float = 60) -> dict[str, Any]:
        """Read and cache one protocol frame, including useful EOF diagnostics."""

        assert self.process is not None and self.process.stdout is not None
        line = await asyncio.wait_for(self.process.stdout.readline(), timeout)
        if not line:
            details = self.log_path.read_text(encoding="utf-8", errors="replace")
            raise AssertionError(f"native runtime exited before sending a frame:\n{details}")
        frame = json.loads(line)
        self._record_frame(frame)
        return frame

    def _record_frame(self, frame: dict[str, Any]) -> None:
        """Cache a frame and reject messages after a request's terminal response."""

        request_id = str(frame.get("id", ""))
        if request_id in self._terminal_ids:
            raise AssertionError(f"late or duplicate terminal frame for {request_id!r}: {frame}")
        self.frames.setdefault(request_id, []).append(frame)
        if frame.get("type") in {"done", "error", "result"}:
            self._terminal_ids.add(request_id)

    async def send(self, request_id: str, operation: str, params: dict[str, Any] | None = None) -> None:
        """Send one protocol request to the owned runtime process."""

        assert self.process is not None and self.process.stdin is not None
        payload = {"v": 1, "id": request_id, "op": operation, "params": params or {}}
        self.process.stdin.write((json.dumps(payload) + "\n").encode())
        await self.process.stdin.drain()

    async def until(self, request_id: str, kind: str = "done", timeout: float = 60) -> list[dict[str, Any]]:
        """Read until a request reaches a specified terminal frame kind."""

        deadline = time.monotonic() + timeout
        while True:
            cached = self.frames.get(request_id, [])
            if any(frame.get("type") == "error" for frame in cached):
                raise AssertionError(f"request {request_id!r} failed: {cached}")
            if any(frame.get("type") == kind for frame in cached):
                return cached
            if kind != "done" and any(frame.get("type") == "done" for frame in cached):
                raise AssertionError(f"request {request_id!r} completed before {kind!r}: {cached}")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AssertionError(f"timed out waiting for {request_id!r} {kind!r}")
            frame = await self.receive(remaining)
            if frame.get("id") == request_id and frame.get("type") == "error":
                raise AssertionError(f"request {request_id!r} failed: {frame}")

    async def until_event(
        self,
        request_id: str,
        predicate: Callable[[dict[str, Any]], bool],
        timeout: float = 60,
        require_active: bool = True,
    ) -> list[dict[str, Any]]:
        """Read until an event matches, optionally requiring the request to stay active."""

        deadline = time.monotonic() + timeout
        while True:
            cached = self.frames.get(request_id, [])
            if any(frame.get("type") == "error" for frame in cached):
                raise AssertionError(f"request {request_id!r} failed: {cached}")
            matched = any(
                frame.get("type") == "event"
                and predicate(frame.get("data", {}))
                for frame in cached
            )
            if matched and (not require_active or not any(frame.get("type") == "done" for frame in cached)):
                return cached
            if any(frame.get("type") == "done" for frame in cached):
                expected = "matching active event" if require_active else "matching event"
                raise AssertionError(f"request {request_id!r} completed without {expected}")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AssertionError(f"timed out waiting for matching event from {request_id!r}")
            frame = await self.receive(remaining)
            if frame.get("id") == request_id and frame.get("type") == "error":
                raise AssertionError(f"request {request_id!r} failed: {frame}")
            if frame.get("id") == request_id and frame.get("type") == "done":
                raise AssertionError(f"request {request_id!r} completed without matching event")


async def terminate_process(process: Any, *, timeout: float = 1) -> None:
    """Terminate a process, escalating to kill when it ignores termination."""

    if getattr(process, "returncode", None) is None:
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            process.kill()
            await asyncio.wait_for(process.wait(), timeout=timeout)
