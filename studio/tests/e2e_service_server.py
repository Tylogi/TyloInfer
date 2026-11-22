"""Start a real TyloInfer HTTP service with a deterministic test-only backend.

The application, authentication middleware, session store, and SSE routes are
the production implementations. Only the ChatBackend is replaced so this gate
does not require a GPU or a model download.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import AsyncIterator, Sequence
from datetime import datetime, timezone
from pathlib import Path
from struct import pack
from typing import Any
from uuid import UUID, uuid4

import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server.api import create_app
from server.api.static import SPAStaticFiles
from server.protocol.models import (
    ModelCapabilities,
    ModelFeatureSet,
    ResponseFormat,
    RuntimeCapabilitiesResource,
    RuntimeInstanceList,
    RuntimeInstanceResource,
    RuntimeInstanceState,
    SamplingParams,
    TokenUsage,
    ToolChoice,
    ToolDefinition,
)
from server.runtime.backend import BackendDelta
from server.runtime.client import BackendError
from server.services.service import ServerService
from server.state.catalog import ModelCatalog
from server.state.storage import SessionStore


class BrowserProtocolBackend:
    """Deterministic backend adapter used only by browser service integration."""

    model = "test-browser-model"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.closed = False
        self._stream_gate: asyncio.Event | None = None

    async def stream(
        self,
        *,
        model: str,
        messages: Sequence[dict[str, Any]],
        sampling: SamplingParams,
        session_id: UUID | None = None,
        tools: Sequence[ToolDefinition] = (),
        tool_choice: ToolChoice = "auto",
        response_format: ResponseFormat | None = None,
    ) -> AsyncIterator[BackendDelta]:
        del tools, tool_choice, response_format
        self.calls.append(
            {
                "model": model,
                "messages": list(messages),
                "sampling": sampling.model_dump(mode="json"),
                "session_id": str(session_id) if session_id else None,
            }
        )
        if messages and messages[-1].get("content") == "trigger-backend-error":
            raise BackendError(
                "test_backend_error",
                "the deterministic test backend rejected this request",
                status_code=503,
            )
        prompt = str(messages[-1].get("content", "")) if messages else ""
        gate = asyncio.Event()
        self._stream_gate = gate
        try:
            yield BackendDelta(content_delta="service echo: ")
            await gate.wait()
            yield BackendDelta(content_delta=prompt)
        finally:
            if self._stream_gate is gate:
                self._stream_gate = None
        yield BackendDelta(
            finish_reason="stop",
            usage=TokenUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
        )

    async def aclose(self) -> None:
        self.closed = True

    async def fork_session(self, source_session_id: UUID, target_session_id: UUID) -> bool:
        del source_session_id, target_session_id
        return True

    async def close_session(self, session_id: UUID) -> bool:
        del session_id
        return True

    async def cancel_response(self, session_id: UUID) -> bool:
        del session_id
        return True

    async def capabilities(self) -> RuntimeCapabilitiesResource:
        return RuntimeCapabilitiesResource(
            model=self.model,
            model_type="deepseek_v4",
            model_capabilities=ModelCapabilities(
                architecture_family="deepseek_v4",
                source="test-browser-backend",
                features=ModelFeatureSet(text=True),
            ),
        )

    async def runtime_status(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "runtime_state": "ready",
            "active_requests": 0,
            "queued_requests": 0,
            "max_context": 4096,
            "reloading": False,
            "mtp_service_enabled": False,
            "chat_template_capabilities": {},
        }

    async def runtime_models(self) -> dict[str, Any]:
        return {"object": "list", "data": [{"id": self.model, "object": "model"}]}

    async def realtime_capabilities(self) -> dict[str, Any]:
        return {"available": False, "input_sample_rate": 16000}

    async def reload_runtime(self, context_size: int) -> dict[str, Any]:
        return {"model": self.model, "runtime_state": "ready", "max_context": context_size}

    async def clear_runtime_cache(self) -> dict[str, Any]:
        return {"status": "ok", "released_snapshots": 0, "prefix_cache_snapshots": 0}

    async def trim_runtime_cache(self, target_bytes: int = 0) -> dict[str, Any]:
        return {"status": "ok", "released_bytes": 0, "target_bytes": target_bytes}


class BrowserFixtureRuntime:
    """Controlled runtime boundary for the real model-management UI flow."""

    def __init__(self) -> None:
        self.instance_id = uuid4()
        self.model: str | None = None
        self.state: RuntimeInstanceState | None = None
        self.release_load = asyncio.Event()

    async def start(self) -> None:
        return None

    async def load(self, context: Any, payload: dict[str, Any]) -> dict[str, Any]:
        self.model = str(payload["model"])
        self.state = RuntimeInstanceState.LOADING
        await context.progress(0.2, message="Fixture runtime is loading the model")
        await self.release_load.wait()
        self.state = RuntimeInstanceState.READY
        await context.progress(1.0, message="Fixture runtime loaded the model")
        return {"instance_id": str(self.instance_id), "model": self.model}

    async def unload(self, context: Any, payload: dict[str, Any]) -> dict[str, Any]:
        del context, payload
        self.state = None
        return {"instance_id": str(self.instance_id), "unloaded": True}

    async def instances(self) -> RuntimeInstanceList:
        if self.model is None or self.state is None:
            return RuntimeInstanceList(data=[])
        return RuntimeInstanceList(data=[RuntimeInstanceResource(
            id=self.instance_id,
            model=self.model,
            state=self.state,
            devices=["fixture"],
            active_sessions=0,
            queued_requests=0,
            context_size=4096,
            context_capacity=4096,
            started_at=datetime.now(timezone.utc),
        )])

    async def runtime_status(self, instance_id: UUID) -> dict[str, Any]:
        if instance_id != self.instance_id or self.model is None or self.state is None:
            raise RuntimeError("fixture runtime instance was not found")
        return {
            "instance_id": str(self.instance_id),
            "model": self.model,
            "runtime_state": self.state.value,
            "active_requests": 0,
            "queued_requests": 0,
            "reloading": self.state is RuntimeInstanceState.LOADING,
            "max_context": 4096,
        }

    async def capabilities(self, instance_id: UUID) -> RuntimeCapabilitiesResource:
        if instance_id != self.instance_id:
            raise RuntimeError("fixture runtime instance was not found")
        return RuntimeCapabilitiesResource(
            model=self.model or "fixture-model",
            model_type="fixture",
            model_capabilities=ModelCapabilities(
                architecture_family="fixture",
                source="browser-fixture-runtime",
                features=ModelFeatureSet(text=True),
            ),
        )

    async def aclose(self) -> None:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--api-key", default="test-browser-key")
    parser.add_argument("--web-root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.database.parent.mkdir(parents=True, exist_ok=True)
    model_root = args.database.parent / f"{args.database.stem}-models"
    model_root.mkdir(parents=True, exist_ok=True)
    model_path = model_root / "fixture-model.gguf"
    names = (b"dense.weight", b"dense.bias", b"output.weight")
    records = b"".join(
        pack("<Q", len(name)) + name + pack("<IQIQ", 1, 32, 0, 0)
        for name in names
    )
    model_path.write_bytes(pack("<4sIQQ", b"GGUF", 3, len(names), 0) + records)

    def inspect_fixture(_: Path) -> dict[str, Any]:
        return {
            "architecture": "fixture",
            "config": {"model_type": "fixture", "max_position_embeddings": 4096},
            "tensors": {
                "model.weight": {"dtype": "GGUF_Q4_0", "nbytes": 1024, "shape": [32, 32]},
            },
        }

    catalog = ModelCatalog([model_root], cache_seconds=0, native_inspector=inspect_fixture)
    backend = BrowserProtocolBackend()
    runtime = BrowserFixtureRuntime()
    service = ServerService(SessionStore(args.database), backend, catalog=catalog, runtime_manager=runtime)
    app = create_app(service=service, api_key=args.api_key)

    @app.get('/__test/backend-calls')
    async def backend_calls() -> dict[str, Any]:
        """Expose fake-backend observations to the integration assertion only."""
        return {'calls': backend.calls}

    @app.post('/__test/release-stream')
    async def release_stream() -> dict[str, str]:
        """Release the current second-delta barrier for deterministic UI checks."""
        if backend._stream_gate is None:
            return {'status': 'no-stream'}
        backend._stream_gate.set()
        return {'status': 'released'}

    @app.post('/__test/release-model-load')
    async def release_model_load() -> dict[str, str]:
        """Release the controlled runtime load barrier used by browser tests."""
        runtime.release_load.set()
        return {'status': 'released'}

    if args.web_root is not None:
        if not args.web_root.is_dir():
            raise ValueError(f'web root is not a directory: {args.web_root}')
        app.mount('/', SPAStaticFiles(directory=args.web_root, html=True), name='web')

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
