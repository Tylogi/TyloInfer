"""Small HTTP server lifecycle helper for Agent bridge integration tests."""

from __future__ import annotations

import asyncio
import socket
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import uvicorn


@contextmanager
def running_agent_bridge(app: Any) -> Iterator[str]:
    """Run an ASGI app on a loopback port and always stop its server thread."""

    ready = threading.Event()

    class ReadyServer(uvicorn.Server):
        async def startup(self, sockets=None):
            await super().startup(sockets=sockets)
            if self.started:
                ready.set()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        server = ReadyServer(uvicorn.Config(app, log_level="error", access_log=False))
        thread = threading.Thread(
            target=lambda: asyncio.run(server.serve(sockets=[listener])),
            daemon=True,
        )
        thread.start()
        try:
            assert ready.wait(10), "Agent bridge API did not start"
            yield f"http://127.0.0.1:{port}"
        finally:
            server.should_exit = True
            thread.join(10)
            assert not thread.is_alive(), "Agent bridge API did not stop"
