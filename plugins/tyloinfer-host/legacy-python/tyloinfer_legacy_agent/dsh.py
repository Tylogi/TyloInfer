"""Isolated DSH SDK adapter with a single bounded local inference tool."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import queue
import secrets
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from uuid import UUID

from tyloinfer.server.api.openai_compat import parse_chat_request, stream_chat_completion
from tyloinfer.server.protocol.models import SamplingParams
from tyloinfer.server.runtime.backend import BackendError, closing_backend_stream

SDK_VERSION = "0.2.2rc2"
TOOL_MODULE = """
export const name = 'tyloinfer-local-tool';
export const inject = ['tools'];
export function apply(ctx) {
  ctx.tools.register({
    name: 'local_inference',
    description: 'Ask the selected local model to perform a bounded text task. No files, shell, or network destinations can be selected.',
    parameters: {type: 'object', properties: {
      prompt: {type: 'string', minLength: 1, maxLength: 65536},
      max_tokens: {type: 'integer', minimum: 1, maximum: 4096}
    }, required: ['prompt'], additionalProperties: false},
    output: {schema: {type: 'object', properties: {
      text: {type: 'string'}, model: {type: 'string'}
    }, required: ['text', 'model'], additionalProperties: false},
      render: (_args, value) => [{type: 'text', text: JSON.stringify(value)}]},
    async execute(args, exec) {
      const response = await fetch(process.env.TYLOINFER_AGENT_BRIDGE + '/tool', {
        method: 'POST', signal: exec.signal,
        headers: {'content-type': 'application/json', 'authorization': 'Bearer ' + process.env.TYLOINFER_AGENT_TOKEN},
        body: JSON.stringify({arguments: args, tool_call_id: String(exec.callId)})
      });
      if (!response.ok) throw new Error('Local inference failed');
      return await response.json();
    }
  });
}
"""


class LocalBridge:
    def __init__(self, service, session, run, emit):
        self.service, self.session, self.run, self.emit = service, session, run, emit
        self.loop = asyncio.get_running_loop()
        self.token = secrets.token_urlsafe(32)
        self.stopped = threading.Event()
        self.failed = False
        self.pending: set = set()
        self.active: set[asyncio.Task] = set()
        self.lock = threading.Lock()
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                self.connection.settimeout(10)
                supplied = self.headers.get("Authorization", "")
                if not secrets.compare_digest(supplied, "Bearer " + bridge.token):
                    self.send_error(403)
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 1048576:
                        raise ValueError("invalid request size")
                    body = json.loads(self.rfile.read(length))
                    if not isinstance(body, dict):
                        raise ValueError("invalid request")
                    if self.path == "/tool":
                        future = bridge.submit(bridge.tool(body))
                        try:
                            result = future.result(timeout=300)
                        finally:
                            if not future.done():
                                future.cancel()
                        content = json.dumps(result, ensure_ascii=False).encode("utf-8")
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(content)))
                        self.end_headers()
                        self.wfile.write(content)
                    elif self.path == "/v1/chat/completions":
                        chunks = queue.Queue(maxsize=128)
                        future = bridge.submit(bridge.chat(body, chunks))
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.send_header("Connection", "close")
                        self.end_headers()
                        try:
                            while not bridge.stopped.is_set():
                                try:
                                    chunk = chunks.get(timeout=1)
                                except queue.Empty:
                                    if future.done():
                                        future.result()
                                        break
                                    continue
                                self.wfile.write(chunk.encode("utf-8"))
                                self.wfile.flush()
                        finally:
                            if not future.done():
                                future.cancel()
                    else:
                        self.send_error(404)
                except (BrokenPipeError, ConnectionResetError):
                    return
                except Exception:
                    self.close_connection = True
                    try:
                        self.send_error(502, "Local inference failed")
                    except OSError:
                        return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = "http://127.0.0.1:" + str(self.server.server_port)

    def submit(self, coroutine):
        if self.stopped.is_set():
            coroutine.close()
            raise RuntimeError("Agent bridge stopped")

        async def tracked():
            task = asyncio.current_task()
            self.active.add(task)
            try:
                if self.stopped.is_set():
                    coroutine.close()
                    raise asyncio.CancelledError
                return await coroutine
            except Exception:
                self.failed = True
                raise
            finally:
                self.active.discard(task)

        future = asyncio.run_coroutine_threadsafe(tracked(), self.loop)
        with self.lock:
            self.pending.add(future)

        def settled(_future):
            with self.lock:
                self.pending.discard(_future)

        future.add_done_callback(settled)
        return future

    async def chat(self, body, chunks):
        if self.stopped.is_set():
            raise RuntimeError("Agent bridge stopped")
        tools = body.get("tools", [])
        if any(tool.get("function", {}).get("name") != "local_inference" for tool in tools):
            raise ValueError("Tool is not permitted")
        parsed = parse_chat_request(
            {
                **body,
                "model": self.session["model"],
                "stream": True,
                "max_tokens": min(body.get("max_tokens", 1024), 4096),
                "enable_mtp": False,
                "enable_thinking": False,
            }
        )
        parsed = replace(
            parsed,
            routing_model=self.service.resolve_model_alias(self.session["model"]),
            session_id=UUID(self.run["id"]),
        )
        bridge = self

        class Backend:
            async def stream(self, **kwargs):
                try:
                    async with closing_backend_stream(
                        bridge.service.backend.stream(**kwargs)
                    ) as stream:
                        async for delta in stream:
                            if delta.content_delta:
                                bridge.emit("text.delta", {"delta": delta.content_delta})
                            yield delta
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    bridge.failed = True
                    raise BackendError(
                        "local_inference_failed", "Local inference failed"
                    ) from error

        async for chunk in stream_chat_completion(Backend(), parsed, keepalive_seconds=5):
            if self.stopped.is_set():
                raise asyncio.CancelledError
            chunks.put_nowait(chunk)

    async def tool(self, body):
        arguments = body.get("arguments")
        if not isinstance(arguments, dict) or set(arguments) - {"prompt", "max_tokens"}:
            raise ValueError("Invalid local inference arguments")
        prompt, limit = arguments.get("prompt"), arguments.get("max_tokens", 256)
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt.encode("utf-8")) > 65536:
            raise ValueError("Invalid local inference prompt")
        if type(limit) is not int or not 1 <= limit <= 4096:
            raise ValueError("Invalid local inference output limit")
        call_id = body.get("tool_call_id")
        if not isinstance(call_id, str) or not 1 <= len(call_id) <= 256:
            raise ValueError("Invalid tool call ID")
        self.emit(
            "tool.started",
            {
                "tool_call_id": call_id,
                "name": "local_inference",
                "status": "running",
                "arguments": arguments,
            },
        )
        output = ""
        try:
            if self.stopped.is_set():
                raise asyncio.CancelledError
            async with closing_backend_stream(
                self.service.backend.stream(
                    model=self.service.resolve_model_alias(self.session["model"]),
                    messages=[{"role": "user", "content": prompt}],
                    sampling=SamplingParams(
                        max_tokens=limit, enable_mtp=False, enable_thinking=False
                    ),
                    session_id=UUID(self.run["id"]),
                    tools=(),
                    tool_choice="none",
                    response_format=None,
                )
            ) as stream:
                async for delta in stream:
                    output += delta.content_delta
                    if len(output.encode("utf-8")) > 65536:
                        raise ValueError("Local inference output limit reached")
                    if delta.tool_calls:
                        raise ValueError("Nested tool calls are not permitted")
            result = {"text": output, "model": self.session["model"]}
            self.emit(
                "tool.completed",
                {
                    "tool_call_id": call_id,
                    "name": "local_inference",
                    "status": "completed",
                    "output": output,
                    "is_error": False,
                },
            )
            return result
        except BaseException:
            self.emit(
                "tool.completed",
                {
                    "tool_call_id": call_id,
                    "name": "local_inference",
                    "status": "failed",
                    "output": "Local inference did not complete",
                    "is_error": True,
                },
            )
            raise

    async def close(self):
        self.stopped.set()
        with self.lock:
            pending = list(self.pending)
        for future in pending:
            future.cancel()
        active = list(self.active)
        for task in active:
            task.cancel()
        await asyncio.gather(*active, return_exceptions=True)
        cancel = getattr(self.service.backend, "cancel_response", None)
        if callable(cancel):
            await cancel(UUID(self.run["id"]))
        await asyncio.to_thread(self.server.shutdown)
        self.server.server_close()
        await asyncio.to_thread(self.thread.join, 2)
        close = getattr(self.service.backend, "close_session", None)
        if callable(close):
            await close(UUID(self.run["id"]))


class DshRuntime:
    def __init__(self):
        self.dsh_bin = os.environ.get("TYLOINFER_AGENT_DSH_BIN")
        self.harness = None
        self.bridge = None
        self.cancelled = False

    @staticmethod
    def capabilities():
        try:
            installed = version("deepseek-harness-sdk")
        except PackageNotFoundError:
            installed = None
        available = (
            installed == SDK_VERSION
            and importlib.util.find_spec("deepseek_harness_runtime") is not None
        )
        return {
            "available": available,
            "version": installed,
            "reason": None if available else "Install the pinned Agent SDK optional dependency",
        }

    async def run(self, *, session, run, root: Path, service, emit):
        from deepseek_harness import DeepSeekHarness

        if os.environ.get("DSH_OFFICE_SIDECAR"):
            raise RuntimeError(
                "Inherited DSH Office resources are not permitted in the Agent runtime"
            )
        directory = root / session["id"]
        directory.mkdir(parents=True, exist_ok=True)
        home, workspace = directory / "home", directory / "workspace"
        home.mkdir(exist_ok=True)
        workspace.mkdir(exist_ok=True)
        plugin = directory / "local-tool.mjs"
        plugin.write_text(TOOL_MODULE, encoding="utf-8")
        self.bridge = LocalBridge(service, session, run, emit)
        disabled = [
            "llm-deepseek",
            "deepseek-llm-api-extensions",
            "session-log-deepseek",
            "plugin-package-inventory-deepseek",
            "persistent-bash",
            "persistent-pwsh",
            "terminal-bash",
            "terminal-pwsh",
            "tool-fs",
            "tool-working-directory",
            "compaction-basic",
            "image-offload",
        ]
        patch = [{"id": key, "disabled": True} for key in disabled]
        patch += [
            {
                "id": "system-prompt",
                "config": {
                    "includeHarnessIdentity": False,
                    "includeRuntimeContext": False,
                    "personaPrefix": "You are a task assistant. Use local_inference for bounded subtasks when useful. You have no shell or file tools.",
                },
            },
            {
                "insert": [
                    {
                        "id": "tyloinfer-model",
                        "name": "@deepseek-ai/dsh-llm-pi-ai",
                        "config": {
                            "providers": {
                                "tyloinfer": {
                                    "api": "openai-completions",
                                    "baseURL": self.bridge.url + "/v1",
                                    "apiKeyEnv": "TYLOINFER_AGENT_TOKEN",
                                    "retryPolicy": {"mode": "normal", "maxRetries": 0},
                                    "compat": {
                                        "supportsDeveloperRole": False,
                                        "maxTokensField": "max_tokens",
                                    },
                                    "models": [
                                        {
                                            "id": session["model"],
                                            "contextWindow": 16384,
                                            "maxTokens": 1024,
                                            "input": ["text"],
                                            "reasoningEfforts": False,
                                        }
                                    ],
                                }
                            }
                        },
                    },
                    {"id": "tyloinfer-tool", "name": str(plugin.resolve())},
                ]
            },
        ]
        patch_path = directory / "runtime.patch.yml"
        patch_path.write_text(json.dumps(patch), encoding="utf-8")
        cache, temporary = home / "cache", home / "tmp"
        cache.mkdir(exist_ok=True)
        temporary.mkdir(exist_ok=True)
        env = {
            key: ""
            for key in os.environ
            if any(part in key.upper() for part in ("TOKEN", "SECRET", "PASSWORD", "KEY"))
            or (key.startswith("DSH_") and key != "DSH_OFFICE_SIDECAR")
        }
        env.update(
            {
                "TYLOINFER_AGENT_BRIDGE": self.bridge.url,
                "TYLOINFER_AGENT_TOKEN": self.bridge.token,
                "DSH_PRIMARY_RUNTIME": "",
                "DSH_RESOURCE_CACHE": str(cache),
                "HOME": str(workspace),
                "USERPROFILE": str(workspace),
                "APPDATA": str(cache),
                "LOCALAPPDATA": str(cache),
                "XDG_CONFIG_HOME": str(cache),
                "TEMP": str(temporary),
                "TMP": str(temporary),
                "HTTP_PROXY": "",
                "HTTPS_PROXY": "",
                "ALL_PROXY": "",
                "NO_PROXY": "127.0.0.1,localhost",
                "NODE_OPTIONS": "",
                "NODE_PATH": "",
            }
        )
        self.harness = DeepSeekHarness(
            provider="tyloinfer",
            model=session["model"],
            max_tokens=1024,
            dsh_bin=self.dsh_bin,
            cwd=str(workspace),
            runtime_cwd=str(workspace),
            dsh_home=str(home),
            profile="sdk-minimal",
            patches=(str(patch_path),),
            env=env,
            initialize_timeout_seconds=30,
            shutdown_timeout_seconds=3,
        )
        marker = directory / "session-ready"

        def execute():
            with self.harness:
                if self.cancelled:
                    raise RuntimeError("Agent run cancelled")
                if marker.exists():
                    selected = self.harness.resume_session(session["id"])
                else:
                    selected = self.harness.start_session(session["id"])
                    selected.get_working_directory()
                    marker.touch()
                result = selected.run(run["input"])
                if result.finish_reason != "completed" or self.bridge.failed:
                    raise RuntimeError("Agent run did not complete")

        worker = asyncio.create_task(asyncio.to_thread(execute))
        try:
            await asyncio.wait_for(asyncio.shield(worker), timeout=300)
        except BaseException:
            await self.cancel()
            await asyncio.gather(worker, return_exceptions=True)
            raise
        finally:
            await self.bridge.close()

    async def cancel(self):
        self.cancelled = True
        if self.harness is not None:
            await asyncio.to_thread(self.harness.close)
