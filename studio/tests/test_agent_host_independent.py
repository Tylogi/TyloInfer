"""Opt-in complete Host lifecycle with an isolated, loopback-only external API fixture."""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from scripts.agent_host import PROFILE, prepare_profile


@pytest.mark.skipif(os.environ.get("TYLOINFER_TEST_AGENT_HOST") != "1", reason="requires the installed complete official Host")
def test_independent_host_external_api_and_cold_restart(tmp_path):
    root = Path(__file__).resolve().parents[2]
    node = shutil.which("node")
    if node is None:
        pytest.skip("requires Node.js")
    models = []

    class ExternalApi(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            content = json.dumps({"service": "external-api-fixture", "protocol_version": "external"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def do_POST(self):
            assert self.path == "/v1/chat/completions"
            assert self.headers.get("Authorization") == "Bearer fixture-only-not-a-real-key"
            length = int(self.headers.get("Content-Length", "0"))
            assert 0 < length <= 2 << 20
            body = json.loads(self.rfile.read(length))
            assert body["model"] == "fixture-external-model"
            models.append(body["model"])
            if body.get("stream"):
                chunks = [
                    {"id": "fixture", "object": "chat.completion.chunk", "model": body["model"], "choices": [{"index": 0, "delta": {"role": "assistant", "content": "fixture-external-response"}, "finish_reason": None}]},
                    {"id": "fixture", "object": "chat.completion.chunk", "model": body["model"], "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11}},
                ]
                content = ("".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks) + "data: [DONE]\n\n").encode()
                content_type = "text/event-stream"
            else:
                content = json.dumps({"id": "fixture", "object": "chat.completion", "model": body["model"], "choices": [{"index": 0, "message": {"role": "assistant", "content": "fixture-external-response"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11}}).encode()
                content_type = "application/json"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

    external = ThreadingHTTPServer(("127.0.0.1", 0), ExternalApi)
    thread = threading.Thread(target=external.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{external.server_port}/v1"
    state = tmp_path / "private-state"
    prepare_profile(state, root / "plugins", endpoint)
    git = shutil.which("git")
    if git is not None:
        subprocess.run([git, "init", "--quiet", str(state / "workspace")], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    patch_path = state / "home" / "profiles" / PROFILE / "cordis.patch.yml"
    patch = json.loads(patch_path.read_text(encoding="utf-8"))
    patch[0]["config"]["providers"]["fixture-external"] = {
        "displayName": "External fixture", "api": "openai-completions", "baseURL": endpoint,
        "apiKeyEnv": "TYLOINFER_FIXTURE_API_KEY",
        "models": [{"id": "fixture-external-model", "contextWindow": 16384, "maxTokens": 256, "input": ["text"], "reasoningEfforts": False}],
        "compat": {"maxTokensField": "max_tokens", "supportsDeveloperRole": False},
        "retryPolicy": {"mode": "normal", "maxRetries": 0},
    }
    patch.append({"id": "system-prompt", "config": {"includeRuntimeContext": False, "includeHarnessIdentity": False, "personaPrefix": "Fixture assistant. Do not use tools."}})
    patch_path.write_text(json.dumps(patch), encoding="utf-8")
    credential_file = state / "home" / ".credentials.yaml"
    with credential_file.open("x", encoding="utf-8") as credential:
        credential.write(json.dumps({"version": 1, "refs": {"TYLOINFER_FIXTURE_API_KEY": "fixture-only-not-a-real-key"}}))
    credential_file.chmod(0o600)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    environment = dict(os.environ)
    environment.pop("DSH_OFFICE_SIDECAR", None)
    process = None

    def stop():
        nonlocal process
        if process is None or process.poll() is not None:
            return
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        else:
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait(timeout=10)

    try:
        for phase in ("before", "after"):
            connection = state / "connection.json"
            previous = connection.stat().st_mtime_ns if connection.exists() else None
            with (tmp_path / (phase + "-host.log")).open("wb") as log:
                process = subprocess.Popen([sys.executable, "-B", "scripts/agent_host.py", "--state-root", str(state), "--port", str(port)],
                    cwd=root, env=environment, stdout=log, stderr=log, start_new_session=os.name != "nt")
                deadline = time.monotonic() + 40
                while not connection.exists() or connection.stat().st_mtime_ns == previous:
                    assert process.poll() is None, "isolated Host exited before readiness"
                    assert time.monotonic() < deadline, "isolated Host readiness deadline exceeded"
                    time.sleep(0.1)
                subprocess.run([node, str(root / "plugins/host/tests/independent-host-smoke.mjs"), str(state), phase],
                    cwd=root / "studio/web", env=environment, check=True, timeout=40)
                stop()
        assert len(models) >= 2 and all(model == "fixture-external-model" for model in models)
        print("INDEPENDENT_HOST_EXTERNAL_API_RESTART_PASS; CORE=unavailable-fixture; DSH=official-complete; MODELS=none; NETWORK=loopback-fixture-only")
    finally:
        stop()
        print('EXTERNAL_FIXTURE_REQUEST_COUNT=' + str(len(models)))
        external.shutdown()
        external.server_close()
        thread.join(5)
