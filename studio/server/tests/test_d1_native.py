"""Hardware-independent process protocol coverage; no claim of GPU correctness."""

from __future__ import annotations

# ruff: noqa: F811 -- imported pytest fixture is intentionally requested by name
import json
import subprocess
import sys

import pytest

from server.tests.test_d1_reference import request, tiny_checkpoint  # noqa: F401
from server.runtime.d1_native import D1Native


@pytest.fixture
def executable(tmp_path, monkeypatch):
    script = tmp_path / "native_fixture.py"
    script.write_text(
        "import json, sys\n"
        "print(json.dumps({'ready': True, 'native_runtime': True, "
        "'backend': 'native-cuda-cublas-lfm3', 'weight_format': 'fixture'}), flush=True)\n"
        "for line in sys.stdin:\n"
        " r = json.loads(line)\n"
        " print(json.dumps({'native_runtime': True, 'backend': 'native-cuda-cublas-lfm3', "
        "'candidate_ids': r['candidate_ids'], "
        "'logits': [[0.0]*len(r['candidate_ids']) for _ in r['slots']]}), flush=True)\n",
        encoding="utf-8",
    )
    real = subprocess.Popen

    def launch(command, **kwargs):
        assert command[0] == str(script.resolve())
        assert kwargs["shell"] is False
        return real([sys.executable, str(script)], **kwargs)

    monkeypatch.setattr("server.runtime.d1_native.subprocess.Popen", launch)
    return script


def test_native_bridge_groups_and_closes_owned_process(tiny_checkpoint, executable):
    directory, _ = tiny_checkpoint
    with D1Native(directory, executable) as runtime:
        output = runtime.decide_batch([request(), request()])
        assert len(output) == 2
        assert len(output[0]["answers"]) == 3
        assert output[0]["usage"]["output_tokens"] == 0
        assert output[0]["usage"]["input_tokens"] > 0
        assert runtime.ready["weight_format"] == "fixture"
    assert runtime.process.poll() is not None
    runtime.close()
    with pytest.raises(RuntimeError, match="closed"):
        runtime.evaluate(
            {"ids": [1], "state": 0, "questions": [1], "slots": [(0, "q", {}, 0)]}, [1]
        )


@pytest.mark.parametrize("warmup,iterations", [(True, 1), (0, True), (-1, 1), (0, 0), (101, 1)])
def test_native_bridge_rejects_bad_benchmark_count(tiny_checkpoint, executable, warmup, iterations):
    directory, _ = tiny_checkpoint
    with D1Native(directory, executable) as runtime, pytest.raises(ValueError, match="iteration"):
        runtime.evaluate({}, [], warmup=warmup, iterations=iterations)


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_native_timeout_must_be_finite(timeout):
    with pytest.raises(ValueError, match="timeout"):
        D1Native("unused", "unused", timeout=timeout)


def test_native_rejects_protocol_and_error_frames():
    import queue

    runtime = object.__new__(D1Native)
    runtime.timeout = 1
    for value in ["not json", "[]", json.dumps({"error": "fixture failure"}), None]:
        runtime._frames = queue.Queue()
        runtime._frames.put(value)
        with pytest.raises(RuntimeError):
            runtime._receive()


@pytest.mark.parametrize('placement', ['stream', 'vram'])
def test_native_placement_is_forwarded_without_optional_frameworks(tmp_path, monkeypatch, placement):
    root = tmp_path / 'assets'
    root.mkdir()
    executable = tmp_path / 'native'
    executable.touch()
    (root / 'calibration.json').write_text('{"default":1,"temperatures":{}}')
    monkeypatch.setattr('server.runtime.d1_native.D1Contract.load', lambda *_args: object())
    monkeypatch.setattr('server.runtime.d1_native.LocalTokenizer', lambda *_args: object())
    class Child:
        def __init__(self):
            import io
            self.stdin = io.StringIO()
            self.stdout = io.StringIO('{"ready":true,"native_runtime":true,"backend":"native-cuda-cublas-lfm3"}\n')
        def wait(self, **_kwargs):
            return 0
    def launch(command, **_kwargs):
        assert command[command.index('--weights-placement') + 1] == placement
        return Child()
    monkeypatch.setattr('server.runtime.d1_native.subprocess.Popen', launch)
    with D1Native(root, executable, weight_placement=placement):
        pass
