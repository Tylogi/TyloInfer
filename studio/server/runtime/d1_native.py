"""Text d1 orchestration shared by standalone and managed native execution.

Python handles tokenization and calibrated readout only; no PyTorch numerical
fallback, checkpoint Python execution or automatic download.
"""

from __future__ import annotations

import json
import math
import queue
import subprocess
import threading
from contextlib import suppress
from pathlib import Path

from server.runtime.d1_contract import D1Calibration, D1Contract
from server.runtime.d1_reference import LocalTokenizer, pack_requests, read_answer


class D1Native:
    backends = frozenset({"native-cuda-cublas-lfm3", "native-metal-mlx-lfm3"})

    def __init__(
        self,
        assets: Path,
        executable: Path,
        *,
        weights: Path | None = None,
        fused: bool = True,
        weight_placement: str = "vram",
        timeout: float = 300,
    ):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("native d1 timeout must be positive and finite")
        if weight_placement not in {"stream", "vram"}:
            raise ValueError("native d1 weight placement must be stream or vram")
        assets, executable = (
            Path(assets).resolve(strict=True),
            Path(executable).resolve(strict=True),
        )
        if not assets.is_dir() or not executable.is_file():
            raise ValueError("native d1 requires local assets and an executable file")
        self.contract = D1Contract.load(assets)
        self.calibration = D1Calibration.parse(
            json.loads((assets / "calibration.json").read_text(encoding="utf-8"))
        )
        self.tokenizer = LocalTokenizer(assets)
        command = [str(executable), "--assets", str(assets)]
        command.extend(["--weights-placement", weight_placement])
        if weights is not None:
            command.extend(["--weights", str(Path(weights).resolve(strict=True))])
        if not fused:
            command.append("--unfused")
        self.timeout = timeout
        self._frames = queue.Queue()
        self._lock = threading.Lock()
        self._closed = False
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            encoding="utf-8",
            bufsize=1,
            shell=False,
        )
        self._reader = threading.Thread(target=self._read_frames, daemon=True)
        self._reader.start()
        try:
            self.ready = self._receive()
            if self.ready.get("ready") is not True or self.ready.get("native_runtime") is not True:
                raise RuntimeError("native d1 executable did not announce readiness")
            backend = self.ready.get("backend")
            if not isinstance(backend, str) or backend not in self.backends:
                raise RuntimeError("unexpected native d1 backend")
            self.backend = backend
        except BaseException:
            self.close()
            raise

    def _read_frames(self):
        try:
            for line in self.process.stdout:
                self._frames.put(line)
        finally:
            self._frames.put(None)

    def _receive(self):
        try:
            line = self._frames.get(timeout=self.timeout)
        except queue.Empty as exc:
            self.close()
            raise RuntimeError("native d1 executable timed out") from exc
        if line is None:
            raise RuntimeError("native d1 executable closed its output")
        try:
            value = json.loads(line)
        except (ValueError, TypeError) as exc:
            raise RuntimeError("invalid native d1 protocol response") from exc
        if not isinstance(value, dict):
            raise RuntimeError("invalid native d1 protocol response")
        if "error" in value:
            raise RuntimeError(f"native d1: {value['error']}")
        return value

    def evaluate(self, row: dict, candidates: list[int], *, warmup=0, iterations=1) -> dict:
        if (
            type(warmup) is not int
            or not 0 <= warmup <= 100
            or type(iterations) is not int
            or not 1 <= iterations <= 1000
        ):
            raise ValueError("invalid native d1 benchmark iteration count")
        request = {
            "tokens": row["ids"],
            "state_tokens": row["state"],
            "question_ids": row["questions"],
            "slots": [s[3] for s in row["slots"]],
            "candidate_ids": candidates,
            "warmup": warmup,
            "iterations": iterations,
        }
        with self._lock:
            if self._closed:
                raise RuntimeError("native d1 runtime is closed")
            try:
                self.process.stdin.write(json.dumps(request, allow_nan=False) + "\n")
                self.process.stdin.flush()
                result = self._receive()
            except (OSError, RuntimeError):
                self.close()
                raise
        logits = result.get("logits")
        if (
            result.get("native_runtime") is not True
            or result.get("backend") != self.backend
            or result.get("candidate_ids") != candidates
            or not isinstance(logits, list)
            or len(logits) != len(row["slots"])
            or any(
                not isinstance(values, list)
                or len(values) != len(candidates)
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
                for values in logits
            )
        ):
            self.close()
            raise RuntimeError("invalid native d1 logits response")
        return result

    def decide_batch(self, requests: list[dict], *, warmup=0, iterations=1) -> list[dict]:
        if not isinstance(requests, list) or not requests:
            raise ValueError("d1 batch must contain at least one request")
        rows = pack_requests(self.tokenizer, self.contract, requests)
        results = [
            {"answers": {}, "usage": {"input_tokens": 0, "output_tokens": 0}, "native_metrics": []}
            for _ in requests
        ]
        for row in rows:
            candidates = sorted({t for s in row["slots"] for group in s[4] for t in group})
            response = self.evaluate(row, candidates, warmup=warmup, iterations=iterations)
            target = results[row["request"]]
            for slot, values in zip(row["slots"], response["logits"], strict=True):
                _, name, question, _, options = slot
                target["answers"][name] = read_answer(
                    question,
                    dict(zip(candidates, values, strict=True)),
                    options,
                    self.calibration,
                )
            target["usage"]["input_tokens"] += len(row["ids"])
            target["native_metrics"].append({k: v for k, v in response.items() if k != "logits"})
        return results

    def decide(self, payload: dict, **kwargs) -> dict:
        return self.decide_batch([payload], **kwargs)[0]

    def close(self):
        if self._closed:
            return
        self._closed = True
        with suppress(OSError):
            self.process.stdin.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self._reader.join(timeout=1)
        self.process.stdout.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
