"""Opt-in native GPU parity probe with a portable synthetic HF checkpoint.

The prepare/check modes allow a CPU reference and a CUDA executable to run on
separate hosts. Fixture artifacts contain synthetic tensors and tokens only.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def prepare(directory: Path):
    import torch
    from safetensors.torch import save_file

    from tests.test_d1_reference import request, tiny_checkpoint
    from tyloinfer.runtime.d1_reference import D1TextReference, pack_requests

    directory.mkdir(parents=True, exist_ok=True)
    _, model = tiny_checkpoint.__wrapped__(directory)
    save_file(
        {
            "model.language_model." + k: v.to(torch.bfloat16).contiguous()
            for k, v in model.state_dict().items()
        },
        str(directory / "model.safetensors"),
    )
    runtime = D1TextReference.load(directory, dtype="bfloat16")
    rows = pack_requests(runtime.tokenizer, runtime.contract, [request()])
    cases = []
    for row in rows:
        candidates = sorted({t for slot in row["slots"] for group in slot[4] for t in group})
        ids = torch.tensor([row["ids"]])
        with torch.inference_mode():
            hidden = runtime.model(
                ids,
                torch.ones_like(ids, dtype=torch.bool),
                torch.tensor([row["state"]]),
                torch.tensor([row["questions"]]),
            )
            positions = torch.tensor([s[3] for s in row["slots"]])
            logits = (
                runtime.model.slot_logits(
                    hidden,
                    torch.zeros_like(positions),
                    positions,
                    torch.tensor(candidates),
                )
                .float()
                .tolist()
            )
        cases.append(
            {
                "request": {
                    "tokens": row["ids"],
                    "question_ids": row["questions"],
                    "state_tokens": row["state"],
                    "slots": positions.tolist(),
                    "candidate_ids": candidates,
                    "warmup": 1,
                    "iterations": 3,
                },
                "expected_logits": logits,
            }
        )
    (directory / "fixture.json").write_text(json.dumps(cases), encoding="utf-8")


def check(executable: Path, directory: Path, weights: Path | None = None):
    cases = json.loads((directory / "fixture.json").read_text(encoding="utf-8"))
    results = []
    for fused in (False, True):
        command = [str(executable.resolve()), "--assets", str(directory.resolve())]
        if weights is not None:
            command.extend(["--weights", str(weights.resolve())])
        if not fused:
            command.append("--unfused")
        done = subprocess.run(
            command,
            input="".join(json.dumps(c["request"]) + "\n" for c in cases),
            text=True,
            encoding="utf-8",
            capture_output=True,
            timeout=120,
            check=True,
        )
        frames = [json.loads(line) for line in done.stdout.splitlines()]
        assert frames[0]["native_runtime"] and frames[0]["ready"]
        if weights is not None:
            assert frames[0]["resident_packed_weight_bytes"] > 0
            assert frames[0]["full_matrix_dequant_cache_bytes"] == 0
            assert all(t["storage"] == "Q8_0" for t in frames[0]["tensors"] if t["rows"] > 0)
        assert len(frames) == len(cases) + 1
        for frame, case in zip(frames[1:], cases, strict=True):
            assert "error" not in frame, frame
            if weights is not None:
                projections = [
                    t
                    for t in frames[0]["tensors"]
                    if t["storage"] == "Q8_0" and t["name"] != "embed_tokens.weight"
                ]
                assert frame["packed_gemm_calls_per_forward"] == len(projections)
                assert frame["bf16_gemm_calls_per_forward"] == 1
                assert frame["packed_embedding_rows_per_forward"] == len(
                    case["request"]["tokens"]
                ) + len(case["request"]["candidate_ids"])
            error = max(
                abs(a - b)
                for actual, expected in zip(frame["logits"], case["expected_logits"], strict=True)
                for a, b in zip(actual, expected, strict=True)
            )
            assert error <= 0.15, {"fused": fused, "max_abs": error}
            results.append({"fused": fused, "max_abs": error, "gpu_ms": frame["gpu_ms"]})
    return results


def test_native_gpu_tiny(tmp_path):
    import pytest

    executable = os.environ.get("TYLOINFER_D1_NATIVE_BINARY")
    if not executable:
        pytest.skip("standalone native d1 GPU executable not provided")
    prepare(tmp_path)
    assert len(check(Path(executable), tmp_path)) == 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", type=Path)
    parser.add_argument("--check", type=Path)
    parser.add_argument("--executable", type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.prepare)
    elif args.check and args.executable:
        print(json.dumps(check(args.executable, args.check)))
    else:
        parser.error("provide --prepare or --check with --executable")
