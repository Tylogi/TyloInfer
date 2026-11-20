"""Opt-in pinned d1-2B answer regression, including a near-median Score case."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from server.tests.d1_native_real_probe import requests
from server.runtime.d1_native import D1Native


@pytest.mark.parametrize("kind", ["hf", "gguf"])
@pytest.mark.parametrize("placement", ["stream", "vram"])
def test_pinned_real_native_answers(kind, placement):
    binary = os.environ.get("TYLOINFER_D1_NATIVE_BINARY")
    assets = os.environ.get("TYLOINFER_D1_NATIVE_ASSETS")
    gguf = os.environ.get("TYLOINFER_D1_NATIVE_GGUF")
    if not binary or not assets or (kind == "gguf" and not gguf):
        pytest.skip("provide native binary and pinned official d1-2B HF/GGUF assets explicitly")
    weights = Path(assets if kind == "hf" else gguf)
    with D1Native(Path(assets), Path(binary), weights=weights, weight_placement=placement) as runtime:
        output = runtime.decide_batch(requests())
        assert runtime.ready['weight_placement'] == placement
        assert runtime.ready['layer_buffer_count'] == (2 if placement == 'stream' else 0)
        assert (runtime.ready['streamed_weight_bytes'] > 0) == (placement == 'stream')
    first, second = [row["answers"] for row in output]
    assert first["refund"]["noul"] > 0.98
    assert first["team"]["choice"] == "billing"
    # Premature BF16 rounding of attention probabilities flipped this median.
    assert first["urgency"]["score"] == 0
    assert second["delivery"]["noul"] > 0.97
    assert second["team"]["choice"] == "shipping"
    assert all(row["usage"]["output_tokens"] == 0 for row in output)
