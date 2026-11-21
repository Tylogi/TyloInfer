"""Focused malformed artifact and numeric edge coverage for the reference path."""

from __future__ import annotations

# ruff: noqa: F811 -- imported pytest fixture is intentionally requested by name
import json

import pytest

from tests.test_d1_reference import config, request, tiny_checkpoint  # noqa: F401
from runtime.d1_contract import D1Contract
from runtime.d1_reference import D1TextReference


@pytest.mark.parametrize("field", ["d1_codes", "d1_prompt"])
def test_boolean_prompt_contract_is_not_an_integer(field):
    value = config()
    value[field] = True
    with pytest.raises(ValueError):
        D1Contract.parse(value)


def test_loader_rejects_missing_weights(tiny_checkpoint):
    directory, _ = tiny_checkpoint
    # Point the index at an existing non-weight file: no path guessing or download.
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"x": "config.json"}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="shard path"):
        D1TextReference.load(directory)


def test_loader_rejects_index_escape_before_opening_outside_file(tiny_checkpoint):
    directory, _ = tiny_checkpoint
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"x": "../config.json"}}), encoding="utf-8"
    )
    # Missing/outside shards must fail, never load a pickle or remote checkpoint.
    with pytest.raises((ValueError, FileNotFoundError)):
        D1TextReference.load(directory)


def test_loader_does_not_execute_checkpoint_python(tiny_checkpoint):
    directory, _ = tiny_checkpoint
    (directory / "modeling_d1_lfm3_vl.py").write_text(
        "raise RuntimeError('checkpoint code executed')", encoding="utf-8"
    )
    assert D1TextReference.load(directory).decide(request())["usage"]["output_tokens"] == 0


def test_bfloat16_local_weights_execute_without_meta_buffers(tiny_checkpoint):
    directory, _ = tiny_checkpoint
    runtime = D1TextReference.load(directory, dtype="bfloat16")
    out = runtime.decide(request())
    assert out["usage"]["output_tokens"] == 0
    assert 0 <= out["answers"]["refund"]["noul"] <= 1
