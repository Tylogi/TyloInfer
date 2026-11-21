from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from runtime.d1_contract import D1Calibration, D1Contract, median_level
from runtime.d1_reference import (
    D1TextReference,
    pack_requests,
    read_answer,
    validate_request,
)


def config():
    return {
        "architectures": ["D1Model"],
        "model_type": "lfm3_vl",
        "bos_token_id": 1,
        "d1_attention": "set",
        "d1_codes": 2,
        "d1_set_size": 16,
        "tie_word_embeddings": True,
        "text_config": {
            "model_type": "lfm3",
            "hidden_size": 16,
            "intermediate_size": 24,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "vocab_size": 64,
            "max_position_embeddings": 2048,
            "conv_L_cache": 7,
            "conv_bias": False,
            "use_sandwich_norm": True,
            "tie_word_embeddings": True,
            "pad_token_id": 0,
            "layer_types": ["conv", "full_attention", "sliding_attention"],
            "num_hidden_layers": 3,
            "norm_eps": 1e-5,
            "sliding_window": 4,
            "rope_parameters": {
                "full_attention": {
                    "rope_type": "default",
                    "rope_theta": 1e6,
                    "partial_rotary_factor": 0.5,
                },
                "sliding_attention": {"rope_type": "default", "rope_theta": 1e4},
            },
        },
    }


def calibration():
    return D1Calibration.parse(
        {
            "default": 1.3,
            "temperatures": {"noul": 2.0, "score:3": 1.0},
            "tables": {"found:text": [[0.1, 0.2], [0.9, 0.8]]},
        }
    )


def request():
    return {
        "state": "blue ticket",
        "questions": {
            "refund": {"type": "noul", "instructions": "Refund?"},
            "team": {
                "type": "choice",
                "instructions": "Team?",
                "criteria": {"billing": "Billing", "support": None},
            },
            "urgency": {
                "type": "score",
                "instructions": "Urgency?",
                "criteria": ["low", "medium", "high"],
            },
        },
    }


def test_model_contract_uses_config_geometry_not_developer_shapes():
    spec = D1Contract.parse(config())
    assert spec.text.layer_types == ("conv", "full_attention", "sliding_attention")
    assert spec.text.hidden_size == 16 and spec.text.num_key_value_heads == 2
    assert spec.set_size == 16


@pytest.mark.parametrize(
    "mutation", ["d3", "causal", "layers", "gqa", "rope", "eps", "bias", "context"]
)
def test_invalid_model_contract(mutation):
    data = config()
    if mutation == "d3":
        data["architectures"] = ["Qwen3_5ForCausalLM"]
    elif mutation == "causal":
        data["d1_attention"] = "causal"
    elif mutation == "layers":
        data["text_config"]["layer_types"][0] = "mamba"
    elif mutation == "gqa":
        data["text_config"]["num_key_value_heads"] = 3
    elif mutation == "rope":
        data["text_config"]["rope_parameters"]["full_attention"]["rope_type"] = "dynamic"
    elif mutation == "eps":
        data["text_config"]["norm_eps"] = float("nan")
    elif mutation == "bias":
        data["text_config"]["conv_bias"] = "false"
    else:
        data["text_config"]["max_position_embeddings"] = True
    with pytest.raises(ValueError):
        D1Contract.parse(data)


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"state": "x", "questions": {}},
        {"state": "x", "questions": {"a": {"type": "text", "instructions": "Extract"}}},
        {
            "state": "x",
            "questions": {
                "a": {
                    "type": "choice",
                    "criteria": {"a": None},
                    "instructions": "?",
                    "multiple": True,
                }
            },
        },
        {
            "state": "x",
            "questions": {"a": {"type": "score", "criteria": ["one"], "instructions": "?"}},
        },
        {**request(), "use_cache": True},
        {**request(), "media": []},
        {**request(), "set_size": True},
        {**request(), "state": float("inf")},
        {**request(), "state": None},
    ],
)
def test_unsupported_requests_are_rejected_explicitly(payload):
    with pytest.raises((ValueError, TypeError)):
        validate_request(payload)


def test_calibration_interpolates_and_temperature_uses_question_bucket():
    c = calibration()
    assert c.temperature("noul", 2) == 2
    assert c.temperature("choice", 3) == 1.3
    assert c.through("found:text", 0.5) == pytest.approx(0.5)
    assert c.through("found:text", 0) == 0.2
    assert c.through("found:text", 1) == 0.8
    assert c.through("missing", 0.4) == 0.4


@pytest.mark.parametrize(
    "bad",
    [
        {"default": 0, "temperatures": {}},
        {"default": 1, "temperatures": {"noul": float("inf")}},
        {"default": 1, "temperatures": {}, "tables": {"x": [[0.5, 0.2], [0.5, 0.8]]}},
        {"default": 1, "temperatures": {}, "tables": {"x": [[0.1, 0.8], [0.9, 0.2]]}},
    ],
)
def test_bad_calibration_is_not_silently_ignored(bad):
    with pytest.raises(ValueError):
        D1Calibration.parse(bad)


def test_score_is_median_not_mean_and_alias_logits_are_max_pooled():
    q = request()["questions"]["urgency"]
    out = read_answer(
        q, [math.log(0.4), math.log(0.35), math.log(0.25)], [[0], [1], [2]], calibration()
    )
    assert out["score"] == 1 and out["score"] != pytest.approx(0.85)
    assert median_level([0.5, 0.25, 0.25]) == 0
    choice = request()["questions"]["team"]
    result = read_answer(choice, [0, 4, 3], [[0, 1], [2]], calibration())
    assert result["choice"] == "billing"
    assert result["confidence"] == pytest.approx(1 / (1 + math.exp(-1 / 1.3)))


@pytest.fixture
def tiny_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    tokenizers = pytest.importorskip("tokenizers")
    from safetensors.torch import save_file

    from runtime.lfm3_reference import Lfm3TextReference

    data = config()
    # A real tokenizer and real random safetensors, not a mocked forward.
    words = [
        "<unk>",
        "<|startoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "yes",
        "Yes",
        "YES",
        "no",
        "No",
        "NO",
        "A",
        "B",
        "C",
        "0",
        "1",
        "2",
        "user",
        "assistant",
        "blue",
        "ticket",
        "QUESTION",
        "Refund",
        "Team",
        "Urgency",
        "low",
        "medium",
        "high",
        "Billing",
        "support",
        "think",
        "Reply",
        "with",
        "only",
        "Options",
        "the",
        "option",
        "code",
    ]
    raw = tokenizers.Tokenizer(
        tokenizers.models.WordLevel({s: i for i, s in enumerate(words)}, unk_token="<unk>")
    )
    raw.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    raw.add_special_tokens(["<|startoftext|>", "<|im_start|>", "<|im_end|>"])
    raw.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "config.json").write_text(json.dumps(data), encoding="utf-8")
    (tmp_path / "calibration.json").write_text(
        json.dumps({"default": 1.3, "temperatures": {}}), encoding="utf-8"
    )
    torch.manual_seed(1001)
    model = Lfm3TextReference(D1Contract.parse(data).text).eval()
    save_file(
        {"model.language_model." + k: v.contiguous() for k, v in model.state_dict().items()},
        str(tmp_path / "model.safetensors"),
    )
    return tmp_path, model


def test_actual_small_hf_load_forward_slots_and_grouping(tiny_checkpoint):
    directory, original = tiny_checkpoint
    runtime = D1TextReference.load(directory)
    payload = request()
    out = runtime.decide(payload)
    assert set(out["answers"]) == set(payload["questions"])
    assert out["usage"]["output_tokens"] == 0 and out["usage"]["input_tokens"] > 0
    assert 0 <= out["answers"]["refund"]["noul"] <= 1
    assert sum(out["answers"]["team"]["probabilities"].values()) == pytest.approx(1)
    for actual, expected in zip(
        runtime.model.state_dict().values(), original.state_dict().values(), strict=True
    ):
        assert actual.equal(expected)
    assert all(
        b.device.type == "cpu" for b in runtime.model.buffers()
    )  # no unmaterialized meta RoPE
    rows = pack_requests(runtime.tokenizer, runtime.contract, [payload])
    assert len(rows) == 1 and len(rows[0]["slots"]) == 3
    lots = {
        "state": "blue",
        "questions": {f"q{i}": payload["questions"]["refund"] for i in range(17)},
    }
    packed = pack_requests(runtime.tokenizer, runtime.contract, [lots])
    assert [len(r["slots"]) for r in packed] == [16, 1]
    assert packed[1]["ids"][: packed[1]["state"]] == packed[0]["ids"][: packed[0]["state"]]
    assert runtime.decide(payload) == out  # no mutable prefix or conv history between requests
    assert runtime.decide_batch([payload, payload]) == [out, out]


def test_real_checkpoint_opt_in_only():
    import os

    path = os.environ.get("TYLOINFER_D1_MODEL")
    if not path:
        pytest.skip("real d1-2B weights not provided; tiny fixtures are not real-model validation")
    runtime = D1TextReference.load(Path(path))
    out = runtime.decide(request())
    assert out["usage"]["output_tokens"] == 0
    assert set(out["answers"]) == set(request()["questions"])
