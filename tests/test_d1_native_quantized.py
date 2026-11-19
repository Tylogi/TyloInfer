"""Small Q8_0 fixtures exercise packed projection tails and token chunk boundaries."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def prepare(directory: Path):
    import gguf
    import numpy as np
    import torch

    from tests.test_d1_reference import config, tiny_checkpoint
    from tyloinfer.runtime.d1_contract import D1Contract
    from tyloinfer.runtime.lfm3_reference import Lfm3TextReference, inverse_frequencies

    directory.mkdir(parents=True, exist_ok=True)
    base = directory / "base"
    base.mkdir(exist_ok=True)
    tiny_checkpoint.__wrapped__(base)
    for name in ("tokenizer.json", "calibration.json"):
        (directory / name).write_bytes((base / name).read_bytes())
    cfg = config()
    cfg["text_config"].update(hidden_size=32, intermediate_size=64, head_dim=8)
    (directory / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    contract = D1Contract.parse(cfg)
    torch.manual_seed(1012)
    model = Lfm3TextReference(contract.text).to(torch.bfloat16).eval()
    suffixes = {
        "operator_norm": "attn_norm",
        "post_operator_norm": "post_attention_norm",
        "ffn_norm": "ffn_norm",
        "post_ffn_norm": "post_ffw_norm",
        "feed_forward.w1": "ffn_gate",
        "feed_forward.w2": "ffn_down",
        "feed_forward.w3": "ffn_up",
        "self_attn.q_proj": "attn_q",
        "self_attn.k_proj": "attn_k",
        "self_attn.v_proj": "attn_v",
        "self_attn.o_proj": "attn_output",
        "self_attn.gate_proj": "attn_gate",
        "self_attn.q_norm": "attn_q_norm",
        "self_attn.k_norm": "attn_k_norm",
        "conv.conv": "shortconv.conv",
        "conv.in_proj": "shortconv.in_proj",
        "conv.out_proj": "shortconv.out_proj",
    }
    writer = gguf.GGUFWriter(str(directory / "weights.gguf"), "lfm3")
    writer.add_string("lfm3.decision.type", "lfm3-d1")
    for name, value in {
        "block_count": 3,
        "context_length": 2048,
        "embedding_length": 32,
        "feed_forward_length": 64,
        "attention.head_count": 4,
        "attention.key_length": 8,
        "attention.value_length": 8,
        "shortconv.l_cache": 7,
        "attention.sliding_window": 4,
        "rope.dimension_count": 8,
    }.items():
        writer.add_uint32("lfm3." + name, value)
    for name, value in {
        "rope.freq_base": 1e6,
        "rope.freq_base_swa": 1e4,
        "attention.layer_norm_rms_epsilon": 1e-5,
    }.items():
        writer.add_float32("lfm3." + name, value)
    writer.add_bool("lfm3.attention.causal", False)
    writer.add_key_value(
        "lfm3.attention.head_count_kv",
        [0, 2, 2],
        gguf.GGUFValueType.ARRAY,
        gguf.GGUFValueType.INT32,
    )
    writer.add_array("lfm3.attention.sliding_window_pattern", [False, False, True])
    writer.add_tensor("rope_freqs.weight", np.array([1, 1, 1e30, 1e30], dtype=np.float32))
    decoded = {}
    packed_bytes = 0
    for name, tensor in model.state_dict().items():
        if name == "embed_tokens.weight":
            target = "token_embd.weight"
        elif name == "embedding_norm.weight":
            target = "output_norm.weight"
        else:
            _, layer, suffix = name.split(".", 2)
            target = f"blk.{layer}.{suffixes[suffix.removesuffix('.weight')]}.weight"
        array = tensor.float().numpy()
        if tensor.ndim == 2:
            packed = gguf.quants.quantize(array, gguf.GGMLQuantizationType.Q8_0)
            writer.add_tensor(target, packed, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
            packed_bytes += packed.nbytes
            array = gguf.quants.dequantize(packed, gguf.GGMLQuantizationType.Q8_0)
        else:
            writer.add_tensor(target, array.squeeze(1) if tensor.ndim == 3 else array)
        decoded[name] = torch.from_numpy(array.copy()).reshape(tensor.shape).to(torch.bfloat16)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    model.load_state_dict(decoded, assign=True)
    for kind, layer in zip(contract.text.layer_types, model.layers, strict=True):
        if kind != "conv":
            layer.self_attn.frequencies = inverse_frequencies(contract.text, kind)
    cases = []
    with torch.inference_mode():
        for n in (257, 513):
            ids = (torch.arange(n) % 63 + 1)[None]
            groups = torch.tensor([[-1] * (n // 2) + [1] * (n // 4) + [2] * (n - n // 2 - n // 4)])
            state = torch.tensor([n // 2])
            hidden = model(ids, torch.ones_like(ids, dtype=torch.bool), state, groups)
            positions = torch.tensor([n // 2 + 1, n - 1])
            candidates = torch.tensor([2, 3, 4, 5])
            logits = (
                model.slot_logits(hidden, torch.zeros_like(positions), positions, candidates)
                .float()
                .tolist()
            )
            cases.append(
                {
                    "request": {
                        "tokens": ids[0].tolist(),
                        "question_ids": groups[0].tolist(),
                        "state_tokens": int(state[0]),
                        "slots": positions.tolist(),
                        "candidate_ids": candidates.tolist(),
                        "warmup": 0,
                        "iterations": 1,
                    },
                    "expected_logits": logits,
                }
            )
    (directory / "fixture.json").write_text(json.dumps(cases), encoding="utf-8")
    return packed_bytes


def test_q8_fixture_preserves_packed_matrix_bytes(tmp_path):
    import pytest

    pytest.importorskip("torch")
    gguf = pytest.importorskip("gguf")
    expected = prepare(tmp_path)
    reader = gguf.GGUFReader(str(tmp_path / "weights.gguf"))
    matrices = [t for t in reader.tensors if t.tensor_type == gguf.GGMLQuantizationType.Q8_0]
    assert expected == sum(t.data.nbytes for t in matrices)
    assert len(matrices) == 22


def test_quantized_native_gpu_opt_in(tmp_path):
    import pytest

    from tests.test_d1_native_gpu import check

    executable = os.environ.get("TYLOINFER_D1_NATIVE_BINARY")
    if not executable:
        pytest.skip("native d1 GPU binary not provided")
    pytest.importorskip("gguf")
    prepare(tmp_path)
    assert len(check(Path(executable), tmp_path, tmp_path / "weights.gguf")) == 4


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", type=Path)
    parser.add_argument("--check", type=Path)
    parser.add_argument("--executable", type=Path)
    args = parser.parse_args()
    if args.prepare:
        print(json.dumps({"packed_bytes": prepare(args.prepare)}))
    elif args.check and args.executable:
        from tests.test_d1_native_gpu import check

        print(json.dumps(check(args.executable, args.check, args.check / "weights.gguf")))
    else:
        parser.error("provide --prepare or --check and --executable")
