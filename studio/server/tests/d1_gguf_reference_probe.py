"""Independent CPU BF16 reference for the standalone Q8_0 loading path.

Requires optional torch, safetensors, tokenizers, numpy and gguf packages.
This is a correctness probe, never a CPU performance proxy for CUDA.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def load_reference(args):
    import gguf
    import torch

    from server.runtime.d1_contract import D1Calibration, D1Contract
    from server.runtime.d1_reference import LocalTokenizer
    from server.runtime.lfm3_reference import Lfm3TextReference, inverse_frequencies

    torch.set_num_threads(args.threads)
    contract = D1Contract.load(args.model)
    tokenizer = LocalTokenizer(args.model)
    calibration = D1Calibration.parse(json.loads((args.model / "calibration.json").read_text()))
    reader = gguf.GGUFReader(str(args.weights))
    available = {t.name: t for t in reader.tensors}
    with torch.device("meta"):
        model = Lfm3TextReference(contract.text)
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
    state = {}
    for name, value in model.state_dict().items():
        if name == "embed_tokens.weight":
            source = "token_embd.weight"
        elif name == "embedding_norm.weight":
            source = "output_norm.weight"
        else:
            match = re.fullmatch(r"layers\.(\d+)\.(.+)\.weight", name)
            if not match:
                raise ValueError(f"unknown reference tensor: {name}")
            source = f"blk.{match[1]}.{suffixes[match[2]]}.weight"
        tensor = available[source]
        array = gguf.quants.dequantize(tensor.data, tensor.tensor_type)
        precision = (
            torch.float16
            if args.projection_dtype == "float16"
            and value.ndim == 2
            and name != "embed_tokens.weight"
            else torch.bfloat16
        )
        state[name] = torch.from_numpy(array.copy()).reshape(value.shape).to(precision)
    model.load_state_dict(state, strict=True, assign=True)
    del state
    for kind, layer in zip(contract.text.layer_types, model.layers, strict=True):
        if kind != "conv":
            layer.self_attn.frequencies = inverse_frequencies(contract.text, kind)
    if args.projection_dtype == "float16":
        from types import MethodType

        from torch.nn import functional

        def projection(self, x):
            return functional.linear(x.to(torch.float16), self.weight).to(torch.bfloat16)

        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                module.forward = MethodType(projection, module)
    model.eval()
    return model, contract, tokenizer, calibration


def main(args):
    import torch

    from server.tests.d1_native_real_probe import requests
    from server.runtime.d1_reference import pack_requests, read_answer

    model, contract, tokenizer, calibration = load_reference(args)
    results = []
    with torch.inference_mode():
        for row in pack_requests(tokenizer, contract, requests()):
            ids = torch.tensor([row["ids"]])
            hidden = model(
                ids,
                torch.ones_like(ids, dtype=torch.bool),
                torch.tensor([row["state"]]),
                torch.tensor([row["questions"]]),
            )
            candidates = sorted({t for s in row["slots"] for group in s[4] for t in group})
            positions = torch.tensor([s[3] for s in row["slots"]])
            logits = (
                model.slot_logits(
                    hidden, torch.zeros_like(positions), positions, torch.tensor(candidates)
                )
                .float()
                .tolist()
            )
            answers = {}
            for slot, values in zip(row["slots"], logits, strict=True):
                answers[slot[1]] = read_answer(
                    slot[2], dict(zip(candidates, values, strict=True)), slot[4], calibration
                )
            results.append(
                {
                    "tokens": len(row["ids"]),
                    "candidate_ids": candidates,
                    "logits": logits,
                    "answers": answers,
                }
            )
            print(json.dumps(results[-1]), flush=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--projection-dtype", choices=("bfloat16", "float16"), default="bfloat16")
    main(parser.parse_args())
