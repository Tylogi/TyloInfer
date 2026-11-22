#!/usr/bin/env python3
"""Generate a short, independent Qwen3.5 reference completion.

This helper intentionally uses the upstream Transformers implementation and the
original Hugging Face checkpoint.  It must not invoke TyloInfer/native runtime
code or derive expected tokens from a native completion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any


DEFAULT_PROMPT = (
    "<|im_start|>user\nCount from one to ten.<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)
DEFAULT_MAX_NEW_TOKENS = 8
REFERENCE_TRANSFORMERS_VERSION = "5.24.0"
REFERENCE_SOURCE_COMMIT = "89c632fd826f68ecadd176320e9b2c070918f4f2"


def sha256(path: Path) -> str:
    """Return the SHA-256 digest for a local file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate(
    model_dir: Path, prompt: str, max_new_tokens: int, dtype_name: str
) -> dict[str, Any]:
    """Run deterministic greedy generation using upstream Transformers."""

    import torch
    import transformers
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM

    if transformers.__version__ != REFERENCE_TRANSFORMERS_VERSION:
        raise ValueError(
            "reference provenance requires Transformers "
            f"{REFERENCE_TRANSFORMERS_VERSION}, found {transformers.__version__}"
        )

    try:
        dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype_name]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {dtype_name}") from exc

    tokenizer_path = model_dir / "tokenizer.json"
    config_path = model_dir / "config.json"
    weights_path = model_dir / "model.safetensors-00001-of-00001.safetensors"
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False
    )
    model = Qwen3_5ForCausalLM.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=dtype,
    )
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_k=1,
            top_p=1.0,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )
    prompt_ids = inputs["input_ids"][0].tolist()
    generated_ids = output.sequences[0, len(prompt_ids) :].tolist()
    top2_margins = []
    for scores in output.scores:
        values = scores[0].topk(2).values if scores.ndim == 2 else scores.topk(2).values
        top2_margins.append(float((values[0] - values[1]).item()))
    return {
        "prompt": prompt,
        "prompt_token_ids": prompt_ids,
        "generated_token_ids": generated_ids,
        "generated_text": tokenizer.decode(generated_ids, skip_special_tokens=False),
        "top2_logit_margins": top2_margins,
        "max_new_tokens": max_new_tokens,
        "sampling": {
            "do_sample": False,
            "temperature": 0,
            "top_k": 1,
            "top_p": 1,
        },
        "model": {
            "config_sha256": sha256(config_path),
            "weights_sha256": sha256(weights_path),
            "tokenizer_sha256": sha256(tokenizer_path),
            "config": json.loads(config_path.read_text(encoding="utf-8")),
        },
        "reference": {
            "implementation": "huggingface/transformers",
            "source_repository": "https://github.com/huggingface/transformers",
            "source_revision": "v5.24.0",
            "source_commit": REFERENCE_SOURCE_COMMIT,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "torch_dtype": str(next(model.parameters()).dtype),
            "device": "cpu",
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
    }


def main() -> int:
    """Parse arguments, generate the fixture, and optionally write JSON."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.max_new_tokens > 64:
        parser.error("--max-new-tokens must be between 1 and 64")
    result = generate(args.model_dir.resolve(), args.prompt, args.max_new_tokens, args.dtype)
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        sys.stdout.write(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
