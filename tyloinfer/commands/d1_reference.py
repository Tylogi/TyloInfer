"""Explicit opt-in d1 text reference command, not native serving support."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def add_parser(sub):
    parser = sub.add_parser(
        "d1-reference",
        help="experimental local HF d1 text reference (not native CUDA/Metal serving)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="local HF d1 checkpoint; never downloaded automatically",
    )
    parser.add_argument(
        "--request", type=Path, help="JSON request or request list; stdin when omitted"
    )
    parser.add_argument(
        "--device", default="cpu", help="explicit PyTorch device, e.g. cpu or cuda:0"
    )
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="validate only model/calibration configuration; do not load weights",
    )
    parser.set_defaults(_impl=run)
    from tyloinfer.commands.d1_native import add_parser as add_native_parser

    add_native_parser(sub)


def run(args):
    from tyloinfer.runtime.d1_contract import D1_REVISION, D1Calibration, D1Contract
    from tyloinfer.runtime.d1_reference import D1TextReference, validate_request

    try:
        if args.check_config:
            contract = D1Contract.load(args.model)
            D1Calibration.parse(
                json.loads((args.model / "calibration.json").read_text(encoding="utf-8"))
            )
            result = {
                "config_valid": True,
                "backbone": "lfm3",
                "layers": len(contract.text.layer_types),
                "set_size": contract.set_size,
                "weights_loaded": False,
            }
        else:
            text = args.request.read_text(encoding="utf-8") if args.request else sys.stdin.read()
            payload = json.loads(text)
            requests = payload if isinstance(payload, list) else [payload]
            if not requests:
                raise ValueError("d1 request batch is empty")
            for request in requests:
                validate_request(request)
            runtime = D1TextReference.load(args.model, device=args.device, dtype=args.dtype)
            done = runtime.decide_batch(requests)
            result = done if isinstance(payload, list) else done[0]
        print(
            json.dumps(
                {
                    "backend": D1TextReference.backend,
                    "reference_revision": D1_REVISION,
                    "native_runtime": False,
                    "result": result,
                },
                ensure_ascii=False,
                allow_nan=False,
            )
        )
        return 0
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        print(f"d1-reference: {exc}", file=sys.stderr)
        return 2
