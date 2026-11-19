"""Standalone native d1 text command, separate from the serving engine."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def add_parser(sub):
    parser = sub.add_parser("d1-native", help="standalone CUDA/cuBLAS d1 text decisions")
    parser.add_argument("--model", type=Path, required=True, help="local matching HF assets")
    parser.add_argument("--weights", type=Path, help="local HF directory or official Q8_0 GGUF")
    parser.add_argument("--weights-placement", choices=("stream", "vram"), default="vram",
                        help="two streamed layer buffers or all weights resident in VRAM")
    parser.add_argument(
        "--executable", type=Path, required=True, help="built tyloinfer-d1-native binary"
    )
    parser.add_argument(
        "--request", type=Path, help="JSON request or request list; stdin otherwise"
    )
    parser.add_argument(
        "--unfused", action="store_true", help="disable projection fusion for comparison"
    )
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument(
        "--timeout", type=float, default=300, help="native response timeout in seconds"
    )
    parser.set_defaults(_impl=run)


def run(args):
    from tyloinfer.runtime.d1_native import D1Native
    from tyloinfer.runtime.d1_reference import validate_request

    try:
        if not 0 <= args.warmup <= 100 or not 1 <= args.iterations <= 1000:
            raise ValueError("invalid native d1 benchmark iteration count")
        text = args.request.read_text(encoding="utf-8") if args.request else sys.stdin.read()
        payload = json.loads(text)
        requests = payload if isinstance(payload, list) else [payload]
        if not requests:
            raise ValueError("d1 request batch is empty")
        for request in requests:
            validate_request(request)
        with D1Native(
            args.model,
            args.executable,
            weights=args.weights,
            fused=not args.unfused,
            weight_placement=args.weights_placement,
            timeout=args.timeout,
        ) as runtime:
            done = runtime.decide_batch(requests, warmup=args.warmup, iterations=args.iterations)
            result = {
                **runtime.ready,
                "result": done if isinstance(payload, list) else done[0],
                "serving_integration": False,
            }
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        print(f"d1-native: {exc}", file=sys.stderr)
        return 2
