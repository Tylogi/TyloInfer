"""Opt-in real-weight native validation/benchmark; outputs are local evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tyloinfer.runtime.d1_native import D1Native
from tyloinfer.runtime.d1_reference import pack_requests, read_answer


def requests():
    teams = {
        "billing": "Charges and refunds",
        "shipping": "Delivery and parcels",
        "technical": "App or site faults",
    }
    return [
        {
            "state": "I was charged twice. Please refund the duplicate payment.",
            "questions": {
                "refund": {"type": "noul", "instructions": "Is a refund requested?"},
                "team": {
                    "type": "choice",
                    "instructions": "Which team should handle this?",
                    "criteria": teams,
                },
                "urgency": {
                    "type": "score",
                    "instructions": "How urgent is this ticket?",
                    "criteria": ["low", "medium", "high"],
                },
            },
        },
        {
            "state": {
                "message": "The parcel has not arrived after two weeks.",
                "language": "English",
            },
            "questions": {
                "delivery": {"type": "noul", "instructions": "Is this about delivery?"},
                "team": {"type": "choice", "instructions": "Which team?", "criteria": teams},
            },
        },
    ]


def run(args):
    output = []
    for fused in (False, True):
        with D1Native(args.model, args.executable, weights=args.weights, fused=fused) as runtime:
            result = {"ready": runtime.ready, "rows": []}
            for row in pack_requests(runtime.tokenizer, runtime.contract, requests()):
                candidates = sorted({t for s in row["slots"] for group in s[4] for t in group})
                frame = runtime.evaluate(
                    row, candidates, warmup=args.warmup, iterations=args.iterations
                )
                answers = {}
                for slot, values in zip(row["slots"], frame["logits"], strict=True):
                    answers[slot[1]] = read_answer(
                        slot[2],
                        dict(zip(candidates, values, strict=True)),
                        slot[4],
                        runtime.calibration,
                    )
                result["rows"].append(
                    {"tokens": len(row["ids"]), "answers": answers, "frame": frame}
                )
            output.append(result)
            print(json.dumps(result), flush=True)
        args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=15)
    run(parser.parse_args())
