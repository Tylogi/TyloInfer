"""Experimental HF text d1 reference execution, separate from native D3 serving.

Prompt/readout compatibility derived from LiquidAI/d1-2B prompt.py/api.py at
 a6bce774e6dde338dcc3036ceb5d91f78bf4c19e; modified text-only implementation.
Those prompt/readout portions are under LFM Open License v1.0; see
LICENSES/LiquidAI-d1-LFM-Open-License.txt. No custom checkpoint code executes.
"""

from __future__ import annotations

import json
import math
import re
import string
from pathlib import Path
from typing import Any

from runtime.d1_contract import D1Calibration, D1Contract, median_level, positive_int


class LocalTokenizer:
    def __init__(self, directory: Path):
        from tokenizers import Tokenizer

        self.raw = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self.specials = [
            t.content
            for t in self.raw.get_added_tokens_decoder().values()
            if t.special and len(t.content) > 1
        ]
        self._pattern = (
            re.compile("|".join(re.escape(s) for s in sorted(self.specials, key=len, reverse=True)))
            if self.specials
            else None
        )

    def encode(self, text: str) -> list[int]:
        return self.raw.encode(text, add_special_tokens=False).ids

    def plain(self, text: str) -> str:
        if self._pattern:
            while self._pattern.search(text):
                text = self._pattern.sub(lambda m: m[0][0] + "\u200b" + m[0][1:], text)
        return text


def validate_request(payload: Any) -> dict:
    if not isinstance(payload, dict) or set(payload) - {"state", "questions", "set_size"}:
        raise ValueError(
            "d1 text requests accept state, questions and optional set_size only; no cache/media/generation options"
        )
    if "state" not in payload:
        raise ValueError("d1 requests require a state")
    # Reject non-JSON values and NaN/Inf rather than rendering different states.
    json.dumps(payload["state"], allow_nan=False, ensure_ascii=False)
    if payload["state"] is None:
        raise ValueError(
            "text-only d1 reference requires a state; media-only requests are unsupported"
        )
    questions = payload.get("questions")
    if (
        not isinstance(questions, dict)
        or not questions
        or any(not isinstance(n, str) or not n for n in questions)
    ):
        raise ValueError("d1 questions must be a nonempty object with nonempty names")
    for q in questions.values():
        if not isinstance(q, dict) or set(q) - {"type", "instructions", "criteria"}:
            raise ValueError(
                "d1 reference supports only text noul, single-choice and score questions"
            )
        kind = q.get("type")
        if kind not in ("noul", "choice", "score") or not isinstance(q.get("instructions"), str):
            raise ValueError("d1 reference supports text instructions and noul/choice/score only")
        criteria = q.get("criteria")
        if kind == "choice" and (
            not isinstance(criteria, dict)
            or not criteria
            or any(
                not isinstance(k, str) or not k or (v is not None and not isinstance(v, str))
                for k, v in criteria.items()
            )
        ):
            raise ValueError("choice criteria must map nonempty labels to descriptions or null")
        if kind == "score" and (
            not isinstance(criteria, list)
            or not 2 <= len(criteria) <= 10
            or any(not isinstance(v, str) for v in criteria)
        ):
            raise ValueError("score criteria must list 2 to 10 levels, lowest first")
        if (
            kind == "noul"
            and criteria is not None
            and (
                not isinstance(criteria, dict)
                or set(criteria) != {"true", "false"}
                or any(not isinstance(v, str) for v in criteria.values())
            )
        ):
            raise ValueError("noul criteria must name true and false descriptions")
    if "set_size" in payload:
        positive_int(payload["set_size"], "set_size")
    return payload


def aliases(tokenizer, labels: list[str], scheme: int) -> list[tuple[str, int]]:
    labels = [s.strip() for s in labels]
    letters = list(string.ascii_uppercase)
    if labels and all(len(s) == 1 and s.isalpha() for s in labels):
        codes = labels
    elif len(labels) <= 26:
        codes = letters[: len(labels)]
    elif scheme == 2:
        codes = (letters + [a + b for a in letters for b in letters])[: len(labels)]
    else:
        codes = [f"{i:02d}" for i in range(len(labels))]
    if len(codes) != len(labels):
        raise ValueError("choice exceeds this d1 code scheme")
    fallback = (
        letters
        + [f"{i:02d}" for i in range(100)]
        + list(string.ascii_lowercase)
        + [f"#{i}" for i in range(200)]
        + [a + b for a in letters for b in letters]
    )
    used, out = set(), []
    for code in codes:
        for candidate in [code] + fallback:
            ids = tokenizer.encode(candidate)
            if len(ids) == 1 and ids[0] not in used:
                used.add(ids[0])
                out.append((candidate, ids[0]))
                break
        else:
            raise ValueError("tokenizer has no distinct single-token answer alias")
    return out


def question_layout(tokenizer, q: dict, scheme: int) -> tuple[str, list[list[int]]]:
    kind, instructions = q["type"], q["instructions"]
    criteria = q.get("criteria")
    if kind == "choice":
        codes = aliases(tokenizer, list(criteria), scheme)
        lines = [
            f"{code} {criteria[label] or label.replace('_', ' ')}"
            for (code, _), label in zip(codes, criteria, strict=True)
        ]
        body = (
            instructions
            + "\n\nOptions:\n"
            + "\n".join(lines)
            + "\n\nReply with the option code only."
        )
        groups = []
        for code, token in codes:
            extra = tokenizer.encode(" " + code)
            groups.append([token] + ([extra[0]] if len(extra) == 1 and extra[0] != token else []))
    elif kind == "noul":
        extra = f"\nYes: {criteria.get('true')}\nNo: {criteria.get('false')}" if criteria else ""
        body = instructions + extra + "\n\nReply with yes or no only."
        groups = []
        for forms in (("yes", "Yes", "YES"), ("no", "No", "NO")):
            groups.append(
                list(
                    dict.fromkeys(
                        ids[0] for text in forms if len(ids := tokenizer.encode(text)) == 1
                    )
                )
            )
    else:
        body = (
            instructions
            + "\n\n"
            + "\n".join(f"{i} {s}" for i, s in enumerate(criteria))
            + f"\n\nReply with a single digit 0-{len(criteria) - 1} only."
        )
        groups = [tokenizer.encode(str(i)) for i in range(len(criteria))]
        if any(len(ids) != 1 for ids in groups):
            raise ValueError("score needs single-token digits")
    if any(not g for g in groups):
        raise ValueError("tokenizer has no single-token answer forms")
    suffix = tokenizer.plain(body) + "<|im_end|>\n<|im_start|>assistant\n<think></think>"
    return suffix, groups


def pack_requests(tokenizer, contract: D1Contract, requests: list[dict]) -> list[dict]:
    """Groups of questions, including repeats of the state for groups past set_size."""
    rows = []
    bos = tokenizer.raw.id_to_token(contract.bos_token_id)
    if bos is None:
        raise ValueError("tokenizer does not contain the configured BOS token")
    for request_id, payload in enumerate(requests):
        validate_request(payload)
        state = payload["state"]
        header = "" if state is None else "QUESTION:\n"
        state_text = (
            ""
            if state is None
            else (
                state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=2)
            )
            + "\n\n"
        )
        prefix = (
            bos
            + "<|im_start|>user\n"
            + (tokenizer.plain(state_text) + "\n" + header if state is not None else "")
        )
        layouts = [
            (name, q, *question_layout(tokenizer, q, contract.codes))
            for name, q in payload["questions"].items()
        ]
        step = payload.get("set_size", contract.set_size)
        for start in range(0, len(layouts), step):
            group = layouts[start : start + step]
            first = tokenizer.encode(prefix + group[0][2])
            boundary = len(first) - len(tokenizer.encode(group[0][2]))
            if not 0 <= boundary < len(first):
                raise ValueError("invalid state/question token boundary")
            ids, question_ids = list(first), [-1] * boundary + [1] * (len(first) - boundary)
            slots = [(request_id, group[0][0], group[0][1], len(ids) - 1, group[0][3])]
            for number, (name, q, suffix, candidates) in enumerate(group[1:], 2):
                own = tokenizer.encode(header + suffix)
                if not own:
                    raise ValueError("empty question tokens")
                ids.extend(own)
                question_ids.extend([number] * len(own))
                slots.append((request_id, name, q, len(ids) - 1, candidates))
            if len(ids) > contract.text.max_position_embeddings:
                raise ValueError(
                    "d1 packed questions exceed model context; reduce set_size or input length"
                )
            rows.append(
                {
                    "ids": ids,
                    "state": boundary,
                    "questions": question_ids,
                    "slots": slots,
                    "request": request_id,
                }
            )
    return rows


def read_answer(
    q: dict, logits: list[float], candidates: list[list[int]], calibration: D1Calibration
) -> dict:
    scores = [max(float(logits[token]) for token in group) for group in candidates]
    if any(not math.isfinite(s) for s in scores):
        raise ValueError("non-finite d1 candidate logits")
    temperature = calibration.temperature(q["type"], len(candidates))
    scores = [s / temperature for s in scores]
    maximum = max(scores)
    exps = [math.exp(s - maximum) for s in scores]
    probabilities = [p / math.fsum(exps) for p in exps]
    kind = q["type"]
    if kind == "noul":
        return {"type": kind, "noul": probabilities[0]}
    best = max(range(len(probabilities)), key=probabilities.__getitem__)
    if kind == "choice":
        labels = list(q["criteria"])
        return {
            "type": kind,
            "choice": labels[best],
            "confidence": probabilities[best],
            "probabilities": dict(zip(labels, probabilities, strict=True)),
        }
    return {
        "type": kind,
        "score": float(median_level(probabilities)),
        "confidence": probabilities[best],
        "probabilities": {str(i): p for i, p in enumerate(probabilities)},
        "legend": {str(i): s for i, s in enumerate(q["criteria"])},
    }


class D1TextReference:
    """Actual LFM3 execution of local HF weights; explicitly not native serving."""

    backend = "pytorch-lfm3-text-reference"

    def __init__(self, contract, model, tokenizer, calibration):
        self.contract, self.model, self.tokenizer, self.calibration = (
            contract,
            model,
            tokenizer,
            calibration,
        )

    @classmethod
    def load(cls, directory: str | Path, *, device: str = "cpu", dtype: str = "float32"):
        import torch
        from safetensors import safe_open

        from runtime.lfm3_reference import Lfm3TextReference

        root = Path(directory).resolve(strict=True)
        if not root.is_dir():
            raise ValueError("d1 reference execution requires a local HF checkpoint directory")
        contract = D1Contract.load(root)
        if not contract.text.tie_word_embeddings:
            raise ValueError("reference d1 currently requires tied output embeddings")
        if dtype not in ("float32", "bfloat16"):
            raise ValueError("d1 reference dtype must be float32 or bfloat16")
        precision = getattr(torch, dtype)
        calibration = D1Calibration.parse(
            json.loads((root / "calibration.json").read_text(encoding="utf-8"))
        )
        tokenizer = LocalTokenizer(root)
        tokenizer.bos = tokenizer.raw.id_to_token(contract.bos_token_id)
        if tokenizer.bos is None:
            raise ValueError("d1 BOS token is missing from tokenizer.json")
        # Meta construction avoids allocating and then discarding a full random model.
        with torch.device("meta"):
            model = Lfm3TextReference(contract.text)
        expected = set(model.state_dict())
        tensors = {}
        index = root / "model.safetensors.index.json"
        if index.is_file():
            manifest = json.loads(index.read_text(encoding="utf-8"))
            mapping = manifest.get("weight_map") if isinstance(manifest, dict) else None
            if (
                not isinstance(mapping, dict)
                or not mapping
                or any(not isinstance(v, str) for v in mapping.values())
            ):
                raise ValueError("invalid safetensors weight_map")
            names = sorted(set(mapping.values()))
        else:
            names = ["model.safetensors"]
        for name in names:
            relative = Path(name)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or relative.suffix != ".safetensors"
            ):
                raise ValueError("invalid safetensors shard path")
            path = (root / relative).resolve(strict=True)
            if not path.is_relative_to(root):
                raise ValueError("safetensors shard escapes the checkpoint directory")
            with safe_open(str(path), framework="pt", device="cpu") as shard:
                for source in shard.keys():  # noqa: SIM118 -- safetensors.safe_open is not iterable
                    prefix = "model.language_model."
                    if not source.startswith(prefix):
                        continue
                    key = source[len(prefix) :]
                    if key not in expected:
                        raise ValueError(f"unexpected LFM3 language weight: {source}")
                    if key in tensors:
                        raise ValueError(f"duplicate LFM3 weight: {source}")
                    value = shard.get_tensor(source)
                    if tuple(value.shape) != tuple(
                        model.state_dict()[key].shape
                    ) or value.dtype not in (torch.float32, torch.bfloat16, torch.float16):
                        raise ValueError(f"invalid LFM3 weight shape/dtype: {source}")
                    tensors[key] = value.to(device=device, dtype=precision)
        missing = expected - set(tensors)
        if missing:
            raise ValueError(f"missing LFM3 language weights: {sorted(missing)[:5]}")
        model.load_state_dict(tensors, strict=True, assign=True)
        # Nonpersistent RoPE buffers were constructed on meta and are not in
        # the checkpoint. Materialize them explicitly on the chosen device.
        from runtime.lfm3_reference import inverse_frequencies

        for kind, layer in zip(contract.text.layer_types, model.layers, strict=True):
            if kind != "conv":
                layer.self_attn.frequencies = inverse_frequencies(
                    contract.text, kind, device=device
                )
        model.eval()
        return cls(contract, model, tokenizer, calibration)

    def decide_batch(self, requests: list[dict]) -> list[dict]:
        import torch

        if not isinstance(requests, list) or not requests:
            raise ValueError("d1 batch must contain at least one request")
        rows = pack_requests(self.tokenizer, self.contract, requests)
        results = [
            {"answers": {}, "usage": {"input_tokens": 0, "output_tokens": 0}} for _ in requests
        ]
        device = self.model.embed_tokens.weight.device
        # One packed question set per forward keeps reference memory bounded.
        # This is not a continuous-batching implementation or a throughput claim.
        with torch.inference_mode():
            for row in rows:
                ids = torch.tensor([row["ids"]], device=device)
                real = torch.ones_like(ids, dtype=torch.bool)
                state = torch.tensor([row["state"]], device=device)
                groups = torch.tensor([row["questions"]], device=device)
                hidden = self.model(ids, real, state, groups, use_cache=False)
                candidates = sorted(
                    {
                        t
                        for _, _, _, _, options in row["slots"]
                        for option in options
                        for t in option
                    }
                )
                selected = torch.tensor(candidates, device=device)
                positions = torch.tensor([slot[3] for slot in row["slots"]], device=device)
                logits = (
                    self.model.slot_logits(hidden, torch.zeros_like(positions), positions, selected)
                    .float()
                    .cpu()
                    .tolist()
                )
                for (_, name, question, _, options), values in zip(
                    row["slots"], logits, strict=True
                ):
                    sparse = dict(zip(candidates, values, strict=True))
                    # read_answer only indexes candidate IDs, never constructs T x vocab.
                    results[row["request"]]["answers"][name] = read_answer(
                        question, sparse, options, self.calibration
                    )
                results[row["request"]]["usage"]["input_tokens"] += len(row["ids"])
        return results

    def decide(self, payload: dict) -> dict:
        return self.decide_batch([payload])[0]
