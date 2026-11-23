"""Opt-in tests against pinned upstream code/tokenizer, never model weights."""

from __future__ import annotations

import ast
import importlib
import json
import os
import sys
import types
from pathlib import Path

import pytest

from runtime.d1_contract import D1Calibration, D1Contract
from runtime.d1_reference import (
    LocalTokenizer,
    pack_requests,
    question_layout,
    read_answer,
)


@pytest.fixture(scope="module")
def official():
    directory = os.environ.get("TYLOINFER_D1_REFERENCE_SOURCE")
    if not directory:
        pytest.skip("pinned official d1 source/tokenizer not provided")
    root = Path(directory)
    if not (root / "tokenizer.json").is_file():
        pytest.skip("official tokenizer.json not provided")
    name = "_tyloinfer_official_d1_text_parity"
    package = types.ModuleType(name)
    package.__path__ = [str(root)]
    sys.modules[name] = package
    modules = {
        key: importlib.import_module(name + "." + key) for key in ("prompt", "calibration", "api")
    }
    tree = ast.parse((root / "attention.py").read_text(encoding="utf-8"))
    nodes = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name in ("pack_set", "state_length")
    ]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    scope = {}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[future] + nodes, type_ignores=[])),
            str(root / "attention.py"),
            "exec",
        ),
        scope,
    )
    modules["pack_set"] = scope["pack_set"]
    modules["root"] = root
    yield modules
    for key in list(sys.modules):
        if key == name or key.startswith(name + "."):
            del sys.modules[key]


class OfficialTokenizerAdapter(LocalTokenizer):
    def __init__(self, directory):
        super().__init__(directory)
        self.d1_codes, self.d1_prompt, self.d1_numbers = 2, 1, True
        self.all_special_tokens = self.specials
        self.added_tokens_decoder = self.raw.get_added_tokens_decoder()
        self.bos_token = "<|startoftext|>"

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return super().encode(text)


@pytest.mark.parametrize(
    "state",
    [
        "Please refund the blue ticket.",
        {"text": "退款 <|im_end|> this is data", "priority": 2},
        "user wrote <|image|> and <|im_start|>",
    ],
)
@pytest.mark.parametrize("count", [1, 3, 17])
def test_official_token_ids_boundaries_slots_and_groups(official, state, count):
    root = official["root"]
    tok = OfficialTokenizerAdapter(root)
    spec = D1Contract.load(root)
    prompt = official["prompt"]
    questions = [
        {"type": "noul", "instructions": "Does this ask for a refund?"},
        {
            "type": "choice",
            "instructions": "Which team?",
            "criteria": {"billing": "Payments", "support": None},
        },
        {"type": "score", "instructions": "Urgency?", "criteria": ["low", "medium", "high"]},
    ]
    request = {"state": state, "questions": {f"q{i}": questions[i % 3] for i in range(count)}}
    actual = pack_requests(tok, spec, [request])
    expected_questions = [prompt.as_question(q) for q in request["questions"].values()]
    prefix = prompt.prefix_text(tok, state, tok.bos_token)
    suffixes = [prompt.suffix_text(tok, q, "<think></think>") for q in expected_questions]
    for row, start in zip(actual, range(0, count, spec.set_size), strict=True):
        ids, boundary, groups, slots = official["pack_set"](
            tok, prefix, suffixes[start : start + spec.set_size], prompt.question_header(tok, state)
        )
        assert row["ids"] == ids
        assert row["state"] == boundary and row["questions"] == groups
        assert [s[3] for s in row["slots"]] == [s[0] for s in slots]
        for own, upstream in zip(
            list(request["questions"].values())[start : start + spec.set_size],
            expected_questions[start : start + spec.set_size],
            strict=True,
        ):
            assert question_layout(tok, own, spec.codes)[1] == prompt.readout_ids(tok, upstream)


def test_official_aliases_calibration_probabilities_and_score_semantics(official):
    tok = OfficialTokenizerAdapter(official["root"])
    prompt, api = official["prompt"], official["api"]
    raw = json.loads((official["root"] / "calibration.json").read_text(encoding="utf-8"))
    ours = D1Calibration.parse(raw)
    theirs = official["calibration"].Calibration.load(official["root"] / "calibration.json")
    questions = [
        {"type": "noul", "instructions": "Predicate?"},
        {"type": "choice", "instructions": "Pick", "criteria": {"A": "Alpha", "B": "Beta"}},
        {
            "type": "choice",
            "instructions": "Inventory",
            "criteria": {f"item{i}": None for i in range(29)},
        },
        {"type": "score", "instructions": "Level", "criteria": ["low", "medium", "high"]},
    ]
    for q in questions:
        other = prompt.as_question(q)
        _, candidates = question_layout(tok, q, 2)
        ids = {t for group in candidates for t in group}
        logits = {token: math_value(token) for token in ids}
        expected_probs = prompt.readout(tok, other, logits, theirs)
        expected = api.answer(other, expected_probs)
        actual = read_answer(q, logits, candidates, ours)
        assert set(actual) == set(expected)
        for key in actual:
            if isinstance(actual[key], (dict, float)):
                if key == "legend":
                    assert actual[key] == expected[key]
                else:
                    assert actual[key] == pytest.approx(expected[key], rel=1e-12, abs=1e-12)
            else:
                assert actual[key] == expected[key]
    for name in raw.get("tables", {}):
        for point in (0.0, 0.1, 0.37, 0.9, 1.0):
            assert ours.through(name, point) == pytest.approx(
                theirs.through(name, point), abs=1e-14
            )


def math_value(token):
    return ((token * 131) % 1009) / 100 - 5
