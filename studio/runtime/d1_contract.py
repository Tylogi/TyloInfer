"""Validated text-only LiquidAI d1 contract; deliberately separate from D3.

Reference: LiquidAI/d1-2B, a6bce774e6dde338dcc3036ceb5d91f78bf4c19e.
No model code is imported or executed when inspecting a checkpoint.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

D1_REVISION = "a6bce774e6dde338dcc3036ceb5d91f78bf4c19e"
D1_REPOSITORY = "LiquidAI/d1-2B"


def positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def finite(value: Any, name: str, *, positive: bool = False) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))
    return float(value)


@dataclass(frozen=True)
class Lfm3TextConfig:
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    max_position_embeddings: int
    conv_L_cache: int  # noqa: N815 -- retain the official config field name
    layer_types: tuple[str, ...]
    norm_eps: float
    sliding_window: int
    rope_parameters: dict[str, dict]
    conv_bias: bool = False
    use_sandwich_norm: bool = True
    tie_word_embeddings: bool = True
    pad_token_id: int | None = None

    @classmethod
    def parse(cls, text: dict) -> Lfm3TextConfig:
        if not isinstance(text, dict) or text.get("model_type") != "lfm3":
            raise ValueError("d1 reference execution requires an LFM3 text backbone")
        names = (
            "hidden_size",
            "intermediate_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "vocab_size",
            "max_position_embeddings",
            "conv_L_cache",
            "sliding_window",
        )
        values = {key: positive_int(text.get(key), key) for key in names}
        layers = text.get("layer_types")
        count = positive_int(text.get("num_hidden_layers"), "num_hidden_layers")
        if (
            not isinstance(layers, list)
            or len(layers) != count
            or any(t not in ("conv", "full_attention", "sliding_attention") for t in layers)
        ):
            raise ValueError("invalid LFM3 layer_types")
        if values["num_attention_heads"] % values["num_key_value_heads"] or values["head_dim"] % 2:
            raise ValueError("LFM3 requires divisible GQA heads and an even head dimension")
        rope = text.get("rope_parameters")
        if not isinstance(rope, dict):
            raise ValueError("missing LFM3 rope_parameters")
        for kind in set(layers) - {"conv"}:
            entry = rope.get(kind)
            if not isinstance(entry, dict) or entry.get("rope_type") != "default":
                raise ValueError("reference LFM3 execution supports default RoPE only")
            finite(entry.get("rope_theta"), "rope_theta", positive=True)
            factor = finite(
                entry.get("partial_rotary_factor", 1.0), "partial_rotary_factor", positive=True
            )
            if factor > 1:
                raise ValueError("partial_rotary_factor must not exceed 1")
        flags = {}
        for name, default in (
            ("conv_bias", False),
            ("use_sandwich_norm", True),
            ("tie_word_embeddings", True),
        ):
            flags[name] = text.get(name, default)
            if type(flags[name]) is not bool:
                raise ValueError(f"{name} must be boolean")
        pad = text.get("pad_token_id")
        if pad is not None and (type(pad) is not int or not 0 <= pad < values["vocab_size"]):
            raise ValueError("pad_token_id is outside the vocabulary")
        return cls(
            **values,
            layer_types=tuple(layers),
            rope_parameters=rope,
            norm_eps=finite(text.get("norm_eps"), "norm_eps", positive=True),
            pad_token_id=pad,
            **flags,
        )


@dataclass(frozen=True)
class D1Contract:
    text: Lfm3TextConfig
    set_size: int
    codes: int
    bos_token_id: int

    @classmethod
    def parse(cls, config: dict) -> D1Contract:
        if (
            not isinstance(config, dict)
            or config.get("model_type") != "lfm3_vl"
            or config.get("architectures") != ["D1Model"]
        ):
            raise ValueError("not a LiquidAI D1Model/LFM3-VL checkpoint (D3 is not compatible)")
        if config.get("d1_attention") != "set":
            raise ValueError("d1 reference execution requires the trained set-attention contract")
        if (
            type(config.get("d1_prompt", 1)) is not int
            or config.get("d1_prompt", 1) != 1
            or type(config.get("d1_codes")) is not int
            or config.get("d1_codes") not in (1, 2)
        ):
            raise ValueError("unsupported d1 prompt or answer-code scheme")
        text = Lfm3TextConfig.parse(config.get("text_config"))
        if not text.tie_word_embeddings or config.get("tie_word_embeddings") is not True:
            raise ValueError("reference d1 requires tied LM-head and token embeddings")
        size = positive_int(config.get("d1_set_size"), "d1_set_size")
        if not 2 <= size <= 128:
            raise ValueError("trained d1_set_size must be between 2 and 128")
        bos = config.get("bos_token_id")
        if type(bos) is not int or not 0 <= bos < text.vocab_size:
            raise ValueError("invalid d1 BOS token")
        return cls(text, size, config["d1_codes"], bos)

    @classmethod
    def load(cls, directory: Path) -> D1Contract:
        return cls.parse(json.loads((directory / "config.json").read_text(encoding="utf-8")))


@dataclass(frozen=True)
class D1Calibration:
    default: float
    temperatures: dict[str, float]
    tables: dict[str, tuple[tuple[float, float], ...]]

    @classmethod
    def parse(cls, data: dict) -> D1Calibration:
        if not isinstance(data, dict) or not isinstance(data.get("temperatures"), dict):
            raise ValueError("invalid d1 calibration temperatures")
        default = finite(data.get("default", 1.0), "default temperature", positive=True)
        temperatures = {
            key: finite(value, "temperature", positive=True)
            for key, value in data["temperatures"].items()
        }
        tables = {}
        if not isinstance(data.get("tables", {}), dict):
            raise ValueError("invalid d1 calibration tables")
        for name, knots in data.get("tables", {}).items():
            if not isinstance(knots, list):
                raise ValueError("calibration knots must be a list")
            parsed = []
            for pair in knots:
                if not isinstance(pair, list) or len(pair) != 2:
                    raise ValueError("calibration knot must be [raw, calibrated]")
                raw, calibrated = (finite(v, "calibration probability") for v in pair)
                if (
                    not 0 <= raw <= 1
                    or not 0 <= calibrated <= 1
                    or (parsed and (raw <= parsed[-1][0] or calibrated < parsed[-1][1]))
                ):
                    raise ValueError("calibration knots must increase and lie in [0,1]")
                parsed.append((raw, calibrated))
            tables[name] = tuple(parsed)
        return cls(default, temperatures, tables)

    def temperature(self, kind: str, count: int) -> float:
        bucket = (
            "noul"
            if kind == "noul"
            else f"score:{count}"
            if kind == "score"
            else "choice:small"
            if count <= 3
            else "choice:mid"
            if count <= 8
            else "choice:large"
            if count <= 26
            else "choice:inventory"
        )
        return self.temperatures.get(bucket, self.default)

    def through(self, name: str, value: float) -> float:
        knots = self.tables.get(name, ())
        if not knots:
            return value
        if value <= knots[0][0]:
            return knots[0][1]
        for (left, low), (right, high) in zip(knots, knots[1:], strict=False):
            if value <= right:
                return low + (value - left) * (high - low) / (right - left)
        return knots[-1][1]


def median_level(probabilities: list[float]) -> int:
    if not probabilities or any(not math.isfinite(p) or p < 0 for p in probabilities):
        raise ValueError("invalid score probabilities")
    total = math.fsum(probabilities)
    if total <= 0:
        raise ValueError("score probability mass must be positive")
    mass = 0.0
    for i, p in enumerate(probabilities):
        mass += p / total
        if mass >= 0.5:
            return i
    return len(probabilities) - 1
