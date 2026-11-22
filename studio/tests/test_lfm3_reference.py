from __future__ import annotations

# torch is an optional extra: skip the module before importing its implementation.
# ruff: noqa: E402, N812
import ast
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn
from torch.nn import functional as F

from tests.test_d1_reference import config
from runtime.d1_contract import D1Contract
from runtime.lfm3_reference import (
    Lfm3TextReference,
    ShortConv,
    inverse_frequencies,
    rotate,
    set_mask,
)


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def batch():
    ids = torch.tensor([[1, 3, 4, 5, 6, 7, 8], [1, 9, 3, 4, 2, 0, 0]])
    real = torch.tensor([[True] * 7, [True] * 5 + [False] * 2])
    state = torch.tensor([2, 2])
    groups = torch.tensor([[-1, -1, 1, 1, 2, 2, 2], [-1, -1, 1, 1, 1, -1, -1]])
    return ids, real, state, groups


def test_set_mask_state_questions_padding_and_chunks():
    _, real, states, groups = batch()
    mask = set_mask(real, states, groups)
    assert mask.shape == (2, 1, 7, 7)
    assert bool(mask[0, 0, 0].all())
    assert bool(mask[0, 0, 2, :4].all()) and not bool(mask[0, 0, 2, 4:].any())
    assert bool(mask[0, 0, 5, :2].all()) and not bool(mask[0, 0, 5, 2:4].any())
    assert not bool(mask[1, :, :, 5:].any())
    assert torch.equal(mask[:, :, 2:5], set_mask(real, states, groups, 2, 3))


def test_partial_rope_uses_full_head_frequency_progression():
    data = config()
    data["text_config"]["head_dim"] = 8
    spec = D1Contract.parse(data).text
    inv = inverse_frequencies(spec, "full_attention")
    torch.testing.assert_close(inv, torch.tensor([1.0, 1e6 ** (-2 / 8), 0, 0]))
    x = torch.randn(1, 2, 4, 8)
    out = rotate(x, torch.arange(4)[None], inv)
    assert torch.equal(out[..., 2:4], x[..., 2:4]) and torch.equal(out[..., 6:8], x[..., 6:8])


def test_conv_gates_are_not_sigmoids_and_future_tokens_cannot_change_prefix():
    spec = D1Contract.parse(config()).text
    conv = ShortConv(spec).eval()
    x = torch.randn(1, 12, spec.hidden_size)
    real = torch.ones(1, 12, dtype=torch.bool)
    y = conv(x, real)
    changed = x.clone()
    changed[:, 7:] += 100
    torch.testing.assert_close(y[:, :7], conv(changed, real)[:, :7])
    b, c, signal = conv.in_proj(x).transpose(1, 2).chunk(3, 1)
    expected = conv.out_proj(
        (
            c
            * F.conv1d(
                F.pad(b * signal, (6, 0)), conv.conv.weight, conv.conv.bias, groups=spec.hidden_size
            )
        ).transpose(1, 2)
    )
    torch.testing.assert_close(y, expected)


def test_real_backbone_padding_equivalence_and_no_cache_reuse():
    torch.manual_seed(42)
    model = Lfm3TextReference(D1Contract.parse(config()).text).eval()
    ids, real, states, groups = batch()
    together = model(ids, real, states, groups)
    alone = model(ids[1:2, :5], real[1:2, :5], states[1:2], groups[1:2, :5])
    torch.testing.assert_close(together[1:2, :5], alone, rtol=1e-5, atol=2e-6)
    for kwargs in ({"use_cache": True}, {"past_key_values": {}}):
        with pytest.raises(ValueError, match="cache"):
            model(ids, real, states, groups, **kwargs)
    wrong = real.clone()
    wrong[0, 1] = False
    with pytest.raises(ValueError, match="right-padded"):
        model(ids, wrong, states, groups)


def official_namespace():
    """Opt-in parity against actual fixed-revision source, not a copied mock.

    Isolate only torch-only definitions so optional Transformers 5.19 is not
    required to exercise the checkpoint's reference equations on CPU.
    No checkpoint custom code is imported by the product loader.
    """
    directory = os.environ.get("TYLOINFER_D1_REFERENCE_SOURCE")
    if not directory:
        pytest.skip("set TYLOINFER_D1_REFERENCE_SOURCE to the pinned official source for parity")
    source = Path(directory) / "modeling_lfm3.py"
    wanted = {
        "Lfm3RotaryEmbedding",
        "Lfm3RMSNorm",
        "Lfm3MLP",
        "rotate_half",
        "repeat_kv",
        "eager_attention_forward",
        "apply_rotary_pos_emb",
        "apply_mask_to_padding_states",
        "Lfm3ShortConv",
        "Lfm3Attention",
        "Lfm3DecoderLayer",
    }
    tree = ast.parse(source.read_text(encoding="utf-8"))
    selected = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in wanted:
            node.decorator_list = []
            for child in ast.walk(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    child.decorator_list = []
            selected.append(node)
    assert {n.name for n in selected} == wanted
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = types.ModuleType("_tyloinfer_official_lfm3_equations")
    module.__dict__.update(
        torch=torch,
        nn=nn,
        F=F,
        GradientCheckpointingLayer=nn.Module,
        can_use_lfm3_conv=lambda *_: False,
        ALL_ATTENTION_FUNCTIONS=SimpleNamespace(get_interface=lambda key, fallback: fallback),
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[future] + selected, type_ignores=[])),
            str(source),
            "exec",
        ),
        module.__dict__,
    )
    # Use the checkpoint's actual d1 attention adapter, not generic eager
    # attention: its BF16 reduction/rounding path is part of this comparison.
    from dataclasses import dataclass, field

    attention_path = Path(directory) / "attention.py"
    attention_tree = ast.parse(attention_path.read_text(encoding="utf-8"))
    attention_names = {"_fused", "_windowed", "causal_attention", "Sees", "d1_attention"}
    attention_nodes = [
        n
        for n in attention_tree.body
        if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in attention_names
    ]
    attention = types.ModuleType("_tyloinfer_official_d1_attention_equations")
    attention.__dict__.update(torch=torch, F=F, dataclass=dataclass, field=field, CHUNK=4096)
    sys.modules[attention.__name__] = attention
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[future] + attention_nodes, type_ignores=[])),
            str(attention_path),
            "exec",
        ),
        attention.__dict__,
    )
    module.Sees = attention.Sees
    module.ALL_ATTENTION_FUNCTIONS = SimpleNamespace(
        get_interface=lambda key, fallback: attention.d1_attention
    )
    return module


@pytest.mark.parametrize("bias,sandwich", [(False, True), (True, True), (False, False)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_full_three_mixer_forward_matches_official_cpu_equations(bias, sandwich, dtype):
    official = official_namespace()
    data = config()
    data["text_config"]["conv_bias"] = bias
    data["text_config"]["use_sandwich_norm"] = sandwich
    spec = D1Contract.parse(data).text
    cfg = SimpleNamespace(
        **data["text_config"],
        _attention_layer_types=("full_attention", "sliding_attention"),
        _attn_implementation="eager",
    )
    torch.manual_seed(125)
    ours = Lfm3TextReference(spec).to(dtype=dtype).eval()
    ids, real, states, groups = batch()
    positions = torch.arange(ids.shape[1])[None].expand(ids.shape[0], -1)
    hidden = ours.embed_tokens(ids)
    for index, kind in enumerate(spec.layer_types):
        layer = official.Lfm3DecoderLayer(cfg, index).to(dtype=dtype).eval()
        layer.load_state_dict(ours.layers[index].state_dict(), strict=True)
        if kind == "conv":
            hidden = layer(hidden, attention_mask=real)
        else:
            inv, _ = official.Lfm3RotaryEmbedding.compute_default_rope_parameters(
                cfg, layer_type=kind
            )
            angles = positions[..., None] * inv
            emb = torch.cat((angles, angles), -1)
            sees = official.Sees(states, groups, real)
            torch.testing.assert_close(
                sees.rows(0, ids.shape[1], real.device), set_mask(real, states, groups)
            )
            hidden = layer(
                hidden,
                position_embeddings=(emb.cos().to(hidden.dtype), emb.sin().to(hidden.dtype)),
                attention_mask=sees,
            )
    norm = official.Lfm3RMSNorm(spec.hidden_size, spec.norm_eps).to(dtype=dtype)
    norm.load_state_dict(ours.embedding_norm.state_dict())
    expected = norm(hidden)
    actual = ours(ids, real, states, groups)
    tolerance = dict(rtol=1e-5, atol=3e-6) if dtype == torch.float32 else dict(rtol=0.02, atol=0.02)
    torch.testing.assert_close(actual[real], expected[real], **tolerance)
    # Also compare selected-slot candidate logits, not just hidden states.
    rows = torch.tensor([0, 0, 1])
    slots = torch.tensor([3, 6, 4])
    candidates = torch.tensor([4, 5, 10, 11])
    expected_logits = F.linear(expected[rows, slots], ours.embed_tokens.weight[candidates])
    logit_tolerance = (
        dict(rtol=1e-5, atol=5e-6) if dtype == torch.float32 else dict(rtol=0.02, atol=0.0625)
    )
    torch.testing.assert_close(
        ours.slot_logits(actual, rows, slots, candidates), expected_logits, **logit_tolerance
    )
