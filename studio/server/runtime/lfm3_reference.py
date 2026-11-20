"""Uncached PyTorch LFM3 text reference, not the native CUDA/Metal runtime.

Equations checked against LiquidAI/d1-2B modeling_lfm3.py at
 a6bce774e6dde338dcc3036ceb5d91f78bf4c19e (HuggingFace, Apache-2.0).
This independently scoped forward does not implement generation, KV reuse,
packed documents, media, quantization or device offload. It projects slots only.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812 -- conventional PyTorch alias

from server.runtime.d1_contract import Lfm3TextConfig


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        y = x.float()
        return self.weight * (y * torch.rsqrt(y.square().mean(-1, keepdim=True) + self.eps)).to(
            x.dtype
        )


def inverse_frequencies(config: Lfm3TextConfig, kind: str, device=None):
    rope = config.rope_parameters[kind]
    count = int(config.head_dim * rope.get("partial_rotary_factor", 1.0)) // 2
    frequencies = torch.zeros(config.head_dim // 2, dtype=torch.float32, device=device)
    frequencies[:count] = 1.0 / (
        rope["rope_theta"]
        ** (torch.arange(count, dtype=torch.float32, device=device) * 2 / config.head_dim)
    )
    return frequencies


def rotate(x, positions, frequencies):
    # Partial rotation zeros frequencies in the full-dimensional progression;
    # shrinking the RoPE dimension gives different values for this checkpoint.
    angles = positions[..., None].float() * frequencies
    angles = torch.cat((angles, angles), -1)[:, None]
    half = x.shape[-1] // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), -1).float()
    # The official rotary module rounds sin/cos to activation dtype before
    # apply_rotary_pos_emb promotes the multiplication back to float32.
    cosine = angles.cos().to(x.dtype).float()
    sine = angles.sin().to(x.dtype).float()
    return (x.float() * cosine + rotated * sine).to(x.dtype)


def set_mask(real, state_lengths, question_ids, start=0, width=None):
    """True means visible. Padding keys are excluded, including for state queries.

    This is only the GLOBAL attention mask. Convolution/sliding attention remain
    causal and may convey information from preceding questions, as trained.
    """
    tokens = real.shape[1]
    width = tokens - start if width is None else width
    q = torch.arange(start, start + width, device=real.device)[None, :, None]
    k = torch.arange(tokens, device=real.device)[None, None, :]
    s = state_lengths[:, None, None]
    own = question_ids[:, start : start + width, None] == question_ids[:, None, :]
    return (((q < s) | (k < s) | own) & real[:, None, :])[:, None]


class ShortConv(nn.Module):
    def __init__(self, config):
        super().__init__()
        h, k = config.hidden_size, config.conv_L_cache
        self.in_proj = nn.Linear(h, 3 * h, bias=config.conv_bias)
        self.conv = nn.Conv1d(h, h, k, groups=h, bias=config.conv_bias)
        self.out_proj = nn.Linear(h, h, bias=config.conv_bias)
        self.kernel = k

    def forward(self, x, real):
        x = x * real[..., None].to(x.dtype)
        b, c, signal = self.in_proj(x).transpose(1, 2).chunk(3, dim=1)
        mixed = (b * signal) * real[:, None].to(x.dtype)
        convolved = self.conv(F.pad(mixed, (self.kernel - 1, 0)))
        # B and C are multiplicative gates, NOT sigmoid gates.
        return self.out_proj((c * convolved).transpose(1, 2).contiguous())


class Attention(nn.Module):
    def __init__(self, config, kind):
        super().__init__()
        h, q, kv, d = (
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
        )
        self.q_proj, self.k_proj = nn.Linear(h, q * d, bias=False), nn.Linear(h, kv * d, bias=False)
        self.v_proj, self.o_proj = nn.Linear(h, kv * d, bias=False), nn.Linear(q * d, h, bias=False)
        self.gate_proj = nn.Linear(h, q, bias=False)
        self.q_norm, self.k_norm = RMSNorm(d, config.norm_eps), RMSNorm(d, config.norm_eps)
        self.register_buffer("frequencies", inverse_frequencies(config, kind), persistent=False)
        self.q_heads, self.kv_heads, self.dim = q, kv, d
        self.window = config.sliding_window if kind == "sliding_attention" else None

    def forward(self, x, real, state_lengths, question_ids, positions):
        b, t, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(b, t, self.q_heads, self.dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(b, t, self.kv_heads, self.dim)).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.kv_heads, self.dim).transpose(1, 2)
        q, k = rotate(q, positions, self.frequencies), rotate(k, positions, self.frequencies)
        outputs = []
        # Bound global-mask materialization. This is a correctness reference,
        # not a long-context speed implementation or a persistent-cache path.
        for start in range(0, t, 256):
            width = min(256, t - start)
            if self.window is None:
                mask = set_mask(real, state_lengths, question_ids, start, width)
                keys, values = k, v
            else:
                left, right = max(0, start - self.window + 1), start + width
                qp = torch.arange(start, right, device=x.device)[:, None]
                kp = torch.arange(left, right, device=x.device)[None, :]
                mask = ((kp <= qp) & (kp > qp - self.window))[None, None] & real[
                    :, None, None, left:right
                ]
                keys, values = k[:, :, left:right], v[:, :, left:right]
            outputs.append(
                F.scaled_dot_product_attention(
                    q[:, :, start : start + width],
                    keys,
                    values,
                    attn_mask=mask,
                    enable_gqa=True,
                    scale=self.dim**-0.5,
                )
            )
        attended = torch.cat(outputs, dim=2).transpose(1, 2)
        attended = attended * torch.sigmoid(self.gate_proj(x)).unsqueeze(-1)
        return self.o_proj(attended.reshape(b, t, -1).contiguous())


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        h, f = config.hidden_size, config.intermediate_size
        self.w1, self.w3, self.w2 = (
            nn.Linear(h, f, bias=False),
            nn.Linear(h, f, bias=False),
            nn.Linear(f, h, bias=False),
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Layer(nn.Module):
    def __init__(self, config, kind):
        super().__init__()
        self.kind = kind
        self.operator_norm, self.ffn_norm = (
            RMSNorm(config.hidden_size, config.norm_eps),
            RMSNorm(config.hidden_size, config.norm_eps),
        )
        if kind == "conv":
            self.conv = ShortConv(config)
        else:
            self.self_attn = Attention(config, kind)
        self.feed_forward = MLP(config)
        self.sandwich = config.use_sandwich_norm
        if self.sandwich:
            self.post_operator_norm = RMSNorm(config.hidden_size, config.norm_eps)
            self.post_ffn_norm = RMSNorm(config.hidden_size, config.norm_eps)

    def forward(self, x, real, state_lengths, question_ids, positions):
        normalized = self.operator_norm(x)
        y = (
            self.conv(normalized, real)
            if self.kind == "conv"
            else self.self_attn(normalized, real, state_lengths, question_ids, positions)
        )
        x = x + (self.post_operator_norm(y) if self.sandwich else y)
        y = self.feed_forward(self.ffn_norm(x))
        return x + (self.post_ffn_norm(y) if self.sandwich else y)


class Lfm3TextReference(nn.Module):
    def __init__(self, config: Lfm3TextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.layers = nn.ModuleList([Layer(config, kind) for kind in config.layer_types])
        self.embedding_norm = RMSNorm(config.hidden_size, config.norm_eps)

    def forward(
        self, input_ids, real, state_lengths, question_ids, *, use_cache=False, past_key_values=None
    ):
        if use_cache or past_key_values is not None:
            raise ValueError("d1 set-attention cannot reuse a state prefix or KV/conv cache")
        if (
            input_ids.ndim != 2
            or input_ids.shape != real.shape
            or input_ids.shape != question_ids.shape
            or state_lengths.shape != (input_ids.shape[0],)
            or real.dtype != torch.bool
        ):
            raise ValueError("invalid d1 packed batch shapes")
        b, t = input_ids.shape
        if not b or not t or t > self.config.max_position_embeddings:
            raise ValueError("d1 sequence is empty or exceeds model context")
        if bool((input_ids < 0).any()) or bool((input_ids >= self.config.vocab_size).any()):
            raise ValueError("d1 token outside the model vocabulary")
        lengths = real.sum(-1)
        positions = torch.arange(t, device=input_ids.device).expand(b, t)
        if not torch.equal(real, positions < lengths[:, None]) or bool((lengths == 0).any()):
            raise ValueError("d1 reference execution requires nonempty right-padded rows")
        if bool((state_lengths < 0).any()) or bool((state_lengths >= lengths).any()):
            raise ValueError("d1 state boundary must leave question tokens")
        if bool(((question_ids <= 0) & real & (positions >= state_lengths[:, None])).any()):
            raise ValueError("d1 question tokens require positive question IDs")
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, real, state_lengths, question_ids, positions)
        return self.embedding_norm(x)

    def slot_logits(self, hidden, rows, positions, candidate_ids):
        slots = hidden[rows, positions]
        return F.linear(slots, self.embed_tokens.weight[candidate_ids])
