"""Plan model loading against live memory, before preallocating CUDA KV."""
from __future__ import annotations

import math

from server.protocol.models import KvMemorySettings, KvQuantizationSettings, ModelLoadPlan
from server.runtime.load_context import supports_kv_quantization
from server.runtime.qsa_cache_policy import qsa_cache_budget, resolve_qsa_cache_mode
from server.services.model_memory import checkpoint_cache_profile

STANDARD_PRECISIONS = (8, 6, 4, 3.5, 3, 2.5, 2)
PREFILL_TILE_ALIGNMENT = 256


def _packed_words(dimension: int, bits: int) -> int:
    rotated = 1 << (dimension - 1).bit_length()
    return 1 + (rotated * bits + 31) // 32


def _cache_bytes(profile, tokens: int, precision: float, *, contiguous: bool, mla_rank: int = 0) -> int:
    total = 0
    for item in profile.components:
        rows = 0 if tokens <= item.active_after else tokens / item.tokens_per_row
        rows = math.floor(rows) if item.row_rounding == 'floor' else math.ceil(rows)
        rows = max(rows, item.minimum_rows)
        if not contiguous and item.allocation == 'power_of_two' and rows:
            rows = 1 << (rows - 1).bit_length()
        width = item.bytes_per_row
        if precision != 16 and item.name == 'raw_kv':
            words = (_packed_words(item.head_dimension, math.floor(precision)) +
                     _packed_words(item.head_dimension, math.ceil(precision)))
            width = item.layers * item.kv_heads * words * 4
        if precision != 16 and mla_rank:
            if item.name == 'latent_kv':
                words = (_packed_words(mla_rank, math.ceil(precision)) +
                         _packed_words(item.head_dimension - mla_rank, math.floor(precision)))
                width = item.layers * words * 4
            elif item.name == 'indexer_pooled':
                width //= 2
        total += rows * width
    return total


def _after_reserve(available: int, reserve: int) -> int:
    return max(0, available) * (100 - reserve) // 100


def plan_model_load(artifact, status: dict, *, maximum: int, requested: int | None,
                    reserve: int, precision: float | None, preallocate: bool,
                    device_id: str, streamed_jev: bool) -> ModelLoadPlan:
    resource = artifact.resource
    backend = status.get('backend')
    cuda = backend == 'cuda'
    profile = checkpoint_cache_profile(artifact, cuda=cuda, include_recurrent=True)
    kind = resource.architecture.casefold()
    components = profile.components if profile is not None else ()
    has_qsa = any(item.group == 'QSA' for item in components)
    has_sparse_attention = has_qsa or any(item.subgroup == 'indexer' for item in components)
    fixed_qsa = (cuda and kind.startswith('qwen4_exp') and has_qsa
                 and resolve_qsa_cache_mode(artifact, KvMemorySettings(), KvQuantizationSettings(), {}) != 'legacy'
                 and getattr(artifact, 'source_cache_layout', None))
    raw = [item for item in components if item.name == 'raw_kv']
    config = getattr(artifact, 'source_config', None) or {}
    text = config.get('text_config', config)
    if not isinstance(text, dict):
        text = {}
    rank = text.get('kv_lora_rank')
    mla_rank = rank if kind == 'hy_v4_flash' and backend == 'metal' and type(rank) is int and any(
        item.name == 'latent_kv' and item.head_dimension and 0 < rank < item.head_dimension for item in components) else 0
    quantized = supports_kv_quantization(kind, backend) and raw and all(
        item.head_dimension and item.kv_heads and item.layers for item in raw)
    precisions = (8, 4) if fixed_qsa else (16, *STANDARD_PRECISIONS) if mla_rank else STANDARD_PRECISIONS if quantized else (16,)
    default = precisions[0]
    selected = precision if precision is not None else default
    separate = status.get('memory_architecture') == 'discrete'
    ram = status.get('host_memory_available_bytes')
    if not separate and status.get('runtime_memory_headroom_bytes') is not None:
        ram = status['runtime_memory_headroom_bytes']
    devices = status.get('gpu_devices', [])
    gpu = next((item.get('memory_available_bytes') for item in devices if item.get('id') == device_id), None)
    if not devices and device_id == '0':
        gpu = status.get('device_free_bytes')
    vram_ceiling = status.get('load_plan_vram_ceiling')
    if gpu is not None and vram_ceiling is not None:
        gpu = min(gpu, vram_ceiling)
    device_budget = None

    def reply(cap: int, context: int | None, reason=None) -> ModelLoadPlan:
        return ModelLoadPlan(max_context=cap, context_size=context, precision=selected, precisions=list(precisions),
                             usable=reason is None, reason=reason,
                             gpu_budget_bytes=device_budget if preallocate and cuda else None)

    if selected not in precisions:
        return reply(0, requested, 'precision_unsupported')
    if requested is not None and requested > maximum:
        return reply(maximum, requested, 'model_limit')
    if not preallocate:
        return reply(maximum, requested or maximum)

    weights = resource.estimated_resident_weight_bytes
    roles = resource.estimated_weight_bytes_by_role
    host_weights = min(weights or 0, artifact.routed_expert_bytes) if separate else 0
    if resource.structured_decisions and resource.jev_weight_memory is not None:
        size = resource.jev_weight_memory
        weights = size.stream_vram_bytes + size.stream_ram_bytes if streamed_jev else size.resident_vram_bytes
        host_weights = size.stream_ram_bytes if streamed_jev and separate else 0
    if weights is None or (ram is None and (not separate or host_weights or fixed_qsa)) or (separate and gpu is None):
        return reply(0, requested, 'memory_unavailable')
    if profile is None and not resource.structured_decisions:
        return reply(0, requested, 'memory_unavailable')

    # CUDA can shrink its auto tile to the same 256-row alignment used by
    # select_prefill_chunk. The KV plan needs room for one tile.
    workspace = 1 << 30 if cuda else 0
    hidden = text.get('hidden_size', 0)
    gqa = any(item.group == 'GQA' for item in raw)
    if cuda and gqa:
        workspace += min(maximum, PREFILL_TILE_ALIGNMENT) * hidden * 8 * 4
    if separate and artifact.routed_expert_bytes:
        host_weights += roles.get('embedding', 0)
    hosted = min(host_weights, _after_reserve(ram or 0, reserve))
    if resource.structured_decisions and hosted < host_weights:
        return reply(0, requested, 'memory_limit')
    pool = (gpu if separate else ram) or 0
    fixed = weights - hosted + workspace + (profile.fixed_bytes if profile else 0)
    if fixed > pool:
        return reply(0, requested, 'memory_limit')
    remaining = pool - fixed
    room_gpu = remaining if separate and has_sparse_attention else _after_reserve(remaining, reserve)
    room_ram = _after_reserve((ram or 0) - hosted, reserve)
    if separate and cuda:
        # Dense GQA preallocates the planned KV once. The device limit also
        # covers temporary work and the allocator's cached pages.
        device_budget = pool if gqa and not artifact.routed_expert_bytes else fixed + room_gpu

    def fits(tokens: int, bits: float) -> bool:
        if fixed_qsa:
            settings = KvMemorySettings(qsa_mode={8: 'affine8', 4: 'tq4'}[bits],
                                        placement='auto_ssd')
            tier = qsa_cache_budget(artifact, tokens, settings, settings.qsa_mode)
            if separate:
                return tier.gpu_budget_bytes <= room_gpu and tier.ram_required_bytes <= room_ram
            return tier.gpu_budget_bytes + tier.ram_required_bytes <= room_gpu
        return profile is None or _cache_bytes(profile, tokens, bits, contiguous=cuda, mla_rank=mla_rank) <= room_gpu

    lo, hi = 0, maximum
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if fits(mid, selected):
            lo = mid
        else:
            hi = mid - 1
    if lo < 512 or (requested is not None and requested > lo):
        return reply(lo, requested, 'memory_limit')
    return reply(lo, requested or lo)
