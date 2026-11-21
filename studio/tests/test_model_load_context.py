import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from pydantic import ValidationError

from server.protocol.models import ModelLoadRequest
from server.runtime.load_context import apply_load_context
from server.runtime.load_plan import plan_model_load
from server.services.service import ServerService


@pytest.mark.parametrize('architecture', ['qwen4_exp', 'qwen3_5', 'qwen3.5-moe', 'llama'])
@pytest.mark.parametrize('automatic', [False, True])
@pytest.mark.parametrize('ssd,ram', [(False, True), (True, True), (True, False)])
def test_load_defaults_resolve_by_model(architecture, automatic, ssd, ram):
    request = ModelLoadRequest(model='test', context_size=32768, context_management={
        'automatic': automatic, 'ssd_kv': ssd, 'ram_preallocated': ram,
        'vram_preallocated': False, 'kv_quantization': {'enabled': True, 'bits': 2.5}})
    result = apply_load_context(request, architecture, 'cuda')
    assert result.context_size == 32768
    assert result.kv_memory.placement == ('auto_ssd' if architecture == 'qwen4_exp' and not automatic and ssd else 'auto')
    assert result.kv_memory.ram_preallocated == (ram if architecture == 'qwen4_exp' and not automatic else True)
    assert result.kv_quantization.enabled == (architecture != 'llama' and (not automatic or architecture != 'qwen4_exp'))
    assert not result.qsa_kv_offload.enabled
    assert 'qsa_kv_offload' in result.model_fields_set
    inherited = architecture == 'qwen4_exp' and automatic
    assert ('kv_memory' in result.model_fields_set) != inherited
    assert ('kv_quantization' in result.model_fields_set) != inherited
    # Replayed requests retain the resolved per-model settings.
    assert apply_load_context(ModelLoadRequest.model_validate(result.model_dump()), architecture, 'cuda') == result


def test_non_sparse_model_ignores_sparse_only_invalid_combination():
    request = ModelLoadRequest(model='test', context_management={'automatic': False, 'ssd_kv': False, 'ram_preallocated': False})
    assert apply_load_context(request, 'qwen3_5', 'cuda').kv_memory.ram_preallocated
    with pytest.raises(ValidationError, match='kv_ssd_required'):
        apply_load_context(request, 'qwen4_exp', 'cuda')


def test_explicit_settings_survive_context_defaults_and_legacy_requests_are_unchanged():
    legacy = ModelLoadRequest(model='test')
    assert apply_load_context(legacy, 'qwen4_exp', 'cuda') is legacy
    request = ModelLoadRequest(model='test', context_management={}, kv_memory={'placement': 'vram'},
        kv_quantization={'enabled': True, 'bits': 8})
    result = apply_load_context(request, 'qwen4_exp', 'cuda')
    assert result.kv_memory.placement == 'vram' and result.kv_quantization.bits == 8


def test_load_api_keeps_absent_fields_absent_for_default_resolution():
    service = SimpleNamespace(runtime_manager=object(), create_job=AsyncMock(return_value=SimpleNamespace(id=uuid4())))
    request = ModelLoadRequest(model='test', context_size=32768, context_management={'automatic': False,
        'ssd_kv': True, 'ram_preallocated': False, 'kv_quantization': {'enabled': True, 'bits': 4}})
    asyncio.run(ServerService.load_model(service, request))
    payload = service.create_job.await_args.args[0].payload
    assert 'kv_memory' not in payload and 'kv_quantization' not in payload
    result = apply_load_context(ModelLoadRequest.model_validate(payload), 'qwen4_exp', 'cuda')
    assert result.kv_memory.placement == 'auto_ssd' and not result.kv_memory.ram_preallocated
    assert result.kv_quantization.enabled


def test_dense_load_plan_keeps_gqa_kv_in_the_selected_gpu():
    gib = 1 << 30
    artifact = SimpleNamespace(
        resource=SimpleNamespace(complete=True, architecture='qwen3_5', estimated_resident_weight_bytes=6*gib,
            estimated_weight_bytes_by_role={}, structured_decisions=False),
        routed_expert_bytes=0,
        source_config=dict(model_type='qwen3_5', max_position_embeddings=262144, num_hidden_layers=32, hidden_size=1024,
            layer_types=['full_attention'] * 32, num_key_value_heads=4, head_dim=128),
    )
    status = dict(backend='cuda', memory_architecture='discrete', host_memory_available_bytes=64*gib,
        gpu_devices=[dict(id='0', memory_available_bytes=12*gib), dict(id='1', memory_available_bytes=8*gib)])

    def plan(reserve=10, requested=None, precision=None, preallocate=True, device='0'):
        return plan_model_load(artifact, status, maximum=262144, requested=requested, reserve=reserve,
            precision=precision, preallocate=preallocate, device_id=device, streamed_jev=True)

    normal = plan()
    assert normal.usable and normal.precision == 8
    assert normal.context_size == normal.max_context
    fixed = 7 * gib + 256 * 1024 * 8 * 4
    kv_budget = (12 * gib - fixed) * 9 // 10
    bytes_per_token = 32 * 4 * (2 + 2 * 32) * 4
    assert normal.max_context == kv_budget // bytes_per_token
    assert normal.gpu_budget_bytes == 12 * gib
    assert plan(reserve=20).max_context < normal.max_context < plan(reserve=5).max_context
    assert plan(device='1').max_context < normal.max_context
    assert plan(requested=normal.max_context + 1, precision=8).reason == 'memory_limit'
    assert plan(requested=normal.max_context + 1).reason == 'memory_limit'
    assert plan(precision=4).max_context > normal.max_context
    assert plan(requested=262144, preallocate=False).context_size == 262144


@pytest.mark.parametrize('mode', ['fixed_qsa', 'legacy_qsa', 'deepseek'])
def test_sparse_load_plan_only_applies_reserve_to_ram(mode):
    gib = 1 << 30
    maximum = 1 << 20
    config = dict(max_position_embeddings=maximum, num_hidden_layers=32, head_dim=256)
    if mode == 'deepseek':
        config.update(model_type='deepseek_v4', index_head_dim=128, sliding_window=128,
                      compress_ratios=[4, 128] * 16)
    else:
        config.update(model_type='qwen4_exp', layer_types=['full_attention'] * 32,
                      num_key_value_heads=1, indexer_head_dim=128, indexer_compress_ratio=4)
    artifact = SimpleNamespace(
        resource=SimpleNamespace(complete=True, architecture=config['model_type'],
            estimated_resident_weight_bytes=4 * gib, estimated_weight_bytes_by_role={},
            structured_decisions=False),
        routed_expert_bytes=2 * gib, source_config=config,
        source_cache_layout=dict(version=1, quantize_tokens=2048, gpu_view_bytes=72,
            host_block_metadata_bytes=512, graph_query_capacity=8) if mode == 'fixed_qsa' else None,
    )

    def plan(reserve, ram=16 * gib):
        status = dict(backend='cuda', memory_architecture='discrete', host_memory_available_bytes=ram,
                      device_free_bytes=6 * gib, load_plan_vram_ceiling=5 * gib)
        return plan_model_load(artifact, status, maximum=maximum, requested=None, reserve=reserve,
            precision=None, preallocate=True, device_id='0', streamed_jev=True)

    aggressive, normal, safe = (plan(reserve) for reserve in (5, 10, 20))
    assert aggressive == normal == safe
    assert normal.usable and normal.max_context < maximum
    assert normal.gpu_budget_bytes == 5 * gib
    if mode == 'fixed_qsa':
        assert plan(20, ram=4 * gib).max_context < plan(5, ram=4 * gib).max_context < normal.max_context


def test_planned_precision_is_applied_to_the_existing_native_configuration():
    request = ModelLoadRequest(model='test', context_management={'reserve_percent': 10, 'precision': 8,
        'vram_preallocated': False})
    dense = apply_load_context(request, 'qwen3_5', 'cuda')
    assert dense.kv_quantization.enabled and dense.kv_quantization.bits == 8
    sparse = apply_load_context(request, 'qwen4_exp', 'cuda', fixed_qsa=True)
    assert sparse.kv_memory.qsa_mode == 'affine8' and sparse.kv_memory.placement == 'auto_ssd'
    assert not sparse.kv_memory.ram_preallocated
