from types import SimpleNamespace

import pytest

from tyloinfer.server.protocol.models import KvMemorySettings, KvQuantizationSettings, ModelLoadRequest
from tyloinfer.server.runtime.qsa_cache_policy import qsa_cache_budget, resolve_qsa_cache_mode


def artifact():
    return SimpleNamespace(
        source_config=dict(model_type='qwen4_exp_text', num_hidden_layers=48,
            layer_types=['linear_attention'] * 36 + ['full_attention'] * 12,
            num_key_value_heads=2, head_dim=256, indexer_head_dim=128,
            indexer_compress_ratio=4, indexer_budget=2048),
        source_tensors={},
        source_cache_layout=dict(version=1, tokens_per_block=4, quantize_tokens=2048,
            gpu_view_bytes=72, host_block_metadata_bytes=512, graph_query_capacity=8),
    )


def resolve(settings=None, environment=None, quantization=None, model=None):
    return resolve_qsa_cache_mode(model or artifact(), settings or KvMemorySettings(),
        quantization or KvQuantizationSettings(), environment or {})


def test_auto_is_explicit_policy_not_a_numeric_precision():
    assert KvMemorySettings().qsa_mode == 'auto'
    assert resolve() == 'auto'
    for bits, mode in [('16', 'fp16'), ('8', 'affine8'), ('4', 'tq4'), ('0', 'legacy')]:
        assert resolve(environment={'TYLOINFER_QSA_RAM_KV_BITS': bits}) == mode
    assert resolve(environment={'TYLOINFER_QSA_ADAPTIVE_TQ': '1'}) == 'auto'
    assert resolve(environment={'TYLOINFER_QSA_ADAPTIVE_TQ': '0'}) == 'legacy'


def test_api_then_named_mode_then_legacy_precision_precedence():
    env = {'TYLOINFER_QSA_CACHE_MODE': 'auto', 'TYLOINFER_QSA_RAM_KV_BITS': '16'}
    assert resolve(environment=env) == 'auto'
    assert resolve(KvMemorySettings(qsa_mode='tq4'), env) == 'tq4'
    assert resolve(KvMemorySettings(qsa_mode='auto'), {'TYLOINFER_QSA_CACHE_MODE': 'fp16'}) == 'auto'
    with pytest.raises(ValueError, match='Invalid QSA cache mode'):
        resolve(environment={'TYLOINFER_QSA_CACHE_MODE': 'invalid'})


@pytest.mark.parametrize('settings', [KvMemorySettings(placement='vram'),
    KvMemorySettings(allocation='dynamic'), KvMemorySettings(placement='ram')])
def test_ordinary_kv_contracts_are_not_silently_changed(settings):
    assert resolve(settings) == 'legacy'
    assert resolve(quantization=KvQuantizationSettings(enabled=True)) == 'legacy'
    model = artifact(); model.source_config['model_type'] = 'qwen3_5'
    assert resolve(model=model) == 'legacy'


def test_budget_prices_actual_images_archives_metadata_and_batch_tails():
    model = artifact(); settings = KvMemorySettings(); ctx = 262144
    auto = qsa_cache_budget(model, ctx, settings, 'auto')
    fixed = {mode: qsa_cache_budget(model, ctx, settings, mode)
             for mode in ('fp16', 'affine8', 'tq4')}
    assert auto.raw_bytes == 6 << 30
    assert fixed['tq4'].ram_required_bytes < auto.ram_required_bytes < fixed['affine8'].ram_required_bytes
    assert fixed['tq4'].ram_required_bytes < fixed['affine8'].ram_required_bytes < fixed['fp16'].ram_required_bytes
    blocks = ctx//4; rows = 8
    fp16, affine, tq4 = 2*rows*512, 2*(24+rows*256), 2*(24+rows*128)
    target = blocks*tq4 + ((blocks+9)//10 + 511)*(affine-tq4)
    archive = 128*8+130*2
    assert auto.ram_required_bytes-auto.ram_working_set_bytes == 12*(target+blocks*archive)
    assert auto.ram_working_set_bytes > 12*blocks*model.source_cache_layout['host_block_metadata_bytes']


def test_shared_vram_cache_is_priced_once_independently_of_context_length():
    model = artifact()
    settings = KvMemorySettings(placement='auto_ssd')
    before = {size: qsa_cache_budget(model, size, settings, 'auto') for size in (131072, 262144)}
    model.source_cache_layout['shared_vram_cache_bytes'] = 100 << 20
    for size, original in before.items():
        updated = qsa_cache_budget(model, size, settings, 'auto')
        assert updated.gpu_budget_bytes - original.gpu_budget_bytes == 100 << 20
        assert updated.ram_required_bytes == original.ram_required_bytes


def test_dense_uses_optional_spare_memory_unless_given_an_explicit_cache_size():
    model = artifact()
    model.source_config = {'model_type': 'qwen4',
                           'text_config': dict(model.source_config, model_type='qwen4_text')}
    settings = KvMemorySettings(placement='auto_ssd')
    assert resolve(settings, model=model) == 'auto'
    model.source_cache_layout.update(shared_vram_cache_bytes=100 << 20, dense_shared_vram_cache_bytes=0)
    automatic = qsa_cache_budget(model, 262144, settings, 'auto')
    model.source_cache_layout['dense_shared_vram_cache_bytes'] = 32 << 20
    explicit = qsa_cache_budget(model, 262144, settings, 'auto')
    assert explicit.gpu_budget_bytes - automatic.gpu_budget_bytes == 32 << 20


def test_ssd_admits_working_set_not_full_context_precision_population():
    model = artifact(); settings = KvMemorySettings(placement='auto_ssd')
    for mode in ('auto', 'fp16', 'affine8', 'tq4'):
        plan = qsa_cache_budget(model, 262144, settings, mode)
        assert plan.ram_required_bytes == plan.ram_working_set_bytes < plan.raw_bytes
        assert plan.ssd_capacity_bytes > 0
    larger = qsa_cache_budget(model, 1048576, settings, 'auto')
    assert larger.ram_required_bytes < larger.raw_bytes//4
    assert larger.ssd_capacity_bytes > qsa_cache_budget(model, 262144, settings, 'auto').ssd_capacity_bytes
    assert larger.ssd_capacity_bytes == qsa_cache_budget(model, 1048576, settings, 'affine8').ssd_capacity_bytes


def test_ssd_capacity_and_mandatory_ram_checks_use_selected_path(tmp_path, monkeypatch):
    from tyloinfer.server.runtime.runtime_pool import RuntimePool
    model = artifact(); settings = KvMemorySettings(placement='auto_ssd', qsa_mode='auto')
    plan = qsa_cache_budget(model, 262144, settings, 'auto')
    calls = []
    monkeypatch.setattr('tyloinfer.server.runtime.runtime_pool.check_ssd_capacity',
                        lambda path, size: calls.append((path, size)))
    pool = SimpleNamespace(backend='cuda', runtime_environment={},
        resolve_context_size=lambda *a, **kw: 262144, _prefix_cache_directory=lambda: tmp_path)
    request = ModelLoadRequest(model='fixture', kv_memory=settings.model_copy(
        update={'ram_budget_bytes': plan.ram_working_set_bytes}))
    actual = RuntimePool._validate_kv_memory_budget(pool, model, request)
    assert actual.cache_mode == 'auto'
    assert calls == [(tmp_path, plan.ssd_capacity_bytes+(64 << 20))]
    request.kv_memory.ram_budget_bytes -= 1
    with pytest.raises(Exception, match='mandatory working set'):
        RuntimePool._validate_kv_memory_budget(pool, model, request)


def test_missing_native_layout_is_not_faked_with_fp16_or_ignored():
    model = artifact(); model.source_cache_layout = None
    with pytest.raises(ValueError, match='rebuild the native runtime'):
        qsa_cache_budget(model, 262144, KvMemorySettings(), 'auto')


def test_serve_wires_auto_ssd_and_gguf_expert_ram_reservation(tmp_path, monkeypatch):
    from tyloinfer.server.runtime.runtime_pool import RuntimePool
    from tyloinfer.server.state.catalog import ModelCatalog
    model = artifact(); model.path = tmp_path/'fixture.gguf'; model.routed_expert_bytes = 40 << 30
    model.resource = SimpleNamespace(name='fixture', format='gguf', structured_decisions=False,
        architecture='qwen4_exp_text')
    monkeypatch.setattr('tyloinfer.server.runtime.runtime_pool.native_tokenizer_arguments', lambda *a: [])
    monkeypatch.setattr('tyloinfer.server.runtime.runtime_pool.native_runtime_environment', lambda *a, **kw: {})
    pool = RuntimePool(ModelCatalog([]), tmp_path/'runtime', backend='cuda', max_requests_per_instance=4)
    monkeypatch.setattr(pool, 'resolve_context_size', lambda *a, **kw: 262144)
    monkeypatch.setattr(pool, '_cuda_expert_host_budget', lambda *a: 2 << 30)
    monkeypatch.setattr(pool, '_cuda_device_vram_limit', lambda *a: 20 << 30)
    monkeypatch.setattr(pool, '_kv_ram_limit', lambda *a: 4 << 30)
    monkeypatch.setattr(pool, '_prefix_cache_directory', lambda: tmp_path)
    request = ModelLoadRequest(model='fixture', kv_memory=dict(qsa_mode='auto', placement='auto_ssd'))
    argv, environment = pool._launch_configuration(model, request, port=0)
    assert environment['TYLOINFER_QSA_CACHE_MODE'] == 'auto'
    assert environment['TYLOINFER_QSA_TQ_DIRECTORY'] == str(tmp_path/'qsa-kv')
    assert environment['TYLOINFER_KV_PLACEMENT'] == 'auto_ssd'
    assert environment['TYLOINFER_MOE_HOST_CACHE_BYTES'] == str(2 << 30)
    assert int(environment['TYLOINFER_KV_RAM_LIMIT_BYTES']) < 4 << 30
    assert argv[argv.index('--continuous-batching')+1] == '4'


def test_concurrent_qsa_does_not_freeze_optional_expert_complement(monkeypatch):
    from tyloinfer.server.runtime.runtime_pool import RuntimePool
    model = artifact()
    model.resource = SimpleNamespace(format='gguf', total_bytes=48 << 30)
    pool = SimpleNamespace(backend='cuda', runtime_environment={})
    for mode in ('auto', 'fp16', 'affine8', 'tq4'):
        request = ModelLoadRequest(model='fixture', kv_memory={'qsa_mode': mode})
        assert RuntimePool._kv_ram_limit(pool, model, request) is None
        request.kv_memory.ram_budget_bytes = 7 << 30
        assert RuntimePool._kv_ram_limit(pool, model, request) == 7 << 30
    model.always_streamed_bytes = 0
    pool._estimated_load_bytes = lambda *a: 8 << 30
    monkeypatch.setattr('tyloinfer.server.runtime.runtime_pool.host_memory_capacity',
                        lambda: (64 << 30, 48 << 30))
    legacy = ModelLoadRequest(model='fixture', kv_memory={'qsa_mode': 'legacy'})
    assert RuntimePool._kv_ram_limit(pool, model, legacy) == 6 << 30


def test_cli_exposes_auto_separately_from_fixed_modes():
    from tyloinfer.cli import _build_parser
    for mode in ('auto', 'fp16', 'affine8', 'tq4', 'legacy'):
        args = _build_parser().parse_args(['serve', '--qsa-cache-mode', mode])
        assert args.qsa_cache_mode == mode


def test_saved_mode_and_old_settings_have_distinct_migration_semantics():
    legacy = KvMemorySettings.model_validate({'placement': 'auto'})
    assert 'qsa_mode' not in legacy.model_fields_set
    assert resolve(legacy, {'TYLOINFER_QSA_RAM_KV_BITS': '16'}) == 'fp16'
    saved = KvMemorySettings.model_validate(KvMemorySettings(qsa_mode='auto').model_dump())
    assert resolve(saved, {'TYLOINFER_QSA_RAM_KV_BITS': '16'}) == 'auto'
