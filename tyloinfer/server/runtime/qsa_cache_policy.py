"""Select the native QSA cache and price its existing encoded storage contract."""
from __future__ import annotations

from typing import Mapping

from tyloinfer.server.runtime.kv_budget import KvTierBudget

_MODES = frozenset({'auto', 'fp16', 'affine8', 'tq4', 'legacy'})
_BITS = {'0': 'legacy', '16': 'fp16', '8': 'affine8', '4': 'tq4'}
_PRECISION_ENV = ('TYLOINFER_QSA_CACHE_MODE', 'TYLOINFER_QSA_RAM_KV_BITS', 'TYLOINFER_QSA_ADAPTIVE_TQ')


def apply_qsa_cache_defaults(artifact, settings, quantization, environment: Mapping[str, str]):
    if (settings.model_fields_set & {'qsa_mode', 'placement'}
            or any(environment.get(name) for name in _PRECISION_ENV)):
        return settings
    if resolve_qsa_cache_mode(artifact, settings, quantization, environment) == 'auto':
        return settings.model_copy(update={'qsa_mode': 'affine8', 'placement': 'auto_ssd'})
    return settings


def resolve_qsa_cache_mode(artifact, settings, quantization, environment: Mapping[str, str]) -> str:
    config = getattr(artifact, 'source_config', None) or {}
    text = config.get('text_config', config)
    model_type = str(text.get('model_type', config.get('model_type', '')))
    if model_type not in {'qwen4', 'qwen4_text'} and not model_type.startswith('qwen4_exp'):
        return 'legacy'
    explicit = 'qsa_mode' in settings.model_fields_set
    if explicit:
        mode = settings.qsa_mode
    elif environment.get('TYLOINFER_QSA_CACHE_MODE'):
        mode = environment['TYLOINFER_QSA_CACHE_MODE'].lower()
    elif environment.get('TYLOINFER_QSA_RAM_KV_BITS'):
        try:
            mode = _BITS[environment['TYLOINFER_QSA_RAM_KV_BITS']]
        except KeyError as error:
            raise ValueError('Invalid legacy QSA RAM precision') from error
    elif environment.get('TYLOINFER_QSA_ADAPTIVE_TQ') == '0':
        mode = 'legacy'
    else:
        mode = settings.qsa_mode
    if mode not in _MODES:
        raise ValueError('Invalid QSA cache mode')
    compatible = (settings.placement in {'auto', 'auto_ssd'} and settings.allocation != 'dynamic'
                  and not quantization.enabled and text.get('indexer_compress_ratio') == 4
                  and 0 < text.get('head_dim', 0) <= 1024)
    if mode != 'legacy' and not compatible:
        if explicit and mode != 'auto':
            raise ValueError('Fixed QSA RAM precision requires four-token QSA blocks and automatic RAM placement')
        return 'legacy'
    return mode


def _align(value: int, boundary: int) -> int:
    return (value + boundary - 1) // boundary * boundary


def qsa_cache_budget(artifact, context: int, settings, mode: str) -> KvTierBudget:
    config = artifact.source_config
    text = config.get('text_config', config)
    layout = getattr(artifact, 'source_cache_layout', None)
    if not layout or layout.get('version') != 1:
        raise ValueError('qsa_cache_unsupported: rebuild the native runtime for explicit QSA cache modes')
    layers = text.get('layer_types')
    if layers is None:
        count = text['num_hidden_layers'] // text.get('full_attention_interval', 4)
    else:
        count = layers.count('full_attention')
    tensors = getattr(artifact, 'source_tensors', None) or {}
    if any(name.startswith(('predictor.block.', 'mtp.')) for name in tensors):
        count += text.get('mtp_num_hidden_layers', 1)
    heads, width, index = text['num_key_value_heads'], text['head_dim'], text['indexer_head_dim']
    if min(context, count, heads, width, index) <= 0 or width > 1024 or text['indexer_compress_ratio'] != 4:
        raise ValueError('qsa_cache_unsupported: invalid adaptive QSA geometry')
    padded = 256 if width <= 256 else 512 if width <= 512 else 1024
    rows = heads * 4 * (padded // 256)
    blocks = (context + 3) // 4
    fp16 = 2 * rows * 512
    affine = 2 * (_align(3 * rows, 4) + rows * 256)
    tq4 = 2 * (_align(2 * rows + (rows + 1)//2 + (3 * rows + 7)//8, 4) + rows * 128)
    quantum = layout['quantize_tokens'] // 4
    if mode == 'auto':
        # ram_precision_target in qsa_ram_policy.h, including complete-batch tails.
        protected = (blocks + 9) // 10
        payload = min(blocks * affine, blocks * tq4 + (protected + quantum - 1) * (affine - tq4))
        # SSD pressure can spill an INT8 image without changing its precision.
        disk_image = affine
    else:
        canonical = {'fp16': fp16, 'affine8': affine, 'tq4': tq4}[mode]
        payload = blocks * canonical
        disk_image = canonical
    archive = index * 8 + (index + 2) * 2
    metadata = count * blocks * layout['host_block_metadata_bytes']
    indexer = count * blocks * ((index + 2) * 2 + layout['gpu_view_bytes'])
    selected = (text.get('indexer_budget', 2048) + 6)//4
    queries = layout['graph_query_capacity']
    # The selected-block staging arena is shared by equal-geometry layers.
    staging = queries * selected * _align(fp16, 16)
    codec_io = 2 * quantum * (fp16 + affine + tq4 + 64)
    tails = count * (4 * heads * (width + padded) * 2 * 4 + 4 * index * 4)
    working = metadata + staging + codec_io + tails
    ssd = settings.placement == 'auto_ssd'
    if ssd:
        # Existing per-layer stores: 1 MiB KV and 4 MiB index archive buffers.
        working += count * (5 << 20)
    ram = working if ssd else working + count * (payload + blocks * archive)
    shared_cache = layout.get('shared_vram_cache_bytes', 0)
    if text.get('model_type', config.get('model_type')) in {'qwen4', 'qwen4_text'}:
        shared_cache = layout.get('dense_shared_vram_cache_bytes', shared_cache)
    required_gpu = (indexer + staging + tails + shared_cache
                    + count * layout.get('private_vram_cache_bytes_per_layer', 0))
    gpu = required_gpu if settings.gpu_budget_bytes is None else settings.gpu_budget_bytes
    if gpu < required_gpu:
        raise ValueError(f'kv_budget_below_indexer: QSA descriptors, indexer and staging need {required_gpu} bytes')
    disk = count * blocks * (disk_image + archive) if ssd else 0
    return KvTierBudget(indexer, count * context * heads * width * 4, gpu, ram, 0., mode, working, disk)
