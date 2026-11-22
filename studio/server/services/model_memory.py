from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from formats.assets import MODEL_CONFIG_ASSET
from formats.io import open_mmap
from server.protocol.models import (
    ModelCacheComponent,
    ModelCacheFixedComponent,
    ModelCacheProfile,
)

if TYPE_CHECKING:
    from server.state.catalog import DiscoveredModel


def checkpoint_cache_profile(artifact: DiscoveredModel, *, cuda: bool = False,
    include_recurrent: bool = False) -> ModelCacheProfile | None:
    if not artifact.resource.complete:
        raise ValueError('checkpoint shards are incomplete')
    if getattr(artifact, 'source_config', None) is not None:
        config = artifact.source_config
        has_predictor = any(name.startswith(('predictor.block.', 'mtp.')) for name in (getattr(artifact, 'source_tensors', None) or {}))
    elif artifact.resource.format == 'tq':
        with open_mmap(artifact.path) as store:
            record = store.records.get(MODEL_CONFIG_ASSET)
            if record is not None:
                if record.nbytes > 4 << 20:
                    raise ValueError('model configuration exceeds 4 MiB')
                config = json.loads(store.read_blob(MODEL_CONFIG_ASSET))
            else:
                config = store.header.extra.get('hf_config', {})
            has_predictor = any(name.startswith(('predictor.block.', 'mtp.')) for name in store.records)
    else:
        path = artifact.path / 'config.json'
        if path.stat().st_size > 4 << 20:
            raise ValueError('model configuration exceeds 4 MiB')
        config = json.loads(path.read_text(encoding='utf-8'))
        index = artifact.path / 'model.safetensors.index.json'
        if index.is_file():
            names = json.loads(index.read_text(encoding='utf-8')).get('weight_map', {})
        else:
            from server.state.catalog import ModelCatalog
            names = {name for shard in artifact.path.glob('*.safetensors') for name in ModelCatalog._safetensors_header(shard)}
        has_predictor = any(name.startswith(('predictor.block.', 'mtp.')) for name in names)
    if not isinstance(config, dict):
        return None
    text = config.get('text_config', config)
    predictor_layers = text.get('mtp_num_hidden_layers', 1) if isinstance(text, dict) and has_predictor else 0
    if type(predictor_layers) is not int or predictor_layers < 0:
        return None
    return cache_profile(config, include_recurrent=include_recurrent, predictor_layers=predictor_layers, cuda=cuda)


def cache_profile(config: dict[str, Any], *, include_recurrent: bool = True, predictor_layers: int = 0,
    cuda: bool = False) -> ModelCacheProfile | None:
    text = config.get('text_config', config)
    if not isinstance(text, dict):
        return None
    kind = text.get('model_type', config.get('model_type'))

    def integer(key: str, default: int | None = None) -> int:
        value = text.get(key, default)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(key)
        return value

    try:
        maximum, layers = integer('max_position_embeddings'), integer('num_hidden_layers')
        if min(maximum, layers) <= 0:
            return None
        fixed_components = []
        components = []

        def state(group: str, name: str, count: int, size: int) -> None:
            if count and size:
                fixed_components.append(ModelCacheFixedComponent(group=group, name=name, layers=count, bytes=size))

        if kind == 'deepseek_v4':
            dim, index, window = integer('head_dim'), integer('index_head_dim'), integer('sliding_window')
            ratios = text.get('compress_ratios')
            if min(dim, index, window) <= 0 or not isinstance(ratios, list) or len(ratios) < layers:
                return None
            state('MLA', 'sliding_window', layers, layers * window * dim * 2)
            for ratio in (4, 128):
                count = ratios[:layers].count(ratio)
                width = dim + index if ratio == 4 else dim
                state('MLA', f'compression_state_{ratio}', count, count * ratio * width * (12 if ratio == 4 else 4))
                if count:
                    components.append(ModelCacheComponent(group='MLA', name='compressed_kv', layers=count,
                        bytes_per_row=count * dim * 2, tokens_per_row=ratio))
                    if ratio == 4:
                        components.append(ModelCacheComponent(group='MLA', name='indexer_key', subgroup='indexer', layers=count,
                            bytes_per_row=count * index * 2, tokens_per_row=ratio))
            if any(value not in (0, 4, 128) for value in ratios[:layers]):
                return None
        elif kind == 'hy_v4_flash':
            types, index = text.get('layer_types'), text.get('index_pool')
            if (not isinstance(types, list) or len(types) != layers or not isinstance(index, dict)
                    or any(value not in ('sliding_attention', 'index_pool_attention', 'full_attention') for value in types)):
                return None
            width = integer('kv_lora_rank') + integer('rotary_dim')
            window = integer('sliding_window')
            index_width, group_size, token_budget = (index.get(key) for key in ('key_head_dim', 'group_size', 'token_budget'))
            if min(width, window) <= 0 or any(type(value) is not int or value <= 0 for value in
                                              (index_width, group_size, token_budget)):
                return None
            sliding = types.count('sliding_attention')
            indexed = types.count('index_pool_attention')
            full = types.count('full_attention')
            window_rows = min(maximum, window + 7) if cuda else window
            state('MLA', 'sliding_window', sliding, sliding * window_rows * width * 2)
            pending_groups = group_size if cuda else group_size - 1
            state('MLA', 'index_pool_pending', indexed, indexed * pending_groups * index_width * 8)
            if not cuda:
                state('MLA', 'indexer_selection', indexed, indexed * token_budget * 4)
            for count, limit in ((indexed, token_budget), (full, None)):
                if count:
                    components.append(ModelCacheComponent(group='MLA', name='latent_kv', layers=count,
                        bytes_per_row=count * width * 2, max_read_rows_per_token=limit,
                        head_dimension=integer('kv_lora_rank'),
                        allocation='exact' if cuda else 'power_of_two', minimum_rows=0 if cuda else 16))
            if indexed:
                row_bytes = (index_width + 2) * 2 if cuda else index_width * 4
                components.append(ModelCacheComponent(group='MLA', name='index_pool', subgroup='indexer',
                    layers=indexed, bytes_per_row=indexed * row_bytes, tokens_per_row=group_size,
                    allocation='exact' if cuda else 'power_of_two', minimum_rows=0 if cuda else 16,
                    row_rounding='ceil' if cuda else 'floor'))
        elif kind == 'inkling2_loop':
            types = text.get('layer_types')
            if (not isinstance(types, list) or len(types) != layers
                    or any(value not in ('sliding_attention', 'full_attention') for value in types)):
                return None
            heads, dim = integer('num_key_value_heads'), integer('head_dim')
            window = integer('sliding_window')
            if min(heads, dim, window) <= 0:
                return None
            sliding, full = types.count('sliding_attention'), types.count('full_attention')
            row_bytes = heads * dim * 4
            window_rows = min(maximum, window + 7) if cuda else window
            state('GQA', 'sliding_window', sliding, sliding * window_rows * row_bytes)
            if include_recurrent and not cuda:
                state('GQA', 'short_convolution', layers, integer('shortconv_state_elements', 0) * 2)
            if full:
                components.append(ModelCacheComponent(group='GQA', name='raw_kv', layers=full,
                    bytes_per_row=full * row_bytes, head_dimension=dim, kv_heads=heads,
                    allocation='exact' if cuda else 'power_of_two', minimum_rows=0 if cuda else 16))
        elif kind in {'qwen4_exp', 'qwen4_exp_text', 'qwen3_5', 'qwen3_5_text', 'qwen3_5_moe', 'qwen3_5_moe_text',
                      'qwen3_next', 'sensenova_loop'}:
            types = text.get('layer_types')
            if kind == 'sensenova_loop':
                pattern = text.get('recurrent_layer_pattern')
                if not isinstance(pattern, list) or any(type(value) is not bool for value in pattern):
                    return None
                types = ['linear_attention' if recurrent else 'full_attention' for recurrent in pattern]
            if not isinstance(types, list):
                interval = integer('full_attention_interval', 4)
                if interval <= 0:
                    return None
                types = ['full_attention' if (i + 1) % interval == 0 else 'linear_attention' for i in range(layers)]
            if len(types) != layers or any(value not in ('linear_attention', 'full_attention') for value in types):
                return None
            linear = types.count('linear_attention')
            if linear and include_recurrent:
                keys, key_dim = integer('linear_num_key_heads'), integer('linear_key_head_dim')
                heads, value_dim = integer('linear_num_value_heads'), integer('linear_value_head_dim')
                kernel = integer('linear_conv_kernel_dim')
                if min(keys, key_dim, heads, value_dim, kernel) <= 0:
                    return None
                channels = 2 * keys * key_dim + heads * value_dim
                if key_dim != value_dim:
                    return None
                state_copies = 1 if kind in {'qwen4_exp', 'qwen4_exp_text'} else 2
                recurrent = linear * 4 * heads * value_dim * value_dim
                convolution = linear * 4 * (kernel - 1) * channels
                state('GDN', 'recurrent_state', linear, recurrent)
                state('GDN', 'convolution_state', linear, convolution)
                state('GDN', 'reset_template', linear, (state_copies - 1) * (recurrent + convolution))
            full = types.count('full_attention') + predictor_layers
            if full:
                heads, dim = integer('num_key_value_heads'), integer('head_dim')
                if min(heads, dim) <= 0:
                    return None
                group = 'QSA' if kind in {'qwen4_exp', 'qwen4_exp_text'} else 'GQA'
                read_rows = integer('indexer_budget') if group == 'QSA' and 'indexer_budget' in text else None
                components.append(ModelCacheComponent(group=group, name='raw_kv', layers=full,
                    bytes_per_row=full * heads * dim * 4, allocation='power_of_two', minimum_rows=16,
                    max_read_rows_per_token=read_rows, head_dimension=dim, kv_heads=heads))
            if kind in {'qwen4_exp', 'qwen4_exp_text'}:
                index, ratio = integer('indexer_head_dim'), integer('indexer_compress_ratio')
                if min(index, ratio) <= 0:
                    return None
                if full and cuda:
                    components.extend([
                        ModelCacheComponent(group='QSA', name='indexer_key', subgroup='indexer', layers=full,
                            bytes_per_row=full * index * 2),
                        ModelCacheComponent(group='QSA', name='indexer_pooled', subgroup='indexer', layers=full,
                            bytes_per_row=full * index * 2, tokens_per_row=ratio),
                    ])
                elif full:
                    state('QSA', 'indexer_tail', full, full * ((ratio + 5) * index * 2 + 2))
                    components.extend([
                        ModelCacheComponent(group='QSA', name='indexer_pooled', subgroup='indexer', layers=full,
                            bytes_per_row=full * index * 4, tokens_per_row=ratio, allocation='power_of_two', minimum_rows=16,
                            row_rounding='floor'),
                    ])
                ple_layers = text.get('ple_layer_ids', [])
                if not isinstance(ple_layers, list):
                    return None
                if ple_layers and include_recurrent:
                    kernel, ngram, hc, hidden = integer('ple_conv_kernel_size'), integer('ngram_size'), integer('hc_count'), integer('hidden_size')
                    if min(kernel, ngram, hc, hidden) <= 0:
                        return None
                    state('PLE', 'convolution_state', len(ple_layers), len(ple_layers) * 4 * (kernel - 1) * ngram * hc * hidden)
        elif kind in {'qwen2', 'qwen3', 'qwen3_moe', 'llama', 'mistral', 'minicpm', 'minicpmo'}:
            heads = integer('num_key_value_heads', integer('num_attention_heads'))
            dim = integer('head_dim', integer('hidden_size') // integer('num_attention_heads'))
            if min(heads, dim) <= 0 or text.get('sliding_window'):
                return None
            query_heads = integer('num_attention_heads')
            group = 'GQA' if query_heads > heads else 'MHA'
            components.append(ModelCacheComponent(group=group, name='raw_kv', layers=layers,
                bytes_per_row=layers * heads * dim * 4, allocation='power_of_two', minimum_rows=16,
                head_dimension=dim, kv_heads=heads))
        else:
            return None
        return ModelCacheProfile(max_context=maximum, fixed_bytes=sum(item.bytes for item in fixed_components),
            fixed_components=fixed_components, components=components)
    except (ValueError, TypeError, OverflowError, ZeroDivisionError):
        return None
