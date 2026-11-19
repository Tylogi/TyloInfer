import math
import re

from tyloinfer.formats.compat import canonical_dtype


def jev_weight_memory(tensors: dict, config: dict, *, trained_readout: bool) -> dict[str, int] | None:
    layers: dict[str, int] = {}
    tables = resident = 0
    d1 = config.get('architectures') == ['D1Model']
    for name, record in tensors.items():
        if d1:
            name = name.removeprefix('model.language_model.')
            name = re.sub(r'^layers\.(\d+)\.', r'model.block.\1.', name)
            name = re.sub(r'^blk\.(\d+)\.', r'model.block.\1.', name)
            name = {'embed_tokens.weight': 'model.token_embedding.weight', 'token_embd.weight': 'model.token_embedding.weight',
                    'embedding_norm.weight': 'model.output_norm.weight', 'output_norm.weight': 'model.output_norm.weight'}.get(name, name)
        layer = re.match(r'^model\.block\.(\d+)\.', name)
        table = name in {'model.token_embedding.weight', 'model.output.weight'}
        if not layer and not table and name != 'model.output_norm.weight':
            continue
        size = int(record['nbytes'])
        shape, dtype = record.get('shape'), record.get('dtype')
        if shape and dtype in {'BF16', 'F16', 'F32'}:
            size = math.prod(shape) * (4 if dtype == 'F32' else 2)
            if d1:
                size = math.prod(shape) * 2
            elif 'norm' in name or layer and '.linear_attention.' in name and name.rsplit('.', 1)[-1] in {'a', 'dt_bias'}:
                size = math.prod(shape) * 4
        if layer:
            key = layer[1]
            layers[key] = ((layers.get(key, 0) + 255) // 256) * 256 + size
        elif table:
            tables += size
        else:
            resident += size
    if not layers:
        return None
    text = config.get('text_config', config)
    if trained_readout:
        hidden = text.get('hidden_size')
        if not isinstance(hidden, int) or hidden <= 0:
            return None
        # The D3 readout has 255 rows and executes in FP32.
        resident += 255 * hidden * 4
    return {'stream_vram_bytes': 2 * max(layers.values()) + resident,
        'stream_ram_bytes': sum(layers.values()) + tables,
        'resident_vram_bytes': sum(layers.values()) + tables + resident}


def weight_role(name: str) -> str:
    if re.search(r'\.mlp\.experts\.(?:\d+\.)?(?:gate|up|down|gate_up)(?:_proj)?\.(?:weight|weight_scale|scales|biases)$', name) or any(
        marker in name for marker in ('.ffn_gate_exps.', '.ffn_up_exps.', '.ffn_gate_up_exps.', '.ffn_down_exps.')
    ):
        return 'experts'
    if name.startswith(('model.token_embedding.', 'token_embd.')) or '.embed_tokens.' in name or name == 'embed_tokens.weight':
        return 'embedding'
    return 'dense'


def estimated_weight_roles(by_role: dict[str, dict[str, int]]) -> dict[str, int]:
    return {role: estimated_resident_weight_bytes(sum(dtypes.values()), dtypes)
            for role, dtypes in by_role.items()}


def estimated_resident_weight_bytes(
    weight_bytes: int, by_dtype: dict[str, int],
) -> int:
    indexed = sum(by_dtype.values())
    if indexed > weight_bytes:
        return weight_bytes + (weight_bytes + 9) // 10
    reserve = weight_bytes - indexed
    for dtype, size in by_dtype.items():
        dtype = canonical_dtype(dtype)
        if dtype not in {'F8_E4M3', 'F16', 'BF16', 'F32', 'F64', 'I8', 'I16', 'I32', 'I64',
                           'U8', 'U16', 'U32', 'U64', 'BOOL'}:
            reserve += size
    return weight_bytes + (reserve + 9) // 10
