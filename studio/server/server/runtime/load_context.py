"""Resolve browser load defaults against the model/backend's capabilities."""
from server.server.protocol.models import (
    KvMemorySettings, KvQuantizationSettings, ModelLoadRequest, QsaKvOffloadPolicy,
)


def apply_load_context(request: ModelLoadRequest, architecture: str, backend: str, *, fixed_qsa: bool = False) -> ModelLoadRequest:
    defaults = request.context_management
    if defaults is None:
        return request
    kind = architecture.casefold()
    qsa = kind.startswith('qwen4_exp') and backend in {'cuda', 'metal'}
    quantized = backend in {'cuda', 'metal'} and kind.startswith(('qwen4_exp', 'qwen3.5', 'qwen3_5', 'qwen35'))
    if defaults.reserve_percent is not None:
        bits = defaults.precision or 16
        changes = {'kv_quantization': KvQuantizationSettings(enabled=quantized and not fixed_qsa and bits < 16,
                   bits=bits if bits < 16 else 4), 'qsa_kv_offload': QsaKvOffloadPolicy()}
        if fixed_qsa:
            changes['kv_memory'] = KvMemorySettings(qsa_mode={16: 'fp16', 8: 'affine8', 4: 'tq4'}[bits],
                placement='auto_ssd', ram_preallocated=defaults.vram_preallocated)
        return request.model_copy(update={key: value for key, value in changes.items() if key not in request.model_fields_set})
    manual = not defaults.automatic
    use_server_qsa_defaults = defaults.automatic and qsa and backend == 'cuda'
    changes = {}
    # A load can still specify exact per-model settings. Browser defaults only
    # fill missing fields, and become explicit before saved settings are read.
    if 'kv_quantization' not in request.model_fields_set and not use_server_qsa_defaults:
        changes['kv_quantization'] = (defaults.kv_quantization if manual and quantized
                                      else KvQuantizationSettings(enabled=quantized, bits=8))
    if 'kv_memory' not in request.model_fields_set and not use_server_qsa_defaults:
        changes['kv_memory'] = KvMemorySettings(
            placement='auto_ssd' if manual and qsa and backend == 'cuda' and defaults.ssd_kv else 'auto',
            ram_preallocated=defaults.ram_preallocated if manual and qsa and backend == 'cuda' else True,
        )
    if 'qsa_kv_offload' not in request.model_fields_set:
        changes['qsa_kv_offload'] = QsaKvOffloadPolicy(enabled=manual and qsa and backend == 'metal' and defaults.ssd_kv)
    return request.model_copy(update=changes)
