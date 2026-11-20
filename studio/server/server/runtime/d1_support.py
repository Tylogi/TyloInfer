"""Local d1 asset recognition and managed worker configuration."""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

from server.formats.gguf import _metadata_stream, _MetadataReader
from server.runtime.d1_contract import D1Calibration, D1Contract
from server.server.runtime.native import NativeRuntimeError, find_native_runtime_resource


def is_d1_config(config: dict | None) -> bool:
    return isinstance(config, dict) and config.get('architectures') == ['D1Model']


def d1_assets(path: Path) -> Path:
    root = path if path.is_dir() else path.parent
    for name in ('config.json', 'tokenizer.json', 'calibration.json'):
        if not (root / name).is_file():
            raise ValueError('d1 requires matching config.json, tokenizer.json and calibration.json beside its weights')
    D1Contract.load(root)
    D1Calibration.parse(json.loads((root / 'calibration.json').read_text(encoding='utf-8')))
    return root


def d1_gguf_metadata(path: Path) -> dict | None:
    """Read only bounded GGUF headers, never quantized tensor payloads."""
    with _metadata_stream(path) as (stream, size):
        reader = _MetadataReader(stream, size)
        identity = {}
        for _ in range(reader.fields):
            name, kind = reader.string(keep=True), reader.integer('<I')
            value = reader.value(kind, keep=name in {b'general.architecture', b'lfm3.decision.type'})
            if name in {b'general.architecture', b'lfm3.decision.type'}:
                if name in identity:
                    raise ValueError('duplicate d1 identity metadata')
                identity[name] = value
        if identity.get(b'general.architecture') != b'lfm3' or identity.get(b'lfm3.decision.type') != b'lfm3-d1':
            return None
        root = d1_assets(path)
        config = json.loads((root / 'config.json').read_text(encoding='utf-8'))
        if reader.tensors > 1 << 20:
            raise ValueError('too many d1 GGUF tensors')
        tensors = {}
        for _ in range(reader.tensors):
            name = reader.string(keep=True).decode('utf-8')
            rank = reader.integer('<I')
            if not 1 <= rank <= 4:
                raise ValueError('invalid d1 GGUF tensor rank')
            shape = [reader.integer('<Q') for _ in range(rank)]
            kind, _offset = reader.integer('<I'), reader.integer('<Q')
            if name in tensors or any(d <= 0 for d in shape):
                raise ValueError('invalid d1 GGUF tensor geometry')
            formats = {0: ('F32', 1, 4), 1: ('F16', 1, 2), 8: ('GGUF_Q8_0', 32, 34), 30: ('BF16', 1, 2)}
            if kind not in formats or (rank == 2 and not name.endswith('.shortconv.conv.weight') and kind != 8):
                raise ValueError('d1 native matrix weights currently require packed Q8_0')
            dtype, block, width = formats[kind]
            count = math.prod(shape)
            if count % block:
                raise ValueError('d1 Q8_0 weights require whole quantization blocks')
            tensors[name] = {'dtype': dtype, 'nbytes': count // block * width, 'shape': list(reversed(shape))}
        return {'architecture': 'lfm3_d1', 'config': config, 'tensors': tensors, 'structured_decisions': True}


def d1_launch_configuration(executable: Path, artifact, request, *, transport: str, port: int,
                            backend: str, environment: dict[str, str]) -> tuple[list[str], dict[str, str]]:
    if backend != 'cuda':
        raise NativeRuntimeError('d1 native serving currently requires the CUDA backend')
    root = d1_assets(artifact.path)
    name = 'tyloinfer-d1-native' + ('.exe' if os.name == 'nt' else '')
    binary = find_native_runtime_resource(executable, name)
    if binary is None:
        raise NativeRuntimeError('the CUDA runtime installation is missing tyloinfer-d1-native')
    command = [sys.executable, '-m', 'server.server.runtime.d1_worker', '--native', str(binary),
               '--assets', str(root), '--weights', str(artifact.path), '--model-name', artifact.resource.name,
               '--ctx-size', str(request.context_size), '--transport', transport,
               '--jev-weights', request.jev_weights]
    if transport == 'http':
        command.extend(['--host', '127.0.0.1', '--port', str(port)])
    result = dict(os.environ)
    result.update(environment)
    if request.device_ids:
        result['CUDA_VISIBLE_DEVICES'] = ','.join(request.device_ids)
    return command, result
