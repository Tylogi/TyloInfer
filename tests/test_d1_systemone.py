from __future__ import annotations

import asyncio
import json
import math
import struct
import threading
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from tests.test_d1_reference import config, request
from tyloinfer.runtime.d1_contract import D1Calibration, D1Contract
from tyloinfer.runtime.d1_reference import LocalTokenizer
from tyloinfer.server.runtime.backend import BackendProtocolError, OpenAIChatBackend
from tyloinfer.server.runtime.client import BackendError, HttpRuntimeClient
from tyloinfer.server.runtime.d1_adapter import D1DecisionBackend, prepare_request
from tyloinfer.server.runtime.d1_worker import D1Worker, create_app
from tyloinfer.server.state.catalog import ModelCatalog


def assets(root: Path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    root.mkdir(exist_ok=True)
    words = ['<unk>', '<|startoftext|>', '<|im_start|>', '<|im_end|>', 'yes', 'Yes', 'YES',
             'no', 'No', 'NO', 'A', 'B', 'C', '0', '1', '2', 'user', 'assistant', 'blue']
    tokenizer = Tokenizer(models.WordLevel({word: i for i, word in enumerate(words)}, unk_token='<unk>'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.add_special_tokens(words[1:4])
    tokenizer.save(str(root / 'tokenizer.json'))
    (root / 'config.json').write_text(json.dumps(config()))
    (root / 'calibration.json').write_text(json.dumps({'default': 1, 'temperatures': {}}))
    header = json.dumps({'model.language_model.embedding_norm.weight':
                         {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}}).encode()
    (root / 'model.safetensors').write_bytes(struct.pack('<Q', len(header)) + header + b'\0' * 4)
    return root


class NativeFixture:
    backend = 'native-cuda-cublas-lfm3'
    def __init__(self, root):
        self.contract = D1Contract.load(root)
        self.calibration = D1Calibration.parse({'default': 1, 'temperatures': {}})
        self.tokenizer = LocalTokenizer(root)
        self.ready = {'resident_packed_weight_bytes': 1024, 'resident_bf16_weight_bytes': 32,
                      'full_matrix_dequant_cache_bytes': 0}
        self.process = SimpleNamespace(poll=lambda: 0 if self.closed else None)
        self.closed = False
        self.calls = 0
        self.entered = self.finish = None
    def evaluate(self, row, candidates):
        self.calls += 1
        if self.entered:
            self.entered.set()
            assert self.finish.wait(5)
        logits = {self.tokenizer.encode(label)[0]: math.log(p) for label, p in
                  [('yes', .7), ('Yes', .7), ('YES', .7), ('no', .3), ('No', .3), ('NO', .3),
                   ('A', .8), ('B', .2), ('0', .4), ('1', .35), ('2', .25)]}
        return {'logits': [[logits.get(token, 0) for token in candidates] for _ in row['slots']],
                'gpu_ms': [1.0], 'packed_kernel_dispatch': {'q8-column-pipeline32-fp32': 1}}
    def close(self):
        self.closed = True


def test_d1_uses_common_private_http_and_keeps_median(tmp_path):
    async def run():
        native = NativeFixture(assets(tmp_path))
        worker = D1Worker(native, 'd1', 2048)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(worker)), base_url='http://runtime') as client:
            backend = D1DecisionBackend(HttpRuntimeClient('http://runtime', client=client))
            body = {'model': 'alias', **request()}
            result = await backend.decide(model='d1', payload=body)
            assert result['answers']['urgency']['score'] == 1
            assert result['answers']['urgency']['score'] != pytest.approx(.85)
            assert result['answers']['refund']['noul'] == pytest.approx(.7)
            assert result['answers']['team']['choice'] == 'billing'
            assert result['usage']['output_tokens'] == 0
            assert result['usage']['input_tokens_details']['cached_tokens'] == 0
            assert native.calls == 1
            assert (await backend.capabilities()).model_capabilities.architecture_family == 'lfm3_d1'
            with pytest.raises(BackendError, match='decision model') as unavailable:
                await backend.preflight(model='d1')
            assert unavailable.value.status_code == 501
            assert 'logits' not in worker.status()['native_metrics']
            assert (await client.get('/runtime/shutdown')).status_code == 501
            assert (await client.get('/runtime/reload')).status_code == 405
        worker.close()
        assert native.closed
    asyncio.run(run())


@pytest.mark.parametrize('score', [0.85, 1.0, 0.0])
def test_common_score_validation_does_not_recompute_model_readout(score):
    async def run():
        class Client:
            async def decide(self, body):
                return {'object': 'structured_decision', 'answers': {'x': {'type': 'score', 'score': score,
                    'probabilities': {'0': .4, '1': .35, '2': .25}, 'confidence': .4,
                    'legend': {'0': 'a', '1': 'b', '2': 'c'}}}, 'usage': {'input_tokens': 3, 'output_tokens': 0},
                    'diagnostics': {'x': {'normalization': 'labels', 'label_mass': None, 'argmax_is_label': None}}}
        backend = OpenAIChatBackend(Client())
        result = await backend.decide(model='any', payload={'questions': {'x': {'type': 'score', 'criteria': ['a', 'b', 'c']}}})
        assert result['answers']['x']['score'] == score
    asyncio.run(run())


@pytest.mark.parametrize('score', [-1, 3, float('nan'), float('inf'), True])
def test_common_score_still_rejects_invalid_range_and_nonfinite_values(score):
    async def run():
        class Client:
            async def decide(self, body):
                return {'object': 'structured_decision', 'answers': {'x': {'type': 'score', 'score': score,
                    'probabilities': {'0': .4, '1': .35, '2': .25}, 'confidence': .4,
                    'legend': {'0': 'a', '1': 'b', '2': 'c'}}}, 'usage': {'input_tokens': 3, 'output_tokens': 0},
                    'diagnostics': {'x': {'normalization': 'labels', 'label_mass': None, 'argmax_is_label': None}}}
        with pytest.raises(BackendProtocolError):
            await OpenAIChatBackend(Client()).decide(model='any', payload={'questions': {'x': {'type': 'score', 'criteria': ['a', 'b', 'c']}}})
    asyncio.run(run())


@pytest.mark.parametrize('change', [
    {'media': []}, {'images': []}, {'prefix_cache': False}, {'full_vocabulary': False},
    {'joint': True}, {'stream': False}, {'chat_template_kwargs': {'chat_template': 'x'}},
    {'questions': {'x': {'type': 'text'}}},
    {'questions': {'x': {'type': 'choice', 'multiple': True, 'criteria': {'a': None}}}},
    {'questions': {'x': {'type': 'number', 'range': [0, 1]}}},
])
def test_unsupported_d1_capabilities_are_rejected_before_forward(tmp_path, change):
    body = {'model': 'd1', **request(), **change}
    with pytest.raises(BackendError) as failed:
        prepare_request(body, 'd1')
    assert failed.value.status_code == 400


def test_d1_request_mapping_accepts_shared_instructions_and_partial_noul_criteria():
    value = prepare_request({'model': 'd1', 'state': {'color': 'blue'}, 'instructions': 'Read the record.',
        'questions': {'x': {'type': 'noul', 'criteria': {'true': 'Blue record'}}}}, 'd1')
    assert value['questions']['x']['instructions'] == 'Read the record.\n\n'
    assert value['questions']['x']['criteria']['false'] == 'No / false'


def test_worker_cancellation_keeps_native_busy_until_completion_and_closes(tmp_path):
    async def run():
        native = NativeFixture(assets(tmp_path))
        native.entered, native.finish = threading.Event(), threading.Event()
        worker = D1Worker(native, 'd1', 2048)
        task = asyncio.create_task(worker.dispatch('decide', {'model': 'd1', **request()}))
        assert await asyncio.to_thread(native.entered.wait, 5)
        native_task = worker.pending
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        with pytest.raises(BackendError) as busy:
            await worker.dispatch('decide', {'model': 'd1', **request()})
        assert busy.value.status_code == 409
        native.finish.set()
        await native_task
        await asyncio.sleep(0)
        assert worker.pending is None
        worker.close()
        worker.close()
        with pytest.raises(BackendError):
            worker.health()
        assert native.closed
    asyncio.run(run())


def test_managed_d1_rejects_chat_before_committing_stream(tmp_path):
    from dataclasses import replace

    from tests.test_runtime_maintenance import runtime
    from tests.test_server_openai_compat import _Service
    from tyloinfer.server.api import create_app as create_public_app

    async def run():
        pool, instance, _ = runtime(tmp_path)
        instance.artifact = replace(instance.artifact, source_config=config())
        instance.backend = D1DecisionBackend(SimpleNamespace())
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_public_app(_Service(pool))),
                                     base_url='http://test') as client:
            response = await client.post('/v1/chat/completions', json={'model': 'probe',
                'messages': [{'role': 'user', 'content': 'Hello'}], 'stream': True})
        assert response.status_code == 501
        assert response.json()['error']['type'] == 'generation_unavailable'
        assert instance.active_requests == instance.queued_requests == 0
    asyncio.run(run())


@pytest.mark.parametrize('backend,options,code', [
    ('hip', {}, 'backend_unsupported'),
    ('cuda', {'device_ids': ['0', '1']}, 'unsupported_device'),
])
def test_managed_d1_rejects_unsupported_load_capabilities(tmp_path, monkeypatch, backend, options, code):
    from tests.test_server_models import _TestJobContext
    from tyloinfer.server.protocol.models import RuntimeMemoryPolicy
    from tyloinfer.server.runtime.runtime_pool import RuntimePool
    from tyloinfer.server.services.jobs import JobExecutionError

    monkeypatch.setattr('tyloinfer.server.runtime.runtime_pool.hardware_identity', lambda: SimpleNamespace(gpu_memory=[]))

    async def run():
        root = assets(tmp_path / 'decision')
        pool = RuntimePool(ModelCatalog([root]), tmp_path / 'runtime', backend=backend)
        pool.memory_policy = RuntimeMemoryPolicy(prefix_directory=str(tmp_path / 'prefix'))
        with pytest.raises(JobExecutionError) as failed:
            await pool.load(_TestJobContext(), {'model': root.name, **options})
        assert failed.value.detail.code == code
        assert not pool._instances
        await pool.aclose()
    asyncio.run(run())


def test_hf_catalog_recognizes_d1_without_a_d3_config(tmp_path):
    root = assets(tmp_path / 'decision')
    def inspect_d3(*args):
        pytest.fail('d1 must not use the D3 native inspector')

    catalog = ModelCatalog([tmp_path], native_inspector=inspect_d3)
    artifact = asyncio.run(catalog.resolve(root.name))
    assert artifact.resource.loadable and artifact.resource.structured_decisions
    assert artifact.resource.architecture == 'lfm3_d1-hf'
    assert not (root / 'decision_config.json').exists()


@pytest.mark.parametrize('backend', ['cuda', 'metal'])
@pytest.mark.parametrize('placement', ['stream', 'vram'])
def test_managed_d1_forwards_weight_placement_and_reports_ram(tmp_path, monkeypatch, backend, placement):
    from tyloinfer.server.protocol.models import ModelLoadRequest
    from tyloinfer.server.runtime.d1_support import d1_launch_configuration
    from tyloinfer.server.runtime.runtime_pool import RuntimePool

    root = assets(tmp_path / 'decision')
    artifact = ModelCatalog._inspect_hf(tmp_path, root)
    monkeypatch.setattr('tyloinfer.server.runtime.d1_support.find_native_runtime_resource', lambda *_args: tmp_path / 'native')
    request = ModelLoadRequest(model=root.name, context_size=2048, jev_weights=placement)
    metallib = tmp_path / 'mlx.metallib'
    metallib.touch()
    command, environment = d1_launch_configuration(tmp_path / 'runtime', artifact, request,
        transport='stdio', port=0, backend=backend, environment={})
    assert command[command.index('--jev-weights') + 1] == placement
    if backend == 'metal':
        assert environment['TYLOINFER_MLX_METALLIB'] == str(metallib)
    native = NativeFixture(root)
    native.ready.update(weight_placement=placement, streamed_weight_bytes=4096 if placement == 'stream' else 0,
                        layer_buffer_bytes=512 if placement == 'stream' else 0)
    status = D1Worker(native, 'd1', 2048).status()
    memory = RuntimePool._memory_resources(status)
    assert memory.ram_weight_bytes == (4096 if placement == 'stream' else 0)
    assert memory.resident_weight_bytes == 1056 + (512 if placement == 'stream' else 0)
    assert memory.ram_embedding_bytes == 0


@pytest.mark.parametrize('gguf', [False, True])
def test_d1_memory_estimate_streams_layers_and_tied_embeddings(gguf):
    from tyloinfer.server.state.model_weights import jev_weight_memory

    names = ['blk.0.ffn_gate.weight', 'blk.1.ffn_gate.weight', 'token_embd.weight', 'output_norm.weight'] if gguf else [
        'model.language_model.layers.0.feed_forward.w1.weight', 'model.language_model.layers.1.feed_forward.w1.weight',
        'model.language_model.embed_tokens.weight', 'model.language_model.embedding_norm.weight']
    tensors = {name: {'nbytes': size, 'dtype': 'Q8_0' if index < 3 else 'F32', 'shape': [8] if index == 3 else None}
               for index, (name, size) in enumerate(zip(names, [512, 768, 4096, 32], strict=True))}
    estimate = jev_weight_memory(tensors, {'architectures': ['D1Model']}, trained_readout=False)
    assert estimate == {'stream_vram_bytes': 1552, 'stream_ram_bytes': 5376, 'resident_vram_bytes': 5392}
