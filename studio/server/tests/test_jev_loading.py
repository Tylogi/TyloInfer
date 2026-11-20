"""Decision-model placement and resource accounting."""
import asyncio

import httpx
import pytest

from server.cli import _build_parser
from server.server.protocol.models import JevWeightMemoryEstimate, ModelLoadRequest
from server.server.runtime.runtime_pool import RuntimePool
from server.server.state.catalog import DiscoveredModel, ModelCatalog
from server.tests.test_server_models import _model
from server.architectures.tensor_schema import canonical_source_tensor_map
from server.server.runtime.cluster import ClusterBackend
from server.server.state.model_weights import jev_weight_memory


def test_jev_memory_uses_two_max_layer_buffers_and_ram_tables():
    tensors = {
        'model.block.0.ffn.weight': {'dtype': 'Q4_K', 'nbytes': 400},
        'model.block.0.attention_norm.weight': {'dtype': 'F16', 'shape': [8], 'nbytes': 32},
        'model.block.1.ffn.weight': {'dtype': 'Q8_0', 'nbytes': 700},
        'model.token_embedding.weight': {'dtype': 'F16', 'shape': [20, 8], 'nbytes': 336},
        'model.output.weight': {'dtype': 'Q8_0', 'nbytes': 180},
        'model.output_norm.weight': {'dtype': 'F16', 'shape': [8], 'nbytes': 32},
        'vision.patch.weight': {'dtype': 'F16', 'nbytes': 100000},
    }
    layers, tables, norm, readout = 544 + 700, 320 + 180, 32, 255 * 8 * 4
    for trained in (False, True):
        resident = norm + (readout if trained else 0)
        assert jev_weight_memory(tensors, {'hidden_size': 8}, trained_readout=trained) == {
            'stream_vram_bytes': 2 * 700 + resident,
            'stream_ram_bytes': layers + tables,
            'resident_vram_bytes': layers + tables + resident,
        }
    assert jev_weight_memory({}, {}, trained_readout=False) is None


@pytest.mark.parametrize('backend', ['cuda', 'metal'])
def test_idle_service_reports_backend_before_model_loading(tmp_path, monkeypatch, backend):
    monkeypatch.setattr('server.server.runtime.runtime_pool.gpu_memory_status', lambda: [])
    pool = RuntimePool(ModelCatalog([tmp_path]), tmp_path / 'runtime', backend=backend)
    status = asyncio.run(pool.runtime_status())
    assert status['runtime_state'] == 'idle'
    assert status['backend'] == backend


def test_jev_bare_backbone_and_packed_weights_have_a_complete_source_map():
    config = {'model_type': 'qwen3_5', 'text_config': {'model_type': 'qwen3_5_text',
        'num_hidden_layers': 1, 'layer_types': ['full_attention']},
        'quantization_config': {'quant_method': 'compressed-tensors'}}
    names = ['language_model.embed_tokens.weight', 'language_model.norm.weight',
        'language_model.layers.0.self_attn.q_proj.weight_packed',
        'language_model.layers.0.self_attn.q_proj.weight_scale']
    mapping = canonical_source_tensor_map(config, names)
    assert mapping['model.token_embedding.weight'] == names[0]
    assert mapping['model.output_norm.weight'] == names[1]
    assert mapping['model.block.0.attention.query.weight'] == names[2]
    assert mapping['model.block.0.attention.query.weight_scale'] == names[3]


@pytest.mark.parametrize('placement', ['stream', 'vram'])
def test_jev_cli_and_managed_worker_placement(tmp_path, placement):
    args = _build_parser().parse_args(['serve', '--jev-weights', placement])
    assert args.jev_weights == placement
    assert _build_parser().parse_args(['serve']).jev_weights == 'stream'

    async def run():
        model = tmp_path / 'decision.tq'
        _model(model, architecture='qwen35')
        catalog = ModelCatalog([tmp_path], cache_seconds=0)
        ordinary = await catalog.resolve_path(model)
        resource = ordinary.resource.model_copy(update={'structured_decisions': True,
            'total_bytes': 200000, 'estimated_weight_bytes_by_role': {'embedding': 10000}})
        artifact = DiscoveredModel(resource, model, source_tensors={
            'model.block.0.ffn.weight': {'nbytes': 30000},
            'model.layers.1.self_attn.q.weight': {'nbytes': 40000},
            'model.block.2.ffn.weight': {'nbytes': 35000}})
        pool = RuntimePool(catalog, tmp_path / 'mfq-runtime', backend='cuda')
        request = ModelLoadRequest(model='decision', jev_weights=placement)
        command, _ = pool._launch_configuration(artifact, request, port=43123)
        assert command[command.index('--jev-weights') + 1] == placement
        estimate = pool._estimated_load_bytes(artifact, request)
        assert estimate == (200000 if placement == 'vram' else 164745)
        planned = DiscoveredModel(resource.model_copy(update={'jev_weight_memory': JevWeightMemoryEstimate(
            stream_vram_bytes=5000, stream_ram_bytes=190000, resident_vram_bytes=191000)}), model)
        assert pool._estimated_load_bytes(planned, request) == (191000 if placement == 'vram' else 5000)
        command, _ = pool._launch_configuration(ordinary, request, port=43123)
        assert '--jev-weights' not in command
    asyncio.run(run())


def test_ram_embedding_and_streamed_weights_do_not_overlap_experts_or_vram():
    memory = RuntimePool._memory_resources({'resident_weight_bytes': 32,
        'ram_expert_payload_bytes': 1024, 'ram_expert_enabled': 1,
        'ram_embedding_bytes': 2048, 'ram_weight_bytes': 4096})
    assert memory.resident_weight_bytes == 32
    assert memory.ram_expert_bytes == 1024
    assert memory.ram_embedding_bytes == 2048
    assert memory.ram_weight_bytes == 4096
    retained = RuntimePool._memory_resources({}, memory)
    assert retained == memory
    cleared = RuntimePool._memory_resources({'ram_weight_bytes': 0, 'ram_embedding_bytes': 0}, memory)
    assert cleared.ram_embedding_bytes == cleared.ram_weight_bytes == 0


def test_service_backend_forwards_decision_calls_to_the_local_runtime():
    async def run():
        class Local:
            async def decide(self, *, model, payload):
                assert model == 'decision' and payload == {'state': 'blue'}
                return {'object': 'structured_decision'}
        async with httpx.AsyncClient(trust_env=False) as client:
            backend = ClusterBackend(Local(), None, client=client)
            assert await backend.decide(model='decision', payload={'state': 'blue'}) == {'object': 'structured_decision'}
    asyncio.run(run())
