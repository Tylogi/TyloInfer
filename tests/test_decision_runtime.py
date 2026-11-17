from __future__ import annotations

import asyncio
import copy
import json

import httpx
import pytest

from tyloinfer.server.runtime.backend import BackendError, BackendProtocolError, OpenAIChatBackend
from tyloinfer.server.runtime.client import HttpRuntimeClient
from tests.test_runtime_maintenance import runtime
from tests.test_runtime_stdio_client import _client


def payload():
    return {"model": "alias", "state": "blue", "questions": {
        "color": {"type": "choice", "criteria": {"blue": None, "green": None}}}}


def response():
    return {"object": "structured_decision", "model": "canonical", "answers": {
        "color": {"type": "choice", "choice": "blue", "probabilities": {"blue": .75, "green": .25}, "confidence": .6}},
        "usage": {"input_tokens": 16, "output_tokens": 0},
        "diagnostics": {"color": {"label_mass": .8, "argmax_is_label": True}}}


def test_decision_clients_use_control_transports_and_canonical_model():
    async def run():
        captured = []
        async def handler(request):
            assert request.url.path == "/runtime/decide"
            captured.append(json.loads(request.content))
            return httpx.Response(200, json=response())
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            backend = OpenAIChatBackend(HttpRuntimeClient("http://test", client=client))
            assert (await backend.decide(model="canonical", payload=payload()))["answers"]["color"]["choice"] == "blue"
            assert captured[0]["model"] == "canonical"
        def frames(request):
            assert request["op"] == "decide" and request["params"] == payload()
            return [{"v": 1, "id": request["id"], "type": "result", "data": response()}]
        ipc = _client(frames)
        try:
            assert await ipc.decide(payload()) == response()
        finally:
            await ipc.aclose()
    asyncio.run(run())


def test_trained_decision_readout_uses_declared_label_normalization():
    async def run():
        value = response()
        value['answers']['color']['confidence'] = .75
        value['diagnostics']['color'] = {'normalization': 'labels', 'label_mass': None, 'argmax_is_label': None}
        class Client:
            async def decide(self, body): return value
        assert await OpenAIChatBackend(Client()).decide(model='canonical', payload=payload()) == value
    asyncio.run(run())


@pytest.mark.parametrize("mutation", ["nan", "sum", "option", "confidence", "type", "usage", "mass", "diagnostics", "winner"])
def test_decision_backend_rejects_malformed_results(mutation):
    async def run():
        value = copy.deepcopy(response())
        answer = value["answers"]["color"]
        if mutation == "nan": answer["probabilities"]["blue"] = float("nan")
        elif mutation == "sum": answer["probabilities"]["blue"] = .5
        elif mutation == "option": answer["probabilities"] = {"other": 1.}
        elif mutation == "confidence": answer["confidence"] = 1.1
        elif mutation == "type": answer["type"] = "score"
        elif mutation == "mass": value["diagnostics"]["color"]["label_mass"] = .7
        elif mutation == "diagnostics": value.pop("diagnostics")
        elif mutation == "winner": answer["choice"] = "green"
        else: value["usage"]["output_tokens"] = 2
        class Client:
            async def decide(self, body): return value
        backend = OpenAIChatBackend(Client())
        with pytest.raises(BackendProtocolError):
            await backend.decide(model="canonical", payload=payload())
    asyncio.run(run())


def test_decision_pool_releases_exclusive_lease_on_cancellation(tmp_path):
    async def run():
        pool, instance, backend = runtime(tmp_path)
        entered, finish = asyncio.Event(), asyncio.Event()
        async def decide(**request):
            assert request["model"] == "probe" and instance.control_leases == 1
            entered.set()
            await finish.wait()
            return response()
        backend.decide = decide
        pending = asyncio.create_task(pool.decide(model="probe", payload=payload()))
        await asyncio.wait_for(entered.wait(), 1)
        assert instance.control_leases == 1
        pending.cancel()
        with pytest.raises(asyncio.CancelledError): await pending
        assert instance.control_leases == 1
        with pytest.raises(BackendError) as busy: await pool.decide(model="probe", payload=payload())
        assert busy.value.status_code == 409
        finish.set()
        await pool._drain_control_lease_releases()
        assert instance.control_leases == 0
        assert await pool.decide(model="probe", payload=payload()) == response()
        assert instance.control_leases == 0
        instance.active_requests = 1
        with pytest.raises(BackendError) as busy: await pool.decide(model="probe", payload=payload())
        assert busy.value.status_code == 409 and instance.control_leases == 0
    asyncio.run(run())
