import pytest
from server.server.protocol.models import TokenUsage
from server.server.runtime.backend import BackendProtocolError, OpenAIChatBackend


def test_cached_usage_survives_backend_validation():
    event = {"choices": [], "usage": {"prompt_tokens": 131120,
        "completion_tokens": 256, "total_tokens": 131376,
        "prompt_tokens_details": {"cached_tokens": 131072}}}
    delta = OpenAIChatBackend._parse_event(event)
    assert delta.usage.prompt_tokens_details.cached_tokens == 131072
    assert delta.usage.model_dump(mode="json", exclude_none=True) == event["usage"]


def test_legacy_usage_does_not_invent_cache_hits():
    usage = {"prompt_tokens": 48, "completion_tokens": 2, "total_tokens": 50}
    parsed = TokenUsage.model_validate(usage)
    assert parsed.prompt_tokens_details is None
    assert parsed.model_dump(mode="json", exclude_none=True) == usage


@pytest.mark.parametrize("details", [{"cached_tokens": -1}, {"cached_tokens": 2, "unknown": 1}, {}])
def test_invalid_cached_usage_is_rejected(details):
    event = {"choices": [], "usage": {"prompt_tokens": 48,
        "completion_tokens": 2, "total_tokens": 50, "prompt_tokens_details": details}}
    with pytest.raises(BackendProtocolError):
        OpenAIChatBackend._parse_event(event)


def test_restore_metrics_do_not_change_ttft_or_prefill_accounting():
    performance = {"prefill_tokens": 48, "cached_tokens": 131072,
        "prefix_restore_ms": 120.0, "ttft_ms": 400.0, "prefill_ms": 250.0,
        "prefill_tps": 192.0, "decode_ms": 1000.0, "decode_tps": 10.0,
        "generation_ms": 1400.0, "generation_tps": 8.0,
        "sampling": {"max_tokens": 256, "temperature": 0, "top_k": 1, "top_p": 1}}
    delta = OpenAIChatBackend._parse_event({"choices": [], "mfq_metrics": performance})
    data = delta.performance.model_dump(mode="json")
    assert data["ttft_ms"] == 400.0
    assert data["prefill_tokens"] == 48
    assert data["cached_tokens"] == 131072
    assert data["prefix_restore_ms"] == 120.0
