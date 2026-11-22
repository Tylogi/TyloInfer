"""Opt-in end-to-end checks for the real native stdio runtime."""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import suppress
from pathlib import Path
from typing import Any

import pytest

from tests.native_stdio_helpers import (
    INDEPENDENT_PROMPT,
    MissingNativeRuntimeAssetsError,
    NativeRuntimeAssets,
    NativeStdioSession,
    ProductionStdioSession,
    generation_request,
    invalid_media_request,
    resolve_native_runtime_assets,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def native_runtime_assets() -> NativeRuntimeAssets:
    """Skip locally, but fail explicitly in the required real-runtime job."""

    try:
        return resolve_native_runtime_assets(os.environ, root=ROOT)
    except MissingNativeRuntimeAssetsError as error:
        message = str(error)
        if os.environ.get("TYLOINFER_REQUIRE_REAL_RUNTIME") == "1":
            pytest.fail(message)
        pytest.skip(message)
    except FileNotFoundError as error:
        pytest.fail(str(error))


async def ready_session(session: NativeStdioSession) -> dict[str, Any]:
    """Verify startup and the model capability contract before a scenario."""

    assert (await session.receive())["type"] == "ready"
    await session.send("models", "models")
    metadata = (await session.until("models", "result"))[0]["data"]["models"][0]
    assert metadata["type"]
    assert metadata["capabilities"]["architecture_family"] != "unknown"
    assert metadata["capabilities"]["source"] == "model-graph+cuda-adapters"
    assert metadata["capabilities"]["features"]["text"]
    return metadata


def unary_result(frames: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the unary result, which has no event discriminator."""

    results = [
        frame["data"]
        for frame in frames
        if frame.get("type") == "event" and "event" not in frame.get("data", {})
    ]
    assert len(results) == 1, frames
    assert {"output", "usage", "finish_reason", "metrics"} <= results[0].keys()
    return results[0]


def stream_completion(frames: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the unique stream completion and validate its required fields."""

    completions = [
        frame["data"]
        for frame in frames
        if frame.get("type") == "event" and frame.get("data", {}).get("event") == "complete"
    ]
    assert len(completions) == 1, frames
    assert {"finish_reason", "metrics"} <= completions[0].keys()
    return completions[0]


@pytest.mark.real_runtime
@pytest.mark.parametrize("batch_size", [0, 2])
def test_native_stdio_loads_and_generates(native_runtime_assets: NativeRuntimeAssets, tmp_path: Path, batch_size: int) -> None:
    """A real model must load and complete one deterministic request."""

    async def scenario() -> None:
        async with NativeStdioSession(native_runtime_assets, batch_size=batch_size, log_path=tmp_path / "runtime.log") as session:
            metadata = await ready_session(session)
            await session.send("plain", "generate", generation_request(False))
            result = unary_result(await session.until("plain"))
            assert result["usage"]["prompt_tokens"] > 0
            assert result["usage"]["completion_tokens"] > 0
            assert "mtp_available" in result["metrics"]
            assert metadata["capabilities"]["features"]["text"]
            await session.shutdown()

    asyncio.run(scenario())


@pytest.mark.real_runtime
def test_native_stdio_production_client_matches_reference_oracle(
    native_runtime_assets: NativeRuntimeAssets, tmp_path: Path
) -> None:
    """Compare one native completion with the independent HF fixture."""

    fixture_path = ROOT / "studio" / "tests" / "fixtures" / "native_reference" / "qwen35_08b.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert fixture["schema_version"] == 1
    assert fixture["case"] == "qwen3.5-0.8b-count-from-one-to-ten"

    model_dir = native_runtime_assets.model
    resources = {
        "config_sha256": model_dir / "config.json",
        "weights_sha256": model_dir / "model.safetensors-00001-of-00001.safetensors",
        "tokenizer_sha256": model_dir / "tokenizer.json",
    }
    for digest_name, path in resources.items():
        assert path.is_file(), path
        assert sha256_file(path) == fixture["model"][digest_name]

    prompt = fixture["input"]["prompt"]
    max_new_tokens = fixture["input"]["max_new_tokens"]
    sampling = fixture["sampling"]
    request = {
        "messages": [{"role": "user", "content": "Count from one to ten."}],
        "mfq_preformatted_prompt": prompt,
        "max_tokens": max_new_tokens,
        "temperature": sampling["temperature"],
        "top_k": sampling["top_k"],
        "top_p": sampling["top_p"],
        "enable_mtp": False,
        "stream": False,
        "mfq_prefix_cache_enabled": False,
    }

    async def scenario() -> None:
        async with ProductionStdioSession(
            native_runtime_assets,
            batch_size=0,
            context_size=512,
            log_path=tmp_path / "reference-runtime.log",
        ) as session:
            assert session.client is not None
            async with session.client.generate(request) as events:
                output_events = [event async for event in events if event is not None]
            result = output_events[-1]
            choice = result["choices"][0]
            actual_text = choice["delta"]["content"]
            expected = fixture["expected"]
            assert actual_text == expected["generated_text"]
            from tokenizers import Tokenizer

            tokenizer = Tokenizer.from_file(str(resources["tokenizer_sha256"]))
            actual_ids = tokenizer.encode(actual_text, add_special_tokens=False).ids
            assert actual_ids == expected["generated_token_ids"]
            assert result["usage"]["completion_tokens"] == max_new_tokens
            assert result["usage"]["prompt_tokens"] == len(
                tokenizer.encode(prompt, add_special_tokens=False).ids
            )

    asyncio.run(scenario())


@pytest.mark.real_runtime
@pytest.mark.parametrize("batch_size", [0, 2])
def test_native_stdio_stream_matches_non_stream(native_runtime_assets: NativeRuntimeAssets, tmp_path: Path, batch_size: int) -> None:
    """Streaming deltas must reconstruct the same output as unary generation."""

    async def scenario() -> None:
        async with NativeStdioSession(native_runtime_assets, batch_size=batch_size, log_path=tmp_path / "runtime.log") as session:
            await ready_session(session)
            await session.send("plain", "generate", generation_request(False))
            plain = unary_result(await session.until("plain"))
            await session.send("stream", "generate", generation_request(True))
            streamed = await session.until("stream")
            deltas = [frame["data"]["delta"] for frame in streamed if frame.get("data", {}).get("event") == "delta"]
            assert "".join(delta.get("content", "") for delta in deltas) == plain["output"]["text"]
            assert "".join(delta.get("reasoning_content", "") for delta in deltas) == plain["output"].get("reasoning", "")
            stream_completion(streamed)
            await session.shutdown()

    asyncio.run(scenario())


@pytest.mark.real_runtime
@pytest.mark.parametrize("batch_size", [0, 2])
def test_native_stdio_rejects_bad_request_and_recovers(native_runtime_assets: NativeRuntimeAssets, tmp_path: Path, batch_size: int) -> None:
    """A request-level validation error must not poison the next request."""

    async def scenario() -> None:
        async with NativeStdioSession(native_runtime_assets, batch_size=batch_size, log_path=tmp_path / "runtime.log") as session:
            await ready_session(session)
            await session.send("plain", "generate", generation_request(False))
            plain = unary_result(await session.until("plain"))
            await session.send("invalid-media", "generate", invalid_media_request())
            error = await session.receive()
            assert error["id"] == "invalid-media"
            assert error["type"] == "error"
            assert error["error"]["status_code"] == 400
            await session.send("after-invalid", "generate", generation_request(False))
            recovered = unary_result(await session.until("after-invalid"))
            assert recovered["output"] == plain["output"]
            assert recovered["usage"]["completion_tokens"] > 0
            await session.shutdown()

    asyncio.run(scenario())


@pytest.mark.real_runtime
@pytest.mark.parametrize("batch_size", [0, 2])
def test_native_stdio_cancel_and_reload_recover(native_runtime_assets: NativeRuntimeAssets, tmp_path: Path, batch_size: int) -> None:
    """Cancellation attempts and reload leave a usable runtime even if generation finishes first."""

    async def scenario() -> None:
        async with NativeStdioSession(native_runtime_assets, batch_size=batch_size, log_path=tmp_path / "runtime.log") as session:
            await ready_session(session)
            await session.send("plain", "generate", generation_request(False))
            plain = unary_result(await session.until("plain"))
            await session.send("cancelled", "generate", generation_request(True, 200))
            await session.until_event("cancelled", lambda data: data.get("event") == "delta")
            await session.send("cancel", "request.cancel", {"target_id": "cancelled"})
            cancelled = await session.until("cancelled")
            await session.until("cancel", "result")
            assert stream_completion(cancelled)["finish_reason"] in {"cancelled", "stop", "length"}
            await session.send("reload-victim", "generate", generation_request(True, 200))
            await session.until_event("reload-victim", lambda data: data.get("event") == "delta")
            await session.send("reload", "reload", {"context_size": 128})
            victim = await session.until("reload-victim")
            await session.until("reload", "result")
            assert stream_completion(victim)["finish_reason"] in {"cancelled", "stop", "length"}
            assert session.frames["reload"][0]["data"]["max_context"] == 128
            await session.send("after", "generate", generation_request(False))
            after = unary_result(await session.until("after"))
            assert after["output"] == plain["output"]
            assert after["usage"]["completion_tokens"] > 0
            await session.shutdown()

    asyncio.run(scenario())


@pytest.mark.real_runtime
@pytest.mark.parametrize("batch_size", [0, 2])
def test_native_stdio_mtp_matches_plain_when_available(
    native_runtime_assets: NativeRuntimeAssets, tmp_path: Path, batch_size: int
) -> None:
    """If the loaded model advertises MTP, its output must match plain decoding."""

    async def scenario() -> None:
        async with NativeStdioSession(
            native_runtime_assets, batch_size=batch_size, log_path=tmp_path / "runtime.log"
        ) as session:
            metadata = await ready_session(session)
            mtp_feature = metadata["capabilities"]["features"]["mtp"]
            if not mtp_feature:
                if os.environ.get("TYLOINFER_STEP_TEST_MTP") == "1":
                    pytest.fail("MTP was required but the loaded model does not advertise it")
                pytest.skip("loaded model does not advertise MTP")
            await session.send("plain", "generate", generation_request(False))
            plain = unary_result(await session.until("plain"))
            assert "mtp_available" in plain["metrics"]
            if not plain["metrics"]["mtp_available"]:
                if os.environ.get("TYLOINFER_STEP_TEST_MTP") == "1":
                    pytest.fail("MTP was required but the runtime did not make it available")
                pytest.skip("runtime did not make MTP available for this model")
            mtp_request = generation_request(False)
            mtp_request["sampling"]["enable_mtp"] = True
            await session.send("mtp", "generate", mtp_request)
            mtp = unary_result(await session.until("mtp"))
            assert mtp["output"] == plain["output"]
            assert mtp["metrics"]["mtp_used"]
            assert mtp["metrics"]["mtp_cycles"] > 0
            await session.shutdown()

    asyncio.run(scenario())


@pytest.mark.real_runtime
@pytest.mark.parametrize("batch_size", [0, 2])
def test_native_stdio_json_schema_output(
    native_runtime_assets: NativeRuntimeAssets, tmp_path: Path, batch_size: int
) -> None:
    """Schema-constrained generation must produce valid JSON satisfying the schema."""

    async def scenario() -> None:
        async with NativeStdioSession(
            native_runtime_assets, batch_size=batch_size, log_path=tmp_path / "runtime.log"
        ) as session:
            await ready_session(session)
            constrained = generation_request(False, 32)
            constrained["template"] = {"enable_thinking": False}
            constrained["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "schema": {
                        "type": "object",
                        "properties": {"ok": {"const": True}},
                        "required": ["ok"],
                        "additionalProperties": False,
                    }
                },
            }
            await session.send("schema", "generate", constrained)
            result = unary_result(await session.until("schema"))
            assert json.loads(result["output"]["text"]) == {"ok": True}
            await session.shutdown()

    asyncio.run(scenario())


@pytest.mark.real_runtime
def test_native_stdio_production_client_multiplexes_and_recovers(
    native_runtime_assets: NativeRuntimeAssets, tmp_path: Path
) -> None:
    """Check two native submissions and preserve B's output during A's cancellation race."""

    async def scenario() -> None:
        async with ProductionStdioSession(
            native_runtime_assets,
            batch_size=2,
            context_size=512,
            log_path=tmp_path / "production-runtime.log",
        ) as session:
            assert session.client is not None
            client = session.client
            models = await client.models()
            assert models["data"]

            def request(prompt: str, tokens: int = 128) -> dict[str, Any]:
                return {
                    "messages": [{"role": "user", "content": prompt}],
                    "mfq_preformatted_prompt": prompt,
                    "max_tokens": tokens,
                    "temperature": 0,
                    "top_k": 1,
                    "enable_mtp": False,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "mfq_prefix_cache_enabled": False,
                }

            prompts = {
                # Distinct prompts expose cross-request output contamination.
                "a": "A " * 96,
                "b": INDEPENDENT_PROMPT,
            }
            async with client.generate(request(prompts["b"])) as events:
                baseline_events = [event async for event in events if event is not None]
            baseline_text = "".join(
                choice.get("delta", {}).get("content", "")
                for event in baseline_events
                for choice in event.get("choices", [])
            )
            assert baseline_text
            baseline_usage = next(
                event["usage"]
                for event in baseline_events
                if isinstance(event.get("usage"), dict)
            )
            observed: list[tuple[str, dict[str, Any]]] = []
            completed: dict[str, bool] = {label: False for label in prompts}
            outputs: dict[str, list[str]] = {label: [] for label in prompts}
            usages: dict[str, dict[str, Any]] = {}
            finish_reasons: dict[str, str] = {}
            outbound_start = len(session.outbound_frames)
            inbound_start = len(session.inbound_frames)

            async def collect(label: str) -> None:
                async with client.generate(request(prompts[label])) as events:
                    async for event in events:
                        if event is None:
                            completed[label] = True
                            return
                        observed.append((label, event))
                        choices = event.get("choices", [])
                        if choices:
                            choice = choices[0]
                            delta = choice.get("delta", {})
                            if isinstance(delta, dict):
                                content = str(delta.get("content", ""))
                                outputs[label].append(content)
                            finish_reason = choice.get("finish_reason")
                            if finish_reason is not None:
                                finish_reasons[label] = str(finish_reason)
                                completed[label] = True
                        if isinstance(event.get("usage"), dict):
                            usages[label] = event["usage"]

            task_a = asyncio.create_task(collect("a"))
            task_b = asyncio.create_task(collect("b"))
            try:
                # The stdin wiretap is a deterministic protocol boundary: both
                # generate frames must be written to native stdin before A is
                # cancelled.  A's non-empty delta then proves it was admitted;
                # no event-arrival ordering assumption is needed for B.
                a_frame, b_frame = await asyncio.gather(
                    session.wait_for_outbound(
                        lambda frame: frame.get("op") == "generate"
                        and frame.get("params", {}).get("input", {}).get("preformatted_prompt")
                        == prompts["a"],
                        start_index=outbound_start,
                    ),
                    session.wait_for_outbound(
                        lambda frame: frame.get("op") == "generate"
                        and frame.get("params", {}).get("input", {}).get("preformatted_prompt")
                        == prompts["b"],
                        start_index=outbound_start,
                    ),
                )
                await session.wait_for_inbound(
                    lambda frame: frame.get("id") == a_frame.get("id")
                    and frame.get("type") == "event"
                    and frame.get("data", {}).get("event") == "delta"
                    and frame.get("data", {}).get("delta", {}).get("content"),
                    timeout=90,
                )

                a_terminal_before_cancel = any(
                    frame.get("id") == a_frame.get("id")
                    and frame.get("type") == "event"
                    and frame.get("data", {}).get("event") == "complete"
                    for frame in session.inbound_frames
                )
                cancel_start = len(session.outbound_frames)
                task_a.cancel()
                with suppress(asyncio.CancelledError):
                    await asyncio.wait_for(task_a, timeout=30)
                cancel_frame: dict[str, Any] | None = None
                if not a_terminal_before_cancel:
                    # The cancellation frame itself is the boundary we can
                    # observe reliably.  Do not infer cancellation from the
                    # Python task being cancelled; the native worker may have
                    # already completed the request in the meantime.
                    cancel_frame = await session.wait_for_outbound(
                        lambda frame: frame.get("op") == "request.cancel"
                        and frame.get("params", {}).get("target_id") == a_frame.get("id"),
                        timeout=30,
                        start_index=cancel_start,
                    )
                a_complete_frame = await session.wait_for_inbound(
                    lambda frame: frame.get("id") == a_frame.get("id")
                    and frame.get("type") == "event"
                    and frame.get("data", {}).get("event") == "complete",
                    timeout=30,
                )
                await asyncio.wait_for(task_b, timeout=90)
                assert completed["b"]
                assert finish_reasons["b"] in {"stop", "length"}
                assert outputs["b"]
                assert "".join(outputs["b"]) == baseline_text
                assert usages["b"]["prompt_tokens"] == baseline_usage["prompt_tokens"]
                assert usages["b"]["completion_tokens"] == baseline_usage["completion_tokens"]

                if a_terminal_before_cancel:
                    assert not cancel_frame
                else:
                    assert cancel_frame is not None
                cancel_frames = [
                    frame
                    for frame in session.outbound_frames[cancel_start:]
                    if frame.get("op") == "request.cancel"
                ]
                assert all(
                    frame["params"].get("target_id") != b_frame["id"]
                    for frame in cancel_frames
                )
                a_finish_reason = a_complete_frame["data"].get("finish_reason")
                assert a_finish_reason in {"cancelled", "stop", "length"}
                # A native stop/length completion is a legal race with the
                # cancellation request; only ``cancelled`` is cancellation
                # evidence.  The deterministic late-cancel contract is
                # covered by the CPU controlled-child test.
                b_complete_frame = await session.wait_for_inbound(
                    lambda frame: frame.get("id") == b_frame["id"]
                    and frame.get("type") == "event"
                    and frame.get("data", {}).get("event") == "complete",
                    timeout=30,
                    start_index=inbound_start,
                )
                assert b_complete_frame["data"]["finish_reason"] in {"stop", "length"}
                a_response_id = a_complete_frame["data"]["request_id"]
                b_response_id = b_complete_frame["data"]["request_id"]
                assert isinstance(a_response_id, str) and a_response_id
                assert isinstance(b_response_id, str) and b_response_id
                assert a_response_id != b_response_id
                # A may be cancelled after its native delta arrives but before
                # its Python consumer resumes. Every delivered event must still
                # belong to its request; B must receive its complete stream.
                assert all(
                    event.get("id") == a_response_id
                    for label, event in observed if label == "a"
                )
                assert {
                    event.get("id") for label, event in observed if label == "b"
                } == {b_response_id}

                # The cancellation must not poison a subsequent request.
                recovery_request = request(INDEPENDENT_PROMPT, tokens=16)
                async with client.generate(recovery_request) as events:
                    recovery_events = [event async for event in events if event is not None]
                assert recovery_events
                usage = next(
                    event["usage"]
                    for event in recovery_events
                    if isinstance(event.get("usage"), dict)
                )
                assert usage["prompt_tokens"] == baseline_usage["prompt_tokens"]
            finally:
                for task in (task_a, task_b):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(task_a, task_b, return_exceptions=True)

    asyncio.run(scenario())
