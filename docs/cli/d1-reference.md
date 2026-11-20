# Experimental d1 text reference execution

This is an **explicit PyTorch correctness/reference path**, not the native CUDA
or Metal runtime. The reference command does not register serving models or
change native dispatch. For the separate CUDA engine and its existing
`/v1/systemone` integration, see [native d1 execution](d1-native.md#managed-serving).

## Supported contract

The initial target is the official
[LiquidAI/d1-2B snapshot](https://huggingface.co/LiquidAI/d1-2B/tree/a6bce774e6dde338dcc3036ceb5d91f78bf4c19e).
Geometry, layer order, convolution width, GQA heads, context, set size and RoPE
parameters are read from `config.json`, not fixed to a development GPU.

| Capability | Reference path |
| --- | --- |
| Local HF safetensors, tied language output weights | Implemented; unsharded or indexed shards |
| LFM3 gated causal short convolution, full and sliding attention | Implemented in PyTorch |
| Q/K RMSNorm, per-head sigmoid attention gate, sandwich/final norms | Implemented |
| Default zero-tail partial RoPE | Implemented; other RoPE variants rejected |
| Set attention and several answer slots | Implemented; requests beyond trained set size repeat the state in groups |
| Text Noul, single-choice Choice, ordinal Score | Implemented; Score uses the median level, not the mean |
| Candidate single-token aliases/max pooling and temperature buckets | Implemented |
| KV/conv/prefix cache reuse | Explicitly refused; each set is a fresh full forward |
| Multiple choice, span/number/date extraction, estimates, images/video/boxes | Explicitly unsupported in this first text subset |
| GGUF/TQ/quantized execution, native CUDA/Metal, SSD offload, continuous batching | Not implemented by this command |
| Real d1-2B checkpoint correctness/latency/quality | Requires separately recorded real-weight validation; tiny fixtures do not establish this |

The global mask lets state queries see every real token, and a question query
see state and its own question. **This is not strict whole-model question
isolation**: convolution and sliding attention remain causal over the packed
sequence, including earlier questions, as in the official trained contract.
State-prefix caching therefore cannot be borrowed from D3.

Calibration tables are validated and their interpolation is implemented for
contract tests. Typed-answer probability tables are not applied to classification
or used to claim unsupported extraction capabilities. Classification uses the
checkpoint's global/per-question-bucket temperatures.

## Run a local checkpoint

Optional dependencies (no Transformers package or remote custom code required):

```sh
uv sync --project studio/server --extra d1-reference
```

The checkpoint must already exist locally, with `config.json`, `tokenizer.json`,
`calibration.json`, and `model.safetensors` (or index and all referenced shards).
The loader reads only language weights. It does not execute checkpoint Python,
download files, use pickle, or load the vision encoder. A model's license still
applies to its weights and use.

```sh
# Configuration validation only; no tensor allocation or weight loading
uv run --project studio/server tyloinfer d1-reference --model ./models/d1-2B --check-config

# CPU FP32 reference. --device cuda:0 --dtype bfloat16 is an explicit alternative;
# availability of that option is not proof it has been validated on your GPU.
uv run --project studio/server tyloinfer d1-reference --model ./models/d1-2B --request request.json
```

`request.json`:

```json
{
  "state": "I was charged twice. Please refund one charge.",
  "questions": {
    "refund": {"type": "noul", "instructions": "Is a refund requested?"},
    "team": {
      "type": "choice",
      "instructions": "Which team?",
      "criteria": {"billing": "Charges and refunds", "support": "Technical support"}
    },
    "urgency": {
      "type": "score", "instructions": "How urgent?", "criteria": ["low", "medium", "high"]
    }
  }
}
```

A JSON list is accepted for several requests. Request-list processing is
sequential in this memory-bounded reference; it is not continuous batching.
The output wrapper identifies `backend=pytorch-lfm3-text-reference` and
`native_runtime=false`. `reference_revision` names the implementation reference,
**not a verified hash of the user's local model weights**. The inner result has
named `answers` and `usage`, with `output_tokens=0`.

## Validation

Basic CPU tests use real small random safetensors and the actual forward, not a
mocked model. They test loading, slots, padding, grouping, repeat-call isolation,
malformed contracts and the CLI. They are not real d1-2B quality evidence.

```sh
python -m pytest studio/server/tests/test_d1_reference.py studio/server/tests/test_lfm3_reference.py studio/server/tests/test_d1_reference_edges.py -q
```

For optional fixed-source comparisons, prepare the official small source and
tokenizer files at the pinned revision, then point
`TYLOINFER_D1_REFERENCE_SOURCE` to that directory. No test auto-downloads them.
Operator tests isolate torch-only definitions from the official source so a
locally older Transformers does not silently replace the checkpoint's equations.
Prompt/readout tests use the actual official prompt/calibration/API modules and
tokenizer. FP32 and BF16 tolerance checks are numerical tests, not speed tests.

```sh
python -m pytest studio/server/tests/test_lfm3_reference.py studio/server/tests/test_d1_official_parity.py -q
```

`TYLOINFER_D1_MODEL` opts into loading a real local checkpoint for the smoke test.
A smoke test alone does not establish official probability parity or real-model
performance. A separate [standalone CUDA/cuBLAS command](d1-native.md) now provides
native text execution and packed Q8_0 weight-only GPU projections. That command
is not native serving/API registration, Metal support or A8 activation computation.

## Licensing

The prompt/readout adaptation retains the **LFM Open License v1.0**, including
its commercial revenue threshold; see `LICENSES/LiquidAI-d1-LFM-Open-License.txt`
and `NOTICE`. It must not be represented as entirely Apache-2.0. The LFM3
operator source carried by the snapshot is attributed to HuggingFace under
Apache-2.0. Model weights are not included in the package.
