# TyloInfer documentation

TyloInfer provides TQ inference, CUDA/Metal runtimes, and local serving.

These source-checkout guides describe the current public `tyloinfer` CLI.
English references are complemented by a Chinese deployment guide; they are
not a complete bilingual manual. Use documentation from the same Git revision
as the installed code. Internal names such as `mfq-runtime`,
`build/mfq-runtime.json`, `mfq_metrics`, and the `MFQ1` container magic remain
valid; there is no public `mfq` command registered by this package.

## Get started

- [Deploy a service and check performance (中文)](deployment.zh-CN.md)
- [Build the C++ runtime](cli/build.md)
- [Run TyloInfer Server and Studio](cli/serve.md)
- [Complete the first API request](cli/serve.md#first-api-request)
- [Record performance and understand reproduction limits](benchmarking.md)
- [Create a self-contained release](packaging.md)
- [Check model and backend support](runtime/support.md)

## Models and runtimes

- [TQ format and execution](formats/tq.md)
- [Runtime sampling profiles](runtime/sampling-profiles.md)
- [Runtime support matrix](runtime/support.md)
- [Python/C++ runtime transport](runtime/transport.md)
- [CUDA KV memory policies](runtime/cuda-kv-memory.md)
- [Flash-Next continuous batching](runtime/cuda-continuous-batching.md)
- [MiniCPM-o 4.5 multimodal runtime](runtime/minicpmo45.md)

## Documentation checks

From the repository root, run `python3 -B docs/check_consistency.py`. This
read-only standard-library check covers public command names, local Markdown
links/anchors, JSON/JSONL examples, the HTTP API table against checked-in
OpenAPI, and selected CLI defaults/options. It does not build the native
runtime, launch the UI, validate external download links, or run inference.

## APIs

TyloInfer provides OpenAI-compatible `/v1/models` and `/v1/chat/completions`,
Anthropic-compatible `/v1/messages`, and an HTTP control API under `/api/v1`.
Token and job updates stream over SSE; audio and realtime sessions use
WebSocket. Function tools are executed by the client and their results returned
through the inference API.

- [HTTP API](api/http.md)
- [WebSocket and realtime API](api/websocket.md)
