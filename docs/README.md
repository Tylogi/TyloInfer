# TyloInfer documentation

TyloInfer provides TQ inference, CUDA/Metal runtimes, and local serving.

## Get started

- [Deploy a service and check performance (中文)](deployment.zh-CN.md)
- [Build the C++ runtime](build.md)
- [Run TyloInfer Server and Studio](serve.md)
- [Run the Agent Host or install the DS Bot plugin (中文)](agent-host.md)
- [Complete the first API request](serve.md#first-api-request)
- [Measure inference performance](benchmarking.md)
- [Create a self-contained release](packaging.md)
- [Check model and backend support](runtime/support.md)

## Models and runtimes

- [TQ format and execution](formats/tq.md)
- [Runtime sampling profiles](runtime/sampling-profiles.md)
- [Runtime support matrix](runtime/support.md)
- [Python/C++ runtime transport](runtime/transport.md)
- [CUDA KV memory policies](runtime/cuda-kv-memory.md)
- [CUDA Flash-Next continuous batching](runtime/cuda-continuous-batching.md)
- [Metal Flash-Next speculative decoding](serve.md#metal-native-serving)
- [MiniCPM-o 4.5 multimodal runtime](runtime/minicpmo45.md)

## APIs

TyloInfer provides OpenAI-compatible `/v1/models` and `/v1/chat/completions`,
Anthropic-compatible `/v1/messages`, and an HTTP control API under `/api/v1`.
Token and job updates stream over SSE; audio and realtime sessions use
WebSocket. Function tools are executed by the client and their results returned
through the inference API.

- [HTTP API](api/http.md)
- [WebSocket and realtime API](api/websocket.md)

## Migration plans

- [Application layout and unified frontend build (中文, not yet implemented)](plans/app-layout-migration.md)
