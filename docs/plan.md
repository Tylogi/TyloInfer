# TyloInfer Refactoring Plan

[architecture.md](architecture.md) defines the contracts. This plan tracks outstanding migration and acceptance.
Paths are relative to the repository root unless a section gives a different base.

## Scope and execution order

- Retain native Engine CLI (`tyloi-engine`) and Server (`tyloinfer-server`). The C++ Server and Studio source are outside the current round. Diagnostics, LibTorch and project Torch C++ bindings stay removed.
- Compare merge `1e10e19b` against architecture parent `0cd82a6f` and incoming main `a2d46607` (common ancestor `ea863676`). Migrate required behavior into `csrc`. Retire imported code once its behavior and callers have a documented destination.
- Preserve settled interfaces, MFQ/TQ format identifiers, protocol/database keys and stored identities. Pure moves wait for P3.
- Keep tests and deprecated Python consumers frozen. Record outstanding test references in the [test porting guide](test-porting-guide.md).
- Avoid concurrent resident models and heavy checks on the 32 GB machine. Let the build choose its own concurrency.

| Order | Work |
| --- | --- |
| First | P0.3 device acceptance and P1.1 CUDA gaps |
| Alongside each slice | P0.0 FFN, P0.1 lifetime, P0.2 source closure and P1.3 acceptance |
| Next | P1.2 Jev acceptance, then P2.2 supported CUDA and Metal convergence |
| Deferred | P0.4 frozen callers and HIP, P2.1 Server and Studio, P2.3 schema/launcher |
| Last | P3 file consolidation |

**Interface fragmentation across `csrc`.** Review real call chains as each area comes up, including apps and commands,
components, transport, Server, Engine and both backend trees. Include tests as callers when tracing an interface.
Record the repeated decision, the callers, its existing owner and the behavior each caller needs. Prioritize cases
where the implementations already disagree. The first Engine cases are under P0.0 (FFN) and P1.1 (prefill and batch admission).

## P0.4 — Retire `TYLOINFER_RUNTIME_CHECK`

- After the test freeze lifts, migrate `csrc/engine/backends/cuda/tests/qsa_indexer_cache_test.cpp` and `float_projection_test.cpp`, then delete the macro and `mfq_runtime_check_failed` from `csrc/engine/backends/cuda/native/tensor/backend.h`.
- Retire the remaining imported callers under P0.2. Handle `csrc/backends/hip/include/backend.h` with the deferred HIP migration.
- Verify invalid shape, dtype and device errors, active-branch retirement, and scheduler and transport error classification. Preserve release-build checks and native CUDA/HIP API error translation.

## P0.0 — FFN migration and acceptance

- Port remaining scheduling and fusion from imported CUDA FFN/cache callbacks into backend ops and Executor. Respect the model, numerical ops and weight preparation responsibilities in [architecture.md](architecture.md).
- Verify independent MLP, routed and shared results, gate precision, clamp, accumulation, calibrated projection rows, 8/16-row tiles, Qwen4 prefill/decode/batch/predictor, CUDA capture and Metal eager execution.

## P0.1 — Lifetime, memory and IO

| Location | Remaining work |
| --- | --- |
| `csrc/engine/engine.*`, `csrc/engine/scheduler/` | Verify serialized reload, pending and active requests, cancellation/shutdown and partial-load cleanup. Cover old/new model co-residency and publication after retirement |
| `csrc/engine/state/session/{prefix_cache,prefix_blocks,cache}.*`, `state/disk_budget.*` under Engine | Verify admission closure, pending IO and budget work, Store indexes, live readers returning `runtime_busy`, refresh failure and retained `.mfqkv`/`.tqkv` compatibility |
| `csrc/engine/backends/cuda/native/{stream,stream_context,stream_signal,graph,memory}.*`, Executor and MoE residency | Reconcile remaining submission, event and retirement deltas. Verify original stream/context, eager and capture ordering, driver ABI, multi-device completion, failure cleanup, backup restore retry and live suballocation leases |
| `csrc/engine/memory/{host,budget,kv}.*`, row and expert caches | Verify OS and process headroom, eviction with live leases, deduplicated physical accounting through final DMA, and conservative sampling failure |
| CUDA QSA, GGML ops and Qwen models | Verify contiguous allocation under fragmentation, resident GDN/PLE/position/QSA payloads, and temporary loans retiring after the last consumer |
| `csrc/engine/io/`, `csrc/engine/weights/source/mapped.cpp` and prefix/session callers | Validate Windows relative, long, UNC and extended paths, concurrent Stores, pressure, partial IO failures and async reader lifetime |

## P0.2 — Source and package closure

- Compare remaining imports in `cpp_runtime/` and `tyloinfer/kernels/cuda/` against the fixed merge parents. Record the replacement, deferred consumer or reason for removal. Resolve raw-header consumers with the relevant migration.
- Verify native release packaging and TQ JIT execution. Recheck affected source lists and shipped assets as files move. Production builds use `csrc` with private SDK definitions and no Torch/Python development or `cpp_runtime` link dependency.

## P0.3 — Cache and LayerStream device acceptance

Paths are relative to `csrc/engine/`.

| Location | Checks |
| --- | --- |
| `weights/cache/`, `memory/budget.cpp`, `generation/`, `backends/cuda/executor/` | Required MFE layout, shared remaining TQ/GGML/MFE capacity, resize/rollback and pressure |
| `weights/cache/ggml.*`, `weights/moe/prepare/projection.h`, CUDA GGML projection | Source IO outside residency locks, two reusable pinned packets, DMA and pageable fallback, wait on the consuming stream, lease lifetime under eviction and cancellation |
| `weights/layer_stream.h`, Qwen35 load/forward, CUDA Executor staging | Streamed/resident parity, transfer overlap, startup transfer volume, peak RAM/VRAM, first allocation fence, retirement on exception and cancellation, unchanged recurrent and KV State |

## P1.1 — CUDA and Qwen

Paths are relative to `csrc/engine/`.

**Let the native batch answer whether it accepts a request.** `engine.cpp::Engine::can_batch` reads four facts and passes them to `generation/batch/ops.cpp::batch_compatible`. The native operations object already holds Sessions, the predictor and the model batch. Move the predicate there and expose a request-level query through `generation/batch/batch.h`. Engine checks for a batch and delegates. Cover decision/media requests, token IDs versus stable prefix, explicit sessions and MTP with an absent, usable or incompatible predictor. Preserve the legacy Metal template caller in `csrc/backends/metal/apps/mfq_decode_mlx.cpp` and track frozen test callers when narrowing the native interface.

| Location | Remaining migration and checks |
| --- | --- |
| `tensor/encoding/`, `weights/{source,prepare}/`, CUDA projection ops and kernels | Verify scaled QKV, gate and fused attention readout for GGML/TQ/NINT/NVQ. Cover workspace fallback, shared scaled RMS input, scales, zero points, shapes, shards, ranges, embedding and reconstruction, FP32 CPU/FP16 device correction, TP fallback, rounding, rotated/corrected fallback, calibrated geometry and workspace limits |
| `weights/{cache,tq,moe}/`, `backends/cuda/ops/{gguf,ffn}/` | Migrate streamed GGML batch and draft verification Graphs, rotated and corrected GGML replay, and remaining routed prefill consumers. Verify checkpoint parity and Gate/Up-before-Down overlap, resident capture, mixed grouped and individual FP16 prefill, DMA fallback, TQ workspace loan retirement and failed upload cleanup |
| Same routed owners and generation batch driver | Validate per-token capacity, aligned bank costs, rejected draft masks, learning freeze, and per-request GGML prefill acceptance for cancelled rows, split batches and failed admission |
| `weights/rows/`, `models/qwen4_exp/`, CUDA row/projection consumers | Verify packet matching, eviction fallback, host limits, virtual ranges, Graph upload fences and backing lifetime |
| `models/qwen4_exp/`, `generation/prefill/`, `backends/cuda/executor/qsa/` | Finish remaining prefill work. Verify phase/storage identity, block positions, warm greedy chains, workspace return before publication and cancellation, and single versus packed tiled admission with live expert loans |
| `models/{qwen35,qwen4_exp}/`, CUDA attention/FFN, `generation/mtp/`, `state/kv/qsa/`, `models/session/` | Verify precision, eligibility and fallback, independent GDN/KV State, rollback, device IDs, readback, DFlash2 seed/commit and draft/target Graphs, mapped codebooks, candidate staging and v1/v2/v3 session compatibility |
| `models/qwen4_exp/`, CUDA model loading and Graph | Run dense Qwen4 checkpoint inference and parity checks. Complete specialized Graph and batch admission after device acceptance |
| `models/deepseek_v4/`, `weights/prepare/`, CUDA projection | Verify packed compressor shapes and reconstruction, short-window calibration and physical host/device/offload memory |

Preserve immutable resident predictor banks, bounded upload buffers, compute-stream loan retirement, and independent request State. Ordinary and MTP batches commit predictor history to emitted tokens. Keep Graph admission explicit.

## P1.2 — Jev acceptance

| Location | Checks |
| --- | --- |
| `csrc/engine/models/source/{decision,hf_source,model}.*` and family loading | Validate `decision_config.json`, trained D3/GGUF metadata, `readout.safetensors`, supported backbone, matrix shape/dtype/content, byte limits and tokenizer codebook |
| `csrc/engine/weights/layer_stream.h`, Qwen35 loading, CUDA Executor | Compare `--jev-weights stream\|vram` outputs, embedding/readout placement, scratch and RAM/VRAM, overlap and failure retirement |
| `csrc/engine/requests/`, `csrc/engine/generation/decision.cpp`, Scheduler and Engine | Validate choice/score/noul modes, source limits, temperatures, conditional label probability, full-vocabulary confidence, mass, argmax, counts and timing |
| `csrc/engine/models/input.h`, shared/family attention and session identity | Verify causal, bidirectional and prefix-bidirectional behavior, cache identity and cancellation/reload |
| `csrc/components/protocol/decision.*`, `csrc/transport/src/{http,stdio}.cpp`, CLI | Verify parsing, results, errors and cancellation for HTTP `/runtime/decide`, `/v1/decisions` and stdio `decide` |

## P1.3 — Acceptance by slice

| Pending check | Coverage |
| --- | --- |
| Device and precision | GPU checkpoint loading and inference, fixed-input independent references, dtype tolerances, fused and unfused output, small and tail rows, encoded formats |
| Qwen4 | Factory/placement/partial-load cleanup, GDN/QSA/PLE, FFN/residual/logits, ordinary/chunked/layer-major prefill and decode, MTP accept/reject/replay, batch isolation and session transactions |
| Other models | Affected Qwen35, Gemma4, DeepSeek, GLM and MiniCPM paths |
| Lifetime | Graph warmup/storage/invalidation, streamed/resident experts, mixed caches, fragmentation, cancel/reload/unload, coroutine scope release and Metal lazy completion |
| Platform and distribution | Supported Metal build, SDK-free public headers, packaging and release closure |
| Frozen callers | Update outdated references listed in the [test porting guide](test-porting-guide.md) once changes are authorized |

Track source, build, numerics, lifetime, performance and packages separately. Static inspection and skipped device checks leave acceptance open.

## P2.1 — Server and Studio follow-up

Deferred from this round.

- Standalone transport health responses in `csrc/transport/src/{http,stdio}.cpp` independently assemble the same capability fields and append Engine/session metrics. When touching these responses, put their common JSON projection in `csrc/components/protocol/`. Compare both payloads for the same snapshot and settings, including loading status. Keep HTTP response handling and stdio framing local.
- Server's `fields()` in `csrc/server/api/{models,runtime,runtime_config,runtime_cache}.cpp` repeats the JSON-object and allowed-key checks. Consolidate those checks in the existing Server HTTP helpers in `api/http.h`. Each handler supplies its allowed keys and diagnostic wording. Verify HTTP 400, `invalid_request` and the offending key for an unknown parameter.
- Publish real decision availability and streamed/resident load options through `csrc/server/models/`, `csrc/server/api/runtime*.cpp` and Studio.
- Align `backend`, `ram_embedding_bytes`, `ram_weight_bytes` and resident/buffer/process metrics. Preserve the distinction between estimated, allocated and unavailable values.
- Verify authentication, aliases, cancellation, reload, persistence and matching results across producer, protocol and UI. Preserve stored IDs, wire `architecture`, database keys and frozen Python contracts.

## P2.2 — Supported CUDA and Metal convergence

Paths are relative to `csrc/engine/` unless stated otherwise.

- Move remaining semantics from legacy `csrc/backends/metal/` and backend-private family headers into family owners. Move numerical work into backend ops. Preserve GLM5Next zero-limit clamp/F32 routing/ordered accumulation and direct KDA/MLA/PLE/compressed-attention phases.
- Verify DeepSeek V4 MFE/NINT host offload staging lifetime, format parity and physical peak. Retain supported placement and execution limits.
- Verify ordinary/MTP batching and joint restore in Qwen35/Qwen4, including independent states, paged KV copies, ragged accepts, emitted-token history, predictor rebuild and output-before-snapshot ordering.
- Verify QSA policy/codec/executor lifetimes and pending test migration in `state/kv/qsa/`, `tensor/encoding/qsa/` and `executor/qsa/`.
- Finish native Metal MFE preload, remaining formats and source closure. Verify actual MLX backing, lazy-reader retirement and shared RAM accounting.
- Replace remaining capability probes and legacy source lists when their consumers migrate. Preserve optional-component validation, media/state behavior and support queries.

## P2.3 — Schema and launcher follow-up

- Verify existing-database upgrade, transaction failure and persistence handling after the schema migration.
- Move `tyloinfer/server/state/schema.sql` to `csrc/server/storage/` with the frozen Python reader. Keep one SQL input and update embedding and packaging together.
- Verify Windows packaging and macOS prepare-only, dylib and DMG workflows. Broad launcher rollout remains deferred.

## P3 — File consolidation

| Current location | Final owner / action |
| --- | --- |
| `csrc/components/tokenizer/` and maintained text preparation | `csrc/engine/requests/text/` |
| `csrc/components/video/` | `csrc/engine/requests/media/` |
| `csrc/components/error.h` | `csrc/engine/error.h` |
| `csrc/components/protocol/` | `csrc/transport/protocol/` |
| Mixed files in `csrc/engine/weights/encoding/` | Pure representation and codecs to `tensor/encoding/`. Stores and preparation stay in weights |
| `csrc/engine/weights/moe/residency/expert_cache_internal.h` | Private declarations with `expert_cache.h` and implementation in `.cpp` |
| Same directory, `cache_types_internal.h` and `cached_source_internal.h` | Move pool, preparation and residency declarations to their owners. Backend consumers use `moe/prepare/projection.h` |
| Remaining bundled dependencies | `csrc/dependencies/<name>` with version and licence |

Move callers, includes, builds and package entries together. Keep `weights/cache/resources.h` private. Resolve imported and frozen consumers under P0.2.

## Deferred scope

HIP and Nemotron-H migration, Qwen4 Metal DFlash2, MTP TTT/LoRA, QSA SSD offload, old paged-cache extensions, and Qwen35 Metal batching.

Delete completed tasks and past build evidence from this plan. Keep outstanding acceptance under its existing task ID.
