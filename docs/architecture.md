# TyloInfer Architecture

This document defines stable responsibilities and contracts. [plan.md](plan.md) lists exact change locations, remaining
work and acceptance; [refactor.md](refactor.md) gives implementation conventions. Source history belongs in Git.
Target ownership does not imply that every source path has migrated or passed device verification.

## Overall Structure

There is one inference library, used by native Engine CLI (`tyloi-engine`) and Server (`tyloinfer-server`). Server calls
Engine in process; Studio is a service client. Diagnostics, project Torch C++ bindings and a LibTorch reference engine
are not products. Independent offline reference generation remains separate. Python consumer migration and HIP work
retain the explicit scope limits in the plan.

| Owner | Responsibility | Does not own |
| --- | --- | --- |
| `server/` | Authentication, catalog/aliases, Engine instances, jobs, uploaded assets, persistent conversations | Model computation, device scheduling, inference cache files |
| `transport/` | External request/result/error conversion and HTTP/stdio listeners | Product database, family selection, model execution |
| `apps/`, `commands/` | Process startup, arguments, signals, library assembly calls | Runtime implementation or model allowlists |
| `engine/requests/` | Typed inputs/results, text/media preparation, output interpretation | Scheduling or model weights |
| `engine/scheduler/` | Admission, queueing, cancellation, backpressure, batch membership, terminal publication | Model math or a second request record |
| `engine/generation/` | Prefill/decode, sampling, MTP and decision progression | Concrete family/device internals |
| `engine/state/` | KV/recurrent payload, pages, checkpoints, prefix/session storage and restore | CUDA allocation or weight loading |
| `engine/weights/` | Read/prepare weights, placement, residency, transfer readiness and leases | Numerical forward, request progression, Graph execution policy |
| `engine/models/` | Family assembly, computation, state semantics and actual capabilities | Executor, active requests or scratch ownership |
| `engine/backends/` | Numerical implementation, execution resources, device primitives | Product policy or a duplicate model runtime |

The five runtime modules are requests, scheduler, generation, state and weights. Tensor/ops/memory/IO, models and
backends support them; directory count is not module count.

```text
Server or CLI/transport → Engine → Scheduler → generation → CausalModel(input, State)
                             │                                  ↓
                             └─ Executor scope              public ops
                                                                ↓
                                                        backend ops → kernels/native
```

Engine depends on neither Server nor transport. Generic generation/state depend on public model, numeric and resource
contracts, not CUDA headers. Models define composition; backends choose implementations by numerical requirements.

Across `csrc`, expose an operation at the module that owns its rules. When several callers must gather the same flags,
repeat an ordered sequence or reproduce the same wire fields, put that shared decision or sequence behind one interface.
Keep small implementation functions local to their owner. Keep separate entry points when callers rely on different
math, lifetime, or protocol behavior.

## Final Target Layout

Paths below are ownership destinations. Keep existing paths until a necessary change moves their consumers together.

```text
csrc/
  engine/
    engine.*, load.*, config.h, executor.h, error.h
    requests/{text,media}/
    scheduler/  generation/  state/  weights/
    models/{source,common,<family>}/
    tensor/encoding/  ops/  memory/  io/
    backends/<device>/{executor,ops,kernels,native}/
  server/{api,models,sessions,media,storage,...}/
  transport/protocol/
  apps/  commands/  dependencies/  tests/
```

`components/tokenizer` and maintained text code belong to requests/text; video codecs/frame sampling to requests/media;
external protocol conversion to transport/protocol; resource errors to engine/error.h. Bundled JSON/HTTP/GGML and other
imports belong to dependencies with licences/version records. Project adapters stay with their consumers.

## Module Internals and File Responsibilities

Paths in this section are relative to `csrc/engine/` unless stated otherwise. A file/module need not introduce a class.

| Files | Contract |
| --- | --- |
| `requests/request.h`, `result.h`, `processor.*`, `output.*` | Submission and result contracts; validate/model-prepare inputs and assemble output |
| `scheduler/request.*`, `scheduler.*`, `batch.*` | One active request record; coordinate execution and publish exactly one terminal result |
| `generation/generation.cpp`, `session.h`, `prefill/`, `decode.*`, `sampling.*`, `mtp/`, `decision.cpp` | Shared progression and cleanup; use model capabilities and explicit State, not family templates or callback workflows |
| `models/source/` | ModelSource, canonical names/config/ModelGraph, format interpretation; no device execution |
| `models/load.*`, `prepare.*`, `<family>/load.*` | Unified family selection and optional-component validation, then family-owned weight assembly |
| `models/<family>/{model,forward,state,session,batch,mtp}.*` | Concrete computation, positions, snapshot layout, predictor and independent request State |
| `models/common/ffn.*`, `mlp.h`, `moe.*` | Model-held weights and residual/norm/router/shared/latent composition |
| `ops/ops.h`, `ops/ffn.h` | Explicit mathematical, dtype/layout/aliasing and mutation contracts; no Executor parameter |
| `weights/source/`, `io/` | Container/range readers versus shared physical IO; retained readers keep sources alive |
| `weights/loader.*`, `placement.h`, `fields.h` | Tensor construction, placement/sharding, ordered groups and model-owned weight enumeration |
| `weights/cache/`, `moe/{prepare,host,residency}/`, `tq/` | Admission, residency, prepared expert data/leases, route learning and cache accounting |
| `weights/rows/` | Bounded encoded-row access, retained packets and stage lifetime |
| `tensor/encoding/` | Encoded representations, logical geometry and pure codecs; no mutable cache or execution manager |
| `state/kv/`, `state/session/`, `models/session/` | Generic storage/prefix lifecycle versus family snapshot composition/validation |
| `memory/{budget,kv,host}.*` | Cross-owner admission, placement and one shared host-memory measurement contract |
| `backends/<device>/executor/` | Scope binding, workspaces, completion, library handles, Graph and device KV/QSA resources |
| `backends/<device>/{ops,kernels,native}/` | Numerical implementations; raw kernels; allocation/streams/events/copies |

Names identify responsibilities. `internal`, `detail`, `details` and `impl` are not business groupings or access control.
Single-file helpers stay in the source file; genuinely shared private headers remain private to their owner. An in-class
`Impl` and representational `TensorImpl` are valid. Do not add a forwarding class merely to rename a file.

## Engine Instance Target Design

Engine exclusively owns `CausalModel` and the primary Executor, plus Scheduler and instance resource owners. Factory
results use `unique_ptr<CausalModel>`; synchronous computation borrows references. Suspended work retires before model
replacement. Shared weights do not require another public model-handle layer.

| Object | Lifetime |
| --- | --- |
| CausalModel / compute units | Read-only config and Tensor weights; no live request State, Executor or temporary scratch |
| Request State / MtpState | KV, recurrent/conv, positions, checkpoints, predictor history and request-specific Graph state |
| Executor | Device resources and submission/completion; forks borrow root resources weakly |
| Budget / weight Cache / Sessions | Their own admission, backing and persistence responsibilities; Engine coordinates teardown |
| Output | Retains request/instance dependencies through completion/cancellation and final cleanup |

`engine/load.*` assembles typed options, model factory and backend resources. Reload is library-owned, shared by CLI
and Server: serialize it, retire old work, release the old model before replacement, and publish completion after scopes
exit. Failed reload preserves the documented unloaded-on-failure behaviour. Shutdown stops admission, cancels/drains
requests, joins workers and releases owners in dependency order.

### Server and Engine

`EngineInstances` manages loaded `EngineInstance` objects. Product sessions are durable conversations; engine sessions
are inference snapshots. Server manages artifacts/jobs/media while Engine manages execution; stable HTTP/DB identifiers
are not renamed to match implementation terminology.

Catalog/support queries and actual assembly use ModelSource/ModelGraph and the same library family selector. Pre-load
support does not promise allocation success. Server calls Engine management operations, never manipulates model,
Executor or cache internals. Unknown artifacts may retain inspection without advertised inference support.

Prefix storage owns parsing, format identity, leases, index publication and reclamation. Instance management holds
admission while storage drains short IO/queued writes; retained readers return busy. All local Stores receive updated
indexes. Server does not implement busy-check/delete/refresh independently. Unrelated files and identity files survive.

Schema/version/migrations belong to server/storage. While the Python reader is frozen, both consumers use the single
SQL resource at its existing path. Resource movement must preserve transaction and database compatibility.

### Request progression

1. Server resolves authentication, instance and product identifiers; protocol converts messages to typed requests.
2. Engine validates/prepares input, Scheduler admits it, and generation advances the owning request State.
3. Model composition calls public ops within the Executor scope; Budget admits growth before allocation.
4. Generation commits positions/accepted tokens, publishes output and snapshots through shared session handling.
5. Completion, cancellation and failure retire asynchronous users before releasing storage. Scope exit alone does not
   mean device completion; service persistence failure is still a service failure after successful inference.

Decision requests reuse Engine/Scheduler. Models own readout weights and causal/bidirectional/prefix-bidirectional
semantics. Text preparation validates codebook/tokenizer agreement and limits. Conditional label probabilities and
full-vocabulary confidence remain distinct. Session identity distinguishes incompatible attention/readout semantics.

## Unified Tensor Contract

`Tensor` is a copyable/movable owning value over a shared `TensorImpl`. Models/state/public ops use Tensor values and
references, not smart-pointer plumbing. TensorImpl provides representation metadata, storage enumeration and lifetime;
it has no numerical forward virtual methods. Dense and encoded implementations reuse TensorStorage/backing identity.

- Copy shares backing; assignment replaces a handle; view shares backing with distinct metadata; clone copies data.
- An empty handle differs from a valid zero-element Tensor. Optional absence is documented by the receiving operation.
- Read inputs through `const Tensor&`, modify explicit outputs through `Tensor&`, return values. Async consumers retain
  handles/leases; references alone do not extend lifetime. Published weight aliases are read-only by convention.
- Ordinary weights have logical `[out,in]`, experts `[experts,out,in]`; grouped results retain declared order.
  Shape/dtype/nbytes describe logical data. Physical accounting includes scales/codebooks and deduplicates real backing.
- Pointer/stride access is a dense or constrained-view operation, not a promise for every encoding. Device copies retain
  encoding/auxiliaries; cast/reshape/slice state aliasing and materialization rules and reject unsupported cases.
- Separate execution placement from current residency. CPU-backed weights do not imply CPU computation. Models do not
  decode encoded payloads, and Tensor creates no execution scope, stream or Executor.
- Cache/Graph identities use actual storage and allocation generation, not Tensor object addresses or equal shapes.
  No ConstTensor, TensorBase, custom reference counting or global identity registry is required.

## Public Ops and Backend Dispatch

Public ops specify shape, dtype, numerical precision, output aliasing, state mutation and errors. Input device selects
ordinary execution; weight operations also validate encoding, shards and placement. Creation/copy specify devices
explicitly. Unsupported combinations fail explicitly without implicit CPU fallback or whole-model dequantization.

Models compose QKV/norm/RoPE/gating/residual and router/top-k/shared/latent math. Fused operations specify their exact
compute range, conversion, clamp and accumulation order. `ops::Ffn::routed_shared` receives resolved routes and explicit
gate input precision, not a model tag or whole-FFN policy. Precision flags cannot select another model algorithm.
Weight preparation and Graph capability are separate operations; preparation success is not capture support.

Backend implementations borrow execution resources from the current scope and call their kernels/native primitives.
There is one dispatch per public operation, no generic registration framework, Task IR or forwarding model runtime.
Shared contracts need matching semantics on CUDA/Metal, not identical kernels or both devices in one build.

## Executor and Execution Scope

Scope temporarily binds device/stream/resource state on the current thread and restores it on exceptions. It owns no
model/request/State and establishes no process-global Engine. Establish scopes at load and each coroutine resume;
release thread-local scopes before suspension. Owners/leases, not scope bindings, retain asynchronous work.

| Responsibility | Rule |
| --- | --- |
| Workspace | Ops define layout/capacity; Executor reuses and isolates storage through the last consumer |
| Parallel work | Ops define independent numerical branches; execution facilities provide fork/join/events |
| Graph | Backend owns capture/replay resources; family driver defines semantic/state conditions; generation selects when to use it |
| Completion | Track eager submission and completion separately; drain before resource replacement or destruction |
| Timing/profiling | Measure evaluated work; scope restores prior policy and reports/retires in its owning execution facility |
| Layer streaming | Weight preparation packs host backing and publishes device views; existing layer events advance Executor transfer slots |

A model does not enumerate mutable Tensor pointers for a second streaming protocol or call CUDA staging in forward.
Streaming prepares host placement directly; genuinely device-required format preparation has bounded/accounted scratch.
Slot reuse waits for final consumers; recurrent/KV State is never rebound with immutable layer weights.

## Weight Preparation, Residency and Memory

ModelSource interprets model semantics; weights/source reads containers/byte ranges. Source/encoding code has no concrete
family dependency. Loader publishes Tensor weights with roles/shapes/group order supplied by models. Container names
such as GGUF do not define a numeric representation when HF sources can produce the same GGML blocks.

`Cache` owns configured capacity, admission, eviction, route transactions and storage constraints. It apportions capacity
across active formats rather than disabling one because another exists. It does not own prefill/decode progression,
Graph selection or capture-resource preparation. Device implementations may use private copy/event primitives.

`ExpertBatch` and preparation receive selected routes/source identity/placement. `PreparedExperts` returns readable
images, work split, readiness and retained leases, not cache internals. Projection consumes those results and records
last use; numerical kernels neither fetch through a concrete cache nor call cache-owned forward methods. Source IO and
uploads stay outside residency locks. Keep phased Gate/Up/Down readiness, bounded staging, heterogeneous CPU/GPU work,
activation reuse and calibrated overlap. Eager and Graph route learning use executed/accepted rows, not rejected drafts.

| Storage case | Required semantics |
| --- | --- |
| Host/expert caches | Eviction removes lookup ownership; physical accounting lasts through DMA/read leases and delayed unregistration |
| TQ retained groups | Price aligned Gate/Up/Down together; selected image leases survive eviction and replay |
| Rows/RowReader | Bounded source reads, per-reader pending packet and retained source; stage output survives async consumers |
| Metal MFE/NINT | Count actual deduplicated MLX backing, including externally retained views/readers; fixed pools retain minimum sets and MFE uses the remaining ceiling |
| Borrowed workspace/KV regions | Retain backing and stream through last allocation/use; normal return rejects live borrowers or detaches outputs; retirement revokes addresses after readers/uploads |
| Encoded formats | Pure codecs/data in tensor/encoding; preparation builds executable layouts; activation quantization and projection are backend math |

Budget coordinates weight/KV/workspace/host reservations before materialization. Geometry belongs to State/model; device
allocation/VMM/hot-cache backing belongs to executor/native. State performs storage/copy through public contracts.
Independent aligned holes are not one contiguous allocation. Resident recurrent/conv/history updates retain storage.
Model snapshot bytes and copy peak come from State; Budget handles eviction/reservation before copying.

`memory/host.*` is the sole OS-memory/trim facility. Linux uses MemAvailable; Windows adds available commit headroom;
macOS reports residency and footprint separately, combining shared Metal use without double counting. Missing commit
information is not zero. Apply process residency only against an explicit process limit, never subtract it again from
system availability. Sampling fails explicitly; optional/background retention may conservatively use zero. Trim is
best-effort and resampled; a memory observation is not a reservation.

## Session, MTP and Graph

- Each request owns logical KV/recurrent/position and predictor State. Batches reference those same States, share
  packed computation and maintain page/request/allocation identity; they do not copy them into aggregate slot State.
- Restore validates compatibility, reserves growth/copy peak, stages independent target/predictor candidates, drains,
  then publishes together. Failure preserves prior state. Snapshot layout/validation belongs to the family; generic
  Sessions owns matching, eviction and storage. CPU-restored tensors retain intended device identity until staging.
- Accepted/emitted tokens determine rollback, route learning and publication. Predictor history returns to that same
  boundary. Normal terminal snapshots follow output flushing; cancellation/failure does not publish a success snapshot.
- QSA mutable precision/history/RAM/SSD policy lives in state/kv/qsa; pure image/TQ codec and row-energy allocation in
  tensor/encoding/qsa; device backing and host workspace integration in executor/qsa. Codec targets never depend on State.
- Graph keys cover storage, geometry, format and request membership. Restore, growth and rebinding trigger required
  invalidation/synchronization. A returned Graph output handle retains the buffer, not its previous contents; clone
  results that must survive replay. CUDA retains capture/replay ordering; Metal retains lazy evaluation and no-Graph use.

### Model and media contracts

Family assembly validates optional objects and reports actual loaded capabilities through the unified factory. No
parallel Backbone enum, model Plan or application family allowlist is introduced. Common code is shared only where
structure and semantics match. Family-specific CUDA Graph adaptation is allowed; model weights do not own an Executor.

Qwen4 PLE keeps hashing/EOS/checkpoints in the family and uses Rows/RowReader. Qwen35 recurrent and softmax attention
remain distinct numerical contracts. GLM5Next's clamp/router/accumulation semantics and DeepSeek compressed-attention
phase order remain explicit. Layer-major prefill reports growth/phase boundaries before allocation; the model bounds
logical advance, Budget bounds compute workspace. MiniCPM duplex state/control belongs to its family runtime.

### Vision Encoder Design

Concrete encoders own their weights and layer math, accept prepared Tensor pixels/patches and explicit mask/grid, and
return features with documented order/width/CLS rules. They do not parse prompt tokens or change text KV. Family vision
adaptation validates media placeholder count/order and produces text-width embeddings; axial output already aligned
and projected must not be projected twice. Request/coroutine retains inputs/results through suspension and completion.

| Encoder owner | Semantics to retain |
| --- | --- |
| `models/minicpmo45/vision.*` | SigLIP separate Q/K/V, GELU tanh, padding and patch/position rules |
| `models/nemotron_h/radio_vision.*` | RADIO fused QKV, GELU erf, CLS/register tokens and position interpolation |
| `models/common/vision/grid_encoder.*` | Shared grid/position/segmented attention and merge rules |
| `models/deepseek_v41/axial_patch_*` | RMSNorm, axial RoPE, gated MLP and spatial alignment |

No generic vision block/base class or callback table is required. Encoders use public ops and weight preparation, not
Executor/Projector or expert caches. Preserve bias/dtype conversion order, softmax precision, masks and fused-QKV slicing.

## Build and Macro Isolation

`csrc` builds the native implementation. Model sources belong to models/CMakeLists.txt, backend sources to their backend;
apps define entry/link relationships, not model assembly. Default `uv run tyloinfer build` produces Engine; Server is
explicit. Backend SDK dependencies remain private, allowing public-header consumers without CUDA/Metal headers/macros.
Native configuration/link/package needs no Torch, Python development library or extension auto-build.

CMake selects sources; private 0/1 backend-availability flags only serve dispatch/assembly, not public type layouts or
alternate math. Remove obsolete native/Torch selectors (`TYLOINFER_NATIVE_CUDA` and legacy runtime variants). Format
compatibility remains target-private; linked production and test objects use identical implementation configuration.
Device compiler/version and real CUDA error/launch helpers remain backend-private. No consumer-side `#undef` workaround.

### Runtime validation and errors

Remove `TYLOINFER_RUNTIME_CHECK` and its generic failure helper after migrating their consumers. Ordinary runtime
validation uses explicit `if`/`throw` at the responsible owner. Do not introduce a replacement CHECK/require macro,
a generic validation framework, or a backend dependency merely to throw an exception.

Preserve checked conditions, single evaluation, failure-only message construction, diagnostic information and existing
exception types. Changing runtime_error to invalid_argument can alter request error mapping and needs a separate
behaviour decision. Input/shape/device/resource checks remain active in release builds; replacing them with assert is
not equivalent. Existing purpose-specific model validation helpers need not be expanded into a new common API.

The CUDA failure helper also serves dtype dispatch; handle those callers before deleting its definition. CUDA/HIP API
status checks remain distinct because they translate device errors and attach diagnostic context. Exact removal sites,
frozen-test dependencies and deferred HIP handling are in the plan.

## Numerical Correctness Verification

The inference library, native engine/server executables and their distribution packages do not depend on Torch/LibTorch.
The independent offline reference generator has its own dependencies and build environment and is not required to run
native inference or compare fixed baseline results. Executing the same backend kernels is not independent evidence of
correctness.
No second Torch execution branch for producing reference answers is kept in models or ops. Numeric verification depends
on native public interfaces, and runtime modules do not depend back on the verification module.

### Modules and Interfaces

The offline verification module takes "cases plus fixed reference results, native execution entry point" as input and
returns a comparison report. Cases and result files form the cross-process seam; the reference framework and the native
program run the same case separately and never link to each other.
Tests run through the existing interface of the layer under test: ops use public ops, compute blocks use model compute
units, complete models use model/Engine entry points; dedicated kernel tests may test a kernel directly, but that does
not replace upper-level composition tests.
Case reading, result alignment, error judgement and reporting are centralised in the verification module, and callers do
not re-implement tolerance rules. The file contract does not require a generic executor interface, plugin registration
or a second model scheduling system.

```text
Fixed case (inputs, weight identity, compute config, observation points, error criteria)
    ├─ Standalone reference generator → fixed reference results
    └─ Native test → actual results
                          │
Fixed reference results ──┴─→ Comparison report
```

| Location | Responsibility |
| --- | --- |
| `tests/numerics/` | Case and result contract, standalone reference generation scripts, fixed samples, comparison and reporting; the reference framework is imported only in the corresponding generation script |
| `csrc/tests/`, `engine/backends/<device>/tests/` | Reuse the existing test entry points to run native op, state and backend cases; simple ops keep an independent CPU math formula as reference |
| `tyloinfer/evaluation/` | Existing model quality evaluation; Python migration remains excluded |

Native test interfaces cover larger compute-block and model cases and their observation points. Pending consumer
migrations and verification are tracked in [plan.md](plan.md).

The reference generator uses PyTorch ops, upstream model implementations or simple CPU math implementations; each case
states its reference source and support scope, and rewriting every model is not required. An independent numeric
comparison must not produce expected values by calling the native op, kernel or the same decode implementation under
test.
Comparison between native paths is implementation-consistency verification. The existing `reference_cuda.py` calls
TyloInfer native programs and can serve as a regression or quality evaluation baseline, but it is not an independent
numeric reference just because it is named reference.

### Case and Result Contracts

A case fixes the actual input data, not only a random seed; model cases fix token IDs, positions, mask semantics and
initial state, and multimodal computation fixes the preprocessed input. Different implementations use the same input
and do not re-sample or re-tokenize on their own.
A case records op/model configuration, input and weight content identity, dtype, layout and the necessary step order,
and defines the outputs and state observation points aligned by name. Complete weights may be referenced externally by
content-validated value, and small samples are stored with the test.
Reference results also record the generating script and reference model version, framework version, device and the
related compute settings, including accumulation precision, TF32 and approximate algorithm choices; native results
record build version and execution configuration. Both sides must satisfy the numeric semantics of the same case;
identical hardware or optimisation algorithms are not required.
Reuse the existing JSON metadata and contiguous binary Tensor export, stating format version, name, shape, logical
dtype, file encoding and byte count. A low-precision result may be losslessly expanded to F32 for export, but the
original compute dtype must be recorded; a high-precision reference keeps its precision.
Results are exported in the agreed logical layout, and cases involving stride, slicing and offsets still execute in the
original layout. Wait for the related device work to finish before reading host results; observation does not change
the actual compute dtype, state advance or Graph path.
The existing `_logit3_` is an evaluation format that compresses and truncates part of the log-probability and stays in
use for its intended purpose; strict numeric comparison of ops, hidden and logits uses results without additional
lossy encoding, and exact reference values cannot be recovered from that format.

### Criteria and Coverage Levels

Comparison first validates case and asset identity, observation point completeness, shape, dtype and layout convention,
and then compares numbers. Missing output, incomplete files or incompatible configuration are failures, not a numeric
pass.
Integers, indices and results explicitly required to be exact are compared element-wise for equality. Floating point
results are judged by `abs(actual - expected) <= atol + rtol * abs(expected)` with tolerances fixed in advance in the
case; tolerances are set per op, dtype and accumulation/approximation semantics, existing numeric criteria are kept, and
one loose tolerance does not cover all models.
NaN fails by default; infinite values that are mathematically allowed must be stated in the case and checked for
position and sign. The report gives the failing observation points, the count of out-of-tolerance elements, the maximum
absolute error with its position and the actual/reference values; relative error or other aggregate metrics are
supplementary only and do not hide element-wise failure.

| Level | Comparison and coverage |
| --- | --- |
| Op | Independent formula or framework reference; covers the dtype actually supported, non-contiguous input, chunk tails and numeric extremes, with the high-precision reference computed from the same already-rounded input |
| Compute block | Fixed weights and input, comparing the agreed observation points of attention, MLP, residual and state update, locating the first deviation |
| Model | A fixed token sequence fed step by step, comparing hidden, logits and the necessary state; text agreement under free sampling is not a primary numeric criterion |
| Execution consistency | Chunked vs whole-segment prefill, prefill/decode, Graph vs ordinary execution, snapshot restore and similar for the same native computation should be equivalent within the agreed tolerance; such tests supplement the independent reference, they do not replace it |

Quantization verifies two separate things: whether the computation is correct for the same packed weights decoded
independently by the quantization kernel, and the quality loss of the quantized model relative to the original weights.
The latter uses evaluation metrics such as perplexity and log-probability, and a quantization error budget is not used
directly as the implementation tolerance of a kernel.
Performance measurement and correctness reporting are separate; the time cost of an observation export path does not
represent production performance, and a performance gain does not offset a numeric failure.

### Baseline Maintenance and Build Constraints

Daily regression reads fixed reference results, and the implementation under test is never allowed to rewrite expected
values automatically. A reference update must come with its generation source, configuration and reason for the
difference; adding dtype, shape or model capability adds the corresponding sample and criteria.
Fixed samples cannot cover every input, so the standalone reference environment is also used for new cases and random
differential testing; a random failure saves the actual input for reproduction. The comparison module itself is verified
with equal, out-of-tolerance, missing-output and non-finite value samples.
The report distinguishes pass, fail and not executed; a missing device, model or optional reference framework cannot be
recorded as a pass, and comparison against fixed results does not depend on the reference framework. GPU cases run in a
verification environment that has the corresponding device.
Native build and packaging do not look for or link LibTorch; the reference framework and model dependencies belong only
to the separate verification environment. Backend source is selected by CMake targets, and `TYLOINFER_NATIVE_CUDA` is
no longer used to switch native/Torch math paths inside the same production implementation.
Preserve the cases, numeric criteria and observation points, and verify that independent references can be regenerated
and fixed regressions run once their native test interfaces are migrated. Source inspection alone is not acceptance.

## Documentation maintenance

Architecture records a contract once under its owner. Plan records next actions with file/symbol, change and acceptance;
completed source work retains only its pending verification. Do not append commit narratives, competing priority lists
or repeated prohibitions. Update an existing statement when evidence changes; keep supported numerical/lifetime
constraints and explicit scope exclusions. File movement and compilation are not numerical acceptance.
