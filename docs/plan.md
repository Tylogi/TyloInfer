# TyloInfer Refactoring Plan

The target architecture is authoritative in [architecture.md](architecture.md). This plan records the remaining work,
its order and acceptance conditions, not a per-commit history.

**Priority: finish Qwen4 exp on the shared CUDA execution chain → completely isolate LibTorch builds → connect Metal
and CUDA through the same public interfaces.** CUDA restructuring is substantially implemented; do not restart the
Tensor, Executor, Engine or Gemma4/Qwen35 migrations as new prerequisite phases. Remaining CUDA work accompanies the
stage that actually needs it. Necessary build, test and consumer wiring is active work, not indefinitely deferred.

## Current baseline and verification limits

Public Tensor ownership, Executor and shared model/state contracts already exist. Qwen4's loading, ordinary and batch
forward, State, predictor and session codec use public Tensor/Device on the common Engine and generation path. CUDA
Graph and layer-major execution stay in family-private CUDA sources; Metal executes without Graph.
The factory covers the supported CUDA families. Ordinary and MTP generation are concrete shared implementations;
batch generation directly coordinates its scheduled requests without a second sequence adapter. Session matching
uses the actual State/Restore contract without callback fallbacks or adapter probes. Optional weights
and predictor belong to the model; request State owns media, positions, recurrent/KV state and checkpoints. Diagnostics
use actual prefill/decode and graph contracts.
MiniCPM keeps duplex state and control in its own runtime, without a duplex contract on CausalModel.
Optional component declarations are validated by each family and loaded directly. Capabilities report the loaded
objects; no model Plan, adapter enum or parallel capability matrix remains in the shared execution chain.

Prefill, decode, warmup and numerical batch jobs report workspace, layer and Graph events. Runtime coordinates `Loan`,
Graph preparation and allocation admission; the root CUDA Executor owns its KV resources and forked Executors borrow
them weakly. Shared Engine holds no CUDA KV types. Shared session restoration
admits and stages separate target/predictor candidates, drains, then publishes them. DeepSeek V4.1 CED is connected to
model prefill with phase admission and raw target history; Qwen4 retains its existing layer-major span and growth plan.
The affected source lists and native diagnostic callers are connected, including the Qwen3.5 layer reference path.

PLE math and profiling use public ops; row loading uses Rows/RowReader with bounded Metal row reads and device decode.
Metal implementations cover GDN, QSA, residual and routed/shared FFN. TQ, GGUF and Affine expert caches receive the same
SwiGLU limit during preparation and execution; typed KV quantization reaches QSA and sessions.
Predictor and independent batch state growth are admitted before execution; retained lazy storage is submitted with
outputs. The CUDA backend holds capture resources,
FFN tile scheduling, profiling policy and replay backing. Fixed buffers, row stages and state-copy backing report growth
before construction. The ordinary preparation pass observes temporary allocation lifetimes by device, stream and size
class; Budget admits graph-pool growth before materialisation. Warmup expands an incomplete plan through another
admission event before retrying that pass. Coroutine suspension releases execution and registration scopes. This is a
measured capture-pool plan, not a static estimate of the ordinary discovery pass or an independent codec-worker arena.

Qwen paged decoding retains CPU tensors and encoded device identity; admitted restore stages target/predictor data and
returns the uploaded hidden handle. MiniCPM-o evaluation and CLI use the family-owned encoders/TTS and explicit duplex
state; prefix restoration uses Program's admitted restore entry. Their affected source lists are connected.

Production CMake no longer composes a LibTorch engine/CLI or discovers Torch/Python development libraries. Native CUDA
model/backend declarations have one fixed implementation; shared dispatch uses private 0/1 availability definitions.
SDK usage requirements are private. Python CUDA JIT uses independent raw launchers and a checked source/include closure
for checkout, sdist and wheel resources; it does not compile the native Tensor runtime.

Load headers expose only complete load/reload. Production and diagnostic loading use typed configuration, Cache and
the model factory directly; the root Executor creates and finalizes KV while forks borrow it. Cache completion retains
pending State growth, budget charges and the owner drain, without public load-phase functions.
Production CUDA and Metal commands share `tyloi-engine`, Engine load/reload and Transport. Python builds, managed-build
selection and Studio/macOS packaging select this entry; diagnostics remain separate. Metal initialization and Qwen
YaRN/placement policy belong to library loading, not the old application. The shared batch owns independent request
State; its CUDA subclass retains Graph and FFN scheduling policy.

This refactoring round reviewed source references, calls, include/package closure and whitespace without building,
running or editing tests. Runtime and distribution acceptance below remains unverified. Other Metal families retain
their legacy implementations; Nemotron-H migration and Program restructuring are postponed at the user's direction. Qwen4's Metal DFlash2,
MTP TTT/LoRA, QSA SSD offload and old paged-cache extensions are
skipped at the user's direction. Qwen3.5 CUDA batching uses the shared batch contract; its Metal batching remains
unsupported. Shared batch admission still excludes MTP and session/prefix requests; those use ordinary execution.
Legacy diagnostic comparison modes without an implementation explicitly reject invocation.

## 1. Finish Qwen4 exp restructuring

The immediate deliverable is Qwen4 exp loading, executing and retiring through the existing public model/Engine chain.
Keep its currently supported math, formats, placement, MTP, batch, session and media capabilities; do not expand its
support scope or require every other family to migrate first.

The model/state, public-op, Graph-resource and native consumer source paths are connected. Remaining acceptance is
listed below; any source inconsistency exposed by those checks belongs to this stage. Preserve the existing plain MoE
and TQ target-Graph design and ordinary predictor path. Do not restore an execution context, resource bundle, callback
dispatch or adjacent-layer Pipeline. Reuse existing Qwen4 TQ fixtures and independent references, including
`tests/numerics/qwen4_fixture.py` and the relevant cases in `flash_ref.py`; generated native output is not its own oracle.
Gemma4/Qwen35 are regression baselines, not new migration milestones. Test execution and edits remain excluded by the
current user instruction.

| Acceptance area | Required behaviour |
| --- | --- |
| Build/load | Fresh native CUDA build; Qwen4 selected by the unified factory; source/config/YaRN/format/placement validation, supported optional components, partial-load failure cleanup |
| Numeric/state | Fixed-input GDN, QSA, PLE, residual, routed/shared experts, embedding and logits; ordinary/chunked/layer-major prefill and decode; compare state and positions as well as output with existing dtype-specific criteria |
| MTP/batch/session | Accept/reject and partial replay, predictor and target state consistency, request/slot isolation, cancellation and storage reuse; session/prefix restore follows prepare/stage/drain/publish and preserves supported media identity |
| Graph/resources | Direct execution versus capture/replay, warmup isolation, state/scratch storage identity, invalidation, cache/KV budget return, borrowed backing kept alive until the final consumer |
| Lifecycle/consumers | Nested scope and exception restoration, coroutine suspend/resume, load failure, cancel/reload/unload, one terminal result and multi-Engine isolation; affected runtime/diagnostic/evaluation callers use the public path |
| Regression | Existing Gemma4/Qwen35 and shared Tensor/ops/state behaviour affected by the change; no changed math, tolerances, ownership or supported placement conditions |

**Completion:** Qwen4 exp no longer holds an execution context or executor in its model/compute units, live request state
is explicit, and its supported production consumers use the unified factory and execution contracts. Required build,
numeric and lifecycle checks pass separately; missing weights/hardware or skipped cases remain outstanding acceptance.

### Production executable and build-command wiring

Complete the production CLI wiring as part of the current execution-chain work. Rename the `mfq-runtime` executable
and CMake target to `tyloi-engine`, with `tyloi-engine.exe` on Windows; do not rename the inference library merely to satisfy this
executable naming requirement. The default `uv run tyloinfer build` must produce `tyloi-engine` and must not build diagnostics.

- Update `apps/cuda.cmake` and the relevant CMake entry-point conditions so the production executable exists when the
  Python build command requests it. Diagnostics remains an independent explicit target, with no dependency from
  `tyloi-engine` that causes diagnostic compilation. Keep libraries actually needed by the production commands.
- Update `tyloinfer/commands/build.py` to request the `tyloi-engine` target only, removing `mfq-diagnostics` from its default
  target list. Synchronise executable lookup, managed-build records, service launch and affected packaging/distribution
  references with the new name. Do not accept a stale `mfq-runtime` binary as evidence that the new build succeeded.
- Complete the production command consumers of public load/reload, Engine and Transport. Keep model construction and
  device resource management in their owning library modules. Independent diagnostic migration remains tracked, but is
  not a prerequisite for producing the production executable.

**Acceptance:** from a clean build directory, `uv run tyloinfer build` configures, compiles and locates `tyloi-engine` without
requesting or compiling a diagnostic executable. Verify its help/startup, managed executable selection and the supported
load/generate/cancel/reload path. Explicit diagnostic builds have separate acceptance; their omission from the default
build does not count as diagnostic migration or verification.

## 2. Completely isolate LibTorch builds and clean macro propagation

The native/LibTorch source and target split is implemented. Public declarations do not depend on native/legacy
selectors, and production SDK dependencies remain private. Useful PyTorch conversion/JIT tools and offline numerics
remain separate consumers; no same-source C++ reference engine is retained. Format compatibility belongs to private
consuming-target configuration and library-owned format queries.

Remaining acceptance covers fresh host/CUDA builds, exported compile/link requirements, public-header consumers and
both sdist-to-wheel and release-wheel resources. The CUDA JIT raw launchers and their MMA dependencies have only static
interface/include review; their compilation and device behaviour are unverified. No tests were run or edited under the
current instruction. SDK-private test dependencies and old source-layout assertions remain unmodified.

**Completion:** fresh host and native CUDA builds succeed without Torch; a consumer compiles the public headers without
backend macros or SDK include paths. Enabling/building the separate reference does not change native compile commands,
public declarations or link/package dependencies. Production and tests use the same implementation definitions. Fixed
numeric regression runs without Torch, and the independent reference can still regenerate its cases. Check the Metal
target's exported requirements here; device build/runtime acceptance is completed in stage 3.

## 3. Connect Metal and CUDA through the shared public interfaces

Reuse the existing Metal Tensor representation, Executor and basic ops. Connect the missing execution chain rather
than adding another Engine facade, universal Backend object or parallel public tensor/model hierarchy. CUDA remains the
regression baseline while the supported Metal paths move to the same contracts.

Library load/reload and Loader placement now use Device. Qwen3.5 and Qwen4 Model/State/loading use the shared model namespace;
the Metal assembly, public ops and Source-backed TQ/GGUF/Affine caches are registered in an independent backend target.
Sessions share host codecs and submit staged candidates before drain/publication; full-KV disk restore uploads only
after admission. CPU cast/cat and Metal lazy generation use the shared contracts. Only static source/call review has
been performed, so the shared Metal path is not runtime-verified.

Remaining source work includes other model families, Qwen4's retained legacy features and paged-cache compatibility,
shared MTP/session batching, CPU encoded offload and legacy MFQ expert formats. The old model runtime still
contains obsolete norm/coordinator dependencies. CUDA family session restoration uses independent admitted candidates;
DeepSeek-V4.1's codec retains host tensors and their destination devices until staging.
Cache metadata planning uses a checked MLX allocator bound; other SDK versions can reject a larger actual backing.

1. **Complete device-neutral loading and weight placement.** Use public Device values and typed load options instead
   of CUDA-only load types or an integer device plus CPU flag. Keep execution placement distinct from CPU-backed
   weight storage. The same library load/reload contract selects backend resources and the model factory; models and
   weights own their semantics, while allocation, copies and readiness belong to the backend.
2. **Complete public-op wiring.** Move CUDA bodies out of generic dispatch into existing backend modules and connect
   the Metal implementations required by the supported models, including projection, attention/KV, recurrent updates,
   FFN/MoE, rope and sampling. Share numeric and mutation contracts, compile each public entry point once, and check
   device/shape/dtype/aliasing consistently. Keep CPU operations available in CUDA builds. Optional fusion may use the
   unfused same-device computation; unsupported combinations fail explicitly without silent CPU transfer.
3. **Connect shared model/state and generation consumers.** Move the remaining Metal family semantics to the owning
   `models/<family>` implementation and connect CausalModel, State, Mtp, Loader/Cache and the unified factory. Replace
   `MlxEngineModel`, the separate Metal request/generation driver and backend-specific instance assembly as their real
   consumers move to Engine/Scheduler/generation. Preserve model-specific media, batching and session behaviour; a
   wrapper around the old complete runtime does not count as shared wiring.
4. **Preserve execution differences behind Executor.** MLX outputs remain lazy until submission/evaluation requires
   execution; timing measures evaluated work, scope exit does not imply completion, and reads/retirement wait for
   dependencies. CUDA retains stream ordering and capture/replay. Metal may use the shared ordinary execution path
   without Graph support. Shared-memory accounting must not double-count CPU/Metal residency. Do not force raw-pointer
   materialisation or CUDA scheduling assumptions into the public Tensor contract.
5. **Finish build and consumer convergence.** Remove SDK dependencies from the shared core and compile device support
   in its own targets. Update affected runtime, test, diagnostic and distribution consumers and remove migrated old
   Metal paths; do not mirror engine/models/state/weights under device directories. No macro compatibility aliases or
   copied upper runtime are retained to keep an old consumer alive.

**Completion:** native CUDA and native Metal build on their supported platforms without Torch and expose the same
public headers and load/Engine/model/ops contracts. Run shared cases for the model/operation combinations supported on
both devices, with recorded dtype-specific criteria and independent fixtures. Verify load, prefill/decode, supported
MTP/batch/session, cancellation, reload/unload and resource retirement on the shared chain. Exercise Metal lazy-output
timing, no-Graph execution and shared-memory budgeting, plus CUDA capture/replay and asynchronous leases. Neither
identical kernels nor both devices in a single platform build are required; matching signatures alone is insufficient.

### Host memory and state/backend dependency cleanup

Host-memory sampling and allocator trimming now live in `components/host_memory.*` under an independent build
target; the agreed target location is `engine/memory/host.*`, and that path migration remains pending.
Session, TQ/MoE, KV and CUDA consumers include that contract directly; the duplicate session/KV platform
queries are removed. Linux uses `MemAvailable`, Windows retains commit headroom, and macOS supplies physical totals,
available pages, process residency and independent footprint. Metal reuses that sample without adding its shared pool
twice. Sampling failure throws or conservatively yields zero headroom in background/optional-cache consumers.

QSA geometry now lives in `state/kv/layout.*`. `Budget::place_kv` owns GPU/RAM/SSD quota selection, host pressure
reclaim and resampling; initial loading does not require a runtime Budget object. CUDA Executor registrations,
allocation, VMM, hot cache and expert-region retention use their own backend sources. The mixed `state/kv_memory.*`
and its placement-ratio header are removed. Offload and fixed gather admission/allocation use the request device.

KV offload/sequence types now belong to shared state. RAM/SSD policy, page identity and arena leases remain with
state/cache owners; allocation, copies and pointer publication use public ops. CUDA pointer upload and fixed gather
execute in backend ops, preserving pinned staging, the mapped-copy kernel and consumer barriers. Shared state no longer
includes backend-private headers or directly calls device APIs for these operations. CPU arena backing accounts for
host allocation without a second manual charge. Source ownership and production consumers are connected; device
execution and lifetime acceptance below remains outstanding.

**Acceptance:** shared state/session/weights use the common host-memory contract without backend-private includes for
that facility; generic state has no direct device API dependency. Check sampling success/failure, absent and exhausted
commit headroom, session/cache budget decisions and CPU/Metal shared-pool accounting. Reuse affected KV growth,
offload, borrowing, Graph and cancel/reload tests to verify admission before allocation and last-reader retirement.
Only static source/link review is complete. Platform builds, metric failure paths and runtime lifetime checks remain
unverified. Unchanged CUDA tests retain removed host/KV header references and old KV options/configuration names;
test migration and execution remain excluded by the user instruction.

### Align source ownership with the five runtime modules

Status: design agreed; source migration and acceptance remain pending. Apply this ownership cleanup to affected work
without adding a prerequisite phase ahead of Qwen4 exp, LibTorch isolation or Metal/CUDA convergence.

- Remove `components` as a source grouping. Move maintained text processing to `engine/requests/text`, media codecs
  and frame sampling to `engine/requests/media`, host-memory sampling to `engine/memory/host.*`, and the public
  resource-exhaustion exception to `engine/error.h`. Preserve current text interfaces and host-memory semantics;
  host sampling stays independently buildable without Budget or CPU numeric targets.
- Place shared file/range/async readers in `engine/io`. Weight-container and model-replica discovery policy stays in
  `weights/source`; KV/session persistence policy stays in state. Do not make state depend on weight loading just to
  read bytes. Do not add peer text, media or platform runtime modules.
- Put bundled dependency sources in `csrc/dependencies/<name>`, preserving licences, notices and upstream version
  records. Project-maintained adapters stay with their engine or transport owner; adapted text-runtime code stays
  in requests with its attribution.
- Keep `weights` for weight loading, preparation and residency. Separate encoded Tensor representations and pure
  codecs into `tensor/encoding`; move QSA mutable cache policies into `state/kv` and CUDA QSA session backing
  integration into the backend executor. Split mixed files rather than moving the entire encoding directory.
  Weight-specific stores/preparation and expert-storage leases remain with their owners. Codec targets must not
  depend back on the state stores that consume them.
- Update affected includes, CMake ownership, JIT source/include manifests, distribution resources and documentation
  together. Track affected test references under the existing test-migration restriction; do not claim acceptance
  while those references or platform builds remain unchecked. No forwarding directory or duplicate source copy is
  introduced to retain the old layout.

Acceptance requires consistent source/build/package references, preserved text/media behaviour, encoded layouts,
async reader and lease lifetimes, and independent host-memory sampling. Directory moves alone are not implementation
or runtime acceptance. The five runtime modules remain requests, scheduler, generation, state and weights; shared
Tensor/ops/memory/IO facilities, models and backends retain their distinct roles.

## Retained follow-up and scope limits

| Item | Remaining scope |
| --- | --- |
| Other CUDA families, including DeepSeek V4.1 | Continue device-neutral numeric work, State/scratch ownership and remaining capability probing through the unified factory and Engine; preserve supported family semantics |
| Remaining generic consolidation | Remove remaining member probing as its consumers migrate; preserve the previously specified budget, lifetime and failure semantics |
| HIP and its old directories | Device migration and platform acceptance remain deferred; old Engine/generation/MTP consumers still reference removed shared template headers and the old namespace; JIT checkout-relative includes and wheel source resources also need migration |
| Unrelated legacy tools and broad packaging cleanup | Keep outstanding tasks visible, but migrate and verify any consumer actually affected by an active stage in that stage; do not defer its necessary acceptance |

Metal wiring, LibTorch isolation and the verification needed by each active stage are no longer blanket deferred items.
No complete CUDA rewrite or second Gemma4/Qwen35 implementation is scheduled. Unsupported new model features remain
outside this refactor. TyloInfer namespace migration does not rename MFQ file-format identifiers or third-party APIs.

## Plan maintenance

- Delete completed source tasks from the remaining-work list; retain unfinished build, numeric and lifecycle checks
  under their actual status. Stable responsibilities live in architecture.md, and Git records the implementation history.
- Distinguish source completion, build success and runtime acceptance. Static inspection, skipped cases, exit 77 and
  `model_branch=skipped` are not runtime passes; keep original error criteria and record unavailable fixtures/hardware.
- Keep source, callers, includes and required CMake wiring consistent in each change. Do not preserve duplicate
  definitions, forwarding headers or old runtime copies as a substitute for migration.
- The machine has 32 GB of memory; avoid several resident models or concurrent heavy verification runs, and do not
  pin `-j`.
