# CUDA KV memory policies

The CUDA Flash-Next/QSA worker exposes `kv_memory` in model load requests and
per-model context settings. Metal keeps its existing allocation controls.

## Explicit QSA cache modes

Managed CUDA Flash-Next defaults to fixed Affine INT8 with SSD KV for supported
four-token QSA blocks. RAM preallocation stays enabled. Saved model settings,
explicit cache mode or placement, and precision environment overrides take
precedence. The independent `kv_quantization` setting still selects the
ordinary packed-KV implementation;
explicit VRAM, mapped-RAM and dynamic raw allocation retain their old paths.
Other attention backends are not changed by the QSA policy.

```json
{"kv_memory": {"qsa_mode": "affine8", "placement": "auto_ssd"}}
```

- `auto` reuses the RAM precision policy: nominal 5% FP16, 15% Affine INT8 and
  80% TQ4, with complete 2048-token conversion batches and bounded batch tails.
  Actual protected fractions depend on the RAM budget and resident population.
- `fp16`, `affine8` and `tq4` retain fixed encoded precision. RAM pressure does
  not silently demote them. With SSD enabled, their original images can spill
  and be read back without refitting. Partial append buffers remain temporary.
- `legacy` retains the ordinary raw-KV implementation and its admission checks.

To opt into adaptive precision without SSD, specify `qsa_mode: "auto"` and
`placement: "auto"` in the load request. `placement: "auto_ssd"` enables the
cold-block SSD tier. Admission then reserves resident index/descriptor metadata
and bounded staging/codec buffers, not the entire historical KV population in
RAM. Disk capacity is checked independently against the selected encoded
format. Without SSD, the budget includes actual packed images, scales,
alignment, index archives, metadata and conversion tails. GGUF experts receive
the remaining RAM budget after this reservation. Cache payload grows with live
tokens rather than preallocating a full-context FP16 image.

Available physical RAM does not guarantee that CUDA can lock more pages. If a
QSA image-pool allocation is rejected, the model thread may release idle GGUF
expert pinned arenas and retry that allocation once. The expert pin ceiling is
reduced by the returned quota so optional cache growth cannot immediately take
it back. Background codecs defer this recovery to the model thread; live graph
and transfer leases remain protected. This does not enable SSD KV or change the
selected precision policy. `native_gguf_pinned_reclaim` reports physical arena
bytes returned, not just deleted cache payload.

Before a managed QSA chunk advances state, its immediate canonical append,
codec batches and required row packets are priced together with an additional
transaction-sized safety margin. Optional GGUF pinned arenas proactively return
quota to satisfy that headroom. The conservative budget uses a previously proven
committed pinned population; it is not advertised as the driver's exact limit.
An unsatisfied known budget is rejected before that chunk advances, rather than
silently reducing precision, context or request capacity. Unknown limits are not
replaced with a device-specific constant.

Without an explicit user RAM cap, managed GGUF QSA can reclaim optional expert
RAM as independent request stores grow. The initial model-load KV reservation
is not frozen into a shared multi-request RAM ceiling. Explicit user caps and
ordinary raw-KV admission keep their existing semantics.

Flash-Next batch prefill acquires required GGUF PLE row-stage leases before
swapping in request state. Plans include the encoded pinned packet, pageable
prefetch packet, GPU packet and decoded output; they use the table's actual
staged representation, including row corrections. Full and tail stages in
layer-major prefill are prepared before reset or forward-state advancement.
Cancellation and failures retain each stage until its finite CPU/GPU consumers
have retired. This is workspace admission, not a global token-slot allocator.

A request-local pinned-allocation scope also covers tensor transfers and mapped
plans that do not use the QSA image pool. On allocation rejection it can return
optional GGUF pinned quota and retry only that uncommitted allocation once.
Optional expert arenas, locked resident-view creation, background threads and
captured/host-served graphs cannot invoke that recovery. Native allocation
failures remain errors when no safe reclaim is possible; numerical chunks are
never replayed as a memory-allocation retry.

`continuous_batching_qsa_*` telemetry reports the independent requests' actual
host payload, deduplicated image-pool allocation and precision-tier tokens. The
base model's `qsa_adaptive_*` counters alone do not include those request stores.

Configuration precedence is explicit API/per-model `qsa_mode`, then
`TYLOINFER_QSA_CACHE_MODE`, then legacy `TYLOINFER_QSA_RAM_KV_BITS`, then the
managed default described above. The native environment accepts `auto`, `fp16`, `affine8`,
`tq4` and `legacy`. Legacy numeric values 16/8/4 migrate to fixed modes, not
Auto; zero selects legacy. Existing configurations that used 16 to request
adaptation must now select `qsa_mode: "auto"` explicitly. The managed service
migrates `TYLOINFER_QSA_ADAPTIVE_TQ=1` to Auto and forwards the resolved mode.
Standalone native calls without any mode keep the ordinary legacy path; the old
standalone GPU-adaptive experimental flag retains its separate implementation.
Prefix compatibility keys include the resolved mode, so old or differently
configured precision caches are not silently reused. Native startup logs report
`native_qsa_cache_mode`; runtime metrics expose `qsa_cache_auto_enabled`,
`qsa_cache_fixed_enabled` and actual RAM/SSD tier occupancy.

## Ordinary raw-KV placement

The following policies apply to `qsa_mode: "legacy"` and ordinary raw KV.
FP16 remains their default precision.

| Placement | Raw K/V storage |
| --- | --- |
| `vram` | Preallocated GPU storage; reject a context that cannot fit. |
| `ram` | Preallocated mapped host memory, accessible to CUDA kernels. |
| `auto` (default) | Budgeted VRAM hot KV and preallocated RAM cold KV; reject insufficient combined capacity. SSD KV is disabled. |
| `auto_ssd` | Budgeted VRAM hot KV, RAM cold KV, then SSD spill. RAM may be preallocated or allocated as KV grows; SSD storage is created on demand. |

The service's automatic VRAM budget includes the entire target indexer cache
plus a smoothly interpolated fraction of raw KV: 100% at zero context, 60% at
128K, 30% at 256K and 10% at 1M. An explicit `gpu_budget_bytes` must cover the
indexer cache. Unused KV capacity can hold extra experts, which are evicted as
KV grows. VRAM and RAM tiers exchange cold rows without retaining a full
duplicate in RAM. `ram_preallocated` defaults to true; disabling it requires
`auto_ssd`. The service checks disk capacity before admitting SSD-backed KV.

Studio's model loading policy includes a collapsible Context management
component. An empty default context means 32768 tokens, capped by the model's
supported limit. Automatic settings use the managed default for CUDA Flash-Next
QSA. Manual settings distinguish sparse-attention RAM/SSD
controls from other models' VRAM preallocation. Defaults are saved per server
in the current browser and resolved against model/backend capabilities when
loading. Explicit API KV settings take precedence over these defaults.

Indexer rows, pooled index keys, recurrent state and compute scratch remain on
the GPU. Capacity admission accounts for dense weights, the minimum expert
working set and compute reserve. Expert residency is reduced by the concrete KV
reservation so prefill transfer buffers retain headroom. Windows additionally
retains shared-memory budget for mapped plans and pinned transfer buffers.

`allocation: "dynamic"` is experimental. The virtual-memory helper is adapted
from local Strata under its MIT license. It reserves stable addresses for the
configured context, initially maps up to 16K tokens of raw KV, and grows in 8K
steps rounded to the driver's allocation granularity. Existing KV is not copied
when pages are added. Indexer storage is currently preallocated.

GPU kernels resolve RAM cache misses while decode/MTP graphs retain stable
addresses. Selected rows remain resident across attention calls; queries larger
than the hot window can read cold RAM rows without an unbounded wait. SSD-backed
attention uses stable GPU staging and performs required host I/O outside graph
replay. Indexer and recurrent state remain on the GPU. Prefix restore and
rewritten speculative positions invalidate stale resident rows. KV precision
is unchanged (FP16 unless explicitly configured).

Native CLI callers that do not supply the service's tier-budget settings retain
the legacy 64K threshold and 32K hot-window policy, configured through
`TYLOINFER_KV_AUTO_STREAM_MIN_CONTEXT` and `TYLOINFER_KV_RESIDENT_CELLS`. Unified-memory
devices do not use that legacy discrete-memory policy.

Long-prompt prefill also budgets sequence-sized residual and FFN injection arrays
before choosing a compute tile. When borrowing expert storage, the two transfer
ring slots use separate allocations so their combined size need not fit one
contiguous region. If the second slot cannot fit, prefill uses one slot.

Dynamic allocation currently requires GPU-resident raw KV (`vram` or `auto`).
Growth headroom is retained in the expert budget; direct physical-page exchange
with expert arenas is not implemented. Additional concurrent sessions are
admitted against available GPU memory; explicit RAM placement uses mapped RAM
for each session. Unsupported model backends do not advertise these controls.

The default CUDA allocation policy is preallocation. Capacity failures without
SSD use `kv_memory_insufficient`; the Studio message offers enabling SSD KV or
reducing context length. Settings are persisted per model after a successful
reload, and a failed reload restores the previous settings.

Local Strata reference `fb58e0dbc8399662c0e47c76578c6e878b14f6cf` has a
4096-token native CLI default. The README's recommended interactive setup instead
selects context from the model and memory budget (typically 128K for Flash-Next
on a 24 GiB GPU). From 64K, when RAM permits, that setup keeps raw KV in RAM
with `--kv-resident 32768`; it normally uses INT8 KV above 8K. MFQ adopts the
long-context placement without changing the user's context or KV precision.
Strata's expert-cache budget accounts for model, KV and workspace costs.

Native regression coverage includes four placements, RAM/SSD spill ordering,
generation parity, reset, prefix restore, MTP, continuous batching and stable
addresses/data across virtual-memory growth.

The Flash-Next service retains compatible CUDA decode graphs across request
reset and prefix restoration when KV addresses are stable. GDN/PLE state is
cleared or restored in place, while QSA pooled keys are refreshed after prefill
and restoration. Expert-budget changes and switching to independent batch
execution invalidate graphs before their transfer buffers are released.

Loading prepares ordinary single-token graphs, target verification widths 1–4
and predictor widths 1–4. Short adaptive catch-up uses the same four-row windows.
The largest predictor
is prepared first so smaller shapes share its transfer buffer. Diverse synthetic
token IDs and two additional replay sweeps exercise the captured widths during
loading. All synthetic model and predictor state is cleared afterwards. This work adds
to model loading rather than interrupting the first generated response.
Long-prompt prefill bounds its compute tile by remaining VRAM while these graphs
are retained; `runtime_prefill_chunk_budget` reports the requested and selected
tile sizes. The logical prompt chunk and configured context capacity do not
change. New, unprepared graph geometries can still incur a first-use capture.
MTP continuations of up to 128 new tokens reuse the existing target windows;
short predictor priming uses windows of up to four tokens. Predictor graph
execution retires prefill expert images before replay. First replay costs after
each request's prefill are excluded from adaptive draft-depth timing.

CUDA Flash-Next also runs up to 64 synthetic output tokens through the complete
MTP generation loop after graph capture, using temperature 0.7, top-p 0.8 and
top-k 20. This exercises stochastic sampling, draft-history
replay and commit/rollback paths before accepting requests. It clears synthetic
KV, generation metrics and adaptive-depth policy afterwards. The extra loading
work is bounded by the model context; `TYLOINFER_QWEN4_WARMUP_MTP=0` disables it.

Whole-layer prefill asynchronously records each target layer's last 32 routed
tokens into a reusable GPU buffer. One combined D2H copy retrieves the samples
at the decode boundary, avoiding tiny per-layer transfers that compete with
prefill expert streaming. Before long MTP predictor priming,
after the target workspace loan has returned, the residency manager prepares
experts using this real prompt tail. Earlier prompt chunks replace their samples
rather than accumulating an entire long prompt's counts. The tail has a four-token
half-life (about six token-equivalents), so a long prompt does not dominate
subsequent decode heat. Usual transactional publication,
rollback and memory limits still apply. This moves an initial exchange into the
reported prefill time. `TYLOINFER_MOE_PREFILL_RESIDENCY=0` disables the observation, and
`runtime_moe_prefill_residency` reports its exchanged projections and elapsed time.

Adaptive MTP calibration revisits speculative widths after the ordinary-decode
baseline. CUDA Flash-Next starts with one draft, three baseline cycles and a
second shallow replay before trying wider verification, including re-entry after
ordinary decoding. Other backends retain
their existing initial order. A fresh CUDA policy repeats the width sweep once
more after initial residency adaptation; inherited continuation policies keep
the shorter calibration. Several cold transfer/cache cycles can survive graph warmup; a single
slow first sample must not suppress a width until the multi-second exploration
interval. Calibration uses the fastest valid sample for each width, followed by
the existing steady-state estimator. Maintenance cycles and widths shortened by
the output/context limit do not consume calibration or probe samples. Acceptance
measurements still include those cycles, and consistently costly speculation
can still select ordinary decoding. After the post-baseline shallow replay,
CUDA defers wider warmup entries whose estimated throughput cannot beat the
baseline, or whose cold projected cost exceeds the startup latency budget.
Deferred widths remain eligible for subsequent exploration.
CUDA parks speculation after six valid consecutive losing cycles following
calibration, avoiding another costly narrow probe in a short response. Other
backends retain the sixteen-cycle exit threshold. Maintenance remains excluded.
Three final scalar samples refresh the baseline after residency adaptation, so
width selection does not compare warm speculation with a cold ordinary step.

A fresh CUDA policy initially caps verification at one draft token. During the
first 48 emitted tokens, widening requires the observed cycle time, scaled by
the next verification width, to fit a 60 ms budget. This is a startup estimate,
not a hard inter-token latency guarantee: actual expert routes and transfers
can change between windows. Inherited continuation policies keep throughput
selection without this cold-start cap. Maintenance samples cannot authorize
widening, and the bounded startup period prevents permanent under-calibration.

During ordinary-decode calibration, predictor history is replayed every four
tokens. Parked speculation uses sixteen-row batches, split into four-row tiles
to amortize expert-exchange boundaries without accumulating a long replay at
request completion. These captured graphs omit final mixing, the
vocabulary projection and token selection. These history-only widths 1–4 are
also captured at load. Only the final tile of a history merge computes the
draft output head. Re-entry combines at most 128 pending history
rows with the current row, avoiding a separate one-row predictor launch while
preserving token positions and committed KV history.
`TYLOINFER_MTP_HISTORY_EARLY_FLUSH=0` restores the previous 512-row flush threshold.

CUDA verification graphs preserve GDN recurrent/convolution state and PLE
convolution/ngram context at every speculative prefix. Partial rejection selects
the accepted checkpoint and truncates QSA KV. Accepted draft tokens do not run
through the target model a second time. Eager execution keeps its existing
rollback/replay behavior. Additional recurrent checkpoints are included in the
runtime's cache memory accounting; packed expert weights retain their layout.

CUDA Flash-Next completes its one-token target bootstrap before publishing the
initial two-token batch. It preserves the prompt-only prefix snapshot before
bootstrap and previews the first token's EOS/stop behavior without changing the
real output or sampling state. Penalties and constraints use prospective copies.
This moves necessary startup work into actual TTFT and avoids a cold bootstrap
stall immediately after the first text; it does not remove that compute cost.
`TYLOINFER_MTP_EARLY_BOOTSTRAP=0` restores separate first-token publication. Other
backends retain their publication order.

Fresh single-device, layer-major Flash-Next prefill can borrow expert-cache tails
while retaining decode graphs. This follows local Strata's `plan_lend` / `lend`
design: count compute workspace and the layer transfer ring together, then
temporarily stream the experts whose GPU storage is borrowed. MFQ preserves the
original allocations and slot books. It saves one compressed RAM copy of the
borrowed tails and builds temporary cold expert maps; two reusable host staging
buffers upload their weights with each layer. RAM prefix storage yields capacity
before these backups are allocated.

Temporary CUDA allocations reuse free slices of the borrowed regions on the
prefill compute stream. Releasing a tensor returns its slice for later phases,
so sequential scratch does not accumulate. Persistent state and graph allocations
bypass the loan. On completion, the final residual is detached, transfer readers
are synchronized, all borrowed tensors are released, and original weights are
restored at the same addresses. Cancellation and exceptions use the same cleanup.
Decode rejects an outstanding loan if cleanup cannot complete.

`runtime_prefill_loan` reports borrowing and return. `TYLOINFER_PREFILL_BORROW=0`
disables this path; `force` is a diagnostic override used by small native tests.
The current compute estimate remains conservative rather than Strata's exact
`Prefill::bytes_needed` formula. Free memory, transfer growth, fragmented expert
fields and the minimum resident expert set can still require a smaller compute
tile. Cached long continuations and other model/prefill paths retain their
existing allocation policy. Ordinary expert-cache resizing remains a separate
operation that invalidates graphs and the contiguous layer-prefill layout.

MTP teacher-forced prompt priming also borrows workspace when its predictor
expert image and compute tiles exceed available memory. A single predictor layer
uses one transfer slot; multiple layers may overlap two slots.
Gate/Up and Down transfer images are allocated separately during a single-layer
loan so they can use separate expert regions rather than requiring one oversized
contiguous allocation. Persistent
predictor positions bypass the loan. A short priming tail returns the loan before
using decode graphs, and cancellation releases it with the speculative session.
Borrowing and restoration time is included in prefill timing.

Speculative generation publishes each committed token batch before computing
the next draft chain. Consumer/network delay is excluded from the next MTP cost
sample. Cancellation between batches updates committed predictor history without
running another speculative chain. The CUDA worker reports early cycles above
55 ms as `runtime_mtp_slow_cycle`, separating draft, verification and commit time
and reporting target expert transfer bytes. `TYLOINFER_TRACE_MOE_SERVE` additionally
reports decode-window reset, graph launch, service and final wait times.

Expert residency exchanges retain their full adaptation rate. Transactions may
mix incoming weights already held in the GPU transfer cache with uncached RAM
weights: only uncached incoming fields need to be preserved before victims are
copied directly into their final RAM locations. When the existing GPU transfer
wire has sufficient payload space, it holds those incoming fields instead,
eliminating the CPU backup copy without allocating more VRAM. Only uncached
fields occupy this compact payload; cached fields leave no unused gaps. Its control header
is preserved; the expert fence retires previous consumers and the next decode
waits for the transaction before reusing it. Insufficient wire space uses pinned
RAM backups. Cached incoming fields refill
from GPU storage. Publication and rollback preserve both kinds of original
weights. GPU-wire exchanges pipeline H2D preservation and D2H victim backups
on separate streams in groups of up to 8 MiB. A group fence protects each RAM
source until its original weights have been preserved; cached incoming fields
can begin their victim backups immediately. Refill and publication wait for all
backups, and rollback drains both streams before restoring bytes.
Large victim fields retain direct D2H DMA. Fields up to 64 KiB inside the
registered RAM arena share one mapped-write kernel after the preservation
fences. This removes small per-field DMA submissions while large transfers
can still overlap the graph tail. Fields outside that arena or beyond descriptor
capacity keep their individual DMA path. `TYLOINFER_MOE_RESIDENCY_SMALL_BACKUP_BYTES=0`
disables this batching; its default limit is 65536 bytes. The exchange-wait log
includes total and batched backup bytes/field counts.
`TYLOINFER_MOE_RESIDENCY_DUPLEX=0` retains the serial transfer path. One reusable pinned
image also supplies expert-map uploads. One mapped-copy kernel publishes all
maps after the previous window retires, avoiding a separate DMA submission for
each small map. Both its descriptors and payload remain alive through completion
and rollback.

Prefix snapshot admission counts occupied KV rows and recurrent state, excluding
unused KV capacity and speculative graph checkpoints. An unsuccessful initial
MTP snapshot is retried at request completion, rather than after every token.
MTP emits a prompt-state boundary after predictor priming and before appending
the first generated token. Initial prefix snapshots run at this boundary, so
their latency is included in TTFT and excluded from the decode interval.
This corrects timing attribution; it does not by itself reduce inference time.
Parser-held trailing text is flushed before the final prefix snapshot. Cache
storage and cleanup still finish before the terminal event; they no longer hold
back the last visible characters of an otherwise completed response.

CUDA expert transfer arenas pack per-projection fields while retaining stable
graph addresses. Preload copies pinned host weights directly into final expert
slots and releases temporary staging after transfer completion. Large MTP
prompt histories can remain in RAM and upload bounded predictor tiles; the
predictor budgets its compute workspace before borrowing expert capacity.

Grid vision weights remain in RAM by default and move to VRAM for each encoder
stage. Its workspace estimate includes decoded projection weights and MLP
activations. Fused tanh GELU avoids retaining a chain of full-sized temporaries.
Windows image requests support the authenticated binary tensor-file transport
used by the Python server, with file ownership, shape and token validation.
