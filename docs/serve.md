# `python -m serve`

`python -m serve` starts the API and Web UI, with or without an initial model. It
selects the C++ executable and manages workers on private loopback ports.

For copyable CUDA deployment commands and performance settings, see the
[service deployment guide (中文)](deployment.zh-CN.md).

## Quick start

Sync the server and backend dependencies once:

```shell
cd /path/to/TyloInfer
uv sync --project studio --extra metal
```

On a CUDA host, use `uv sync --project studio` instead. Native CUDA inference
does not require PyTorch or LibTorch.

Start an idle server:

```shell
uv run --project studio python -m serve \
  --running-executable build/mfq-runtime
```

Open the model catalog in TyloInfer or the Web UI to load a discovered model.
To load one model before the server becomes available, pass its existing path:

```shell
uv run --project studio python -m serve --model /models/model.gguf \
  --running-executable build/mfq-runtime
```

The server listens on `http://127.0.0.1:8090` by default. `Ctrl-C` stops the
server and its private workers. `--model` accepts an existing GGUF or TQ file;
`.mfq` remains a recognized container suffix. For a sharded model, select the
first shard and keep all sibling shards and required companion files together.
Supported native Hugging Face model directories can also be selected; exact
architecture, artifact and backend coverage is in the
[support matrix](runtime/support.md). A recognized suffix is not a guarantee
that every tensor format or model graph is supported. Legacy inference formats
require the [compatibility build](build.md#native-format-profiles).

Build the native runtime first; see [build the C++ runtime](build.md) for the toolchain
each backend needs. `--running-executable` points at the built worker, and packaged
runtimes need no CMake, `nvcc`, or source tree.

## First API request

After startup loading finishes, or after the UI reports the model ready, run
these requests from another terminal. This example uses a trusted loopback
listener with authentication disabled:

```shell
curl --fail-with-body http://127.0.0.1:8090/health
curl --fail-with-body http://127.0.0.1:8090/v1/models
```

Health should return `"status":"ok"`; it only checks the service, not that a
model can generate. Copy the exact model `id` returned by `/v1/models` and save
this body as `request.json`, replacing `MODEL_ID`:

```json
{
  "model": "MODEL_ID",
  "messages": [{"role": "user", "content": "Say hello in one sentence."}],
  "max_tokens": 128,
  "temperature": 0,
  "enable_thinking": false,
  "enable_mtp": false,
  "stream": false
}
```

```shell
curl --fail-with-body http://127.0.0.1:8090/v1/chat/completions \
  -H 'Content-Type: application/json' --data-binary '@request.json'
```

`--fail-with-body` requires curl 7.76 or newer; with older curl, explicitly
check the HTTP status so an error response is not mistaken for success.
In Windows PowerShell use `curl.exe`, not the `curl` alias, and put the last
command on one line (or use PowerShell backticks, not shell backslashes).
When authentication is enabled, add an `Authorization: Bearer <token>` header
to the `/v1` requests; `/health` remains public. Do not paste real keys into
shared logs or example files.

Success means HTTP `200`, `object: "chat.completion"`, and an assistant message
in `choices[0].message` with nonempty `content`. This smoke test is not a
performance measurement. For API-loaded models, `202 Accepted` is only queue
acceptance: inspect the operation through the jobs API and wait for the
instance to become `ready` before testing. For `401` check the credential;
for model-not-found check the exact ID and loaded instance; for `503` inspect
worker startup/load logs and runtime state.

## How the native runtime is selected

`python -m serve` detects the host backend and selects the runtime in this order:

1. `--running-executable PATH` uses that prebuilt worker and skips every other lookup.
2. Otherwise it reads `<repo>/build/mfq-runtime.json`, a record left by older
   installations, and uses the executable it names when that record matches this
   checkout, operating system, machine architecture and backend, and the binary exists.
3. Otherwise startup stops with an error naming the backend it looked for.

Nothing in this repository compiles the native runtime or writes that record; build it
with CMake and pass `--running-executable`. Packaged distributions pass their bundled
worker the same way.

## Application and private listeners

`--host` and `--port` control only the user-facing FastAPI/Uvicorn listener:

```shell
uv run --project studio python -m serve \
  --model /models/model.gguf \
  --host 0.0.0.0 \
  --port 9000
```

The native worker always binds to `127.0.0.1` on a dynamically reserved port.
TyloInfer passes requests to that private address and disables environment HTTP proxy
handling for internal connections.

`0.0.0.0` is a bind address, not a browser destination. On the same machine,
open `http://127.0.0.1:9000/`; from another machine, use the server's reachable
IP address or hostname. Any non-loopback bind can expose both the API and Web UI
to the network. Non-loopback binds require an API key. Configure it before
starting the listener:

```shell
TYLOINFER_SERVER_API_KEY='replace-with-a-secret' \
  uv run --project studio python -m serve --model /models/model.gguf --host 0.0.0.0
```

`--api-key-env` changes the name of the environment variable read by the
server; it does not accept a secret value on the command line.

`--anthropic-port` opens a second public listener dedicated to the Anthropic
Messages API (`/v1/messages`); it defaults to the saved listener port or
`8091`. Both listeners serve the same models, and the running port/protocol
can be changed later through `PUT /api/v1/runtime/listener`.

When the selected environment variable is unset or empty, API authentication is
disabled. When it is set, its value is the administrator credential. HTTP
requests under `/api/` must send it as a bearer token, and realtime WebSocket
connections must authenticate as well. Static Web UI files and `/health` remain
readable.

```shell
curl -H 'Authorization: Bearer replace-with-a-secret' \
  http://127.0.0.1:8090/api/v1/models
```

To use another variable name:

```shell
TYLOINFER_PRODUCTION_KEY='replace-with-a-secret' \
  uv run --project studio python -m serve --model /models/model.gguf --api-key-env TYLOINFER_PRODUCTION_KEY
```

In PowerShell, set `$env:TYLOINFER_SERVER_API_KEY = 'replace-with-a-secret'` before
running `uv run --project studio python -m serve`. The current browser UI and Studio show an
**API Key** sign-in page when authentication is required. Browser credentials
are held in memory; reloading the page may require signing in again. Studio
can persist credentials through the system credential store. Use a key with
appropriate permissions; a valid inference key need not grant administrative
access. For remote access, use a trusted network or HTTPS through a correctly
configured reverse proxy: API-key authentication alone does not encrypt HTTP.

## Web UI

Web UI is enabled unless you explicitly pass `--no-web-ui`. Open the listener
root, such as `http://127.0.0.1:8090/`. Asset selection follows this order:

1. `--no-web-ui` disables the Web UI.
2. `--web-root` uses the specified prebuilt directory.
3. `TYLOINFER_SERVER_WEB_ROOT` supplies a prebuilt directory when the CLI option is
   absent.
4. Packaged Studio servers use their bundled `web-ui` assets without Node.js/pnpm.
5. Otherwise, TyloInfer looks for `studio/web` in the source checkout.

Studio starts both the API and Web UI on the same listener; it does not open a
browser automatically. The desktop client and browsers share that server.

A prebuilt directory must contain `index.html`. TyloInfer checks explicit and
environment-provided directories before starting the backend or loading a
model.

In automatic source mode, TyloInfer rebuilds the UI when the sources are newer than
`dist/index.html`:

1. `pnpm install --frozen-lockfile`
2. `pnpm run build`

This needs Node.js/pnpm, write access to the checkout, and network access when
pnpm must download packages. Missing assets, unavailable pnpm when a build is
needed, or a failed build stop startup before model loading, with instructions
to supply `--web-root` or explicitly select `--no-web-ui` for an API-only server.

Explicit Web UI modes:

```shell
# Serve an existing Web UI build.
uv run --project studio python -m serve --model /models/model.gguf --web-root /srv/tyloinfer-web

# Start only the API.
uv run --project studio python -m serve --model /models/model.gguf --no-web-ui
```

`--web-root` and `--no-web-ui` are mutually exclusive. When neither is given,
`TYLOINFER_SERVER_WEB_ROOT` can select a prebuilt directory.

## Model discovery and runtime pool

`--model` loads the initial model and adds its parent directory to the catalog.
Without it, the server stays idle until Studio, the Web UI, or the model API
loads a model. Repeat `--model-dir` to add discovery roots:

```shell
uv run --project studio python -m serve \
  --model /models/default.tq \
  --model-dir /models/team-a \
  --model-dir /models/team-b
```

If no `--model-dir` is supplied, `TYLOINFER_SERVER_MODEL_DIRS` is read as an
OS-separated path list. If neither is configured, models are discovered in
`<data-dir>/models`. `--max-runtime-instances` limits managed workers and
`--max-requests-per-runtime` limits concurrent inference requests accepted by
each worker.

Load a catalog model after startup with:

```shell
curl -X POST http://127.0.0.1:8090/api/v1/models/load \
  -H 'Content-Type: application/json' \
  -d '{"model":"model","context_size":32768}'
```

The request returns `202 Accepted` and an `operation_id` for the background
load job.

For file checkpoints, the catalog name omits the suffix (`.gguf`, `.tq`, or
`.mfq`) and shard numbering. For example,
`model-00001-of-00003.tq` belongs to catalog entry `model`. Native HF directory
models use their discovered catalog ID. Always query `/v1/models` rather than
guessing an ID from a path.

Catalog names must be unique across discovery roots. Startup and refresh fail
if two artifacts have the same name. Runtime model IDs and inference routing
use the catalog name; a separate hash tracks the artifact and profile drift.

## Storage and workspace

By default, managed server data is stored in `./.tyloinfer`: `server.sqlite3` holds
persistent state, while `media/`, `models/`, and `components/` hold managed
files. User-selected tool-job paths remain restricted to the current working
directory.

```shell
uv run --project studio python -m serve \
  --model /models/model.gguf \
  --data-dir /var/lib/tyloinfer \
  --work-dir /srv/tyloinfer-work
```

For long-running servers, place the data and work directories on writable,
persistent paths. TyloInfer resolves data, work, and model paths before startup.

## Runtime controls

```shell
uv run --project studio python -m serve \
  --model /models/model.gguf \
  --context-size 32768 \
  --prefill-chunk-size auto \
  --runtime-startup-timeout 1800
```

- `--context-size 0` uses the saved model context policy, otherwise the model's
  declared capacity; it does not impose a 32K limit. Set an explicit target to
  avoid budgeting KV for a larger context than needed.
- `--prefill-chunk-size` defaults to `auto`. CUDA selects a compute tile up to
  `8192` from the remaining input and workspace budget, reserving expert-transfer
  buffers before growing the tile. Ordinary and MTP prefill share this policy.
  Explicit integers from `1` to `8192` set a smaller limit and are always passed
  to the worker. On Metal and HIP, `auto` preserves their native default policy.
- `--max-requests-per-runtime N` configures request capacity per worker. It
  defaults to 8 on CUDA and Metal and 1 on HIP. Batching eligibility depends on
  the backend, model, and request. See [CUDA Flash-Next continuous batching](runtime/cuda-continuous-batching.md)
  and [Metal native serving](#metal-native-serving).
- `--runtime-startup-timeout` covers native worker startup and model loading.

Persistent Session KV caching is configured with `--prefix-cache-dir`,
`--prefix-cache-disk-size`, `--prefix-cache-hot-size`, and
`--prefix-cache-block-tokens`; `--no-prefix-cache` disables it entirely. The
cache survives restarts in the configured directory and its entries can be
inspected and purged through the [runtime API](api/http.md).

Resident-model admission is capped by `--max-runtime-memory`, and
`--moe-gpu-cache-gb` bounds the resident routed-expert cache (`0` keeps
experts fully resident). Omitting it lets the service choose an automatic
budget on supported CUDA and Metal paths. `--no-memory-guard` disables the automatic Metal model-residency
budget. `--max-queued-requests-per-runtime` bounds waiting requests per
worker, and `--runtime-idle-timeout` is the default unload delay for idle
unpinned models; instances may override it through the runtime API.
`--no-access-log` silences the per-request access-log line.

## Speculative decoding (MTP / DFlash2)

Native MTP requires predictor weights compatible with the target graph.
DFlash2 requires a matching draft checkpoint. The managed `python -m serve`
CLI exposes the following explicit companion controls for CUDA.

```shell
# Native MTP companion for a CUDA GGUF target.
uv run --project studio python -m serve --backend cuda --model /models/target.gguf \
  --mtp-native-draft /models/matching-mtp.gguf --no-web-ui

# DFlash2 checkpoint directory (the CLI describes the CUDA Flash-Next path).
uv run --project studio python -m serve --backend cuda --model /models/target.gguf \
  --dflash2-draft /models/matching-dflash2 --no-web-ui
```

Replace the paths with compatible artifacts. The draft switches are mutually
exclusive and require CUDA. The native companion must be an existing file;
the DFlash2 path must be an existing directory.

| Option | Effect | Default |
| --- | --- | --- |
| `--mtp-native-draft PATH` | Native MTP GGUF companion for a CUDA GGUF target. | No explicit override |
| `--dflash2-draft PATH` | DFlash2 checkpoint directory for the supported CUDA path. | No explicit override |
| `--spec-min-p P` / `--mtp-min-probability P` | CUDA greedy draft probability threshold, in `[0,1]`; `0` disables the threshold. | No CLI override |
| `--draft-conf P` / `--dflash2-draft-confidence P` | CUDA greedy DFlash2 cumulative confidence, in `[0,1]`; nonzero overrides `--spec-min-p`. | No CLI override |
| `--mtp-cost-aware` / `--no-mtp-cost-aware` | Select CUDA greedy native MTP depth using measured cost and calibrated probabilities. | No CLI override |

Explicit probability/cost switches also require CUDA. If omitted, the CLI does
not override the corresponding runtime environment policies; record inherited
`TYLOINFER_MTP_SPEC_MIN_P`, `TYLOINFER_DFLASH2_DRAFT_CONF`, and
`TYLOINFER_MTP_SPEC_COST` when comparing runs.

For an OpenAI chat request, `"enable_mtp": false` requests an ordinary decode
baseline; `"enable_mtp": true` permits speculation and
`"mtp_max_draft_tokens": 3` caps draft depth. For `/v1/chat/completions`,
`X-TyloInfer-MTP: on` or `off` supplies `enable_mtp` only when the body omits it;
an explicit body value takes precedence. Other header values are ignored.
Without either override, the chat parser defaults to `true`. This is a
request-local chat control, not a `/v1/systemone` option. Availability still depends on
model weights, runtime support, and scheduling. Inspect
reported `mfq_metrics` such as `mtp_used` and `mtp_acceptance_rate` when present;
do not assume the requested path actually ran. The metrics name is a retained
wire-format identifier, not the public CLI name. See the
[performance guide](benchmarking.md) for measurement steps.

### Metal native serving

The native Metal HTTP server supports continuous batching on Flash-Next,
including ordinary decode, native MTP with a compatible single-layer
predictor without predictor LoRA, and DFlash2 with a matching draft checkpoint.
`--continuous-batching N` sets the maximum concurrent requests for this worker.
Values of `0` and `1` run serially. The managed CLI's explicit
`--dflash2-draft` and `--mtp-native-draft` options are CUDA-only.

To serve Flash-Next with DFlash2 directly, use the binary from the
[Metal build](build.md#quick-start) and pass the draft directory to the worker.

```shell
TYLOINFER_METAL_DFLASH2_TREES=1 build/metal/mfq-decode-metal \
  --transport http --model /models/target.gguf \
  --dflash2-draft /models/matching-dflash2 --continuous-batching 8 \
  --host 127.0.0.1 --port 8080 --ctx-size 32768
```

`TYLOINFER_METAL_DFLASH2_TREES=1` enables tree verification for eligible
greedy requests. The DFlash2 batched path requires text requests with
`enable_mtp` enabled, an output budget above one token, zero presence and
frequency penalties, and a repetition penalty of one. Grammar, JSON schema,
and tool constraints can affect path selection.

Send test requests to the worker's `/runtime/generate` route. Check
`metrics.mtp_used` in the response. Agent, DS Bot, and other OpenAI-compatible
clients connect to the managed API. See [native HTTP transport](runtime/transport.md#private-http).

```shell
curl -fsS http://127.0.0.1:8080/runtime/generate \
  -H 'Content-Type: application/json' \
  --data '{
    "input": {"messages": [{"role": "user", "content": "Explain continuous batching."}]},
    "sampling": {"enable_mtp": true, "temperature": 0, "max_new_tokens": 128},
    "stream": false
  }'
```

## Jev decision-model weight placement

For supported CUDA decision models, `--jev-weights stream` is the default.
Layer weights remain in RAM and use two GPU streaming buffers sized for the
largest layer. Eligible embedding/output tables normally attempt mapped RAM
placement; unavailable mapping or an ineligible layout can fall back to device
tables and require additional VRAM. Explicit mapping overrides may report failure
instead of falling back. Output normalization and any trained readout also consume
resident GPU memory. `--jev-weights vram` places
these weights in VRAM instead, increasing residency and avoiding layer-weight
streaming transfers. Neither mode guarantees that a model fits: execution
workspaces and other allocations also require memory.

```shell
uv run --project studio python -m serve --backend cuda --model ./models/decision.tq --jev-weights stream --no-web-ui
```

Replace the example artifact with a supported decision checkpoint. The setting
is forwarded to CUDA workers only for artifacts marked `structured_decisions`;
it is not a general MoE expert-cache switch. API loading uses the corresponding
`jev_weights` field, also defaulting to `"stream"`. See the
[decision request contract](api/http.md#82-structured-decisions) for request
fields and exclusive-runtime limits. The d1 text adapter supports CUDA and
Metal, both with streamed and resident weights.
For API loading, omit `jev_weights` to use d1 residency automatically, or select
`"vram"` explicitly; explicit `"stream"` is rejected. See
[model discovery and runtime pool](#model-discovery-and-runtime-pool) above.

## Command options

| Option | Meaning | Default |
| --- | --- | --- |
| `--model PATH` | Optional initial GGUF/TQ file or supported native HF directory. | None |
| `--running-executable PATH` | Prebuilt native worker; the normal choice for a source checkout and for packaged deployments. | Record from an older installation |
| `--host HOST` | Public API bind address. | `127.0.0.1` |
| `--port PORT` | OpenAI API bind port. | Saved listener port or `8090` |
| `--anthropic-port PORT` | Dedicated Anthropic Messages API port. | Saved listener port or `8091` |
| `--access-log / --no-access-log` | Emit one HTTP access-log line per request. | Enabled |
| `--jev-weights {stream,vram}` | Weight placement for supported CUDA decision models: RAM with layer streaming or VRAM residency. | `stream` |
| `--context-size N` | Target context; `0` uses saved policy or declared model capacity. | `0` |
| `--prefill-chunk-size auto\|N` | Native compute chunk limit; CUDA selects a fitting tile up to 8192. | `auto` |
| `--runtime-startup-timeout SECONDS` | Time allowed for native startup. | `1800` |
| `--data-dir PATH` | Managed database, media, models, and components. | `./.tyloinfer` |
| `--db PATH` | Deprecated explicit database override retained for compatibility. | None |
| `--web-root PATH` | Prebuilt Web UI directory. | Auto-detect |
| `--no-web-ui` | Disable Web UI discovery and building. | Off |
| `--model-dir PATH` | Add a model discovery root; repeatable. | Environment/model parent |
| `--work-dir PATH` | Workspace boundary for tool jobs. | Current directory |
| `--api-key-env NAME` | Environment variable containing the administrator API key. | `TYLOINFER_SERVER_API_KEY` |
| `--max-runtime-instances N` | Maximum managed native workers. | `2` |
| `--max-requests-per-runtime N` | Concurrent requests per worker. | CUDA and Metal `8`, HIP `1` |
| `--max-queued-requests-per-runtime N` | Waiting requests per worker. | `max(32, 4 × concurrency)` |
| `--max-runtime-memory SIZE` | Aggregate resident-model admission budget, such as `96G`. | None |
| `--moe-gpu-cache-gb N` | Resident routed-expert cache in GiB; `0` keeps experts resident. | Automatic where supported |
| `--no-memory-guard` | Disable the automatic Metal model-residency budget. | Off |
| `--prefix-cache-dir PATH` | Persistent Session KV cache directory. | Managed `prefix-cache` directory beside the server database |
| `--prefix-cache-disk-size SIZE` | Maximum persistent Session KV cache size, such as `100G`. | None |
| `--prefix-cache-hot-size SIZE` | Maximum hot prefix-cache size, such as `2G`; its memory tier depends on the backend. | None |
| `--prefix-cache-block-tokens N` | Tokens per content-addressed Session KV block. | Runtime default |
| `--no-prefix-cache` | Disable persistent Session KV caching. | Off |
| `--runtime-idle-timeout SECONDS` | Default unload delay for an idle unpinned model. | Runtime default |
| `--log-level LEVEL` | Uvicorn log level. | `info` |
| `--backend {auto,cuda,hip,metal}` | Select or detect the native backend. | `auto` |
| `--transport {stdio,http}` | Python-to-native-runtime communication transport. | `stdio` |

When the current database does not yet exist, an existing legacy Studio
database named `tyloinfer-server.sqlite3` and its matching media directory are reused
automatically from `--data-dir`.

## Troubleshooting

### The model file is rejected immediately

When supplied, `--model` must be an existing model file or supported native HF directory. Check the path before
debugging runtime or server startup.

### Native startup times out

Model loading is included in the startup deadline. Increase
`--runtime-startup-timeout` for large models or slower storage, then inspect the
native worker output printed by `python -m serve`.

### The public port is already in use

Choose another `--port`. The private worker port is allocated automatically and
does not need configuration.

### Web UI assets are missing

Install Node.js/pnpm so TyloInfer can build `studio/web`, pass a valid
`--web-root`, or use `--no-web-ui` for an API-only process.
