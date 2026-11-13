# Performance measurement guide

## Using this guide

The README tables contain previously reported project measurements, with test
conditions listed alongside them. This guide provides a protocol for collecting
new measurements. To reproduce a specific result, use that run's saved
configuration and artifacts.

## Record before running

Keep these inputs and outputs together in a separate result directory:

- Exact source commit (`git rev-parse HEAD`), local diff, build command, CMake
  arguments, chosen native executable and `build/mfq-runtime.json`. Rebuild
  after changing source; `serve` can reuse an existing executable.
- OS, GPU/CPU, memory, driver, toolkit/compiler versions, storage location and
  concurrent workloads. Record Windows versus WSL2 explicitly.
- Model and tokenizer repository revisions, quantization, complete filenames
  and SHA-256 hashes, including all shards, PLE/Engram tables and MTP/DFlash2
  companions. Record draft quantization independently from target quantization.
- Exact startup command and non-secret environment overrides; context, prefill
  chunking, expert residency, KV quantization/offload, prefix-cache policy,
  concurrency and loaded runtime profile. Do not archive API keys.
- Actual prompt bytes and tokenizer/chat-template identity, reported prompt and
  completion token counts, sampling parameters/seed, thinking policy, draft
  mode/depth/probability/confidence/cost settings, and stop conditions.
- Warm-up and repetition policy, every raw response and worker log, errors,
  actual `mtp_used`/acceptance when reported, and the aggregation method.

For each engine being compared, record its exact revision, command and these
settings, including differences in activations, output heads, draft precision
and host environment.

## CUDA measurement example

First complete the [API smoke test](cli/serve.md#first-api-request) and download
compatible model artifacts. Replace the example paths below with your files.
Start one model and one request at a time. For a no-prefix-reuse baseline, run
these commands in the server terminal:

```sh
uv run tyloinfer build --backend cuda
uv run tyloinfer serve --backend cuda --model /models/target.gguf \
  --context-size 32768 --max-runtime-instances 1 \
  --max-requests-per-runtime 1 --no-prefix-cache --no-web-ui
```

Choose a context size that fits your model and hardware; 32768 is an example.
Record automatic choices from worker logs. Keep expert residency and KV policy
identical between variants. For a supported CUDA native MTP companion, add
`--mtp-native-draft /models/matching-mtp.gguf`; for a supported CUDA DFlash2
checkpoint, use `--dflash2-draft /models/matching-dflash2` instead. These CLI
switches require CUDA and are mutually exclusive. See
[speculative decoding](cli/serve.md#speculative-decoding-mtp--dflash2) for
applicability and optional policies.

In another terminal, query `/v1/models`. Save the following as `request.json`,
replacing `MODEL_ID` and the prompt with a saved representative input:

```json
{
  "model": "MODEL_ID",
  "messages": [{"role": "user", "content": "REPLACE_WITH_SAVED_PROMPT"}],
  "max_tokens": 256,
  "temperature": 0,
  "seed": 42,
  "enable_thinking": false,
  "enable_mtp": false,
  "stream": false
}
```

For speculation, save a separate request with `"enable_mtp": true` and
`"mtp_max_draft_tokens": 3` (or your declared depth), keeping the remaining
settings fixed. A requested 256-token output can stop earlier, so use actual
completion counts and record finish reasons. Seed and greedy sampling specify
the protocol; they do not guarantee bitwise equality across engines/backends.

This POSIX-shell example saves one warm-up response and five measured responses
in a new directory. Choose and record the repetition count for your experiment.
`curl --fail-with-body` requires curl 7.76 or newer; with older curl, check HTTP
status explicitly before recording a response as a successful measurement.

```sh
mkdir benchmark-run
cp request.json benchmark-run/request.json
git rev-parse HEAD > benchmark-run/commit.txt
git diff > benchmark-run/source.diff
curl --fail-with-body http://127.0.0.1:8090/v1/models \
  -o benchmark-run/models.json
curl --fail-with-body http://127.0.0.1:8090/v1/chat/completions \
  -H 'Content-Type: application/json' --data-binary '@benchmark-run/request.json' \
  -o benchmark-run/warmup.json
for i in 1 2 3 4 5; do
  curl --fail-with-body http://127.0.0.1:8090/v1/chat/completions \
    -H 'Content-Type: application/json' --data-binary '@benchmark-run/request.json' \
    -o "benchmark-run/run-$i.json" || break
done
```

Use a new output directory per variant; keep startup/build logs alongside it.
These examples assume local authentication is disabled. If enabled, provide a
Bearer credential without recording it in published logs. PowerShell users
should use `curl.exe` and adapt shell continuations and loops, rather than
pasting POSIX syntax unchanged.

## Interpret and report

- Preserve the complete `usage` and `mfq_metrics` objects when present; the
  latter name is a retained response identifier. Missing metrics mean
  unavailable data, not zero. Streaming requires
  `"stream_options":{"include_usage":true}` to receive final usage/metrics.
- Record PP (`prefill_tps`), generation (`decode_tps`), TTFT (`ttft_ms`),
  accepted/drafted token counts and MTP usage when available. TTFT and decode
  rate measure different intervals; record external wall-clock timing
  separately for end-to-end measurements.
- A repeated prompt with prefix caching enabled can avoid prompt work. Report
  no-prefix-reuse and warm-prefix runs separately, including hit counts. The
  example disables prefix reuse, but its warm-up can still warm GPU kernels,
  expert residency and OS file caches. Distinguish these conditions from a
  cold-SSD measurement.
- Use real tokenized 4K/8K prompts for long-prefill tests, not character counts
  or the short smoke-test prompt. Record the actual count, template overhead,
  output length and context; ensure input plus output fits the loaded context.
- Report all successful samples and failures, along with median and range or
  another declared statistic. Separate warm-up from measured repetitions and
  quality/correctness evaluation from speed measurements.
- Continuous batching changes scheduling and speculation/prefix availability;
  measure it as a separate experiment. See the [batching limits](runtime/cuda-continuous-batching.md).
