# Performance recording and reproduction limits

## Status of the README tables

The README numbers are existing project-reported results, not measurements
produced during this documentation repair. The tables include hardware and
some comparison conditions, but this checkout does not currently provide a
complete per-row chain of target/draft artifact revisions, project commit,
commands, prompts, raw outputs, repetition counts and aggregation rules.
**The tables are not yet independently reproducible from this guide.** This is
an evidence gap, not a new validation or a finding that the numbers are wrong.

Do not invent missing logs or label a new run as a reproduction of those rows.
The commands below are a starting protocol for new measurements. They have been
checked against the CLI/API implementation but were not run against a model
in this documentation revision. No new speed, quality or memory claim is made.

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

For each engine being compared, record its exact revision, command and all of
these settings too. Different activations, output heads, draft precisions and
host environments must remain visible as comparison caveats.

## New-run example (CUDA, not a README-table recipe)

First complete the [API smoke test](cli/serve.md#first-api-request). Download
compatible artifacts separately; these placeholder paths are not download
instructions or a validated minimum-memory configuration. Start one model and
one request at a time. For a no-prefix-reuse baseline, in the server terminal:

```sh
uv run tyloinfer build --backend cuda
uv run tyloinfer serve --backend cuda --model /models/target.gguf \
  --context-size 32768 --max-runtime-instances 1 \
  --max-requests-per-runtime 1 --no-prefix-cache --no-web-ui
```

Use a context that actually fits your artifact and hardware; 32768 is an
example, not a measured resource requirement. Record automatic choices from
worker logs. Keep expert residency and KV policy identical between variants.
For a supported CUDA native MTP companion, add
`--mtp-native-draft /models/matching-mtp.gguf`; for a supported CUDA DFlash2
checkpoint, use `--dflash2-draft /models/matching-dflash2` instead. The switches
are mutually exclusive. See [speculative decoding](cli/serve.md#speculative-decoding-mtp--dflash2)
for applicability and optional policies. These CUDA switches are not Metal
recipes; exact Metal benchmark companion settings remain to be documented.

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

For speculation, create a separate request with `"enable_mtp": true` and
`"mtp_max_draft_tokens": 3` (or your declared depth); do not silently modify a
previous run's request file. Hold the remaining settings fixed. A requested
256-token output can stop earlier, so use actual completion counts and record
finish reasons. Seed and greedy sampling document the protocol; they do not
guarantee bitwise equality across engines/backends.

This POSIX-shell example saves one warm-up response and five measured responses
in a new directory; five is an example protocol, not the historical table's
unknown repetition count. `curl --fail-with-body` requires curl 7.76 or newer.
For older curl, check HTTP status explicitly rather than treating an error JSON
as a measurement.

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
  rate measure different intervals; do not infer end-to-end speed from decode
  rate alone. External wall-clock timing should be labelled separately.
- A repeated prompt with prefix caching enabled can avoid prompt work. Report
  no-prefix-reuse and warm-prefix runs separately, including hit counts. The
  example disables prefix reuse but its warm-up can still warm GPU kernels,
  expert residency and OS file caches: do not call it a cold-SSD benchmark.
- Use real tokenized 4K/8K prompts for long-prefill tests, not character counts
  or the short smoke-test prompt. Record the actual count, template overhead,
  output length and context; ensure input plus output fits the loaded context.
- Report each successful sample and failures, then a declared statistic such as
  median and range; do not select only the fastest run. Separate warm-up from
  measured repetitions. Compare quality/correctness separately from speed.
- Continuous batching changes scheduling and speculation/prefix availability;
  it is a separate experiment. See the [batching limits](runtime/cuda-continuous-batching.md).

A result is reproducible only when another person can identify the exact
artifacts and revision, rerun the declared commands and inspect the raw
responses. Until that record exists, keep README-table comparisons labelled as
reported results rather than verified reproductions.
