# Inference performance

## First-request baseline

Use the first inference request after loading the model. Send it as soon as
the model is ready, without an extra generation warm-up or an idle delay.
Disable prefix reuse and use the same prompt, output limit and sampling
settings for every engine. Set input lengths using token counts, not characters.

Start a single-model, single-request server:

```sh
uv run tyloinfer serve --model /models/model.gguf \
  --context-size 32768 --max-runtime-instances 1 \
  --max-requests-per-runtime 1 --no-prefix-cache --no-web-ui
```

Get the loaded model's ID from `/v1/models`. Save a public or synthetic test
prompt in `request.json`, replacing the placeholders:

```json
{
  "model": "MODEL_ID",
  "messages": [{"role": "user", "content": "REPLACE_WITH_TEST_PROMPT"}],
  "max_tokens": 256,
  "temperature": 0,
  "seed": 42,
  "enable_thinking": false,
  "enable_mtp": false,
  "stream": false
}
```

Send one request:

```sh
curl --fail-with-body http://127.0.0.1:8090/v1/chat/completions \
  -H 'Content-Type: application/json' --data-binary '@request.json'
```

In PowerShell, use `curl.exe` and put the command on one line. Authentication
and curl compatibility are covered in the [serving guide](cli/serve.md#first-api-request).

For native MTP or DFlash2, configure a matching
[companion](cli/serve.md#speculative-decoding-mtp--dflash2) before launch, then
set `"enable_mtp": true` and choose `mtp_max_draft_tokens`. Run each mode in a
fresh process and check the returned `mfq_metrics` for the mode actually used.

## Timing

- **Prompt processing:** input tokens divided by full prefill wall time.
- **TTFT:** elapsed time from sending the request to receiving its first token.
- **Generation comparisons:** exclude the first complete decode round's time
  and tokens together, then divide the remaining tokens by elapsed wall time
  after that round. Apply the same rule to ordinary decode, native MTP and
  DFlash2; count all later CPU, GPU, I/O and scheduling delays.

First-round exclusion needs a per-round timeline; it cannot be reconstructed
from aggregate `decode_tps` alone. Service rates in `mfq_metrics` remain the
API's own metrics. Streaming requests need
`"stream_options":{"include_usage":true}` for final usage and metrics.

For repeated samples, reload in a fresh process and measure its first request
again. Keep later requests and prefix-hit measurements separate. Preserve each
sample and use a consistent summary, such as the median. Match expert residency,
KV settings and concurrency across engines; measure
[continuous batching](runtime/cuda-continuous-batching.md) separately.
