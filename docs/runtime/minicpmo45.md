# MiniCPM-o 4.5 Runtime

TyloInfer supports the official `OpenBMB/MiniCPM-o-4_5` composite graph in the
Python/CUDA runtime and in the native C++ CUDA and Apple MLX runtimes. The
Python path delegates graph ownership to the official remote-code modules and
replaces supported `Linear` and `Embedding` weights with packed operators. The
native paths implement SigLIP, the Resampler, Whisper, the Qwen3-8B language
backbone, TTS, attention masks, and streaming caches directly.

The runtime requires the official model directory for its tokenizer, processor,
remote-code definitions, and Token2wav assets. TyloInfer does not copy those assets
into the weight file.

## Environment

Install the model-specific extra:

```bash
pip install -e ".[minicpmo45]"
```

The extra pins `transformers==4.51.0` and the official Torch/Torchaudio 2.3–2.8
range. TTS and streaming install `minicpmo-utils[all]>=1.0.5`.

Apple realtime serving uses a separate media extra so the native model worker
does not inherit Python runtime dependencies:

```bash
pip install -e ".[metal,minicpmo45-realtime]"
```

## Load

The legacy NINT/MFQ examples below require the corresponding reference path or
an explicit [native compatibility build](../cli/build.md#native-format-profiles).
They are not the default GGUF/TQ quick-start recipe. Retained native executable
names are internal interfaces, not old public CLI aliases.

```python
from tyloinfer.runtime import load_minicpmo45

runtime = load_minicpmo45(
    "/models/OpenBMB/MiniCPM-o-4_5",
    "/models/MiniCPM-o-4_5-NINT4.mfq",
    device="cuda:0",
)

model = runtime.model
print(runtime.load_report)
```

`runtime` delegates the official model methods, including `forward`, `chat`,
streaming, duplex, and TTS entry points. Call them with the same processor data
and arguments used by the official repository.

## Native C++ composite graph

The native CUDA model implements SigLIP, the Resampler, Whisper, the audio
projector, Qwen3, and the TTS code decoder. The same graph is
available on Apple platforms through `MlxMiniCPMO45Runtime`; the Metal HTTP
server loads the full visual graph for MiniCPM-o models.

TyloInfer Server applies the MiniCPM-o image pipeline to HTTP input:

- optional 448-pixel slicing;
- bicubic resize and `[0, 1]` conversion;
- mean/std normalization;
- 14-pixel patch packing and contiguous patch masks;
- 64-query placeholders.

Video is decoded at one frame per second without per-frame slicing. CUDA and
Metal receive the same versioned `pixel_values`, `patch_mask`, and
`target_sizes`. The native server applies the chat template, tokenizes it,
derives `image_bounds` from `<image>` and `<slice>` spans, and injects Resampler
outputs into Qwen3 prefill.

The converter embeds the NumPy-generated Resampler position table as a
versioned BF16 runtime asset, avoiding platform-dependent sin/cos rounding in
cross-attention.

The former CUDA tensor-file diagnostic CLI and its Program owner have been
removed. Native requests use the Engine and transport interfaces below.

## Native full-duplex session

The native CUDA Engine accepts processor-produced units through its MiniCPM
duplex controls. The Apple runtime exposes the equivalent state machine through
`prepare_duplex()` and `duplex_step()`. Both preserve the official streaming
Whisper, Qwen3, and TTS caches across units and follow the listen/speak,
chunk-end, and turn-end state transitions used by `MiniCPMODuplex`.

On Apple, `MlxMiniCPMO45DuplexResult` reports all three cache positions and the
streaming audio index after every unit. TTS only returns codes that have already
advanced the TTS KV cache, so the next unit can verify its expected position
exactly. `reset()` destroys the duplex state and clears all modality caches.

### Realtime service

Start the native CUDA or Apple worker. MiniCPM-o models expose their realtime
duplex capability automatically; it is not a server launch option:

```bash
CUDA_VISIBLE_DEVICES=0 tyloi-engine \
  --model /models/MiniCPM-o-4_5.mfq \
  --transport http \
  --host 127.0.0.1 \
  --port 8081
```

On Apple Silicon, use the Metal executable with the same server options:

```bash
mfq-decode-metal \
  --model /models/MiniCPM-o-4_5.mfq \
  --server \
  --host 127.0.0.1 \
  --port 8081
```

Start the media gateway with the official `assets/token2wav` directory and
`system_ref_audio.wav` available below the assets root:

```bash
tyloinfer-minicpmo-realtime \
  --backend http://127.0.0.1:8081 \
  --assets /models/OpenBMB/MiniCPM-o-4_5 \
  --host 127.0.0.1 \
  --port 8090
```

Open `http://127.0.0.1:8090/`. Realtime-capable models add voice and playback
controls to the chat composer. Text is added to the conversation. Audio is
always generated; the playback toggle only controls browser playback. Other UI
pages proxy their API requests to the native worker.

Voice defaults match the MiniCPM-o demo:

- one 16 kHz audio unit per second;
- three initial force-listen units;
- at most 20 generated text tokens per unit;
- temperature 0.7, top-k 20, and top-p 0.8;
- repetition penalty 1.05 over 512 tokens;
- turn length penalty 1.05.

TTS uses temperature 0.8 and repetition penalty 1.05. Token2wav uses 10 flow
steps, and the browser buffers 200 ms before playback. The Web UI selects the
Chinese or English duplex prompt unless the user sets a system prompt. Generic
chat sampling options do not change these voice defaults. Token2wav prefers
CUDA, then Apple MPS, then CPU.

The media protocol is `WS /v1/realtime?mode=audio`. Clients send base64 float32
mono PCM at 16 kHz and receive separate text deltas and base64 float32 mono PCM
at 24 kHz.

Streaming Mel uses a 1030 ms first window and two frames of CNN boundary
context. Flow and HiFT render S3 codes on MPS, with CPU fallback. The first TTS
chunk uses early flush; later chunks keep three S3 lookahead codes and consume
25 codes at a time.

The current public media gateway is audio-only. Native callers can still use
the composite graph interfaces for image and mixed-modality requests. Quality
and performance results require a calibrated checkpoint and modality-specific
reference evaluation; structural support alone does not establish those
results.
