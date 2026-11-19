# Standalone native d1 text execution

`tyloinfer d1-native` runs the LFM3 text backbone in a separate CUDA/cuBLAS
executable. Python performs tokenization, question packing and calibrated answer
readout; it does not run a PyTorch numerical fallback or checkpoint Python.

The standalone command is a diagnostic entry point. Managed serving uses the
same native engine through the existing `/v1/systemone` and model lifecycle;
see [Managed serving](#managed-serving). The [reference command](d1-reference.md)
remains separate. Chat/Agent generation is not supported by this decision model.

## Build and run

Requirements: CMake 3.26+, a C++17/CUDA compiler, CUDA 11.2+ Runtime/cuBLAS, and a CUDA
device supporting BF16 Tensor Core GEMM (compute capability 8.0 or newer). The
executable rejects unsupported devices. No driver or toolkit is installed by
this command.

```sh
cmake -S cpp_runtime/backends/cuda/d1 -B build/d1 -DCMAKE_BUILD_TYPE=Release
cmake --build build/d1 --target tyloinfer-d1-native --config Release
```

The standalone target defaults to native architecture detection. Explicit
`CMAKE_CUDA_ARCHITECTURES` or `CUDAARCHS` overrides are respected; cross-compilation
and distributable multi-architecture binaries need an explicit target list.
The executable is normally `build/d1/bin/tyloinfer-d1-native`; multi-configuration
and Windows generators may add a configuration directory and `.exe` suffix.
The standard CUDA CMake build also builds this target beside the general runtime.
Custom packages must keep the two executables together.

Managed d1 serving defaults to `--jev-weights stream`: weights stay in RAM,
with two reusable GPU layer buffers. The next layer is copied while the
current layer computes. Embedding and answer rows are gathered from RAM.
Select `--jev-weights vram` to keep all text weights in VRAM instead.
The standalone command retains its resident default; use
`--weights-placement stream` to select the same streaming implementation.

Prepare matching local HF assets (`config.json`, `tokenizer.json`,
`calibration.json`) and weights. Nothing is downloaded automatically.

```sh
# HF BF16 language weights; request format is shared with d1-reference.
tyloinfer d1-native --model ./models/d1-2B \
  --executable ./build/d1/bin/tyloinfer-d1-native --request request.json

# Official Q8_0 GGUF weights with the matching HF configuration/tokenizer/calibration.
tyloinfer d1-native --model ./models/d1-2B \
  --weights ./models/d1-2B-Q8_0.gguf \
  --executable ./build/d1/bin/tyloinfer-d1-native --request request.json
```

The initial source contracts are pinned to HF
`LiquidAI/d1-2B@a6bce774e6dde338dcc3036ceb5d91f78bf4c19e` and GGUF
`LiquidAI/d1-2B-GGUF@4a181094d1b6addfaa3a0956f78e7ab2aa406a0c`.
The loader checks structure and GGUF identity/geometry/mixer/RoPE metadata, not
a cryptographic provenance guarantee for arbitrary user files. Supply assets
and weights from the same checkpoint; the command does not validate every
GGUF tokenizer entry against the HF tokenizer.

## Supported scope

- HF BF16 safetensors language weights, tied candidate output projection.
- GGUF declaring `general.architecture=lfm3`,
  `lfm3.decision.type=lfm3-d1`, Q8_0 matrices and floating auxiliary tensors.
- Gated causal short convolution, full set attention, causal sliding attention,
  Q/K RMSNorm, per-head output gate, sandwich norms and zero-tail partial RoPE.
  Attention probabilities and the probability/value product retain FP32 until
  the attention output is rounded to BF16; rounding probabilities earlier can
  change near-boundary median Score answers.
- Text Noul, single-choice Choice and median-level Score. Question sets larger
  than the configured set size repeat the state in independent groups.
- Fused Q/K/V/gate and FFN gate/up projections by default. `--unfused` selects
  the explicit non-fused baseline; no model-size or device-name whitelist.

**GGUF matrices remain Q8_0-packed in RAM and on the GPU.** Projection calls reuse the
shared GGUF grouped-GEMM backend with BF16 activations and FP32 accumulation.
Only bounded 64-by-128 weight tiles are decoded into shared memory; no complete
weight matrix or layer is expanded into a persistent or temporary BF16 image.
Block products use ordinary FP32 partial-sum accumulation without extra
compensation. This is W8A16 weight-only execution, not an A8/integer-activation implementation.
`weight_format=gguf-q8_0-packed-w8a16` identifies the actual path.

Token embeddings and tied answer weights are fetched directly from packed rows.
Only requested token/candidate rows are decoded as activations; the full embedding
table remains packed. Norms and short-convolution kernels are ordinary floating
auxiliary tensors. Unsupported matrix formats are rejected instead of silently
falling back to full-matrix dequantization. Projection fusion concatenates packed
rows without changing their compressed byte count.

Requests are processed sequentially, one packed question set per forward.
There is no KV/conv/prefix reuse, SSD offload, continuous batching, generation,
Metal/CPU backend, media, typed extraction or multiple-choice selection.
All required language weights must fit the selected GPU. Set attention is not
whole-model question isolation: convolution/sliding layers remain causal across
preceding questions, as described in the reference contract.

## Managed serving

Build the CUDA runtime with `tyloinfer build --backend cuda`. Its
`tyloinfer-d1-native` companion must remain beside `mfq-runtime` (with `.exe`
suffixes on Windows). No checkpoint Python is executed or downloaded.

For HF BF16, keep the safetensors weights and matching `config.json`,
`tokenizer.json`, `calibration.json` in one model directory. For Q8_0 GGUF, put
those same three matching HF assets beside the GGUF file. Select the directory
or GGUF through the existing model catalog/load API, or start with:

```sh
tyloinfer serve --model ./models/d1-2B-Q8_0.gguf --jev-weights vram
```

Send Noul, Choice and Score requests to the existing
[`POST /v1/systemone`](../api/http.md#request). Loading, aliases, busy admission,
authentication and unloading use the ordinary server APIs. Score remains the
official d1 median, while D3 retains its mean; there is no readout mode option.
Both existing private HTTP and stdio transports are supported.

Only CUDA text decisions are available. Chat generation returns `501`; media,
typed questions, multiple selection and nonempty template overrides are rejected.
Weights use the selected device with streamed or full-VRAM placement.
There is no prefix/KV reuse or dynamic offload budget. Cancelled callers
do not release the exclusive decision lease until the native call completes.
Managed unload stops the worker and its native child. This does not imply
Studio/Agent chat can generate text with d1, or that custom desktop packages
already contain the required native companion.

## Outputs and timing

The output identifies `native_runtime=true`,
`backend=native-cuda-cublas-lfm3`, `serving_integration=false`, actual
`weight_format` and `fused_projections`. Answers have `output_tokens=0`.
`load_ms` covers language weight loading, transfers, floating-auxiliary conversion
and fusion. `weight_conversion_ms` includes only floating auxiliary conversion;
Q8_0 matrices are copied without host dequantization. It is a subset of `load_ms`,
not an extra additive cost. Neither field measures full CLI startup.

The ready frame includes resident packed/floating weight bytes, streamed RAM
weight bytes, layer-buffer bytes and a per-tensor storage manifest.
Per-forward dispatch counters identify the packed projections,
packed embedding row reads and the small BF16 candidate-only projection.
`bf16_gemm_calls_per_forward` counts weight projections, not the separate attention
QK/PV activation products. Device-used bytes and CUDA temporary-pool high-water
marks are provided for memory accounting; they are not interchangeable with a
process-wide or sampled device peak including CUDA context/workspace overhead.

`--warmup N --iterations N` enables explicit repeated evaluation. Per-set
`native_metrics` contains CUDA-event `gpu_ms` samples and their median. The
measured interval includes forward, candidate projection and in-interval
allocation/synchronization; it is not a pure kernel-only figure. Input copies,
Python/tokenizer/readout, process startup and loading are outside that interval.
`benchmark_wall_ms` includes warmups and result transfer, so do not divide it by
only the measured iteration count. These metrics are not end-to-end TTFT.

## Validation and licensing

```sh
python -m pytest tests/test_d1_native.py tests/test_d1_native_gpu.py tests/test_d1_native_quantized.py tests/test_d1_native_real.py -q
```

Protocol tests use a fake executable and do not prove CUDA correctness. The
GPU test is opt-in via `TYLOINFER_D1_NATIVE_BINARY`; it compares real CUDA output
with a small BF16 reference fixture. The quantized test adds real small Q8_0 GGUF
matrices, incomplete output tiles, multiple question masks, 257/513-token chunk
boundaries, packed-residency assertions and actual projection dispatch checks.
Creating that fixture additionally requires the optional `gguf` Python package;
it is not a runtime dependency of the C++ loader.
The pinned real-answer regressions additionally
require `TYLOINFER_D1_NATIVE_ASSETS`, and `TYLOINFER_D1_NATIVE_GGUF` for the GGUF case.
These inputs must be the official pinned d1-2B checkpoints, not arbitrary models.
Real checkpoint probes are available as
`python -m tests.d1_native_real_probe --help` and
`python -m tests.d1_gguf_reference_probe --help`. The latter independently
loads the same Q8_0 weights into a CPU BF16 correctness reference; its runtime
must never be used as a GPU performance proxy.

Prompt/readout adaptation retains the LFM Open License v1.0 and its commercial
revenue threshold. See the reference command's licensing section and the
repository `NOTICE`; model weights are not redistributed in the package.
