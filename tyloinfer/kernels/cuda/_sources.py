"""Define the shared CUDA JIT sources and their compiler search paths."""

from pathlib import Path

SOURCES = (
    'engine/backends/cuda/kernels/kv/cache.cu',
    'engine/backends/cuda/kernels/elementwise/activation.cu',
    'engine/backends/cuda/kernels/embedding/lookup.cu',
    'engine/backends/cuda/kernels/moe/routing.cu',
    'engine/backends/cuda/kernels/projection/mx/matmul.cu',
    'engine/backends/cuda/kernels/projection/sq/fp8.cu',
    'engine/backends/cuda/kernels/projection/sq/mxfp4.cu',
    'engine/backends/cuda/kernels/projection/nint/matmul.cu',
    'engine/backends/cuda/kernels/projection/nint/short_decode.cu',
    'engine/backends/cuda/kernels/projection/nint/q8_decode.cu',
    'engine/backends/cuda/kernels/projection/nvq/matmul.cu',
    'engine/backends/cuda/kernels/projection/nvq/decode.cu',
    'engine/backends/cuda/kernels/projection/nvq/routed_reuse.cu',
    'engine/backends/cuda/kernels/projection/nepq/residual.cu',
    'engine/backends/cuda/kernels/rope/jit.cu',
    'engine/backends/cuda/kernels/sampling/jit.cu',
    'engine/backends/cuda/kernels/recurrent/gdn/jit.cu',
    'bindings/cuda/module.cpp',
    'bindings/cuda/elementwise/acc.cpp',
    'bindings/cuda/norm.cpp',
    'bindings/cuda/elementwise/activation.cpp',
    'bindings/cuda/models/qwen4_exp.cpp',
    'bindings/cuda/attention/selected.cpp',
    'bindings/cuda/rope.cpp',
    'bindings/cuda/sampling.cpp',
    'bindings/cuda/embedding.cpp',
    'engine/backends/cuda/kernels/norm/jit.cu',
    'engine/backends/cuda/kernels/elementwise/acc_jit.cu',
    'engine/backends/cuda/kernels/attention/jit.cu',
    'engine/backends/cuda/kernels/recurrent/convolution_jit.cu',
    'engine/backends/cuda/kernels/projection/nepq/jit.cu',
    'engine/backends/cuda/kernels/qwen4_jit.cu',
    'engine/backends/cuda/kernels/attention/selected_jit.cu',
    'engine/backends/cuda/kernels/attention/sparse_mma_jit.cu',
    'bindings/cuda/kv.cpp',
    'bindings/cuda/recurrent/convolution.cpp',
    'bindings/cuda/models/glm5_next.cpp',
    'bindings/cuda/attention/dense.cpp',
    'bindings/cuda/recurrent/gdn.cpp',
    'bindings/cuda/moe.cpp',
    'bindings/cuda/projection/mx/forward.cpp',
    'bindings/cuda/projection/packed/backward.cpp',
    'engine/backends/cuda/kernels/projection/packed/backward.cu',
    'bindings/cuda/projection/mx/backward.cpp',
    'engine/backends/cuda/kernels/projection/mx/backward.cu',
    'bindings/cuda/projection/nint/forward.cpp',
    'bindings/cuda/projection/nint/backward.cpp',
    'engine/backends/cuda/kernels/projection/nint/backward.cu',
    'bindings/cuda/projection/nvq/forward.cpp',
    'bindings/cuda/projection/nvq/backward.cpp',
    'engine/backends/cuda/kernels/projection/nvq/backward.cu',
    'bindings/cuda/projection/nepq/forward.cpp',
    'bindings/cuda/projection/nepq/residual.cpp',
    'bindings/cuda/projection/nepq/residual_backward.cpp',
    'engine/backends/cuda/kernels/projection/nepq/residual_backward.cu',
    'bindings/cuda/projection/sq/fp8.cpp',
    'bindings/cuda/projection/sq/fp8_backward.cpp',
    'bindings/cuda/projection/sq/mxfp4.cpp',
    'bindings/cuda/projection/sq/mxfp4_backward.cpp',
    'engine/backends/cuda/kernels/projection/sq/mxfp4_backward.cu',
    'engine/backends/cuda/native/env.cpp',
)


def source_root(package: Path) -> Path:
    """Prefer a checkout's canonical sources and otherwise use wheel resources."""
    checkout = package.resolve().parents[2]
    if (checkout / 'csrc/bindings/cuda/module.cpp').is_file():
        return checkout
    return package / '_src'


def include_roots(root: Path) -> tuple[Path, ...]:
    """Return compiler include directories in their original search order."""
    csrc = root / 'csrc'
    cuda = csrc / 'engine/backends/cuda'
    ggml = csrc / 'components/ggml'
    return (
        root, cuda, cuda / 'kernels', cuda / 'native',
        csrc, ggml / 'include', ggml / 'src', ggml / 'src/ggml-cuda',
    )


def sources(root: Path) -> tuple[Path, ...]:
    """Return canonical CUDA JIT sources, including Torch bindings."""
    csrc = root / 'csrc'
    return tuple(csrc / name for name in SOURCES)
