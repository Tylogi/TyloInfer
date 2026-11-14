"""Define the shared CUDA JIT sources and their compiler search paths."""

from pathlib import Path

KERNELS = (
    'kv_cache.cu', 'activation.cu', 'embedding.cu',
    'moe.cu', 'mx_matmul.cu', 'fp8_sq.cu', 'mxfp4_sq.cu', 'nint_matmul.cu',
    'nint_short_decode.cu', 'nint_q8_decode.cu', 'nvq_matmul.cu',
    'nvq_decode.cu', 'nvq_routed_reuse.cu',
    'nepq_residual.cu',
)

ADAPTERS = (
    'mfq_cuda.cpp', 'acc.cpp', 'norm.cpp', 'activation.cpp', 'qwen4_exp.cpp',
    'selected_attention.cpp', 'rope.cpp', 'rope_jit.cu', 'sampling.cpp', 'embedding.cpp',
    'norm_jit.cu', 'acc_jit.cu', 'attention_jit.cu', 'gated_delta_net_jit.cu',
    'sampling_jit.cu', 'ssm_conv_jit.cu', 'nepq_jit.cu',
    'qwen4_exp_jit.cu', 'selected_attention_jit.cu',
    'sparse_mma_raw_jit.cu',
    'kv_cache.cpp', 'ssm_conv.cpp', 'glm5_next.cpp', 'attention.cpp',
    'gated_delta_net.cpp', 'moe.cpp', 'mx_matmul.cpp',
    'packed_backward.cpp', 'packed_backward.cu', 'mx_backward.cpp', 'mx_backward.cu',
    'nint_matmul.cpp', 'nint_backward.cpp', 'nint_backward.cu',
    'nvq.cpp', 'nvq_backward.cpp', 'nvq_backward.cu',
    'nepq.cpp', 'nepq_residual.cpp', 'nepq_residual_backward.cpp', 'nepq_residual_backward.cu',
    'fp8_sq.cpp', 'fp8_sq_backward.cpp', 'mxfp4_sq.cpp',
    'mxfp4_sq_backward.cpp', 'mxfp4_sq_backward.cu',
)


def source_root(package: Path) -> Path:
    """Prefer a checkout's canonical sources and otherwise use wheel resources."""
    checkout = package.resolve().parents[2]
    if (checkout / 'csrc/engine/backends/cuda/kernels/activation.cu').is_file():
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


def sources(root: Path, package: Path) -> tuple[Path, ...]:
    """Return raw CUDA, Torch adapter and environment provider source paths."""
    cuda = root / 'csrc/engine/backends/cuda'
    return (
        *(cuda / 'kernels' / name for name in KERNELS),
        *(package / name for name in ADAPTERS),
        cuda / 'native/env.cpp',
    )
