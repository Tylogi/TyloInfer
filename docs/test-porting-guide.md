# TyloInfer C++ 测试移植指南

仓库：`/home/zmx/mine/workspace/MFQ/TyloInfer`，源码树 `csrc/`，构建目录 `/tmp/mfq-try`（已配置好，Release + CUDA）。

## 构建与验证

```bash
cmake --build /tmp/mfq-try --target <test-target-name> -j     # 单个测试
ctest --test-dir /tmp/mfq-try -R <test-name> --output-on-failure   # 运行
```

测试 target 名在 `csrc/tests/CMakeLists.txt` 和 `csrc/engine/backends/cuda/CMakeLists.txt`（BUILD_TESTING 段）里查 `add_test(NAME ...)`。

## API 迁移映射（旧 → 新，测试引用仍待迁移）

| 旧 | 新 |
|---|---|
| `mfq_tensor_backend::Tensor` | `tyloi::DenseTensor`（engine/tensor/tensor.h） |
| `mfq_tensor_backend::TensorOptions` | `tyloi::TensorOptions` |
| `mfq_tensor_backend::kFloat32 / kCUDA / kCPU 等` | `tyloi::kFloat32` 等（tyloi 层，不是 tyloi::cuda） |
| `mfq_tensor_backend::zeros/ones/randn/arange/cat/matmul/where/topk/softmax/...` | `tyloi::cuda::`（engine/backends/cuda/native/tensor/tensor.h） |
| `mfq_tensor_backend::NoGradGuard / manual_seed / from_blob` | `tyloi::cuda::` |
| `mfq::cuda::X` | `tyloi::cuda::X` |
| `tyloi::cuda::Ffn` / `ops::FfnWeights` | 权重使用 `models::Mlp` / `models::Moe`；数值调用使用无状态 `ops::Ffn`，逐项传入 Tensor 和参数（测试尚未修改） |
| `cpp_runtime/transport/src/decision_protocol.h` | 决策协议迁至 `csrc/components/protocol/decision.h`。旧 `decision_test.cpp` 的包含路径和 Engine 事件断言待迁移 |
| 旧 `DecisionText`、`DecisionTokens` 控制命令 | 决策进入普通请求调度，结果是 `DecisionReadResult` 事件。旧 `request_executor_test.cpp` 和 `mfq_runtime_transport_test.cpp` 的控制命令断言待迁移 |
| `cpp_runtime/backends/cuda/runtime/layer_weight_stream.h` | 旧的逐层换绑已删除。流式权重通过原生 Loader、Executor 层事件和 Qwen 前向验证。旧 `layer_weight_stream_test.cpp` 待迁移 |
| `HfModelSource` / `HfSafetensorsSource` | 使用 `models::HfSource` 和 `Safetensors`。旧的 `model_source_test.cpp`、`mlx_affine_model_source_test.cpp`、`channel_fp8_test.cpp`、`server_inference_test.cpp` 和 Python 源码断言待迁移 |
| `MfqModelSource` | 普通、分片和映射读取使用 `models::MfqSource` 和 `weights/source/mapped.cpp`。旧的 `model_source_test.cpp` 及其中独立 Engram 分片的入口待迁移 |
| `cpp_runtime/backends/cuda/commands/runtime.cpp` 和 `mfq-runtime` | 命令入口是 `csrc/commands/runtime.cpp`，可执行文件是 `tyloi-engine`。旧 CUDA CLI 测试和 `test_cpp_runtime_architecture_boundaries.py` 待迁移 |
| `cpp_runtime/scheduler/src/scheduler.cpp` 和 `cpp_runtime/transport/src/` | 调度使用 `csrc/engine/scheduler/`，协议和独立传输使用 `csrc/components/protocol/` 和 `csrc/transport/`。旧传输测试及 `test_cpp_runtime_continuous_batching_source.py`、`test_cpp_runtime_session_state_source.py` 和 `test_cpp_runtime_architecture_boundaries.py` 待迁移 |
| `TYLOINFER_RUNTIME_CHECK` | 生产代码使用普通 `if` 和 `throw std::runtime_error`。CUDA 头文件暂时保留原定义，供 `qsa_indexer_cache_test.cpp:13` 和 `float_projection_test.cpp:385` 使用 |
| `ops::Linear` / `cuda::LinearCore` | 已删除；层组合使用 `models::LinearAttention`，状态使用现有 `state/recurrent`，数值调用使用 `ops::linear_*`（测试尚未修改） |
| `DenseLinearGroup` / `make_fp32_quant_group` | 权重使用 packed Tensor；FP32 重建使用 weights 私有准备入口 `fp32_group`；执行使用 `ops::project` 后按原边界拆分（测试尚未修改） |
| `mfq::ModelGraph / ModelSource / TensorMetadata` | `tyloi::` 同名 |
| `mfq_cuda_synchronize()` | `cudaStreamSynchronize(::tyloi::cuda::current_stream())` |
| `namespace tb = mfq_tensor_backend` | `namespace tb = tyloi::cuda` + 需要 tyloi 层符号时用全限定名 |

测试文件顶部若用了裸符号（`Tensor`、`kFloat32`），用这段开头：
```cpp
using namespace tyloi;
using namespace tyloi::cuda;
using Tensor = ::tyloi::DenseTensor;
```

## 头文件迁移映射

| 旧 include | 新 |
|---|---|
| `tyloinfer/runtime.h` | `engine/backends/cuda/native/context.h`（按实际用到的类选 native/*.h） |
| `tyloinfer/model_source.h` | `engine/models/source/model.h` |
| `execution/execution.h`、`execution/options.h` | `engine/backends/cuda/executor/execution.h`、`.../options.h` |
| `storage/session/state.h` | `engine/models/common/session/state.h` |
| `storage/session/text_cache.h` | **已删除**（TextCache 不存在了） |
| `storage/weights/loaders/weight.h` | `engine/weights/loader.h` |
| `storage/weights/moe/dma_copy_batch.h` | `engine/backends/cuda/native/copy_batch.h` |
| `storage/weights/moe/native_dma_batch.h` | `engine/backends/cuda/native/dma_batch.h` |
| `mfq_cuda_kernels.h` 等裸内核名 | `engine/backends/cuda/kernels/<同名>` |
| `mx.h` | `engine/weights/prepare/mx.h` |
| `engine/backends/cpu/native/host_memory.h` | `components/host_memory.h`（独立的 `mfq-host-memory` target；现有两处 CUDA 测试引用尚未修改） |
| `engine/state/kv_memory.h` / `state/kv_budget.h` | CUDA 存储用 `engine/backends/cuda/executor/kv/memory.h`；几何用 `state::QsaLayout`；预算策略用 `engine::KvOptions` / `Budget::place_kv`；offload 配置用 `state::KvOffloadConfig`（测试尚未修改） |
| `cuda::KvOffloadStore` / `cuda::KvOffloadSequence` / `KvOffloadStore::GatherWorkspace` | `state::KvOffloadStore` / `state::KvOffloadSequence` / `ops::KvGatherWorkspace`（测试尚未修改） |
| `models/common/axial_patch_vision_ops.h` | `engine/models/common/vision/axial_patch_vision.h` |
| `cuda_model_plan.h` | **已删除**（模型直接校验声明，并通过 `CausalModel::capabilities()` 报告实际能力） |
| `engine/ops/linear.h` / `engine/backends/cuda/ops/attention/linear.h` | **已删除**；按职责使用 `engine/models/common/linear.h`、`engine/ops/ops.h` 和 `engine/state/recurrent/state.h` |
| `runtime_config.h` / `cuda_runtime_config.h` | `engine/load.h` |
| `env.h`（kernels 头文件内） | `engine/backends/cuda/native/env.h` |

## Buffer / 内存类
`tyloi::cuda::Buffer`、`HostBuffer` 在 `engine/backends/cuda/native/memory.h`。

## 决策规则（对每个坏测试）

1. 被测的模块/函数在 `engine/` 下还存在 → 最小改动移植（改 include、改类型名、按新签名调整调用）。不要重写测试逻辑。
2. 被测机制已彻底删除（如 TextCache、QuantLinearKind）→ 删除测试文件 + CMakeLists 里对应的 add_executable/add_test 块，并在汇报里说明删了什么、为什么。
3. 拿不准 → 保持文件不动，汇报疑问，不要瞎改。

## 已隔离（不要动）
- 旧 diagnostic/eval 工具、Program 和对应诊断视图已删除。测试中的旧引用仍待迁移；当前限制下不修改或运行测试。
- bench 可执行目标挂在 `TYLOINFER_BUILD_BENCHES`（默认 OFF）下。
- hip/metal 后端不动。
