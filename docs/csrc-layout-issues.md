# csrc 目录结构与文件命名问题

检查日期：2026-11-15。

范围：整个 `csrc`，重点检查 CUDA 后端及其与 `engine/weights`、`engine/models` 的边界。
以下问题来自当前工作区的静态检查，状态见各条目。文件数量和行数为初次检查时的快照。
各批次记录中的原路径保留为检查快照；2026-11-16 的统一目录整理见第 11 项。

目标职责以 [architecture.md](architecture.md) 为准，实施范围和顺序沿用 [plan.md](plan.md)。

## 1. 算子目录承担权重缓存管理

位置：`csrc/engine/backends/cuda/ops/gguf/`。

- `gguf_experts.cpp` 有 5043 行，包含 `GgufExpertCache`、主机内存分配、设备缓存淘汰和推测路由提交。
- 同目录还有 `gguf_host_snapshot.h`、`gguf_speculative_journal.h`、`gguf_adaptive_slots.h` 等缓存和驻留相关文件。
- 影响：数值计算与权重生命周期混在同一目录，难以判断某项缓存策略应由哪里维护。
- 建议：缓存、驻留和路由事务归入现有 `engine/weights`；数值计算留在 CUDA `ops`；设备分配和传输原语由 CUDA `native/executor` 提供。按职责拆分现有实现，保留错误处理和资源释放逻辑。

## 2. Tensor 层算子接口放在 kernels 目录

状态：所列 Tensor 层声明已迁入 CUDA `ops/`，测试中的旧引用待同步。

位置：`csrc/engine/backends/cuda/kernels/`。

- 原 `mfq_cuda_activation_ops.h` 声明接收和返回 `DenseTensor` 的激活算子，现位于 `ops/activation.h`。
- 原 `mfq_cuda_norm_ops.h` 同样声明 Tensor 层归一化接口，现位于 `ops/norm/launch.h`，相关实现位于 `ops/norm/rms.cpp`。
- 底层原始指针和 CUDA stream 接口分别位于 `kernels/elementwise/activation_raw.h`、`kernels/norm/raw.h`。
- 影响：同一目录同时暴露 Tensor 算子和 kernel 启动接口，目录名不能准确表达接口层级。
- 处理：Tensor 层声明归入对应 `ops` 目录；底层启动接口及设备实现留在 `kernels`。其余历史 `mfq_cuda_*` 通用声明也已按同一规则归组并清理文件前缀。

## 3. 跨模块直接引用私有实现头

- `cuda/ops/dispatch.cpp` 包含 `engine/weights/moe/batch_internal.h`，直接依赖 `ExpertBatch::Impl` 的实现定义。
- `cuda/ops/projection/expert_projection.cpp` 在 MFQ 兼容分支中包含 `engine/weights/moe/cached_source_internal.h`。
- `weights/moe/expert_cache_internal.h` 有 1822 行，包含缓存类定义和大量实现，同时直接依赖 CUDA 执行和设备设施。
- 影响：`_internal` 后缀没有形成模块边界，修改缓存内部结构会波及后端算子。
- 建议：通过 weights 的准备、查询和借用接口提供后端所需数据；单文件使用的实现留在 `.cpp`，确需共享的私有声明按具体职责命名。结合实际调用关系处理这些文件。

## 4. 单消费者实现被拆成额外头文件

状态：已处理（2026-11-15）。

位置：`csrc/engine/backends/cuda/ops/`。

| 已删除的实现头文件 | 原行数 | 合并至 |
| --- | ---: | --- |
| `attention/csa_core.h` | 608 | `attention/csa.cpp` |
| `attention/gdn_core.h` | 727 | `attention/gdn/attention.cpp` |
| `attention/qsa_core.h` | 1023 | `attention/qsa.cpp` |
| `ffn/mixer_core.h` | 366 | `ffn/gated_residual.cpp` |

- 原问题：阅读一个算子的实现需要在两个文件之间跳转，`core` 也没有说明独立职责。
- 处理：将四个单消费者实现头的内容并回对应 `.cpp`，删除原头文件，合并重复 include。数值计算、校验、错误处理和资源生命周期代码保持原有内容。

## 5. 文件名称与实际职责不符，历史前缀残留

状态：生产源文件和头文件已整理；测试文件命名及旧引用保持冻结。

以下路径均相对 `csrc/engine/backends/cuda/`：

| 原文件 | 当前文件 | 职责 |
| --- | --- | --- |
| `kernels/nepq.cu` | `kernels/hadamard/transform.cu` | 通用 Hadamard 变换 |
| `ops/ffn/mixer.cpp` | `ops/ffn/gated_residual.cpp` | gated residual 算子及其私有实现 |
| `ops/attention/wrappers.cpp` | `ops/attention/dense.cpp` | 稠密注意力及其缓存计算 |
| `ops/attention/gdn_wrappers.cpp` | `ops/attention/gdn/recurrence.cpp` | GDN Tensor 算子 |
| `ops/ffn/moe_wrappers.cpp` | `ops/ffn/experts/launch.cpp` | MoE 数值算子 |
| `ops/norm/wrappers.cpp` | `ops/norm/rms.cpp` | RMS 归一化及融合算子 |
| `ops/rope/wrappers.cpp` | `ops/rope/rotary.cpp` | 旋转位置编码及融合算子 |
| `ops/projection/fp8_sq_wrappers.cpp` | `ops/projection/fp8_sq/launch.cpp` | FP8 SQ 数值算子 |
| `ops/projection/mx_wrappers.cpp` | `ops/projection/mx/launch.cpp` | MX 数值算子 |
| `ops/projection/mxfp4_sq_wrappers.cpp` | `ops/projection/mxfp4_sq/launch.cpp` | MXFP4 SQ 数值算子 |
| `ops/projection/nepq_residual_wrappers.cpp` | `ops/projection/vq/residual.cpp` | NEPQ 稀疏残差算子 |
| `ops/projection/nepq_wrappers.cpp` | `ops/projection/hadamard.cpp` | Hadamard Tensor 算子 |
| `ops/projection/nint_wrappers.cpp` | `ops/projection/nint/launch.cpp` | NINT 数值算子 |
| `ops/projection/nvq_wrappers.cpp` | `ops/projection/vq/launch.cpp` | NVQ/NEPQ 数值算子 |

命名约定：文件名只补充目录没有表达的职责；`ops/` 下的实现不再重复 `_ops` 后缀。同一编码的接口和多份实现收拢为格式目录，目录提供编码名称，文件区分 kernel 调用和编码 Tensor 计算：

```text
ops/projection/
├── nint/
│   ├── launch.cpp
│   ├── tensor.cpp
│   └── tensor.h
├── mx/          # 同上
├── fp8_sq/      # 同上
├── mxfp4_sq/    # 同上
└── vq/
    ├── launch.cpp
    ├── tensor.cpp
    ├── tensor.h
    ├── check.cu
    └── residual.cpp
```

`launch.cpp` 负责 DenseTensor 参数校验和底层 kernel 调用；`tensor.*` 提供编码 Tensor 的计算接口、投影及 CPU 回退。

原有编码 Tensor 实现及相关文件的当前位置：

| 原文件 | 当前文件 | 职责 |
| --- | --- | --- |
| `ops/norm/rms.cpp` | `ops/norm/dispatch.cpp` | 归一化精度、设备和融合路径分派 |
| `ops/projection/fp8_sq.cpp` | `ops/projection/fp8_sq/tensor.cpp` | FP8 SQ 编码 Tensor 的投影和 CPU 解码 |
| `ops/projection/mx.cpp` | `ops/projection/mx/tensor.cpp` | MX 编码 Tensor 的投影、查表和 CPU 回退 |
| `ops/projection/mxfp4_sq.cpp` | `ops/projection/mxfp4_sq/tensor.cpp` | MXFP4 SQ 编码 Tensor 的投影和 CPU 解码 |
| `ops/projection/nint.cpp` | `ops/projection/nint/tensor.cpp` | NINT 编码 Tensor 的投影、查表和 CPU 回退 |
| `ops/projection/vq.cpp` | `ops/projection/vq/tensor.cpp` | NVQ 编码 Tensor 的投影、查表和 CPU 回退 |
| `ops/projection/nvq_check.cu` | `ops/projection/vq/check.cu` | NVQ/NEPQ 数值检查 |
| `ops/projection/nint.h` | `ops/projection/nint/tensor.h` | NINT 编码 Tensor 计算接口 |
| `ops/projection/mx.h` | `ops/projection/mx/tensor.h` | MX 编码 Tensor 计算接口 |
| `ops/projection/fp8_sq.h` | `ops/projection/fp8_sq/tensor.h` | FP8 SQ 编码 Tensor 计算接口 |
| `ops/projection/mxfp4_sq.h` | `ops/projection/mxfp4_sq/tensor.h` | MXFP4 SQ 编码 Tensor 计算接口 |
| `ops/projection/vq.h` | `ops/projection/vq/tensor.h` | NVQ 编码 Tensor 计算接口 |

- 已同步生产 CMake 源码路径、生产和诊断 include，以及诊断源码中的路径注释。JIT 源码清单及 wheel 打包清单没有引用本次更名的文件。
- 通用 Hadamard 声明已从 `kernels/nepq.h` 移到 `kernels/hadamard/transform.h`，其他历史通用头文件也已更名。按“测试不要运行和修改”的要求，测试文件名和其中的旧引用保持不变，后续允许同步测试时再处理。
- 头文件下沉后，`tests/nint_row_pipeline_test.cpp` 和 `tests/decode_window_attention_test.cpp` 中的 `#include "nint.h"` 仍需同步；这里的测试路径相对 CUDA 后端目录。按测试冻结要求未修改这两处引用。
- 算子符号、MFQ 文件格式及编码标识保持原有名称。

本批验证：在 `build/csrc-layout` 中以 `BUILD_TESTING=OFF`、`TYLOINFER_ENABLE_MFQ_COMPAT=ON` 配置 CUDA 构建。四个实现头合并后的编译检查已通过；目录和命名调整后，受影响的 18 个生产编译单元全部编译通过。合并后的实现内容与原 include 展开内容一致；移动文件只同步了必要的 include 路径。未运行或修改测试，未进行完整链接和运行验证。

## 6. kernels 根目录平铺过多

状态：已处理（2026-11-16）；根目录从 326 个文件减为 13 个。

位置：`csrc/engine/backends/cuda/kernels/`。

- 根目录有 326 个 Git 跟踪文件，子目录仅有 `moe/`。
- 根目录包含 71 个 `gguf_*`、29 个 `mfe_*`、26 个 `nvq_*`、26 个 `gr_*` 和 14 个 `tq_*` 文件。
- 影响：目录主要依赖文件前缀分组，算子入口、算法变体、格式辅助代码和共享原语混排。
- 处理：按算子族收拢为 `attention/`、`projection/`、`ffn/`、`recurrent/`、`kv/`、`norm/`、`rope/`、`sampling/` 等目录；projection 内部再按格式分组。类型、warp 和跨算子的共享原语保留在根目录。

## 7. CUDA 构建和测试组织过于集中

- `csrc/engine/backends/cuda/CMakeLists.txt` 有 1303 行，约从第 442 行开始配置测试。
- CUDA `tests/` 根目录平铺 126 个 Git 跟踪文件，混有测试入口、fixture、测试片段和 benchmark。
- 例如 `kv_cache_bench.h` 由 `cuda_runtime_state_test.cpp` 包含，性能测试逻辑需要进入较大的测试文件才能发现。
- 影响：定位测试归属和调整源文件路径时，需要查阅较长的清单及包含关系。
- 建议：测试配置放入 `tests/CMakeLists.txt`；按 `native`、`ops`、`executor` 等实际测试对象归组。具有独立执行需求的 benchmark 再设置单独入口。

## 8. csrc 同时保留两套后端目录布局

- 当前同时存在 `csrc/backends/` 和 `csrc/engine/backends/`。
- CUDA 已位于 `engine/backends/cuda`；Metal 的新构建入口仍引用 `backends/metal` 下的 runtime、storage、models 和 tests 源码。
- HIP 仍位于 `backends/hip`，其中有 `engine/cuda_engine.cpp`、`engine/cuda_batching.cpp`、`ops/common/cuda_execution.cpp` 等历史名称。
- 影响：仅凭路径难以判断一个后端的完整实现范围，也容易误把旧目录视为可删除内容。
- 建议：按现有架构和实施计划逐步收敛归属；迁移时核对实际构建引用。Metal/HIP 的迁移范围沿用实施计划。

## 9. engine 根目录平铺多组不同职责的文件

状态：已处理，`mfq-tensor` 和 `mfq-engine` 构建通过（2026-11-15）。

原根目录有 30 个文件，包含公共算子、Tensor、内存预算和实例入口。现将同一职责的头文件与实现收拢为：

```text
engine/
├── engine.*、config.*、load.*、executor.*、program.*
├── metrics.h、CMakeLists.txt、cuda.cmake
├── ops/
│   ├── ops.h
│   ├── math.cpp、storage.cpp、projection.cpp
│   ├── rope.cpp、sampling.cpp、read.cpp、kv.cpp
│   └── 原有算子族接口
├── tensor/
│   ├── tensor.h、tensor.cpp
│   ├── dense.cpp
│   └── views.cpp
└── memory/
    ├── budget.h、budget.cpp
    └── kv.h、kv.cpp
```

- 16 个文件移入对应目录；公共算子实现移入现有 `ops/` 后去掉 `_ops` 后缀。
- `resource_metrics.h` 仅由 `engine.cpp` 包含，已将实现并回该 `.cpp` 并删除头文件。根目录文件从 30 个减少到 13 个。
- Executor 公共接口维持 `engine/executor.*`。算子、Tensor 和预算的类型、职责及编译目标保持原有语义。
- 已同步生产 include、CMake 源码路径和架构文档。按测试冻结要求保留测试中的旧 `engine/ops.h`、`engine/tensor.h` 引用，待允许修改测试时同步。
- 验证：以 `BUILD_TESTING=OFF` 构建 `mfq-tensor` 和 `mfq-engine`，包括所需的 CUDA native 依赖，均编译和静态库链接通过。移动文件仅同步 include 路径，资源指标实现与原 include 展开内容一致。未进行最终可执行文件链接或运行验证。

## 10. CUDA executor 根目录混有多组执行设施和数值实现

状态：源码归组已处理；局部编译通过，完整构建受已有错误限制。

原 `csrc/engine/backends/cuda/executor/` 根目录有 36 个文件。现按实际职责收拢：

| 职责 | 当前目录或文件 |
| --- | --- |
| 执行入口、配置、动态选项、工作集 | `executor/` 根目录的 `executor.*`、`config.*`、`options.*`、`working_set.*` |
| Graph 捕获、执行窗口和注册 | `executor/graph/decode.*`、`window.*`、`registry.*` |
| KV 分配和内存管理 | `executor/kv/allocation.cpp`、`memory.*` |
| QSA 缓存和会话 backing | `executor/qsa/ram_cache.*`、`tq_cache.*`、`session.h` |
| TQ GEMM 和 Graph 资源 | `executor/tq/resources.*` |
| TQ 专家计算 | `ops/ffn/tq/banks.*` |
| 混合专家数值批处理 | `ops/ffn/mfe/batch.cpp` |
| 采样算子 | `ops/sampling/sampler.h` |
| CUDA driver stream signal | `native/stream_signal.h` |

- 共移动 21 个生产文件。TQ 专家计算和混合专家批处理由现有 `mfq-cuda-ops` 目标编译，执行设施仍由 `mfq-cuda-execution` 编译。
- `moe_batch.h` 仅由原 `moe_batch.cpp` 包含，已合并至移动后的源文件。
- `residual_trace.h` 仅供 `ops/ffn/gated_residual.cpp` 使用，已并回该源文件。保留原有 trace 限制、错误处理和资源生命周期逻辑。
- 根目录从 36 个文件减少到 13 个，其中 8 个为执行入口和公共配置文件。
- 另外 5 个头文件仅供测试使用或被这些测试辅助文件直接依赖：`gdn_recorded_replay_batch.h`、`graph_pool_lease.h`、`qsa_indexer.h`、`recurrent_copy_batch.h`、`recurrent_copy_kernel.h`。按测试冻结要求保持内容和路径不变。
- 已同步生产 include 和 CMake 路径。测试中的旧路径引用保持不变，待允许修改测试时同步。QSA 缓存源码原本未纳入当前生产目标，本次保留其构建状态。
- 验证：`BUILD_TESTING=OFF`，检查了 29 个受影响的生产编译单元，27 个通过；本次移动的、已纳入生产目标的实现均编译通过。合并后的实现与原 include 展开内容一致，其他文件仅更名和同步 include。
- 完整 `mfq-cuda-ops` / `mfq-cuda-execution` 构建在未改动的 `ops/ffn/experts/shared.cpp` 遇到旧错误：依赖模板调用缺少 `template`，以及 `NintTensor` 与 `NintWeight` 参数不匹配。
- 两个调用方编译仍有旧错误：`models/qwen35/batch_ops.cpp` 使用了当前 `models::Batch` 不提供的 `Options` 等成员；`models/minicpmo45/duplex.cpp` 的 `ops::read_integers` 解析到错误命名空间。这两个文件本次仅同步 include 路径，相关类型、命名空间和调用内容未改动。
- 测试文件、上述 5 个测试辅助头和测试 CMake 配置均保持本批开始时的内容。未运行测试，未验证最终可执行文件链接和运行。

## 11. CUDA 与公共层的统一目录整理

日期：2026-11-16。范围：CUDA 和公共层；旧 Metal/HIP 目录不重组，仅同步其中显式引用共享文件的路径。第三方源码和测试保持原有内容。

状态：本轮目录整理完成，源码、构建和打包引用已同步；最终 GGUF 分组的 CUDA 内核库编译通过。

按职责收拢接口和实现，并去掉目录已表达的文件前后缀。同一个模型、同一个容器接口内部的文件维持其模块归属。下表为本批完成时的统计，后续 Python 包内 CUDA 源码迁移见第 12 项。

| 目录 | 整理前根目录文件数 | 当前根目录文件数 | 分组 |
| --- | ---: | ---: | --- |
| `engine/backends/cuda/kernels` | 326 | 13 | attention、projection、ffn、recurrent、kv、norm、rope、sampling、embedding、elementwise、hadamard 等算子族 |
| `engine/weights` | 47 | 14 | cache、prepare、rows |
| `engine/weights/moe` | 40 | 10 | residency、host、prepare |
| `engine/weights/encoding` | 42 | 15 | dense、mx、sq、tq、quant、qsa、readers |
| `diagnostics` | 49 | 3 | checks、models、runtime、data、profiling |
| `engine/state` | 22 | 4 | kv、recurrent，保留已有 session |
| `engine/models` | 21 | 13 | views；ModelSource 接口和工厂归入已有 source |
| `engine/generation` | 18 | 7 | batch、prefill，保留已有 mtp |
| `engine/backends/cuda/ops/ffn` | 26 | 11 | experts、tq、mfe |
| `engine/backends/cuda/kernels/projection/gguf` | 73 | 16 | dot、compact、dense、quad、mmq、prefill、residual、routes |

- 共整理 540 个源文件的位置和名称。CPU backend 下由共享编码目标编译的 QSA codec、mirrored reader 实现与对应头文件合并到同一目录；编译目标、数值算法、类型、校验和资源生命周期保持原有语义。
- Tensor 层 CUDA 声明归入对应 `ops` 目录。通用 `mfq_cuda_*` 文件前缀已清理；MFQ 文件格式、符号及编码名称保持原有语义。
- 已同步所有生产 CMake 源码路径、生成 MMQ 模板引用、CUDA JIT 清单、源发行包清单和 Python TQ 资源查找。
- TQ 源码与其他 CUDA JIT 源码共用 wheel 内的 `_src/csrc/...` 目录结构。JIT 的 163 个源码及引用文件闭包检查通过，TQ 的 14 个本地源码依赖在暂存目录中均可解析。
- wheel 构建通过。源码发行包构建通过，解包后可再次解析完整 JIT 闭包并找到 TQ 源码。
- 2086 个 C/C++ 文件的实现内容与整理前一致，差异仅为必要的 include 路径；562 个测试及指定测试辅助文件保持原内容，没有运行测试。
- 公共 Tensor、Engine 和 CUDA 执行设施的生产编译单元通过编译检查；CUDA 内核库在最终 GGUF 分组后重新编译通过。组合构建仍在 `ops/ffn/experts/shared.cpp` 遇到第 10 项记录的旧模板和类型错误，未修改相关业务逻辑。Qwen35 batch、MiniCPM-o duplex 的旧接口错误也维持原记录，未验证最终可执行文件链接和运行。
- 本轮未将目录移动视为职责迁移完成。第 1 项的 GGUF 缓存/计算混合、第 3 项的私有实现跨模块依赖，以及旧后端的迁移仍需按实际接口处理。

## 12. Python 包内仍保留 CUDA 原生源码

日期：2026-11-16。范围：仅 CUDA；HIP 和 Metal 源码及加载方式保持原样。

状态：63 个 C++、CUDA、头文件和 include 片段已迁出 `tyloinfer/kernels/cuda`。该目录在源码树中只保留 Python 接口、加载器和源码清单。

- 40 个 PyTorch 绑定文件归入 `csrc/bindings/cuda`，按 attention、elementwise、recurrent、projection 和模型接口分组。绑定层位于原生 engine 之外，生产 CMake 目标不增加 Torch 依赖。
- 23 个设备实现和原始接口文件归入已有 `csrc/engine/backends/cuda/kernels` 算子目录，包括 JIT 内核、反向计算、MMA 头文件和 GDN include 片段。保留独立 JIT 实现的数值语义，未用原生实现替换。
- 63 个文件的实现内容与本批开始时一致，仅调整文件位置、名称及必要的 include 路径；源码编译清单及顺序保持不变。
- 已同步 Python JIT 加载器、源码根目录探测、源码包清单和 wheel 收集逻辑。所有 JIT 原生源码统一从 `csrc` 读取，wheel 内统一暂存为 `_src/csrc/...`，不再依赖包目录中的平铺源码。
- Torch JIT 以源文件名生成平铺目标文件。不同算子目录中的 `matmul.cu`、`jit.cu`、`forward.cpp` 和 `backward.cpp` 会发生冲突；加载器在 Torch 构建缓存中生成带序号的编译入口，直接 include 原文件，并携带原文件内容哈希以保持增量重编译。仓库不保留第二份 kernel 实现。
- 163 个 JIT 源码及引用文件可完整解析。迁移的 14 个 CUDA 编译单元使用原 JIT 编译选项全部编译通过。
- 最终路径的 wheel 和源码包构建通过，两者均包含与源码树一致的 163 个 JIT 依赖文件；wheel 解包后可通过 Python 加载器定位全部编译源文件与 TQ 源码。61 个编译入口在 Torch 的目标文件命名规则下互不冲突；入口生成的重复调用、源码变更检测和实际 CUDA 编译检查通过。
- 当前 Python 环境未安装 Torch，完整 PyTorch 扩展编译、链接和设备运行未验证；测试未运行、未修改，已有测试源码路径断言保留原样。

## 后续整理顺序与关联范围

第 4、5、6、9、10 项的生产文件整理已完成，第 11 项覆盖其他 CUDA 与公共层目录，第 12 项统一 CUDA JIT 和绑定源码位置。后续在允许修改测试时同步测试命名和旧引用；完整构建仍需解决第 10 项记录的已有接口错误。目录归组不替代第 1、3、8 项的接口和迁移工作。

每批移动或重命名需要同步检查：

- CMake 源文件列表、include 路径和生成源码模板。
- C++ 调用者、测试、fixture 及诊断工具。
- CUDA JIT 源码清单 `tyloinfer/kernels/cuda/_sources.py`。
- wheel 源码收集逻辑 `packaging/python/build.py`、`MANIFEST.in` 和相关打包测试。
- 记录源码路径的文档与源码结构断言。

已有陈旧引用示例：`tests/test_kernel_package_data.py` 仍断言旧路径 `csrc/backends/cuda/kernels`，并读取当前 `_ext.py` 已不提供的 `_KERNEL_DIR` 等字段。整理时应同步这些消费者。
