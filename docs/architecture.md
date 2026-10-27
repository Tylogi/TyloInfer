# TyloInfer 架构

本文记录已确认的目标目录与 TyloInfer `refactor/arch` 的迁移现状。职责约定见
[refactor.md](refactor.md)，剩余工作及验收见 [plan.md](plan.md)。当前实现说明
以 `c68c707` 为源码核对基点，包含当前工作区的 TQ 源码迁移；完整 native runtime
与服务尚未验收。下述目标目录
尚未整体落地；`source/` 和 CUDA native 目录已归位，`backends/` 的其余职责仍待迁移。

## 最终目标目录

按职责组织源码，只有设备原语与内核固定按后端分目录：

```text
csrc/
├── engine/
│   ├── requests/       # 请求、媒体输入、约束、结果
│   ├── models/         # family 配置与算子组合
│   ├── state/          # KV/递归状态、快照、恢复、生命周期
│   ├── weights/        # 权重加载、驻留、预算、租约
│   ├── batch/          # 批次组织与 prefill/decode 计划
│   └── executor/       # 执行推进、MTP、Graph、执行资源所有者
├── ops/                # 算子数学与实现，按计算职责组织
│   ├── attention/
│   ├── ffn/
│   ├── projection/
│   ├── norm/
│   ├── embedding/
│   ├── sampling/
│   └── vision/
├── kernels/            # 设备计算内核
│   ├── cuda/
│   ├── metal/
│   ├── hip/
│   └── cpu/
├── native/             # tensor、分配、复制、同步、设备通信
│   ├── cuda/
│   ├── metal/
│   ├── hip/
│   └── cpu/
├── formats/            # 格式定义、编码布局、解码
├── source/             # 模型输入、metadata、tensor inventory
├── components/         # I/O、JSON、tokenizer 等基础模块
├── scheduler/          # 逻辑准入、排队、取消、背压
├── transport/          # 协议解析与输出
├── apps/               # 可执行入口与依赖装配
├── commands/           # 命令处理
├── eval/
├── diagnostics/
└── tests/
```

`model_source/` 简化为 `source/`，目标结构取消 `backends/` 与 `third_party/`
聚合目录，也不设置顶层 `storage/`、`execution/`。source 的 22 个文件及 CUDA/公共
生产 include、CMake 路径已迁移，沿用 `mfq-model-source` 目标。重复的
`csrc/third_party/strata.LICENSE` 已删除，源码许可引用统一指向
`LICENSES/Strata-MIT.txt`，原有 `NOTICE` 与许可证内容保留。

`state`、`weights` 各自共享策略和生命周期，设备载荷允许使用不同具体类型；
不预设 `cuda/`、`metal/` 平行子目录，也不为目录统一新增通用 Tensor 或插件表。
必要的设备实现差异留在所属模块内部，只有实际复杂度需要时才局部拆分。
`ops` 按算子职责组织，模型组合留在 `engine/models`；设备无关的 attention、MLP
数学从模型公共辅助代码中归入 `ops`。`native` 不管理模型、请求或驻留策略。

依赖方向为 Engine 使用 ops，ops 使用 kernels/native；算子不依赖 Engine 驱动、
加载器或状态/权重管理者。加载链为 source 使用 formats 和 components/io，权重
所有者装配计算所需的数据。设备专有编译依赖通过各自 CMake 目标隔离。

该结构覆盖最终各后端；迁移先处理 CUDA 及必要公共模块，Metal/HIP 仍按后续阶段
推进。它们未迁移前保留的旧路径不代表目标结构仍采用 `backends/`。

## 产品与数据流

```text
TyloInferStudio → HTTP / WebSocket → tyloinfer/server
                                      ↓ 本地 native worker
                             Transport → Scheduler → Engine
                                                       ↓
                                               CUDA ops / kernels

模型文件 / TQ 权重 → source → 权重加载与模型装配
```

Python 包和 CLI 为 `tyloinfer`，入口见 [pyproject.toml](../pyproject.toml)。
`tyloinfer/server/` 管理 API、任务与 native worker；`formats/`、`architectures/`
提供格式及模型配置，`runtime/`、`evaluation/`、`tools/` 保留运行和评测工具。
离线量化、校准不是 TyloInfer 的产品入口。

`Runtime = Transport + Scheduler + Engine` 表示装配职责，没有独立的顶层
Runtime 类或 `csrc/runtime/` 模块。CUDA 可执行入口位于 `apps/`、`commands/`；
已有 CMake 目标及可执行文件仍使用 `mfq-` 前缀。

## 公共模块与请求执行

以下为当前路径，相对 `csrc/`。公共层已按职责拆分，Engine 当前包含六个目录。

| 模块 | 当前职责 |
| --- | --- |
| `components/io/` | 文件范围读取、异步读取和读统计 |
| `formats/` | 格式定义、模型图 schema、量化行与范围解码 |
| `source/` | tensor inventory、metadata、输入读取与兼容名称规范化 |
| `backends/cpu/` | host 内存/线程、CPU 算术与执行资源、RAM/SSD 物理存储 |
| `engine/requests/` | 请求、媒体、文本准备、约束、输出与请求表 |
| `engine/models/` | family 配置、共享数学组合及现有模型适配约定；已迁移的 Qwen35 MTP predictor 只生成 `Task` |
| `engine/state/` | 页分配、prefix/session、快照与序列状态规则 |
| `engine/weights/` | 缓存策略、路由排序、dispatch plan、TP 与成本模型 |
| `engine/batch/` | 连续批处理、prefill 预算及 batch MTP 策略 |
| `engine/executor/` | Engine 接口、装配模板、请求推进、采样与 MTP 流程；CUDA `Executor` 只分发固定算子 `Task` |
| `scheduler/`、`transport/` | 逻辑准入、取消和输出队列 / HTTP、stdio 协议 |

公共 Engine 接口位于 [executor/engine.h](../csrc/engine/executor/engine.h)，
请求值位于 `engine/requests/`；`mfq-engine` 当前公开整个 `engine/` include 根。
Scheduler 调用 `Engine::step()`，再发布返回的 owned typed events；终态为
`Completed`、`Cancelled`、`Failed`。

当前已闭合的 CUDA 执行切口：`engine/models/qwen35/predictor.*` 生成
`std::vector<Task>`，`engine/executor/run.*` 以 `Inputs` 执行固定的 Norm、Concat、
Project、Attention、Dense；Qwen35 MTP 的 predictor、request 和 state 路径不再接收
`CudaExecutionContext`。`Dense`、`FullAttention`、`LinearAttention` 算子也通过
显式 Projector、state view 和执行资源工作，不依赖该上下文。

CUDA 由 `EngineInstance<CudaBackend>` 装配。共享 `ContinuousBatch` 已被
[batch/batch.h](../csrc/backends/cuda/engine/batch/batch.h) 中的
`QwenBatchOperations`、`FlashNextBatchOperations` 使用，但两者仍有不同的模型
状态适配与 Graph 实现。Duplex 通过 `ControlRequest` 和 `Engine::control()`
进入后端，其生成、取消和背压仍需与普通请求流程统一。

## CUDA 模型、算子与资源

CUDA context、tensor、环境配置及 allocation/matmul 辅助代码已迁至
`csrc/native/cuda/`，与先前迁移的 cuBLAS 句柄、分支调度、collectives 和 profiler
使用同一目录。旧 `csrc/backends/cuda/native/` 已删除。生产 include、CMake
源码与 include 根均已同步；Python JIT 的 native include、tensor include 和 env
源码路径也已更新。仍沿用现有 CUDA 目标与编译配置，没有旧位置转发头。
资源耗尽异常现定义在 `csrc/native/error.h`，CUDA 分配器不再反向 include
Engine 头；请求与传输层共享同一异常类型，原有捕获和错误映射保持不变。

共享 CUDA kernels 的 140 个文件已迁至 `csrc/kernels/cuda/`，生产 include、
CMake 和 Python JIT 路径已同步。旧 `csrc/backends/cuda/kernels/` 仅保留
`deepseek_v41.h`、`deepseek_v4_attention.h`、`deepseek_v4_hc.h` 三份原始声明，
供 CUDA/HIP 共用；待 HIP 消费者一并迁移后再归位。

以下未迁移路径相对 `csrc/backends/cuda/`。`ops/` 提供算子；`engine/` 已承载 state、weights、
batch、executor 和部分模型装配，顶层 `models/` 仍保留未迁移的 family 实现。
CUDA 已没有原来的 `core/`、`runtime/`、`storage/`、`execution/` 目录。

上述是当前布局。CUDA 状态和权重虽已落入后端 `engine/`，仍需与公共实现一起按
职责归入顶层 `csrc/engine/`；其 `ops` 也须迁入顶层对应模块。
legacy 模型装配随消费者迁移收口，见 [迁移计划](plan.md)。

通用线性加载位于 `engine/weights/load.*`，线性计算位于
`ops/common/quant_linear.cpp`。TQ/MoE 的物理加载、缓存和驻留位于
`engine/weights/tq/`、`engine/weights/moe/`，而 `ops/tq/` 只保留计算实现；这个
分离尚未覆盖全部调用边界，`engine/models/load.*` 和主 FFN 仍接收
`CudaExecutionContext`。

MoE pipeline 的实现已在 `engine/executor/moe_ffn_pipeline.*` 中移除
`CudaExecutionContext` 参数：共享回调接收 Projector，构造时绑定 profiler、MMQ 和
类型化配置，运行时显式传入 batch serial/tile 参数。其 registry、drain 和全局资源
所有权仍在 `CudaExecutionContext`；这些借用资源不是完整的上下文移除，后续不得再
扩大该上下文接口。

`644585d` 已完成 RouteStats 抽离，当前实现已归入
`csrc/diagnostics/route_stats.{h,cpp}`，不再位于 native。记录/写出接口不接收
`CudaExecutionContext`，GPU 统计张量、layer map、输出路径和 energy
选项由 `RouteStats` 持有；但 `CudaExecutionContext` 仍持有 `routes`，主
`engine/executor/ffn.cpp` 仍通过它记录，`eval/kl.cpp` 仍通过它清空/写出，
`engine/models/load.cpp` 仍通过它判断启用。因此这是一个已完成的 leaf extraction，
不是全局 Context 移除。

`449566f` 将 Qwen4 的 `Gr` 数学移入 `ops/common/gated_residual.*` 的
`GatedResidual`。算子持有 norm、投影权重和维度，加载时绑定投影执行依赖。
`bdd8efb` 将 `pre(input)` 的结果改为具名的 `mixed`、`residual`、`gate`，支持
结构化绑定；`pre_after(input, residual, gate)` 和 `post(input, residual, gate)`
显式接收计算张量。CUDA 模型层直接组合 GR、Attention 与 FFN，延后回写使用
可选的 GR 结果，共享模型模板及其他后端的调用约定保持原样。
普通前向、MTP、batch 和 graph 调用均已迁移；原 Context 参数与长参数函数已删除。
`f21dfb2` 将 TQ 投影改为绑定 Projector 后直接计算，删除每份权重的 Executor
及临时 Task/Inputs。Projector 借用现有 owner 的原生资源与类型化配置，主设备单独
绑定；MoE/NINT 动态选项在实际投影时读取，诊断配置和 graph 标志保持实时生效。
模型须先于资源 owner 销毁。GDN/QSA 的 Context 参数、加载器及根资源迁移尚未完成。

共享 [CausalModelBase](../csrc/engine/models/common/causal_model.h) 仍继承后端与
`CausalState`。CUDA 的 `CausalResources`、`CudaCausalOps::ForwardOps`、
`models/common/block.h` 和 `FullBlock` 均仍在使用。Qwen35 通过共享
[causal_lm.h](../csrc/engine/models/qwen35/causal_lm.h) 与 CUDA
[ops.h](../csrc/backends/cuda/models/qwen35/ops.h) 装配，尚未改为外部 StateStore
提供显式状态的模型。

| 当前所有者或路径 | 已有实现与剩余耦合 |
| --- | --- |
| `engine/executor/engine.cpp` | `CudaBackend` 持有 `CudaStores`、`CudaExecutionContext` 和模型；卸载先 drain，再释放模型、stores 和 execution |
| `engine/executor/execution.*` | stream、通信、scratch、配置及缓存访问仍集中在管理上下文；MoE registry/drain 也仍在此处，资源借用边界需收窄 |
| `csrc/diagnostics/route_stats.*` | RouteStats 独立持有 GPU 统计张量、路径和 energy 选项；`CudaExecutionContext` 仍持有其 `routes` owner |
| `models/common/full_block.h` | `FullBlock` 同时持有权重、KVCache 和 decode scratch；reset、预分配与 forward 仍在同一对象 |
| `engine/batch/qwen35.*`、`models/qwen4_exp/ops.cpp` | 请求状态已有独立载荷，但仍通过 adapter 绑定或交换到模型 |
| `engine/state/` | KV、Recurrent、PagedKv、预算、VMM 与 offload 已由独立 owner 承载；统一状态身份和事务尚未覆盖全部模型消费者 |
| `engine/state/session/`、`models/common/session_codec.h` | snapshot 载荷与捕获/恢复仍依赖具体 family 和 block 成员 |
| `engine/weights/moe/` | 专家 backing、读取、缓存和驻留交换已集中；装配与执行仍跨越旧上下文 |
| `engine/models/components.*` | 模型加载、MTP 和多模态装配；grid vision 的具体计算与 prompt 组件仍在这里 |

公共 `csrc/engine/state/` 已有页、快照和 prefix 规则，尚未提供统一 StateStore；
`csrc/engine/weights/` 也还不是统一 WeightStore。原生内存管理与已有缓存是后续
迁移的基础，不能用新增空接口替代实际消费者接入。

环境读取已集中到 `csrc/native/cuda/env.*`，Engine 加载时调用 `env_freeze()`，并有
`tests/native/env_test.cpp`。线性算子已使用 `ops/common/include/quant_linear.h`
定义的 `InputGateMode`。后续工作应复用这些入口，检查配置传播与原始算子接口。

## 构建与源码来源

[顶层 CMake](../csrc/CMakeLists.txt) 已接入独立的 I/O、format、model source、CPU、
Engine、Scheduler 和 Transport 目标。[CUDA CMake](../csrc/backends/cuda/CMakeLists.txt)
将 `mfq-cuda-ops`、`mfq-cuda-execution`、`mfq-cuda-loading` 对象目标与具体
Engine/model 实例一起组成 `mfq-cuda-runtime`；native tensor 目标仍名为
`mfq-cuda-core`。这些目标拆分尚不意味着加载和计算的头文件依赖已完全解耦。
CUDA production source list 已包含 `ops/common/expert_projection.cpp`，并从顶层
`csrc/diagnostics/route_stats.cpp` 编译路由统计；这只说明源码接线，不表示构建已验证。

默认 native 构建包含浮点/FP8/MX 与 TQ；旧 MFQ 算子通过
`TYLOINFER_ENABLE_MFQ_COMPAT=ON` 开启。CUDA CMake 已将计算、执行和加载目标
与可选通信分开，服务可执行入口仅在通信目标存在时创建；通信关闭的实际构建
仍待验证。Torch reference runtime 仍显式要求通信开启。

共享 CUDA kernel 实现位于 `csrc/kernels/cuda/`，Python
`tyloinfer/kernels/cuda/_ext.py` 仍从这里编译原有的源码子集。CMake 的
`OBJECT_DEPENDS`、逐文件数值选项及寄存器设置均保持原样。TQ 的 13 个源码文件
`tq_ops.*`、`tq_*.cuh` 也已迁至 `csrc/kernels/cuda/`，由 native 与源码运行时的
`tq.py` 共用。`packaging/python/build.py` 负责 Python 发行包装配：将 Python 包、
TQ 资源、元数据和许可证放入临时发行目录，再交给普通 setuptools 构建 wheel。
TQ 资源在发行目录中归入 `tyloinfer/kernels/cuda/`；安装后的 JIT 从包内加载，
editable 安装读取仓库源码。Windows/macOS 自包含发行脚本安装该 wheel 后，
由 PyInstaller 收集其包和资源，不再自行拼接 TQ 资源。
通用 Torch 扩展仍显式编译 MFQ 兼容算子，不等于默认 native 配置。

保留的三份 DeepSeek 共享声明仍需随 HIP 消费者归入 `csrc/kernels/cuda/`。

构建迁移尚有源码可见的遗漏：部分 host/CUDA 测试和 bench 仍链接已无目标定义的
`mfq-core`，部分 Python 架构测试仍读取旧目录。具体修复与验证顺序见
[plan.md](plan.md)，构建用法见 [cli/build.md](cli/build.md)。
