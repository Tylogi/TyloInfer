# TyloInfer CUDA 重构计划

本计划按 TyloInfer `refactor/arch` 当前源码整理，核对基点为 `c68c707`，
包含当前工作区的 TQ 源码与打包路径迁移。
只维护剩余任务与验收；目标边界见 [refactor.md](refactor.md)，当前模块见
[architecture.md](architecture.md)。最终目录采用职责优先的
[目标结构](architecture.md#最终目标目录)。`source/` 生产路径已归位，冗余的
`third_party/` 已删除，后续继续迁移 `backends/`。实现范围仍为 CUDA 及必需公共模块；
Metal/HIP 源码及构建保持原样。CUDA CMake 和 JIT 的源码、include 路径随迁移
同步，Python 算子与执行逻辑、测试实现不改。

公共模块和 Engine 六个职责目录已有实现；CUDA 的 state、weights、batch 和 executor
已落在 `engine/`，顶层 `models/` 仍保留过渡装配。后续结合依赖、加载边界和实际
消费者完成迁移。`models/common/block.h` 仍存在，通用 Block、模型内状态和管理上下文
仍在生产调用链中；显式 StateStore/WeightStore 是待接入的目标，不能按参考分支中已有
同名文件假定 TyloInfer 已实现。

本次源码核对未执行构建或数值测试。下列源码事实用于安排工作，不能据此宣称完整
runtime 已通过或已失败，也不沿用参考分支的失败数量与历史测试结果。

截至上述基点，最近实现批次涉及 CUDA 生产源码、必要公共模块、许可证路径、
CMake 和 CUDA JIT 路径。CMake 中直接引用迁移文件的路径已同步；测试实现、
Metal/HIP 和 Python 算子与执行逻辑保持原样，未运行构建或测试。下列构建、
测试和 JIT 验证均延期，不是本轮的完成条件。

## 当前进展

source 目录、CUDA/公共生产调用方与 CMake 路径已迁移；重复许可证已删除，
许可内容与 NOTICE 保留。CUDA native 已整体迁至 `csrc/native/cuda/`，包括
context、tensor、env 及先前迁移的通信资源。生产 include、CMake 和 CUDA JIT
路径同步，旧 native 目录已删除。资源耗尽异常已归入
`csrc/native/error.h`，解除 CUDA 分配器对 Engine 头的反向依赖；RouteStats 已归入
`csrc/diagnostics/`。140 个共享 CUDA kernel 文件已迁至 `csrc/kernels/cuda/`，
CMake 与 JIT 路径同步；三份 DeepSeek 共享声明因 HIP 消费者仍保留原位，
TQ 仍由 Python 包分发。上述目录迁移的整体构建与运行验收仍延期。

当前工作区已将 TQ 的 13 个 `tq_ops.*`、`tq_*.cuh` 文件原样迁入
`csrc/kernels/cuda/`，同步 native CMake、C++ include 与 JIT 路径。
`packaging/python/build.py` 统一装配临时 Python 发行目录并生成 wheel，根目录不新增
`setup.py`、`MANIFEST.in` 或 setuptools 自定义命令。源码/editable 与 wheel 的
JIT 分别读取仓库和包内路径，自包含发行脚本消费该 wheel。测试文件保持原样，
本批后续不修改或运行测试；完整 native runtime 与发行包验收仍延期。

截至 `c68c707`，已完成以下执行切口：

- `8ca567f`：Qwen35 MTP predictor 生成 `std::vector<Task>`；CUDA `Executor` 以
  `Task`/`Inputs` 分发固定算子，MTP predictor/request/state 路径不再接收上下文。
- `7364bf5`：专家 projection 使用 Projector，TQ topology slices 提前加载；旧的
  executor/experts 文件及其 Context-taking API 已删除。
- `ffd9332`：MoE FFN pipeline 不再接收 `CudaExecutionContext`；共享回调使用
  Projector，profiler/MMQ/类型化配置在构造时绑定，batch serial/tile/tiles 按次传入。
- `644585d`：RouteStats 抽离，统计张量、layer map、输出路径和 energy 选项由
  无 Context 的 `RouteStats` 持有；当前源码位于 `csrc/diagnostics/route_stats.*`，
  `CudaExecutionContext` 仍持有 `routes` owner。
- `449566f`：Qwen4 的 `Gr` 移入 `GatedResidual` 算子，持有权重并在加载时绑定
  执行依赖；前向、MTP、batch 和 graph 调用只传计算张量及可选 marker。
  旧 GR 长参数函数和 `ResidualLinear` 已删除；配置与 batch serial 标志仍借用现有
  owner，普通 Linear/Routed 回调和加载器仍待迁移。
- `bdd8efb`：GR 的 `pre` 返回具名 Tensor 结果 `mixed/residual/gate`；`post` 和
  `pre_after` 显式传入 `input/residual/gate`。CUDA Qwen4 前向、MTP、graph、batch
  和分块 prefill 已迁移，移除模型调用方的 GR 数字下标协议。
- `f21dfb2`：GR 的 TQ 投影直接调用绑定后的 Projector，删除每份权重的 Executor
  与临时 Task/Inputs。Projector 借用类型化配置，动态选项在使用时读取；主设备
  从配置中分离，Dense/TP/EP 调用已同步。GDN/QSA 的 Context 参数仍待迁移。

这些切口不等于主链完成：MoE registry、drain 和根资源所有权仍在
`CudaExecutionContext`；`ffn.cpp`、主模型与加载/装配仍大量使用该上下文，
`Block`、`CausalModelBase`、模型元数据及 Qwen 主任务路径尚未完成迁移。Dense、
FullAttention、LinearAttention 算子已是显式资源接口，但不能据此宣称全局上下文已移除。

RouteStats leaf extraction 已完成，但全局 Context 移除仍未完成：主 `ffn.cpp`、
`eval/kl.cpp` 和 `engine/models/load.cpp` 仍通过 `CudaExecutionContext` 访问
`routes`，主 FFN 也仍 Context-bound。

## 按职责组织的目标目录与迁移方式

最终仅保留一套 `csrc/engine/` 职责模块，将 CUDA 实现与共享规则放进相应职责。
`state`、`weights` 不预设 `cuda/`、`metal/` 子目录；设备载荷可保留不同具体类型，
实际差异留在所属模块内部，不复制生命周期或策略。`ops/` 按计算职责组织，只有
`kernels/`、`native/` 固定按设备分目录。下表来源与目标均相对 `csrc/`。

| 现有来源 | 目标归属与职责 |
| --- | --- |
| `engine/state/`、`backends/cuda/engine/state/`、模型内状态和 session codec、CPU 状态存储 | `engine/state/`：统一 KV、递归状态、快照、恢复及生命周期，包含各设备/RAM/SSD 载荷的内部实现 |
| `engine/weights/`、`backends/cuda/engine/weights/` 与算子内加载和缓存管理 | `engine/weights/`：统一加载、预算、路由、驻留、租约、物理映射与传输 |
| `engine/models/`、`backends/cuda/models/` 与后端 `engine/models/` | `engine/models/`：按 family 组织配置、权重角色和算子组合；通用数学提取到 ops |
| `engine/batch/` 与 `backends/cuda/engine/batch/` | `engine/batch/`：统一批次组织及执行适配，复用 `ContinuousBatch` |
| `engine/executor/` 与 `backends/cuda/engine/executor/` | `engine/executor/`：请求执行、MTP、Graph 和执行资源 Owner，保留 `Task`/`Inputs` 接口 |
| `backends/*/ops/` 与 `engine/models/common/` 中可复用的算子数学 | `ops/`：按 attention、FFN、projection 等职责组织，设备差异在算子内部表达，不引入通用 Tensor 或动态后端注册层 |
| 尚未迁移的 `backends/*/kernels/` 与 CPU 计算内核；TQ CUDA 源码已归位 | `kernels/cuda/`、`metal/`、`hip/`、`cpu/`：设备计算内核；native/JIT/wheel 共用唯一源码 |
| 尚未迁移的后端 native 与通用 host/设备原语 | `native/<device>/`：CUDA 目录已迁移；其他设备后续归位，不接管状态和权重管理策略 |
| 后端 `apps/`、`commands/`、`eval/`、`diagnostics/`、`tests/` | 顶层同名职责目录；入口组合依赖，测试和工具覆盖相应模块，不复制执行循环 |

最终不保留 `backends/`、`model_source/`、`third_party/` 或独立 storage/execution
职责层。算子借用低层资源和显式 view，不依赖 Engine 驱动及 state/weights 管理者。
公共请求处理直接复用 `engine/requests/`，不为目录对称创建空实现。

source、CUDA native、140 个共享 kernel 文件及 13 个 TQ 文件已迁移，
生产消费者、CMake、JIT 与 TQ 打包路径已同步。下一批迁移 ops。
三份 DeepSeek 共享声明留待 HIP 消费者一并调整后归位。
随后随第 2—5 节合并 Engine 对应职责及其调用方。
Metal/HIP 仍在后续阶段迁移，期间旧路径只作为迁移来源，
不重建另一套共享层。每批同时调整所有权、调用方、include、CMake、JIT/打包
清单和测试，构建并验证实际消费者
后删除旧文件与目录；不留转发头、源码副本或第二条执行链。后文旧路径均表示迁移
来源，最终落点以本表为准。设备专有头和编译依赖必须由独立 CMake 目标隔离，
不能因目录归并混入 host 或其他设备构建。当前 tests/bench/JIT 的延期项在对应
迁移验收前补齐，不将仅更新生产 source list 当成目录迁移完成。

## 1. 后续：构建与测试依赖迁移（延期）

本节只记录后续需要处理的依赖问题；本轮不修改 tests/bench，不运行构建或 CTest。

| 当前证据 | 剩余工作 | 验收 |
| --- | --- | --- |
| 顶层已取消 `mfq-core` 目标定义；`csrc/tests/CMakeLists.txt` 的 `mfq_add_core_test`、部分 bench，以及 CUDA 测试仍链接它 | 按真实依赖链接 `mfq-engine`、`mfq-model-source`、format、I/O 和 CPU 目标；清理失效 include 根，不恢复聚合目标 | host 通信 on/off、CUDA 默认/兼容配置的受影响目标重新构建并运行对应 CTest |
| `tests/test_cpp_runtime_architecture_boundaries.py` 仍读取 `engine/include/mtp_policy.h` 等旧路径 | 更新第一方测试、bench、JIT 清单与 include；删除或改写只断言旧继承/目录形状的测试，保留数值及生命周期判据 | 相关 Python 测试能收集并运行，所有引用路径可解析；测试失败不能用跳过旧路径掩盖 |
| CUDA CMake 已分开计算/加载与可选通信；服务入口按通信目标创建 | 迁移目录时保持目标依赖隔离，核对通信关闭路径的实际消费者 | 完整 CUDA 服务在通信开启时构建；独立 CUDA 数值目标在通信关闭时构建；Torch reference 的通信要求保持显式 |
| CUDA native、共享 kernel 和 TQ 的 CMake/JIT 路径已更新，TQ wheel 从唯一源码收集资源；ops 与三份 DeepSeek 共享声明仍在旧路径 | 随剩余源码迁移更新 source list、include 根与 JIT，并补齐整体构建及安装验收 | 默认/兼容 native 与相关 JIT 构建通过，TQ wheel 安装后可加载内核资源；不保留手工同步的源码副本 |

后续进入验证阶段时，先用当前代码和固定输入建立对照。若发现真实构建或数值失败，记录目标、配置、
输入与命令并修复相应依赖，不能将“目录已拆分”作为阶段完成证据。

## 2. 完成 TQ/MoE 加载与算子边界

通用线性加载已在 `engine/weights/load.*`，计算在
`ops/common/quant_linear.cpp`。接下来覆盖以下仍耦合的路径；本节 CUDA 路径相对
`csrc/backends/cuda/`，仅表示当前来源；目标均位于顶层 `csrc/` 对应职责。

- `ops/tq/tq.cpp`、`ops/tq/include/tq.h`：继续收窄算子与加载边界；TQ 权重加载和
  专家缓存管理已在 `engine/weights/tq/`、`engine/weights/moe/`，格式解码归公共
  `csrc/formats/`，输入读取归 `csrc/source/`，加载与 backing 归顶层
  `csrc/engine/weights/`，模型装配归 `csrc/engine/models/`。
- `engine/models/load.*`、`engine/executor/ffn.*`、`engine/weights/moe/`：分清加载参数、
  完整 FFN 数学和缓存执行选择；只在全部
  消费者迁移后删除旧入口。
- `engine/models/components.*`：将视觉数学交给 `csrc/ops/`，family 装配交给
  `csrc/engine/models/`，可执行依赖装配归 `csrc/apps/`；保留现有 TQ、分片、映射和 CPU
  offload 行为。
- `engine/models/common/attention.h`、`gated_mlp.h` 等公共数学按实际职责提取到
  `csrc/ops/`，消除算子对模型公共辅助代码的反向依赖；family 的层序与配置仍归 models。
- `csrc/native/cuda/env.*` 已提供严格读取、冻结和覆盖接口，线性算子也已使用
  `InputGateMode`。复用这些入口检查配置传播；继续分离推理与 tool/backward
  kernel，收窄 activation/embedding/KV 原始接口。

后续验收：默认 TQ 与显式 MFQ 兼容加载分别构建并对照原始张量；ops 不再依赖模型
文件读取。覆盖映射保活、分片、非法输入、环境冻结/覆盖及加载失败清理。native
和实际受影响的 Python JIT 冷构建通过，TQ wheel 的源码资源可安装和加载。

## 3. 收口执行与权重所有权

以已有 `CudaBackend → CudaStores / CudaExecutionContext` 和先 drain 再卸载的
流程为起点，不重复创建第二套 Owner；`CudaExecutionContext` 不是目标接口。
`engine/executor/moe_ffn_pipeline.*` 已完成 Context-free pipeline 切口，
`engine/executor/execution.*`、`engine/executor/tq.*` 中的执行资源管理仍需收口；
目标中资源 Owner 归 `csrc/engine/executor/`，通用设备原语归 `csrc/native/`，
完整数学归 `csrc/ops/`。可写
scratch、stream/event、collective 和分支资源必须有明确借用与回收契约，算子
不负责 reset 或退休 Owner。

复用 `csrc/engine/weights/` 已有缓存策略、排序、dispatch 和 TP 规则，将 CUDA
`engine/weights/moe/residency.*` 及 TQ 缓存的晋升/淘汰、路由历史和交换决策
归入同一权重模块。backing、映射、传输及设备完成作为该模块的内部实现，按实际
载荷类型表达差异，不预设 CUDA/Metal 平行结构；普通、映射、流式权重都必须持有
到最后消费者结束。

后续验收：真实 CUDA 多 Engine、失败加载、reload、取消与 drain；B 的准备失败不
影响 A，预算可归还，scratch/传输在途时不提前释放。只有在完成事件和租约覆盖
真实消费者后才缩小 drain 范围，不先删除已有清理。

## 4. 接入显式状态与具体模型组合

先用 Qwen35 的 full attention 与递归/卷积两种状态完成一条生产调用链，再扩展
其他 family。共享 `engine/state/` 已有页、prefix 和 snapshot 规则，但仍须接入
状态身份、generation、独占写访问、checkpoint/restore 与输入位置提交。

| 当前消费者 | 必须完成的改动 |
| --- | --- |
| `engine/models/common/causal_model.h`、CUDA `models/common/causal_model_ops.*` | CUDA 接入具体组合，移除其对模型内 `CausalState`、`CausalResources` 继承与 `ForwardOps` 转发的依赖；加载后模型保留只读权重和数学配置 |
| CUDA `models/common/block.h`、`full_block.*`、`engine/executor/ffn.*` 与各 family | 用具体 attention/FFN 组合替代通用 Block；KV、递归/卷积状态和 decode scratch 交给相应所有者 |
| CUDA `engine/batch/qwen35.*`、`models/qwen4_exp/ops.cpp` | 删除把请求状态绑定/交换到模型的路径，直接传 state view；保持已有 batch、CPU/layer placement 和上下文预算语义 |
| CUDA `engine/state/`、`engine/state/session/`、`models/common/session_codec.h` | 物理 backing 已归 state owner；快照校验、跨层恢复、prefix/RAM/SSD 恢复仍需作为事务接通 |
| CUDA `engine/executor/engine.cpp`、`engine/batch/batch.*` | 执行逻辑归 `engine/executor/`，批次执行归 `engine/batch/`；显式传入输入、状态和执行资源，复用共享请求推进与 `ContinuousBatch` |

当前消费者路径相对 `csrc/`，标注 CUDA 的来源路径相对 `csrc/backends/cuda/`；
所有目标均是顶层 `csrc/engine/` 内的对应职责。先迁移 CUDA 消费者；共享模板若
必须改变 Metal/HIP 调用约定，列为后续依赖，
不得通过代理 Model、转发 facade 或新通用 Block 隐藏旧生命周期。

后续验收：从真实 Engine 入口比较普通/chunked prefill、decode、交错请求及恢复后的
logits 和有效状态。覆盖过期 handle、页/slot 复用、跨层恢复失败隔离和实际输入
位置；独立 attention 或递归方程测试通过不能代替模型接入。

## 5. 迁完优化路径与全部现有消费者

- MTP：收口 CUDA `engine/executor/mtp.cpp` 和各 family predictor，统一 target/predictor、
  递归/卷积及采样状态的接受、拒绝、部分 replay 和位置提交。
- Graph：迁移 `engine/executor/decode_graph.*`、batch 与 family 内 Graph；移除扫描旧
  `blocks` 或交换模型状态的 warmup，捕获实际 view 并保活相关 backing。
- family：覆盖 `engine/models/` 和 CUDA `models/` 中已有 Qwen4、Gemma4、GLM、
  DeepSeek、Nemotron-H、MiniCPM-o 等消费者，按其真实数学与状态接入，不能只修
  Qwen35。RADIO、grid vision、音频与 Duplex 的计算、缓存和请求生命周期各归其主。
- 工具与服务：同步 generation、session/prefix、eval、diagnostics、reference、
  bench 和测试。诊断生成复用生产 step；Duplex/media 统一单终态、取消和背压，
  不把现有 `Engine::control()` 入口当成生命周期已经统一的证明。

后续验收：各 family 与优化路径按实际改动单独验证；Graph/eager 数值一致，warmup
不污染生产状态，地址保活和失效生效；MTP 覆盖接受/拒绝与部分 replay。服务覆盖
准入、deadline、背压、输入上限、异步文本准备、取消和恰好一个终态。迁移一批
消费者就删除相应失去用途的旧实现，不把统一清理拖到最后。

## 6. 后续阶段完成与后续范围

后续 CUDA 阶段完成须同时满足上述源码接入、默认/兼容 runtime、diagnostics、eval、
host、相关 JIT 和行为验证。依赖图无回环，第一方 include/清单闭合，不存在已删
接口的活动调用或为保留旧执行链而添加的包装。

本轮不以构建、测试、bench 或 Metal/HIP 结果作为完成依据；这些项目均未验证。

目录验收按最终职责结构逐项检查：

- `csrc/source/` 的生产源码、include 与 CMake 路径已迁移，无转发头；构建与运行验收延期。
- `csrc/third_party/` 目录及其旧分发路径已移除，必要许可在仓库既有约定位置保留。
- CUDA 的 Engine、ops、kernels、native 和入口/工具已归入顶层相应职责，当前
  `backends/cuda/` 下的生产源码随调用方迁移删除；Metal/HIP 后续迁完后删除整个 `backends/`。
- `state`、`weights` 共享策略和生命周期，设备载荷在模块内部表达；没有预设的
  CUDA/Metal 平行框架、重复共享规则或空目录。
- 数学与 native 原语不依赖 Engine 驱动；include 可明确解析，设备依赖由 CMake
  目标隔离，不能靠 include 搜索顺序误选另一份头文件。
- native、JIT 和 wheel 均使用唯一 kernel 源码，TQ 安装资源及加载行为保持可用。

Metal/HIP 的适配以及必须改变其调用约定的共享接口重构留到后续阶段；在此之前
不宣称跨后端统一完成。没有新增需求的路径重命名、一次性 probe、阶段记录包或
离线产品恢复不列为任务。

## 尚未验证与记录规则

- 当前完整构建矩阵和服务主链；本轮文档更新不替代运行验证。
- TQ/MoE 真实大权重、多 GPU TP/EP、随机采样分布、峰值内存与性能。
- 显式状态迁移后的 session、MTP、Graph、取消退役与多 Engine 预算归还。
- CPU offload 的整模型数值，以及其他后端的设备数值和性能。
- 构建矩阵、测试及 JIT 运行均未验证；本批仅更新并静态核对 CMake/JIT 路径。

进入后续验证阶段后，按 [refactor.md](refactor.md) 的规则构建实际目标再运行测试。77、skip、
缺少模型参数或 `model_branch=skipped` 均为未验证；构建问题、环境不足和数值
缺陷分开报告。保留固定输入、原始张量、数值设置及可复现命令，不通过改 dtype、
放宽容差或跳过失败输入消除差异。

文档只记录有当前源码或运行证据的结论，不引用不存在的缺陷清单、基线文件或
历史日志作为验收依据。完成后移除对应剩余条目，变更历史由 Git 追溯。
