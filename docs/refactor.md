# TyloInfer CUDA 重构约定

本文维护稳定的目标职责、范围和不变量。当前结构见
[architecture.md](architecture.md)，剩余任务与完成条件只维护在
[plan.md](plan.md)，不在本文记录逐提交进展。

## 目标与范围

以 TyloInfer 当前实现为起点，分离模型数学、活动请求状态、权重驻留和执行资源，
完成真实生产消费者的接入。设计参考为 MFQ refactor/cuda 的
a93e73a27a30e566a440eb5dc146873d75064e94；参考只用于职责、接口和数值约定，
不复制中间实现，也不引用其构建或测试结果作为 TyloInfer 的完成证据。

执行顺序、当前模型主线与暂缓范围唯一以 [plan.md](plan.md) 为准；接口变更同步必需消费者，
不借共享接口扩展其他模型能力。加载、生成、batch、MTP、session、Graph、取消和卸载按已有支持范围验收。

**本阶段只重构 CUDA 及其必需公共代码。** Metal/HIP 的可编译、可运行和兼容性
不作为约束或验收条件；CUDA 所需的共享接口与目录可以直接调整，不等待其他后端
适配，不为其保留旧接口、兼容层、转发头或源码副本。Metal/HIP 的适配留待后续。
Python 算子与执行逻辑不在当前范围；CMake、JIT/打包和测试的同步按计划恢复后处理。

拆分过程中，若当前代码直接暴露出职责、依赖或对象所有权错误，可在同一批次修复；
接口和装配按计划推进。全过程保持数学、dtype、校验、错误传播和生命周期约束。

默认 TQ 路径遵循同一所有权与执行规则，MFQ 兼容格式保持显式开启。
不恢复已移除的离线量化、校准产品入口。

## 目标职责

目标目录和职责唯一以[architecture.md](architecture.md)为准，本文不复制另一张目录表。
最终结构中，模型位于 `csrc/engine/models`，负责模型配置、权重角色、前向、模型专属
State/session、batch、MTP、Graph 适配和装配；通用 `engine/generation` 负责生成推进、
采样和 MTP 流程，`engine/scheduler` 负责请求生命周期、准入、合批、背压和终态，
`engine/requests` 负责请求/结果契约与输入输出解释，`engine/state` 负责通用状态载荷、
页、快照和 session，`engine/weights` 负责读取、准备、驻留、传输及权重统计。
设备执行统一归 `csrc/engine/backends/<device>` 的 executor、ops、kernels、native；
apps/commands 负责产品装配、命令和 reload，transport 只负责外部协议。
各目录的文件职责、依赖方向和最终结构见[最终目标目录](architecture.md#最终目标目录)
及[模块内部结构与文件职责](architecture.md#模块内部结构与文件职责)。
各设备的 `backends/<device>` 仅承载 executor、ops、kernels、native；根 `backends` 提供共享 dtype.h 与设备目录，设备目录不镜像 `engine` 上层模块，通用机制留在对应 `engine` 模块，模型知识留在 `models`，设备 API 调用不改变业务归属。

TyloInfer 自有 C++ 命名空间统一为 tyloi，公共抽象接口为 tyloi::Tensor，DenseTensor 与各编码实现继承它；删除 mfq、mfq_tensor_backend 旧命名空间及兼容别名。
Torch 转换留在适配边缘；MFQ 文件格式/编码标识和第三方命名空间保持其原有语义。

文件归属由模型知识决定，不由 State、Run、Batch、MTP 等类名决定。
Model 保存只读共享权重，每请求 State 独立；模型 State 的类型与行为归 models，
通用载荷和事务机制归 engine/state。models 目录中允许有状态实现。

模型组合 Attention、MLP、Encoder 等计算单元，持有只读权重与配置，通过 forward 调用公共 ops。
不强制公共 Block/Attention 基类；模型差异通过模型侧数学步骤表达，融合算子定义明确的计算范围。
公共 ops 负责数值契约和设备分派，backend ops 实现算法；不接收完整 Model 或回调模型前向，
不新增 Plan/Bindings、公共 Task IR 或第二生成循环。模型装配与 family 实例化归 models。

Engine 的生产实现和公共头不依赖具体 family 类型、名称或私有布局。
模型计算单元使用公共 ops、Tensor、权重与状态契约，不接收或保存 Executor，不借构造注入绕过这一规则。
公共模型接口不暴露后端私有 Task/Tensor 或管理者；设备适配中的模型语义仍归 family，数值实现归 backend ops。
使用 requires、别名或转发头不改变实际依赖归属。

模型实现由独立编译目标承载，通用 Engine 不编译或传递链接具体模型实现；
最终 runtime/入口组合两者。设备依赖由 CMake 目标隔离，
不靠新增 include 根或源码副本掩盖依赖问题。

## Engine 实例职责

最终 Engine 采用构造函数注入的 `CausalModel` 与 `Executor`，并内置 Scheduler；
Scheduler 在 Engine 内直接推进 generation。Engine 协调请求、预算、失败清理、卸载
和生命周期，但不解析模型路径、不持有加载回调，也不把完整上下文传给模型或算子。
目标结构见[架构设计](architecture.md#engine-实例目标设计)。

apps/commands 负责根据模型与设备配置选择工厂、打开 weights、创建 Executor，装配
Engine，并协调 reload；模型工厂返回一个绑定单一模型的 `CausalModel`，reload 不维护
全部 family 的 variant。迁移期间复用已有生成/MTP、batch、Graph、session 和 Budget
逻辑，不复制请求循环或算法。

NativeEngine、CudaEngine、EngineInstance 及 Context/CudaExecutionContext 属于过渡实现，按目标职责收拢。
Executor 负责执行作用域、stream/event、工作区、Graph 执行资源与完成/排空，
不提供 project/norm/attention 等重复数值入口，不接管模型加载或请求策略。

## 所有权与接口约束

- CudaExecutionContext 按实际所有者拆除，不能整体改名成 Executor 或执行 scope。
  执行资源归后端执行设施，状态与权重管理者留在各自模块；模型不提取 Projector、profiler、stream 或 scratch 逐项传递。
- 不引入或恢复模型专属 Runtime 转发壳；模型装配复用已有加载结果与共享执行接口。
- 复用已有 Loader、缓存、执行资源、Batch、Graph 和预算机制；计算单元组织实际数学步骤，
  不增加空的 Model/Block 转发层次或模型计算单元的通用继承框架，
  不新增 registry、callback 表、代理 Model、旧 Run 桥接或第二执行循环。
  新接口允许不向后兼容，必须同步 CUDA 及其必要公共消费者；Metal/HIP 的适配不阻塞本阶段。
- 请求/批次持有模型 State；模型和计算单元不保存活动 State 或执行工作区。
  算子定义 workspace 布局，后端执行设施保证并行隔离、复用和保活；Engine 协调跨模块预算与生命周期。
- 激活、权重和状态载荷共用 Tensor 抽象接口，以引用/智能指针使用；基类只承诺共性数据与生命周期契约，各表示的字段和布局校验归具体实现，遵循[统一 Tensor 契约](architecture.md#统一-tensor-契约)。
  格式实现复用 encoding，数值计算归 ops；不另建公共权重体系或把完整旧权重/执行管理者搬入子类。对象共享、view 别名和数据克隆语义明确，权重及其别名只读。
  统一逻辑 shape/dtype 与展开字节语义，实际计量枚举全部 backing 并去重；布局受限操作在修改或提交前校验，引用不替代保活，租约覆盖最后消费者。
- 公共 ops 按设备分派并校验数据相容性；backend ops 从当前执行作用域借用资源，
  不调用模型前向或请求调度，不依赖 source、加载器或 state/weights 管理者；
  native 只提供设备原语和通用分配记录作用域，不处理模型、请求、权重类型或驻留预算策略。

## 不变量

- 保留数学公式、dtype、累积顺序、编译数值选项、原始容差和所有错误检查。
- scope 在同步推进或协程 resume 的外层建立，退出、异常或挂起后恢复线程环境；重新恢复/换线程时重新进入。
  纯元数据/CPU 操作无需 CUDA scope；加载等 CUDA 提交入口建立所属 scope，Tensor 不隐式创建 Executor 或 stream。
  不把 scope 放进跨挂起点存活的协程局部变量；不绑定当前模型、请求或 State，不使用进程级当前 Engine。
  scope 不按 token 重建 stream，退出不隐含设备同步；异步完成、依赖与保活规则见[执行作用域](architecture.md#executor-与执行作用域)。
- forward 使用显式传入的请求 State；不经 bind/swap/unbind 或作用域代理把状态
  装回共享 Model，不传一个 State 却实际访问另一个。模型 reset/checkpoint/restore
  组合归 models，底层事务复用 Engine。
- 加载、执行、存储资源有明确 owner；失败加载、reload、取消和销毁等待真实
  消费者结束后再退休状态、权重和执行资源。保留原有失败清理与异常传播。
- 句柄必须拒绝跨所有者或复用后的陈旧访问；准备另一请求失败不能污染无关请求。
  模型跨层恢复保持事务边界，已有候选恢复、缓存失败策略与位置校验不能丢失。
- 位置按实际完成 forward 的输入提交。MTP target/predictor、递归/卷积和采样状态
  共同遵循接受、拒绝及部分 replay 的规则。
- Graph 捕获实际 state view，保活 storage、scratch 和传输到最后消费者结束；
  warmup 隔离生产状态。请求/slot/page 身份及 Graph 约束见
  [architecture.md 的 session、MTP 与 Graph](architecture.md#sessionmtp-与-graph)，先核对失效、同步和复用，
  不在缺少证据时增加统一 generation 机制或认定 ABA 缺陷。
- 目标 Scheduler 属于 Engine 内部并直接推进 generation；保留过渡逻辑中请求提交、
  调度、取消和背压的既有语义。过渡阶段的 step 仍返回 owned typed events 后再发布，
  每个已接受请求恰有一个终态，媒体和 Duplex 使用一致的取消与背压规则。
- 源码只有一份。同步本阶段 CUDA 调用方及 include；暂缓的构建、测试和 JIT/打包引用登记待恢复，
  不计为迁移完成。共享文件迁移以 CUDA 生产消费者接通为条件，
  不等待 Metal/HIP 适配，也不以复制共享实现维持其旧路径。
- CUDA kernels 以 csrc/engine/backends/cuda/kernels 为唯一源码根，native/JIT/wheel 共用。
  移除 third_party 或调整发行路径时保留必要许可证与 NOTICE。

## 验证与文档维护

当前约束仍为不修改或运行测试，相关验收保留待办，待用户调整该约束后执行。
本次文档更新不运行构建或数值测试，也不改变这一限制。
源码接通、静态核对、构建通过和运行通过必须分别表述；
未完成必要验证时，计划中的 Qwen35 阶段保持未完成。

后续验证构建实际受影响的当前目标，再运行对应测试，避免旧产物。
Qwen35 的生产、batch、reference、eval、diagnostics 和服务消费者均在验收范围，
独立算子测试或源码搜索不能替代整链接入。具体完成条件以计划为准。

按影响范围覆盖 host 通信 on/off、CUDA 默认 TQ 与显式 MFQ 兼容配置；
独立计算/加载支持通信关闭，完整服务与 Torch reference 的通信要求保持显式。
涉及源码/JIT/发行路径时验证安装后的实际资源加载。

CPU 策略、格式与 reader 使用直接行为和真实文件验证；设备存活、Graph、异步回收
和数值使用真实 CUDA 路径。缺少硬件/模型、返回 77、skip 或 model_branch=skipped
均记为未验证。保留固定输入、原始张量、数值设置和可复现命令，
区分旧实现已有差异与迁移引入的问题，不放宽容差、改 dtype 或跳过失败输入。

机器内存约 32 GB，避免同时运行多个重型任务；不固定限制 -j 或 MAX_JOBS。
验证范围与改动匹配，纯文档只检查源码描述、路径、链接与文档一致性。

文档分工保持固定：architecture 写目标架构与职责，refactor 写稳定约束，
plan 统一维护当前状态、剩余工作与完成条件。完成条目从计划移除，提交历史由 Git 追溯；
不追加逐提交摘要、静态审计流水或重复的“当前进展”。
