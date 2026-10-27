# TyloInfer 原生架构重构

## 目标与范围

以 TyloInfer 当前代码为起点，分离模型数学、活动请求状态、权重驻留和执行资源，
接通现有生产调用链。当前结构见 [architecture.md](architecture.md)，剩余任务和
顺序统一维护在 [plan.md](plan.md)。

设计参考为 MFQ `refactor/cuda` 的 `a93e73a27a30e566a440eb5dc146873d75064e94`，
其比较基线为 `d906b257e10353948152ba51c824787045271cd9`。参考用于核对职责、
接口和数值约定；不重放已撤销的目录调整，不复制中间实现或编译断点，也不把参考
分支的完成度、失败数量和测试日志当成 TyloInfer 的现状。

**当前实施范围是 CUDA 及其必需的公共模块。Metal/HIP 的源码与构建、Python
算子与执行逻辑不在本阶段修改范围；CUDA JIT 的源码和 include 路径随迁移同步。** 最终目录覆盖所有后端，当前先迁移 CUDA 及必要公共
职责，其他后端后续迁移。公共 I/O、formats、source、CPU 和 Engine
已经拆分，继续完成其 CUDA 调用方、测试和构建依赖，不重新建立聚合层。公共改动
须保持其他后端现有调用约定；必须同步修改 Metal/HIP 的接口重构列为跨后端后续
工作，不以复制共享实现或转发外壳绕过。

TyloInfer 的默认 TQ 路径必须遵循同一所有权和执行规则，MFQ 兼容格式保持显式
开启。不恢复已移除的离线量化、校准入口；Python JIT 与 TQ wheel 资源随受影响的
native 接口同步更新。

## 目标职责

接口包括类型、错误、调用顺序、生命周期和性能约束。沿用现有六个 Engine 职责
目录，让真实所有者承担完整行为，不新增通用 Tensor/KV 接口、插件表或转发 facade。
目标目录以 [architecture.md](architecture.md#最终目标目录) 为准：取消
`backends/` 聚合，将 Engine、ops、kernels、native 按职责放在 `csrc/` 顶层；
`model_source/` 简化为 `source/`，移除 `third_party/` 目录。状态和权重的物理
backing 归所属模块，执行资源 Owner 归 executor，不另设 storage/execution 层。

`state`、`weights` 不预设 CUDA/Metal 平行目录，统一策略与生命周期，保留实际
需要的载荷类型差异。ops 按计算职责组织；必要的设备实现差异在模块内部表达，
不复制共享逻辑或创建空目录。只有 kernels/native 固定按设备分目录。
以下为目标职责，路径均相对 `csrc/`；迁移对应见 [plan.md](plan.md)。

| 模块 | 接口与职责 |
| --- | --- |
| `components/io` | 有界 range/batch 读取、完成与统计；拥有文件句柄和异步读取，不依赖模型或设备 |
| `formats` | 持久化 schema、规范 dtype、码表和编码行解码；不决定模型执行或驻留 |
| `source` | 规范 tensor inventory、metadata 与保活的范围 reader；在输入边界规范名称和 dtype |
| `engine/requests` | owned 请求、媒体、文本准备、约束、采样参数与结果；不携带设备或协议对象 |
| `engine/models` | family 配置、权重角色、层序与具体算子组合；已迁移切口生成 `std::vector<Task>`，不拥有活动请求状态 |
| `engine/state` | 状态身份、generation、独占写访问、checkpoint/restore、输入位置提交及设备/RAM/SSD 载荷；具体载荷与共享生命周期在同一职责内 |
| `engine/weights` | 权重加载、预算、路由历史、驻留、准备/提交、最后消费者租约及物理映射和传输 |
| `engine/batch` | 物理 batch 规则、prefill 预算及批次推进；不同 family 复用流程而保留各自数学 |
| `engine/executor` | 公共 Engine 接口、有界 step、请求推进、采样、MTP、Graph 与执行资源 Owner；Executor 只分发固定 `Task` |
| `ops` | 按 attention、FFN、projection 等计算职责提供完整数学；只接收已装配权重、输入、显式 state view 和执行资源，不接收管理上下文 |
| `kernels/<device>` | CUDA/Metal/HIP/CPU 计算内核；不处理模型加载、请求或驻留策略 |
| `native/<device>` | 无模型知识的 tensor、内存分配、复制、同步、通信及 host 执行原语 |
| `apps` / `commands` | 可执行入口、依赖装配与命令处理；不复制 Engine 执行循环 |
| Scheduler / Transport | 唯一逻辑准入与发布 / 线路解析与编码；Transport 不执行模型 |

模型与执行器之间的 seam 固定为 `Task`/`Inputs`；不引入 `Plan` 或 `Bindings` 命名。
`CudaExecutionContext` 是待拆除的旧 Owner，不是目标接口；在构造时拆出 profiler、
MMQ 和类型化配置只是过渡，不得恢复通过 `execution.x` 访问服务；typed-resource
borrowing 也不等于全局上下文已经移除。

## 不变量

- 保留数学公式、dtype、累积顺序、编译数值选项、原始容差及所有错误检查。
- 迁移现有消费者后删除 CUDA 通用 `Block`、`CausalResources` 继承和 `ForwardOps`
  转发约定；family 直接组合具体算子，不替换成另一套通用层基类或类型擦除表。
- `forward` 显式接收状态。不能通过 bind/swap/unbind、作用域安装器或代理 Model
  把请求状态装回权重对象；reset/checkpoint/restore 归 state 模块，不回调 Model 生命周期。
- 一个 Engine 的加载、执行和存储资源有明确 Owner。保留已有卸载清理语义；失败
  加载、reload、取消与销毁都须等真实消费者结束，再退休状态、权重和执行资源。
- KV 页、slot、快照和 pending 工作携带 generation；旧 handle 不能访问复用后的
  资源。B 的准备失败不得修改 A 的已提交状态，跨层恢复必须作为事务。
- 位置按实际完成 forward 的输入提交。MTP target/predictor、递归/卷积状态与
  采样状态属于同一验证、拒绝及部分 replay 事务。
- Graph 捕获实际 state view，并保活 Graph、scratch 与传输 backing 到最后消费者
  完成。warmup 使用隔离状态，不从旧 `blocks` 扫描或交换生产状态。
- 保留 `Transport → Scheduler → Engine`。step 返回 owned typed events 后才发布；
  每个已接受请求恰有一个终态，Duplex/media 复用取消、背压和请求生命周期。
- 共享模型数学不 include 设备专有头；模型组合不 include 引擎驱动。算子不 include
  source、加载器、state/weights 管理者或 executor 驱动，只借用低层资源与显式
  view。通用数学归 ops，ops/kernels/native 不反向依赖 Engine 驱动。
- 源码只有一份，迁移消费者后删除旧路径。CUDA kernel 的目标源码根为
  `csrc/kernels/cuda/`，native/JIT 共用；TQ wheel 从唯一源码收集安装资源并同步
  加载路径，不保留另一份手工维护的源码。移除 third_party 目录时保留必要许可信息。

## 完成条件与验证

以下是后续迁移的验收条件。最近实现批次涉及 CUDA 生产源码、必要公共模块、
许可证路径及 CMake/JIT 接线；CMake 中直接引用迁移文件的路径同步调整。
测试实现、Metal/HIP 和 Python 算子与执行逻辑保持原样，未运行构建或测试。

后续每项迁移覆盖实际受影响的实现、全部调用方、CMake、JIT、测试与文档。构建目标后
再运行 CTest，避免使用旧产物。新接口允许不向后兼容，但同一迁移必须接通当前
范围内的消费者；独立新层测试或源码搜索不能代替 Engine 接入。仍被其他后端使用
的公共接口在对应消费者迁移后再删除，不为 CUDA 增加新的兼容执行链。

目录归位与职责、所有权迁移同批完成。CUDA 阶段检查其旧路径与引用消失，最终
各后端迁完后删除 `backends/`；`source/` 重命名和 `third_party/` 移除一并检查
构建、清单及许可引用。共享规则不得复制，实际消费者必须通过对应验证。

后续 host 验证覆盖通信开启/关闭；CUDA 验证覆盖默认 TQ 与 MFQ 兼容配置，并验证
通信关闭的独立计算/加载路径。完整服务与 Torch reference 的通信要求保持显式。
runtime、diagnostics、eval 及实际受影响测试必须可构建，不能以分支合并
为由豁免受影响目标。

后续 CPU 策略、格式函数和 reader 使用直接行为测试与真实临时文件。设备存活、Graph、
异步回收和数值行为使用实际 CUDA 路径；缺少硬件/模型、返回 77 或 skip 都记为
未验证。数值差异保留实际输入，区分迁移引入与旧实现已有结果，不放宽判据。

机器内存约 32 GB，注意构建与模型验证的内存占用，避免同时运行多个重型任务；
不固定限制 `-j` 或 `MAX_JOBS`。根据改动选择验证，纯文档改动只核对路径、链接、
目标和源码描述，不触发全量编译或数值测试。

后续完成一个职责及对应验证后再提交，保留不可重建的原始数值证据。变更历史由 Git
追溯，剩余工作只维护在计划中。CUDA 阶段完成不等于跨后端架构统一；没有实际
设备验证的后端不能宣称完成数值或性能验收。
