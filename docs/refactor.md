# Native architecture migration

## 范围与完成条件

将 MFQ `refactor/cuda` 在 `a93e73a27a30e566a440eb5dc146873d75064e94`
的架构改动完整迁入 TyloInfer。比较基线为该分支已经合入的 master
`d906b257e10353948152ba51c824787045271cd9`；按最终实现迁移，已撤销的目录调整、
被后续接口取代的中间实现不重复搬运。

**当前执行范围仅 CUDA。Metal 和 HIP 的源码、构建及 Python 实现保持不动。**
先迁移 CUDA 私有的加载、数学、执行资源、物理状态、Engine 装配与 Graph，及其
CUDA JIT、测试和文档。现有公共 Interface 继续使用；需要改变其他后端调用约定的
共享 Engine/core/CPU/format/model-source 重构先保留在后续工作中，不在本阶段实施。

完整参考涉及的共享层和其他后端适配仍列在设计中，用于保留总目标的差距，不能将
本阶段 CUDA 完成写成整个迁移完成。TyloInfer 的 TQ 主路径必须接入同一所有权和
执行规则；MFQ 兼容格式保持显式启用。不恢复已移除的离线量化、校准产品入口。

参考分支仍有未闭合的模型迁移，其完整 runtime 不能构建。迁移不能以复制这个编译
断点作为完成：所有现有调用方都必须接通新接口，删除已失去消费者的旧实现，实际受
影响目标必须能构建并通过对应行为验证。参考设计尚未实现的产品能力不当作已有功能
宣称迁入；已实现的新接口则不能以保留旧链路或增加兼容外壳代替接入。

工作在 `refactor/arch` 进行。每个职责闭合、完成相应验证后提交，不把完整迁移缩减为
目录更名、单个算子或能够单独编译的中间层。

## Module 与 Interface

以 `codebase-design` 的 depth、locality 和 seam 为依据。Interface 同时包含类型、
错误、顺序、生命周期和性能约束。尽量由既有真实所有者承担完整行为，不新增通用
Tensor/KV 接口、插件表、转发 facade 或第二条执行链。

| Module | Interface / seam | Implementation 与唯一所有者 |
| --- | --- | --- |
| `components/io` | 有界 range/batch reader、完成与统计 | 文件句柄、异步读取；不依赖模型、格式、Engine 或设备 |
| `formats` | 持久化 schema、规范 dtype、编码行 reader | 码表、TQ/NINT/NVQ 等格式及范围解码；不依赖 model source 或后端 |
| `model_source` | 规范 tensor inventory、metadata、保活的范围 reader | 输入解析、别名规范化；不执行模型或决定驻留 |
| `engine/include` | owned 请求、媒体、采样、结果、资源错误 | 公共值唯一声明；不携带 native Tensor、stream 或协议对象 |
| `engine/model` | 具体 family 的数学组合 | 层序、位置公式和配置；直接组合完整算子，不拥有活动请求状态 |
| `engine/state` | generation、独占写访问、checkpoint、实际输入位置提交 | 请求状态与恢复事务；native storage 提供物理载荷 |
| `engine/weights` | 驻留计划、准备/提交、最后消费者租约 | 预算、路由历史与晋升/淘汰规则；native 执行映射、传输与回收 |
| CUDA `execution` | 借用的 scratch、stream/event、通信与完成 | Owner 创建、drain、回收资源；算子不能 reset 或退休 Owner |
| CUDA `ops` / `kernels` | 完整 attention、FFN、projection 数学 | 接收已装配权重、输入、显式 state view 和执行资源；不读模型文件 |
| CUDA `storage` | 物理 backing、状态 view、传输与完成 | KV/递归状态、权重映射、分配和复制；不回调 Model 生命周期 |
| CUDA `models` | 加载并装配具体算子 | 解析 family 参数角色；共享行为不能藏在某个 family 目录 |
| `engine/executor` 与 CUDA `engine` | 有界 step、物理 batch、typed 结果 | 请求推进、采样、Graph、MTP 与事务；CUDA 根只装配依赖 |
| Scheduler / Transport | 准入与发布 / 线路解码与编码 | Scheduler 唯一逻辑队列；Transport 不执行模型 |

CPU 算术、内存中的策略和格式函数直接跨 Interface 测试。文件 reader 使用真实临时
文件及既有内存 ModelSource。设备存活、Graph 和异步退役必须使用实际 CUDA 路径；
host mock 或源码字符串不能证明设备行为。

## 不变量

- 保留数学公式、dtype、累积顺序、编译数值选项、原始容差及所有错误检查。
- 不保留通用 `Block`，也不换成另一个通用层基类、类型擦除表或回调转发层。
  family 按自身数学使用具体组合；移除 `CausalResources` 继承与 `ForwardOps` 约定。
- `forward` 显式接收状态。禁止通过 bind/swap/unbind、作用域安装器或代理 Model
  把活动请求重新装回权重对象；reset/checkpoint/restore 归 StateStore/native storage。
- 一个 Engine 的加载、执行和存储资源有明确 Owner。失败加载、reload、取消与销毁
  必须等待真实消费者结束，再依次退休状态、权重与执行资源。
- KV 页、slot、快照和 pending 工作携带 generation；旧 handle 不得访问复用后的资源。
  B 的准备失败不能改动 A 已提交的状态；跨层恢复必须作为事务验证。
- 位置按实际完成 forward 的输入提交。MTP target/predictor、递归/卷积状态及采样状态
  属于同一验证、拒绝和部分 replay 事务。
- Graph 捕获保活的实际 view。warmup 不污染生产状态，不从旧 `blocks` 扫描或交换状态；
  Graph、scratch 和传输 backing 保活到最后消费者完成。
- 保留 `Transport → Scheduler → Engine`；step 返回 owned typed events 后才发布。
  每个已接受请求只产生一个终态，Duplex/media 不拥有第二套请求生命周期。
- 格式与物理存储不知道模型品牌。Shared model 不 include 后端头；CUDA model 不 include
  CUDA Engine；算子不 include model source、加载器或 Engine。
- 源码只有一份。迁移调用方后删除旧路径，不提供转发头或复制 kernel；JIT 与 native
  直接编译共同来源。TQ wheel 资源必须继续可安装和加载。

## 迁移顺序与剩余工作

每组都覆盖实现、全部调用方、CMake、Python/JIT、测试和文档。以下是完整目标的
工作分解，不是任选其中一组即可完成的列表。

当前先执行下表中 CUDA 私有部分：完成加载分离，收口 CUDA 目录与依赖，再迁移
执行 Owner、显式 state 和真实模型/Engine/MTP/Graph 消费者。共享层、CPU I/O 及
Metal/HIP 的改动暂缓；若某一步需要改变它们的 Interface，登记依赖后继续其他
CUDA 工作，不用复制共享实现或增加转发包装绕过。

| 顺序 | 剩余改动 | 必须提供的证据 |
| --- | --- | --- |
| 1 | 拆出 `components/io`、`formats`、`model_source`；host worker/copy 归 CPU；共享请求、模型定义与状态策略归 Engine；删除 `mfq-core` 聚合 | host/CUDA × communication on/off 的实际构建；reader、host copy、dtype、范围与依赖检查 |
| 2 | 完成全部格式与 MoE/TQ 加载分离；区分规范输入、格式解码、算子装配、物理 backing | 默认 TQ 和兼容模式均构建；加载/分片/映射保活及原判据的数值对照；ops 无 ModelSource 依赖 |
| 3 | CUDA 消除 `core/`、`runtime/` 混合层；按最终职责放置 execution、state、weights、models 与 kernels；移除失去用途的前缀和单消费者片段 | 所有第一方 include 可解析，真实依赖为 DAG；native/JIT manifest 闭合，无旧路径或源码副本 |
| 4 | 统一环境配置入口；typed gate mode；拆开 inference 与 tool/backward kernel；收窄原始 activation/embedding/KV kernel Interface | 环境读取、冻结/覆盖、非法值测试；新旧原始张量数值一致；实际冷构建 JIT 与原生算子测试 |
| 5 | typed load options、family strategy、Engine composition root；共享 Scheduler/Transport 控制与 owned 请求；诊断生成走生产 step | 公共头独立编译，通信关闭仍可构建；取消、背压、输入上限、异步文本准备及单终态行为测试 |
| 6 | 执行 Owner、collective/branch/scratch；WeightStore backing/租约；路由历史和驻留规则归 Engine | 多 Engine、失败加载、reload、取消与 drain；普通/映射/流式权重保活，准备失败隔离与预算归还 |
| 7 | StateStore/页 generation/快照版本；attention、递归状态与 KV backing 分离；具体模型组合，删除旧 Block/继承生命周期 | 至少 full attention 与递归两种状态的真实调用；普通/chunked prefill、decode、交错请求和 checkpoint/restore 数值对照 |
| 8 | 接通全部现有 family、batch、prefix/session、MTP、Graph、eval、diagnostics、bench 与多模态消费者 | 无已删接口的活动引用；target/predictor 接受/拒绝与部分 replay；Graph/eager 对照、warmup 隔离、失效与地址保活 |
| 9 | 按最终源码与上游差异审计，删除中间兼容链、失效测试与重复实现，更新实际架构和支持说明 | 默认/兼容 runtime、diagnostics、eval、host 与相关 JIT 构建；依赖、资源与数值 gates 的逐项结果 |

当前已有：顶层 `csrc` 更名、CUDA native tensor/kernel 单一来源、通用线性加载与计算
分离。它们只覆盖上述目标的一部分。TQ 的专用加载/缓存、旧管理执行上下文、模型内
请求状态、通用 Block、Graph/session/MTP 消费者及 core 聚合仍须处理。

目录调整应直接朝最终所有者推进；不为重放历史提交恢复已撤销的布局。当前阶段
不提交 Metal/HIP 文件改动，也不进行会迫使这些后端同步修改的公共 Interface 更换。

## 验证与完成审计

构建目标后再运行 CTest，避免旧产物造成假通过。机器约 32 GB，原生构建使用有界
并行，Torch JIT 使用 `MAX_JOBS=1`，不与其他 CUDA 重编同时运行。缺少硬件/模型、
返回 77 或 skip 均记录为未验证，不计为对应能力通过。

已知基线：TQ GPU Graph 检查有 273 项既有失败；默认 diagnostics 缺少 8 个旧 MFQ
符号；communication 关闭的全量 host 构建有 `chat.h` 依赖问题；旧 NVQ Python 测试
缺少已移除模块而无法收集。原始日志位于忽略的 `build/logs`，不能靠放宽判据或恢复
离线产品模块使其变绿。公共依赖和默认构建闭合时必须解决相应架构阻断；数值缺陷
保留实际失败输入，并明确区分迁移引入与既有结果。

完成审计须逐项核对上表与不变量，比较固定参考的最终文件/Interface，而不只比较
commit 数量。每项需要实际源码、调用链、构建输出或运行结果；只有设计、独立新层
测试、源码搜索无结果或历史通过记录都不足以证明生产接入。对没有实际硬件的后端
仅报告源代码/公共接口检查覆盖，不能声称完成其设备数值或性能验证。

整套工作未闭合时目标保持进行中。每阶段提交说明职责变化、实际验证与剩余缺口；
最终文档按实际代码更新，不把参考分支的未完成说明写成 TyloInfer 已有能力。
