# TyloInfer 重构计划

目标架构以 [architecture.md](architecture.md) 为准。本文只列剩余工作、暂缓项和验收条件，不保留已完成任务或逐提交进展。
**先完成 Gemma4 的剩余公共契约与 Engine 装配，再清理通用 Block，随后推广并验收 Qwen35 全链路。**
CMake、旧目录收尾、Metal/HIP、工具接线、测试迁移和运行继续暂缓，不作为当前源码重构的前置条件。
源码接通、静态核对、构建和运行分别记录；尚无构建、数值或整链验收结论。

## 1. 完成 Gemma4 剩余接口与所有权

Gemma4 保持当前支持范围或明确拒绝，不新增媒体、MTP、session 或 batch 能力；其他 family 的迁移不作为其公共接口收敛的前置条件。

1. 实现[Tensor 抽象接口及具体数据表示](architecture.md#统一-tensor-契约)：DenseTensor 与 TQ/NINT/NVQ/MX 等实现继承 Tensor，统一逻辑 shape/dtype、布局和存储枚举；复用现有编码、缓冲区与数值实现。
2. 将 Loader、模型字段、共享 Attention/MLP、视觉、State、公共 ops 及必要 CUDA 消费者迁至引用/智能指针契约，清除抽象 Tensor 的按值构造、容器与返回接口；明确对象、view、只读别名及异步结果的所有权。
   拆除 Weights、QuantLinear/QuantLinearGroup/Experts 体系及旧对象强转，不将其整体搬入 Tensor 子类；分组使用有序权重指针列表，保留联合投影与输出边界。
3. 补齐各编码、专家与分片的准备、完整缓冲区遍历、设备搬运和 view/cast 规则，复用 backing 去重统计；统一 TQ、MX 等实现的逻辑信息与物理字节语义。
   [相容性检查](architecture.md#公共-ops-与后端分派)覆盖 embedding、expert、attention 状态 view/元数据、CPU 暂存、实际 host 计算、分片及异步结果；不新增注册系统或公共 Task IR。
4. 将 Engine 收敛为构造注入 CausalModel 与 Executor，移出 CUDA friend 写入内部字段、向下转换及设备装配职责，保持当前单请求支持范围。
   其余旧 Engine 装配随对应 family 迁移；资源管理者留在 state、weights 和后端执行设施，不新增 Runtime 转发壳或资源袋。
5. 将多态 Tensor、view 及全部 backing 接入[执行完成与资源保活契约](architecture.md#executor-与执行作用域)，协调外部权重及 offload，补齐取消、加载失败、卸载/reload、异步输出与租约的退休接线。
   收拢共享视觉的设备选择、权重暂存和空间估算，移除写死 CUDA 0 的转换与剩余计算图回调壳；保持输出迁出借用存储后再归还的顺序。
   补齐剩余 legacy、媒体及资源退休入口的作用域覆盖；复用已有 scope/drain，不将整设备同步优化另列为当前前置任务。

**源码完成条件：** 一阶段 CUDA 路径经 Tensor 抽象接口及引用/智能指针访问各数据表示，基类无格式字段或数值实现，编码、view、执行相容性与生命周期契约落地；Gemma4 的装配、生成和释放经注入的公共模型/执行契约组织，模型不依赖 Executor，通用运行代码不访问 family 私有布局。
所有资源 owner、作用域和完成时点可定位。嵌套 scope、同线程多 Engine、协程暂停/恢复、异常恢复、并行工作区隔离和异步保活待验证工作恢复后验收。

## 2. 拆除剩余通用 Block 与旧执行适配

Qwen4Exp、DeepSeek V4.1 及其共享消费者仍须拆除公共 Block 继承树；内存块、KV 页块和模型内部的层权重记录不属于删除范围。

- 将 Block/FullBlock 中的数学组合归具体模型计算单元，KV、递归/卷积及推测状态归请求 State，scratch 归执行设施；不新增统一层联合类型或兼容壳。
- 同步加载、predictor、batch、session、Graph 和预分配消费者，移除 CudaCausalOps::blocks、通用 load_block 工厂及识别旧派生类的 dynamic_cast。
- 迁移相关 family 的旧 Engine 装配和 CudaExecutionContext 借用壳，清理 Executor 中失去用途的整层执行辅助；CUDA 生产消费者迁完后删除旧实现、模板和转发头。
- 收敛模型中剩余的 Transform 与计算图回调，由具体模型计算单元解释执行顺序。

保留数学、dtype、校验、CPU/layer placement、投影顺序、回滚事务和存储身份。暂缓的 Metal/HIP、测试与工具引用登记待恢复，不为其保留源码副本或旧抽象。
**源码完成条件：** 活动 CUDA 生产代码不再依赖公共 Block 运行框架及其类型探测，各模型只组合所需计算单元，状态与执行资源的所有者清楚。

## 3. 推广 Qwen35 并验收全链路

前两阶段完成后，将 Qwen35 的 embedding、forward、logits、replay 及已有 predictor/batch/session/Graph 适配接入公共模型、状态和 ops 契约，移除逐层 Executor/Collectives 参数及旧 Task 驱动依赖。
构建接线和测试工作恢复后，验证从加载、输入准备到输出和释放的真实路径；reference、eval、diagnostics、服务与受影响发行消费者一并验收。

| 检查面 | 必须覆盖的行为 |
| --- | --- |
| 加载与权重 | 默认 TQ、显式 MFQ 兼容、source/config/YaRN/placement 校验、RoPE/权重顺序、CPU 层数及实际支持或拒绝条件；编码 Tensor 的逻辑/物理字节、布局、别名和分组输出 |
| 前向与生成 | Full/Linear 混合层、普通/chunked prefill、decode、MRoPE、mask、显式位置、output norm、lm_head；比较 logits、有效状态与提交位置 |
| batch 与隔离 | 请求/slot 双向归属、页代次、批次变化、独立 scratch、取消及存储复用；模型 batch 状态只在模型侧解释 |
| MTP | 接受/拒绝、部分 replay、位置提交、target/predictor 及递归/卷积状态一致性；不支持的 suffix 保持明确拒绝 |
| session/prefix | 文本与媒体身份、候选 prepare/stage/drain/publish、命中记账、失败 fallback/reset、缓存预算及无关请求隔离 |
| Graph | 捕获与直接执行一致、warmup 隔离、state/scratch 地址和存储身份匹配、失效同步、最后消费者保活 |
| 执行与生命周期 | 后端分派及设备校验、scope 恢复、异步保活、加载失败、取消、背压、排空、reload、卸载、唯一终态、多 Engine 隔离和预算归还 |
| 输入输出 | 已支持的文本与多模态路径、模板/tokenizer、停止条件、增量文本及结构化结果 |
| 工具与发行 | reference 逐层低内存加载、独立 State/Executor 与执行作用域；eval/diagnostics 真实入口；受影响 JIT/wheel 安装后加载 |

保持固定输入、dtype、采样设置和原数值误差判据，不改变错误传播、恢复和释放顺序。
按受影响目标覆盖通信 on/off、默认 TQ 和显式 MFQ；TP/EP、CPU/layer placement 使用既有支持条件，不扩大能力声明。

## Qwen35 阶段完成条件

- CUDA 生产消费者完成公共契约迁移；Qwen35 的数学、状态及 batch/MTP/session/Graph 和媒体语义均归 models，通用运行模块无 family 私有依赖。
- 暂缓的构建、测试、工具及发行适配恢复并完成；生产、batch、reference、eval、diagnostics 和服务路径均有实际验证。
- 上表数值、状态、生命周期和失败路径通过验收。静态检查、skip、返回 77 或 model_branch=skipped 不构成运行通过；必要验证未完成时阶段保持未完成。

## 暂缓事项

| 项目 | 恢复后处理 |
| --- | --- |
| CMake 与旧构建入口 | 更新源码列表（含 Gemma4 model.cpp、engine/engine.cpp、scheduler/request.cpp）、include roots、目标依赖及 native/Torch 接线，切换旧 CUDA 目录作用域；保留数值选项和输出路径 |
| Metal/HIP 及旧目录收尾 | 按 tyloi 命名空间与 Tensor 抽象接口、指针/引用契约适配，同步 Gemma4 MLP 与共享 Attention 的加载接口，不保留兼容别名，不阻塞 CUDA 主线 |
| 工具接线 | 同步 reference、eval、KL、diagnostics 和 minicpmo45 命令中的模型/State、请求输出、加载与执行资源接口及 Torch 边缘转换，纳入整链验收 |
| 测试迁移、修改与运行 | 同步多态 Tensor 所有权、旧 Block/Qwen35 接口与路径假设，包括 shared_execution_test、kernel package 和 source archive 检查；保留原数值判据 |
| JIT、打包与构建/运行验收 | 同步源码、资源声明与安装后加载，验证真实生产路径 |

暂缓项不记为完成，不以旧接口、转发头或源码副本绕过；其他 CUDA family 的独立功能与整链验收后置。
命名空间迁移覆盖 TyloInfer 自有代码及必要消费者；MFQ 文件格式/编码标识和第三方命名空间按各自语义保留。

## 计划维护

- 完成的源码任务从计划删除，尚未完成的构建和运行验收保留；稳定职责以架构文档为准，Git 记录过程。
- 本次只更新文档，不修改或运行测试；后续按实际改动恢复必要验证，不降低原判据。
- 机器内存 32 GB，避免多份模型或多套重型验证同时驻留，不固定限制 -j。
