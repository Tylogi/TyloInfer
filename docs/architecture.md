# TyloInfer 架构

本文定义目标架构和模块职责，不表示源码已完成迁移。[实施计划](plan.md)负责当前状态、剩余任务与验收，
[重构约定](refactor.md)负责实施规则；架构描述存在冲突时以本文为准。

文中的示例代码和调用签名只说明职责、依赖与数据流，不要求逐字实现。
名称、参数形式和代码写法可按实际需要调整；正文明确的模块职责、所有权、数值和生命周期约束仍须满足。

本阶段先以 Gemma4 建立公共接口与 CUDA Engine 的完整链路，再由 Qwen35 复用，覆盖各自已支持的能力。
Metal/HIP 的编译、运行和兼容性不作为本阶段约束。

## 总体结构

`csrc/engine/` 是完整推理库。Engine 对象分别持有注入的 CausalModel 和 Executor，
拥有内部 Scheduler，并协调请求、生成流程及资源生命周期。
调用方提交请求、消费输出；Scheduler 在 Engine 内直接推进 generation。

```text
应用 / Transport → Engine：submit、cancel、读取输出、管理操作
                      │
              Scheduler → generation
                            ├─ Executor：建立执行作用域、管理执行资源与完成
                            └─ CausalModel::forward(input, state)
                                 → 模型内 Attention / MLP / Encoder.forward
                                 → 公共 ops：数值契约与设备分派
                                 → backend ops：借用当前作用域的执行资源
                                 → 同一 backend 的 kernels / native
```

requests、scheduler、generation、state、weights 是五个运行模块。
generation 负责生成推进；models 组合计算单元并调用公共 ops，既不接收也不保存 Executor。
`engine/ops.*` 定义公共数值接口并分派后端；`engine/executor.h` 定义执行作用域与生命周期契约，
设备实现位于 `engine/backends/<device>/executor/`。Executor 不转发模型前向或逐个算子调用。

TyloInfer 自有 C++ 命名空间统一为 `tyloi`，公共张量接口为抽象类 `tyloi::Tensor`。
激活、权重和状态载荷共用该接口；DenseTensor 与各编码 Tensor 实现具体数据表示，通过引用和智能指针使用，设备执行归后端。

## 最终目标目录

```text
csrc/
├── engine/
│   ├── engine.*            # 推理实例接口与生命周期协调
│   ├── executor.h          # 执行作用域、完成跟踪与生命周期契约
│   ├── tensor.*            # Tensor 抽象接口、公共元信息与所有权契约
│   ├── dense_tensor.*      # 稠密 Tensor 与 view 实现
│   ├── ops.*               # 公共数值接口、参数与后端分派
│   ├── config.h            # 实例级配置
│   ├── budget.*            # 跨模块资源预算协调
│   ├── requests/           # 请求/结果契约、输入准备、输出解释
│   ├── scheduler/          # 请求生命周期、调度、背压与合批
│   ├── generation/         # prefill/decode、采样与 MTP 策略
│   ├── state/              # 通用状态载荷、页、快照和 session
│   ├── weights/            # 模型文件读取、权重准备与驻留
│   ├── models/             # 模型定义、装配、前向和专属状态
│   └── backends/
│       ├── cuda/           # executor、ops、kernels、native
│       ├── metal/
│       ├── hip/
│       └── cpu/
├── components/             # 通用异常、IO、JSON、tokenizer、媒体编解码
├── transport/              # 外部协议解析与结果传输
├── apps/                   # 可执行入口与依赖装配
├── commands/               # 管理命令及装配操作
├── eval/
├── diagnostics/
└── tests/
```

下文的运行模块、models、backends 路径均相对 `csrc/engine/`。
source 指 `weights/source/`；公共 ops 位于根目录 `ops.*`，设备算子、kernels、native 位于对应 backend。
各设备共享运行模块的策略和生命周期，设备计算与原语集中在 backends。

## 模块内部结构与文件职责

`.*` 表示按实际需要使用 `.h/.cpp/.cu`。文件划分表达职责，不要求每个文件新增一个类；
共享实现以实际复用为依据，其他模型和后端按各自能力组织文件。

### Engine 根目录

| 文件 | 职责 |
| --- | --- |
| engine.* | 提交、取消、输出访问、状态查询、会话控制、卸载与依赖重新绑定 |
| executor.h | 执行作用域、工作完成与排空契约；不包含 project、norm、attention 等数值接口 |
| tensor.*、dense_tensor.* | Tensor 抽象接口与 DenseTensor 实现；逻辑元信息、存储枚举及指针/引用契约见[统一 Tensor 契约](#统一-tensor-契约) |
| ops.* | 公共数值契约、参数与设备分派；接口不暴露设备私有类型，实现在构建允许的后端中选择 |
| config.h | 实例级配置，模块专属选项由对应模块定义 |
| budget.* | state、weights、执行工作区之间的预算协调，以及批量 prefill 预算辅助，各模块管理自身资源 |
| metrics.h | Engine 实例资源聚合与统计 |
| CMakeLists.txt | 按真实依赖组织运行核心、模型与设备支持的编译目标 |
| cuda.cmake | CUDA 实例装配及运行时构建定义 |

### requests

| 文件 | 职责 |
| --- | --- |
| request.h、result.h | 外部请求、增量结果、用量、终态与错误契约 |
| media.h | 图片、音频、视频输入的数据契约与顺序信息 |
| processor.* | 输入校验、模板应用、tokenize、约束准备及模型输入准备的组织 |
| tokenizer.* | 通用 tokenizer 的推理侧适配，供输入和输出共用 |
| constraint.*、sampling.h | 语法约束构造与状态、采样参数定义 |
| output.* | token 转文本、结构化结果解析、结束结果 |
| emitter.* | UTF-8 增量拼接、停止串与文本片段处理 |

requests 通过模型公开契约获取输入规则、tokenizer 资产和协议语义，形成 owned 输出结果。
请求排队、取消、输出通道和终态发布由 Scheduler 管理。

### scheduler

| 文件 | 职责 |
| --- | --- |
| scheduler.* | 请求表、调度循环、准入、取消、截止时间、背压和唯一终态发布 |
| request.* | 活动请求记录：准备状态、进度、模型状态持有、生成句柄与输出通道 |
| batch.* | 请求组合、slot 分配、批次成员及本轮工作列表 |
| policy.* | 优先级、prefill 预算与工作选择 |

requests/request.h 是提交契约，scheduler/request.* 是唯一的运行记录。
Scheduler 决定运行哪些请求及多少工作；generation 推进计算，models 定义批次状态布局。
批次引用同一份活动请求，不独立决定请求终态。

### generation

| 文件 | 职责 |
| --- | --- |
| generation.h | 分块 prefill、decode 推进，调用模型、采样及 session 操作 |
| sequence.*、step.* | 生成协程、有界推进、进度与中断点 |
| prefill_activity.* | prefill 活动计数与近期活动记录 |
| batch.* | 通用批量生成推进，调用模型批量能力 |
| prefill_budget.h | 纯 prefill 预算计算与批次选择辅助 |
| sampling.* | 采样策略、惩罚与约束推进，设备计算交给后端 |
| mtp/sequence.* | 通用 MTP 推进、验证和接受流程，并承载执行条件检查 |
| mtp/policy.*、mtp/depth_controller.* | 推测策略和深度选择 |
| mtp/metrics.* | MTP 通用指标 |
| mtp/state.h | 实例级深度策略状态与上一轮生成统计，由 Engine 持有 |
| mtp/batch_policy.* | 批量推测策略与深度选择 |

generation 借用 Engine 持有的模型与执行器，在推进或恢复协程的外层建立执行作用域，
再调用模型前向及公共 ops 完成采样。forward 返回 hidden 等模型结果，按需经模型侧 logits 投影后采样；
具体结果语义由模型契约定义。生成代码不持有 family 私有布局，也不向模型传递执行器。
通用 MTP 推进、验证和接受逻辑归 `generation/mtp`；模型专属深度策略归
`models/<family>`，由预测器提供深度策略和 draft 采样参数；默认自适应与批量策略仍由
`generation/mtp` 提供。通用生成代码不依赖具体 family 类型。
深度策略状态与生成统计由 Engine 持有；预测器的缓存及位置归请求的 MTP State。

### state

| 文件 | 职责 |
| --- | --- |
| kv_cache.*、recurrent.* | 通用 KV、递归及卷积状态载荷与操作 |
| pages.h、paged_kv.* | 页身份、引用、分配代次、分页管理及 KV 绑定 |
| kv_memory.*、kv_budget.h | KV 容量、预算、分配与回收 |
| kv_offload.* | 设备/RAM/SSD 转移；KV 页映射策略留在 state |
| qsa_blocks.*、disk_budget.h | 不透明 KV 字节块的 RAM/SSD 存储及磁盘预留协调 |
| session/cache.* | session 匹配、存储、预算与淘汰 |
| session/prefix_cache.* | 前缀匹配、共享与复用 |
| session/prefix_blocks.*、session/paged.* | 前缀字节块存储与会话归属 |
| session/snapshot.h | 通用会话快照的匹配、预算与淘汰 |

state 管通用存储机制及 session/cache 生命周期；模型的快照布局、编码与校验、完整边界判定、
联合恢复和位置语义由对应 family 定义。缓存仅消费共用元信息和 Tensor 枚举，按实际 backing 身份去重统计。

### weights

| 文件或目录 | 职责 |
| --- | --- |
| loader.* | 原始资产读取与具体 Tensor 准备；权重返回 `shared_ptr<const Tensor>`，分组返回有序权重指针列表 |
| placement.h、group.* | 执行放置、分片及有序分组；具体分片布局留在实现内 |
| cache.*、workspace.* | 驻留、预算、租约及权重预取/展开/重排缓冲区 |
| memory.* | 记录权重常驻与 PLE 统计 |
| source/source.h、source/read.* | 只读 Source 契约与 tensor/资产读取辅助，不包含模型架构、布局或 graph |
| source/gguf.*、safetensors.*、hf.*、mfq.* | 容器、分片索引与字节范围读取 |
| encoding/ | TqTensor、NintTensor、NvqTensor、MxTensor 等编码实现与布局校验；复用位布局、scale/codebook、行/块解码和打包逻辑 |
| mx.h、sq.h、nint.h、vq.h | 对应编码 Tensor 的准备与格式辅助；复用 encoding 实现，数值计算归 backend ops |
| nint_cache.h、nint_row_cache.h | 单层 prefill 展开缓存、原始编码行及借用租约 |
| row_stage.h | 已准备的表行上传、解码与稳定输出存储契约，编码实现由对应权重准备持有 |
| bias.h | 不带计算方法的投影权重/bias 数据与稠密行切片准备；读取和校验由 Loader 提供 |
| fields.h | 模型 Tensor 字段遍历、传输和字节统计；按实际存储身份去重 |
| moe/、tq/ | 对应权重的运行时表示、准备、传输与缓存实现 |
| moe/pool.h、dispatch.h、prepare.* | 专家池、激活几何、混合编码调度元数据及异步上传 backing 生命周期 |

权重准备、驻留、传输和缓存策略归共享的 `engine/weights`，计算 scratch 与设备传输原语
分别归后端 Executor/native；backend 不按权重或上层模块再建镜像目录。
source 是 weights 的内部实现，模型工厂通过统一入口访问数据和准备权重。

### models

```text
models/
├── CMakeLists.txt     # 模型实现目标与源码清单
├── source.*           # ModelSource 的模型语义契约与文件格式选择工厂
├── causal_model.h     # 模型行为接口、能力与模型状态契约
├── input.h            # 模型前向输入：token、位置、embedding 等
├── prepare.*          # 主机侧模型输入准备与停止 token 解析
├── load.*             # 模型工厂，返回 CausalModel
├── session.h          # 模型会话组合与 Tensor 枚举
├── session_codec.*    # codec 选择与 prefix 身份
├── common/
│   ├── causal_metadata.h # 共用的模型能力与配置描述
│   ├── model_config.*、model_graph.*、capabilities.h # 共享配置、结构和能力判定
│   ├── mfq_source.*、hf_source.*、gguf_source.* # 各容器的模型名称、布局和 graph 解释
│   ├── tq_fixture_source.* # 显式合成输入的模型视图
│   ├── weight_loading.h、load.*、components.* # 实际共享的权重装配与计算单元
│   ├── transformer_layer.h # 实际共享的层序与 HC/MHC 组合，无通用 Block 继承树
│   ├── batch.h、prompt.h、prefetch.h # 共享批次、输入与权重行预取适配
│   ├── generation_ops.h、mtp_ops.h、mtp.h # 共享模型行为与预测器契约，推进循环归 generation
│   ├── session/       # state.*、paged.*、size.h、cache.*：快照、读写、峰值与恢复
│   └── vision/
│       ├── grid_vision.* # grid 配置、几何与位置规则
│       ├── grid_forward.h # grid 编码、merge 与输入准备模板
│       ├── grid_encoder.* # 共享 grid 编码器、私有层权重和前向
│       ├── axial_patch_vision.* # axial patch 配置与几何规则
│       └── axial_patch_encoder.* # axial patch 编码器及设备适配
├── qwen35/
│   ├── config.*       # 配置解释与校验
│   ├── model.*        # CausalModel 实现、模型定义及组件持有
│   ├── weights.h      # 按层和角色组织的模型权重类型
│   ├── load.*         # 权重名称映射、模型装配
│   ├── forward.*      # embedding、层序、norm、hidden/logits 计算
│   ├── attention.*    # Full/Linear Attention 计算单元及数学步骤
│   ├── mlp.*          # MLP/MoE 计算单元；是否拆文件按实际规模决定
│   ├── state.*        # 请求/批次状态、位置语义、Graph 状态访问
│   ├── session.*      # 模型快照校验与联合恢复
│   ├── batch.*        # 模型批次状态及接口
│   ├── batch_ops.*    # 批次状态操作与模型批量执行
│   ├── predictor.*    # MTP predictor 定义与权重
│   ├── mtp.*          # 预测、提交、回滚适配
│   ├── mtp_state.*    # 请求相关的 MTP 可变状态
│   ├── input.*        # 特殊 token、媒体位置与预处理规则
│   └── vision.*       # 视觉编码、投影与文本融合
└── <family>/           # 其他 family 的配置、前向、状态和适配
```

模型按 family 组合 Layer、Attention、MLP 和 Encoder，计算单元持有只读权重与配置并提供 forward。
不强制统一基类、通用层联合类型或每单元一个文件；结构与契约真正一致的部分才归 common。
例如层前向依次调用 `attention.forward(x, state.attention, step)` 和 `mlp.forward(x)`，
单元内部调用 `ops::project` 等数值接口；残差、归一化、门控及其顺序由模型定义。
Qwen35 的线性递归与 softmax attention 使用不同数值契约，不能靠全 family 标志拼成万能 attention。
KV、递归/卷积、位置和推测状态随请求或批次持有；模型和计算单元均不保存 Executor、活动 State 或 scratch。
Graph 的模型适配可留在 model.*、state.*；需要设备适配的模型语义仍归对应 family。
模型计算快照数据量及复制峰值；实例协调缓存淘汰和空间预留，通用生成代码不读取 family 私有缓存布局。
模型决定 prefill 的逻辑推进范围；计算块大小和空间预留仍受实例预算约束。
视觉编码器的共享范围、输入输出及资源契约见[视觉编码器设计](#视觉编码器设计)。

### backends

```text
backends/
├── dtype.h                # 共用标量类型定义，不含设备句柄
├── cuda/
│   ├── executor/
│   │   ├── executor.*     # 执行作用域、资源复用、完成跟踪与排空
│   │   ├── config.h       # 构造时读取的现有 CUDA 执行配置
│   │   ├── options.*      # 动态 CUDA 执行选项及该配置的读取
│   │   ├── batch_graph.*        # 批次图状态、键和失效
│   │   ├── decode_graph.*       # 捕获、输入传输、缓冲区、复用与失效
│   │   ├── graph_registry.*     # 捕获图登记与执行生命周期
│   │   ├── moe_ffn_pipeline.*   # MoE 流水线执行与工作区
│   │   └── pipelines.cpp       # 流水线持有、准备及排空
│   ├── ops/               # attention、ffn、projection、norm、rope、vision、sampling 数值实现
│   ├── kernels/           # 按算子族组织的 .cu/.cuh 内核
│   └── native/            # Tensor 存储与设备原语、内存、stream/event、通信
├── metal/
├── hip/
└── cpu/
    ├── native/            # 线程、内存复制、能力检测与计时
    ├── executor/          # 专家行任务的 CPU worker pool
    └── ops/projection/    # 压缩权重的主机数值计算
```

backend 负责设备执行，共用 Engine、Scheduler 和 generation。不同设备按实际能力实现，
不要求具有相同的 Graph 机制或文件清单。以下路径相对 backends/cuda：

各设备的 `backends/<device>/` 仅承载 `executor`、`ops`、`kernels` 和 `native`；根 `backends/`
提供共享 `dtype.h` 与设备目录。设备目录禁止再次镜像
`engine` 下的 `requests`、`generation`、`scheduler`、`state`、`weights` 或 `models` 等上层模块目录。
通用机制归外层 `engine` 的同名模块，模型知识归 `models`；设备 API 的调用位置不改变上层业务归属。

| 文件或目录 | 职责 |
| --- | --- |
| ops/attention/ | softmax attention、递归更新等明确数值契约及融合实现；不组织模型整层 Attention |
| ops/ffn/ | dense、MLP/MoE、MFE、激活、残差、路由及工作区布局；存储复用与保活归执行设施 |
| ops/projection/ | 投影、bias、embedding 及编码相关实现；读取、切片和权重准备归 weights |
| ops/norm/、ops/rope/ | 归一化、位置编码及 YaRN 数值实现；模型提供位置规则和公共参数 |
| ops/vision/attention.h | 非因果/分段 attention 与轴向 RoPE 数值辅助；不包含完整编码器或 Transformer 层定义 |
| ops/sampling.* | 采样、惩罚、约束及 MTP 概率计算；生成策略归 generation |
| kernels/ | 内核和 prepack 入口；所需 GGML host helper 符号由 native/ggml.cu 提供 |
| native/device.*、scope.h | 错误检查、设备/context 选择和作用域恢复；不包含模型或实例管理 |
| native/tensor/、memory.*、allocation.h | 各 Tensor 实现复用的设备内存、临时 kernel view、分配和映射原语；不按设备复制 Tensor 继承树或处理驻留预算 |
| native/stream.*、graph.* | stream/event 所有权及 Graph 捕获、上传、重放和销毁原语 |
| native/copy_batch.h、dma_batch.h、vmm.* | DMA、虚拟地址及物理页映射；权重缓存、上层批次与 KV 映射策略归所属模块 |
| native/collectives.*、native/branches.*、native/profile.* | peer access 与通信、设备执行分支与 profiling |

backend ops 负责已定义数值操作的算法、融合、kernel 选择及启动，通过内部执行设施使用工作区与同步原语。
不能接收整个 Model、按 family 分支或回调模型前向；模型 Attention/MLP 的步骤留在 models。
kernels 实现设备计算，native 提供设备原语；stream 所有权与当前执行选择分别管理。

## Engine 实例目标设计

Engine 通过构造函数分别接收 CausalModel、Executor 和实例选项；资源装配见下文。

| 对象 | 所有权与行为 |
| --- | --- |
| Engine | 持有模型、主 Executor、Scheduler 及资源所有者，协调实例生命周期和预算 |
| CausalModel | 组合计算单元、持有只读权重；提供能力、输入准备、State 创建、forward/logits 及 batch/MTP/session 行为 |
| Executor | 管理执行作用域、stream/event、工作区、Graph 执行资源和异步完成；不包含模型数学或数值转发接口 |
| 模型 State | 由 CausalModel 创建，由请求或批次持有；具体布局和恢复行为由模型定义 |

模型前向例如 `model.forward(input, state)`，计算单元只接收输入、所需状态和本步数学参数。
不在构造或 forward 中逐层注入 Executor，也不提取 Projector、profiler、stream 或 scratch 传给计算单元。
模型不绑定某个 Executor 对象；权重、状态和算子仍须满足设备相容性，切换后端不等于可直接复用原设备数据。

| Engine 公共能力 | 契约 |
| --- | --- |
| submit | 接收 owned 请求，返回输出句柄；成功提交表示进入管理，不保证立即执行 |
| cancel | 标记并唤醒取消处理，内部循环完成清理与终态发布 |
| 请求输出 | 有界事件通道，包含增量结果、用量及完成/取消/失败事件 |
| info/status、session/control | 能力与状态查询、会话及控制操作；冲突由内部循环协调 |
| 卸载、重新绑定、shutdown | 静止请求、等待设备消费者、释放或接纳依赖，完成线程及资源收尾 |

Engine 自己推进执行。准入和有界 step 属于内部机制，Scheduler 直接调用 generation。
设备作用域在实际执行线程建立；外部线程通过公共接口提交请求和控制操作。

### 模型装配与重载

apps/commands 根据模型与设备配置选择相容的模型工厂和后端执行器，分别注入 Engine。
Engine 不解析模型路径或持有加载回调。以下代码仅示意装配关系，函数名与参数不要求照搬：

```cpp
auto weight_loader = weights::open(model_path, weight_options, weight_resources);
auto executor = make_executor(device_options, execution_resources);
std::unique_ptr<models::CausalModel> model;
{
    auto scope = executor->enter(); // 加载中的设备操作也使用所属执行环境
    model = models::load_causal_model(weight_loader, model_options);
}
auto engine = std::make_unique<Engine>(
    std::move(model), std::move(executor), engine_options);
```

工厂返回 CausalModel，完成配置、权重角色和组件定义的装配；活动 State 在请求或批次建立时创建。
伪代码省略根资源的显式移交，权重、reader、设备资源的持有关系必须覆盖所有借用者。
加载失败清理未移交对象和设备工作，不发布可用实例。

reload 由装配入口协调：Engine 停止接收，清理请求并等待设备消费者，释放 Graph、
批次、session、模型状态和执行依赖；随后装配入口构造新依赖，Engine 校验并绑定后恢复接收。
默认先释放旧资源再加载，避免两份模型同时驻留；失败时实例保持未加载状态并返回错误。
请求、状态和 Graph 不跨依赖绑定复用，Scheduler 始终由同一 Engine 管理。

### 公共 ops 与后端分派

公共 ops 定义数学语义、shape/dtype、输出与别名、显式状态修改及错误契约，参数不包含 Executor。
模型和算子通过 Tensor 引用读取数据，以智能指针持有结果；普通运算按输入设备分派，权重运算结合布局、执行放置与实际驻留校验相容性。
后端按形状、逻辑 dtype 和存储编码选择实现；保留既有 host 计算、暂存、分片及按需专家路径，不以 CPU backing 推断执行后端。
普通算子校验输入、权重和状态的设备相容性；copy 等跨设备操作明确源和目标，无输入的创建操作显式指定 device。
先用简单分派，不建立通用注册系统；每次公共调用选择一次后端，内部 kernel 不重复做跨后端分派。
未实现的组合明确拒绝，不隐式搬数据或回退 CPU。张量设备不能决定 stream，stream 由执行作用域选择。

模型 Attention 组织 QKV、QK norm、RoPE、门控和输出投影等步骤；公共 attention 算子只表达明确的数值契约。
融合算子可以覆盖多步，但须明确计算范围，保留残差、dtype 转换、累积顺序和原数值判据，不根据模型名称选择算法。
requests/request.h、models/input.h 与算子输入分别表达请求、模型前向和数值操作；公共接口不暴露 CUDA 私有类型。
现有 CUDA Task/Inputs 仅属后端私有实现，迁移后无用途的部分删除，不保留公共 Task IR 或第二套算子入口。

### Executor 与执行作用域

Executor 协调下列执行资源，内部复用 native、工作区和 Graph 实现；模型层序、请求调度、权重及 KV 管理各归原所有者。

| 执行职责 | 契约 |
| --- | --- |
| 当前执行环境 | 进入时保存并激活线程上的设备、stream 和执行资源绑定；退出或异常时恢复，后端算子从内部入口借用 |
| 分支与依赖 | 算子决定哪些数值步骤可并行，执行设施提供 stream/event、分支与汇合；不选择运行哪个请求 |
| 计算工作区 | 算子定义布局和容量需求；执行设施负责复用、并行隔离及最后消费者结束后的回收，不接管权重或 KV 缓存 |
| 完成与排空 | 跟踪提交和完成，保活 Tensor backing 与借用租约；取消后停止后续提交并收尾，卸载/reload 前排空 |
| Graph 与辅助资源 | 管理捕获/重放资源、固定缓冲区、计算库句柄和可选 profiler；模型提供状态条件，generation 决定何时使用 Graph |

scope 是当前线程的临时绑定，不是资源所有者；不保存“当前模型/请求/State”，不设置进程级当前 Engine。
复用已有 ContextGuard、StreamGuard 等机制，资源由所属执行器或租约保活，不把旧 CudaExecutionContext 整体改名搬入。
scope 覆盖一段连续提交工作的主机执行区间：通常是一轮 decode、一个批次或 prefill chunk，也可包含多个 MTP forward。
不固定对应一个 token，不覆盖整个请求或等待期，也不在每次进入时重建 stream。

协程挂起不会析构其局部变量，因此 scope 必须在恢复协程的外层建立，不能作为协程内跨挂起点存活的局部对象：

```cpp
{
    auto scope = executor.enter();
    sequence.resume(); // 本次执行到下一次挂起或结束；forward 不接收 executor
}                      // 恢复线程环境，不隐含 GPU 同步
```

每次恢复或转交其他线程执行时重新进入所属作用域；同步路径按相同规则包住实际提交区间。
退出 scope 不意味着 GPU 完成，不回收仍在使用的工作区、输出 backing、state view 或权重租约；
Tensor 结果可先于设备完成返回，跨 stream 消费建立依赖，主机读取和资源退休按完成事件或 drain 等待。

## 模型文件、权重与模型语义

| 解释与管理对象 | 唯一归属 |
| --- | --- |
| 文件容器、原始 metadata/资产、tensor 名称与字节范围 | weights/source |
| 数值编码的位布局、scale/codebook、block/row 规则 | weights/encoding |
| 权重准备、放置、传输、驻留、按需专家缓存与租约 | weights |
| 模型配置、graph、权重角色、名称映射及模型专属布局 | models |

模型工厂通过 weights 读取配置、资产和 tensor；Source 提供原始 metadata、字节范围和有界 reader，
ModelSource 解释模型架构、名称和 graph。模型解释归 mfq-model-source，原始读取与准备归 mfq-weight-source。
source 读取容器记录的 shape/dtype/编码标识，encoding 解释编码，models 决定权重角色和模型布局。
初次加载与按需加载共用 reader、文件和映射持有者，不默认展开整个量化 tensor；配置、tokenizer 和模板共用解析结果。
纯编码布局可被 source、ops/kernels 使用，不依赖模型或缓存管理对象。

### 统一 Tensor 契约

`tyloi::Tensor` 是带虚析构的抽象数据接口，基类只承诺各实现共有的逻辑 shape/dtype、布局识别、存储枚举/访问与生命周期契约。
DenseTensor 表达稠密数据；TqTensor、NintTensor、NvqTensor、MxTensor 等按数据表示继承 Tensor，格式字段与布局校验归各实现，复用 weights/encoding。
具体实现复用 TensorStorage 的缓冲区所有权与 backing 身份；基类不堆格式字段，派生类不持有完整 QuantLinear/Experts、router、加载器或执行管理者。

拥有对象使用 `shared_ptr<Tensor>`，只读权重使用 `shared_ptr<const Tensor>`；只读算子输入为 `const Tensor&`，原地修改使用 `Tensor&`，结果返回拥有对象的智能指针。
抽象 Tensor 不按值构造、复制或返回；复制智能指针共享对象，张量 view 使用独立元信息对象并共享 backing，clone 创建独立数据，借用引用不延长寿命。

| 数据 | 逻辑表示 | 算子使用方式 |
| --- | --- | --- |
| 激活、norm、bias、KV/递归载荷 | DenseTensor 或相容的编码实现 | 公共算子处理，模型 State 组织状态语义 |
| 普通/量化权重 | 逻辑 `[输出维度, 输入维度]` | project、embedding 等通过 Tensor 引用读取 |
| 专家权重 | 逻辑 `[专家数量, 输出维度, 输入维度]` | 专家算子显式接收路由；混合编码、分片和按需读取由具体表示组织 |
| 分组权重 | 有序只读 Tensor 指针列表或具名模型字段 | project_group 返回同序结果指针；保留联合投影与物理打包 |

所有实现的 shape/dtype 表达逻辑数值，nbytes 表示逻辑展开字节数；实际存储计量枚举全部数值、scale/codebook 等缓冲区，按 backing 身份去重。
stride、data_ptr 和直接访问的 TensorView 属于稠密实现或受约束的 view；基类不承诺任意布局都能裸指针访问，模型不解释编码载荷。
设备拷贝保留编码并覆盖全部辅助数据；cast、reshape、切片、索引和分组逐项定义布局、别名与物化规则，未支持组合在修改或提交前拒绝，不隐式整模解量化。
执行放置与动态驻留分别表达，分片校验覆盖所有分区；Tensor 派生类按数据表示划分，设备差异由 backend 处理。

数值入口统一归公共 ops，按设备、逻辑 dtype 与布局选择已有后端算法；Tensor 不承担计算虚函数，backend 直接调用 native/kernel，避免反调公共算子。
纯元数据/CPU 操作无需 CUDA scope；CUDA 提交由装配、加载或调度入口建立所属 Executor 作用域，Tensor 不自行创建 Executor 或 stream。

Loader 创建具体实现并通过只读 Tensor 指针发布，模型提供名称、角色、分组顺序和形状要求；所有别名遵守权重只读约束。
Tensor 对象保活其全部 backing；reader、映射、驻留缓存、预取与传输租约归 weights，计算 scratch 归执行设施，按需专家向后端提供本次可读 view 与租约。
对象/指针寿命与设备完成分开管理：租约覆盖最后消费者，输出在归还借用存储前迁出；加载失败、取消、卸载和 reload 遵循同一退休规则。
缓存与 Graph 根据实际 backing、租约和存储身份复用，不以 Tensor 对象地址或形状相等代替，不新增全局身份注册系统。

## 一个请求的完整执行链

1. **提交与排队。** Engine 返回请求输出句柄；Scheduler 登记请求，按优先级、容量和截止时间准入。
2. **准备输入。** requests 完成校验、模板、tokenize、约束准备，调用模型输入处理。
   token、媒体位置、embedding 与上下文占用保持一致，耗时准备可分段推进并响应取消。
3. **准备状态。** 模型创建请求或批次 State；匹配文本和媒体身份后恢复 session/prefix，或初始化新状态。
4. **prefill。** Scheduler 选择 chunk 或批次；generation 在执行作用域内调用模型前向，消费未复用的 prompt，
   更新 State，并通过模型侧 logits 投影获得所需位置的采样分数。
5. **采样与 decode。** generation 选择 token，随后反复将生成 token 输入模型，计算下一轮结果；
   MTP 和 Graph 按能力启用；每段推进重新激活执行环境，模型计算单元通过公共 ops 完成数值运算。
6. **输出。** requests 解释 token、处理停止条件并形成增量结果；Scheduler 发布到有界通道，按背压暂停后续工作。
7. **收尾。** 正常结束、取消和失败均完成 session 保存策略、设备与请求资源清理，再发布唯一终态。

首个生成 token 来自 prefill 最后位置的 logits；已输出 token 通常在下一轮 forward 中消费。
状态位置按实际完成的输入提交，MTP 的接受、拒绝与 replay 遵守同一规则。
一次内部推进可以处理媒体片段、prefill chunk、多个 token 或等待批次结果，不固定对应一个输出 token。

## 文本与多模态输入输出

| 阶段 | 归属 |
| --- | --- |
| 外部消息、媒体字段和顺序解析 | transport；requests/media.h 保留输入对应关系 |
| JPEG/PNG、音频和视频字节编解码 | components 中的通用工具 |
| 模板执行、tokenize、通用语法约束、增量结果解析 | requests，复用 components/tokenizer |
| 缩放、裁剪、归一化、patch/grid、音频特征和时间/位置规则 | models/<family>；实际共享部分放 models/common |
| 媒体占位符、特殊 token、视觉/音频编码、投影及文本融合 | models/<family>，通过公共 ops 完成计算 |
| 模型专属输出解码与协议语义 | models 提供；requests 组织结果，transport 编码并传输 |

模型输入输出规则通过公开契约提供，requests 与 Scheduler 不维护按 family 分支的处理表。
上述是目标职责划分，输入类型和媒体能力的支持范围由模型公开能力声明。

### 视觉编码器设计

对外模块是具体视觉编码器，入口例如 `encode(input)`，有状态时显式传入；各编码器定义自己的输入与结果契约。
层权重、attention/MLP 组合和残差顺序是编码器内部实现，不设通用 VisionBlock、vision::Layer、
视觉基类或回调装配表。Engine 通过 CausalModel 组织输入准备，不直接选择视觉编码器。

| 模块 | 归属与共享范围 |
| --- | --- |
| SigLIP 编码器 | `models/minicpmo45/vision.*`；保留独立 Q/K/V、GELU tanh、padding mask 及本模型的 patch/位置规则 |
| RADIO 编码器 | `models/nemotron_h/radio_vision.*`；保留融合 QKV 的布局解释、GELU erf、CLS/register token 和位置插值规则 |
| grid 编码器 | `models/common/vision/grid_encoder.*`；供已有 grid 模型复用，拥有 grid 层权重、位置编码与分段 attention 的组合 |
| axial patch 编码器 | `models/common/vision/axial_patch_encoder.*`；保留 RMSNorm、轴向 RoPE、gated MLP 及已有空间对齐计算，不套用 grid 或 SigLIP 的层定义 |
| 视觉结果到文本输入的适配 | 对应 `models/<family>/vision.*`；组织 resampler/merger/projector/aligner，完成输出选择、媒体占位与文本融合；相同数值模型按实际复用归 common |

共享限于已有 `pre_norm_layer`/`mlp` 辅助、backend 数值算子，以及实际使用同一结构和契约的编码器。
不为相同的 pre-norm 层序再封装公共 Block。CUDA 模型前向仍在 models，可用 `_cuda.*` 区分设备实现。

**输入输出。** 输入为已预处理的 pixels/patches 及该编码器需要的 grid/mask；明确 shape、dtype 和有效区域。
输出明确特征宽度、patch 顺序和 CLS/register 保留规则，复用已有 Tensor/grid 类型；axial patch 的输出
已包含空间对齐和文本宽度投影，调用方不重复投影。family 适配生成 `[N_media, D_text]` 的媒体 embedding
（N_media 为媒体 token 数），校验各段数量、顺序与占位范围后融合文本。编码器不解释 prompt token 或修改文本 KV。

**执行与所有权。** 编码器持有只读配置和权重句柄，通过公共 ops 计算，不注入或提取 Executor/Projector。
耗时编码由外层在每次 StepSequence 恢复时建立 scope；激活、进度与结果归请求或协程，媒体缓存由 state 管生命周期、模型定义身份。
编码器不持有主执行器或活动请求，不直接操作 MoeExpertCache。权重准备与租约归 weights，计算 scratch 归 executor；
复用底层存储时，结果须在归还前迁出，租约覆盖最后设备消费者，yield/取消/失败遵守同一生命周期。

**权重与算子。** 模型定义 QKV 角色、层权重枚举及 mask/位置规则；weights 提供通用权重表示、字段枚举、
读取、切片准备和传输，不依赖模型层。带 bias 的投影归 ops/projection，供视觉、音频和 TTS 共用；norm、
attention、RoPE、GELU 归对应算子族。权重数据与计算分开，不以视觉类封装 `load/rows/forward`。
投影保留两种计算契约：按权重 dtype 完成 dense+bias 后转回输入 dtype，以及按 projector 结果 dtype 加 bias。
GELU tanh/erf、softmax 精度、attention 输出 dtype、mask、contiguous 行为和融合 QKV 的 dense 切片限制均保持不变。

## 状态、权重与生命周期

活动请求与批次拥有模型 State，state 管通用载荷，weights 管文件及权重驻留，
后端 Executor/native 管执行工作区和设备资源。Engine 持有实际所有者并协调依赖顺序。

- 请求逻辑状态独立；物理存储共享时保留请求、slot、页和分配代次的身份。
- Graph、scratch、state view、传输 backing 和权重租约存活到最后设备消费者结束。
- session/control、预算调整和依赖替换由内部循环协调；需要静止的操作等待冲突请求和设备工作结束。
- shutdown 停止接收，取消并清理请求，等待设备消费者和执行线程退出，再按依赖释放资源。
- 每个成功提交的请求都有唯一终态；输出尾部和清理结果先处理，错误区分请求失败与实例不可用。

### session、MTP 与 Graph

| 能力 | 归属 |
| --- | --- |
| Session/prefix 存储、匹配、预算与淘汰 | state/session |
| 共享 KV 快照、编码与恢复 | models/common/session/state.* |
| 模型快照字节读写工具 | models/common/session/paged.* |
| 当前会话组合及 codec 选择 | models/session.h、models/session_codec.* |
| family 快照布局、编码与校验、完整边界判定、target/predictor 联合恢复、位置语义，以及持有 Tensor 的枚举 | models/<family> |
| MTP draft/验证/接受策略与生成推进 | generation/mtp |
| CUDA Graph 捕获、缓存、重放、失效和资源保活 | backends/cuda/executor/ |
| 模型捕获条件、状态访问与恢复 | models/<family> |
| CUDA Graph 底层句柄操作 | backends/cuda/native/graph.* |

session 恢复先准备、校验候选并等待设备消费者，再提交；失败保持一致的恢复或重置语义。
模型块快照负责自身布局，并通过 `tensors(visitor)` 枚举持有的 Tensor。缓存消费遍历结果，
按实际 backing 身份去重，计量共享存储的物理字节数，不重复累加别名 view。
MTP 的递归/卷积状态、位置与必要 hidden 数据共同遵循接受和回滚规则。
Graph 先取得捕获 stream 和所属 Context，再写入成员；上传后的预算检查使用持有的 Context。
Graph 复用依赖实际批次布局、state/scratch 地址和存储身份；warmup 隔离生产状态，
slot/page 复用、批次变化和状态恢复触发相应同步或失效。generation 协调何时使用和释放图。

## 产品装配与依赖方向

产品装配为 `Runtime = Transport + Engine`。apps/commands 创建模型和执行器并注入 Engine，
组合 Transport；加载与 reload 命令走装配入口，再调用 Engine 的生命周期接口。
Transport 的推理请求只使用 Engine 公共接口。

- requests、scheduler、generation 使用公共行为契约，不依赖具体模型或设备私有实现头。
- models 组合计算单元，使用公共 ops、Tensor、权重及状态契约；模型语义及设备适配归 family，不引用 Executor。
- 公共 ops 分派设备实现；backend ops 接收权重、输入和显式状态 view，从当前执行环境借用资源，kernels/native 提供设备能力。
- 文件来源与编码实现不依赖具体模型，设备原语不依赖请求调度或 Engine 驱动。

运行核心、模型实现和设备支持按真实依赖构建，最终推理库或入口完成组合；
公共头不隐式引入具体 family，不按每个子目录机械增加编译目标。
源码、CMake、JIT/include、工具与打包最终须一致；当前实施顺序和暂缓范围以计划为准。
CUDA 的三个可执行入口位于 `csrc/apps/`，命令、评测和诊断实现分别位于
`csrc/commands/`、`csrc/eval/` 和 `csrc/diagnostics/`。设备与模型各有 CMakeLists.txt，实例装配由 engine/cuda.cmake 组合，
诊断与入口分别由 diagnostics/cuda.cmake、apps/cuda.cmake 定义；旧构建作用域的切换与删除按计划恢复后处理。
CUDA 内核的唯一源码根是 `csrc/engine/backends/cuda/kernels`，native/JIT/wheel 共用。
Python 包与 CLI 为 `tyloinfer`，服务入口见 [pyproject.toml](../pyproject.toml)。
架构调整保持模型数学、dtype 和数值校验要求；离线量化、校准产品入口不在本架构范围。
