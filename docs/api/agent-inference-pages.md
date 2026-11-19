# Agent 推理页面接口

七个推理页面使用外部 Host 插件的独立 `/tyloinfer` Connection RPC。原生 `/api` 仍由 DSH Gateway 持有。

## 页面与数据

| 页面 | 受限读取 | 明确操作与验收边界 |
|---|---|---|
| 概览 | `health/status/loaded_models/instances/resources/serving` | 显示核心连接、模型、请求统计与资源；未加载模型时提供模型页入口，不自动加载。 |
| 模型 | `models/loaded_models/instances`，`models/analysis` | 已登记工件与运行实例分开；既有加载、卸载、票据与取消见[模型桥](agent-model-bridge.md)。 |
| 服务 | `listener/memory_policy/inference_policy/context_policy/model_aliases/authentication_policy` | 显示实际监听与策略；`service/configure` 只允许下述四类策略。连接和凭据复用官方 provider 设置，凭据不由 Client 读取。 |
| 资源 | `resources/status/metrics/prefix_cache/memory_policy` | 不把未上报的 null 当零；缓存读取失败需保留错误，不冒充空缓存。清除、裁剪与磁盘删除不属于读取。 |
| 下载 | `download_directory/job_kinds/jobs` | 队列只把 `download.huggingface` 和 `download.modelscope` 当模型下载；无对应执行器时明确说明，不触发网络探测或自动下载。 |
| 分析 | `models`，`models/analysis` | 使用核心工件 ID 读取模型元数据，无需加载权重；无工件时引导登记模型，不能显示虚构分析。 |
| 日志 | `logs/metrics/jobs` | 保留原生时间、级别、序列和分页；空记录与断连不同。作业列表读取不代表开放任意作业创建、重试、删除或取消。 |

## 调用

```javascript
const result = await ctx.connection.rpc.call('/tyloinfer', 'core/read', {
  resource: 'logs',
  params: { limit: 100, level: 'warning', after: 0 }
}, signal);

const analysis = await ctx.connection.rpc.call('/tyloinfer', 'models/analysis', {
  model_id: selectedArtifact.id
}, signal);
```

`params` 可省略。只允许表中与该 resource 对应的命名字段；不接受 URL、任意路径、HTTP 方法、原始 query 或其他字段。Host 固定构造 GET 路由和查询编码。

| resource | 固定核心 GET | 可选 params |
|---|---|---|
| `health` | `/health` | 无 |
| `status` | `/api/v1/runtime/status` | `instance_id`：UUID |
| `instances` | `/api/v1/runtime/instances` | 无 |
| `models` | `/api/v1/models` | `refresh/include_components`：boolean |
| `loaded_models` | `/api/v1/runtime/models` | 无 |
| `resources` | `/api/v1/runtime/resources` | 无 |
| `capabilities` | `/api/v1/runtime/capabilities` | `instance_id`：UUID |
| `listener` | `/api/v1/runtime/listener` | 无 |
| `memory_policy` | `/api/v1/runtime/memory-policy` | 无 |
| `inference_policy` | `/api/v1/runtime/inference-policy` | 无 |
| `context_policy` | `/api/v1/runtime/context-policy` | 无 |
| `model_aliases` | `/api/v1/runtime/model-aliases` | 无 |
| `authentication_policy` | `/api/v1/auth/policy` | 无；只含认证策略状态，不读取密钥 |
| `serving` | `/api/v1/runtime/serving` | `scope`：session/alltime；`model`：最多255字符 |
| `metrics` | `/api/v1/runtime/metrics` | `instance_id`：UUID；`since`：带时区的ISO时间；`limit`：1–2000 |
| `logs` | `/api/v1/runtime/logs` | `instance_id`：UUID；`level`：debug/info/warning/error；`after`：非负安全整数；`limit`：1–2000 |
| `jobs` | `/api/v1/jobs` | `status`：原生作业状态；`kind`：原生作业类型标识；`limit`：1–200；`offset`：非负安全整数 |
| `job_kinds` | `/api/v1/jobs/kinds` | 无 |
| `download_directory` | `/api/v1/hub/download-directory` | 无 |
| `model_directories` | `/api/v1/models/directories` | `directory_id`：32位小写hex；不接受任意本地path |
| `prefix_cache` | `/api/v1/runtime/cache/entries` | `namespace`：64位小写hex；`offset`：非负安全整数；`limit`：1–200；`query`：最多200字符；`search_in`：all/text/id；`text_filter`：all/saved/missing；`chain_filter`：all/complete/incomplete；`sort`：recent/oldest/length/size |

`models/analysis` 只接受 `{model_id}`，对应固定 `/api/v1/models/{model_id}/analysis`；ID 必须是核心工件协议的32位小写hex，不接受模型名、实例UUID或文件路径。核心404、422、501等错误通过 `tyloinfer/core-rejected` 的 `details.http_status/core_error` 保留，不变成成功空数据。

### 模型组件补充只读接口

| endpoint | payload | 固定核心GET与原生成功值 |
|---|---|---|
| `models/cache-profile` | `{model_id,include_recurrent?}` | `/api/v1/models/{model_id}/cache-profile`；仅显式携带boolean时编码 `include_recurrent=true/false`，省略沿用核心false默认值。返回 `ModelCacheProfile` 或null，保留 `max_context/fixed_bytes/fixed_components/components` 及完整原生字段。 |
| `models/directory` | `{model_id}` | `/api/v1/models/{model_id}/directory`；返回完整 `ModelDirectoryList`，包括 `current_id/current_name/current_path/parent_id/model_file_count/files/can_open_directory/can_open_in_finder/data`。 |

两者仍仅owner可读，model_id必填且为32位小写hex工件ID；拒绝未知字段、模型名、实例UUID、任意path／URL／query／HTTP方法。目录由核心根据已登记工件确定，HF使用checkpoint目录，其余工件使用文件所在目录；不会启动系统文件管理器、打开文件、登记目录、删除文件或加载模型。返回的目录ID可用于既有受限目录浏览，不把 `current_path` 反向当成允许提交任意路径的授权。

成功封装保持 `{ok:true,value:<原生JSON>}`；cache-profile的null表示当前元数据没有可用预算描述，不能显示为零容量或虚构默认预算。核心404（工件不存在）、422（缓存元数据不可读）、501（目录／catalog能力不可用）仍通过 `tyloinfer/core-rejected` 的 `details.http_status/core_error` 保留。两个读取复用普通请求期限、取消、响应限额和凭据保护，不触发外部Hub、下载或推理。

## 官方清单本地快照

`hub/official-local`精确payload为 `{}`，只允许owner；对应固定 `GET /api/v1/hub/official/local`，不带query或body。成功为 `{ok:true,value:{system,data,refreshing}}`，复用原生 `OfficialModelList`：system是当前核心的 `HubSystemProfile`，data是完整 `OfficialModelInfo[]`，refreshing为boolean。

该读取只投影核心当前已知的官方spec与已载入内存的缓存元数据，不调用发现、官方刷新、详情补取或缓存重载，不更新官方缓存内容、时间戳或文件，不创建／取消后台任务，不加载或下载模型。当前核心没有硬编码的四模型清单，官方spec来自已发现仓库与启动时已载缓存；首次无缓存时data可以为空，不能拿本地模型工件列表冒充，也不伪造卡片。

refreshing仅反映读取时已有的刷新任务，true不表示该读取启动了任务，也不授权Client另发联网请求。可用于首次进入／重进页读取本地快照；后续刷新需适度，不能覆盖用户选择或表单，也不能因自动选择首项再隐式调用 `hub/info/search/readme/probe`。已有用户显式启动的后台刷新不被此读取停止，其后续缓存变化不属于快照请求产生的写入。

旧核心缺少固定路由时返回 `tyloinfer/operation-not-supported`、`retryable:false`，绝不回退到旧 `hub/official` 或用未识别的 `local_only` query冒充保证；核心配置的目录实现不支持快照时，保留原生501及 `local_official_catalog_unavailable`。请求取消、期限、禁止重定向、响应限额和凭据保护沿用既有读取契约。现有用户显式 `hub/official {refresh?}`语义保持不变。

## 用户显式动作

以下调用必须来自明确用户操作；挂载、刷新、重连和草稿恢复不得触发。Hub 读取可能访问外部模型站点，页面首次进入只读本地目录、执行能力与队列，用户点击检索／查看／刷新官方清单后才联网。

| endpoint | payload | 固定核心调用与返回 |
|---|---|---|
| `hub/search` | `{provider,query,formats?,sort?,direction?,cursor?,limit?}` | GET `/api/v1/hub/models`；固定 `browse=true`。provider 仅 huggingface/modelscope；formats 为 tq/gguf/mlx 数组，空数组省略；sort 为 relevance/downloads/likes/updated；direction 为 asc/desc；limit 1–100；query 最多255字符；cursor 为原生16–64字符标识。返回原生搜索页。 |
| `hub/info` | `{provider,repo_id,revision?}` | GET `/api/v1/hub/models/{provider}/{owner}/{name}`；repo_id 仅 owner/name，不接受URL、相对路径或其它站点。返回原生元数据。 |
| `hub/official` | `{refresh?}` | GET `/api/v1/hub/official`；refresh仅boolean。省略或false不等于local-only：首次／过期会启动后台联网刷新，未过期也可能补取缺失详情。当前没有保证只读本地官方清单的参数，不得用作挂载／轮询自动读取。返回原生清单，包括 `refreshing`；不能将仍刷新中的空清单当成最终不存在。 |
| `models/register` | `{directory_id}` | POST `/api/v1/models/directories/register`；只用核心已提供的32位hex目录ID，不接收本地path。返回原生 `ModelArtifactList`；登记不加载模型。 |
| `downloads/create` | `{provider,request:{repo_id,destination,revision?,include?,expected_bytes?,max_workers?,download_directory?}}` | 先读核心 `job_kinds` 确认执行器；仅POST `/api/v1/jobs`，kind固定为 `download.huggingface` 或 `download.modelscope`，payload为原生request。核心202返回原生 `JobResource`，`id` 为真实票据，不伪造成 `OperationAccepted`。 |
| `downloads/get` / `downloads/cancel` | `{operation_id}` | GET `/api/v1/jobs/{id}`；取消先读再按原生状态POST `/api/v1/jobs/{id}/cancel`，只接受上述两类下载作业。 |
| `service/configure` | `{resource,request}` | 固定PUT对应策略路由，白名单见下表。同步策略仅原生200和有效策略字段算保存成功；memory_policy仅原生202 accepted算受理。 |
| `service/get` / `service/cancel` | `{operation_id}` | 固定作业读取／取消，仅 `runtime.memory.configure`；不开放模型重载、测速或任意作业。 |

下载 request 保留调用方字段，不补写下载并发、revision或文件选择。destination 必须为相对路径，禁止上级跳转；include 为原生文件选择数组，expected_bytes 可为null或非负安全整数，max_workers为原生1–16。可选 download_directory 仍由核心下载目录管理器核实；不接受API密钥、任意job kind、dataset/space类型或自定义下载URL。未安装下载执行器时在发送POST前失败，不能为显示成功而创建空作业。

| service/configure resource | request 白名单 | 完成语义 |
|---|---|---|
| `inference_policy` | 必填 `mtp_enabled` boolean | 返回原生同步策略。 |
| `model_aliases` | 必填 `aliases`，模型名到别名的字符串对象，最多128项 | 返回原生同步映射；模型存在性与冲突由核心验证。 |
| `context_policy` | 必填 `max_context_size`，null或512–2147483647整数 | 只保存策略，不隐式重载现有模型，不更写Auto／固定精度。 |
| `memory_policy` | 至少一个 `total_limit_bytes/model_limit_bytes/prefix_limit_bytes/prefix_disk_limit_bytes/prefix_ram_limit_bytes`；null重置，total/model为正整数，prefix可为0 | 返回真实 `OperationAccepted`，用 `service/get` 跟踪完成；不把accepted当已应用。不接受prefix_directory或任何路径字段。 |

写类请求上限64 KiB，不重试；POST或PUT发出后断链、取消、5xx或无效成功响应均保守返回 `tyloinfer/outcome-unknown`。取消作业不等于取消HTTP传输；已取消请求、取消中与最终cancelled按原生状态区分。模型作业、下载作业和内存策略作业的跟踪／取消入口互相隔离，不因读取日志的全部作业清单而扩大写权限。

## 维护接口

## 产品边界接口

缓存、下载目录和作业事件见[维护接口](agent-maintenance-bridge.md)。模型加载、工件删除和下载重试见[模型与下载补充动作](agent-model-actions.md)。连接和工件选择见[产品边界](agent-product-boundaries.md)。
