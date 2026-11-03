# TyloInfer 原生 Server HTTP API

`tyloinfer-server`（`cpp_runtime/server`）为 Studio、脚本和 SDK 提供常驻 HTTP 接口。
本文就是接口契约本身：字段、约束、响应形状和生命周期语义都在这里规定，
服务端不再发布机器可读 schema，也没有 `/openapi.json` 或 Swagger 入口。

默认地址：

```text
http://127.0.0.1:8090
```

版本化管理根是 `http://127.0.0.1:8090/api/v1`。`/health`、`/health/identity`
和 `/v1` 兼容入口不在版本化根下。启动参数见 `tyloinfer-server --help`：
`--host`、`--port`（默认 8090，`0` 表示随机端口）、`--data-dir`、`--db`、
`--web-root`、`--api-key-env`、`--model-dir`、`--exit-on-stdin-close`、
`--studio-instance-id`。监听地址不是回环时必须启用鉴权，否则启动失败。

## 1. 通用约定

### 1.1 鉴权与 scope

未配置根凭据时，`/api/` 与 `/v1` 路由不要求凭据。配置 `TYLOINFER_SERVER_API_KEY`
（或 `--api-key-env` 指定的变量）后，所有 `/api/` 和 `/v1` 请求必须携带：

```http
Authorization: Bearer <token>
```

Anthropic 风格请求（`/v1/messages*`、`/v1/models/*`，或带 `Anthropic-Version`、
`Anthropic-Beta`、`X-Api-Key` 头的 `/v1`、`/v1/models`）也可以把凭据放在
`X-Api-Key`。根凭据拥有 `admin`。子密钥由 `POST /api/v1/auth/keys` 创建，
可指定 scope 列表和/或角色；`admin` 覆盖一切。

| 角色 | 授予的 scope |
|---|---|
| `viewer` | `inference` |
| `operator` | `inference`、`models`、`jobs` |
| `administrator` | `admin` |

| 路由区域 | 所需 scope |
|---|---|
| `/api/v1/auth/*` | `admin` |
| `PUT /api/v1/runtime/model-aliases` | `admin` |
| `/api/v1/models/directories*`、`/api/v1/models/{artifact_id}/directory`、`/api/v1/hub/download-directory` | `models` |
| 其余 `/api/v1/models`、`/api/v1/runtime`、`/api/v1/hub`：读 | `inference` |
| 其余 `/api/v1/models`、`/api/v1/runtime`、`/api/v1/hub`：写（含 `POST /models/load`、`/models/unload`） | `models` |
| `/api/v1/jobs*` | `jobs` |
| `/api/v1/sessions*`、`/api/v1/presets*`、`/api/v1/inference/*` | `inference` |
| `/v1*` 兼容入口 | `inference` |

`/health`、`/health/identity` 不在 `/api/` 下，永不要求凭据。

请求可带 `X-TyloInfer-Instance-ID`；若该值与当前进程身份不符，服务返回
`409 server_instance_changed`，客户端必须重连并确认新实例（重启后的服务
不会接受旧实例的请求）。

### 1.2 类型与严格校验

| 记号 | 含义 |
|---|---|
| `UUID` | 标准 UUID 字符串；所有 `*_id` 路径参数都是 UUID |
| `DateTime` | ISO 8601 时间字符串 |
| `Object` | 自由 JSON 对象，字段可由服务端扩展 |
| `String[]` | 字符串数组 |

管理接口（`/api/v1/*`）的 JSON 是严格的：拒绝未知字段、错误的字段类型、
超界整数和未声明允许的 `null`，一律返回 `400 invalid_request`。
`Content-Type` 必须是 `application/json`，否则返回 `415 unsupported_media_type`。
请求体上限为 1 MiB，超出由 HTTP 层直接拒绝。

`/v1` 兼容层按各自协议校验：OpenAI chat 请求不拒绝未知字段，Anthropic
Messages 请求拒绝未知字段。

### 1.3 管理错误信封

`/api/v1/*` 的错误统一为：

```json
{
  "error": {
    "code": "invalid_request",
    "message": "unknown parameter: foo",
    "retryable": false,
    "details": {}
  }
}
```

`retryable` 只在 `429` 和 `503` 为 `true`，仅表示可能重试，不承诺重试安全；
丢失受理响应时先核对任务与实例状态，不要盲目重放。`401` 附带
`WWW-Authenticate: Bearer`；`429` 与 `503` 附带 `Retry-After: 1`。
`/v1` 兼容入口改用各自协议的错误信封，见 6.5。

常见稳定错误码：

| 状态 | `code` | 触发条件 |
|---|---|---|
| `400` | `invalid_request` | 未知字段、类型错误、超界整数、非法筛选值 |
| `400` | `invalid_model_directory_source`、`invalid_provider`、`invalid_model_alias`、`invalid_download_*`、`invalid_hub_*` | 领域参数非法 |
| `400` | `job_kind_not_supported`、`repo_type_not_supported`、`unsupported_requirement` | 请求了未支持的类型 |
| `400` | `search_cursor_expired` | Hub 搜索游标过期或筛选条件变化 |
| `401` / `403` | `authentication_error`、`authentication_required` | 凭据缺失、无效、撤销或过期 |
| `404` | `not_found`、`operation_not_found`、`model_artifact_not_found`、`model_directory_not_found`、`session_not_found`、`response_not_found` | 对象不存在或已归档 |
| `409` | `job_state_conflict`、`model_already_loaded`、`model_operation_busy`、`model_busy`、`model_alias_conflict`、`revision_conflict`、`response_in_progress`、`request_cancelled`、`model_delete_busy`、`model_delete_unsafe`、`model_delete_changed`、`model_delete_failed` | 状态或标识冲突 |
| `410` | `event_cursor_expired` | 事件游标已被淘汰 |
| `413` | `request_too_large` | 会话消息或导出超过上限 |
| `415` | `unsupported_media_type` | `Content-Type` 不是 `application/json` |
| `422` | `checkpoint_analysis_unavailable`、`model_cache_profile_unavailable` | 检查点不完整或元数据读不出 |
| `429` | `operation_queue_full`、`stream_connections_full` | 后台队列或流式连接已满 |
| `500` | `response_save_failed`、`model_registration_save_failed`、`model_alias_save_failed`、`operation_failed` | 落盘或执行失败 |
| `501` | `cache_reclaim_not_supported`、`media_unavailable`、`session_mode_unavailable` | 当前服务不提供该能力 |
| `502` | `invalid_tool_output`、`model_card_unavailable`、下载源或校验失败 | 上游/原生输出不可信 |
| `503` | `server_stopping`、`model_stopping`、`backend_unavailable`、`resource_exhausted` | 停机中或资源不可用 |

### 1.4 列表、删除与异步受理

列表响应统一为 `{"data": [...]}`。成功删除返回 `204 No Content`。

模型加载/卸载返回 `202 Accepted`，并带 `Location: /api/v1/jobs/{id}`：

```json
{"operation_id": "uuid", "status": "accepted"}
```

`operation_id` 就是 jobs 记录的 `id`，不存在独立的操作表。受理只表示任务已
可靠落库并排队，不表示执行成功；受理后的失败记录在任务的 `error` 字段，
不再改写原 HTTP 响应。

后台执行队列为单线程，最多排队 16 个任务，超出返回 `429 operation_queue_full`。

### 1.5 并发与停机

HTTP 层共 4 个工作线程。流式请求（`/v1/chat/completions`、`/v1/messages`、
会话回复流、任务事件流）合计最多占用 3 个，保留 1 个处理管理与取消请求；
超出返回 `429 stream_connections_full`。断开流式连接只关闭连接：任务订阅
不会取消任务，推理流会请求取消原生生成。

停止信号到达后进入 draining：新的非读、非 `/cancel` 请求返回
`503 server_stopping`；查询与取消接口保留到 HTTP 关闭。服务先通知任务、
会话和实例停止，再等待同步加载返回、回复落库和实例回收。重启时未完成的
任务记录标记为 `interrupted`，不会自动重跑。

## 2. 端点总览

破折号表示没有 JSON 请求体。字段表和语义要点见后续小节。

### 2.1 鉴权管理（`admin`）

| 方法 | 路径 | 请求 | 成功 |
|---|---|---|---|
| `GET` | `/api/v1/auth/policy` | — | `200 Object` |
| `PUT` | `/api/v1/auth/policy` | `{"enabled":bool,"api_key":string}` | `200 Object` |
| `GET` | `/api/v1/auth/keys` | — | `200 {"data":[ApiKey]}` |
| `POST` | `/api/v1/auth/keys` | `{"name":string,"scopes":String[],"role":string\|null,"expires_at":DateTime\|null}` | `201 {"key":ApiKey,"token":string}` |
| `POST` | `/api/v1/auth/keys/{key_id}/revoke` | — | `200 ApiKey` |
| `POST` | `/api/v1/auth/keys/{key_id}/rotate` | — | `200 {"key":ApiKey,"token":string}` |

`name` 为 1-64 个字符；`scopes` 最多 4 项且只能取 `inference`、`models`、
`jobs`、`admin`；没有 scope 时必须给角色。`api_key` 为 1-512 个不含空格的
可打印 ASCII 字符。

非回环监听时禁用鉴权会被拒绝。策略变更持久化到重启。

### 2.2 模型生命周期与运行时

| 方法 | 路径 | 请求 | 成功 | scope |
|---|---|---|---|---|
| `GET` | `/api/v1/models` | `?refresh=true\|false` | `200 {"data":[Artifact]}` | `inference` |
| `GET` | `/api/v1/models/directories` | `?directory_id=` 或 `?path=` | `200 Directory` | `models` |
| `POST` | `/api/v1/models/directories/register` | `{"path":string}` 或 `{"directory_id":string}` | `200 {"data":[Artifact]}` | `models` |
| `GET` | `/api/v1/models/{artifact_id}/directory` | — | `200 Directory` | `models` |
| `GET` | `/api/v1/models/{artifact_id}/capabilities` | — | `200 Capabilities` | `inference` |
| `GET` | `/api/v1/models/{model_id}/analysis` | — | `200 CheckpointAnalysis` | `inference` |
| `GET` | `/api/v1/models/{model_id}/cache-profile` | `?include_recurrent=true\|false` | `200 ModelCacheProfile\|null` | `inference` |
| `GET` | `/api/v1/models/{model_id}/deletion` | — | `200 DeletionPlan` | `inference` |
| `POST` | `/api/v1/models/{model_id}/delete` | `{"token":string}` | `200 {"files":int,"total_bytes":int}` | `models` |
| `POST` | `/api/v1/models/load` | `LoadRequest` | `202 Accepted` | `models` |
| `POST` | `/api/v1/models/unload` | `UnloadRequest` | `202 Accepted` | `models` |
| `GET` | `/api/v1/runtime/instances` | — | `200 {"data":[Instance]}` | `inference` |
| `GET` | `/api/v1/runtime/instances/{instance_id}` | — | `200 Instance` | `inference` |
| `GET` | `/api/v1/runtime/status` | `?instance_id=UUID` | `200 Status` | `inference` |
| `GET` | `/api/v1/runtime/models` | — | `200 {"data":[{"id":String}]}` | `inference` |
| `GET` | `/api/v1/runtime/resources` | — | `200 Resources` | `inference` |
| `GET` | `/api/v1/runtime/serving` | `?scope=session\|alltime&model=` | `200 ServingStatistics` | `inference` |
| `GET` | `/api/v1/runtime/metrics` | `?instance_id&since&limit` | `200 {"data":[MetricSnapshot]}` | `inference` |
| `GET` | `/api/v1/runtime/logs` | `?instance_id&level&after&limit` | `200 {"data":[LogEntry]}` | `inference` |
| `GET` | `/api/v1/runtime/model-aliases` | — | `200 Aliases` | `inference` |
| `PUT` | `/api/v1/runtime/model-aliases` | `Aliases` | `200 Aliases` | `admin` |

### 2.3 任务

| 方法 | 路径 | 请求 | 成功 |
|---|---|---|---|
| `GET` | `/api/v1/jobs` | `?limit&offset&status&kind` | `200 {"data":[Job]}` |
| `POST` | `/api/v1/jobs` | `{"kind":string,"payload":Download}` | `202 Job` |
| `GET` | `/api/v1/jobs/kinds` | — | `200 {"data":[{"kind":String}]}` |
| `GET` | `/api/v1/jobs/{id}` | — | `200 Job` |
| `DELETE` | `/api/v1/jobs/{id}` | — | `204` |
| `POST` | `/api/v1/jobs/{id}/cancel` | — | `200 Job` |
| `POST` | `/api/v1/jobs/{id}/retry` | — | `202 Job` |
| `GET` | `/api/v1/jobs/{id}/events` | `?after&limit&tail` | `200 {"data":[JobEvent]}` |
| `GET` | `/api/v1/jobs/{id}/events/stream` | `?after` + `Last-Event-ID` | `200` SSE |
| `DELETE` | `/api/v1/jobs/completed` | — | `204` |

### 2.4 会话与回复（`inference`）

| 方法 | 路径 | 请求 | 成功 |
|---|---|---|---|
| `GET` | `/api/v1/sessions` | `?limit=1..200（默认 50）&offset` | `200 {"data":[Session]}` |
| `POST` | `/api/v1/sessions` | `CreateSession` | `201 Session` |
| `POST` | `/api/v1/sessions/import` | `SessionArchive` | `201 {"session","messages_imported","media_imported"}` |
| `GET` | `/api/v1/sessions/{session_id}` | — | `200 Session` |
| `PATCH` | `/api/v1/sessions/{session_id}` | `{"title","mode","metadata"}`（至少一项） | `200 Session` |
| `DELETE` | `/api/v1/sessions/{session_id}` | — | `204` |
| `GET` | `/api/v1/sessions/{session_id}/messages` | — | `200 {"data":[SessionMessage]}` |
| `POST` | `/api/v1/sessions/{session_id}/messages` | `{"expected_revision","role","parts"}` | `201 {"session","message"}` |
| `POST` | `/api/v1/sessions/{session_id}/fork` | `{"at_message_id","include_message","title","model"}` | `201 Session` |
| `POST` | `/api/v1/sessions/{session_id}/rewind` | `{"expected_revision","at_message_id","include_message"}` | `200 Session` |
| `GET` | `/api/v1/sessions/{session_id}/export` | — | `200 SessionArchive` |
| `GET` | `/api/v1/sessions/{session_id}/responses` | `?limit=1..1000（默认 200）` | `200 {"data":[SessionResponse]}` |
| `POST` | `/api/v1/sessions/{session_id}/responses` | `CreateResponse` | `200 SessionResponse` 或 SSE |
| `POST` | `/api/v1/sessions/{session_id}/responses/cancel` | — | `200 SessionResponse` |

### 2.5 生成预设（`inference`）

| 方法 | 路径 | 请求 | 成功 |
|---|---|---|---|
| `GET` | `/api/v1/presets` | — | `200 {"data":[GenerationPreset]}` |
| `POST` | `/api/v1/presets` | `PutGenerationPreset` | `201 GenerationPreset` |
| `PUT` | `/api/v1/presets/{preset_id}` | `PutGenerationPreset` | `200 GenerationPreset` |
| `DELETE` | `/api/v1/presets/{preset_id}` | — | `204` |

`PUT` 整体替换预设，不是部分更新。预设创建不要求模型已加载。

### 2.6 Hub 与下载目录

| 方法 | 路径 | 请求 | 成功 | scope |
|---|---|---|---|---|
| `POST` | `/api/v1/hub/resolve` | `{"reference":string,"fallback_provider":"huggingface"\|"modelscope"}` | `200 HubRepo` | `models` |
| `GET` | `/api/v1/hub/models` | `?provider&query&limit&format&sort&formats&direction&cursor&browse` | `200 HubSearchPage` | `inference` |
| `GET` | `/api/v1/hub/official` | `?refresh=bool` | `200 OfficialCatalog` | `inference` |
| `GET` | `/api/v1/hub/models/{provider}/{owner}/{name}` | `?revision=` | `200 HubRepo` | `inference` |
| `GET` | `/api/v1/hub/models/{provider}/{owner}/{name}/readme` | `?revision=` | `200 {"content":string\|null}` | `inference` |
| `POST` | `/api/v1/hub/download-reuse` | `{"provider":string,"repo_id":string,"revision"?:string,"destination":string,"download_directory"?:string}` | `200 {"reusable_files":{文件名:字节}}` | `models` |
| `POST` | `/api/v1/hub/download-sources/probe` | `{"sources":[{"provider":string,"repo_id":string,"revision"?:string}]}` | `200 {"data":[ProbeRoute]}` | `models` |
| `GET` | `/api/v1/hub/download-directory` | — | `200 DownloadDirectory` | `models` |
| `PUT` | `/api/v1/hub/download-directory` | `{"directory_id":string\|null,"evict_prefix_cache":bool}` | `200 DownloadDirectory` | `models` |

`resolve` 接受 `owner/name` 或 `huggingface.co`/`modelscope.cn` 的 HTTPS 仓库
URL；任意 URL、凭据、查询串和 URL 后缀 revision 一律拒绝。仓库详情列出文件并
分组可下载权重；估算与热度元数据为 `null`。Hugging Face 固定 commit，
ModelScope 需要每个文件的 SHA-256。

Hub 搜索：`provider` 必填，`query`（≤255）、`limit`（`1..100`，默认 `20`）、
`format`（`all\|tq\|gguf\|mlx`）、`sort`（`relevance\|downloads\|likes\|updated`，
默认 `downloads`）、`direction`（`asc\|desc`）、`formats`（逗号分隔的 `tq`/`gguf`/`mlx`）、
`cursor` 和 `browse`。带 `browse`、`formats`、`cursor`、`sort=relevance` 或
`direction=asc` 任一项即进入浏览器路径：按格式分别翻页、合并去重，
`HubSearchPage` 为 `{"data":[HubSummary],"next_cursor":string\|null}`，
`next_cursor` 在服务端保存 15 分钟；游标过期或筛选条件变化返回
`400 search_cursor_expired`。ModelScope 不支持 `direction=asc`，返回
`400 invalid_hub_search`。单页搜索不返回游标（`next_cursor` 恒 `null`）。
`HubSummary` 字段：`provider`、`repo_id`、`source_url`、`author`、`description`、
`downloads`、`likes`、`total_bytes`、`updated_at`、`formats`、`trending_score`；
Hugging Face 不提供 `total_bytes`，恒为 `0`。

官方目录：`refresh=true` 强制刷新。响应为 `{"system":SystemProfile,"data":[OfficialModel],
"refreshing":bool}`，刷新在后台线程进行，`refreshing` 为 `true` 时客户端应稍后重试。
`system` 报告 `platform`、`machine`、`backend`、`cpu_cores`、物理/可用内存和单个
`ram` 内存池；`cpu_name`、GPU 字段和 `runtime_memory_budget_bytes` 为 `null`。
官方条目按 `updated_at` 倒序排列，字段集与 Python 侧一致，但驻留权重、参数分解、
cache profile、attention profile、license 和 `published_at` 一律为 `null`，
`configuration` 恒为 `unknown`（`available_memory_bytes` 取物理内存）。

模型卡：`readme` 返回 `{"content":string|null}`，仓库没有模型卡时为 `null`；
上游不可达返回 `502 model_card_unavailable`。

下载复用：`destination` 相对下载目录（`models/` 前缀会被剥离），未指定
`download_directory` 时使用当前下载目录。响应 `reusable_files` 只列出磁盘上已存在且
字节数一致的 `.tq`/`.mfq` 文件，键为仓库内文件名，值为字节数。

下载源探测：`sources` 接受 1 到 2 个条目。`ProbeRoute` 为
`{provider,repo_id,bytes_per_second,route}`，`route` 取 `direct` 或 `proxy`；
仓库没有 ≥256 KiB 的权重文件时省略 `route` 且 `bytes_per_second` 为 `0`，
仓库元数据不可用时同样返回 `0` 而不是报错。

下载目录：`directory_id` 来自模型目录浏览器，`null` 重置为工作区模型目录；
`evict_prefix_cache: true` 返回 `501 cache_reclaim_not_supported`。响应字段：
`directory`、`actual_directory`、`default_directory`、`disk_total_bytes`、
`disk_free_bytes`、`disk_available_bytes`、`evict_prefix_cache`（恒 `false`）。

### 2.7 推理取消（`inference`）

| 方法 | 路径 | 请求 | 成功 |
|---|---|---|---|
| `POST` | `/api/v1/inference/requests/{request_id}/cancel` | `{"instance_id":UUID}` | `200 {"cancel_requested":bool}` |

`request_id` 属于具体实例，取消使用实例与请求标识的组合。`cancel_requested`
只表示已受理取消意图，不表示生成已终止或资源已清理；终态从原推理流读取。
同步原生加载只能在工厂返回后收尾。

### 2.8 健康与身份

`GET /health` 返回 `{"status":"ok","service":"tyloinfer-server","protocol_version":"1.0"}`。
`GET /health/identity` 返回进程身份：`service`、`protocol_version`、
`studio_protocol_version`、`version`、`instance_id`、`workspace_id`、`pid`、
`launch_mode`。

## 3. 模型生命周期

### 3.1 `POST /api/v1/models/load`

```json
{
  "artifact_id": "catalog-artifact-id",
  "requirements": {"context_size": 8192},
  "policy": {"pin": false, "idle_ttl_seconds": 300}
}
```

| 字段 | 类型 | 约束 |
|---|---|---|
| `artifact_id` | String | 必填；必须是目录中已注册产物的 ID，不接受模型名或别名 |
| `requirements.context_size` | Integer | 可省略；`1..2147483647` |
| `policy.pin` | Boolean | 默认 `false`；表示服务保留实例，不表示锁定物理内存 |
| `policy.idle_ttl_seconds` | Integer | `1..2147483647`；固定实例（`pin: true`）不能同时指定空闲超时 |

`requirements` 未填写时把原生默认上下文交给现有工厂决定，服务端不补 8192
或 32768。显式要求无法满足时加载失败，加载结果报告实际配置。空闲卸载不取消
正在运行的请求，每次使用后重新计时；显式卸载不受 `pin` 限制。服务策略不会
传给原生工厂。加载请求不接受设备放置、预填充分块、批处理、KV 或后端缓存预算，
也不提供任意后端参数字典。

加载任务只报告 `resolving`、`loading`、`initializing` 三个真实检查点；
进行中的 `progress` 为 0 表示没有可量化进度，成功后为 1。

### 3.2 `POST /api/v1/models/unload`

| 字段 | 类型 | 约束 |
|---|---|---|
| `instance_id` | UUID | 必填 |
| `force` | Boolean | 默认 `false`；普通卸载遇到活动请求失败，强制卸载取消并等待必要清理 |

成功前不移除实例，关闭失败时保留故障状态。卸载成功结果包含
`{"instance_id":..., "state":"unloaded"}`；加载成功结果是当时的实例表示。

### 3.3 能力与实例

`GET /api/v1/models/{artifact_id}/capabilities`：

| 字段 | 含义 |
|---|---|
| `artifact_id` | 产物 ID |
| `supported` | 产物能否被所接工厂识别执行，不保证当前资源足够 |
| `reason` | 不支持的原因，支持时为 `null` |
| `declared_context_capacity` | 目录声明的上下文容量 |
| `load.cancellation` | 恒 `after_return` |
| `load.progress` | 恒 `phase` |

`GET /api/v1/runtime/instances` 返回 `{"data":[Instance]}`。`Instance` 含
`instance_id`、`artifact_id`、`model`、`backend`、`state`
（`loaded`\|`unloading`\|`failed`）、`context_size`、`max_requests`、
`active_requests`、`input_bytes`、`capabilities`、`metrics`、`session_metrics`、
`policy{pin,idle_ttl_seconds}`、`metrics_age_ms`。`metrics_age_ms` 是生成该
实例表示时原生指标快照距采样完成的毫秒数；请求计数另行读取，不用该年龄推断
请求状态。实例存在与任务历史成功是不同事实。

### 3.4 模型别名

`GET/PUT /api/v1/runtime/model-aliases` 读写原模型名到公开别名的映射，正文为
`{"aliases":{"model-name":"public-name"}}`。读取需要 `inference`，写入需要
`admin`。别名去除 ASCII 首尾空白后最长 255 个 Unicode 字符，不得含 ASCII 控制
字符。空值或等于原名的值清除映射；别名必须唯一，不得遮蔽原名、产物 ID 或
实例 ID。目标必须是已登记或已加载的模型。校验或持久化失败时保留原有映射与
路由；并发模型操作返回 `409`。卸载不删除别名配置。

公开列表同时公布原生支持的登记产物与健康实例；同名产物/实例用 ID 区分。
列表、详情和设置别名都不触发加载。普通响应与 SSE 回显请求中的名称。

### 3.5 模型分析、缓存画像与磁盘删除

`GET /api/v1/models/{model_id}/analysis` 读取产物元数据，返回 `CheckpointAnalysis`：

| 字段 | 含义 |
|---|---|
| `model_id`、`name`、`format`、`architecture`、`complete` | 产物身份与格式 |
| `layer_count` | `num_hidden_layers`，或与张量名推出的最大层号 + 1 的较大值 |
| `hidden_size`、`attention_heads`、`kv_heads`、`head_dim`、`linear_key_heads`、`linear_value_heads`、`experts_per_token` | 配置里的结构数字，缺失为 `null` |
| `expert_count` | 配置声明与专家条目推出的最大值 |
| `parameters`、`stored_bytes`、`average_bpw` | 全部张量的参数与字节合计，平均位宽排除 `predictor`、`vision` |
| `attention_distribution` | 注意力类型到层号列表的映射 |
| `layers[]` | `layer`、`attention_type`、`parameters`、`stored_bytes`、`average_bpw` |
| `projections[]` | 按 `(category, name)` 排序的投影聚合，含 `tensor_count`、`formats` |
| `tensors[]` | `name`、`shape`、`category`、`projection`、`layer`、`parameters`、`stored_bytes`、`bpw`、`format` |
| `experts[]` | 路由专家行：`layer`、`expert`、`projection`、`parameters`、`stored_bytes`、`bpw`、`format`、`granularity` |
| `graph` | `hc_count`、`has_ple`、`has_shared_expert` |
| `advanced` | 结构细节：FFN 与专家宽度、HC、激活函数、GDN/QSA/PLE 卷积与压缩参数、`normalizations[]` 等 |
| `cache_profile` | 按 `include_recurrent=false` 计算的缓存画像，见下文 |
| `warnings[]` | 无法解析的张量说明 |

`{model_id}` 接受产物 ID 或模型名；不存在返回 `404 model_artifact_not_found`，
产物不完整或元数据读不出返回 `422 checkpoint_analysis_unavailable`。容器没有
保存逻辑形状时从张量头恢复几何，恢复不了的张量记入 `warnings` 并从
`parameters`、`stored_bytes` 中排除。

`GET /api/v1/models/{model_id}/cache-profile` 返回 `ModelCacheProfile`，架构没有
缓存模型时返回 `null`。字段为 `max_context`、`fixed_bytes`、
`fixed_components[]{group,name,layers,bytes}` 与
`components[]{group,name,subgroup,layers,bytes_per_row,tokens_per_row,allocation,minimum_rows,row_rounding,active_after,max_read_rows_per_token,head_dimension,kv_heads}`。
`include_recurrent` 只决定是否计入循环状态（GDN、PLE）的固定分量，默认 `false`。
读不出元数据返回 `422 model_cache_profile_unavailable`。

`GET /api/v1/models/{model_id}/deletion` 返回删除计划：

```json
{"model_id": "catalog-artifact-id", "name": "model-name",
 "files": ["/abs/path/config.json"], "total_bytes": 123, "token": "<64 hex>"}
```

`files` 就是会被删除的确切文件：分片模型按同基名同分片数匹配同目录兄弟分片，
HF 目录含 `config.json`、`*.safetensors` 与索引 `weight_map` 引用的分片。目录和
伴随文件不在计划内。`token` 绑定这些文件的设备号、inode、大小与时间戳。

`POST /api/v1/models/{model_id}/delete` 正文只有 `{"token":string}`，必须是计划
返回的 64 位十六进制摘要。

| 状态 | `code` | 条件 |
|---|---|---|
| `400` | `invalid_request` | `token` 缺失、类型错误或不是 64 位十六进制 |
| `404` | `model_artifact_not_found` | 产物不存在 |
| `409` | `model_delete_busy` | 该产物或同名模型已加载、正在卸载或正在加载 |
| `409` | `model_delete_unsafe` | 文件集合越出模型目录，例如 `weight_map` 指向 `..`、绝对路径或非 `.safetensors` 目标 |
| `409` | `model_delete_changed` | 计划与当前文件状态不一致，需要重新取计划 |
| `409` | `model_delete_failed` | 删除中途失败，已删除的文件不会恢复 |

删除只作用于计划里的文件，不删除模型目录或伴随文件；成功后服务重新扫描目录，
产物从列表消失。成功响应为 `{"files":N,"total_bytes":M}`。

### 3.6 运行时观测

原生服务没有共享的 runtime worker，也没有 Python 侧的指标与日志存储。观测资源
只报告原生能够取到的事实，取不到的字段保持 `null` 或空集合，不会用估算值填充。

`GET /api/v1/runtime/instances/{instance_id}` 返回与列表元素相同的 `Instance`；
未知实例返回 `404 runtime_instance_not_found`，UUID 格式错误返回 `400`。

`GET /api/v1/runtime/status` 不带 `instance_id` 时返回第一个已加载实例的投影，
没有已加载实例时为 `{"source":"native-instances","status":"ok","runtime_state":"idle"}`；
带 `instance_id` 时选择该实例，错误码与实例详情相同。投影字段为 `source`
（恒 `native-instances`）、`status`、`instance_id`、`model`、`backend`、
`model_type`、`runtime_state`（`ready`\|`busy`\|`unloading`\|`failed`）、
`max_context`、`context_capacity`、`vocab_size`、`max_requests`、`active_requests`、
`reloading`、`model_capabilities`、`metrics_age_ms`。Python 侧由 runtime worker
上报的 `started_at`、`uptime_seconds`、`total_requests`、`failed_requests`、
`total_prompt_tokens`、`total_completion_tokens`、`last_request`、`live_requests`
以及内存与前缀缓存计数在原生服务不存在，响应里不出现这些键。

`GET /api/v1/runtime/models` 只列出已加载实例的模型名，同名实例合并为一条；
不登记产物，也不触发加载。

`GET /api/v1/runtime/resources` 采样服务所在主机：

| 字段 | 原生来源 |
|---|---|
| `sampled_at` | 采样时刻（Unix 秒） |
| `interval_seconds` | 与上一次采样的间隔；首次采样或间隔超过 10 秒为 `null` |
| `cpu_cores` | `std::thread::hardware_concurrency()` |
| `cpu_utilization_percent` | 相邻两次采样的 CPU 时间差；首次采样为 `null` |
| `cpu_name` | 恒 `null`，原生服务不上报 CPU 型号 |
| `memory_total_bytes`、`memory_available_bytes`、`memory_used_bytes` | 系统物理与可用内存，`used = total - available` |
| `gpus`、`disks` | 恒为空数组，原生服务不采样 GPU 利用率与磁盘 IO |
| `memory_bandwidth_bytes_per_second`、`memory_bandwidth_limit_bytes_per_second`、`memory_bandwidth_utilization_percent` | 恒 `null` |
| `weights` | 每个已加载实例一条：`instance_id`、`model` 加五个字节速率字段 |

`weights` 的速率由实例原生指标快照差分得到：`expert_read_bytes_per_second`
取 `ssd_expert_bytes_read`，`ple_read_bytes_per_second` 取 `ple_source_bytes_read`，
`engram_read_bytes_per_second` 取 `engram_bytes_read`，`kv_read_bytes_per_second`
取 `qsa_kv_ssd_read_bytes`，`kv_write_bytes_per_second` 取
`qsa_kv_ssd_written_bytes`。计数器不存在、上一次快照缺少该计数器或采样间隔
超过 10 秒时该速率为 `null`。

`GET /api/v1/runtime/serving` 接受 `scope`（默认 `session`，或 `alltime`）与
`model` 过滤，其它取值返回 `400 invalid_request`。原生服务不持久化完成请求
记录，`totals` 恒为零计数与 `null` 速率，`activity` 恒为空数组，`started_at`
是服务进程启动时刻。

`GET /api/v1/runtime/metrics` 与 `GET /api/v1/runtime/logs` 保留 Python 的查询
参数：`instance_id` 必须是 UUID，`since` 必须是 ISO 8601 时间，`level` 取
`debug`\|`info`\|`warning`\|`error`，`after` 不小于 `0`，`limit` 为 `1..2000`
（默认 `200`），违反者返回 `400 invalid_request`。原生服务没有指标历史与日志
存储，两者恒返回 `{"data":[]}`；实时指标见 `GET /api/v1/runtime/instances` 与
`GET /api/v1/runtime/status`。

## 4. 任务与事件

### 4.1 任务记录与状态机

`Job` 字段：`id`、`kind`、`status`、`payload`（原始业务请求）、`progress`
（`0..1`）、`progress_data`、`cancel_requested`、`result`、`error`、
`last_sequence`、`created_at`、`updated_at`、`started_at`、`completed_at`。

状态为 `queued`、`running`、`cancelling`、`succeeded`、`failed`、`cancelled`、
`interrupted`。`queued` 可直接取消；`running` 收到取消后进入 `cancelling`；
取消受理与取消完成分开，完成必要清理后才发布 `cancelled`。成功发布实例后的
迟到取消不能改写成功事实。服务重启把未完成记录标记为 `interrupted`，不自动
重跑模型加载。

列表按创建时间与 ID 倒序；`limit` 为 `1..1000`（默认 100），`offset` 只跳过
匹配的可见行；`status` 与 `kind`（`^[a-z][a-z0-9_.-]{0,63}$`）可联合筛选，
空值或非法筛选值返回 `400`。

### 4.2 事件轮询与 SSE

`JobEvent` 字段：`job_id`、`sequence`（任务内从 1 单调递增）、`type`
（`state`\|`progress`\|`log`\|`artifact`）、`level`、`message`、`progress`、
`data`、`created_at`。

`GET /jobs/{id}/events` 支持 `after`（默认 0）、`limit`（`0..1000`，默认 256）、
`tail`（默认 `false`）。`GET /jobs/{id}/events/stream` 用 SSE，`id` 等于
`sequence`；`after` 与 `Last-Event-ID` 同时提供时取较大值，客户端按任务和
序号去重。每个任务最多保留最近 256 个事件；游标过期返回
`410 event_cursor_expired`，客户端改读任务快照并用其 `last_sequence` 续接。
终态后发送完已提交事件再结束订阅。事件写入与对应快照更新在同一事务提交。

SSE 开始前的失败用 HTTP 错误；开始后的失败发送 `event: error` 和结构化错误体。
断开订阅只关闭连接，不取消任务。

### 4.3 归档

`DELETE /jobs/{id}` 归档一条终态任务，返回 `204`；`queued`/`running`/
`cancelling` 返回 `409 job_state_conflict`。`DELETE /jobs/completed` 原子归档
所有可见终态任务（含空历史），活动任务保持可见。归档后单条查询、取消和事件
接口返回 `404 operation_not_found`；已打开的事件流若观察到归档则发送错误并
结束。任务行、事件和产物关联仍保存，不卸载实例或删除文件；事务失败不会部分
隐藏历史。

### 4.4 下载任务

`GET /api/v1/jobs/kinds` 返回当前服务接受的任务类型，不再有内嵌 schema：

```json
{"data":[{"kind":"download.huggingface"},{"kind":"download.modelscope"}]}
```

`POST /api/v1/jobs` 只接受这两种 `kind`，其他返回 `501 job_kind_not_supported`。
`payload` 字段：

| 字段 | 类型 | 约束 |
|---|---|---|
| `repo_id` | String | 必填，`owner/name` 形式 |
| `destination` | String | 必填，`1..1024`；相对所选下载根，开头的 `models/` 会被去掉；已存在目录不覆盖 |
| `download_directory` | String\|null | 使用已配置的下载目录 ID |
| `revision` | String\|null | 固定版本 |
| `repo_type` | String | 仅 `model`（默认） |
| `include` | String[] | 最多 4096 项；只支持 `*` 与 `?` 通配 |
| `exclude` | String[] | 最多 64 项 |
| `max_workers` | Integer | `1..16`，默认 8 |
| `direct` | Boolean | 默认 `false` |
| `expected_bytes` | Integer\|null | 容量提示；校验以远端文件大小与校验和为准 |

文件在校验后作为单个目录发布。源或文件系统故障记录在任务上；停机和取消保留
可续传的半成品。`POST /jobs/{id}/retry` 只对 `failed`/`cancelled`/`interrupted`
的下载开放，创建新任务并在复用半成品前校验仓库清单；活动或已成功任务返回
`409`，没有自动重启重放。

## 5. 会话与回复

### 5.1 会话与消息

`Session`：`id`、`model`、`mode`（`text`\|`voice`\|`full_duplex`）、`state`
（`idle`\|`processing`\|`interrupted`\|`error`\|`closed`\|`listening`\|`speaking`\|
`reconnecting`）、`revision`、`title`、`runtime_instance_id`、`created_at`、
`updated_at`、`metadata`。原生服务只创建 `text` 会话。

`CreateSession`：`model`（≤255，默认 `""`）、`mode`（仅 `text`）、`title`
（≤512 或 `null`）、`metadata`（Object）。

内容部件 `TextPart`：

| `type` | 字段 |
|---|---|
| `text`、`reasoning` | `text` |
| `transcript` | `text`、`language`、`start_ms`、`end_ms` |
| `tool_call` | `call_id`（≤255）、`name`（≤255）、`arguments`（Object，默认 `{}`） |
| `tool_result` | `call_id`、`result`（任意）、`is_error`（默认 `false`） |

部件数组为 `1..1024` 项。`POST /messages` 需要 `expected_revision`、`role`
（`system`\|`user`\|`assistant`\|`tool`）和 `parts`，原子追加并返回
`{"session","message"}`；`expected_revision` 不符返回 `409 revision_conflict`。
历史读取最多 10000 条消息、16 MiB 部件；旧数据库中的媒体部件在读取时保留。

`fork` 可指定 `at_message_id`、`include_message`（默认 `true`）、`title`、
`model`。`rewind` 需要 `expected_revision` 与 `at_message_id`。`export` 返回
`SessionArchive`（`format: "mfq-session-v1"`、`session`、`messages` ≤10000、
`media` 恒为空数组）；含媒体的会话导出返回 `501 media_unavailable`。
`POST /sessions/import` 原子导入，返回 `messages_imported` 与恒为 0 的
`media_imported`。有活动回复的会话拒绝删除。

### 5.2 创建回复与幂等

`POST /sessions/{session_id}/responses`：

| 字段 | 类型 | 约束 |
|---|---|---|
| `request_id` | UUID | 必填，幂等键 |
| `expected_revision` | Integer | 必填，`≥0` |
| `input` | `TextPart[]` | 必填，`1..1024` |
| `stream` | Boolean | 默认 `true`；不计入幂等指纹 |
| `sampling` | Object | 只记录显式覆盖值：`max_tokens`、`temperature`、`top_k`、`top_p`、`presence_penalty`、`frequency_penalty`、`repetition_penalty`、`seed`、`enable_thinking`、`enable_vision`、`enable_mtp`、`mtp_max_draft_tokens`（`1..5`）、`reasoning_effort`（仅 `null`） |
| `model` | String\|null | `1..255` |
| `system_prompt` | String\|null | ≤32768 |
| `include_reasoning_history` | Boolean | 默认 `true` |
| `input_role` | String | `user`（默认）\| `tool` |
| `tools` | Object[] | ≤128 |
| `tool_choice` | String\|Object | `auto`\|`none`\|`required` 或选择器 |
| `response_format` | Object | — |

只接受上表列出的字段，其余返回 `400`。缺省的采样字段在执行时取所选模型的
原生运行配置；回复的 `settings.sampling` 保留覆盖值，`performance.sampling`
报告实际执行参数。

`SessionResponse`：`id`、`request_id`、`session_id`、`status`
（`running`\|`completed`\|`failed`\|`cancelled`）、`output_message_id`、
`output`（`TextPart[]`）、`finish_reason`、`usage`、`performance`、`settings`、
`error`、`created_at`、`completed_at`。

回复先在短事务中校验请求、版本与幂等键，保存输入和 `running` 回复并冻结历史；
模型加载在事务外进行。加载期间会话为 `processing`、`runtime_instance_id` 为
`null`，查询与取消保持可用。加载错误和原生 profile 引起的参数组合错误保存为
该回复的 `failed` 终态；取消加载等待进入 `cancelled`。原生提交前的
`cancelled`/`failed` 回复其 `usage` 与 `performance` 为 `null`。同一
`request_id` 重放已保存的终态结果，不重新加载；内容变化则冲突。取消发起者
仍须等同步工厂返回后收尾。`POST /responses/cancel` 返回当前记录，它可能仍是
`running`；终态从原流或回复列表读取。

### 5.3 会话 SSE

`stream` 为 `true` 时返回 `text/event-stream`。每帧：

```text
event: <payload.type>
id: <sequence>
data: {"protocol_version":"1.0","session_id":"...","sequence":0,"timestamp":"...","payload":{...}}

```

`sequence` 从 0 开始。`payload.type` 与事件名相同，取值：

| `payload.type` | 字段 |
|---|---|
| `session.state` | `state`、`revision`（开场一帧，终态后再送一帧） |
| `response.text.delta` | `response_id`、`delta` |
| `response.reasoning.delta` | `response_id`、`delta` |
| `response.tool_call.delta` | `response_id`、`index`、`call_id`、`name`、`arguments_delta` |
| `response.completed` | `response_id`、`finish_reason`、`usage`、`performance` |
| `response.interrupted` | `response_id`、`reason`（`cancelled`） |
| `error` | 管理错误体 |

回复不是 `running`（幂等重放）时，先按已保存的 `output` 重放增量片段，再送
终态快照。终态事件在落盘成功后才发送。会话 SSE 不支持 `Last-Event-ID` 续传；
连接中断后用相同 `request_id` 重发以走重放语义。

### 5.4 保存失败

落盘失败返回 `response_save_failed`，`details` 分别给出 `execution_status` 与
`saved: false`。同一回复的后续读取仍报告该失败，不自动重试，也不会把
`running` 快照重放成成功。数据库事务失败则回滚：历史查询可能仍读到此前提交的
`running` 记录，该记录不表示执行仍在继续。重启恢复会把遗留回复标记为
`failed`、会话标记为 `interrupted`，并保留已提交的用户输入。

## 6. `/v1` 兼容层

`/v1` 直接服务 OpenAI 与 Anthropic 客户端。`/v1/messages*` 与 `/v1/models/{id}`
恒为 Anthropic；`/v1` 与 `/v1/models` 上出现 `Anthropic-Version`、
`Anthropic-Beta` 或 `X-Api-Key` 头时改用 Anthropic 形状，否则为 OpenAI。
这些入口共用鉴权、`inference` scope 和连接限额。`model` 字段按别名表解析：
先别名，再登记模型名，最后是已加载实例 ID。

### 6.1 服务广告与模型列表

`GET /v1` 返回 `{"status":"ok","service":"tyloinfer-server","endpoints":
["/v1/models","/v1/chat/completions"]}`，Anthropic 请求下第二项为
`/v1/messages`。

`GET /v1/models` 公布原生支持的登记模型与健康实例，用别名或原名，名称排序；
元数据查询不加载模型。OpenAI 形状为 `{"object":"list","data":[{"id","object":
"model","created":0,"owned_by":"mfq"}]}`，`created` 是纪元哨兵而非加载时间。
Anthropic 形状为 `{"data":[{"id","type":"model","display_name","created_at"}],
"has_more","first_id","last_id"}`，支持 `limit`（`1..1000`，默认 20）、
`after_id`、`before_id`。`GET /v1/models/{model_id}` 返回 Anthropic 详情，
未知 ID 返回 `404`。

### 6.2 `POST /v1/chat/completions`

字段校验通过并预留连接额度后才获取登记或已加载模型；同一产物的并发请求共享
一次加载结果与错误，等待者可独立取消。成功流的响应头在加载与原生提交成功后
发出，并带 `X-Request-ID`。断开会取消等待或原生生成，但不会中断共享的同步
加载，也不会在加载后替该客户端提交推理。

| 字段 | 类型 | 约束 |
|---|---|---|
| `model` | String | 必填；登记模型名、别名或已加载实例 ID |
| `messages` | Object[] | 必填且非空；`role` 为 `system`\|`developer`\|`user`\|`assistant`\|`tool`，`content` 为字符串或部件数组 |
| `tools` | Object[] | `{type:"function",function:{name,description,parameters}}` |
| `tool_choice` | String\|Object | `none`\|`auto`\|`required` 或 `{type:"function",function:{name}}`；命名函数必须出现在 `tools` 中，`required` 需要至少一个工具 |
| `parallel_tool_calls` | Boolean | — |
| `response_format` | Object | `text`\|`json_object`\|`json_schema`；`json_schema` 也可作为顶层字段，两者同时给出返回 `400` |
| `max_tokens` | Integer | `1..2147483647`；`max_completion_tokens` 优先 |
| `temperature` | Number | `0..10` |
| `top_p` | Number | `(0,1]`；本采样器下 `top_p < 1` 要求 `top_k > 0` |
| `top_k` | Integer | `0..1024` |
| `stop` | String\|String[] | 最多 16 个非空串 |
| `seed` | Integer\|null | 省略时随机 |
| `presence_penalty`、`frequency_penalty` | Number | `-2..2` |
| `repetition_penalty` | Number | `(0,10]` |
| `n` | Integer\|null | 只支持 `1`，其他值返回 `400 unsupported_parameter` |
| `stream` | Boolean | 默认 `false` |
| `stream_options.include_usage` | Boolean\|null | 默认 `false` |

未列出的标准变体（如 `max_completion_tokens`、`repetition_penalty`）也被接受，
未知字段不会被拒绝；但 `logprobs: true` 与非零 `min_p` 返回
`400 unsupported_parameter`。采样缺省取自原生模型 profile。请求体上限 1 MiB。

非流式响应：

```json
{
  "id": "chatcmpl-<uuid>",
  "object": "chat.completion",
  "created": 1750000000,
  "model": "requested-model",
  "choices": [{"index": 0, "message": {"role": "assistant", "content": "..."}, "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
}
```

`choices` 恒为 1 项。`message` 在模型产生推理输出时含 `reasoning_content`，
有工具调用时含 `tool_calls[]`（`id`、`type:"function"`、
`function{name,arguments}`，`arguments` 是 JSON 编码字符串）。`finish_reason`
为 `stop`\|`length`\|`tool_calls`。`created` 是受理时刻的 Unix 秒。

流式响应是 `chat.completion.chunk` 序列：首帧 `delta` 为 `{"role":"assistant"}`，
随后是文本、`reasoning_content` 或 `tool_call` 增量片段，最后一个 chunk 的
`delta` 为空并带 `finish_reason`。`stream_options.include_usage` 时额外发送一个
`choices` 为空数组、带 `usage` 块的 chunk。流总是以 `data: [DONE]` 结束；
失败时用 error 对象取代成功终态。工具标识在流出后发生变化返回
`502 invalid_tool_output`；流开始前的原生取消返回 `409 request_cancelled`，
开始后的取消以 SSE error 帧结束，不发成功终态。

### 6.3 兼容错误信封

`/v1` 不用管理错误信封。OpenAI 形状：

```json
{"error":{"code":"invalid_request","message":"...","retryable":false,
  "details":{},"type":"invalid_request","param":"top_p"}}
```

`type` 复制 `code`，`param` 在运行时报出被拒字段时为字段名，否则为 `null`。
流开始后的失败把同一对象作为 SSE `data` 帧发出。

Anthropic 形状：

```json
{"type":"error","error":{"type":"invalid_request_error","message":"..."}}
```

`error.type` 按状态映射：`401` → `authentication_error`，`403` →
`permission_error`，`404` → `not_found_error`，`413` → `request_too_large`，
`429` → `rate_limit_error`，`503` → `overloaded_error`，其余 4xx →
`invalid_request_error`，其他 → `api_error`。

### 6.4 `POST /v1/messages`

Anthropic `2023-06-01` 子集，严格校验：未知字段返回
`400 unsupported_parameter`；显式 `Anthropic-Version` 只接受 `2023-06-01`。

| 字段 | 类型 | 约束 |
|---|---|---|
| `model` | String | 必填 |
| `max_tokens` | Integer | 必填，`1..2147483647` |
| `messages` | Object[] | 必填且非空；`role` 为 `user`\|`assistant`，`content` 为字符串或 `text`\|`image`\|`tool_use`\|`tool_result`\|`thinking` 块数组 |
| `system` | String\|Object[] | 文本块数组 |
| `stream` | Boolean | 默认 `false` |
| `temperature` | Number | `0..1` |
| `top_p` | Number | `(0,1]` |
| `top_k` | Integer | `0..1024` |
| `seed` | Integer\|null | `≥0` |
| `enable_mtp`、`enable_thinking` | Boolean | 原生开关 |
| `mfq_session_id` | String\|null | `1..128` 字节的安全标识；原生 KV 缓存会话，不是持久化聊天会话 |
| `metadata.user_id` | String\|null | — |
| `tools` | Object[] | ≤128；`{type:"custom",name,input_schema}`，名称唯一 |
| `tool_choice` | Object | `auto`\|`none`\|`any` 或 `{type:"tool",name}`；命名选择必须存在于 `tools`；`disable_parallel_tool_use: true` 返回 `400` |
| `thinking` | Object | `adaptive`\|`disabled` 映射到原生 `enable_thinking`；显式 `budget_tokens` 在校验后返回 `501`；与 `enable_thinking` 冲突返回 `400` |
| `stop_sequences` | String[] | ≤16；空数组可接受，非空返回 `501`（原生终态事件尚未暴露命中的停止串） |

`tool_use`/`thinking` 只允许出现在 `assistant` 消息，`tool_result` 只允许在
`user` 消息。采样缺省取自原生模型 profile。图片只透传给原生输入校验，本路由
不抓取或解码。请求体上限 1 MiB；累计输出或保留的工具参数上限 16 MiB 与 128 次
调用。

非流式返回 `Message`：`id`、`type:"message"`、`role:"assistant"`、`model`、
`content`（`text`\|`tool_use`\|`thinking` 块）、`stop_reason`
（`end_turn`\|`max_tokens`\|`tool_use`）、`stop_sequence`（恒 `null`）、
`usage{input_tokens,output_tokens}`。响应头含 `X-Request-ID` 与 `request-id`。

流式事件：`message_start`（初始 `usage` 为 0）、`content_block_start`、
`content_block_delta`、`content_block_stop`、可选 `ping`、`message_delta`
（最终原生 `usage`）、`message_stop`；失败时 `error` 取代成功终态。本地
thinking 的 `signature` 为空串。

### 6.5 `POST /v1/messages/count_tokens`

原生 runtime 不暴露精确 prompt 计数，鉴权通过后恒返回
`501 unsupported_parameter`。

## 7. Python 后端

Python 服务（`tyloinfer/server`）仍是独立入口，其独有接口——媒体与文档、
实时/语音、量化、数据集与评估、集群节点、运行时 profile 与缓存管理等——以
`tyloinfer/server` 代码及其 `tyloinfer/server/protocol/openapi.json` 为准。
本文只描述原生 C++ server 已实现的能力，不能把 Python 服务的接口清单当作
原生服务已实现的能力。
