# Agent 产品模型管理桥

这是外部 Host 插件提供的受限 Connection RPC，不是新的推理核心 HTTP API。仅允许已认证的 owner；所有调用使用独立 `/tyloinfer` 通道，不占用官方 Gateway 的 `/api`。

## 读取接口

```javascript
const result = await ctx.connection.rpc.call('/tyloinfer', 'core/read', { resource: 'loaded_models' }, signal);
```

| resource | 核心固定 GET | 含义 |
|---|---|---|
| `models` | `/api/v1/models` | 已登记模型工件，不等于已加载 |
| `loaded_models` | `/api/v1/runtime/models` | 已加载模型列表，`value.data` |
| `instances` | `/api/v1/runtime/instances` | 运行实例清单，`value.data` |

`loaded_models` 行保留核心返回的 `id`、`model`、`state`、`instance_id`；卸载使用 `instance_id`，不能把工件 ID、模型名称和运行实例 ID 混用。原有 `health/status/resources/capabilities` 读取保持不变。

## 显式加载与卸载

```javascript
const accepted = await ctx.connection.rpc.call('/tyloinfer', 'models/load', {
  request: { model: selectedModel, context_size: selectedContext }
}, signal);

const acceptedUnload = await ctx.connection.rpc.call('/tyloinfer', 'models/unload', {
  request: { instance_id: selectedInstanceId, force: false }
}, signal);
```

桥分别调用固定的 `POST /api/v1/models/load` 和 `POST /api/v1/models/unload`。`request` 使用核心现有 `ModelLoadRequest`／`ModelUnloadRequest` 字段；桥不补写默认值、不改 Auto／固定精度、SSD 或内存策略。请求必须为JSON对象，大小上限64 KiB；完整原生字段校验仍由核心执行。

配套资产使用顶层 `companion_ids` 或 `companion_paths`。前者由 Host 向核心解析已登记 GGUF 工件，后者接收用户明确选择的核心所在设备的绝对路径。原生加载器验证文件和兼容性。见[产品边界](agent-product-boundaries.md)。

成功仅表示核心返回 HTTP 202 和真实 `{operation_id, status:'accepted'}`。这不是完成、不是模型 ready，也不能据此显示加载成功。加载／卸载只能由明确用户动作触发；页面挂载、刷新、重连和状态恢复不得触发写操作。强制卸载必须是明确选择，不得把 `force` 默认为 true。

## 操作状态与取消

```javascript
const operation = await ctx.connection.rpc.call('/tyloinfer', 'operations/get', {
  operation_id: accepted.value.operation_id
}, signal);

const cancelRequest = await ctx.connection.rpc.call('/tyloinfer', 'operations/cancel', {
  operation_id: accepted.value.operation_id
}, signal);
```

- `operations/get` 对应固定 `GET /api/v1/jobs/{operation_id}`。
- `operations/cancel` 先读同一操作，再按需要调用固定 `POST /api/v1/jobs/{operation_id}/cancel`。
- 仅暴露 `kind=model.load` 或 `model.unload`，不是任意后台作业管理或代理。
- 返回完整原生 `JobResource`；运行态为 `queued/running/cancelling`，终态为 `succeeded/failed/cancelled/interrupted`。
- `queued/running` 且未请求取消时可以提交取消请求；已经请求取消时只返回现有状态，不重复 POST；不接受对终态的新取消。
- `cancel_requested=true` 或 `cancelling` 仍不是取消完成，必须继续读取原生状态。HTTP 请求的 `AbortSignal` 只取消该次传输，不能冒充后台操作已取消。
- 操作 `succeeded` 后重新读取已加载模型与实例清单，按核心实际状态更新页面。

## 返回与失败语义

成功为 `{ok:true,value:<核心 JSON>}`；失败为 `{ok:false,error:{code,message,details}}`。

- `tyloinfer/bad-request`：不支持的方法、非法包装、标识或传输大小。
- `tyloinfer/forbidden`：不是 owner。
- `tyloinfer/core-incompatible`：核心身份或协议不匹配。
- `tyloinfer/core-unavailable`／`tyloinfer/cancelled`：未提交写操作前的连接或取消问题。
- `tyloinfer/core-rejected`：核心明确拒绝，`details.http_status` 和 `details.core_error` 保留其结构化错误；不自动重试写操作。
- `tyloinfer/core-invalid-response`：无效、过大或包含受保护凭据材料的读取响应。
- `tyloinfer/operation-not-supported`／`operation-not-cancellable`：非模型作业或不能提交新取消。
- `tyloinfer/outcome-unknown`：写操作可能已经提交，但传输、服务端错误或响应校验无法确认结果。`details.retryable=false`，且可能没有 `operation_id`。不得伪造票据、成功或取消；先刷新状态并人工核实，不能自动重发。

Host 不重试写请求，也不跟随 HTTP 重定向。固定官方 Web Connection 的 unary 调用本身为单次发送；Client 同样不得添加写重试、重连重放或重复点击提交。Client 到 Host 的传输失败也应保守视为结果未知，而不是立即重发。

Host 从官方 provider 设置读取端点，通过官方 credentials 解析凭据；凭据不进入 Client 参数或返回值。只接受固定资源和方法，不接受 URL、HTTP 方法、任意路径或原始查询参数。页面读取新增的白名单命名过滤与分析入口见[推理页面接口](agent-inference-pages.md)。响应有 4 MiB 传输上限；调用期限和调用方取消覆盖凭据等待与网络阶段。以上不是推理性能限制或模型形状限制。
