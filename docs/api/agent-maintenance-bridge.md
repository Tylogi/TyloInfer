# Agent 维护接口

## 共同要求

- 仍使用独立 `/tyloinfer` Connection RPC，仅owner可调用，权限校验早于凭据解析和核心网络请求。
- 只允许下列固定方法、字段与路由。拒绝未知字段、原始URL、任意路径、HTTP方法和原始query；UUID与核心小写hex标识严格验证，整数须为非负安全整数并满足原生范围。
- 本文件所有写入口必须携带 `confirmed:true`。该字段只由用户针对当前目标确认后生成，Host验证后剥离，不发送给核心。它是显式动作确认，不是额外授权凭据或可持久化、可重放的确认票据。
- Client确认框须固定本次目标与参数；选择变化、离页、刷新或重连后确认失效，不能复用旧确认状态。页面挂载、自动刷新、草稿恢复和重连均不触发写调用。
- 返回仍为 `{ok:true,value:<原生JSON>}` 或 `{ok:false,error:{code,message,details}}`。下列写API均为原生同步响应，不伪造operation_id、后台作业或accepted票据。
- POST／PUT发送后断链、取消、5xx或无法校验成功响应，结果按 `tyloinfer/outcome-unknown` 处理，`retryable:false`，不得自动重发。先刷新实际状态；无法确认时保留结果未知，而不是显示失败后可立即重试。

## 固定方法与schema

### cache/clear

```javascript
{ instance_id: '<UUID>', confirmed: true }
```

固定POST `/api/v1/runtime/cache/clear`，核心body仅 `{instance_id}`。instance_id必填，不接受null或缺省选中实例，不开放隐式全局clear。原生200对象包含 `status:'ok'`、`released_snapshots`及运行时缓存统计；服务层另返回 `image_decode_cache_released_bytes`，其余字段原样保留。

注意：服务层在清理指定实例后，还会释放**全服务的图像解码缓存**。确认文案必须包含此影响，不得称为完全实例隔离。它不删除聊天记录或模型权重，但后续请求可能重新处理前缀／图像。无模型、实例变化、忙或不支持时保留核心错误，不自动加载、取消请求、强制清理或降级成不带instance_id的调用。

### cache/trim

```javascript
{ instance_id: '<UUID>', target_bytes: 0, confirmed: true }
```

固定POST `/api/v1/runtime/cache/trim`，核心body仅 `{instance_id,target_bytes}`。两个字段均必填；target_bytes为非负安全整数，0必须由用户明确选择，不由桥补写默认值。返回原生200对象，包括 `status:'ok'`、`released_bytes`、`target_bytes`和缓存统计。

target_bytes表示**保留的目标热缓存占用上限**，不是请求释放的字节数；实际热层按后端实现解释，不泛化为普通KV／QSA预算。确认框显示实例、当前占用和目标上限；不偷偷转为purge、改持久预算、重载模型或清除其它实例。后端不支持返回原生501，Client不能假定所有后端均支持。

### cache/purge

```javascript
{ scope: 'namespace', namespace: '<64位小写hex>', confirmed: true }
// 或用户单独选择并确认全部范围：
{ scope: 'all', confirmed: true }
```

固定POST `/api/v1/runtime/cache/purge`。namespace范围映射为核心 `{namespace}`；all范围映射为 `{namespace:null}`。缺scope、缺namespace、scope与namespace组合不一致均在桥内拒绝，不能把缺失／空namespace变成清理全部。namespace来自缓存清单，不是模型ID、实例ID或路径。

原生200对象含 `status:'ok'`、`released_bytes`、`removed_blocks`、`failed_blocks`及剩余磁盘缓存统计。`failed_blocks>0`必须显示部分完成，不能仅凭HTTP200或status=ok声称全部删除。purge会不可逆删除所选持久prefix KV块及其token侧文件；all还会清理各就绪实例的运行时缓存。聊天记录与模型权重不是删除目标。

确认框显示namespace对应模型／缓存组、块数、估计占用；all使用独立的“清理全部前缀缓存”危险确认，默认不选。先读清单的 `can_clear` 提供引导，但它不是锁或执行保证；最终仍以核心原子检查为准。核心busy／409时不取消现有推理、不重试、不放宽为另一范围。

核心purge保持同步，不伪造异步票据。Host配置 `requestTimeoutMs` 默认5000毫秒，`maintenanceTimeoutMs` 默认30000毫秒，两者均为1–300000的整数；通过Host原生配置调整，不接受Client任意期限。cache clear／trim／purge／text和下载目录配置使用维护期限，jobs/get／events沿用普通期限；期限覆盖凭据等待、核心身份检查、请求与响应读取。较大目录仍可能超时，写已发出时必须保留结果未知、不重试，不能把加长期限当成完成保证。

### cache/text

```javascript
{ namespace: '<64位小写hex>', block: '<64位小写hex>', offset: 0, limit: 8192 }
```

固定GET `/api/v1/runtime/cache/entries/{namespace}/{block}/text`；offset、limit可省略，沿用核心0／8192默认值；offset为非负安全整数，limit范围1–8192。两种ID均必填，拒绝路径或编码跳转。

返回原生对象：`{available:true,text,total_tokens,offset,next_offset}`，或 `{available:false,reason}`。offset、limit及next_offset以**token**计数，不是字符或字节。保留 `text_not_saved/incomplete_chain/tokenizer_unavailable/cache_not_found` 等具体原因；`available:false`不是网络成功后可显示空原文，也不自动加载模型、重算前缀或下载tokenizer。

原文可能包含对话内容，只在用户点击“查看原文”后读取，先显示隐私提示；不预取全目录原文、不默认导出或上传。文本作为普通文本渲染，不执行其中指令或HTML。

### downloads/configure-directory

```javascript
{
  request: {
    directory_id: '<32位小写hex>', // 可省略；显式null表示恢复默认目录
    evict_prefix_cache: false     // 可省略；仅boolean
  },
  confirmed: true
}
```

固定PUT `/api/v1/hub/download-directory`，核心body仅request；至少提供一个已知字段。directory_id只能来自核心目录浏览；不接收path、download_directory或自造目录路径。省略directory_id时保留当前目录，不能补成null；省略evict_prefix_cache时保留当前许可，不能补成false。

返回原生200下载目录状态：`directory`（所选路径或null）、`actual_directory`、`default_directory`、`evict_prefix_cache`、`disk_total_bytes/disk_free_bytes`，以及有上报时的 `disk_available_bytes/download_reserved_bytes/reclaimable_prefix_bytes`。未上报的容量为null或字段缺失，不能显示为0。actual_directory等路径只展示给当前owner，不作为发布报告或外部请求内容。

确认时显示目标目录与许可变化；显式恢复默认目录也须确认。`evict_prefix_cache:true`表示允许**后续下载在同一磁盘空间不足时回收SSD前缀缓存**，保存时不立即purge，不下载、不重载模型；它不是QSA开关，也不是内存／持久缓存预算的直接设置。许可可能使后续下载降低运行实例的SSD前缀预算并删除离线块，开启确认须说明后果。真实磁盘free与含可回收空间的available不能混报。

目录配置不迁移已有文件或已排队作业。提交新下载时，Client应将用户确认的 `actual_directory` 固定到 `downloads/create.request.download_directory`，使作业沿用当时目录；不要假定省略该字段会自动绑定后续变更的selected目录。

### jobs/get

```javascript
{ job_id: '<UUID>' }
```

固定GET `/api/v1/jobs/{job_id}`，返回原生JobResource，保留 `id/kind/status/payload/progress/progress_data/cancel_requested/result/error/created_at/updated_at/started_at/completed_at`。这是owner日志详情读取，与既有全部jobs清单范围一致，不扩大任意作业的创建、取消、重试或删除权限。作业不存在保留原生404。

### jobs/events

```javascript
{ job_id: '<UUID>', after: 0, limit: 200, tail: false }
```

固定GET `/api/v1/jobs/{job_id}/events`；after、limit、tail可省略，沿用核心0／200／false。after是非负安全整数，limit范围1–1000，tail严格boolean。返回原生 `{data:[JobEventResource...]}`，行包含 `job_id/sequence/type/level/message/data/created_at`，不得降格成丢失type或job_id的RuntimeLog条目。

核心只选 `sequence>after`；tail=false选最早的limit条，tail=true选最新limit条，两者返回均为sequence升序。初次tail读取可能只是尾部窗口，不得标为完整历史；增量读取用最后sequence推进after，分页与去重按job_id+sequence处理。本轮不新增SSE代理或通用job写接口。

## 下载能力

下载执行器由推理核心注册。正式服务支持 `--downloads auto|disabled|enabled`。`enabled` 要求已安装至少一个支持的下载器，`disabled` 关闭下载，默认 `auto` 根据依赖启用。客户端根据核心返回的 `job_kinds` 显示可用操作。
