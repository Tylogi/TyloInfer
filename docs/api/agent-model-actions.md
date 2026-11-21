# Agent 模型与下载补充动作

## 共同边界

全部通过独立 `/tyloinfer` Connection RPC，仅owner可调用。成功为 `{ok:true,value}`，失败为 `{ok:false,error:{code,message,details}}`。拒绝未知字段、客户端URL、原始query、HTTP方法和任意job kind。工件ID为32位小写hex，实例／作业ID为UUID，删除token为64位小写hex。

本批状态修改均必带 `confirmed:true`，Host校验后剥离。确认固定目标和完整参数；刷新、重连、恢复草稿、切换目标或重复点击不得自动重发。模型删除确认必须展示最新原生plan的范围、文件列表与总容量；上下文修改须明确将重载目标实例并清除活动KV。不能将accepted当成已完成。

写请求含PATCH、DELETE、POST；发送后断链、调用方取消、5xx或无效成功响应均为 `tyloinfer/outcome-unknown`，`retryable:false`。HTTP取消不代表核心作业取消。新增Hub的POST只做读取／网络探测，不创建作业或修改设置，其失败不伪装为“状态写入结果未知”，仍不自动重试。读取和写入均保留原生错误，禁止重定向、受响应限额及凭据回显保护约束。

## Hub读取与显式网络动作

三个入口都可能访问外部Hub，必须由用户点击查看模型卡／探测／检查复用触发，不在页面挂载、轮询或重连时触发；不接受密钥或自定义域名。provider仅 `huggingface|modelscope`；repo_id仅合法 `owner/name`，不接受URL或跳转段。revision为1–255字符无控制字符字符串。

| RPC | 精确payload | 固定核心调用与value |
|---|---|---|
| `hub/readme` | `{provider,repo_id,revision}` | GET `/api/v1/hub/models/{provider}/{owner}/{name}/readme?revision=...`；`{content:string|null}`。原文是外部不可信内容，Client须安全呈现，不当指令执行。 |
| `hub/probe-sources` | `{sources:[{provider,repo_id,revision?}]}` | POST `/api/v1/hub/download-sources/probe`；sources为1–2项，revision可省略或null；value为原生 `{data:[{provider,repo_id,bytes_per_second,route?}]}`，route为direct/proxy。0保留为原生0，不伪造成功速度。 |
| `hub/download-reuse` | `{request:{provider,repo_id,destination,revision?,download_directory?}}` | POST `/api/v1/hub/download-reuse`；value为原生 `{reusable_files:{文件名:字节数}}`。revision可省略或null。只检查已有文件，不启动下载，不创建目录。 |

reuse的destination必须是1–1024字符相对路径，拒绝绝对路径、盘符、URL、冒号、控制字符、`.`和`..`路径段。download_directory可省略或null，保留核心原有解释；显式字符串必须与核心当前 `/api/v1/hub/download-directory` 返回的 `actual_directory` 完全相同，Host先读取并核对，再发送Hub POST。不能用它选择任意目录；修改选中目录仍走已实现的目录ID配置接口。核心继续执行注册根、目录存在性与子路径约束，符号链接等最终路径安全不由Client判断。

复用检查以核心实际实现为准：当前按已知文件的类型、存在性和字节数匹配，不构成内容哈希或权重完整性认证。不得将同字节数直接宣传为同模型。

## 实例策略与上下文

| RPC | 精确payload | 固定核心调用与value |
|---|---|---|
| `instances/configure` | `{instance_id,request:{pinned?,idle_ttl_seconds?},confirmed:true}` | PATCH `/api/v1/runtime/instances/{instance_id}`；request至少一项，pinned为boolean，idle_ttl_seconds为非负安全整数或null。返回完整原生 `RuntimeInstanceResource`，ID必须与目标相同。不是重载操作。 |
| `instances/yarn` | `{instance_id}` | GET `/api/v1/runtime/yarn/{instance_id}`；返回原生 `YarnContextInfo`，保留supported、native_context、maximum_context、maximum_factor、enabled、effective_factor。 |
| `context/configure` | `{request:{instance_id,context_size,yarn_enabled?,qsa_kv_offload?,kv_quantization?,kv_memory?},confirmed:true}` | POST `/api/v1/runtime/context`；仅明确单实例，返回原生202 `JobResource`，kind必须是 `runtime.context.configure`。 |
| `context/get` | `{operation_id}` | GET `/api/v1/jobs/{operation_id}`；仅允许kind为 `runtime.context.configure`。 |

context_size必填，为null或512–2147483647整数。null保留核心自动配置语义；不能将缺实例变成全局重载。不接受 `apply_global_cap`、模型路径、任意作业字段或另一个reload入口。yarn_enabled可省略、null或boolean。

三个嵌套设置可省略或null；省略／null均原样发核心，不补写Host默认值。非null对象只允许下列原生字段：

- qsa_kv_offload：`enabled?` boolean；`budget_bytes?`整数1–2^50；`ram_budget_bytes?`为null或整数0–2^50。
- kv_quantization：`enabled?` boolean；`bits?`为2、2.5、3、3.5、4、6、8；`algorithm?`仅turboquant。
- kv_memory：`qsa_mode?`为auto/fp16/affine8/tq4/legacy；`placement?`为auto/auto_ssd/vram/ram；`allocation?`为preallocated/dynamic；`gpu_budget_bytes?`为null或整数1–2^50；`ram_preallocated?` boolean；`ram_budget_bytes?`为null或整数0–2^50。ram_preallocated显式false时，placement必须显式auto_ssd，与核心校验一致。

Auto和固定精度不混用、不替换；通用桥不写死开发设备容量，也不改变QSA与普通KV的原生含义。

**不存在 `context/cancel` RPC。** 核心对queued/running/cancelling的上下文重载明确返回409 `context_reload_in_progress`，提交后不可取消。Client应继续跟踪真实作业直到终态，不能显示可用取消按钮，也不能将HTTP断开当成撤销。通用jobs/get可用于日志只读，不因此扩大取消、重试或删除权限。

## 工件删除

| RPC | 精确payload | 固定核心调用与value |
|---|---|---|
| `models/deletion-plan` | `{model_id}` | GET `/api/v1/models/{model_id}/deletion`；完整原生 `ModelDeletionPlan`：model_id、name、files、total_bytes、token。 |
| `models/delete` | `{model_id,token,confirmed:true}` | POST `/api/v1/models/{model_id}/delete`，body仅 `{token}`；原生200 `{files,total_bytes}`。 |

plan的model_id必须匹配请求目标；Client只允许用本次展示且明确确认的plan token。token不是认证密钥，但不能用作跨目标、持久确认或自动重试票据。核心校验内容变化、加载中、运行引用、下载冲突和文件范围；409不强制、不自动卸载或取消其他作业。删除可能部分完成并失败，不能回滚承诺或显示“未删除任何文件”；出现结果未知时先刷新目录与模型清单，再由用户决定。

## 下载重试与移除记录

| RPC | 精确payload | 固定核心调用与value |
|---|---|---|
| `downloads/retry` | `{operation_id,confirmed:true}` | 先GET `/api/v1/jobs/{id}`核对kind和状态；仅failed/cancelled/interrupted允许POST同ID `/retry`。返回原生202 `JobResource`，采用实际返回的id，不假定必然新建或仍为原id。 |
| `downloads/delete` | `{operation_id,confirmed:true}` | 先GET `/api/v1/jobs/{id}`核对kind和终态；DELETE同ID，原生204映射为 `{ok:true,value:null}`。Client须允许该入口成功值为null。 |

仅允许 `download.huggingface|download.modelscope`，不开放通用jobs/retry、jobs/delete或clear-completed。delete仅允许succeeded/failed/cancelled/interrupted；运行中、排队中和取消中不自动取消或强制删除。删除是归档队列记录，不删除已下载模型文件，不回收磁盘空间；文件删除必须走工件plan与token闭环。

retry须再次核实该下载kind执行器可用；无执行器不发送POST。核心可能复用另一条相同目标的活动下载票据，Host保持返回原值；用户界面自动跟踪真实id，不要求用户填写票据。先读再写不是锁，最终状态冲突仍以核心原子校验为准。
