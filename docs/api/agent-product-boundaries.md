# Agent 产品解耦与剩余受限接口

## 已挂载固定接口

全部仍通过 `/tyloinfer`，仅owner可用。拒绝未知字段、URL／路径／方法透传；写请求发出后断链、取消、5xx或无效成功响应均保留 `tyloinfer/outcome-unknown`、`retryable:false`，不自动重放。

| RPC | 精确payload | 返回与边界 |
|---|---|---|
| `connection/info` | `{}` | 固定读取已配置TyloInfer核心的health和listener；返回 `{base_url,openai_base_url,anthropic_base_url,anthropic_port,protocol_version}`。地址来自核心provider，不使用Agent origin；拒绝userinfo、query、fragment及包含已解析凭据值的地址。Anthropic未监听时两个相关字段为null，不猜默认端口。不回传密钥或凭据引用。核心不兼容／不可达时失败，不伪造可用地址。 |
| `models/load` | `{request,companion_ids?}` | request保持原生ModelLoadRequest，但禁止Client传 `gguf_companions` 路径。companion_ids为0–16个唯一的32位小写hex工件ID。Host固定GET每个 `/api/v1/models/{id}/companion-reference`，核对ID、GGUF格式和路径后，才将解析出的路径写入原生gguf_companions并POST固定load。全部解析成功才发一次load；不查Host本机文件，不退回Client路径。返回原生OperationAccepted。 |
| `jobs/delete` | `{job_id,confirmed:true}` | 先固定GET作业并核对身份及终态，再DELETE同ID。允许任意kind的succeeded/failed/cancelled/interrupted记录；活动态不取消、不强制。原生204映射 `{ok:true,value:null}`。只归档记录，不删除模型或下载文件。 |
| `jobs/clear-completed` | `{scope:'all-completed',confirmed:true}` | 固定DELETE `/api/v1/jobs/completed`，由原生数据库原子筛选终态；原生204映射null，不编造清理数量。不接受省略scope或泛化all，不清理活动队列，不运行／重试／取消任何作业。确认须明确包含非下载作业的全部终态记录。 |
| `cache/configure-directory` | `{directory_id,scope:'all-loaded-models',confirmed:true}` | 目录ID必须为核心已提供的32位小写hex。Host固定GET `/api/v1/models/directories?directory_id=...`，核对current_id与current_path后，仅PUT `/api/v1/runtime/memory-policy` 的 `{prefix_directory:<核心路径>}`。返回原生202 OperationAccepted，通过既有service/get和service/cancel跟踪原生memory作业。不能直接传路径；此入口不支持null恢复默认。 |

companion-reference由核心根据其当前catalog解析，仅返回完整且实际存在的已登记GGUF工件：`{model_id,format:'gguf',path}`。可包含已登记表组件，最终与主模型、架构和后端的兼容性仍由原生加载器校验。缺失／不完整／非GGUF明确返回404或422；老核心缺此路由时明确失败，不从Host猜路径。

缓存目录变更不是无害的路径保存：原生全局内存规划可能先卸载再重载所有已加载实例，清除活动KV；忙时返回真实作业失败，不替用户取消请求。确认必须显示全局范围与重载影响，不能隐藏在自动保存、轮询或草稿恢复里。PUT保留核心对现有预算的部分更新合并，不由Host写默认容量。目录存在、可写性及模型恢复最终由核心验证；失败不能声称原子完成或完全无副作用。

远程路由、监听／认证修改以及apply_global_cap批量重载不在本批开放；现有provider配置不能宣称等价恢复它们。

## 独立产品与遗留门面

- 实际运行包与依赖清单归位dsh/，官方DSH版本由dsh/package.json固定到0.2.2-rc.2；官方源码不随本仓库分发，未宣称源码逐位重现二进制。
- 核心不再默认导入、挂载或关闭旧Python Agent任务／设置门面；相关源码迁到Host插件下非分发的legacy-python兼容测试目录，未删除任何旧数据库、配置或凭据。
- 核心Python包移除旧SDK extra；已离线更新锁文件并重生成公开OpenAPI，协议回归通过。旧门面测试显式加载兼容工厂，不把它重新挂回正式核心。核心隔离测试明确阻断DSH／旧门面导入，默认应用无旧Agent state与路由。
- Studio显示设置与纯推理入口由前端独立保留，Agent版使用原生DSH设置，不要求Studio依赖DSH。

## 下载启用

正式核心提供 `--downloads auto|disabled|enabled`，默认auto保留已有依赖探测语义；disabled明确不注册两类Hub下载handler，enabled要求至少一个SDK或外部程序已存在，否则启动时明确报错，不安装依赖。

开发预览默认关闭下载。客户端按核心返回的 `job_kinds` 显示可用下载来源。核心注册下载能力时不会自动联网、下载或加载模型。

## 版本与恢复

启动器从所选官方CLI读取实际版本，要求CLI/base/Web属于同一版本，再按两个产品插件的明确engines声明检查兼容性，不再以Python启动器内写死的版本阻止独立选择runtime。当前插件仍只声明已验证的0.2.2-rc.2；未测新版本不会自动放行，不伪装成任意版本升级都可用。

版本拒绝发生在准备／写入profile之前；保留已有配置、会话与插件链接，回到原支持runtime可继续启动。Launcher将已核实运行版本交给Host，Host报告实际运行版本与插件包版本；未通过产品启动器运行时，未知运行版本应为null而非伪造标签。
