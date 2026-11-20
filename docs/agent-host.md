# 独立 Agent Host 启动器

TyloInfer 推理服务与完整 DSH Host 分别启动。产品启动器位于 `scripts/agent_host.py`，仅依赖 Python 标准库，不导入推理核心或 Python Agent SDK。它通过官方 `dsh --profile` 入口启动完整 base＋Web 组合，再加载外部 Host／Client 插件，不复制或修改 DSH 内部服务。

## 两种独立产品形态

TyloInfer Studio持续作为纯推理引擎桌面软件和WebUI保留，不因Agent产品接入而删除或退役。Studio使用自己的 `studio/web/package.json` 构建入口：`npm run build` 只执行TypeScript与Vite构建，桌面Tauri入口也只调用这一路径；不要求构建、安装或启动DSH与新版Agent页面。推理核心继续独立提供HTTP API。

Agent产品使用完整DSH Host及外部插件；`plugins/studio/` 是Agent中的推理Client插件，不是独立Studio的替代安装。其 `node build.mjs` 单独生成插件产物，允许单向复用Studio的纯展示组件与样式，但不能让Studio反向导入DSH服务、插件运行时或Agent页面。当前插件构建还复用了Studio的Vite工具安装；共享构建工具和源码不等于Studio必须构建Agent。

两种入口的功能与测试分别报告。当前独立Studio无模型预览的特定设置／执行能力限制，不代表整个Studio是待淘汰的过渡产品；旧Python Agent任务／设置门面的迁出也不包括删除Studio或推理核心API。

两种形态在同机同时启动时，Agent与核心OpenAI／Anthropic监听须使用不同端口；此启动器只选择Agent端口，不替核心修改监听或启动另一个推理服务。

## 安装布局与版本

- `dsh/upstream/`：完整官方参考源码。
- `dsh/runtime/`：独立官方运行包、固定版本依赖清单与锁文件；当前开发安装已归入此处，尚未发布最终安装包。
- `plugins/host/`：推理适配、协议握手和默认组合。
- `plugins/studio/`：产品 Client 及构建产物。
- `--state-root`：使用者显式指定的私有 profile、工作区和临时目录，不属于源码或发布资源。

启动器读取所选CLI的实际发行版，要求CLI、base、Web版本一致，并与两个产品插件明确的engines声明匹配；不再将Python脚本中的固定版本当成独立选择runtime的限制。当前插件仍仅声明已验证的DSH `0.2.2-rc.2`；模拟未来版本元数据只验证预检机制，不能当作真实新发行版的兼容证据。完整源码固定到官方同版本标签 `dsh-v0.2.2-rc.2`，无本地 Core 补丁；运行包仍来自官方发布产物，未验证从源码逐位重现二进制。运行包缺失、版本不匹配或 Client 未构建时会明确退出，不回退读取其他安装或当前 DS Bot 的 profile。

## 使用

从仓库根目录运行，私有状态目录可按部署需求另行指定：

```sh
python scripts/agent_host.py --state-root .tyloinfer/agent --check
python scripts/agent_host.py --state-root .tyloinfer/agent --prepare-only
python scripts/agent_host.py --state-root .tyloinfer/agent
```

上述命令要求目标 runtime 已安装；仅有依赖清单时 `--check` 会失败。已有独立官方安装可通过 `--runtime-root <runtime-directory>` 显式选择。`--plugins-root` 可另行指定两个产品插件所在目录。启动器不会执行下载、安装或编译。

默认 Agent 监听回环地址的 8091 端口，推理模型提供方初始连接 `http://127.0.0.1:8090/v1`。`--port` 仅选择 Agent 端口；`--core-url` 仅用于首次创建 profile，已有用户设置不覆盖。默认模型列表为空，启动不加载权重。实际模型选择、外部 API 与凭据继续由官方设置服务管理。

`--check` 校验版本、产物及既有profile的产品包绑定，不写文件、不启动进程。`--prepare-only` 创建完整默认profile和两个产品包的本地链接，但不启动服务；创建前先统一检查全部既有绑定，不因后一个插件冲突而先创建前一个链接或默认文件。已有profile、用户设置和正确链接保持不变；指向其他安装或断开的旧绑定明确拒绝，不自动覆盖或改绑。检查兼容性和保留私有状态的方法见[升级说明](agent-upgrade.md)。

私有状态与缓存集中在 `--state-root`，不继承其他 DSH profile、密钥环境或 Node 注入选项。默认不打开浏览器、不打印带认证信息的地址。首次浏览器连接所需信息由 Host 插件写入该私有目录的 `connection.json`，不得作为公开链接、日志或发布资源传播。

打开已经运行的 Host 使用独立模式：

```sh
python scripts/agent_host.py --state-root .tyloinfer/agent --port 8091 --open-existing
```

该模式仅校验并读取私有连接文件，将同一回环端口的认证入口交给本机默认浏览器；不打印 URL，不校验或启动新的运行包，不重启服务。它拒绝外部地址、不同端口、嵌入式用户名密码和非入口路径。输出 `AGENT_BROWSER_OPEN_REQUESTED; EXISTING_HOST_ONLY=true` 只表示本机浏览器接受了打开请求，不代表另一台设备上的浏览器已打开。

## 旧版接口

旧 Python Agent 接口保留在 Host 插件的兼容性测试目录。独立 Studio 使用推理 API，Agent 产品使用 DSH 的会话和设置服务。
