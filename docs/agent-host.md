# 独立 Agent Host 启动

TyloInfer 推理服务与完整 DSH Host 分别启动。启动只用官方 `dsh` CLI：不导入推理核心，不导入 Python Agent SDK，不复制或修改 DSH 内部服务。本仓库不提供额外启动脚本；profile 的创建、bundle 安装和 patch 叠加都由官方 CLI 完成。

## 两种独立产品形态

TyloInfer Studio持续作为纯推理引擎桌面软件和WebUI保留，不因Agent产品接入而删除或退役。Studio使用自己的 `studio/web/package.json` 构建入口：`pnpm run build` 只执行TypeScript与Vite构建，桌面Tauri入口也只调用这一路径；不要求构建、安装或启动DSH与新版Agent页面。推理核心继续独立提供HTTP API。

`plugins/src/client/` 是 Agent 中的推理界面，不是独立 Studio 的替代安装。插件构建单向复用 Studio 的展示组件和样式源码，但依赖和工具由 `plugins/` 自己的 npm 清单和锁文件管理。不需要安装 `studio/web/` 的依赖，也不需要构建独立 Studio。

两种入口的功能与测试分别报告。当前独立Studio无模型预览的特定设置／执行能力限制，不代表整个Studio是待淘汰的过渡产品；旧Python Agent任务／设置门面的删除也不包括删除Studio或推理核心API。

两种形态在同机同时启动时，Agent与核心OpenAI／Anthropic监听须使用不同端口。Agent端口由 `dsh --port` 选择；启动DSH不替核心修改监听，也不启动另一个推理服务。

## 安装布局与版本

- `dsh/`：独立官方运行包的安装目录、固定版本依赖清单与锁文件（`package.json`／`package-lock.json`）；当前开发安装已归入此处，尚未发布最终安装包。
- `plugins/`：独立 Node 工作区，一个 DSH bundle `@tyloinfer/dsh-plugin`。
- `plugins/src/host/`：推理适配和协议握手的 TypeScript 源码；Host 运行入口是构建产物 `lib/host/index.js`。
- `plugins/src/client/`：产品 Client 源码；浏览器入口构建到 `plugins/lib/client.js`。
- `DSH_HOME`：使用者显式指定的私有 profile、缓存与临时目录，不属于源码或发布资源。

启动前必须核对三项。官方 CLI 不做这些核对，会照常启动并给出难定位的失败（Web 空白或 bundle 未加载）：

1. `@deepseek-ai/dsh`、`@deepseek-ai/dsh-base`、`@deepseek-ai/dsh-web-app` 三个包版本完全一致：`npm --prefix dsh ls --depth=0`。
2. `plugins/package.json` 的 `engines.dsh` 等于 CLI 版本：`./dsh/node_modules/.bin/dsh --version`。当前插件只声明已验证的 DSH `0.2.2-rc.2`；模拟未来版本元数据只验证声明机制，不能当作真实新发行版的兼容证据。
3. `plugins/lib/client.js` 已构建。缺失时不得从另一安装补读产物。

上游版本由 `dsh/package.json` 固定到官方 `0.2.2-rc.2` 发布包，官方源码不随本仓库分发，无本地 Core 补丁；运行包仍来自官方发布产物，未验证从源码逐位重现二进制。

## 使用

从仓库根目录运行；私有目录可按部署需求另行指定：

```sh
npm --prefix dsh ci
npm --prefix plugins ci
npm --prefix plugins run build

export DSH_HOME=$PWD/.tyloinfer/agent/home
./dsh/node_modules/.bin/dsh --profile tyloinfer-web --from-default-profile web --dump-config
./dsh/node_modules/.bin/dsh plugin --profile tyloinfer-web add file:$PWD/plugins
./dsh/node_modules/.bin/dsh --profile tyloinfer-web --host 127.0.0.1 --port 8092
```

`--from-default-profile web` 只在需要新建 profile 时使用，它用官方 Web 模板替换同名 profile。`--dump-config` 只打印组合出来的 profile 树，不挂载、不写用户设置。`dsh plugin` 把其余参数原样转给 profile 目录里的 pnpm，上面一行把本仓库 `plugins/` 作为 `file:` 依赖装入 profile；这一步需要网络和 pnpm。上面的命令形式来自 `dsh --help` 与 `dsh plugin --help`；本次文档改动没有执行过这一步安装。

已有独立官方安装时，把 `./dsh/node_modules/.bin/dsh` 换成该安装的 CLI 路径即可，`DSH_HOME` 仍决定 profile 落在哪里。不要复用你已有的 DS Bot profile 目录。

Provider 配置只有一处：bundle 自带的 `plugins/cordis.patch.yml`。它声明 `tyloinfer` provider 的 `baseURL`（默认 `http://127.0.0.1:8090/v1`）与 `tyloinfer-host` 挂载。不存在第二层由脚本写入的 profile patch。要改核心端点或增加 Host 选项，在自己的 profile 或 `--patch <path>` 叠加层里覆盖。

Agent 默认监听回环 8092；Core 的 OpenAI 与 Anthropic 监听默认占用 8090 和 8091，选端口时避开它们。默认模型列表为空，启动不加载权重。实际模型选择、外部 API 与凭据继续由官方设置服务管理。

私有 profile、缓存与临时文件都落在 `DSH_HOME` 下。Web 首次连接的认证入口由官方 Web 输出提供，不得作为公开链接、日志或发布资源传播。

Host 插件的 `connectionFile` 是可选项。只有在 profile 或叠加层里显式给出 `tyloinfer-host` 的 `connectionFile`，Host 才把私有认证入口写到该路径；未给出时不写任何连接文件。检查兼容性与保留私有状态的方法见[升级说明](agent-upgrade.md)。
