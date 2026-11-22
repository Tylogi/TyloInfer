# Agent 独立升级

TyloInfer推理核心、完整官方DSH运行包、外部产品插件是三个独立更新单元。升级动作全部由官方 `dsh` CLI 和 npm/pnpm 完成：本仓库不下载、不安装、不重写Core，也不自动迁移用户数据。独立Studio继续使用自己的构建与设置，不因DSH升级而要求构建Agent Client。

## 更新前的只读预检

从项目根目录执行；下面的路径是使用者选择的安装与私有状态位置。预检只用只读入口：

```sh
DSH_HOME=<private-state>/home ./dsh/node_modules/.bin/dsh --version
DSH_HOME=<private-state>/home ./dsh/node_modules/.bin/dsh --profile tyloinfer-web --dump-config
npm --prefix dsh ls --depth=0
```

这三条核对的是同一组事实：CLI／base／Web 版本一致、产品 bundle 的 `engines.dsh` 声明、Host／Client 入口产物存在。它们不启动Host、不推理、不打开浏览器、不写profile，也不重建、删除或替换链接。

仓库现在没有自动拒绝：版本不一致或缺少 `app/plugins/lib/client.js` 时，官方 CLI 仍会启动。核对是人工步骤，结果不一致就停止升级。

当前插件只声明0.2.2-rc.2。未测的新版本应被拒绝安装，不应为了通过核对把兼容声明改成任意版本，或仅修改候选manifest版本。缺少客户端产物时须重新构建，不从另一安装补读。

旧双包布局（`@tyloinfer/dsh-host`＋`@tyloinfer/dsh-studio`）的 Agent profile 不会自动迁移；当前产品是一个 bundle `@tyloinfer/dsh-plugin`。要保留旧 profile，可继续用旧版 checkout，或为新布局另建一个 profile 与私有状态目录。

## 状态与回退

- 替换运行包前先停止对应Host；不得把活动服务的普通目录复制冒充一致的恢复快照。当前工具没有自动停止、快照或恢复入口。
- 私有状态独立于安装目录（`DSH_HOME`）。需要备份时应保留完整的原生会话、设置、凭据、媒体和工作区状态；仅保留浏览器连接文件不能恢复会话或授权。
- 升级前先复制一份私有状态，在副本上演练新运行包与新插件安装。核对不一致时回到原运行包与原插件安装；不需要为一次核对失败重建profile或清空用户设置。
- profile 已绑定某个插件安装、而你想改用另一个安装时，不要删掉旧链接来强行通过：先在私有副本里演练。跨插件安装的原子改绑／回退仍未实现。
- 真正的新发行版运行过后，是否能读回旧版本、是否需要恢复备份，必须由该发行版的数据兼容证据决定。升级前应在隔离状态副本中检查会话、设置和凭据能否恢复。
