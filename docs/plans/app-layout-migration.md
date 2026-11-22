# 应用层目录与统一前端构建迁移计划

状态：方案已选定，迁移尚未执行。本文中的目标目录和目标命令尚不存在；本次仅编写计划。

## 1. 已确定的方向

采用 `app/` 统一控制前端依赖、类型检查、测试和构建的方案，同时保留两个完整应用的归属：

- `app/studio/`：Studio 必需的 Python 服务、运行时、工具、Web 入口和 Tauri 桌面壳。
- `app/plugins/`：插件必需的 Host、Client 入口、RPC 接入、清单和分发打包。
- `app/shared/`：两端实际共用的展示源码、样式、素材、纯函数和必要类型。
- `app/package.json`：共用前端工具链与构建入口，不代表第三个应用。

**统一的是前端构建，不是把 Python、Rust 和 C++ 都交给 npm 管理。** Python 仍由 Studio 的 `pyproject.toml` 管理；Tauri 保留 Cargo 配置；原生运行时仍由 `csrc/` 的 CMake 管理；完整发行包仍由 `packaging/` 编排。

不采用以下方案：

- 新增 `packages/inference-ui` 或独立发布的 UI 包。
- 把 Studio 的 Python 部分改名为 `python/`，打散应用归属。
- 只把 `studio/web` 改名，然后继续让插件导入另一个应用的内部文件。
- 取消 `app/package.json`，让两个前端各自维护完整工具链。

## 2. 迁移目标与非目标

### 目标

1. Studio 和插件各自保留应用专用实现，但不再互相引用私有前端源码。
2. 两端使用同一份共用展示实现，不复制组件和样式。
3. 从 `app/` 可以分别构建 Studio、普通插件和 DS Bot 插件，也可以构建两个默认目标。
4. 删除按 Studio 内部路径替换依赖的构建规则，改为明确的展示接口和应用适配。
5. 更新目录移动影响到的调用方、配置、测试、原生构建、发布流程和文档。
6. 保持既有功能、协议、包导出和用户数据行为。

### 非目标

- 不重写推理实现，不重新划分 Python/C++ 的职责。
- 不统一 Studio HTTP 与插件 RPC，不新增通用传输框架。
- 不强制统一 React 大版本，不升级 DSH、DS Bot 或其他无关依赖。
- 不重做页面，不补齐现有 Studio 与插件的功能差异。
- 不迁移用户模型、缓存、凭据、浏览器存储或 DSH profile。
- 不引入 Nx、Turborepo、独立 UI 发布流程或仓库级 npm 根工程。

## 3. 目标目录

以下省略未变化的文件。共用源码的具体清单以两端实际依赖为准，不整体搬走 Studio 的 `features/`。

```text
TyloInfer/
├── app/
│   ├── package.json                   # 前端工具链、统一构建与检查命令
│   ├── pnpm-lock.yaml                 # 应用层 Node 依赖锁文件
│   ├── pnpm-workspace.yaml            # 纳入插件清单，不纳入 dsh/
│   ├── tsconfig.json                  # 编排共享代码与两端的类型检查
│   │
│   ├── shared/                        # 源码目录，无 package.json 或独立产物
│   │   ├── features/
│   │   │   ├── models/
│   │   │   ├── runtime/
│   │   │   ├── connections/
│   │   │   └── analysis/
│   │   ├── ui/                        # display、列表、提示等共用展示
│   │   ├── styles/
│   │   ├── assets/
│   │   ├── types/                     # 展示所需数据、回调及控件接口
│   │   ├── PresentationProvider.tsx   # 确有共用需要的展示环境
│   │   └── formatters.ts
│   │
│   ├── studio/
│   │   ├── pyproject.toml
│   │   ├── uv.lock
│   │   ├── setup.py
│   │   ├── _version.py
│   │   ├── serve.py
│   │   ├── server/
│   │   ├── runtime/
│   │   ├── kernels/
│   │   ├── architectures/
│   │   ├── formats/
│   │   ├── tools/
│   │   ├── tests/
│   │   └── web/
│   │       ├── index.html
│   │       ├── vite.config.ts
│   │       ├── vitest.config.ts
│   │       ├── playwright.config.ts
│   │       ├── src/                   # Studio 页面组装、状态和 HTTP 接入
│   │       ├── src-tauri/             # 保留 Cargo、Tauri 配置与桌面实现
│   │       ├── tests/
│   │       ├── e2e/
│   │       └── dist/                  # Studio 前端构建产物
│   │
│   └── plugins/
│       ├── package.json               # Host 依赖、插件身份和分发清单
│       ├── build.mts                  # 插件目标的打包实现，由 app 命令调用
│       ├── tsconfig.host.json
│       ├── tsconfig.client.json
│       ├── cordis.patch.yml
│       ├── dsbot.patch.yml
│       ├── src/
│       │   ├── host/
│       │   └── client/                # 插件页面组装、状态和 RPC 接入
│       ├── lib/                       # 普通插件产物
│       └── dist/dsbot/                # DS Bot 分发产物
│
├── csrc/                              # 原生推理工程，位置不变
├── dsh/                               # 官方宿主安装工程，位置和锁文件不变
├── packaging/                         # 整体发行包编排，位置不变
├── scripts/
└── docs/
```

`app/studio/web/package.json` 的前端依赖和命令合并到 `app/package.json`，不保留第二份前端依赖清单。`app/plugins/package.json` 必须保留，因为宿主通过它识别插件及其导出；它不再维护另一套 Vite 等通用前端工具链。

## 4. 源码归属与依赖规则

### 4.1 允许的依赖方向

```text
app/studio/web/src ────────┐
                          ├──→ app/shared
app/plugins/src/client ───┘

Studio 页面 → Studio 状态、HTTP、桌面能力
插件页面   → 插件状态、RPC、宿主能力
```

生产前端代码遵守以下规则：

- 两个应用不能直接导入对方的私有前端源码，包括类型、样式、素材、重导出和动态导入。
- `app/shared` 不得反向导入任一应用。
- `app/shared` 不创建真实 HTTP/RPC 客户端，不读取某个应用的凭据、路由或持久化设置。组件可以调用调用方显式传入的异步操作。
- 应用专用页面、状态、传输和入口仍留在各自应用，不能为减少目录数量而全部塞进共享区。
- 集成测试可以同时验证两端；这不构成生产源码互相依赖的许可。

这不是禁止所有跨目录调用，而是禁止一个应用依赖另一个应用的私有实现。

### 4.2 迁移映射

| 当前内容 | 目标 | 处理方式 |
| --- | --- | --- |
| `studio/` 的 Python 工程 | `app/studio/` | 保留 Python 包名、模块名和 CLI；更新仓库相对路径 |
| `studio/web/` 的应用入口、状态、请求、路由 | `app/studio/web/` | 留在 Studio，仅调整共享展示的引用 |
| `studio/web/src/features/` 中两端共用的展示及纯函数 | `app/shared/features/` | 移动实际使用的依赖闭包，并拆除应用专用依赖 |
| `studio/web/src/app/display.tsx`、共用格式化函数、素材与样式 | `app/shared/ui/`、`formatters.ts`、`assets/`、`styles/` | 只迁移共用部分，保留样式顺序及相对资源引用 |
| 展示组件引用的必要数据类型 | `app/shared/types/` | 不再用 Studio 请求实现或业务 Hook 的返回类型定义展示接口 |
| `plugins/` 的 Host、清单、Client 接入 | `app/plugins/` | 保留插件功能和分发协议，改用共享展示 |
| `plugins/src/client/shared-presenters.tsx` | 原文件随插件移动后改造 | 保留插件任务数据转换，接入共享展示 Provider |
| 两端的 Dialog、Tooltip、Switch 实现 | 各自应用内 | 保留现有实现，通过明确接口供共享展示使用 |
| 共用展示的测试 | 与迁移后的展示源码相邻 | 使用共享展示接口构造输入，不继续依赖 Studio store |
| 应用及协议集成测试 | 各自应用内 | 验证真实的状态接入、请求与生命周期 |

不得把纯数据定义和请求实现一起移动。`ApiError` 与 `errorMessage` 需要一起检查：若共享展示仍需要错误类型，只共享不带传输依赖的定义，保留原有错误语义和类型身份。

### 4.3 需要先拆除的隐式依赖

当前静态检查发现 21 个插件 Client 源文件含有 55 条直接指向 `studio/web` 的 import；构建和测试另有跨应用读取。

| 当前耦合 | 目标接口 |
| --- | --- |
| 展示组件直接调用 Studio `useSettings()` | 从共享展示 Provider 或已有 props 接收翻译函数；设置持久化仍由应用负责 |
| 展示组件读取 Studio `jobStore` | 接收任务快照；订阅、轮询和任务转换仍由应用负责 |
| `LocalCheckpoints` 内部调用 `readLoadContext()` | 由调用方提供加载配置或读取回调，不读取 Studio 的 API 地址与本地存储 |
| `ReturnType<typeof useModelCatalog>` 等类型耦合 | 明确声明展示实际需要的数据和操作接口，不用 `any` 绕过 |
| 导入 Studio 的 Dialog、Tooltip、Switch 后在构建时替换 | 共享代码声明所需控件接口，Studio 和插件分别提供现有实现 |
| 插件测试通过 Studio 的 `package.json` 解析 React | 从插件对应的测试依赖解析，验证真实宿主兼容版本 |

保留已有错误处理、操作确认、结果不明状态、任务恢复、取消语义及无障碍行为。不要在这次迁移中把两端不同的业务策略合并成一套默认行为。

## 5. 统一构建设计

### 5.1 依赖与锁文件

- 以现有 Studio 的 pnpm 工具链为基础，建立 `app/` workspace，仅包含应用层所需的 Node 工程。
- 根清单管理 TypeScript、Vite、测试工具和共用前端依赖；插件清单保留 Host 运行依赖、宿主兼容声明和分发信息。
- 合并当前 `studio/web/pnpm-lock.yaml` 与 `plugins/package-lock.json` 的应用层依赖意图，生成 `app/pnpm-lock.yaml`。迁移时核对已锁定版本，不借合并机会升级依赖。
- 移除旧应用层锁文件前，先通过新安装路径下的构建和测试。
- `dsh/package-lock.json`、Studio 的 `uv.lock` 和 Tauri 的 `Cargo.lock` 保持各自职责，不纳入 Node 统一锁文件。
- `app/shared` 没有独立依赖、版本和构建产物；由两个目标直接编译。

### 5.2 React 与插件产物

当前 Studio 使用 React 19，插件面向 React 18.2 的宿主。统一构建不能把这一差异抹掉：

- Studio 保留现有 React 运行时；插件产物继续使用宿主提供的 React。
- 插件清单可以保留 React 18.2 测试依赖，用于验证宿主兼容性；不得因此把另一份 React 打进 Client。
- 共享代码必须在两端均可运行；不得仅因根工程类型检查通过就引入 React 19 专属能力。
- 保留普通插件的单个惰性 `window.__ModuleLoader__.load` factory、外部 React 和不创建第二个 React 根的约束。
- 保留 DS Bot 目标原有的 ReactDOM portal 接入，不把它带入普通插件目标。
- 明确配置每个目标的 JSX 编译策略，不再依靠“文件位于 Studio 源码路径”来补 React import。

### 5.3 目标命令与产物

下表是待实现的命令接口，不是当前可执行命令。根 scripts 直接调用现有工具和调整后的目标配置，不增加通用构建框架。

| 从仓库根执行的目标命令 | 职责 | 主要输出 |
| --- | --- | --- |
| `pnpm --dir app install --frozen-lockfile` | 安装应用层 Node 依赖 | `app/` workspace 依赖 |
| `pnpm --dir app dev:studio` | Studio 前端开发服务 | 无分发产物 |
| `pnpm --dir app build:studio` | Studio 类型检查与 Vite 构建 | `app/studio/web/dist/` |
| `pnpm --dir app build:plugins` | 编译插件 Host 并打包普通 Client | `app/plugins/lib/host/`、`app/plugins/lib/client.js` |
| `pnpm --dir app build:dsbot` | 编译 Host 并组装 DS Bot 包 | `app/plugins/dist/dsbot/` |
| `pnpm --dir app build` | 依次构建 Studio 与普通插件 | 上述两个默认目标；DS Bot 显式构建 |
| `pnpm --dir app typecheck` | 编排共享代码与两端已有类型检查 | 不生成分发产物 |
| `pnpm --dir app test` | 编排共享、Studio 和插件单元测试 | 测试结果；插件产物测试前先构建 |

Studio 不因构建入口统一而必须构建或启动 DSH。插件目标也不应先构建 Studio。Python、原生构建和桌面发行包不隐式塞进默认 `app build`。

Tauri 与 `packaging/build_release.py` 调用 `app` 的 Studio 目标，不再执行 `studio/web` 私有的 npm/pnpm 构建。Tauri 的项目位置和产物相对位置保持在 `app/studio/web/` 内，避免为此次调整另外搬动桌面工程。

### 5.4 配置与版本

- 保留 Studio 目标的 Vite 配置、开发代理和 HTML 入口，显式设置工作目录、资源根和输出目录。
- 更新 TypeScript、Vite、Vitest、Playwright 对共享源码的解析和测试收集；别名只能指向共享源码，不能掩盖应用间引用。
- 插件 CSS 仍需要作用域隔离，输入改为 `app/shared`。删除服务/Provider 的路径替换，不删除 CSS 隔离及宿主打包所需处理。
- 保留产品版本、插件包版本、宿主版本的不同语义。产品版本源暂留 `app/studio/_version.py`；构建期读取版本是明确的元数据依赖，不是允许插件导入 Studio 实现。
- 更新 Studio Vite 对原 `./package.json` 的版本读取、Tauri `build.rs` 的版本路径与变更追踪。版本、release 标识、Git revision 的展示行为不得改变。

## 6. 全仓库影响清单

不能只修改前端 import。已发现以下需要随目录移动一起处理的调用方：

| 范围 | 已知位置与注意事项 |
| --- | --- |
| 服务启动与静态资源 | `studio/serve.py` 的 `WEB_DIR`、自动安装/构建逻辑；保留 `--web-root`、环境变量和冻结发行包的 `web-ui` 查找行为 |
| 仓库根定位 | `studio/server/runtime/native_build.py` 的 `parents[3]`；`studio/runtime/minicpmo45_realtime.py`、`studio/tools/pack_runtime_assets.py`、测试中的相对层级；逐处确认含义，不能统一机械加一 |
| CUDA/HIP 原生构建 | `csrc/backends/{cuda,hip}/CMakeLists.txt` 的 kernel root；原生源码中的 `studio/kernels/...` include；移动后更新为正确位置，不顺带重构 kernel 归属 |
| Python 扩展与 HIP staging | `studio/kernels/{cuda,hip}/_ext.py`、`studio/kernels/hip/_build.py`；源码打包、构建缓存和测试 fixture 的路径必须一致 |
| 桌面配置 | `studio/web/src-tauri/tauri*.conf.json`、`build.rs`、资源路径、启动工作目录、Cargo 检查入口；保留应用标识与用户配置位置 |
| 发行打包 | `packaging/build_release.py` 的 `STUDIO`、`PYTHON_PROJECT`、前端安装命令、Tauri 查找与静态资源收集；安装包内的既有布局不因仓库布局改变 |
| 插件分发 | `package.json` 的 exports/files、两种 patch、Host locale、`lib/host`、单文件 Client、DS Bot manifest；保留宿主协议和插件 ID |
| 跨应用集成测试 | `studio/tests/test_agent_*` 调用的插件 smoke 脚本；保留集成测试职责，更新脚本位置与 cwd |
| 测试及源码规则 | Python `conftest.py`、包布局测试、HIP portability 测试、原生 fixture、源码归档与 ignore 规则测试；Vite/Vitest/Playwright 的 include、setup、webServer 命令 |
| 文档与资源 | 根 README 中的图标路径、运行/安装命令、`CONTRIBUTING.md`、`docs/`、各应用 README、`.gitignore`；保留迁移素材的版权声明 |

保留 Python 发布名 `tyloinfer`、`tyloinfer-server` CLI、现有 Python import 名称、插件包名及导出。源码 checkout 的路径命令需要更新，不默认增加旧目录软链接或永久兼容转发文件。

安装后的用户数据位置、环境变量、浏览器存储键、桌面应用标识和 DSH profile 不属于源码目录，不得做全局字符串替换。

## 7. 实施顺序

每阶段以可运行状态为检查点。阶段间允许短暂保留旧展示引用，但不能把中间态标为迁移完成。

### 阶段 A：记录迁移基线

1. 记录当前 Studio、普通插件和 DS Bot 的安装、构建、类型检查、单元测试结果。
2. 列出两端实际共用的展示依赖闭包及样式顺序，记录各节点是否依赖应用状态、传输、平台或持久化。
3. 记录 Node/React/DSH 版本、已有失败和缺少的测试环境，区分迁移问题与既有问题。
4. 确认七个插件页面、Studio 对应页面及桌面启动的最小回归场景。

检查点：有可对照的命令和结果；未运行的项目标为未验证。

### 阶段 B：移动完整应用，修复路径

1. 将 `studio/` 移到 `app/studio/`，将 `plugins/` 移到 `app/plugins/`，保留各自内部结构。
2. 暂时保留现有构建入口，优先更新第 6 节的仓库根、原生 include、打包和测试路径。
3. 验证服务与原生构建的根定位、前端构建、插件产物和发布脚本的路径解析。
4. 同步当前使用说明中的目录；不复制旧树、不移动用户数据。

检查点：目录移动本身不改变产品行为；不能只以 JS 构建通过作为此阶段成功依据。

### 阶段 C：建立统一构建入口

1. 将 Studio 前端清单和通用工具配置提升到 `app/`，纳入插件清单，建立统一锁文件。
2. 实现第 5 节目标命令；分别设置 Studio、普通插件和 DS Bot 的入口与输出。
3. 调整 Studio 开发服务、自动构建、Tauri、Playwright 和发行脚本，使其使用根构建入口。
4. 修复插件测试的 React 解析，验证 React 18/19 的目标隔离。
5. 在新安装路径验证后，删除旧的前端清单、重复工具依赖和旧应用层 Node 锁文件。

检查点：干净 checkout 在 `app/` 安装一次 Node 依赖即可分别构建两个前端；不依赖旧目录中的 `node_modules` 或构建产物。

### 阶段 D：迁移共用展示并显式接入

1. 先移动素材、样式、纯函数和不依赖应用环境的展示组件；同步两端调用及对应测试。
2. 按模型、运行时、连接配置、分析等实际依赖逐组迁移，保留应用专用控制逻辑。
3. 拆除设置、任务 store、加载默认值、请求实现类型等反向依赖。
4. 为确有差异的基础控件提供明确接口，保留两端现有实现及焦点、键盘、关闭行为。
5. 改造已有 `shared-presenters.tsx`，保留任务归一化等插件逻辑，不新增通用宿主注册框架。

检查点：每组共用代码均通过展示接口测试，并在两端至少覆盖一个实际消费场景。

### 阶段 E：删除旧替换规则，固定依赖方向

1. 删除插件构建对 `SettingsProvider`、`jobStore`、`loadContextDefaults` 和 Studio UI 路径的替换。
2. 移除旧 Studio 样式读取、旧文件转发和重复实现；保留必要的 CSS 作用域、JSX 及宿主输出处理。
3. 添加一个聚焦的前端依赖方向检查，纳入 `app` 检查命令：禁止两应用前端互相导入，禁止共享源码反向导入应用。
4. 检查应基于解析后的目标路径覆盖相对路径、别名和类型导入；不能只搜索旧的 `studio/web` 字符串。构建元数据读取与集成测试单独界定范围。
5. 更新架构说明及安装、构建、测试、发布文档，明确旧命令的替代入口。

检查点：不再需要借助构建替换才能把 Studio 展示运行在插件中。

### 阶段 F：最终验证

1. 执行下节验证矩阵，记录平台、依赖版本、命令和结果。
2. 从清洁安装环境构建三个前端目标，确认未读取旧路径或另一应用的私有源码。
3. 验证桌面安装包及实际宿主接入；仅完成静态检查或单元测试时，不宣称发行链已通过。
4. 清理旧路径引用、失效脚本和重复依赖，保留本计划作为迁移决策记录。

## 8. 验证矩阵与完成标准

| 层次 | 必须验证的内容 |
| --- | --- |
| 依赖与类型 | 新锁文件可冻结安装；共享源码和 Studio 通过类型检查；插件 Host、逻辑模块及受影响的展示接入契约通过检查；无应用间私有源码导入 |
| 共用展示 | 迁移组件的原有测试；翻译、任务变化、加载默认值、错误显示、Dialog/Tooltip/Switch 适配；样式顺序与素材路径 |
| Studio Web | Vite 构建与预览；页面导航/刷新；模型、运行时、下载、分析、日志等受影响页面；现有 Playwright 场景 |
| 插件 | Host 与 Client 单元测试；单个惰性 factory、React 外部化；七个推理页面、原生会话与设置保留、卸载清理；DS Bot 目标的专用产物测试 |
| Python 与协议 | 包布局、服务静态资源、根定位、运行时路径及 `test_agent_*` 桥接测试；HTTP/RPC 契约与结果不明处理不变 |
| 原生构建 | CUDA/HIP 的 CMake 配置、kernel include、Python 扩展及 HIP staging；Metal 相关路径与资源检查；修改到的构建链需要实际构建证据 |
| 桌面与发行 | Tauri 检查和前端构建调用；服务/sidecar 启动；macOS、Windows 打包的资源收集；安装包内静态资源与版本一致 |
| 用户状态 | 不改变桌面配置、profile、凭据、模型与缓存位置；浏览器恢复状态和插件卸载行为不变 |

当前插件 `.tsx` 页面未全部纳入严格类型检查，这是已知基线。迁移必须检查共享源码和受影响的适配接口，但不得把 Vite 转译成功描述为全 Client 类型检查成功，也不应为隐藏错误全局放宽检查。无关页面的全量类型补齐不属于本次迁移。

没有 CUDA、HIP、Metal、Windows、macOS 或实际宿主环境时，对应项必须记录为“未验证”，不能用另一个平台的通过结果替代。UI fixtures 也不代表真实权重推理或下载执行已通过。

迁移完成条件：

- [ ] 目录符合第 3 节，两个应用的专用实现归属未被打散。
- [ ] `app/` 控制两个前端的依赖、检查和构建；共享源码不是独立包。
- [ ] Studio、普通插件、DS Bot 各自可独立选择构建，产物与分发协议正确。
- [ ] 两端只共享明确的展示接口，不再引用对方私有前端源码或依赖路径替换。
- [ ] Python、原生 include、桌面、发布、测试和文档路径均已更新。
- [ ] 已完成的验证有实际结果，无法执行的项目和发布风险明确列出。
- [ ] 用户数据、公开协议、包身份、CLI 和原有错误处理保持不变。

## 9. 提交与回退

按上述阶段形成可审查的改动，尽量把目录移动与展示接口调整分开。每个阶段应包含其必要的调用方、配置和测试修复，不提交已知断裂的半次移动。

若需要回退，回退对应源码、锁文件及构建配置，并按回退后的工具链重建。不要用旧构建产物掩盖新代码失败，不删除用户模型、缓存、凭据或 profile，也不通过保留第二份长期源码来维持兼容。

本计划描述后续迁移的实施范围，不代表已经开始迁移。本次没有执行目录移动、依赖安装、构建或运行验证。
