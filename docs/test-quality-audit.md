# 测试质量审计 · `2a91af16`

**结论：测试有价值，但绿灯暂时不能作为推理正确性的充分发布依据。** 最值得修的不是测试数量，而是数值比较会放过 NaN、GPU 用例在跳过条件后还引用已不存在的模块，以及 Python 的常规入口没有稳定基线。不建议重写整套测试。

## 本机实测

Linux x86-64、RTX 4090、Python 3.12（`studio/.venv`，无 torch）、Node 24。使用已有依赖；构建和变异实验不修改跟踪源码。

| 层 / 口径 | 结果 | 解读 |
|---|---|---|
| Python：143 个 `test_*.py`，单元、ASGI/存储/进程集成和可选硬件测试混合 | 第 1 轮 1794 过 / 6 败 / 84 跳，117.76 s；第 2、3 轮均为 1795 / 5 / 84，分别 112.23、114.64 s。 | 5 个参数例同一固定故障；首轮还观察到任务取消状态失败。模块级 skip 不等价于只漏跑一个函数。 |
| Web Vitest：`src` 83 文件、`tests` 19 文件 | 946/946 连过 3 轮；27.98 / 25.90 / 25.77 s | 有真正的组件行为测试；也有明确标注为非行为的源码检查。 |
| 插件 `node:test`：Host 8、Client 7 个 `.test.ts` | 每轮 652 过、1 跳、0 败；共 3 轮，0.28–0.31 s/轮 | 使用新构建产物；跳过的是 DSBot 打包模式。另行执行 8 个旧 `.test.mjs`，8/8 导入失败；不计入 TS 套件。 |
| 后端无关 CTest：Release，关闭 CUDA/HIP 后端 | 注册的 57/57 连过 3 轮；21.97 / 22.02 / 22.16 s | 不是 GPU 全量。仓库另有 198 个 `*_test.{cpp,cu,hip}` 源文件（通用 52、tokenizer 1、CUDA 65、HIP 14、Metal 66）；源文件不是 CTest 用例数。 |
| 原生 CUDA：单独编译现有 GGUF row residual 测试，真实设备运行 | 基线连续 3 次通过；单次约 0.15 s；两个定向变异杀死 1 个 | 见下节。这是单目标的 1/2，不是全仓或整个 CUDA 后端的分数。 |
| Playwright：36 个 spec，桌面/移动项目共收集 196 例 | 构建和预览成功；前 2 例在启动浏览器时失败后停，194 未跑 | 需要 Chromium headless shell 1255，本机缓存为 1243。没有执行页面断言，不能判为 UI 回归或通过。 |
| Tauri：2 个 Rust 文件中 26 处 `#[test]`，含条件编译和显式 ignore | `cargo test --offline` 编译依赖时缺 `glib-2.0.pc`，未执行 | 不把系统开发库缺失计为代码缺陷。Metal/HIP 真机、完整原生 CUDA、真实模型权重也未验证。 |

复跑方式：根目录 `HF_HUB_OFFLINE=1 app/studio/.venv/bin/python -m pytest app/studio/tests -q --durations=10`；`app/` 工作区下 `pnpm test:studio`、`pnpm test:plugins`、`pnpm test:e2e:studio`，插件测试前需 `pnpm build:plugins`；下列表格中的路径是迁移前的 `studio/`、`plugins/` 路径。原生用 `cmake -S csrc -B <tmp> -G Ninja -DCMAKE_BUILD_TYPE=Release -DTYLOINFER_BUILD_CUDA_RUNTIME=OFF -DTYLOINFER_BUILD_HIP_RUNTIME=OFF`、构建、`ctest --test-dir <tmp> --output-on-failure`。首轮 pytest 最慢单例是 jobs 事件尾页测试（9.62 s）；写入 1005 条再验证翻页边界，不建议为了变快删掉该边界。

## 优先处理

1. **高 · 真实假绿。** `csrc/backends/cuda/tests/gguf_row_residual_test.cpp:20–23` 用 `abs(actual-expected) > tol` 判错。有限参考值遇到 NaN 时条件为假。在临时副本中只改 `csrc/backends/cuda/kernels/gguf_row_residual.cu` 的 `y[slot*rows+row]+=sum`，重编 kernel，以同一份未改测试跑相同 fixture：给结果加 `1.f`，退出 **1**；改成写 quiet NaN，仍打印 **PASS**、退出 **0**。这些 NaN 出现在有残差的行，不是未执行的分支。先修这一处，比较时拒绝非有限值，并在失败时打印行/下标、实际值、参考值。再排查下表 CUDA、Metal 中相同写法；`std::max(max_error, abs(delta))` 也可能吞 NaN，检查最终的 `max_error` 不够。**不是所有原生比较都有此问题**，HIP attention MMA 就先检查了每个差值。
2. **高 · 平台条件掩盖失效的测试。** 三个 CUDA Python 模块在 `importorskip('torch')` / CUDA 检查之后，才导入 `formats.nepq`、`formats.nint8_zero` 等。当前树中 7 个不同的仓内模块/测试辅助模块找不到（10 处 `from`，本机 `find_spec` 也返回 `None`）；3 文件共 21 个 `test_` 函数，尚未展开参数。当前环境显示 skip，满足硬件条件则会在收集时失败。先确认这些格式的**现行受支持入口**，迁移断言，或明确退役哪些重复用例，不要再套一层 `importorskip`。独立问题：`test_kernels_cuda.py:13` 对所有系统强制要求 Windows `cl`，Linux 即使有 torch、CUDA 和 C++ 编译器也会跳过整个文件（44 个测试函数）。生产加载器 `studio/kernels/cuda/_ext.py:65` 只在 Windows 处理 MSVC；测试应使用相同的平台条件。
3. **高 · 5 个常规测试在目标断言前就失败。** `studio/tests/test_cuda_kv_tier_budget.py:131–136` 把 `SimpleNamespace` 当 `RuntimePool`，传入 `artifact=None`；现在 `studio/server/runtime/runtime_pool.py:4957` 先读 `self.backend` 和 CUDA artifact 架构。5 个 SSD 预分配组合都报 `AttributeError: ... backend`，没有一个执行 `assert checks == [expected_ssd]`。用真正的 pool 和最小非 Hy4 CUDA artifact，固定预算/容量探针，保留这 5 个期望字节数。不需要 GPU，也不宜靠跳过测试恢复绿灯。
4. **高 · 任务取消存在时序信号。** `studio/tests/test_server_jobs.py:170–177` 等第二个任务开始，然后 `close()`，再要求第一个已持久化为 `cancelled`。同一代码首轮读到 `cancelling`，后两轮通过。`studio/server/services/jobs.py:217–229, 240–249` 的关闭和显式取消都会取消任务，终态写入另见 `_run` 的异常分支。这个失败不能直接归因为“测试太快”：它可能是真实的关闭竞争。用事件把「已释放执行槽、尚未写入终态」固定下来；分开验证槽复用和 shutdown 的持久化契约。不要加 sleep、重试到绿，或把 `cancelling` 也算完成。
5. **中 · 迁移后的插件有两套入口。** 8 个 Host `.test.mjs` 都从 `../core-bridge.js` 导入，构建后该源码路径依然不存在；同名 `.ts` 用的是 `lib/host/`。另有 5 个 Python/Node 集成测试只检查 `node` 是否存在，却要执行依赖 `plugins/lib` 的 smoke。对 `git archive HEAD plugins` 的未构建副本执行 `core-api-smoke.ts` 可复现 `ERR_MODULE_NOT_FOUND`，而本机有构建产物时可过。保留真正的跨进程集成测试，统一以构建后的 TS 入口运行；在 Python 集成入口前显式构建并检查 Node 版本。确认无调用方后删除旧 `.mjs` 副本，而不是再维护另一套路径。

### 可读性（Readability）

`test_server_models.py` 4140 行，其中单个流程覆盖加载、重复加载拒绝和卸载。`test_server_service.py` 的 `FakeBackend` 还被 18 个测试文件直接导入。先把跨文件依赖的 fixture 移到测试支持模块；只按可独立失败的行为拆，保留一条完整生命周期，不按行数机械拆文件。

### 可靠性（Reliability）

已观测到同一取消用例红绿变化；未在 Vitest、Node TS 或 CPU CTest 的三轮中观察到失败。轮询终态、模拟耗时的子进程和无条件等待不是同一件事，不应扫到 `sleep` 就全删。E2E 浏览器及桌面系统库问题单独列示，不算 flaky。

### 诊断价值（Diagnostic）

数值断言的首要问题是根本不失败，其次才是报错模糊。可以仿照 `channel_fp8_test.cpp:24–25` 输出元素位置及实测/期望。保留 `test_server_jobs.py` 最终状态的严格断言，它正是发现竞争的信号。

### 设计（Design）

`studioContracts.test.ts` 和 `voiceContracts.test.ts` 共 30 个 `it`，两者自己也声明是“非行为测试”。例如声称验证跨 audio worklet 块重采样相位的用例，只确认源码设了 16 kHz 并创建 `AudioContext`；另一例靠全 App 出现确认调用至少 6 次判断破坏性操作有确认。保留确实是架构约束的依赖方向检查；行为改用输入输出或用户操作断言，优先处理取消、删除、语音缓冲边界。不把所有读源码或检查 CSS 的测试一概计为缺陷。

### AI 生成（AI-generated）

来源不明，不根据命名或代码风格推断作者。可验证的是失效导入和会存活的变异。没有全仓变异分数，也不从定向的 1/2 推算总体检错率。

### 覆盖（Coverage）

有真实的负例、独立数值参考和并发测试：例如 QSA 能量分配器穷举独立目标函数；磁盘预算验证竞争、越限后状态和释放；插件桥验证网络请求次数及错误语义。浏览器测试中的 `mockStudioServer` 会拦截 `/api/v1/**`，适合测 UI，不等于整条浏览器到 Python 服务的链路已验过。真实权重 opt-in 测试有意义，但开发机 skip 不应算发布平台上的验证。

## 可执行的检查与验收

以下是本次实际做过的静态检查的最小门禁版本：在根目录执行，不装 torch/MLX、不导入 GPU 模块。当 `formats`、`kernels` 或 `tests` 下的绝对 `from` 指向不存在的本地模块时非零退出。当前应报上面的 3 个文件，修完应为 0。它不验证导出符号，也不能代替 GPU 收集测试。

```sh
python3 - <<'PY'
import ast
from pathlib import Path
root = Path('studio')
missing = []
for f in sorted((root / 'tests').rglob('test_*.py')):
    for n in ast.walk(ast.parse(f.read_text(encoding='utf-8'))):
        if not isinstance(n, ast.ImportFrom) or n.level or not n.module:
            continue
        parts = n.module.split('.')
        if parts[0] in {'formats', 'kernels', 'tests'}:
            p = root.joinpath(*parts)
            if not (p.is_dir() or p.with_suffix('.py').is_file()):
                missing.append(f'{f}:{n.lineno}: {n.module}')
print('\n'.join(missing))
raise SystemExit(bool(missing))
PY
```

数值验收比设全仓百分比有用：对上述 CUDA kernel 的同一行分别注入有限偏差与 NaN，要求原测试两次都以失败退出，原实现保持绿。当前基线已经红，且抽样非随机；本轮不宜设置全仓变异率阈值或全仓禁用源码断言的 lint。此次仅审计，未更改既有测试、生产代码或 CI。

## 文件抽查索引

表中抽查 60 个文件，约占上述命名入口、旧 `.test.mjs` 及 Rust 内联测试文件的 12%；这是定向抽样，不是随机样本。优先选最近迁移的插件、体积大的服务/算子测试和运行失败处；用正常的独立参考/行为测试作对照。每行是一个文件的**抽查部位**，不是全文认证或缺陷占比。高＝可假绿、固定红灯或关键时序；中＝覆盖/维护边界；低＝该处无优先修复项。S＝约半天以内，M＝约 1–2 天，按同类合并估计，不逐行相加；「—」不用改。

| 文件:行 | 严重度 / 工作量 | 证据或处置 |
|---|---|---|
| `csrc/backends/cuda/tests/gguf_row_residual_test.cpp:22` | 高 / S | NaN 变异实测存活；拒绝非有限值并报告行号。 |
| `csrc/backends/cuda/tests/gguf_quant_test.cpp:53–68` | 高 / S | decoder 检 finite，embedding 对同一类数据却只用误差大于阈值判错；补齐后者及图重放。 |
| `csrc/backends/cuda/tests/qsa_tq_prefill_test.cpp:153–156` | 高 / S | 单独比较 history mass 的近似断言容许 NaN；不要只依赖其他张量的 exact 检查。 |
| `csrc/backends/cuda/tests/qsa_ram_runtime_test.cpp:169–172` | 高 / S | 分块 attention 和原路径比较时 NaN 不会触发该异常。 |
| `csrc/backends/cuda/tests/gr_bf16_test.cpp:44` | 高 / S | FP64 点积是有价值的参考；修正 prefill 标量比较对 NaN 的处理。 |
| `csrc/backends/cuda/tests/gguf_runtime_test.cpp:2613–2621` | 中 / S | 最大误差归约后才检查 finite；逐项检查。随后还有 `check_moe`，不能说整个用例一定假绿。 |
| `csrc/backends/cuda/tests/attention_mma_prefill_test.cpp:26–34` | 低 / — | 先验证 shape 和逐元素有限性，再验证误差；另有独立 FP64 oracle。 |
| `csrc/backends/cuda/tests/channel_fp8_test.cpp:19–25` | 低 / — | 失败指出下标和数值，对有限输入的输出显式判定 finite。 |
| `csrc/backends/metal/tests/mlx_moe_ops_test.cpp:34–44` | 高 / S | 标量 `require_close` 遇 NaN 不抛，须在公共比较处拒绝。 |
| `csrc/backends/metal/tests/mlx_transformer_test.cpp:25,286–287` | 高 / S | 数组路径有保护，直接比较 normalized 标量的调用没有；修标量 helper。 |
| `csrc/backends/metal/tests/mlx_linear_attention_test.cpp:27,1156` | 高 / S | 卷积标量走无 finite 检查的 helper，与本文件向量检查不一致。 |
| `csrc/backends/metal/tests/mlx_nint8_zero_test.cpp:146` | 高 / S | 解码结果比较可放行 NaN；保留另有的损坏 blob 负例。 |
| `csrc/backends/metal/tests/mlx_moe_test.cpp:6528` | 高 / S | 拼接专家的输出维度正确但数值为 NaN 时，此处逐元素比较不会失败。 |
| `csrc/backends/metal/tests/mlx_ssd_expert_arena_test.cpp:49–53` | 高 / S | 从 0 开始的 `max` 可能吞掉 NaN，最终误差仍是 0；逐元素检查再归约。 |
| `csrc/backends/metal/tests/mlx_tq_test.cpp:1657–1672` | 低 / — | CPU 点积前已有全张量 allclose；不把后面的单个谓词当成整个用例的假绿证据。 |
| `csrc/backends/hip/tests/attention_mma_test.cu:71–81` | 低 / — | 逐项拒绝非有限差值，再汇总相对/绝对误差。 |
| `csrc/backends/hip/tests/decode_hotspots_test.cpp:152–159` | 低 / — | cached prefill 对非有限值和与 FP64 的误差分别判错。 |
| `csrc/tests/speculative_sequence_test.cpp:794–796` | 中 / S | 接受率近似比较不拒绝 NaN；应报告实际/期望，保留后续独立的策略状态断言。 |
| `csrc/tests/qsa_uniform_fit_test.cpp:60–66` | 低 / — | 在小问题上穷举，独立检验代价最优；不是重复调用生产算法生成期望。 |
| `csrc/tests/qsa_precision_policy_test.cpp:25–32` | 低 / — | 预算不足、保护窗口、冷热块选择有具体期望。 |
| `studio/tests/test_cuda_kv_tier_budget.py:131–136` | 高 / S | 单个过期的假 pool 使 5 个预分配组合没有测到 SSD 检查。 |
| `studio/tests/test_server_jobs.py:170–177` | 高 / M | 槽释放不等于终态落库；保留断言，固定交错顺序再判定关闭语义。 |
| `studio/tests/test_kernels_cuda.py:10–14` | 高 / S | 对非 Windows 错用 MSVC 门槛，整模块 44 个测试函数可能被跳过。 |
| `studio/tests/test_format_completion_cuda.py:11,20` | 高 / M | 两个 NINT8-0 模块不存在，连同同文件的 MX 用例也收集不到。 |
| `studio/tests/test_nepq_kernels_cuda.py:12–31` | 高 / M | GPU 条件满足后将导入缺失的实现及 `_tensor` 来源；对齐现行 NEPQ 接口。 |
| `studio/tests/test_nepq_a_kernels_cuda.py:10–26` | 高 / M | 除实现外 `_a_tensor` 的来源也不存在；随同格式迁移或退役。 |
| `studio/tests/test_agent_model_bridge.py:41–65` | 中 / S | 使用真实 API 和受控 runtime 有价值；Node 存在不代表插件已构建。 |
| `studio/tests/test_agent_maintenance_bridge.py:63,127` | 中 / S | 相同集成前置缺口；保持真实的范围与文件保存断言。 |
| `studio/tests/test_agent_model_actions_bridge.py:82,137` | 中 / S | 相同构建前置；保留对可复用文件内容的精确检查。 |
| `studio/tests/test_agent_model_details_bridge.py:29,59` | 中 / S | 相同前置；禁止解码 tensor payload 是值得保留的反向断言。 |
| `studio/tests/test_agent_page_bridge.py:66,108` | 中 / S | 相同前置；继续检查收到哪些 hub 调用，而不是只看进程退出码。 |
| `studio/tests/test_server_models.py:1727–1790` | 中 / M | 一例串接列表、加载、重复加载、卸载；重复轮询可归并，保留端到端主流程。 |
| `studio/tests/test_server_service.py:37` | 中 / S | `FakeBackend` 被其他测试模块直接导入；移到共享测试支持文件。 |
| `studio/tests/test_runtime_log_output.py:16,80–99` | 低 / — | 真实 pipe 和超长行检查值得保留；未来共享 `_model` 等 fixture 不必导入巨型测试模块。 |
| `studio/tests/test_native_stdio_e2e.py:16–20,77–88` | 低 / — | 需要权重是合理前置；检查非流式与流式结果一致，不计入无权重机器的通过数。 |
| `studio/tests/test_d1_native.py:58–73` | 低 / — | 明确只测进程协议；包含非法参数、非有限超时负例。 |
| `studio/tests/test_d1_native_real.py:17–35` | 低 / — | 显式 opt-in，检查具体答案和接近阈值的分数，保留为发布机检查。 |
| `studio/tests/test_d1_official_parity.py:24–39,99–110` | 低 / — | opt-in 引用外部金标准并比对 token 序列；与硬编码假后端的能力不同。 |
| `studio/tests/test_disk_budget.py:38–58` | 低 / — | 竞争预留、超限失败后保留旧值和释放都用实际状态验证。 |
| `studio/tests/test_hip_portability.py:21–45` | 低 / — | 验证并发暂存、内容寻址、源文件不被更改；读文件是测试行为本身。 |
| `studio/tests/test_source_archive_rules.py:8–31` | 低 / — | 在独立 git 仓库执行 `check-ignore`；规则测试不是简单检查配置文本。 |
| `studio/tests/test_server_hub.py:88–105` | 低 / — | 分页/过滤使用 MockTransport，能断言请求参数和结果数量。 |
| `studio/tests/test_tq_cuda.py:40–55` | 中 / S | 不补回全局 TF32 设置；局部保存/恢复，避免干扰后续 GPU 用例。 |
| `studio/tests/test_formats/test_mx.py:11–31` | 中 / S | 名称说 roundtrip/streamed rows，本例只检查类型和 shape；补值/指定行的比对，保留后面的 NaN 编码负例。 |
| `plugins/src/host/tests/core-bridge.test.ts:40–45` | 低 / — | 拒绝非 owner 必须在发网络请求之前发生；同时断言错误语义和零 I/O。 |
| `plugins/src/host/tests/core-bridge.test.mjs:3` | 中 / S | 旧入口实际导入失败；其余七个同类 `.mjs` 单元入口也已用 runner 验证。 |
| `plugins/src/client/tests/plugin.test.ts:6–9,43–45` | 中 / S | 从 Studio 工作区取 React，而插件依赖里也有 React 且版本不同；明确要模拟的宿主版本和构建前置。 |
| `plugins/src/client/tests/page-api.test.ts:30–35` | 低 / — | 指定一个资源失败，验证其错误不被空值伪装，其他资源仍保留。 |
| `studio/web/tests/studioContracts.test.ts:104–106` | 中 / M | 数全局确认调用次数，保护不到某一次具体的删除；测取消后未发写请求。 |
| `studio/web/tests/voiceContracts.test.ts:43–48` | 中 / M | 固定采样率不是跨块相位正确的充分条件；用跨块波形/长度作期望。 |
| `studio/web/tests/branding.test.ts:29–34` | 中 / S | 测 Rust 绘制坐标的源码拼写；若要保视觉结果，测产出图标而非绘制代码。 |
| `studio/web/tests/architecture.test.ts:15–31` | 低 / — | 依赖方向适合作静态守卫；行数或 JSX 原文不应代替路由行为。 |
| `studio/web/tests/chatToolbarLayout.test.ts:10–34` | 中 / S | 搭人造 DOM 查 CSS，真实工具栏结构变了仍可通过；若要保证组件布局，渲染实际组件。 |
| `studio/web/tests/runtimeContracts.test.tsx:80–96` | 低 / — | 真正 rerender hook 后读 sampling，不把文件名含 Contracts 的测试一概归为源码检查。 |
| `studio/web/src/features/models/useModelCatalog.test.tsx:80–97` | 低 / — | 同时检查不可加载的组件没有产生请求或改变选中项，以及可见的提示。 |
| `studio/web/e2e/analysis-numerals.spec.ts:18–39` | 低 / — | 原意是视觉契约，浏览器实际 computed style 有价值；本次未运行页面。 |
| `studio/web/e2e/studio.spec.ts:29–57` | 低 / — | 受控 API 遥测驱动真实 meter 更新；适合 UI 回归，不是后端集成。 |
| `studio/web/e2e/asset-recovery.spec.ts:22–39` | 低 / — | 注入持久 chunk 失败，验证仅一次自动重载和手动恢复；保留负例。 |
| `studio/web/src-tauri/src/main.rs:877–891` | 低 / — | 用两个真实子进程区分 owned 与 external；本机未执行。 |
| `studio/web/src-tauri/src/updater.rs:1221–1243` | 低 / — | Windows 条件用例验证更新失败不破坏原文件；Linux 结果不能替它背书。 |
