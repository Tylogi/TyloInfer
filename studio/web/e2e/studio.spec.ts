
import { expect, test, type Page } from '@playwright/test';
import { mockStudioServer, officialCatalog } from './mockServer';
import { openModelSettings } from './modelSettings';

for (const language of ['zh', 'en'] as const) {
  test(`tool navigation stays visible without clipping (${language})`, async ({ page }, testInfo) => {
    await mockStudioServer(page, { language: language === 'zh' ? 'zh-CN' : 'en' });
    await page.goto('/inference');
    await expect(page.locator('main h1')).toBeVisible();
    if (testInfo.project.name === 'mobile') {
      await page.getByRole('button', { name: language === 'zh' ? '打开侧栏' : 'Open sidebar' }).click();
    }
    const sidebar = page.locator('#studio-sidebar');
    await expect(sidebar).toBeVisible();
    const engine = sidebar.getByRole('button', { name: language === 'zh' ? '推理引擎' : 'Inference engine', exact: true });
    await expect(engine).toHaveAttribute('aria-expanded', 'true');
    await expect(sidebar.getByRole('button', { name: language === 'zh' ? '新建会话' : 'New session', exact: true })).toHaveCount(0);
    await expect(sidebar.getByRole('button', { name: language === 'zh' ? '分析' : 'Analysis', exact: true })).toBeVisible();
    for (const button of await sidebar.locator('button').all()) {
      expect(await button.evaluate(element => element.scrollWidth <= element.clientWidth)).toBe(true);
    }
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await sidebar.screenshot({ path: testInfo.outputPath(`compact-sidebar-${language}.png`) });
  });
}

test('resource telemetry and service profiles have separate responsive homes', async ({ page }, testInfo) => {
  await mockStudioServer(page);
  let samples = 0;
  await page.route('**/api/v1/runtime/resources', (route) => {
    samples++;
    return route.fulfill({ json: { sampled_at: 100 + samples * 2, interval_seconds: 2,
      cpu_name: 'Apple M5 Max', cpu_cores: 18, cpu_utilization_percent: samples > 1 ? 43 : 36,
      gpus: [{ name: 'Apple M5 Max', core_count: 40, utilization_percent: 24 }],
      memory_bandwidth_bytes_per_second: null, memory_bandwidth_limit_bytes_per_second: 614e9,
      memory_bandwidth_utilization_percent: null, disks: [{ name: 'disk0', read_bytes_per_second: 2 ** 30,
        write_bytes_per_second: 2 ** 20, busy_percent: null, bandwidth_utilization_percent: null }],
      weights: [{ instance_id: 'test-model', model: 'Qwen3.8-Flash-Next', expert_read_bytes_per_second: 0,
        ple_read_bytes_per_second: 2 ** 20, engram_read_bytes_per_second: null }] } });
  });
  await page.goto('/resources');
  await expect(page.getByText('Resource monitoring', { exact: true })).toBeVisible();
  await expect(page.getByRole('meter', { name: 'CPU: Apple M5 Max 18C' })).toHaveAttribute('aria-valuenow', '36');
  await expect(page.getByRole('meter', { name: 'GPU 0: Apple M5 Max 40C' })).toBeVisible();
  await expect(page.getByRole('meter', { name: 'CPU: Apple M5 Max 18C' })).toHaveAttribute('aria-valuenow', '43');
  await expect(page.getByText('Runtime profiles', { exact: true })).toHaveCount(0);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.locator('.resource-monitor-panel').screenshot({ path: testInfo.outputPath('resource-monitor.png') });
  await page.goto('/runtime');
  await expect(page.getByRole('heading', { name: 'Service', exact: true })).toBeVisible();
  await expect(page.getByText('Runtime profiles', { exact: true })).toHaveCount(0);
  await expect(page.getByText('Resource monitoring', { exact: true })).toHaveCount(0);
  await page.goto('/resources');
  await expect(page.getByRole('meter', { name: 'CPU: Apple M5 Max 18C' })).toHaveAttribute('aria-valuenow', '43');
});

declare global {
  interface Window {
    revokedPreviews: string[];
  }
}


async function navigateClient(page: Page, path: string) {
  await page.evaluate((next) => {
    window.history.pushState(null, '', next);
    window.dispatchEvent(new PopStateEvent('popstate'));
  }, path);
}

test('服务页不再包含旧聊天设置和可执行文件入口', async ({ page }, testInfo) => {
  await mockStudioServer(page);
  await page.route('**/api/v1/runtime/models', (route) => route.fulfill({ json: { data: [{ id: 'Test Model' }, { id: 'Another Model' }] } }));
  await page.goto('/runtime');
  await expect(page.getByText('Runtime executable', { exact: true })).toHaveCount(0);
  await expect(page.getByRole('textbox', { name: 'System prompt' })).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Alias mapping' })).toBeEnabled();
  await page.getByRole('tab', { name: 'Cache', exact: true }).click();
  await expect(page.getByText('/data/tyloinfer/prefix-cache', { exact: true })).toBeVisible();

});

test('每个模型独立配置 ctx，保存整套设置且不改变另一个模型', async ({ page }, testInfo) => {
  await mockStudioServer(page);
  const instances = [
    { id: 'flash', model: 'Qwen3.8-Flash-S4-L', state: 'ready', devices: ['metal'], active_sessions: 0,
      queued_requests: 0, context_size: 32768, context_capacity: 131072,
      memory: { resident_weight_bytes: 78.6 * 2 ** 30 } },
    { id: 'dense', model: 'Qwen3.8-27B-S4-M', state: 'ready', devices: ['metal'], active_sessions: 0,
      queued_requests: 0, context_size: 16384, context_capacity: 262144,
      memory: { resident_weight_bytes: 17.7 * 2 ** 30 } },
  ];
  await page.route('**/api/v1/runtime/instances', (route) => route.fulfill({ json: { data: instances } }));
  await page.route('**/api/v1/runtime/models', (route) => route.fulfill({ json: { data: instances.map((item) => ({ id: item.model })) } }));
  await page.route(/\/api\/v1\/runtime\/status(?:\?.*)?$/, (route) => {
    const id = new URL(route.request().url()).searchParams.get('instance_id');
    const item = instances.find((entry) => entry.id === id) ?? instances[0];
    return route.fulfill({ json: { instance_id: item.id, model: item.model,
      runtime_state: 'ready', max_context: item.context_size, context_capacity: item.context_capacity } });
  });
  let complete!: () => void;
  const pending = new Promise<void>((resolve) => { complete = resolve; });
  const reloads: { instance_id: string; context_size: number }[] = [];
  const policy = { max_context_size: null, model_overrides: {} as Record<string, number>, fallback_context_size: 32768 };
  await page.route('**/api/v1/runtime/context-policy', route => route.fulfill({ json: policy }));
  await page.route('**/api/v1/runtime/context', async (route) => {
    const body = route.request().postDataJSON();
    reloads.push(body);
    await pending;
    const item = instances.find((entry) => entry.id === body.instance_id)!;
    item.context_size = body.context_size;
    policy.model_overrides[item.model] = body.context_size;
    await route.fulfill({ json: { id: 'context-save-job', kind: 'runtime.context.configure', status: 'succeeded',
      payload: body, progress: 1, result: { model: item.model, instance_id: item.id, max_context: item.context_size,
        context_override: body.context_size }, cancel_requested: false,
      created_at: '2026-10-09T00:00:00Z', updated_at: '2026-10-09T00:01:00Z' } });
  });
  const dialogs: string[] = [];
  page.on('dialog', async (dialog) => { dialogs.push(dialog.message()); await dialog.dismiss(); });
  await openModelSettings(page, instances[0].model);
  const flash = page.getByRole('spinbutton', { name: 'Qwen3.8-Flash-S4-L maximum context' });
  const dense = page.getByRole('spinbutton', { name: 'Qwen3.8-27B-S4-M maximum context' });
  const save = page.getByRole('button', { name: 'Save to this model', exact: true });
  await expect(save).toBeEnabled();
  await expect(flash).toHaveValue('');
  const contexts = page.locator('.model-context-settings');
  await expect(contexts.locator('.model-context-controls svg')).toHaveCount(0);
  await expect(page.getByRole('combobox', { name: 'Per-model context model' })).toHaveCount(0);
  await flash.fill('8192');
  await save.click();
  await expect.poll(() => reloads).toEqual([{ instance_id: 'flash', context_size: 8192, yarn_enabled: false }]);
  await expect(save).toHaveText('Saving…');
  await page.screenshot({ path: testInfo.outputPath('per-model-context.png'), animations: 'disabled' });
  complete();
  await expect(save).toHaveText('Save to this model');
  await expect(flash).toHaveValue('8192');
  await page.getByRole('button', { name: 'Close model settings' }).click();
  await openModelSettings(page, instances[1].model);
  await expect(dense).toHaveValue('');
  await expect(save).toBeEnabled();
  await dense.fill('65536');
  await save.click();
  await expect.poll(() => reloads).toHaveLength(2);
  expect(reloads[1]).toMatchObject({ instance_id: 'dense', context_size: 65536 });
  expect(dialogs).toEqual([]);
  expect(await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth)).toBe(false);
});

test('网页运行服务允许编辑端口，无效端口不发送更改', async ({ page }) => {
  await mockStudioServer(page);
  const updates: number[] = [];
  await page.route('**/api/v1/runtime/listener', (route) => {
    if (route.request().method() === 'PUT') updates.push(route.request().postDataJSON().port);
    return route.fulfill({ json: { host: '127.0.0.1', port: 8090, configurable: true } });
  });
  await page.goto('/runtime');
  const port = page.getByRole('spinbutton', { name: 'OpenAI port', exact: true });
  await expect(port).toBeEnabled();
  await expect(port).toHaveValue('8090');
  await port.fill('65536');
  await port.blur();
  await expect(page.getByText('Port must be an integer between 1 and 65535')).toBeVisible();
  expect(updates).toEqual([]);
});

test('不同架构模型在资产与下载页面显示对应标识', async ({ page }) => {
  const state = await mockStudioServer(page);
  const models = [
    { name: 'Qwen3.8-Flash-Next-S4-L', architecture: 'qwen4_exp', vendor: 'qwen' },
    { name: 'DeepSeek-V4.1-Flash', architecture: 'deepseek_v4', vendor: 'deepseek' },
    { name: 'GLM-5.3', architecture: 'glm5_next', vendor: 'zai' },
  ];
  const artifacts = models.map((item, index) => ({ ...item, id: `model-${index}`, format: 'mfq',
    shard_count: 1, total_bytes: 1 << 20, tensor_count: 1, record_count: 1, dtypes: [],
    complete: true, loadable: true, modified_at: '2026-01-01' }));
  const instances = models.map((item, index) => ({ id: `instance-${index}`, model: item.name,
    state: 'ready', devices: ['metal'], active_sessions: 0, queued_requests: 0, context_size: 32768 }));
  await page.route('**/api/v1/runtime/models', (route) => route.fulfill({ json: { data: models.map((item) => ({ id: item.name })) } }));
  await page.route('**/api/v1/runtime/instances', (route) => route.fulfill({ json: { data: instances } }));
  await page.route(/\/api\/v1\/runtime\/status(?:\?.*)?$/, (route) => route.fulfill({ json: {
    model: models[0].name, model_type: models[0].architecture, runtime_state: 'ready', instance_id: 'instance-0', max_context: 32768,
  } }));
  await page.route(/\/api\/v1\/models(?:\?.*)?$/, (route) => route.fulfill({ json: { data: artifacts } }));
  await page.route(/\/api\/v1\/hub\/official(?:\?.*)?$/, (route) => route.fulfill({ json: {
    ...officialCatalog, data: models.map((item, index) => ({ ...officialCatalog.data[0], ...item, id: `catalog-${index}` })),
  } }));
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('/inference');
  await expect(page.locator('.runtime-hero-actions [data-model-vendor="qwen"]')).toBeVisible();
  const heroMark = await page.locator('.runtime-hero-actions .model-vendor-mark').boundingBox();
  const heroButton = await page.locator('.runtime-hero-actions button').first().boundingBox();
  expect(heroMark!.x + heroMark!.width).toBeLessThan(heroButton!.x);
  for (const item of models) await expect(page.locator(`.overview-model-grid [data-model-vendor="${item.vendor}"]`)).toBeVisible();
  await navigateClient(page, '/models');
  await expect(page.locator('.loaded-model-panel')).toHaveCount(0);
  await expect(page.locator('.loaded-model-panel').getByRole('button', { name: /Use in chat|Current/ })).toHaveCount(0);
  for (const item of models) {
    await expect(page.locator(`.model-library-panel [data-model-vendor="${item.vendor}"]`)).toBeVisible();
  }
  await navigateClient(page, '/runtime');
  await expect(page.locator('.server-model-control')).toHaveCount(0);
  await navigateClient(page, '/model-hub');
  for (const item of models) await expect(page.locator(`.official-model-card [data-model-vendor="${item.vendor}"]`)).toBeVisible();
  await expect(page.locator('.model-detail-heading [data-model-vendor="qwen"]')).toBeVisible();
  await page.locator('.official-model-card').last().click();
  await expect(page.locator('.model-detail-heading [data-model-vendor="zai"]')).toBeVisible();
  await expect(page.locator('.model-variant .model-vendor-mark')).toHaveCount(0);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  expect(errors).toEqual([]);
  expect(state.unexpected).toEqual([]);
});

test('推理预算按内存架构显示正确容量和带宽', async ({ page }) => {
  await mockStudioServer(page);
  await page.goto('/model-hub');
  await expect(page.locator('.detected-hardware-summary strong')).toContainText('128 GiB URAM');
  const unified = page.locator('.detected-memory-pool');
  await expect(unified).toHaveCount(1);
  await expect(unified.locator('strong')).toHaveText('128 GiB URAM');
  await expect(unified.locator('small')).toHaveText('571.8 GiB/s');
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.route(/\/api\/v1\/hub\/official(?:\?.*)?$/, (route) => route.fulfill({ json: {
    ...officialCatalog, system: { platform: 'Windows', machine: 'AMD64', backend: 'cuda',
      cpu_name: 'AMD Ryzen 5 9600X', gpu_names: ['NVIDIA GeForce RTX 5090'], physical_memory_bytes: 64 * 2 ** 30,
      memory_pools: [{ kind: 'vram', capacity_bytes: 32 * 2 ** 30, bandwidth_bytes_per_second: 1792e9 },
        { kind: 'ram', capacity_bytes: 64 * 2 ** 30, bandwidth_bytes_per_second: 104 * 2 ** 30 }] },
  } }));
  await page.getByRole('button', { name: 'Refresh', exact: true }).click();
  await expect(page.locator('.detected-memory-pool strong')).toHaveText(['32 GiB VRAM', '64 GiB RAM']);
  await expect(page.locator('.detected-memory-pool small')).toHaveText(['1,668.9 GiB/s', '104 GiB/s']);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  expect(errors).toEqual([]);
});

test('注册资产显示文件总大小', async ({ page }) => {
  const state = await mockStudioServer(page);
  const nextModel = 'Loaded without registration';
  const assets = [32, 8].map((size, index) => ({ id: `asset-${index}`, name: `Registered checkpoint ${index}`,
    architecture: 'test', format: 'mfq', total_bytes: size * 2 ** 30, shard_count: 1,
    tensor_count: 1, record_count: 1, dtypes: [], complete: true, loadable: true, modified_at: '2026-01-01' }));
  await page.route(/\/api\/v1\/models(?:\?.*)?$/, (route) => route.fulfill({ json: { data: assets } }));
  await page.route('**/api/v1/runtime/models', (route) => route.fulfill({ json: { data: [] } }));
  await page.route('**/api/v1/runtime/instances', (route) => route.fulfill({ json: { data: [
    { id: 'instance-1', model: 'Studio Test Model', state: 'ready', devices: ['cpu'] },
    { id: 'instance-2', model: nextModel, state: 'ready', devices: ['cpu'] },
  ] } }));
  await page.goto('/models');
  await expect(page.locator('.model-library-summary > span').last()).toContainText('Total file size');
  await expect(page.locator('.model-library-summary > span').last().locator('strong')).toHaveText('40 GiB');
  await expect(page.getByText('Switching keeps the registered asset')).toHaveCount(0);
  expect(state.unexpected).toEqual([]);
});

test('运行资源按模型分段并显示正确预算和服务端点', async ({ page }, testInfo) => {
  await mockStudioServer(page);
  const instances = [1, 2, 3, 4].map((index) => ({
    id: `resource-${index}`, model: `Resource Model ${index}`, state: 'ready', devices: ['metal'],
    active_sessions: 0, queued_requests: 0, started_at: `2026-01-01T00:00:0${index}Z`,
    memory: { resident_weight_bytes: index * 2 ** 30, kv_bytes: index * 2 ** 20,
      wired_available: true, wired_bytes: index * 2 ** 30, wired_limit_bytes: 20 * 2 ** 30,
      context_count: index, prefix_cache_blocks: index * 2, ssd_experts: true,
      ssd_expert_bytes: index * 2 ** 30, ssd_ple: true, ssd_ple_bytes: index * 2 ** 30 },
  }));
  await page.route('**/api/v1/runtime/instances', (route) => route.fulfill({ json: { data: instances } }));
  await page.route(/\/api\/v1\/runtime\/status(?:\?.*)?$/, (route) => route.fulfill({ json: {
    runtime_state: 'ready', model: 'Resource Model 1', instance_id: 'resource-1',
    runtime_memory_budget_bytes: 32 * 2 ** 30, runtime_memory_effective_budget_bytes: 20 * 2 ** 30,
  } }));
  await page.goto('/inference');
  await expect(page.getByText('Resource overview', { exact: true })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Runtime resources', exact: true })).toBeVisible();
  await expect(page.locator('.memory-model-legend i')).toHaveCount(4);
  await expect(page.getByTitle('Resource Model 1 · Metal wired 1 GiB / 20 GiB')).toBeVisible();
  await expect(page.getByText('10 contexts · 20 cache blocks')).toBeVisible();
  for (const tier of ['weights', 'kv', 'experts', 'ple']) {
    const bars = page.locator(`[data-tier="${tier}"] .memory-tier-track > span`);
    await expect(bars).toHaveCount(4);
    const width = parseFloat(await bars.first().evaluate((segment) => (segment as HTMLElement).style.width));
    expect(width).toBeCloseTo(tier === 'weights' ? 5 : tier === 'kv' ? 1 / 1024 / 10 * 100 : 10);
  }
  await expect(page.locator('[data-tier="weights"] .memory-tier-heading > span')).toHaveText('10 GiB / 20 GiB');
  await expect(page.locator('[data-tier="kv"] .memory-tier-heading > span')).toHaveText('10 MiB / 10 GiB');
  await expect(page.getByText(/Colors show each model/)).toHaveCount(0);
  const serviceOrigin = new URL(page.url()).origin;
  await expect(page.locator('.overview-endpoint-panel code')).toHaveText([
    `${serviceOrigin}/v1`, 'http://127.0.0.1:8091/v1/messages', `${serviceOrigin}/v1/systemone`,
  ]);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.locator('.overview-memory-panel').screenshot({ path: testInfo.outputPath('resource-hierarchy.png') });
});

test('加载条显示实际任务进度，部分缺失的资源明细保留已知数值', async ({ page }, testInfo) => {
  await mockStudioServer(page);
  const job = { id: 'loading-job', kind: 'model.load', payload: { model: 'Loading Model' },
    progress: 0.25, status: 'running', cancel_requested: false,
    created_at: '2026-01-01', updated_at: '2026-01-01' };
  const instances = [{ id: 'loading', model: 'Loading Model', state: 'loading', devices: ['metal'],
    active_sessions: 0, queued_requests: 0, context_size: 32768 }];
  let advance!: () => void;
  const progressEvent = new Promise<void>((resolve) => { advance = resolve; });
  await page.route(/\/api\/v1\/jobs(?:\?.*)?$/, (route) => route.fulfill({ json: { data: [job] } }));
  await page.route('**/api/v1/jobs/loading-job/events/stream', async (route) => {
    await progressEvent;
    job.progress = 0.75;
    await route.fulfill({ contentType: 'text/event-stream', body: `data: ${JSON.stringify({
      job_id: job.id, sequence: 1, type: 'progress', level: 'info', progress: 0.75,
      data: {}, created_at: job.updated_at,
    })}\n\n` }).catch(() => undefined);
  });
  await page.route('**/api/v1/runtime/instances', (route) => route.fulfill({ json: { data: instances } }));
  await page.route(/\/api\/v1\/runtime\/status(?:\?.*)?$/, (route) => route.fulfill({ json: {
    runtime_state: 'loading', model: 'Loading Model', instance_id: 'loading',
  } }));
  await page.route(/\/api\/v1\/models(?:\?.*)?$/, (route) => route.fulfill({ json: { data: [{
    id: 'artifact', name: 'Loading Model', architecture: 'qwen35', loadable: true,
    total_bytes: 1024, shard_count: 1,
  }] } }));
  await page.goto('/inference');
  await expect(page.locator('.runtime-hero progress')).toHaveAttribute('value', '0.25');
  advance();
  await expect(page.locator('.runtime-hero progress')).toHaveAttribute('value', '0.75');
  await navigateClient(page, '/models');
  await expect(page.locator('.loaded-model-panel progress')).toHaveCount(0);
  await expect(page.locator('.model-library-panel progress')).toHaveAttribute('value', '0.75');
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({ path: testInfo.outputPath('model-loading-progress.png') });
  job.status = 'succeeded';
  job.progress = 1;
  await page.route('**/api/v1/runtime/instances', (route) => route.fulfill({ json: { data: [
    { id: 'known', model: 'Known Model', state: 'ready', devices: ['metal'], active_sessions: 0, queued_requests: 0, memory: {
      resident_weight_bytes: 2 ** 30, kv_bytes: 0, context_count: 0, prefix_cache_blocks: 0,
      ssd_experts: false, ssd_expert_bytes: 0, ssd_ple: false, ssd_ple_bytes: 0,
    } }, { id: 'unknown', model: 'Unknown Model', state: 'ready', devices: ['metal'], active_sessions: 0, queued_requests: 0, memory: null },
  ] } }));
  await page.route(/\/api\/v1\/runtime\/status(?:\?.*)?$/, (route) => route.fulfill({ json: {
    runtime_state: 'ready', model: 'Known Model', runtime_memory_effective_budget_bytes: 4 * 2 ** 30,
  } }));
  await page.goto('/inference');
  await expect(page.locator('[data-tier="weights"] .memory-tier-heading > span')).toHaveText('≥ 1 GiB / 4 GiB');
  await expect(page.locator('[data-tier="weights"] .memory-tier-notice')).toContainText('Unknown Model');
  await expect(page.locator('[data-tier="weights"] .memory-tier-track > span')).toHaveCount(1);
  await page.unrouteAll({ behavior: 'ignoreErrors' });
});

for (const kind of ['model.load', 'download.modelscope']) {
  test(`${kind} 事件流中断后切页返回仍同步进度`, async ({ page }) => {
    await mockStudioServer(page);
    const loading = kind === 'model.load';
    const job = { id: 'resume-job', kind, payload: loading ? { model: 'Loading Model' } : { repo_id: 'Tylogi/test-MFQ' },
      progress: 0.25, status: 'running', cancel_requested: false,
      created_at: '2026-01-01T00:00:00Z', updated_at: '2026-01-01T00:00:00Z' };
    await page.route(/\/api\/v1\/jobs(?:\?.*)?$/, (route) => route.fulfill({ json: { data: [job] } }));
    await page.route(/\/api\/v1\/jobs\/resume-job\/events\/stream(?:\?.*)?$/, (route) => route.fulfill({ status: 503, json: { error: 'stream offline' } }));
    if (loading) {
      await page.route('**/api/v1/runtime/instances', (route) => route.fulfill({ json: { data: [{
        id: 'loading', model: 'Loading Model', state: 'loading', devices: ['metal'], active_sessions: 0, queued_requests: 0,
      }] } }));
      await page.route(/\/api\/v1\/runtime\/status(?:\?.*)?$/, (route) => route.fulfill({ json: {
        runtime_state: 'loading', model: 'Loading Model', instance_id: 'loading',
      } }));
    }
    const path = loading ? '/inference' : '/model-hub';
    await page.goto(path);
    if (!loading) await page.locator('.download-circle').click();
    const progress = () => page.locator(loading ? '.runtime-hero progress' : '.download-queue-item progress');
    await expect(progress()).toHaveAttribute('value', '0.25');
    await navigateClient(page, '/models');
    job.progress = 0.75;
    job.updated_at = '2026-01-01T00:00:05Z';
    await navigateClient(page, path);
    if (!loading) await page.locator('.download-circle').click();
    await expect(progress()).toHaveAttribute('value', '0.75');
    job.progress = 0.9;
    job.updated_at = '2026-01-01T00:00:06Z';
    await expect(progress()).toHaveAttribute('value', '0.9');
  });
}

test('页面按需请求自己的资源，概览不预载其他业务列表', async ({ page }) => {
  const state = await mockStudioServer(page);
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('/inference');
  await expect(page.getByRole('heading', { name: 'Overview', exact: true })).toBeVisible();
  for (const path of ['sessions', 'datasets', 'datasets/catalog', 'evaluations', 'evaluations/tools', 'models', 'runtime/logs', 'runtime/profiles', 'cluster/nodes']) {
    expect(state.requests).not.toContain(`GET /api/v1/${path}`);
  }
  const routes = [
    ['/model-hub', null],
    ['/versions', null],
    ['/runtime', null],
    ['/resources', '/api/v1/runtime/resources'],
    ['/logs', '/api/v1/runtime/logs'],
  ] as const;
  for (const [path, endpoint] of routes) {
    await navigateClient(page, path);
    await expect(page.locator('main h1')).toBeVisible();
    if (endpoint) await expect.poll(() => state.requests.includes(`GET ${endpoint}`)).toBe(true);
    if (path === '/runtime') {
      expect(state.requests).not.toContain('GET /api/v1/cluster/nodes');
      await page.getByRole('tab', { name: 'Advanced', exact: true }).click();
      await expect.poll(() => state.requests.includes('GET /api/v1/cluster/nodes')).toBe(true);
    }
    if (path === '/versions') expect(state.requests).not.toContain('GET /api/v1/presets');
  }
  expect(errors).toEqual([]);
  expect(state.unexpected).toEqual([]);
});

test('模型下载留在本页，飞入圆圈后打开第三个队列标签', async ({ page }, testInfo) => {
  const state = await mockStudioServer(page);
  const jobs: unknown[] = [{ id: 'load', kind: 'model.load', status: 'succeeded', progress: 1,
    payload: { repo_id: 'Not a download' }, cancel_requested: false, created_at: '2026-01-01', updated_at: '2026-01-01' }];
  await page.route('**/api/v1/jobs/kinds', (route) => route.fulfill({ json: { data: [{ kind: 'download.modelscope', payload_schema: {} }] } }));
  await page.route(/\/api\/v1\/jobs(?:\?.*)?$/, async (route) => {
    if (route.request().method() === 'GET') return route.fulfill({ json: { data: jobs } });
    const { kind, payload } = route.request().postDataJSON();
    expect(kind).toBe('download.modelscope');
    const job = { id: 'download', kind, payload, status: 'succeeded', progress: 1,
      cancel_requested: false, created_at: '2026-01-01', updated_at: '2026-01-01' };
    jobs.unshift(job);
    return route.fulfill({ json: job });
  });
  await page.goto('/model-hub');
  await expect(page.getByRole('img', { name: 'Apple · METAL' })).toBeVisible();
  await expect(page.getByText('Apple M5 Max · 18 CPU / 40 GPU · 128 GiB URAM')).toBeVisible();
  await expect(page.getByRole('tab')).toHaveText(['Recommended models', 'Model browser', 'Download queue']);
  await page.getByRole('button', { name: 'Download', exact: true }).first().click();
  await expect(page).toHaveURL('/model-hub');
  await expect(page.getByRole('tab', { name: 'Recommended models' })).toHaveAttribute('aria-selected', 'true');
  await expect(page.locator('.download-flight')).toBeVisible();
  expect(await page.locator('.download-flight').evaluate((node) => getComputedStyle(node).animationName)).toBe('download-fly');
  expect(await page.locator('.download-circle').evaluate((node) => {
    const rect = node.getBoundingClientRect();
    return document.elementFromPoint(rect.left + rect.width / 2, rect.top + rect.height / 2)?.closest('.download-circle') === node;
  })).toBe(true);
  await page.getByRole('button', { name: 'Download queue', exact: true }).click();
  await expect(page.getByRole('tab', { name: 'Download queue' })).toHaveAttribute('aria-selected', 'true');
  await expect(page.locator('.download-queue-item')).toHaveCount(1);
  await expect(page.locator('.download-queue-item')).toContainText('example/studio-layout-test');
  await expect(page.locator('.download-queue-item')).toContainText('100%');
  await expect(page.getByText('Not a download')).toHaveCount(0);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({ path: testInfo.outputPath('download-queue.png'), animations: 'disabled' });
  expect(state.requests).not.toContain('GET /api/v1/artifacts/lineage');
  expect(state.unexpected).toEqual([]);
});

test('路由导航及模型目录弹窗键盘焦点', async ({ page }, testInfo) => {
  const state = await mockStudioServer(page);
  await page.goto('/models');
  const addModel = page.getByRole('button', { name: 'Add model', exact: true }).first();
  await expect(addModel).toBeEnabled();
  await addModel.click();
  const dialog = page.getByRole('dialog', { name: 'Choose model folder' });
  await expect(dialog).toBeVisible();
  await page.keyboard.press('Tab');
  expect(await dialog.evaluate((element) => element.contains(document.activeElement))).toBe(true);
  await page.screenshot({ path: testInfo.outputPath('model-dialog.png'), fullPage: true });
  await page.keyboard.press('Escape');
  await expect(dialog).toBeHidden();
  await expect(addModel).toBeFocused();
  if (testInfo.project.name === 'mobile')
    await page.getByRole('button', { name: 'Open sidebar', exact: true }).click();
  await page.locator('#studio-sidebar').getByRole('button', { name: 'Overview', exact: true }).click();
  await expect(page).toHaveURL('/inference');
  await expect(page.getByRole('heading', { name: 'Overview', exact: true })).toBeVisible();
  await expect(page.locator('.sidebar')).not.toHaveClass(/open/);
  await expect(page.getByRole('main')).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(
    true,
  );
  expect(state.unexpected).toEqual([]);
  await page.screenshot({ path: testInfo.outputPath('overview.png'), fullPage: true, animations: 'disabled' });
});

test('明暗主题的页面内容保持在工作区内且可滚动访问', async ({ page }, testInfo) => {
  const state = await mockStudioServer(page);
  const routes = [['/inference', 'Overview'], ['/models', 'Models'], ['/runtime', 'Service'],
    ['/model-hub', 'Model downloads']] as const;
  for (const theme of ['light', 'dark']) {
    await page.goto('/inference');
    await page.locator('#dashboard-overview .appearance-controls').getByRole('button', { name: theme === 'light' ? 'Light' : 'Dark', exact: true }).click();
    await expect(page.locator('html')).toHaveAttribute('data-theme', theme);
    for (const [path, title] of routes) {
      await navigateClient(page, path);
      await expect(page.getByRole('heading', { name: title, exact: true })).toBeVisible();
      if (path === '/model-hub') {
        await expect(page.locator('.model-detail-panel')).toBeVisible();
        await expect(page.locator('.model-variant')).toHaveCount(3);
      }
      const layout = await page.locator('.dashboard-view').evaluate((element) => ({
        width: element.clientWidth,
        scrollWidth: element.scrollWidth,
        height: element.clientHeight,
        bottom: element.getBoundingClientRect().bottom,
        viewport: window.innerHeight,
      }));
      expect(layout.scrollWidth).toBeLessThanOrEqual(layout.width + 1);
      expect(layout.height).toBeGreaterThan(0);
      expect(layout.bottom).toBeLessThanOrEqual(layout.viewport + 1);
      await page.screenshot({ path: testInfo.outputPath(`${title.replaceAll(' ', '-')}-${theme}.png`), fullPage: true, animations: 'disabled' });
    }
  }
  expect(state.unexpected).toEqual([]);
});
