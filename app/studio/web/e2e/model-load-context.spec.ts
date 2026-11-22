import { expect, test } from '@playwright/test';
import { mockStudioServer } from './mockServer';
import { openModelSettings } from './modelSettings';

test('load context defaults reach the model load request and remain readable in both themes', async ({ page }, testInfo) => {
  await mockStudioServer(page, { language: 'zh-CN' });
  const writes: Record<string, unknown>[] = [];
  await page.route(/\/api\/v1\/models(?:\?.*)?$/, route => route.fulfill({ json: { data: [{ id: 'local', name: 'Qwen3.5',
    architecture: 'qwen3_5', format: 'mfq', shard_count: 1, total_bytes: 40 * 2 ** 30,
    estimated_resident_weight_bytes: 8 * 2 ** 30, context_capacity: 131072, tensor_count: 1,
    record_count: 1, dtypes: [], complete: true, loadable: true, modified_at: '2026-10-14T00:00:00Z' }] } }));
  await page.route('**/api/v1/models/local/load-plan?*', route => {
    const query = new URL(route.request().url()).searchParams;
    const maximum = query.get('preallocate') === 'false' ? 131072 : (30 - Number(query.get('reserve'))) * 2048;
    return route.fulfill({ json: { max_context: maximum, context_size: Number(query.get('context_size')) || maximum,
      precision: Number(query.get('precision')) || 8, precisions: [8, 4], usable: true, reason: null } });
  });
  await page.route('**/api/v1/runtime/status*', route => route.fulfill({ json: {
    runtime_state: 'ready', model: 'Studio Test Model', model_type: 'Test', max_context: 8192,
    total_requests: 0, active_requests: 0, sampling_defaults: {}, memory_architecture: 'discrete',
    host_memory_available_bytes: 12 * 2 ** 30,
    gpu_devices: [{ id: '0', name: 'GPU', memory_available_bytes: 3 * 2 ** 30 }],
  } }));
  await page.route('**/api/v1/models/load', route => {
    writes.push(route.request().postDataJSON());
    return route.fulfill({ json: { operation_id: 'load', status: 'accepted' } });
  });
  await openModelSettings(page, 'Qwen3.5', 'zh-CN');
  const panel = page.getByRole('region', { name: '加载选项' });
  await expect(panel.getByRole('spinbutton')).toHaveValue('40960');
  await expect(panel.getByRole('radio', { name: /普通/ })).toBeChecked();
  await expect(panel.getByRole('combobox', { name: 'KV 精度' })).toContainText('自动 · INT8');
  expect(writes).toHaveLength(0);
  await panel.getByRole('radio', { name: /安全/ }).check();
  await expect(panel.getByRole('spinbutton')).toHaveValue('20480');
  await panel.getByRole('combobox', { name: 'KV 分配' }).selectOption('dynamic');
  await expect(panel.getByRole('spinbutton')).toHaveValue('131072');
  await panel.getByRole('combobox', { name: 'KV 精度' }).selectOption('4');
  await panel.getByRole('spinbutton').fill('65536');
  for (const theme of ['light', 'dark']) {
    await page.evaluate(value => { document.documentElement.dataset.theme = value; }, theme);
    await panel.scrollIntoViewIfNeeded();
    await page.screenshot({ path: testInfo.outputPath(`context-${theme}.png`), fullPage: true });
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  }
  await page.reload();
  await openModelSettings(page, 'Qwen3.5', 'zh-CN');
  await expect(panel.getByRole('spinbutton')).toHaveValue('65536');
  await page.keyboard.press('Escape');
  await page.getByRole('button', { name: '加载', exact: true }).click();
  await expect.poll(() => writes.length).toBe(1);
  expect(writes[0]).toMatchObject({ context_size: 65536, context_management: {
    reserve_percent: 20, precision: 4, vram_preallocated: false,
  } });
  expect(writes[0].context_management).not.toHaveProperty('context_size');
});
