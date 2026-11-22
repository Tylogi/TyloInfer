import { expect, type Page } from '@playwright/test';
import type { ModelArtifact } from '../src/shared/api/types';

/** Build a registered model fixture with the fields required by the settings page. */
export function modelArtifact(id: string, name: string, overrides: Partial<ModelArtifact> = {}): ModelArtifact {
  return {
    id, name, architecture: 'qwen35', format: 'tq', shard_count: 1,
    total_bytes: 4 * 2 ** 30, estimated_resident_weight_bytes: 4 * 2 ** 30,
    context_capacity: 262144, tensor_count: 1, record_count: 1, dtypes: ['F16'],
    complete: true, loadable: true, modified_at: '2026-11-01T00:00:00Z',
    ...overrides,
  };
}

/** Open a registered or running model's settings through its visible configure action. */
export async function openModelSettings(page: Page, name: string, language = 'en') {
  await page.goto('/models');
  const library = page.getByRole('button', { name: language === 'zh-CN' ? `配置模型：${name}` : `Configure model: ${name}`, exact: true });
  const running = page.locator('.loaded-model-panel .model-row').filter({ hasText: name }).getByRole('button', { name: language === 'zh-CN' ? '配置' : 'Configure', exact: true });
  await expect(library.or(running)).toBeVisible();
  await library.or(running).click();
  const dialog = page.getByRole('dialog', { name, exact: true });
  await expect(dialog).toBeVisible();
  return dialog;
}
