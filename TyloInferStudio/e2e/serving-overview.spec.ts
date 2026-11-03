import { expect, test } from '@playwright/test';
import { mockStudioServer } from './mockServer';

test('overview keeps the serving panel honest about missing native statistics', async ({ page }) => {
  const state = await mockStudioServer(page);
  // The serving resource is not part of the native server; no request may leave the page.
  await page.route('**/api/runtime/serving*', route => route.abort());
  await page.goto('/');
  const footer = page.locator('.overview-footer-grid');
  await expect(footer.locator('.overview-endpoint-panel')).toBeVisible();
  await expect(page.locator('.configuration-effect').first()).toBeVisible();
  await expect(page.locator('.metric-tile').filter({ hasText: 'Prefill' })).toContainText('--');
  expect(state.unexpected).toEqual([]);
});
