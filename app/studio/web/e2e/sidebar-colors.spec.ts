import { expect, test } from '@playwright/test';
import { mockStudioServer } from './mockServer';

test('session sidebar keeps neutral controls and inference navigation stays secondary', async ({ page }, testInfo) => {
  await mockStudioServer(page);
  await page.goto('/');
  await expect(page.locator('main h1')).toBeVisible();
  if (testInfo.project.name === 'mobile') await page.getByRole('button', { name: 'Open sidebar', exact: true }).click();
  const sidebar = page.locator('#studio-sidebar');
  const engine = sidebar.getByRole('button', { name: 'Inference engine', exact: true });
  await expect(sidebar.getByRole('button', { name: 'Models', exact: true })).toBeVisible();
  for (const theme of ['light', 'dark', 'system'] as const) {
    for (const scheme of theme === 'system' ? ['light', 'dark'] as const : [theme]) {
      await page.emulateMedia({ colorScheme: scheme });
      await page.evaluate(value => document.documentElement.dataset.theme = value, theme);
      const color = await engine.evaluate(element => getComputedStyle(element).color);
      const channels = color.match(/[\d.]+/g)!.slice(0, 3).map(Number);
      expect(Math.max(...channels) - Math.min(...channels)).toBeLessThanOrEqual(4);
      await expect(engine.locator('.ui-icon').first()).toHaveCSS('color', color);
    }
  }
  await engine.click();
  await expect(sidebar.getByRole('button', { name: 'Models', exact: true })).toBeHidden();
  await engine.click();
  await expect(sidebar.getByRole('button', { name: 'Models', exact: true })).toBeVisible();
});
