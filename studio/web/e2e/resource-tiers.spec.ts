import { expect, test } from '@playwright/test';
import { mockStudioServer } from './mockServer';

for (const language of ['zh-CN', 'en'] as const) {
  test(`RAM/SSD categories and accessible tier details (${language})`, async ({ page }, testInfo) => {
    const state = await mockStudioServer(page, { language });
    const errors: string[] = [];
    page.on('pageerror', error => errors.push(error.message));
    for (const path of ['/', '/resources']) {
      await page.goto(path);
      await expect(page.locator('[data-memory-group]')).toHaveCount(2);
      await expect(page.locator('[data-memory-group="RAM"] [data-tier]')).toHaveCount(2);
      await expect(page.locator('[data-memory-group="SSD"] [data-tier]')).toHaveCount(3);
      await expect(page.locator('.memory-capacity-note')).toContainText(language === 'zh-CN' ? '扣除系统和其他进程占用' : 'after system and other process usage');
      await expect(page.locator('[data-memory-group="RAM"] h3')).toHaveText(language === 'zh-CN' ? '内存' : 'RAM');
      const trigger = page.locator('.memory-heading-title').getByRole('button', { name: language === 'zh-CN' ? '资源分层说明' : 'Memory tier details' });
      const help = page.locator('.studio-tooltip');
      await expect(help).not.toBeVisible();
      await trigger.scrollIntoViewIfNeeded();
      await trigger.hover();
      await expect(help).toBeVisible();
      await expect(help).toContainText(language === 'zh-CN' ? '前缀缓存' : 'prefix caches');
      const bounds = await help.boundingBox();
      expect(bounds!.x).toBeGreaterThanOrEqual(0);
      expect(bounds!.x + bounds!.width).toBeLessThanOrEqual(page.viewportSize()!.width);
      await page.screenshot({ path: testInfo.outputPath(`${path.slice(1) || 'overview'}-help-${language}.png`), fullPage: true });
      await page.keyboard.press('Escape');
      await expect(help).not.toBeVisible();
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    }
    expect(errors).toEqual([]);
    expect(state.unexpected).toEqual([]);
  });
}
