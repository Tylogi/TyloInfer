import { expect, test } from '@playwright/test';
import { mockStudioServer } from './mockServer';

// The native listener binds one port at startup: the Anthropic endpoint shares it and the
// connections page states that instead of offering a configurable port.
for (const language of ['zh-CN', 'en'] as const) {
  test(`overview shows and copies the shared native API URLs (${language})`, async ({ page }, testInfo) => {
    const state = await mockStudioServer(page, { language });
    await page.addInitScript(() => {
      Object.defineProperty(navigator, 'clipboard', { configurable: true, value: {
        writeText: async (value: string) => { document.documentElement.dataset.copiedEndpoint = value; },
      } });
    });
    await page.goto('/');
    const panel = page.locator('.overview-endpoint-panel');
    await expect(panel.locator('code')).toHaveText(['http://127.0.0.1:5187/v1', 'http://127.0.0.1:5187/v1/messages']);
    await panel.getByRole('button', { name: language === 'en' ? 'Copy OpenAI endpoint' : '复制 OpenAI 端点' }).click();
    await expect(page.locator('html')).toHaveAttribute('data-copied-endpoint', 'http://127.0.0.1:5187/v1');
    await panel.getByRole('button', { name: language === 'en' ? 'Copy Anthropic endpoint' : '复制 Anthropic 端点' }).click();
    await expect(page.locator('html')).toHaveAttribute('data-copied-endpoint', 'http://127.0.0.1:5187/v1/messages');
    await panel.scrollIntoViewIfNeeded();
    const endpointBounds = await panel.boundingBox();
    const sessionBounds = await page.locator('.overview-session-panel').boundingBox();
    expect(endpointBounds!.height).toBeLessThanOrEqual(190);
    expect(sessionBounds!.height).toBeLessThanOrEqual(190);
    expect(await panel.locator('.overview-api-entry').first().evaluate(element => {
      const label = element.querySelector('strong')!.getBoundingClientRect();
      const url = element.querySelector('code')!.getBoundingClientRect();
      return Math.abs((label.top + label.height / 2) - (url.top + url.height / 2)) < 2;
    })).toBe(true);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await page.locator('.overview-footer-grid').screenshot({ path: testInfo.outputPath('overview-api-endpoints.png'), animations: 'disabled' });
    expect(state.requests.filter(request => /^(POST|PUT|DELETE) /.test(request))).toEqual([]);
  });
}
