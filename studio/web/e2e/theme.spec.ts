import { expect, test } from '@playwright/test';
import { mockStudioServer } from './mockServer';

function contrast(foreground: string, background: string) {
  const luminance = (color: string) => {
    const channels = color.match(/[\d.]+/g)!.slice(0, 3).map(Number).map(value => {
      const channel = value / 255;
      return channel <= 0.04045 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4;
    });
    return channels[0] * 0.2126 + channels[1] * 0.7152 + channels[2] * 0.0722;
  };
  const a = luminance(foreground), b = luminance(background);
  return (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);
}

for (const mode of ['light', 'dark', 'system-light', 'system-dark'] as const) {
  test(`theme follows the selected mode and keeps controls readable (${mode})`, async ({ page }) => {
    const dark = mode.endsWith('dark');
    const theme = mode.startsWith('system') ? 'system' : mode;
    const state = await mockStudioServer(page);
    await page.emulateMedia({ colorScheme: dark ? 'dark' : 'light' });
    await page.addInitScript(value => localStorage.setItem('tyloinfer.studio.generation.v1',
      JSON.stringify({ language: 'en', theme: value })), theme);

    await page.goto('/runtime');
    await expect(page.locator('html')).toHaveAttribute('data-theme', theme);
    await expect(page.locator('html')).toHaveCSS('color-scheme', dark ? 'dark' : 'light');
    const vision = page.getByRole('switch', { name: 'Vision input', exact: true });
    await expect(vision).toBeChecked();
    const paint = await vision.evaluate(element => ({
      track: getComputedStyle(element).backgroundColor,
      thumb: getComputedStyle(element.querySelector('.studio-switch-thumb')!).backgroundColor,
    }));
    expect(contrast(paint.thumb, paint.track)).toBeGreaterThanOrEqual(3);
    await vision.click();
    await expect(vision).not.toBeChecked();

    await page.goto('/inference');
    await expect(page.locator('main h1')).toBeVisible();
    const actionColors = await page.locator('.runtime-hero-actions button').evaluate(element => {
      const style = getComputedStyle(element);
      return { foreground: style.color, background: style.backgroundColor };
    });
    expect(contrast(actionColors.foreground, actionColors.background)).toBeGreaterThanOrEqual(4.5);
    expect(state.unexpected).toEqual([]);
  });
}
