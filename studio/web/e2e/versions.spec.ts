import { expect, test } from '@playwright/test';
import packageInfo from '../package.json' with { type: 'json' };
import { mockStudioServer } from './mockServer';

for (const language of ['zh-CN', 'en'] as const) {
  test(`settings version manager browses Releases and retains offline notes (${language})`, async ({ page }, testInfo) => {
    await mockStudioServer(page, { language });
    const en = language === 'en';
    const release = { tag_name: 'v0.3.3', name: 'TyloInfer 0.3.3', body: '# Faster inference\n\n- Packed projection improvements\n- Updated application layout\n\n## Compatibility\n\nExisting models and settings are retained.',
      draft: false, prerelease: false, published_at: '2026-10-10T00:00:00Z', assets: [] };
    let offline = false;
    await page.route('https://api.github.com/repos/Tylogi/TyloInfer/releases?**', route => offline ? route.abort() : route.fulfill({ json: [release,
      { ...release, tag_name: 'v0.3.2', name: 'TyloInfer 0.3.2', body: 'Previous Release notes' },
      { ...release, tag_name: 'v0.4.0rc1', prerelease: true }, { ...release, tag_name: 'nightly' }] }));
    await page.goto('/inference');
    if (testInfo.project.name === 'mobile') await page.getByRole('button', { name: en ? 'Open sidebar' : '打开侧栏', exact: true }).click();
    await page.getByRole('button', { name: en ? 'Settings' : '设置', exact: true }).click();
    const settings = page.getByRole('dialog', { name: en ? 'Settings' : '设置', exact: true });
    await expect(settings.getByText(`TyloInfer Studio ${packageInfo.version}`, { exact: true })).toBeVisible();
    await settings.getByRole('button', { name: en ? 'Versions' : '版本管理', exact: true }).click();
    await expect(page).toHaveURL(/\/versions$/);
    await expect(page.getByRole('heading', { name: en ? 'Version manager' : '版本管理', exact: true })).toBeVisible();
    await expect(page.getByRole('heading', { name: en ? 'Release' : '正式版', exact: true })).toBeVisible();
    await expect(page.getByRole('button', { name: en ? 'View all Releases' : '查看全部正式版', exact: true })).toBeVisible();
    await expect(page.locator('.release-version')).toHaveCount(2);
    await expect(page.getByRole('heading', { name: 'Faster inference' })).toBeVisible();
    const previous = page.locator('.release-version').nth(1);
    await previous.locator('summary').click();
    await expect(previous).toContainText('Previous Release notes');
    await expect(page.locator('.experimental-versions')).not.toHaveAttribute('open', '');
    await expect(page.getByRole('button', { name: /Install & restart|安装并重启/ })).toHaveCount(0);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await page.locator('.workspace').screenshot({ path: testInfo.outputPath('versions.png'), animations: 'disabled' });
    offline = true;
    await page.getByRole('button', { name: en ? 'Check now' : '检查更新', exact: true }).click();
    await expect(page.getByText(en ? 'Could not retrieve online releases' : '未能获取在线版本')).toBeVisible();
    await expect(page.locator('.release-version')).toHaveCount(2);
  });
}
