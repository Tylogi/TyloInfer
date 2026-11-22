import { expect, test } from '@playwright/test';

const servicePort = Number(process.env.TYLOINFER_SERVICE_PORT ?? 5190);
const serviceUrl = `http://127.0.0.1:${servicePort}`;
const apiKey = 'test-browser-key';

async function signIn(page: import('@playwright/test').Page): Promise<void> {
  await page.getByRole('textbox', { name: /API Key/i }).fill(apiKey);
  await page.getByRole('button', { name: 'Sign in' }).click();
  await expect(page.getByRole('heading', { name: 'Overview' })).toBeVisible();
}

test.beforeEach(async ({ page }) => {
  await page.addInitScript((url) => {
    localStorage.setItem('tyloinfer.studio.service-url', url);
  }, serviceUrl);
});

test('signs in through the real service and rejects invalid credentials visibly', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByRole('heading', { name: 'Sign in' })).toBeVisible();

  await page.getByRole('textbox', { name: /API Key/i }).fill('wrong-key');
  await page.getByRole('button', { name: 'Sign in' }).click();
  await expect(page.getByRole('alert')).toContainText('incorrect or expired');

  await signIn(page);
  await expect(page.getByText('test-browser-model')).toBeVisible();
});

test('loads a local model through the production models page and follows the real job stream', async ({ page }) => {
  await page.goto('/');
  await signIn(page);
  await page.getByRole('complementary', { name: 'Inference Studio' })
    .getByRole('button', { name: 'Models', exact: true }).click();
  await expect(page).toHaveURL(/\/models$/);

  const row = page.locator('.model-row', { hasText: 'fixture-model' });
  await expect(row).toBeVisible();
  await expect(row.getByRole('button', { name: 'Load' })).toBeEnabled();

  const loadResponse = page.waitForResponse((response) =>
    response.request().method() === 'POST' && response.url().endsWith('/api/v1/models/load'),
  );
  const jobEventStream = page.waitForResponse((response) =>
    response.status() === 200 && /\/api\/v1\/jobs\/[^/]+\/events\/stream$/.test(response.url()),
  );
  await row.getByRole('button', { name: 'Load' }).click();
  expect((await loadResponse).status()).toBe(202);
  expect((await jobEventStream).status()).toBe(200);

  // The load endpoint has returned HTTP 202, but the instance is still loading.
  // The page must show the intermediate state rather than treating acceptance as readiness.
  await expect(row.locator('.model-library-runtime')).toContainText('Loading');
  await expect(row.getByRole('progressbar', { name: /fixture-model load progress/i })).toBeVisible();
  await expect(row.locator('.model-library-runtime')).not.toContainText('Ready');

  // Release only the controlled runtime barrier; all state transitions and observations remain
  // on the production jobs, SSE, RuntimeProvider, catalog, and models-page implementations.
  const release = await page.request.post('/__test/release-model-load');
  expect(release.ok()).toBe(true);

  await expect(row.locator('.model-library-runtime')).toContainText('Ready');
  await expect(row.locator('.model-load-progress')).toHaveCount(0);
});
