import { expect, test } from '@playwright/test';
import { mockStudioServer } from './mockServer';

test('recovers a stale route chunk by refreshing the same page once', async ({ page }) => {
  await mockStudioServer(page);
  let failures = 0;
  let reloads = 0;
  let unavailable = true;
  await page.route('**/assets/OverviewPage-*.js', route => {
    if (!unavailable) return route.continue();
    failures += 1;
    return route.fulfill({ status: 404, body: 'Not Found' });
  });
  page.on('request', request => {
    if (request.resourceType() === 'document' && new URL(request.url()).pathname === '/inference') {
      reloads += 1;
      // Keep both preload and import unavailable until recovery reloads the page.
      unavailable = false;
    }
  });
  await page.goto('/model-hub');
  await expect(page.getByRole('tab', { name: 'Model browser', exact: true })).toBeVisible();
  await page.evaluate(() => { history.pushState(null, '', '/inference'); dispatchEvent(new PopStateEvent('popstate')); });
  await expect(page.getByRole('heading', { name: 'Overview', exact: true })).toBeVisible();
  await expect(page).toHaveURL(/\/inference$/);
  expect(reloads).toBe(1);
  expect(failures).toBeGreaterThan(0);
  await expect(page.locator('.failure-page')).toHaveCount(0);
});

test('stops automatic reloads for a persistent failure and makes manual retry reload the page', async ({ page }) => {
  await mockStudioServer(page);
  let unavailable = true;
  let reloads = 0;
  await page.route('**/assets/OverviewPage-*.js', route => unavailable
    ? route.fulfill({ status: 404, body: 'Not Found' }) : route.continue());
  page.on('request', request => {
    if (request.resourceType() === 'document' && new URL(request.url()).pathname === '/inference') reloads += 1;
  });
  await page.goto('/model-hub');
  await expect(page.getByRole('tab', { name: 'Model browser', exact: true })).toBeVisible();
  await page.evaluate(() => { history.pushState(null, '', '/inference'); dispatchEvent(new PopStateEvent('popstate')); });
  await expect(page.getByRole('heading', { name: 'Page unavailable', exact: true })).toBeVisible();
  expect(reloads).toBe(1);
  unavailable = false;
  await page.getByRole('button', { name: 'Retry page', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Overview', exact: true })).toBeVisible();
  expect(reloads).toBe(2);
});
