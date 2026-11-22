import { expect, test, type Page } from '@playwright/test';

// This is a test-only ChatProvider/ChatPage harness; the published Studio has no chat route.

const harnessPort = Number(process.env.TYLOINFER_CHAT_HARNESS_PORT ?? 5203);
const harnessUrl = `http://127.0.0.1:${harnessPort}`;
const apiKey = 'test-browser-key';
const expectedDefaultSampling = { temperature: 0.7, max_tokens: 4096 };

async function signIn(page: Page): Promise<void> {
  await page.getByRole('textbox', { name: /API Key/i }).fill(apiKey);
  await page.getByRole('button', { name: 'Sign in' }).click();
  await expect(page.getByRole('textbox', { name: 'Message' })).toBeVisible();
}

async function startChat(page: Page): Promise<void> {
  const newChat = page.getByRole('button', { name: 'New chat' });
  const start = page.getByRole('button', { name: 'Start chat' });
  await expect(newChat.or(start).first()).toBeVisible();
  await newChat.or(start).first().click();
}

async function releaseStream(page: Page): Promise<void> {
  const result = await page.evaluate(async () => {
    const response = await fetch('/__test/release-stream', { method: 'POST' });
    return response.json() as Promise<{ status: string }>;
  });
  expect(result.status).toBe('released');
}

test.beforeEach(async ({ page }) => {
  await page.addInitScript((url) => {
    localStorage.setItem('tyloinfer.studio.service-url', url);
  }, harnessUrl);
});

test('renders streamed chat deltas and reloads persisted history in the component harness', async ({ page }) => {
  await page.goto('/');
  await signIn(page);
  await startChat(page);
  const composer = page.getByRole('textbox', { name: 'Message' });
  await expect(composer).toBeEnabled();
  await composer.fill('hello from browser');
  await page.getByRole('button', { name: 'Send' }).click();

  const live = page.locator('.live-message .message-body');
  await expect(live).toContainText('service echo:');
  await expect(page.getByRole('button', { name: 'Stop generation' })).toBeVisible();
  await expect(page.locator('.message-assistant .message-content')).toHaveCount(0);
  await releaseStream(page);
  await expect(page.locator('.message-assistant .message-content').filter({ hasText: 'service echo: hello from browser' })).toHaveCount(1);

  const calls = await page.evaluate(async () => {
    const response = await fetch('/__test/backend-calls');
    return response.json();
  });
  expect(calls.calls.at(-1)).toMatchObject({
    model: 'test-browser-model',
    messages: [{ role: 'user', content: 'hello from browser' }],
    sampling: expectedDefaultSampling,
  });

  await page.reload();
  await signIn(page);
  await expect(page.locator('.message-assistant .message-content').filter({ hasText: 'service echo: hello from browser' })).toHaveCount(1);
});

test('shows a backend stream failure in the component UI and recovers', async ({ page }) => {
  await page.goto('/');
  await signIn(page);
  await startChat(page);
  const composer = page.getByRole('textbox', { name: 'Message' });
  await expect(composer).toBeEnabled();
  await composer.fill('trigger-backend-error');
  await page.getByRole('button', { name: 'Send' }).click();
  await expect(page.getByRole('alert')).toContainText('deterministic test backend rejected');

  await composer.fill('recovery works');
  await page.getByRole('button', { name: 'Send' }).click();
  await expect(page.locator('.live-message .message-body')).toContainText('service echo:');
  await expect(page.getByRole('button', { name: 'Stop generation' })).toBeVisible();
  await releaseStream(page);
  await expect(page.locator('.message-assistant .message-content').filter({ hasText: 'service echo: recovery works' })).toHaveCount(1);
});
