import { defineConfig } from '@playwright/test';
import os from 'node:os';
import path from 'node:path';

const servicePort = Number(process.env.TYLOINFER_SERVICE_PORT ?? 5190);
const frontendPort = servicePort;
const harnessPort = Number(process.env.TYLOINFER_CHAT_HARNESS_PORT ?? 5203);
const python = process.env.TYLOINFER_TEST_PYTHON ?? 'python';
const database = process.env.TYLOINFER_SERVICE_DATABASE ?? path.join(
  os.tmpdir(),
  `tyloinfer-e2e-service-${process.pid}-${Date.now()}.sqlite3`,
);

function quote(value: string): string {
  if (process.platform === 'win32') return `"${value.replaceAll('"', '\\"')}"`;
  return `'${value.replaceAll("'", "'\\''")}'`;
}

const serviceCommand = [
  `pnpm --dir .. build && ${quote(python)}`,
  '../../tests/e2e_service_server.py',
  '--host', '127.0.0.1',
  '--port', String(servicePort),
  '--database', quote(database),
  '--api-key', 'test-browser-key',
  '--web-root', '../dist',
].join(' ');

export default defineConfig({
  testDir: '.',
  outputDir: './artifacts/playwright-service',
  fullyParallel: false,
  workers: 1,
  timeout: 30_000,
  expect: { timeout: 10_000 },
  reporter: 'list',
  use: {
    baseURL: `http://127.0.0.1:${frontendPort}`,
    locale: 'en-US',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  projects: [
    {
      name: 'service-desktop',
      testMatch: 'service-protocol.spec.ts',
      use: { baseURL: `http://127.0.0.1:${servicePort}`, browserName: 'chromium', viewport: { width: 1440, height: 1000 } },
    },
    {
      name: 'chat-component',
      testMatch: 'chat-component.spec.ts',
      use: {
        baseURL: `http://127.0.0.1:${harnessPort}`,
        browserName: 'chromium',
        viewport: { width: 1440, height: 1000 },
      },
    },
  ],
  webServer: [
    {
      command: serviceCommand,
      url: `http://127.0.0.1:${servicePort}/api/v1/runtime/status`,
      reuseExistingServer: false,
      timeout: 90_000,
    },
    {
      command: `pnpm --dir .. exec vite --config e2e-service/harness/vite.config.ts --host 127.0.0.1 --port ${harnessPort} --strictPort`,
      url: `http://127.0.0.1:${harnessPort}`,
      reuseExistingServer: false,
      timeout: 60_000,
    },
  ],
});
