
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vitest/config';
import { fileURLToPath } from 'node:url';

export default defineConfig({
  root: fileURLToPath(new URL('./', import.meta.url)),
  resolve: { alias: { '@shared': fileURLToPath(new URL('../shared', import.meta.url)) } },
  plugins: [react()],
  test: {
    environment: 'jsdom',
    restoreMocks: true,
    unstubGlobals: true,
    projects: [
      {
        extends: true,
        test: {
          name: 'web',
          setupFiles: ['./tests/setup.ts'],
          include: ['src/**/*.test.{ts,tsx}', 'tests/**/*.test.{ts,tsx}'],
        },
      },
      {
        extends: true,
        root: fileURLToPath(new URL('../shared', import.meta.url)),
        test: {
          name: 'shared',
          setupFiles: ['./tests/setup.ts'],
          include: ['**/*.test.{ts,tsx}'],
        },
      },
    ],
  },
});
