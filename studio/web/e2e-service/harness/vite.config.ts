import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const harnessRoot = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  root: harnessRoot,
  plugins: [react()],
  server: {
    proxy: {
      '/api': `http://127.0.0.1:${process.env.TYLOINFER_SERVICE_PORT ?? 5190}`,
      '/__test': `http://127.0.0.1:${process.env.TYLOINFER_SERVICE_PORT ?? 5190}`,
    },
  },
});
