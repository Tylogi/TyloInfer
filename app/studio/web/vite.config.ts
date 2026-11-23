import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

// The frontend toolchain and version live in app/package.json, so the Studio
// target names its own root, assets and output instead of relying on the cwd.
const webRoot = new URL("./", import.meta.url);
const { version } = JSON.parse(readFileSync(new URL("../../package.json", webRoot), "utf8"));

function git(...args: string[]) {
  try { return execFileSync("git", args, { encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] }).trim(); }
  catch { return ""; }
}

const revision = git("rev-parse", "--short=8", "HEAD");
const modified = Boolean(git("status", "--porcelain", "--untracked-files=normal"));
const release = !modified && git("tag", "--points-at", "HEAD").split("\n").includes(`v${version}`);

export default defineConfig({
  root: fileURLToPath(webRoot),
  resolve: { alias: { '@shared': fileURLToPath(new URL('../../shared', webRoot)) } },
  plugins: [react()],
  define: {
    __TYLOINFER_STUDIO_BUILD__: JSON.stringify({
      version: release ? version : `${version}+dev.${revision || 'local'}${modified ? '.modified' : ''}`,
      release,
    }),
  },
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8090",
    },
  },
});
