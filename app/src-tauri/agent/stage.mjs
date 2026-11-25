import { spawnSync } from 'node:child_process';
import { chmodSync, closeSync, copyFileSync, existsSync, mkdtempSync, openSync, readdirSync, readSync, renameSync, rmSync, statSync } from 'node:fs';
import { basename, dirname, join, resolve } from 'node:path';

const here = import.meta.dirname;
const app = resolve(here, '../..');
const root = resolve(app, '..');
const windows = process.platform === 'win32';
const platform = windows ? 'windows' : process.platform;
const arch = { arm64: 'aarch64', x64: 'x86_64' }[process.arch];
const [major, minor] = process.versions.node.split('.').map(Number);
if (major !== 24 || minor < 9) throw new Error('Build with the official Node.js 24 distribution (24.9 or newer).');
if ((process.env.TAURI_ENV_PLATFORM ?? platform) !== platform || (process.env.TAURI_ENV_ARCH ?? arch) !== arch ||
    (platform === 'darwin' && process.arch !== 'arm64') || (windows && process.arch !== 'x64')) {
  throw new Error('Stage the Agent with native Node on the target architecture (Apple Silicon or Windows x64).');
}
const nodeDir = dirname(process.execPath);
const nodeLicense = [join(nodeDir, 'LICENSE'), join(nodeDir, '..', 'LICENSE')].find(existsSync);
if (!nodeLicense) throw new Error('Use the official Node.js archive with its LICENSE alongside node.exe or above bin/node.');

function run(command, args, cwd) {
  // Windows pnpm is a .cmd file; only fixed commands and shell-safe relative arguments use cmd.
  const cmd = windows && command === 'pnpm';
  const result = spawnSync(cmd ? 'cmd.exe' : command, cmd ? ['/d', '/c', command, ...args] : args, { cwd, stdio: 'inherit' });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`${command} failed (exit ${result.status ?? result.signal}).`);
}

function signMacRuntime(stage, node) {
  const identity = process.env.APPLE_SIGNING_IDENTITY;
  if (platform !== 'darwin' || !identity || identity === '-') return;
  const sign = (path, options = []) => run('codesign', ['--force', '--sign', identity, '--options', 'runtime', '--timestamp', ...options, path], stage);
  const machO = new Set([0xfeedface, 0xfeedfacf, 0xcefaedfe, 0xcffaedfe, 0xcafebabe, 0xcafebabf, 0xbebafeca, 0xbfbafeca]);
  const magic = Buffer.alloc(4);
  function signNative(dir) {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      const path = join(dir, entry.name);
      // Keep upstream-signed helpers (including their entitlements and TCC identity) intact.
      if (entry.isDirectory() && !entry.name.endsWith('.app')) signNative(path);
      if (!entry.isFile() || (!/\.(node|dylib|so)$/.test(entry.name) && !(statSync(path).mode & 0o111))) continue;
      const fd = openSync(path, 'r');
      try {
        if (readSync(fd, magic, 0, 4, 0) === 4 && machO.has(magic.readUInt32BE())) sign(path);
      } finally {
        closeSync(fd);
      }
    }
  }
  signNative(join(stage, 'node_modules'));
  sign(node, ['--entitlements', join(here, 'node-entitlements.plist')]);
}

const stage = mkdtempSync(join(here, '.stage-'));
try {
  run('pnpm', ['run', 'build:plugins'], app);
  // Deploy the app's locked production graph as real directories for Tauri's resource copier.
  run('pnpm', ['--filter', '@tyloinfer/app', '--config.node-linker=hoisted',
    '--config.package-import-method=copy', 'deploy', '--prod', '--ignore-scripts',
    `src-tauri/agent/${basename(stage)}`], app);
  // The app boots DSH through its Node entry point, not the package manager's bin links.
  rmSync(join(stage, 'node_modules', '.bin'), { recursive: true, force: true });

  const node = join(stage, windows ? 'node.exe' : 'node');
  copyFileSync(process.execPath, node);
  if (!windows) chmodSync(node, 0o755);
  copyFileSync(nodeLicense, join(stage, 'LICENSE.node'));
  for (const file of ['LICENSE', 'NOTICE']) copyFileSync(join(root, file), join(stage, file));
  copyFileSync(join(here, 'boot.mjs'), join(stage, 'boot.mjs'));
  run(node, ['node_modules/@deepseek-ai/dsh-subprocess-local/scripts/ensure-spawn-helper.mjs'], stage);
  run(node, ['boot.mjs'], stage);
  signMacRuntime(stage, node);
  const runtime = join(here, 'runtime');
  rmSync(runtime, { recursive: true, force: true });
  renameSync(stage, runtime);
} finally {
  rmSync(stage, { recursive: true, force: true });
}
