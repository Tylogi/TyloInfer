import { readFileSync, statSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { initProfile, loadLayeredEnv, loadProfileDirectory, resolveProfileDir } from '@deepseek-ai/dsh-app-boot';
import { runProfile } from '@deepseek-ai/dsh/profile-boot';

const base = '@deepseek-ai/dsh-base';
const web = '@deepseek-ai/dsh-web-app';
const plugin = '@tyloinfer/dsh-plugin';
const manifest = name => JSON.parse(readFileSync(new URL(`./node_modules/${name}/package.json`, import.meta.url), 'utf8'));
const dsh = manifest('@deepseek-ai/dsh');
const bundle = manifest(plugin);
if ([manifest(base).version, manifest(web).version, bundle.engines.dsh].some(version => version !== dsh.version)) {
  throw new Error('The bundled DSH, Web and TyloInfer plugin versions are incompatible.');
}
for (const path of [bundle.exports['.'].default, bundle.exports['./client'], bundle.dsh.bundle.patch]) {
  if (!statSync(new URL(`./node_modules/${plugin}/${path}`, import.meta.url)).isFile()) throw new Error('Incomplete TyloInfer plugin.');
}

// With no arguments the package builder checks the payload without creating user state.
if (process.argv[2] === 'launch') {
  if (!process.env.DSH_HOME) throw new Error('The Agent must have a dedicated DSH_HOME.');
  const name = 'tyloinfer-agent';
  const dir = resolveProfileDir(name);
  const bundles = [base, web, plugin];
  initProfile(dir, bundles);
  // User state lives outside the installation; the resolver must start at the app's dependency graph.
  const installAnchor = fileURLToPath(new URL('package.json', import.meta.url));
  const profile = loadProfileDirectory('dsh', dir, installAnchor);
  if (bundles.some(bundle => !profile.layers.some(layer => layer.packageName === bundle))) {
    throw new Error('The Agent profile is missing a bundled package.');
  }
  process.env.TYLOINFER_DSH_VERSION = dsh.version;

  // EOF works on Windows too and reaches the same graceful shutdown path as SIGTERM.
  process.stdin.once('end', () => { if (!process.emit('SIGTERM', 'SIGTERM')) process.exit(0); });
  process.stdin.resume();
  process.stdin.unref?.();
  const { ctx, shutdown } = await runProfile({
    environment: loadLayeredEnv('dsh'),
    profile: name,
    resolvedProfile: { profile, installAnchor },
    args: ['--host', '127.0.0.1', '--port', '0', '--no-open'],
    patchFiles: [],
  });
  const webServer = ctx.get('webServer');
  const connection = ctx.get('connection');
  if (!webServer || !connection) {
    await shutdown.shutdown(1);
    throw new Error('The Agent profile did not start the Web Host.');
  }
  process.stdout.write(`tyloinfer-agent ready: ${connection.authenticatedUrl(webServer.getOrigin())}\n`);
}
