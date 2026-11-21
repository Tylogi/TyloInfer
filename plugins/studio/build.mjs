import { build } from '../../studio/web/node_modules/vite/dist/node/index.js';
import { copyFile, mkdir, readFile, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
const root = fileURLToPath(new URL('.', import.meta.url));
const dsbot = process.argv.includes('--dsbot');
const output = new URL(dsbot ? './dist/dsbot/' : './', import.meta.url);
const versionSource = await readFile(new URL('../../studio/_version.py', import.meta.url), 'utf8');
const productVersion = versionSource.match(/^__version__\s*=\s*["']([^"']+)["']/m)?.[1];
if (!productVersion) throw new Error('TyloInfer product version is missing');
const { default: postcss } = await import('../../studio/web/node_modules/postcss/lib/postcss.mjs');
const stylePaths = ['shared/styles/tokens.css', 'shared/styles/base.css', 'shared/styles/components.css', 'shared/styles/lists.css', 'shared/styles/dialogs.css', 'shared/ui/primitives.css', 'features/models/model-workbench.css', 'shared/styles/configuration-layout.css', 'features/models/model-load-context.css', 'features/models/model-delete.css', 'features/runtime/runtime.css', 'features/runtime/serving.css', 'features/runtime/cache.css', 'features/runtime/logs.css', 'features/connections/connections.css', 'features/connections/service-layout.css', 'features/models/model-browser.css', 'features/models/community-model-browser.css', 'features/analysis/analysis.css', 'app/model-vendor.css', 'shared/styles/responsive.css', 'shared/styles/selects.css', 'shared/styles/numeric-values.css', 'shared/styles/typography.css'];
const scopedStyles = [];
for (const path of stylePaths) {
  const tree = postcss.parse(await readFile(new URL('../../studio/web/src/' + path, import.meta.url), 'utf8'));
  tree.walkAtRules('import', rule => rule.remove());
  tree.walkRules(rule => {
    if (rule.parent.type === 'atrule' && /keyframes$/.test(rule.parent.name)) return;
    rule.selectors = rule.selectors.map(selector => selector.includes(':root')
      ? selector.replaceAll(':root', '.tylo-studio-inference').replaceAll('data-theme', 'data-studio-theme')
      : `.tylo-studio-inference ${selector}`);
  });
  scopedStyles.push(tree.toString());
}
const studioRoot = fileURLToPath(new URL('../../studio/web/src/', import.meta.url)).replaceAll('\\', '/');
const sharedContext = fileURLToPath(new URL('./src/shared-presenters.jsx', import.meta.url)).replaceAll('\\', '/');
const primitives = fileURLToPath(new URL('./src/shared-primitives.jsx', import.meta.url)).replaceAll('\\', '/');
const result = await build({ configFile: false, root, logLevel: 'error', plugins: [{
  name: 'tyloinfer-shared-studio-presenters', enforce: 'pre',
  resolveId(source, importer) {
    if (source === 'tyloinfer:studio-styles') return '\0tyloinfer:studio-styles';
    if (source.endsWith('.css') && importer?.replaceAll('\\', '/').startsWith(studioRoot)) return '\0tyloinfer:empty-style';
    if (!importer?.replaceAll('\\', '/').startsWith(studioRoot)) return;
    if (/(?:SettingsProvider|stores\/jobStore|loadContextDefaults)$/.test(source)) return sharedContext;
    if (/shared\/ui\/(?:Dialog|Tooltip|Switch)$/.test(source)) return primitives;
    if (/shared\/api\/(?:client|resources\/)|RuntimeProvider|react-router|stores\/toastStore/.test(source)) throw new Error(`Shared presenter imports a standalone service: ${source}`);
  },
  load(id) {
    if (id === '\0tyloinfer:studio-styles') return `export default ${JSON.stringify(scopedStyles.join('\n'))};`;
    if (id === '\0tyloinfer:empty-style') return 'export default "";';
  },
  transform(code, id) {
    const path = id.replaceAll('\\', '/');
    if (path.startsWith(studioRoot) && /\.tsx$/.test(path) && !/import\s+React\b/.test(code)) return `import React from 'react';\n${code}`;
  },
}], define: { __TYLOINFER_PRODUCT_VERSION__: JSON.stringify(productVersion), __TYLOINFER_DSBOT__: JSON.stringify(dsbot) }, oxc: { jsx: { runtime: 'classic' } },
  build: { write: false, minify: false, lib: { entry: 'src/client.jsx', name: 'TyloInferStudioClient', formats: ['iife'] },
    rollupOptions: { external: ['react'], output: { globals: { react: 'React' } } } } });
const outputs = Array.isArray(result) ? result : [result];
const chunks = outputs.flatMap((item) => item.output).filter((item) => item.type === 'chunk');
if (chunks.length !== 1) throw new Error('Expected one self-contained Client factory');
await mkdir(new URL('lib/', output), { recursive: true });
const portal = dsbot ? "const createDSBotPortal = require('react-dom').createPortal;\n" : '';
await writeFile(new URL('lib/client.js', output), `window.__ModuleLoader__.load({ id: '@tyloinfer/dsh-studio', factory(require) {\nconst React = require('react');\n${portal}${chunks[0].code}\nreturn TyloInferStudioClient;\n} });\n`);
await copyFile(new URL('./src/index.js', import.meta.url), new URL('lib/index.js', output));

if (dsbot) {
  const manifest = JSON.parse(await readFile(new URL('./package.json', import.meta.url), 'utf8'));
  const host = new URL('../host/', import.meta.url);
  const hostManifest = JSON.parse(await readFile(new URL('package.json', host), 'utf8'));
  if (manifest.engines.dsh !== hostManifest.engines.dsh) throw new Error('DSBot plugin components require different DSH versions');
  const hostOutput = new URL('lib/host/', output);
  await mkdir(hostOutput, { recursive: true });
  for (const filename of ['package.json', ...hostManifest.files.filter(filename => filename.endsWith('.js'))]) {
    await copyFile(new URL(filename, host), new URL(filename, hostOutput));
  }
  const { name, version, type, engines, exports, dsh, peerDependencies } = manifest;
  const packageManifest = {
    name, version, type, engines, dsh,
    description: 'TyloInfer inference controls for DSBot',
    license: hostManifest.license,
    exports: { ...exports, './bridge': './lib/host/index.js' },
    peerDependencies: { ...peerDependencies, ...hostManifest.peerDependencies, 'react-dom': peerDependencies.react },
    files: ['lib', 'cordis.patch.yml'],
  };
  await copyFile(new URL('dsbot.patch.yml', import.meta.url), new URL('cordis.patch.yml', output));
  await writeFile(new URL('package.json', output), JSON.stringify(packageManifest, null, 2) + '\n');
}
console.log('CLIENT_BUILD_OK external_react=true factory_count=1');
