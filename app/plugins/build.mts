import { build } from 'vite';
import postcss from 'postcss';
import { copyFile, cp, mkdir, readFile, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import manifest from './package.json' with { type: 'json' };

const root = fileURLToPath(new URL('.', import.meta.url));
const dsbot = process.argv.includes('--dsbot');
const output = new URL(dsbot ? './dist/dsbot/' : './', import.meta.url);
const clientEntry = fileURLToPath(new URL('./src/client/client.tsx', import.meta.url));
const sharedSource = new URL('../shared/', import.meta.url);
const versionSource = await readFile(new URL('../server/_version.py', import.meta.url), 'utf8');
const productVersion = versionSource.match(/^__version__\s*=\s*["']([^"']+)["']/m)?.[1];
if (!productVersion) throw new Error('TyloInfer product version is missing');
// The Client shows the Studio presentation inside the host shell, so the shared
// styles are scoped to one class instead of leaking onto the host document.
const stylePaths = ['styles/tokens.css', 'styles/base.css', 'styles/components.css', 'styles/lists.css', 'styles/dialogs.css', 'ui/primitives.css', 'features/models/model-workbench.css', 'styles/configuration-layout.css', 'features/models/model-load-context.css', 'features/models/model-delete.css', 'features/runtime/runtime.css', 'features/runtime/serving.css', 'features/runtime/cache.css', 'features/runtime/logs.css', 'features/connections/connections.css', 'features/connections/service-layout.css', 'features/models/model-browser.css', 'features/models/community-model-browser.css', 'features/analysis/analysis.css', 'styles/model-vendor.css', 'styles/responsive.css', 'styles/selects.css', 'styles/numeric-values.css', 'styles/typography.css'];
const scopedStyles = [];
for (const path of stylePaths) {
  const tree = postcss.parse(await readFile(new URL(path, sharedSource), 'utf8'));
  tree.walkAtRules('import', rule => rule.remove());
  tree.walkRules(rule => {
    if (rule.parent.type === 'atrule' && /keyframes$/.test(rule.parent.name)) return;
    rule.selectors = rule.selectors.map(selector => selector.includes(':root')
      ? selector.replaceAll(':root', '.tylo-studio-inference').replaceAll('data-theme', 'data-studio-theme')
      : `.tylo-studio-inference ${selector}`);
  });
  scopedStyles.push(tree.toString());
}
const sharedRoot = fileURLToPath(sharedSource).replaceAll('\\', '/');
const result = await build({ configFile: false, root, logLevel: 'error', plugins: [{
  name: 'tyloinfer-plugin-client-bundle', enforce: 'pre',
  async resolveId(source, importer) {
    if (source === 'tyloinfer:studio-styles') return '\0tyloinfer:studio-styles';
    if (!importer?.replaceAll('\\', '/').startsWith(sharedRoot)) return;
    if (source.endsWith('.css')) return '\0tyloinfer:empty-style';
    if (/(?:SettingsProvider|stores\/jobStore|stores\/toastStore|RuntimeProvider|react-router)|\/api\/(?:client|resources\/)/.test(source)) throw new Error(`Shared presentation imports an application service: ${source}`);
    if (/^[@\w][^:]*$/.test(source) && !source.startsWith('@shared/')) {
      // Shared presentation uses this workspace's dependencies, not Studio's installation.
      return await this.resolve(source, clientEntry, { skipSelf: true }) ?? this.error(`Missing plugin dependency: ${source}`);
    }
  },
  load(id) {
    if (id === '\0tyloinfer:studio-styles') return `export default ${JSON.stringify(scopedStyles.join('\n'))};`;
    if (id === '\0tyloinfer:empty-style') return 'export default "";';
  },
  transform(code, id) {
    const path = id.replaceAll('\\', '/');
    if (path.startsWith(sharedRoot) && /\.tsx$/.test(path) && !/import\s+React\b/.test(code)) return `import React from 'react';\n${code}`;
  },
}], define: { __TYLOINFER_PRODUCT_VERSION__: JSON.stringify(productVersion), __TYLOINFER_PLUGIN_NAME__: JSON.stringify(manifest.name), __TYLOINFER_DSBOT__: JSON.stringify(dsbot) }, oxc: { jsx: { runtime: 'classic' } },
  resolve: { alias: { '@shared': fileURLToPath(new URL('../shared', import.meta.url)) } },
  build: { write: false, minify: false, lib: { entry: clientEntry, name: 'TyloInferClient', formats: ['iife'] },
    rollupOptions: { external: ['react'], output: { globals: { react: 'React' } } } } });
const outputs = Array.isArray(result) ? result : [result];
const chunks = outputs.flatMap((item) => item.output).filter((item) => item.type === 'chunk');
if (chunks.length !== 1) throw new Error('Expected one self-contained Client factory');
await mkdir(new URL('lib/', output), { recursive: true });
const portal = dsbot ? "const createDSBotPortal = require('react-dom').createPortal;\n" : '';
await writeFile(new URL(manifest.exports['./client'], output), `window.__ModuleLoader__.load({ id: ${JSON.stringify(manifest.name)}, factory(require) {\nconst React = require('react');\n${portal}${chunks[0].code}\nreturn TyloInferClient;\n} });\n`);

if (dsbot) {
  // The Host half is already compiled to lib/host/ by `pnpm build:plugins`; the DSBot bundle carries it.
  await cp(new URL('./lib/host/', import.meta.url), new URL('lib/host/', output), { recursive: true });
  const { name, version, type, engines, exports, dsh } = manifest;
  const packageManifest = {
    name, version, type, engines, dsh,
    description: 'TyloInfer inference controls for DSBot',
    license: manifest.license,
    exports: { ...exports, './bridge': './lib/host/index.js' },
    peerDependencies: { react: manifest.devDependencies.react, 'react-dom': manifest.devDependencies.react },
    files: ['lib', 'cordis.patch.yml'],
  };
  await copyFile(new URL('dsbot.patch.yml', import.meta.url), new URL('cordis.patch.yml', output));
  await writeFile(new URL('package.json', output), JSON.stringify(packageManifest, null, 2) + '\n');
}
console.log('CLIENT_BUILD_OK external_react=true factory_count=1');
