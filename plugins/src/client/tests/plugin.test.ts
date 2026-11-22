import { readFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import test from 'node:test';
const require = createRequire(new URL('../../../../studio/web/package.json', import.meta.url));
const React = require('react');
const { renderToStaticMarkup } = require('react-dom/server');
const artifact = await readFile(new URL('../../../lib/client.js', import.meta.url), 'utf8');
function harness(sessionStorage?: any, source: string = artifact) {
  let definition: any;
  const imports: any[] = [];
  const styles = new Set<any>();
  const classes = new Set<any>();
  const icons = new Set<any>();
  const document = { querySelectorAll: () => [], createElement: (tag: any) => ({ dataset: {}, remove() { styles.delete(this); icons.delete(this); }, tag }), head: { append: (item: any) => item.tag === 'link' ? icons.add(item) : styles.add(item) },
    body: { classList: { add: (value: any) => classes.add(value), remove: (value: any) => classes.delete(value) } } };
  vm.runInNewContext(source, { document, AbortController, setTimeout, clearTimeout, window: { sessionStorage, __ModuleLoader__: { load: (value: any) => { definition = value; } } } });
  const plugin = definition.factory((name: any) => {
    imports.push(name);
    assert.ok(name === 'react' || name === 'react-dom');
    return name === 'react' ? React : require(name);
  });
  const registrations: any[] = [];
  const disposers: any[] = [];
  const selected: any[] = [];
  let tokens = false;
  const ctx = {
    effect: (callback: any) => { const dispose = callback(); if (typeof dispose === 'function') disposers.push(dispose); },
    on: () => () => {},
    locale: { register: () => () => {} },
    theme: { overrideTokens: () => { tokens = true; return () => { tokens = false; }; }, getTheme: () => ({ preference: 'system' }), setTheme() {} },
    connection: { rpc: { call: async () => ({ ok: true, value: { data: [] } }) } },
    modules: { manifest: { release: { version: '9.8.7-test' } } },
    layout: { selectPanel: (key: any) => selected.push(key) },
    slots: { inject: (_name: any, callback: any) => { const dispose = callback(); if (dispose) disposers.push(dispose); }, register: (options: any, Component: any) => {
      const row = { options, Component }; registrations.push(row); return () => registrations.splice(registrations.indexOf(row), 1);
    } },
  };
  plugin.apply(ctx);
  return { definition, imports, registrations, selected, styles, classes, icons, tokens: () => tokens, dispose() { for (const dispose of disposers.reverse()) dispose();  } };
}
test('DSBot build adds a native panel without replacing the existing shell', { skip: process.env.TYLOINFER_TEST_DSBOT !== '1' }, async () => {
  const output = new URL('../../../dist/dsbot/', import.meta.url);
  const manifest = JSON.parse(await readFile(new URL('package.json', output), 'utf8'));
  const patch = await readFile(new URL(manifest.dsh.bundle.patch, output), 'utf8');
  assert.match(patch, /name: '@tyloinfer\/dsh-plugin'/);
  assert.doesNotMatch(patch, /agent-default-model|connectionFile/);
  await readFile(new URL(manifest.exports['./bridge'], output), 'utf8');
  const source = await readFile(new URL(manifest.exports['./client'], output), 'utf8');
  const scope = harness(undefined, source);
  try {
    assert.equal(scope.definition.id, manifest.name);
    assert.deepEqual(scope.imports, ['react', 'react-dom']);
    assert.deepEqual(scope.registrations.map(row => row.options.name), ['sidebar.footer.action', 'main']);
    const [access, panel] = scope.registrations;
    assert.equal(access.options.id, panel.options.key);
    const content = renderToStaticMarkup(React.createElement(access.Component, {
      wide: true, t: (key: any) => key, useNavSelection: (select: any) => select('overview'),
      usePanelInfo: (select: any) => select({ activePanelId: null }), ...access.options.inject(),
    }));
    assert.match(content, /tylo-studio-engine/);
    assert.match(content, /overview.*models.*service.*resources.*downloads.*analysis.*logs/);
    assert.doesNotMatch(content, /tylo-studio-settings/);
    assert.equal(scope.classes.size, 0);
    assert.equal(scope.icons.size, 0);
    assert.equal(scope.tokens(), false);
  } finally { scope.dispose(); }
});
test('uses a lazy Host factory with external React and preserves the native root and conversation', () => {
  const scope = harness();
  try {
    assert.equal(scope.definition.id, '@tyloinfer/dsh-plugin');
    assert.deepEqual(scope.imports, ['react']);
    assert.deepEqual(scope.registrations.map((row: any) => row.options.name), ['sidebar.brand.mark', 'sidebar.brand.name', 'conversation.hero.brand.mark', 'shell.overlay', 'settings.launcher', 'main', 'shell.overlay']);
    assert.equal(scope.registrations.find((row: any) => row.options.name === 'main').options.key, 'tyloinfer.inference');
    assert.equal(scope.registrations.find((row: any) => row.options.name === 'settings.launcher').options.priority, -100);
    // The unminified artifact carries module path comments, so match the calls
    // that would mean a bundled React runtime, not the word "react-dom".
    assert.ok(!/createRoot|startRun|createSession/.test(artifact));
    assert.equal(scope.styles.size, 1);
    assert.equal(scope.classes.has('tylo-studio-product'), true);
    assert.equal(scope.tokens(), true);
  } finally { scope.dispose(); }
  assert.equal(scope.styles.size, 0);
  assert.equal(scope.icons.size, 0);
  assert.equal(scope.classes.size, 0);
  assert.equal(scope.registrations.length, 0);
  assert.equal(scope.tokens(), false);
});
test('uses product artwork and a sidebar name without replacing native navigation', () => {
  const scope = harness();
  try {
    const mark = scope.registrations.find((row: any) => row.options.name === 'sidebar.brand.mark');
    const name = scope.registrations.find((row: any) => row.options.name === 'sidebar.brand.name');
    assert.equal(mark.options.priority, -100);
    assert.equal(name.options.priority, -100);
    assert.match(renderToStaticMarkup(React.createElement(mark.Component, { size: 24 })), /data:image\/svg\+xml/);
    assert.match(renderToStaticMarkup(React.createElement(name.Component)), />TyloInfer</);
    assert.equal(scope.icons.size, 1);
    assert.match([...scope.styles][0].textContent, /data-slot=['"]conversation\.hero\.brand\.mark['"]/);
    assert.ok(!scope.registrations.some((row: any) => ['root', 'sidebar', 'main.conversation', 'conversation.composer'].includes(row.options.name)));
  } finally { scope.dispose(); }
});
test('scopes settings presentation without replacing native settings or configuration consumers', () => {
  const scope = harness();
  try {
    const css = [...scope.styles][0].textContent;
    assert.match(css, /\[data-shortcut-modal=(?:['"])?settings(?:['"])?\]/);
    assert.ok(!scope.registrations.some((row: any) => ['sidebar.settings', 'settings.section', 'settings.general.item'].includes(row.options.name)));
    assert.ok(!/settings\/(?:update|mutate|replace)/.test(artifact));
  } finally { scope.dispose(); }
});
test('renders inventory through the native selector hook contract', () => {
  const scope = harness();
  try {
    const row = scope.registrations.find((row: any) => row.options.name === 'main');
    const state = { inventory: { models: [], loadedModels: [], instances: [] }, loading: false };
    const markup = renderToStaticMarkup(React.createElement(row.Component, { t: (key: any) => key,
      useSelection: (selector: any) => selector('models'), useModelState: (selector: any) => selector(state), ...row.options.inject() }));
    assert.match(markup, /Local checkpoints/);
    assert.match(markup, /No local models yet/);
    assert.match(markup, /model-library-summary/);
    assert.match(markup, /screen-header/);
    assert.doesNotMatch(markup, /Track an existing operation|Operation ID|tylo-model-restore/);
    assert.doesNotMatch(markup, /tylo-model-registration/);
  } finally { scope.dispose(); }
});
test('reloading the model page unlocks controls left by a legacy untracked write', () => {
  const legacy = JSON.stringify({ version: 1, id: null, kind: null, unknown: true, label: '' });
  const scope = harness({ getItem: (key: any) => key === 'tyloinfer.page-operation.v1' ? legacy : null });
  try {
    const row = scope.registrations.find((item: any) => item.options.name === 'main');
    const model = { id: 'a'.repeat(32), name: 'Fixture', architecture: 'qwen35', format: 'gguf',
      complete: true, loadable: true, total_bytes: 1024, shard_count: 1 };
    const state = { inventory: { models: [model], loadedModels: [], instances: [] }, loading: false };
    const markup = renderToStaticMarkup(React.createElement(row.Component, { t: (key: any) => key,
      useSelection: (selector: any) => selector('models'), useModelState: (selector: any) => selector(state), ...row.options.inject() }));
    assert.doesNotMatch(markup, /Outcome unknown/);
    for (const action of ['load', 'settings']) {
      const button = markup.match(new RegExp('<button class="model-' + action + '-action(?: [^"]*)?"[^>]*>'))?.[0];
      assert.ok(button, action);
      assert.doesNotMatch(button, /\sdisabled/);
    }
  } finally { scope.dispose(); }
});
test('reuses Studio presentation without importing its standalone transport or renderer', () => {
  const scope = harness();
  try {
    const css = [...scope.styles][0].textContent;
    for (const selector of ['.metric-grid', '.overview-memory-tiers', '.model-library-summary', '.configuration-tabs', '.resource-monitor-grid', '.logs-console', '.model-browser-layout', '.analysis-layout']) assert.ok(css.includes('.tylo-studio-inference ' + selector), selector);
    assert.equal((artifact.match(/(?:var|const|let) PresenterContext =/g) || []).length, 1);
    assert.ok(!/fetch\(|getApiToken|setApiBaseUrl|\/api\/v1\//.test(artifact));
    assert.match(artifact, /function AnalysisPanels/);
    assert.match(artifact, /function RuntimeHeroView/);
    assert.match(artifact, /function DownloadQueueView/);
    assert.match(artifact, /function LocalCheckpoints/);
    assert.doesNotMatch(css, /MFQ Small Numerals|unicode-range/);
    assert.match(css, /lining-nums tabular-nums/);
    assert.match(css, /--control-font-size: 12px/);
    assert.match(css, /text-align: end/);
  } finally { scope.dispose(); }
});
test('keeps all seven restored page structures inside the native main slot', () => {
  const scope = harness();
  try {
    const row = scope.registrations.find((row: any) => row.options.name === 'main');
    const expected = { overview: 'overview-footer-grid', models: 'model-library-summary', service: 'configuration-tabs', resources: 'resource-budget-summary', downloads: 'model-browser-layout', analysis: 'analysis-checkpoint-bar', logs: 'logs-console' };
    for (const [view, marker] of Object.entries(expected)) {
      const state = { inventory: { models: [], loadedModels: [], instances: [] }, loading: false };
      const markup = renderToStaticMarkup(React.createElement(row.Component, { t: (key: any) => key, useSelection: (selector: any) => selector(view), useModelState: (selector: any) => selector(state), ...row.options.inject() }));
      assert.ok(markup.includes(marker), view);
      assert.ok(markup.includes('screen-header'), view);
    }
  } finally { scope.dispose(); }
});
test('places inline default-expanded inference options above a separate native settings launcher', () => {
  const scope = harness();
  try {
    const row = scope.registrations.find((row: any) => row.options.name === 'settings.launcher');
    const markup = renderToStaticMarkup(React.createElement(row.Component, { wide: true, settingsOpen: false, openSettings() {}, t: (key: any) => key, useNavSelection: (selector: any) => selector('overview'), usePanelInfo: (selector: any) => selector({ activePanelId: 'tyloinfer.inference' }), ...row.options.inject() }));
    assert.match(markup, /aria-expanded="true"/);
    assert.ok(markup.indexOf('<nav') < markup.indexOf('tylo-studio-engine'));
    assert.ok(markup.indexOf('tylo-studio-engine') < markup.indexOf('tylo-studio-settings'));
    assert.match(markup, /aria-haspopup="dialog"/);
    assert.match(markup, /TyloInfer <!-- -->|TyloInfer /);
    assert.match(markup, /based on DSH 9\.8\.7-test/);
    assert.ok(markup.indexOf('tylo-studio-settings') < markup.indexOf('tylo-studio-version'));
    row.options.inject().showInference('models');
    assert.deepEqual(scope.selected, ['tyloinfer.inference']);
    scope.registrations.find((row: any) => row.options.name === 'main').options.inject().backToChat();
    assert.deepEqual(scope.selected, ['tyloinfer.inference', null]);
  } finally { scope.dispose(); }
});
