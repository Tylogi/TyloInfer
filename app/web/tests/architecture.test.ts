
import { readFileSync, readdirSync } from 'node:fs';
import { dirname, isAbsolute, join, relative, resolve, sep } from 'node:path';
import { describe, expect, it } from 'vitest';

const root = resolve(import.meta.dirname, '../src');
function source(path: string) { return readFileSync(join(root, path), 'utf8'); }
function files(path: string): string[] {
  return readdirSync(path, { withFileTypes: true }).flatMap((entry) => entry.isDirectory() ? files(join(path, entry.name)) : /\.tsx?$/.test(entry.name) && !entry.name.includes('.test.') ? [join(path, entry.name)] : []);
}

const sharedRoot = resolve(import.meta.dirname, '../../shared');
const pluginRoot = resolve(import.meta.dirname, '../../plugins/src');

function insideRoot(path: string, directory: string): boolean {
  const delta = relative(resolve(directory), path);
  return delta !== '' && !isAbsolute(delta) && delta !== '..' && !delta.startsWith(`..${sep}`);
}

// 方向规则看解析后的目标，所以拼接多级 ../ 的相对路径不能绕过检查。
function importTargets(file: string): string[] {
  return [...readFileSync(file, 'utf8').matchAll(/(?:from|import)\s*['"](\.[^'"]+)['"]/g)]
    .map((match) => resolve(dirname(file), match[1]));
}

function directionViolations(directory: string, forbidden: string[]): string[] {
  const offenders = files(directory).filter((file) => importTargets(file).some((target) => forbidden.some((rootDirectory) => insideRoot(target, rootDirectory))));
  return offenders.map((file) => file.slice(directory.length));
}

describe('业务职责边界', () => {
  it('三个前端源码树只按允许方向依赖', () => {
    const rules: [string, string[]][] = [
      [root, [pluginRoot]],
      [sharedRoot, [root, pluginRoot]],
      [pluginRoot, [root]],
    ];
    for (const [directory, forbidden] of rules) {
      expect(directionViolations(directory, forbidden).sort(), directory).toEqual([]);
    }
  });
  it('依赖方向按解析目标判断，不按导入路径原文', () => {
    const sharedFile = join(sharedRoot, 'ui/Dialog.tsx');
    const detour = resolve(dirname(sharedFile), '../types/../../web/src/shared/ui/Dialog');
    expect(insideRoot(detour, root)).toBe(true);
    expect(insideRoot(detour, sharedRoot)).toBe(false);
    expect(insideRoot(resolve(dirname(sharedFile), './local/Dialog'), root)).toBe(false);
  });
  it('App仅组织Provider和路由，不拥有请求、业务状态或业务表单', () => {
    const app = source('App.tsx');
    expect(app).not.toMatch(/\buseState\b|\buseEffect\b|\bapi\.|\bfetch\(|<form\b|<input\b/);
    expect(app.split('\n').length).toBeLessThan(100);
    for (const route of ['models', 'runtime', 'resources', 'logs', 'model-hub']) {
      expect(app).toContain(`path="${route}"`);
    }
    expect(app).toContain('path="inference"');
    expect(app).toContain('<Navigate to="/inference" replace />');
    expect(app).not.toMatch(/AgentProvider|AgentWorkspace|EvaluationsPage/);
    expect(app).not.toContain('QuantizationPage');
    expect(app).not.toContain('ApplicationsPage');
    expect(app).not.toContain('SettingsRoute');
  });
  it('共享传输层不反向引用React、页面或业务状态', () => {
    for (const file of files(join(root, 'shared/api'))) {
      expect(readFileSync(file, 'utf8')).not.toMatch(/from ['"][^'"]*(?:features\/|App|react|zustand)/);
    }
  });
  it('业务模块不能依赖App，运行时启动不请求页面专属资源', () => {
    for (const file of files(join(root, 'features'))) expect(readFileSync(file, 'utf8')).not.toMatch(/from ['"][^'"]*\/App['"]/);
    const runtime = source('app/RuntimeProvider.tsx');
    for (const request of ['datasets', 'evaluations', 'modelArtifacts', 'runtimeLogs', 'generationPresets', 'artifactLineage']) {
      expect(runtime).not.toContain(`.${request}(`);
    }
  });
  it('业务模块按领域依赖资源 API，不依赖统一聚合入口', () => {
    for (const file of files(root)) {
      const relative = file.replace(`${root}${sep}`, '').replaceAll(sep, '/');
      expect(relative).not.toBe('api.ts');
      // The shared type barrel is a contract mirror, not the runtime aggregate.
      expect(readFileSync(file, 'utf8')).not.toMatch(/from ['"]\.\.?\/[^'"]*\/api['"]/);
      expect(readFileSync(file, 'utf8')).not.toMatch(/(?<![:/])\bapi\.[A-Za-z]/);
    }
  });
  it('全局样式入口只保留有序导入，各业务规则在独立样式中', () => {
    const styles = source('styles.css');
    expect(styles).not.toContain('{');
    expect(styles).toContain('features/chat/chat.css');
    expect(styles).toContain('shared/styles/overrides.css');
  });
});
