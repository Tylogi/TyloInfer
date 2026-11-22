
import { readFileSync, readdirSync } from 'node:fs';
import { join, resolve, sep } from 'node:path';
import { describe, expect, it } from 'vitest';

const root = resolve('src');
function files(path: string): string[] {
  return readdirSync(path, { withFileTypes: true }).flatMap((entry) => entry.isDirectory() ? files(join(path, entry.name)) : /\.tsx?$/.test(entry.name) && !entry.name.includes('.test.') ? [join(path, entry.name)] : []);
}

describe('业务职责边界', () => {
  it('共享传输层不反向引用React、页面或业务状态', () => {
    for (const file of files(join(root, 'shared/api'))) {
      expect(readFileSync(file, 'utf8')).not.toMatch(/from ['"][^'"]*(?:features\/|App|react|zustand)/);
    }
  });
  it('业务模块不能反向依赖App', () => {
    for (const file of files(join(root, 'features'))) expect(readFileSync(file, 'utf8')).not.toMatch(/from ['"][^'"]*\/App['"]/);
  });
  it('业务模块按领域依赖资源 API，不依赖统一聚合入口', () => {
    for (const file of files(root)) {
      const relative = file.replace(`${root}${sep}`, '').replaceAll(sep, '/');
      expect(relative).not.toBe('api.ts');
      expect(readFileSync(file, 'utf8')).not.toMatch(/from ['"][^'"]*\/api['"]/);
    }
  });
});
