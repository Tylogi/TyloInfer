import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { expect, it } from 'vitest';

const read = (path: string) => readFileSync(resolve(import.meta.dirname, '..', path), 'utf8');

it('uses the TyloInfer mark across the desktop and web shell', () => {
  const mark = read('../shared/assets/tylogi-ai-lab-mark.svg');
  expect(mark).toContain('aria-label="Tylogi AI Lab"');
  expect(mark).toContain('M48 48H86M67 48V77C67 84 71.5 88 79 88H84');
  expect(mark).not.toContain('M16 44V20L32 37 48 20V44');
  expect(read('index.html')).toContain('../shared/assets/tylogi-ai-lab-mark.svg');
  expect(read('src/app/StudioSidebar.tsx')).toMatch(/>TyloInfer<\//);
  expect(read('../shared/ui/display.tsx')).toContain('assets/tylogi-ai-lab-mark.svg');
  for (const component of ['StudioSidebar', 'LoadingPage', 'NotFoundPage', 'DesktopWindowFrame', 'LoginPage', 'FailurePage']) {
    const source = read(`src/app/${component}.tsx`);
    expect(source).toContain('assets/tylogi-ai-lab-mark.svg');
    expect(source).not.toContain('mfq-mark.svg');
  }
});

it('uses TyloInfer in context-management descriptions and exported filenames', () => {
  const context = read('../shared/features/runtime/ModelContextSettingsView.tsx');
  expect(context).toContain('留空由 TyloInfer 管理');
  expect(context).not.toContain('由 MFQ 管理');
  expect(read('src/features/runtime/LogsPage.tsx')).toContain('`tyloinfer-${tab}-');
});
