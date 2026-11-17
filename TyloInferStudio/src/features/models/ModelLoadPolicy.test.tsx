import { fireEvent, render, screen } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import { ModelLoadPolicy } from './ModelLoadPolicy';
import { DEFAULT_LOAD_CONTEXT } from './loadContextDefaults';
import type { useModelCatalog } from './useModelCatalog';

vi.mock('../settings/SettingsProvider', () => ({ useSettings: () => ({ tr: (_zh: string, en: string) => en }) }));

it('offers explicit full VRAM placement for Jev and keeps ordinary model settings unchanged', () => {
  const setLoadContext = vi.fn();
  const catalog = { runtime: { backend: 'cuda' }, artifacts: [{ name: 'decision', structured_decisions: true }],
    settingsModel: 'decision', loadContext: DEFAULT_LOAD_CONTEXT, setLoadContext,
    loadPinned: false, setLoadPinned: vi.fn(), loadIdleTtl: null, setLoadIdleTtl: vi.fn(),
  } as unknown as ReturnType<typeof useModelCatalog>;
  const { rerender } = render(<ModelLoadPolicy catalog={catalog} />);
  const setting = screen.getByRole('switch', { name: 'Load all Jev weights into VRAM' });
  expect(setting).not.toBeChecked();
  fireEvent.click(setting);
  expect(setLoadContext).toHaveBeenCalledWith({ ...DEFAULT_LOAD_CONTEXT, jev_weights: 'vram' });
  rerender(<ModelLoadPolicy catalog={{ ...catalog, loadContext: { ...DEFAULT_LOAD_CONTEXT, jev_weights: 'vram' } }} />);
  expect(screen.getByRole('switch', { name: 'Load all Jev weights into VRAM' })).toBeChecked();
  rerender(<ModelLoadPolicy catalog={{ ...catalog, settingsModel: 'ordinary' }} />);
  expect(screen.queryByRole('switch', { name: 'Load all Jev weights into VRAM' })).not.toBeInTheDocument();
});
