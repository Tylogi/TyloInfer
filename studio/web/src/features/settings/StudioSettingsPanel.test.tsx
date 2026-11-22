import { fireEvent, render, screen } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import { SettingsProvider } from './SettingsProvider';
import { StudioSettingsPanel } from './StudioSettingsPanel';
import { DEFAULT_SETTINGS, SETTINGS_KEY, loadSettings } from './configuration';
import { normalizeFontFamily } from './fontPreferences';

beforeEach(() => {
  localStorage.clear();
  localStorage.setItem(SETTINGS_KEY, JSON.stringify({ ...DEFAULT_SETTINGS, language: 'en' }));
});

it('applies and persists local font preferences while preserving generation settings', () => {
  render(<SettingsProvider><StudioSettingsPanel onOpenVersions={vi.fn()} /></SettingsProvider>);
  fireEvent.change(screen.getByRole('slider', { name: 'Text size' }), { target: { value: '18' } });
  expect(document.documentElement.style.getPropertyValue('--studio-content-font-size')).toBe('18px');
  fireEvent.click(screen.getByText('More font settings'));
  fireEvent.change(screen.getByRole('textbox', { name: 'Text font' }), { target: { value: 'Example Font, serif' } });
  fireEvent.change(screen.getByRole('textbox', { name: 'Code font' }), { target: { value: 'monospace' } });
  fireEvent.change(screen.getByRole('slider', { name: 'Code size' }), { target: { value: '15' } });
  expect(document.documentElement.style.fontFamily).toContain('Example Font');
  expect(document.documentElement.style.getPropertyValue('--studio-code-font-size')).toBe('15px');
  expect(document.documentElement.style.getPropertyValue('--studio-code-font-family')).toBe('monospace, monospace');
  expect(loadSettings()).toMatchObject({ fontSize: 18, codeFontSize: 15, temperature: DEFAULT_SETTINGS.temperature });
  expect(screen.getByRole('spinbutton', { name: 'Terminal size' })).toBeDisabled();
  expect(normalizeFontFamily('"Example Font", serif, <bad>\\font')).toBe('"Example Font", serif, "badfont"');
});

it('identifies the independent inference Studio without implying an Agent dependency', () => {
  render(<SettingsProvider><StudioSettingsPanel onOpenVersions={vi.fn()} /></SettingsProvider>);
  fireEvent.click(screen.getByText('Chat display'));
  expect(screen.getByRole('combobox', { name: 'Transcript view' })).toBeDisabled();
  expect(screen.getByRole('combobox', { name: 'Enter while busy' })).toBeDisabled();
  for (const name of ['Account', 'Models', 'Agent presets', 'Plugins']) {
    fireEvent.click(screen.getByRole('button', { name }));
    expect(screen.getByText(/TyloInfer Studio is a standalone inference interface/)).toBeVisible();
    expect(screen.getByRole('status')).toHaveTextContent('This section belongs to the optional Agent product');
    expect(screen.queryByText(/transitional interface/)).toBeNull();
    expect(screen.queryByText(/DSH configuration services are not connected/)).toBeNull();
    expect(screen.queryByRole('button', { name: /save/i })).toBeNull();
    expect(screen.queryByRole('textbox')).toBeNull();
  }
});

it('keeps standalone display preferences usable offline and restores them without Agent APIs', () => {
  const network = vi.fn(() => Promise.reject(new Error('Network must not be used for display settings')));
  vi.stubGlobal('fetch', network);
  try {
    const view = render(<SettingsProvider><StudioSettingsPanel onOpenVersions={vi.fn()} /></SettingsProvider>);
    expect(screen.getByText(/without connecting to the inference core or Agent/)).toBeVisible();
    fireEvent.change(screen.getByRole('combobox', { name: 'Appearance' }), { target: { value: 'dark' } });
    fireEvent.change(screen.getByRole('slider', { name: 'Text size' }), { target: { value: '19' } });
    expect(network).not.toHaveBeenCalled();
    view.unmount();
    render(<SettingsProvider><StudioSettingsPanel onOpenVersions={vi.fn()} /></SettingsProvider>);
    expect(screen.getByRole('combobox', { name: 'Appearance' })).toHaveValue('dark');
    expect(screen.getByRole('slider', { name: 'Text size' })).toHaveValue('19');
    expect(network).not.toHaveBeenCalled();
  } finally { vi.unstubAllGlobals(); }
});

it('reports local storage failure without claiming a successful save', () => {
  const write = vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new DOMException('quota', 'QuotaExceededError'); });
  try {
    render(<SettingsProvider><StudioSettingsPanel onOpenVersions={vi.fn()} /></SettingsProvider>);
    expect(screen.getByRole('alert')).toHaveTextContent('storage is unavailable');
    fireEvent.change(screen.getByRole('slider', { name: 'Text size' }), { target: { value: '17' } });
    expect(document.documentElement.style.getPropertyValue('--studio-content-font-size')).toBe('17px');
  } finally { write.mockRestore(); }
});
