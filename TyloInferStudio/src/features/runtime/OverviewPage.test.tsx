import { act, fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import type { RuntimeStatus, RuntimeInstance } from '../../shared/api/contracts/runtime';
import { OverviewPage } from './OverviewPage';

const state = vi.hoisted(() => ({ runtime: null as RuntimeStatus | null, models: [], instances: [] as RuntimeInstance[],
  connectionRevision: 0, refreshRuntime: vi.fn() }));
vi.mock('../../app/RuntimeProvider', () => ({ useRuntime: () => state }));
vi.mock('../settings/SettingsProvider', () => ({ useSettings: () => ({ tr: (_zh: string, en: string) => en,
  settings: { theme: 'system', language: 'system' }, updateSettings: vi.fn() }) }));
vi.mock('./RuntimeHero', () => ({ RuntimeHero: () => null }));
vi.mock('./MemoryHierarchy', () => ({ MemoryHierarchy: () => null }));

beforeEach(() => { state.runtime = null; state.models = []; state.instances = []; });
afterEach(() => { vi.restoreAllMocks(); });

it('places the combined appearance control beside Refresh in the overview header', async () => {
  await act(async () => { render(<OverviewPage />); });
  const controls = screen.getByRole('group', { name: 'Appearance and language' });
  expect(controls.closest('.screen-header-trailing')).toContainElement(screen.getByRole('button', { name: 'Refresh' }));
  expect(controls.querySelectorAll('.appearance-controls-divider')).toHaveLength(1);
});

it('uses placeholders without statistics, not unsupported or old request speed', async () => {
  state.runtime = { model: 'model', last_request: { decode_tps: 987.6 }, prefix_cache_supported: 0 };
  await act(async () => { render(<OverviewPage />); });
  expect(screen.getByText('Prefix reuse').closest('.metric-tile')).toHaveTextContent('--');
  expect(screen.queryByText('987.6')).not.toBeInTheDocument();
  expect(screen.queryByText('Live performance')).not.toBeInTheDocument();
  expect(screen.queryByText('Processed tokens / total prefill time')).not.toBeInTheDocument();
  expect(screen.queryByText('Generated tokens / total decode time')).not.toBeInTheDocument();
  expect(screen.getByText('Concurrency').closest('.metric-tile')).toHaveTextContent('--');
});

it('shows native aggregate statistics as unavailable and uses its shared Anthropic endpoint', async () => {
  state.runtime = { source: 'native-instances', active_requests: 2 };
  await act(async () => { render(<OverviewPage />); });
  expect(screen.getByText(/does not provide aggregate statistics/)).toBeInTheDocument();
  expect(screen.getByText(`${window.location.origin}/v1/messages`)).toBeInTheDocument();
  expect(screen.getByText('Prefill').closest('.metric-tile')).toHaveTextContent('--');
});

it('renders multiple native instances with reported backend and request counts', async () => {
  state.runtime = { source: 'native-instances' };
  state.instances = ['First', 'Second'].map((model, index) => ({ id: `native-${index}`, model,
    state: 'ready', devices: [], backend: 'cuda', active_requests: index, context_size: 8192 }));
  const { container } = render(<OverviewPage />);
  await act(async () => {});
  expect(screen.getByRole('button', { name: /First/ })).toHaveTextContent('cuda · 8,192 ctx · 0 active requests');
  expect(screen.getByRole('button', { name: /Second/ })).toHaveTextContent('cuda · 8,192 ctx · 1 active requests');
  expect(container).not.toHaveTextContent('undefined sessions');
});
