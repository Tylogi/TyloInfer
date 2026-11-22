import { act, fireEvent, render, screen } from '@testing-library/react';
import { MemoryRouter, useLocation } from 'react-router';
import { beforeEach, expect, it, vi } from 'vitest';
import type { RuntimeStatus } from '../../shared/api/contracts/runtime';
import type { JobResource, RuntimeInstance } from '../../shared/api/types';
import { useJobStore } from '../../stores/jobStore';
import { RuntimeHero } from './RuntimeHero';
import { getApiBaseUrl, setApiBaseUrl } from '../../shared/api/client';

const state = vi.hoisted(() => ({ runtime: null as RuntimeStatus | null, instances: [] as RuntimeInstance[], studio: null, chinese: false }));
vi.mock('../../app/RuntimeProvider', () => ({ useRuntime: () => state }));
vi.mock('../settings/SettingsProvider', () => ({ useSettings: () => ({ tr: (zh: string, en: string) => state.chinese ? zh : en }) }));

function Location() {
  const location = useLocation();
  return <output data-testid="location">{location.pathname}{location.hash}</output>;
}

beforeEach(() => { state.runtime = null; state.instances = []; state.chinese = false; useJobStore.getState().setJobs([]); });

it.each([false, true])('opens local checkpoints from the empty-model Load action (Chinese: %s)', chinese => {
  state.chinese = chinese;
  render(<MemoryRouter><RuntimeHero /><Location /></MemoryRouter>);
  expect(screen.queryByRole('button', { name: chinese ? '模型' : 'Models' })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: chinese ? '加载' : 'Load' }));
  expect(screen.getByTestId('location')).toHaveTextContent('/models#local-checkpoints');
});

it('uses the serving Studio origin rather than a guessed port when no remote service is configured', () => {
  const previous = getApiBaseUrl();
  setApiBaseUrl('');
  try {
    const { container } = render(<MemoryRouter><RuntimeHero /></MemoryRouter>);
    expect(container.querySelector('.runtime-endpoint')).toHaveTextContent(`${window.location.origin}/v1`);
    expect(container.querySelector('.runtime-endpoint')).not.toHaveTextContent(':8090/v1');
  } finally { setApiBaseUrl(previous); }
});

it('retains the normal Models action when a model is running', () => {
  state.runtime = { model: 'model-a', runtime_state: 'ready' };
  render(<MemoryRouter><RuntimeHero /><Location /></MemoryRouter>);
  expect(screen.getByText('Running')).toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Load' })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Models' }));
  expect(screen.getByTestId('location').textContent).toBe('/models');
});

it('shows a failed instance without presenting it as a running model', () => {
  state.instances = [{ id: 'failed-load', model: 'broken-model', state: 'failed',
    devices: [], active_sessions: 0, queued_requests: 0 }];
  render(<MemoryRouter><RuntimeHero /></MemoryRouter>);
  expect(screen.getByRole('heading', { name: 'broken-model' })).toBeInTheDocument();
  expect(screen.getByText('Failed')).toBeInTheDocument();
  expect(screen.queryByText('Running')).not.toBeInTheDocument();
});

it('shows an interrupted load after runtime cleanup and lets a new retry show its progress', () => {
  const interrupted = { id: 'old-load', kind: 'model.load', status: 'interrupted', progress: .2,
    cancel_requested: false, payload: { model: 'fixture-model' },
    created_at: '2026-01-01', updated_at: '2026-01-01' } as JobResource;
  useJobStore.getState().setJobs([interrupted]);
  render(<MemoryRouter><RuntimeHero /></MemoryRouter>);
  expect(screen.getByText('Interrupted')).toBeInTheDocument();
  expect(screen.queryByText('Loading')).not.toBeInTheDocument();
  expect(screen.queryByRole('progressbar')).not.toBeInTheDocument();
  act(() => useJobStore.getState().setJobs([{ ...interrupted, id: 'new-load', status: 'running', progress: .25 }, interrupted]));
  expect(screen.getByText('Loading')).toBeInTheDocument();
  expect(screen.getByRole('progressbar')).toHaveAttribute('value', '0.25');
  expect(screen.queryByText('Interrupted')).not.toBeInTheDocument();
});
