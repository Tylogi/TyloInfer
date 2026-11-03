import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { LogsPage } from './LogsPage';
import { jobsApi } from '../../shared/api/resources/jobs';
import { useJobStore } from '../../stores/jobStore';
import { studioConfirm } from '../../studio';
import type { JobResource, RuntimeInstance, RuntimeLogEntry, RuntimeStatus } from '../../shared/api/types';

const state = vi.hoisted(() => ({ ready: true, connectionRevision: 0, refreshRuntime: vi.fn().mockResolvedValue(true),
  refreshError: null as string | null,
  runtime: null as RuntimeStatus | null, artifactNames: {} as Record<string, string>,
  instances: [] as RuntimeInstance[] }));
vi.mock('../../app/RuntimeProvider', () => ({ useRuntime: () => state }));
vi.mock('../settings/SettingsProvider', () => ({ useSettings: () => ({ tr: (_zh: string, en: string) => en }) }));
vi.mock('../../studio', () => ({ studioConfirm: vi.fn().mockResolvedValue(true) }));

const stamp = () => new Date().toISOString();
function entry(sequence: number): RuntimeLogEntry {
  return { sequence, level: 'info', message: 'Weights ready', instance_id: 'a', fields: { source: 'runtime.stdout' }, created_at: stamp() };
}
function job(id = 'job-a'): JobResource {
  return { id, kind: 'model.load', status: 'succeeded', payload: { model: 'Qwen3.8' }, progress: 1,
    cancel_requested: false, result: { instance_id: 'a' }, created_at: stamp(), updated_at: stamp() };
}
beforeEach(() => {
  state.ready = true; state.connectionRevision = 0;
  state.runtime = null; state.artifactNames = {};
  state.refreshError = null; state.refreshRuntime.mockReset().mockResolvedValue(true);
  state.instances = [{ id: 'a', model: 'Qwen3.8', state: 'ready', devices: ['metal'], active_sessions: 0, queued_requests: 0 }];
  useJobStore.getState().setJobs([job()]);
  vi.spyOn(jobsApi, 'jobEvents').mockResolvedValue([entry(1)]);
  vi.mocked(studioConfirm).mockResolvedValue(true);
});
afterEach(() => { vi.restoreAllMocks(); vi.useRealTimers(); });

it('shows native job names and phases without a runtime history fetch', async () => {
  state.artifactNames = { 'artifact-a': 'Native Model' };
  useJobStore.getState().setJobs([{ ...job(), status: 'running', payload: { artifact_id: 'artifact-a' },
    progress: 0.9, progress_data: { phase: 'loading' } }]);
  render(<LogsPage />);
  expect(await screen.findByText('Native Model')).toBeInTheDocument();
  expect(screen.getByText('Loading weights')).toBeInTheDocument();
  expect(screen.getByRole('progressbar')).not.toHaveAttribute('value');
  expect(screen.queryByText('90%')).not.toBeInTheDocument();
});

it('lists job records newest first and expands their events', async () => {
  render(<LogsPage />);
  expect(await screen.findByRole('button', { name: /Load model/ })).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: /Load model/ }));
  await screen.findByText('Weights ready');
});

it('shows task events and deletes only the selected completed record after confirmation', async () => {
  const remove = vi.spyOn(jobsApi, 'deleteJob').mockResolvedValue(undefined);
  render(<LogsPage />);
  await screen.findByRole('button', { name: /Load model/ });
  fireEvent.click(screen.getByRole('button', { name: /Load model/ }));
  await screen.findByText('Weights ready');
  fireEvent.click(screen.getByRole('button', { name: 'Delete this record' }));
  await waitFor(() => expect(remove).toHaveBeenCalledWith('job-a'));
  expect(studioConfirm).toHaveBeenCalled();
  expect(useJobStore.getState().jobs).toHaveLength(0);
});

it('retains expanded job events when a later event query fails', async () => {
  render(<LogsPage />);
  await screen.findByRole('button', { name: /Load model/ });
  fireEvent.click(screen.getByRole('button', { name: /Load model/ }));
  await screen.findByText('Weights ready');
  let reject!: (error: Error) => void;
  vi.mocked(jobsApi.jobEvents).mockReturnValueOnce(new Promise((_resolve, fail) => { reject = fail; }));
  await act(async () => { useJobStore.getState().setJobs([{ ...job(), updated_at: new Date(Date.now() + 1000).toISOString() }]); });
  await act(async () => { reject(new Error('job events offline')); });
  expect(screen.getByText('Weights ready')).toBeInTheDocument();
  expect(screen.getByRole('status')).toHaveTextContent('job events offline');
});

it('pauses updates, keeps the snapshot frozen and refreshes manually', async () => {
  render(<LogsPage />);
  await screen.findByRole('button', { name: /Load model/ });
  fireEvent.click(screen.getByRole('button', { name: 'Pause updates' }));
  expect(screen.getByText('Paused')).toBeInTheDocument();
  vi.mocked(jobsApi.jobEvents).mockResolvedValue([entry(1), { ...entry(2), message: 'New task output' }]);
  fireEvent.click(screen.getByRole('button', { name: 'Refresh' }));
  await act(async () => {});
  fireEvent.click(screen.getByRole('button', { name: /Load model/ }));
  await screen.findByText('New task output');
  expect(screen.getByText('Paused')).toBeInTheDocument();
});

it('filters job records by status and search text', async () => {
  useJobStore.getState().setJobs([job(), { ...job('job-b'), kind: 'model.unload', status: 'failed' }]);
  render(<LogsPage />);
  await screen.findByRole('button', { name: /Load model/ });
  expect(screen.getByRole('button', { name: /Unload model/ })).toBeInTheDocument();
  fireEvent.change(screen.getByRole('combobox', { name: /Status/ }), { target: { value: 'failed' } });
  expect(screen.queryByRole('button', { name: /Load model/ })).not.toBeInTheDocument();
  expect(screen.getByRole('button', { name: /Unload model/ })).toBeInTheDocument();
  fireEvent.change(screen.getByRole('combobox', { name: /Status/ }), { target: { value: '' } });
  fireEvent.change(screen.getByRole('searchbox'), { target: { value: 'unload' } });
  expect(screen.queryByRole('button', { name: /Load model/ })).not.toBeInTheDocument();
});
