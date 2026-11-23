import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import { ModelBrowserView, type ModelBrowserServices } from '@shared/features/models/ModelBrowserView';
import type { OfficialModelList } from '@shared/types/api';
import { useJobStore } from '../../stores/jobStore';

const tr = (_zh: string, en: string) => en;
const configuration = { status: 'unknown' as const, recommendation: 'unknown' as const, reasons: [] };
function catalog(refreshing = false): OfficialModelList {
  return { refreshing, system: { platform: 'test', machine: 'test', backend: 'unknown' }, data: ['Qwen3.5-9B', 'GLM-5.3-Flash'].map((name, index) => {
    const source = { provider: 'modelscope' as const, repo_id: `example/model-${index}`, revision: 'main', available: true, url: `https://modelscope.cn/models/example/model-${index}` };
    return { id: name, name, family: name, architecture: index ? 'glm' : 'qwen3', description: 'Cached catalog metadata', description_zh: '', modalities: ['text'], capabilities: [], precision_options: [], supports_ssd_streaming: false, sources: [source], selected_source: source, revision: 'main', downloads: 0, likes: 0, variants: [], configuration };
  }) };
}
function setup(data = catalog()) {
  const local = vi.fn().mockResolvedValue(data);
  const remote = vi.fn().mockResolvedValue(data);
  const info = vi.fn(); const probe = vi.fn();
  const services: ModelBrowserServices = { modelsApi: { localOfficialHubModels: local, officialHubModels: remote, hubModelInfo: info, probeDownloadSources: probe, browseHubModels: vi.fn(), resolveHubModel: vi.fn(), hubModelReadme: vi.fn() }, createJob: vi.fn(), openExternal: vi.fn(), useDownloadReuse: () => ({ checking: false, reusableFiles: {} }) };
  const props = { services, tab: 'official' as const, onTabChange: vi.fn(), jobKinds: [], onError: vi.fn(), onJobCreated: vi.fn(), tr, autoLoadCatalog: false, autoSearch: false, manualMetadataActions: true };
  return { local, remote, info, probe, props };
}
afterEach(() => { vi.useRealTimers(); useJobStore.getState().setJobs([]); });

it('automatically reads only the local official snapshot and preserves original model and vendor marks', async () => {
  const { props, local, remote, info, probe } = setup();
  const { container } = render(<ModelBrowserView {...props} />);
  await screen.findByRole('heading', { name: 'Qwen3.5-9B' });
  expect(local).toHaveBeenCalledTimes(1); expect(remote).not.toHaveBeenCalled(); expect(info).not.toHaveBeenCalled(); expect(probe).not.toHaveBeenCalled();
  const title = container.querySelector('.official-model-card-title')!;
  expect(title.firstElementChild).toHaveTextContent('Q');
  expect(title.querySelector('[data-model-vendor="qwen"]')).toBeInTheDocument();
});
it('rereads local metadata on manual page refresh without resetting selection or download concurrency', async () => {
  const { props, local, remote } = setup();
  const view = render(<ModelBrowserView {...props} actionRevision={0} />);
  await screen.findByRole('heading', { name: 'Qwen3.5-9B' });
  fireEvent.click(screen.getByRole('button', { name: /GLM-5.3-Flash/ }));
  fireEvent.change(screen.getByRole('combobox', { name: 'Maximum concurrent downloads' }), { target: { value: '2' } });
  view.rerender(<ModelBrowserView {...props} actionRevision={1} />);
  await waitFor(() => expect(local).toHaveBeenCalledTimes(2));
  expect(screen.getByRole('heading', { name: 'GLM-5.3-Flash' })).toBeInTheDocument();
  expect(screen.getByRole('combobox', { name: 'Maximum concurrent downloads' })).toHaveValue('2');
  expect(remote).not.toHaveBeenCalled();
});
it('waits for initial local state and issues only one snapshot read after a page refresh settles', async () => {
  const { props, local } = setup();
  const view = render(<ModelBrowserView {...props} initialLocalStateReady={false} actionRevision={0} />);
  await act(async () => {}); expect(local).not.toHaveBeenCalled();
  view.rerender(<ModelBrowserView {...props} initialLocalStateReady actionRevision={0} />);
  await waitFor(() => expect(local).toHaveBeenCalledTimes(1));
  view.rerender(<ModelBrowserView {...props} initialLocalStateReady={false} actionRevision={1} />);
  await act(async () => {}); expect(local).toHaveBeenCalledTimes(1);
  view.rerender(<ModelBrowserView {...props} initialLocalStateReady actionRevision={1} />);
  await waitFor(() => expect(local).toHaveBeenCalledTimes(2));
});
it('polls an already-refreshing local snapshot without falling through to network reads', async () => {
  vi.useFakeTimers();
  const { props, local, remote } = setup(catalog(true));
  local.mockResolvedValueOnce(catalog(true)).mockResolvedValue(catalog(false));
  const view = render(<ModelBrowserView {...props} />);
  await act(async () => {});
  await act(async () => { await vi.advanceTimersByTimeAsync(1000); });
  expect(local).toHaveBeenCalledTimes(2); expect(remote).not.toHaveBeenCalled();
  view.unmount();
  await act(async () => { await vi.advanceTimersByTimeAsync(5000); });
  expect(local).toHaveBeenCalledTimes(2);
});
it('rereads the local catalog on download state changes, not every progress sample', async () => {
  const { props, local, remote } = setup();
  render(<ModelBrowserView {...props} />);
  await screen.findByRole('heading', { name: 'Qwen3.5-9B' });
  const job = { id: 'background', kind: 'download.modelscope', status: 'running' as const, progress: 0, cancel_requested: false, payload: {}, created_at: '', updated_at: '' };
  await act(async () => useJobStore.getState().setJobs([job]));
  expect(local).toHaveBeenCalledTimes(2);
  await act(async () => useJobStore.getState().setJobs([{ ...job, progress: 0.5 }]));
  expect(local).toHaveBeenCalledTimes(2);
  await act(async () => useJobStore.getState().setJobs([{ ...job, status: 'succeeded', progress: 1 }]));
  expect(local).toHaveBeenCalledTimes(3); expect(remote).not.toHaveBeenCalled();
});
it('keeps an unsupported local snapshot as an error without a network fallback', async () => {
  const { props, local, remote } = setup();
  local.mockRejectedValue(new Error('Local snapshot unavailable'));
  render(<ModelBrowserView {...props} />);
  expect(await screen.findByRole('alert')).toHaveTextContent('Local snapshot unavailable');
  expect(remote).not.toHaveBeenCalled();
});
it('distinguishes an empty local cache from an online catalog and discloses explicit refresh networking', async () => {
  const { props, local, remote } = setup({ ...catalog(), data: [] });
  render(<ModelBrowserView {...props} />);
  await screen.findByText(/No recommended models are cached locally/);
  expect(local).toHaveBeenCalledTimes(1); expect(remote).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button', { name: 'Refresh' }));
  await waitFor(() => expect(remote).toHaveBeenCalledWith(true, expect.any(AbortSignal)));
});
