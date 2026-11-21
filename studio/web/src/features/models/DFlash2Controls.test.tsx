import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import type { ModelArtifact, RuntimeInstance } from '../../shared/api/types';
import { DFlash2Controls } from './DFlash2Controls';

const model: ModelArtifact = {
  id: 'a'.repeat(32), name: 'Flash-Next', architecture: 'qwen4_exp_text', format: 'gguf',
  shard_count: 1, total_bytes: 1024, tensor_count: 1, record_count: 1, dtypes: ['BF16'],
  complete: true, loadable: true, modified_at: '',
};
const draft: ModelArtifact = {
  ...model, id: 'b'.repeat(32), name: 'dflash2', architecture: 'dflash2',
  format: 'hf', asset_kind: 'dflash2', dflash2_target: 'Qwen3.8-Flash-Next', loadable: false,
};
const instance: RuntimeInstance = {
  id: 'instance', model: model.name, devices: ['metal'], state: 'ready',
  active_sessions: 0, queued_requests: 0,
};

it('downloads an explicit draft and enables loading after it appears in the catalog', async () => {
  const download = vi.fn(async () => ({ id: 'download' }));
  const load = vi.fn(async () => undefined);
  const props = { model, backend: 'metal', tr: (_zh: string, en: string) => en, onDownload: download, onLoad: load };
  const unrelated = { ...draft, id: 'c'.repeat(32), name: 'Qwen3.8-27B-DFlash2', dflash2_target: 'Qwen3.8-27B' };
  const { rerender } = render(<DFlash2Controls {...props} artifacts={[model, unrelated]} />);
  expect(screen.getByText('Qwen3.8-Flash-Next · Draft')).toBeVisible();

  expect(screen.getByRole('button', { name: 'Load DFlash2 draft' })).toBeDisabled();
  fireEvent.click(screen.getByRole('button', { name: 'Download DFlash2 draft' }));
  await screen.findByText('Added to download queue');
  expect(download).toHaveBeenCalledOnce();

  rerender(<DFlash2Controls {...props} artifacts={[model, unrelated, draft]} />);
  fireEvent.click(screen.getByRole('button', { name: 'Load DFlash2 draft' }));
  await waitFor(() => expect(load).toHaveBeenCalledWith(draft.id));
});

it('confirms the reload of a running model and offers a separate switch after loading', async () => {
  const load = vi.fn(async () => null);
  const toggle = vi.fn(async () => null);
  const props = { model, artifacts: [model, draft], backend: 'metal', tr: (_zh: string, en: string) => en,
    onDownload: vi.fn(), onLoad: load, onToggle: toggle };
  const { rerender } = render(<DFlash2Controls {...props} instance={instance} />);
  fireEvent.click(screen.getByRole('button', { name: 'Load DFlash2 draft' }));
  expect(load).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button', { name: 'Reload and load' }));
  await waitFor(() => expect(load).toHaveBeenCalledWith(draft.id));
  rerender(<DFlash2Controls {...props} instance={{ ...instance, dflash2_draft: draft.name, dflash2_enabled: true }} />);
  const control = screen.getByRole('switch', { name: 'Enable DFlash2' });
  await waitFor(() => expect(control).not.toBeDisabled());
  fireEvent.click(control);
  expect(toggle).toHaveBeenCalledWith(false);
});
