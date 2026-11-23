import { useStudioSharedPresentation } from '../../../tests/shared-presentation';
import { useState } from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, expect, it, vi } from 'vitest';
import { ModelLoadContext } from './ModelLoadContext';
import { DEFAULT_LOAD_CONTEXT, readLoadContext, saveLoadContext } from './loadContextDefaults';
import { setApiBaseUrl } from '../../shared/api/client';
import { modelsApi } from '../../shared/api/resources/models';
import type { ModelArtifact } from '@shared/types/api';

vi.mock('../settings/SettingsProvider', () => ({ useSettings: () => ({ tr: (zh: string) => zh }) }));
useStudioSharedPresentation((zh: string) => zh);

const model: ModelArtifact = { id: 'fixture', name: 'Dense model', architecture: 'qwen3_5', format: 'gguf',
  shard_count: 1, total_bytes: 2 ** 30, tensor_count: 1, record_count: 1, dtypes: [],
  complete: true, loadable: true, modified_at: '2026-01-01T00:00:00Z' };

beforeEach(() => {
  vi.restoreAllMocks(); localStorage.clear(); setApiBaseUrl('');
  vi.spyOn(modelsApi, 'modelLoadPlan').mockImplementation(async (_id, query) => {
    const cap = !query.preallocate ? 262144 : (30 - query.reserve) * 2048;
    return { context_size: query.context_size || cap, max_context: cap, precision: query.precision || 8,
      precisions: [8, 4], usable: (query.context_size || cap) <= cap,
      reason: (query.context_size || cap) > cap ? 'memory_limit' : null };
  });
});

function Form() {
  const [value, setValue] = useState(readLoadContext);
  return <ModelLoadContext models={[model]} value={value} onChange={next => { setValue(next); saveLoadContext(next); }} />;
}

it('puts the computed default in the field and recalculates it for another tier', async () => {
  render(<Form />);
  const context = screen.getByRole('spinbutton', { name: '上下文长度' });
  await waitFor(() => expect(context).toHaveValue(40960));
  expect(screen.getByRole('radio', { name: /普通/ })).toBeChecked();
  expect(screen.getByRole('option', { name: '自动 · INT8' })).toHaveProperty('selected', true);
  fireEvent.click(screen.getByRole('radio', { name: /安全/ }));
  await waitFor(() => expect(context).toHaveValue(20480));
  expect(readLoadContext()).toMatchObject({ context_size: null, reserve_percent: 20 });
});

it('preserves a manual context, applies the memory limit only with preallocation, and resets to auto', async () => {
  render(<Form />);
  const context = screen.getByRole('spinbutton');
  await waitFor(() => expect(context).toHaveValue(40960));
  fireEvent.change(context, { target: { value: '131072' } });
  expect(await screen.findByRole('alert')).toHaveTextContent('超出可用容量');
  expect(screen.getByRole('combobox', { name: 'KV 分配' })).toHaveValue('preallocate');
  fireEvent.change(screen.getByRole('combobox', { name: 'KV 分配' }), { target: { value: 'dynamic' } });
  await waitFor(() => expect(screen.queryByRole('alert')).not.toBeInTheDocument());
  fireEvent.change(screen.getByRole('combobox', { name: 'KV 精度' }), { target: { value: '4' } });
  expect(readLoadContext()).toMatchObject({ context_size: 131072, precision: 4, vram_preallocated: false });
  fireEvent.click(screen.getByRole('button', { name: '恢复自动' }));
  await waitFor(() => expect(context).toHaveValue(262144));
  expect(readLoadContext().context_size).toBeNull();
});

it('isolates stored defaults by server and model', () => {
  setApiBaseUrl('http://127.0.0.1:8090');
  saveLoadContext({ ...DEFAULT_LOAD_CONTEXT, reserve_percent: 5, context_size: 8192 }, model.name);
  expect(readLoadContext(model.name)).toMatchObject({ reserve_percent: 5, context_size: 8192 });
  expect(readLoadContext()).toEqual(DEFAULT_LOAD_CONTEXT);
  setApiBaseUrl('http://127.0.0.1:8092');
  expect(readLoadContext(model.name)).toEqual(DEFAULT_LOAD_CONTEXT);
});

it.each([
  [new Error('Fixture connection failure'), '无法读取加载方案'],
  [{ status: 404, code: 'http_404' }, '当前核心版本缺少加载方案接口'],
])('keeps context editing and KV allocation independent when the load plan fails', async (error, message) => {
  vi.mocked(modelsApi.modelLoadPlan).mockRejectedValueOnce(error);
  render(<Form />);
  expect(await screen.findByRole('alert')).toHaveTextContent(message);
  const allocation = screen.getByRole('combobox', { name: 'KV 分配' });
  expect(allocation).toHaveValue('preallocate');
  const context = screen.getByRole('spinbutton');
  fireEvent.change(context, { target: { value: '8192' } });
  fireEvent.blur(context);
  expect(allocation).toHaveValue('preallocate');
  expect(readLoadContext()).toMatchObject({ context_size: 8192, vram_preallocated: true });
  await waitFor(() => expect(screen.getByText(/上限/)).toBeInTheDocument());
});
