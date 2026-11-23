import { render, screen, within } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import type { ModelArtifact } from '@shared/types/api';
import type { useModelCatalog } from './useModelCatalog';
import { LocalCheckpoints } from '@shared/features/models/LocalCheckpoints';
import userEvent from '@testing-library/user-event';
import { DEFAULT_LOAD_CONTEXT } from './loadContextDefaults';

vi.mock('../settings/SettingsProvider', () => ({ useSettings: () => ({ tr: (_zh: string, en: string) => en }) }));
vi.mock('@shared/features/models/ModelLoadProgress', () => ({ ModelLoadProgress: () => null }));

it('shows streamed Jev placement and updates pressure when full VRAM is selected', () => {
  const GiB = 2 ** 30;
  const artifact = { id: 'decision', name: 'd3-turbo', architecture: 'qwen35', format: 'hf',
    shard_count: 1, total_bytes: 20 * GiB, estimated_resident_weight_bytes: 20 * GiB,
    jev_weight_memory: { stream_vram_bytes: GiB / 2, stream_ram_bytes: 16 * GiB, resident_vram_bytes: 16 * GiB },
    structured_decisions: true, tensor_count: 1, record_count: 1, dtypes: [], complete: true,
    loadable: true, modified_at: '' } as ModelArtifact;
  const catalog = { runtime: { memory_architecture: 'discrete', device_memory_total_bytes: 24 * GiB,
    host_memory_total_bytes: 64 * GiB, device_free_bytes: 16 * GiB,
    host_memory_available_bytes: 50 * GiB }, artifacts: [artifact], filteredArtifacts: [artifact], instances: [],
    busy: false, modelFilter: '', setModelFilter: vi.fn(), unloadInstance: vi.fn(), loadArtifact: vi.fn(),
    chooseModelDirectory: vi.fn(), openModelFiles: vi.fn(), loadContextForModel: () => DEFAULT_LOAD_CONTEXT,
  } as unknown as ReturnType<typeof useModelCatalog>;
  const { rerender } = render(<LocalCheckpoints catalog={catalog} />);
  expect(screen.getByText('Estimated resident VRAM', { exact: false })).toHaveTextContent('0.5 GiB · RAM-streamed weights 16 GiB');
  expect(screen.getByRole('progressbar', { name: 'VRAM + RAM pressure: 25.0%' })).toBeInTheDocument();
  rerender(<LocalCheckpoints catalog={{ ...catalog,
    loadContextForModel: () => ({ ...DEFAULT_LOAD_CONTEXT, jev_weights: 'vram' }) }} />);
  expect(screen.getByText('Estimated resident VRAM', { exact: false })).toHaveTextContent('16 GiB');
  expect(screen.queryByText('RAM-streamed weights', { exact: false })).toBeNull();
  expect(screen.getByRole('progressbar', { name: 'VRAM pressure: 100.0%' })).toBeInTheDocument();
});

it('passes an explicit external Engram path only for the selected GGUF model', async () => {
  const artifact = { id: 'xq', name: 'XQ', architecture: 'deepseek_v41', format: 'gguf', shard_count: 16,
    gguf_companion_required: true,
    total_bytes: 100, tensor_count: 1, record_count: 1, dtypes: [], complete: true, loadable: true, modified_at: '' } as ModelArtifact;
  const loadArtifact = vi.fn();
  const catalog = { runtime: null, artifacts: [artifact], filteredArtifacts: [artifact], instances: [], busy: false,
    modelFilter: '', setModelFilter: vi.fn(), unloadInstance: vi.fn(), loadArtifact,
    chooseModelDirectory: vi.fn(), openModelFiles: vi.fn() } as unknown as ReturnType<typeof useModelCatalog>;
  const view = render(<LocalCheckpoints catalog={catalog} />);
  expect(screen.getByRole('button', { name: 'Load' })).toBeDisabled();
  await userEvent.click(screen.getByText('Engram companion'));
  expect(screen.getByPlaceholderText('Server-side GGUF/TQ/MFQ table file or Engram folder')).toBeInTheDocument();
  await userEvent.type(screen.getByRole('textbox', { name: 'Engram path: XQ' }), '/models/engram');
  await userEvent.click(screen.getByRole('button', { name: 'Load' }));
  expect(loadArtifact).toHaveBeenCalledExactlyOnceWith('XQ', ['/models/engram']);
  catalog.connectionRevision = 1;
  view.rerender(<LocalCheckpoints catalog={catalog} />);
  expect(screen.getByRole('textbox', { name: 'Engram path: XQ' })).toHaveValue('');
  expect(screen.getByRole('button', { name: 'Load' })).toBeDisabled();
});

it('selects only complete registered GGUF companion identities when embedded in the Agent product', async () => {
  const artifact = { id: 'a'.repeat(32), name: 'Main GGUF', architecture: 'deepseek_v41', format: 'gguf', shard_count: 1,
    gguf_companion_required: true, total_bytes: 100, complete: true, loadable: true, modified_at: '' } as ModelArtifact;
  const table = { ...artifact, id: 'b'.repeat(32), name: 'Registered table', asset_kind: 'engram', gguf_companion_required: false, loadable: false } as ModelArtifact;
  const incomplete = { ...table, id: 'c'.repeat(32), name: 'Incomplete table', complete: false };
  const loadArtifact = vi.fn();
  const catalog = { runtime: null, artifacts: [artifact, table, incomplete], filteredArtifacts: [artifact], instances: [], busy: false,
    modelFilter: '', setModelFilter: vi.fn(), unloadInstance: vi.fn(), loadArtifact, chooseModelDirectory: vi.fn(), openModelFiles: vi.fn() } as unknown as ReturnType<typeof useModelCatalog>;
  const view = render(<LocalCheckpoints catalog={catalog} companionMode="registered-id" />);
  expect(screen.getByRole('button', { name: 'Load' })).toBeDisabled();
  await userEvent.click(screen.getByText('Engram companion'));
  const picker = screen.getByRole('combobox', { name: 'Engram companion: Main GGUF' });
  expect(within(picker).queryByRole('option', { name: 'Incomplete table' })).toBeNull();
  expect(screen.queryByRole('textbox', { name: /Engram path/ })).toBeNull();
  await userEvent.selectOptions(picker, table.id);
  await userEvent.click(screen.getByRole('button', { name: 'Load' }));
  expect(loadArtifact).toHaveBeenCalledExactlyOnceWith('Main GGUF', [table.id]);
  catalog.connectionRevision = 1;
  view.rerender(<LocalCheckpoints catalog={catalog} companionMode="registered-id" />);
  expect(picker).toHaveValue('');
  expect(screen.getByRole('button', { name: 'Load' })).toBeDisabled();
});

it('uses the instance list for unload, reload and deletion while the summary is stale', async () => {
  const artifact = { id: 'a', name: 'Model', architecture: 'qwen35', format: 'tq', shard_count: 1,
    total_bytes: 100, tensor_count: 1, record_count: 1, dtypes: [], complete: true, loadable: true, modified_at: '' } as ModelArtifact;
  const prepareModelDeletion = vi.fn();
  const catalog = { runtime: null, artifacts: [artifact], filteredArtifacts: [artifact], instances: [], busy: false,
    modelFilter: '', setModelFilter: vi.fn(), unloadInstance: vi.fn(), loadArtifact: vi.fn(),
    chooseModelDirectory: vi.fn(), openModelFiles: vi.fn(), prepareModelDeletion } as unknown as ReturnType<typeof useModelCatalog>;
  const view = render(<LocalCheckpoints catalog={catalog} />);
  expect(screen.getByRole('option', { name: 'TQ' })).toHaveValue('tq');
  expect(screen.queryByRole('option', { name: 'MFQ' })).not.toBeInTheDocument();
  await userEvent.click(screen.getByRole('button', { name: 'Delete model: Model' }));
  expect(prepareModelDeletion).toHaveBeenCalledExactlyOnceWith('a');
  catalog.runtime = { model: 'Model' };
  catalog.instances = [{ id: 'instance', model: 'Model', state: 'ready' } as typeof catalog.instances[number]];
  view.rerender(<LocalCheckpoints catalog={catalog} />);
  expect(screen.getByRole('button', { name: 'Delete model: Model' })).toBeDisabled();
  await userEvent.click(screen.getByRole('button', { name: 'Unload' }));
  expect(catalog.unloadInstance).toHaveBeenCalledExactlyOnceWith('instance');

  catalog.instances = [];
  view.rerender(<LocalCheckpoints catalog={catalog} />);
  const load = screen.getByRole('button', { name: 'Load' });
  expect(load).toBeEnabled();
  await userEvent.click(load);
  expect(catalog.loadArtifact).toHaveBeenCalledExactlyOnceWith('Model');
  expect(screen.getByRole('button', { name: 'Delete model: Model' })).toBeEnabled();
});

it('updates the combined MoE pressure and SSD warning from the same live memory sample', () => {
  const GiB = 2 ** 30;
  const artifact: ModelArtifact = { id: 'a', name: 'MoE', architecture: 'qwen4_exp', format: 'tq',
    shard_count: 1, total_bytes: 133 * GiB, estimated_resident_weight_bytes: 83 * GiB, ssd_ple_bytes: 50 * GiB,
    estimated_weight_bytes_by_role: { dense: 2 * GiB, experts: 80 * GiB, embedding: GiB },
    tensor_count: 1, record_count: 1, dtypes: [], complete: true, loadable: true, modified_at: '2026-10-11' };
  const catalog = { runtime: { memory_architecture: 'discrete', device_memory_total_bytes: 24 * GiB,
    host_memory_total_bytes: 64 * GiB, host_memory_available_bytes: 56 * GiB, device_free_bytes: 16 * GiB },
    artifacts: [artifact], filteredArtifacts: [artifact], instances: [], busy: false,
    modelFilter: '', setModelFilter: vi.fn(), unloadInstance: vi.fn(), loadArtifact: vi.fn(),
    chooseModelDirectory: vi.fn(), openModelFiles: vi.fn() } as unknown as ReturnType<typeof useModelCatalog>;
  const { rerender } = render(<LocalCheckpoints catalog={catalog} />);
  expect(screen.getByRole('progressbar', { name: 'VRAM + RAM pressure: 115.3%' })).toBeInTheDocument();
  expect(screen.getByText(/Experts that do not fit/)).toBeInTheDocument();
  catalog.runtime = { ...catalog.runtime, host_memory_available_bytes: 64 * GiB, device_free_bytes: 24 * GiB };
  rerender(<LocalCheckpoints catalog={catalog} />);
  expect(screen.getByRole('progressbar', { name: 'VRAM + RAM pressure: 94.3%' })).toBeInTheDocument();
  expect(screen.queryByText(/Experts that do not fit/)).not.toBeInTheDocument();
  expect(screen.getAllByRole('progressbar')).toHaveLength(1);
  expect(document.querySelector('.variant-memory-caption')).toBeNull();
  expect(screen.getByText(/SSD PLE 50/)).toBeInTheDocument();
});

it('shows the same zero shared budget as the model-loading estimate, not raw free memory', () => {
  const artifact: ModelArtifact = { id: 'a', name: 'MoE', architecture: 'qwen4_exp', format: 'tq',
    shard_count: 1, total_bytes: 2 ** 30, estimated_resident_weight_bytes: 2 ** 30,
    tensor_count: 1, record_count: 1, dtypes: [], complete: true, loadable: true, modified_at: '2026-10-18' };
  const catalog = { runtime: { memory_architecture: 'unified', host_memory_available_bytes: 3 * 2 ** 30,
    runtime_memory_headroom_bytes: 0 }, artifacts: [artifact], filteredArtifacts: [artifact], instances: [], busy: false,
    modelFilter: '', setModelFilter: vi.fn(), unloadInstance: vi.fn(), loadArtifact: vi.fn(),
    chooseModelDirectory: vi.fn(), openModelFiles: vi.fn() } as unknown as ReturnType<typeof useModelCatalog>;
  render(<LocalCheckpoints catalog={catalog} />);
  expect(screen.getByText('Remaining available memory', { exact: false })).toHaveTextContent('Remaining available memory 0 GiB');
  expect(screen.getByRole('progressbar', { name: 'Estimated share of remaining available memory: No available memory' })).toBeInTheDocument();
});

it('shows remaining memory once beside Filter instead of repeating it on model rows', () => {
  const base = { architecture: 'qwen4_exp', format: 'tq', shard_count: 1, total_bytes: 2 ** 30,
    estimated_resident_weight_bytes: 2 ** 30, tensor_count: 1, record_count: 1, dtypes: [],
    complete: true, loadable: true, modified_at: '' };
  const artifacts = [{ ...base, id: 'a', name: 'Model A' }, { ...base, id: 'b', name: 'Model B' }] as ModelArtifact[];
  const catalog = { runtime: { memory_architecture: 'unified', runtime_memory_headroom_bytes: 85.3 * 2 ** 30 },
    artifacts, filteredArtifacts: artifacts, instances: [], busy: false, modelFilter: '', setModelFilter: vi.fn(),
    unloadInstance: vi.fn(), loadArtifact: vi.fn(), chooseModelDirectory: vi.fn(), openModelFiles: vi.fn(),
  } as unknown as ReturnType<typeof useModelCatalog>;
  const { container } = render(<LocalCheckpoints catalog={catalog} />);
  const memory = screen.getByText('Remaining available memory', { exact: false });
  expect(memory).toHaveTextContent('Remaining available memory 85.3 GiB');
  expect(memory.previousElementSibling).toHaveClass('list-filters');
  expect(container.querySelectorAll('.checkpoint-available-memory')).toHaveLength(1);
  expect(container.querySelectorAll('.model-row')).toHaveLength(2);
  expect(container.querySelector('.model-row')).not.toHaveTextContent('Remaining available memory');
  expect(screen.getAllByRole('progressbar')).toHaveLength(2);
});

it.each(['unified', 'discrete'] as const)('allows loading a low-memory MoE and describes SSD streaming on %s memory', async (memory_architecture) => {
  const GiB = 2 ** 30;
  const artifact: ModelArtifact = { id: 'a', name: 'MoE', architecture: 'new-moe-architecture', format: 'tq',
    shard_count: 1, total_bytes: 100 * GiB, estimated_resident_weight_bytes: 80 * GiB, ssd_ple_bytes: 20 * GiB,
    estimated_weight_bytes_by_role: { dense: 4 * GiB, experts: 76 * GiB },
    tensor_count: 1, record_count: 1, dtypes: [], complete: true, loadable: true, modified_at: '' };
  const loadArtifact = vi.fn();
  const catalog = { runtime: { memory_architecture, host_memory_available_bytes: 16 * GiB,
    runtime_memory_headroom_bytes: 16 * GiB, device_free_bytes: 8 * GiB }, loadDeviceIds: [],
    artifacts: [artifact], filteredArtifacts: [artifact], instances: [], busy: false,
    modelFilter: '', setModelFilter: vi.fn(), unloadInstance: vi.fn(), loadArtifact,
    chooseModelDirectory: vi.fn(), openModelFiles: vi.fn() } as unknown as ReturnType<typeof useModelCatalog>;
  const view = render(<LocalCheckpoints catalog={catalog} />);
  const button = screen.getByRole('button', { name: 'Load' });
  expect(button).toBeEnabled();
  expect(button).toHaveAccessibleDescription('Insufficient remaining memory. Experts that do not fit will be streamed from SSD during inference.');
  await userEvent.click(button);
  expect(loadArtifact).toHaveBeenCalledExactlyOnceWith('MoE');
  catalog.busy = true;
  view.rerender(<LocalCheckpoints catalog={catalog} />);
  expect(button).toBeDisabled();
});

it.each([
  { experts: 0, resident: 80, loadable: true, available: 16 },
  { experts: 76, resident: 16, loadable: true, available: 16 },
  { experts: 76, resident: 80, loadable: false, available: 16 },
  { experts: 76, resident: 80, loadable: true, available: undefined },
])('does not claim SSD expert streaming for a dense, fitting, invalid or unmeasured model: %j', ({ experts, resident, loadable, available }) => {
  const artifact: ModelArtifact = { id: 'a', name: 'Model', architecture: 'qwen4_exp', format: 'tq',
    shard_count: 1, total_bytes: 100, estimated_resident_weight_bytes: resident,
    estimated_weight_bytes_by_role: { experts }, tensor_count: 1, record_count: 1,
    dtypes: [], complete: loadable, loadable, modified_at: '' };
  const catalog = { runtime: { runtime_memory_headroom_bytes: available },
    artifacts: [artifact], filteredArtifacts: [artifact], instances: [], busy: false,
    modelFilter: '', setModelFilter: vi.fn(), unloadInstance: vi.fn(), loadArtifact: vi.fn(),
    chooseModelDirectory: vi.fn(), openModelFiles: vi.fn() } as unknown as ReturnType<typeof useModelCatalog>;
  render(<LocalCheckpoints catalog={catalog} />);
  expect(screen.queryByText(/Experts that do not fit/)).not.toBeInTheDocument();
  if (!loadable) expect(screen.getByRole('button', { name: 'Load' })).toBeDisabled();
});

it('keeps incomplete-shard and streaming notices in the same slot before memory, outside the actions', async () => {
  const GiB = 2 ** 30;
  const base = { architecture: 'qwen4_exp', format: 'tq', shard_count: 8, total_bytes: 80 * GiB,
    estimated_resident_weight_bytes: 80 * GiB, estimated_weight_bytes_by_role: { experts: 76 * GiB, dense: 4 * GiB },
    tensor_count: 1, record_count: 1, dtypes: [], complete: true, loadable: true, modified_at: '' } as Omit<ModelArtifact, 'id' | 'name'>;
  const artifacts = [{ ...base, id: 'streamed', name: 'Streamed model' },
    { ...base, id: 'incomplete', name: 'Incomplete model', complete: false, loadable: false, missing_shards: 2 }];
  const catalog = { runtime: { runtime_memory_headroom_bytes: 16 * GiB }, artifacts, filteredArtifacts: artifacts, instances: [], busy: false,
    modelFilter: '', setModelFilter: vi.fn(), unloadInstance: vi.fn(), loadArtifact: vi.fn(),
    chooseModelDirectory: vi.fn(), openModelFiles: vi.fn() } as unknown as ReturnType<typeof useModelCatalog>;
  const view = render(<LocalCheckpoints catalog={catalog} />);
  const notices = view.container.querySelectorAll('.local-checkpoint-notice');
  expect(notices).toHaveLength(2);
  expect(notices[0]).toHaveTextContent('Insufficient remaining memory.');
  expect(notices[1]).toHaveTextContent('Incomplete shards');
  const incomplete = view.container.querySelectorAll('.model-row')[1];
  const load = incomplete.querySelector<HTMLButtonElement>('.model-load-action')!;
  expect(load).toHaveTextContent('Load');
  expect(load).toBeDisabled();
  expect(load).toHaveAccessibleDescription('Incomplete shards');
  await userEvent.click(load);
  expect(catalog.loadArtifact).not.toHaveBeenCalled();
  for (const notice of notices) expect(notice.nextElementSibling).toHaveClass('local-checkpoint-memory');
  for (const actions of view.container.querySelectorAll('.model-row-controls')) {
    expect(actions).not.toHaveTextContent('Insufficient remaining memory.');
    expect(actions).not.toHaveTextContent('Incomplete shards');
  }
});

it('keeps quantized versions under their model and loads the exact selected variant', async () => {
  const base = { format: 'tq', shard_count: 1, total_bytes: 100,
    tensor_count: 1, record_count: 1, dtypes: [], complete: true, loadable: true, modified_at: '' };
  const artifacts = [
    { ...base, id: 'flash-s2', name: 'Qwen3.8-Flash-Next-EW-S2-L', architecture: 'qwen4_exp_text' },
    { ...base, id: 'deepseek', name: 'DeepSeek-V4.1-Flash-1017-Q3', architecture: 'deepseek_v41_text' },
    { ...base, id: 'dense', name: 'Qwen3.8-27B-V1_M', architecture: 'qwen3_5_text' },
    { ...base, id: 'flash-iq3', name: 'Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS', architecture: 'qwen4_exp_text', format: 'gguf' },
  ] as ModelArtifact[];
  const loadArtifact = vi.fn();
  const catalog = { runtime: null, artifacts, filteredArtifacts: artifacts, instances: [], busy: false,
    modelFilter: '', setModelFilter: vi.fn(), unloadInstance: vi.fn(), loadArtifact,
    chooseModelDirectory: vi.fn(), openModelFiles: vi.fn() } as unknown as ReturnType<typeof useModelCatalog>;
  const view = render(<LocalCheckpoints catalog={catalog} />);
  expect([...view.container.querySelectorAll('.checkpoint-family-heading h3')].map(element => element.textContent)).toEqual(['Qwen', 'DeepSeek']);
  const flash = screen.getByRole('region', { name: 'Qwen3.8-Flash-Next' });
  expect(flash.querySelectorAll('.model-row')).toHaveLength(2);
  expect(screen.getByRole('region', { name: 'Qwen3.8-27B' }).querySelectorAll('.model-row')).toHaveLength(1);
  const variant = [...flash.querySelectorAll<HTMLElement>('.model-row')].find(row => row.textContent?.includes('GSQ-RCO'))!;
  await userEvent.click(within(variant).getByRole('button', { name: 'Load' }));
  expect(loadArtifact).toHaveBeenCalledExactlyOnceWith('Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS');
});

it('lists independent tables with file access but no load controls or pressure bars', async () => {
  const base = { architecture: 'qwen4_exp_text', format: 'tq', shard_count: 2, total_bytes: 2 ** 30,
    tensor_count: 1, record_count: 1, dtypes: [], complete: true, loadable: false, modified_at: '' };
  const artifacts = [
    { ...base, id: 'model', name: 'Qwen3.8-Flash-Next-EW-S2-L', loadable: true },
    { ...base, id: 'ple', name: 'Qwen3.8-Flash-Next-PLE', asset_kind: 'ple' },
    { ...base, id: 'engram', name: 'Qwen3.8-Flash-Next-Engram', asset_kind: 'engram' },
  ] as ModelArtifact[];
  const openModelFiles = vi.fn();
  const catalog = { runtime: null, artifacts, filteredArtifacts: artifacts, instances: [], busy: false,
    modelFilter: '', setModelFilter: vi.fn(), unloadInstance: vi.fn(), loadArtifact: vi.fn(),
    chooseModelDirectory: vi.fn(), openModelFiles } as unknown as ReturnType<typeof useModelCatalog>;
  render(<LocalCheckpoints catalog={catalog} />);
  const group = screen.getByRole('region', { name: 'Qwen3.8-Flash-Next' });
  expect(group.querySelectorAll('.model-row')).toHaveLength(3);
  expect(screen.getByText('1 local models · 2 companion assets')).toBeInTheDocument();
  for (const kind of ['PLE', 'Engram']) {
    const row = screen.getByText(`Independently usable ${kind}`).closest('.model-row')!;
    expect(row).toHaveTextContent('1 GiB · 2 shards');
    expect(row.querySelector('.local-checkpoint-resources')).toBeNull();
    expect(within(row as HTMLElement).queryByRole('button', { name: 'Load' })).toBeNull();
    expect(within(row as HTMLElement).getAllByRole('button')).toHaveLength(1);
    await userEvent.click(within(row as HTMLElement).getByRole('button', { name: 'Model files' }));
    expect(openModelFiles).toHaveBeenLastCalledWith(kind.toLowerCase());
  }
  expect(screen.getAllByRole('button', { name: 'Load' })).toHaveLength(1);
});
