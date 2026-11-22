import { expect, it } from 'vitest';
import { availableLoadMemory, estimateModelWeights } from './loadMemoryEstimate';
import type { ModelArtifact } from '../../shared/api/types';

it('counts unified memory once even for CUDA', () => {
  expect(availableLoadMemory({ memory_architecture: 'unified', backend: 'cuda', host_memory_available_bytes: 48,
    gpu_devices: [{ id: '0', name: 'Spark', memory_available_bytes: 48, unified_memory: true }] })).toBe(48);
});

it('uses the shared backend headroom including zero instead of raw system availability', () => {
  expect(availableLoadMemory({ memory_architecture: 'unified', host_memory_available_bytes: 3,
    runtime_memory_headroom_bytes: 0 })).toBe(0);
  expect(availableLoadMemory({ memory_architecture: 'unified', host_memory_available_bytes: 3,
    runtime_memory_headroom_bytes: 5 })).toBe(5);
  expect(availableLoadMemory({ memory_architecture: 'unified', runtime_memory_headroom_bytes: 5 })).toBe(5);
});

it('uses actual free RAM and the chosen GPU rather than installed capacity or other GPUs', () => {
  const runtime = { memory_architecture: 'discrete' as const, host_memory_available_bytes: 40,
    host_memory_total_bytes: 128, gpu_devices: [
      { id: '0', name: 'GPU0', memory_total_bytes: 24, memory_available_bytes: 3 },
      { id: '1', name: 'GPU1', memory_total_bytes: 24, memory_available_bytes: 5 }] };
  expect(availableLoadMemory(runtime)).toBe(43);
  expect(availableLoadMemory(runtime, ['1'])).toBe(45);
  expect(availableLoadMemory(runtime, ['0', '1'])).toBe(48);
  expect(availableLoadMemory(runtime, ['2'])).toBeNull();
});

it('keeps missing telemetry unknown and treats zero free space as measured', () => {
  expect(availableLoadMemory({ host_memory_total_bytes: 128 })).toBeNull();
  expect(availableLoadMemory({ memory_architecture: 'discrete', host_memory_available_bytes: 0,
    device_free_bytes: 0 })).toBe(0);
  expect(availableLoadMemory({ memory_architecture: 'discrete', host_memory_available_bytes: 12 })).toBeNull();
});

it('counts Jev layer buffers in VRAM and the backing weights in RAM', () => {
  const model = { structured_decisions: true, estimated_resident_weight_bytes: 10000,
    jev_weight_memory: { stream_vram_bytes: 400, stream_ram_bytes: 9000, resident_vram_bytes: 9100 } } as ModelArtifact;
  expect(estimateModelWeights(model)).toEqual({ total: 9400, vram: 400, ram: 9000 });
  expect(estimateModelWeights(model, 'vram')).toEqual({ total: 9100, vram: 9100, ram: 0 });
  expect(estimateModelWeights({ ...model, jev_weight_memory: undefined })).toBeNull();
  expect(estimateModelWeights({ ...model, structured_decisions: false })).toEqual({ total: 10000, vram: null, ram: null });
});
