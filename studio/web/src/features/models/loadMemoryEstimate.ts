import type { ModelArtifact, RuntimeStatus } from '../../shared/api/types';
import { availableResidentMemory, hasSeparateVram } from '../runtime/memoryArchitecture';

function measured(value: number | null | undefined): value is number {
  return value != null && Number.isFinite(value) && value >= 0;
}

export function availableLoadPools(runtime?: RuntimeStatus | null, deviceIds: string[] = []) {
  const ram = measured(runtime?.host_memory_available_bytes) ? runtime.host_memory_available_bytes : null;
  const devices = runtime?.gpu_devices ?? [];
  const selected = deviceIds.length ? devices.filter(device => deviceIds.includes(device.id)) : devices.slice(0, 1);
  if (selected.length && (!deviceIds.length || selected.length === deviceIds.length)) {
    const vram = selected.some(device => !measured(device.memory_available_bytes))
      ? null : selected.reduce((sum, device) => sum + device.memory_available_bytes!, 0);
    return { vram, ram };
  }
  return { ram, vram: !deviceIds.length && measured(runtime?.device_free_bytes) ? runtime.device_free_bytes : null };
}

export function availableLoadMemory(runtime?: RuntimeStatus | null, deviceIds: string[] = []): number | null {
  if (!hasSeparateVram(runtime)) return availableResidentMemory(runtime);
  const { vram, ram } = availableLoadPools(runtime, deviceIds);
  return vram == null || ram == null ? null : vram + ram;
}

export function estimateModelWeights(model: ModelArtifact, placement: 'stream' | 'vram' = 'stream') {
  if (!model.structured_decisions) {
    const total = model.estimated_resident_weight_bytes;
    return measured(total) ? { total, vram: null, ram: null } : null;
  }
  const estimate = model.jev_weight_memory;
  if (!estimate) return null;
  const vram = placement === 'stream' ? estimate.stream_vram_bytes : estimate.resident_vram_bytes;
  const ram = placement === 'stream' ? estimate.stream_ram_bytes : 0;
  return measured(vram) && measured(ram) ? { total: vram + ram, vram, ram } : null;
}
