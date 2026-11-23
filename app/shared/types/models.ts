// Model data shapes shared by the Studio web app and the plugin Client.

import type { KvQuantizationSettings } from './runtime';

export interface ModelLoadContextSettings {
  reserve_percent?: 5 | 10 | 20;
  precision?: number | null;
  automatic?: boolean;
  ssd_kv?: boolean;
  ram_preallocated?: boolean;
  vram_preallocated: boolean;
  kv_quantization?: KvQuantizationSettings;
}

export interface ModelLoadPlan {
  max_context: number;
  context_size: number | null;
  precision: number;
  precisions: number[];
  usable: boolean;
  reason: 'model_limit' | 'memory_limit' | 'memory_unavailable' | 'precision_unsupported' | null;
}

export interface ModelLoadPlanQuery {
  reserve: 5 | 10 | 20;
  context_size?: number;
  precision?: number;
  preallocate: boolean;
  device?: string;
  streamed_jev: boolean;
  ssd_kv?: boolean;
  ram_preallocated?: boolean;
}

export interface DownloadDirectoryStatus {
  directory: string | null;
  actual_directory: string;
  default_directory: string;
  disk_total_bytes?: number | null;
  disk_free_bytes?: number | null;
  disk_available_bytes?: number | null;
  download_reserved_bytes?: number;
  reclaimable_prefix_bytes?: number;
  evict_prefix_cache?: boolean;
}
