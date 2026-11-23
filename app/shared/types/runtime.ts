// Runtime data shapes shared by the Studio web app and the plugin Client.

export interface RuntimeMemoryPolicy {
  total_limit_bytes?: number | null;
  effective_total_limit_bytes?: number | null;
  automatic_total_limit_bytes?: number | null;
  total_capacity_limit_bytes?: number | null;
  model_limit_bytes: number | null;
  prefix_limit_bytes: number | null;
  prefix_disk_limit_bytes?: number | null;
  prefix_ram_limit_bytes?: number | null;
  effective_prefix_ram_limit_bytes?: number;
  prefix_directory: string | null;
  actual_prefix_directory: string;
}

export interface RuntimeAuthenticationPolicy {
  enabled: boolean;
  has_key: boolean;
}

export interface RuntimeContextPolicy {
  fallback_context_size: number;
  max_context_size: number | null;
  model_overrides: Record<string, number>;
  model_yarn_enabled?: Record<string, boolean>;
  model_qsa_kv_offload?: Record<string, QsaKvStorageSettings>;
  model_kv_quantization?: Record<string, KvQuantizationSettings>;
  model_kv_memory?: Record<string, KvMemorySettings>;
}

export interface KvMemorySettings {
  placement: 'auto' | 'auto_ssd' | 'vram' | 'ram';
  allocation: 'preallocated' | 'dynamic';
  ram_preallocated?: boolean;
}

export interface KvQuantizationSettings {
  enabled: boolean;
  bits: 2 | 2.5 | 3 | 3.5 | 4 | 6 | 8;
  algorithm: 'turboquant';
}

export interface QsaKvStorageSettings {
  enabled: boolean;
  budget_bytes: number;
  ram_budget_bytes?: number | null;
}

export interface YarnContextInfo {
  supported: boolean;
  native_context: number | null;
  maximum_context: number | null;
  maximum_factor: number | null;
  enabled: boolean;
  effective_factor: number;
}

export interface QsaKvOffloadInfo {
  supported: boolean;
  architecture: string;
  policy: QsaKvStorageSettings & { target_context: number };
  required_index_bytes: number | null;
  buffer_bytes: number;
  target_context: number;
  reason: string | null;
}

export interface PrefixCacheGroup {
  id: string;
  model_name: string | null;
  architecture: string | null;
  codec: string | null;
  context_size: number | null;
  bytes: number;
  blocks: number;
  prefixes: number;
  block_tokens: number;
  max_prefix_tokens: number;
  incomplete_prefixes: number;
  text_blocks: number;
  last_used_at: string;
}

export interface PrefixCacheBlock {
  id: string;
  parent: string;
  tokens: number;
  prefix_tokens: number;
  complete_chain: boolean;
  text_available: boolean;
  bytes: number;
  last_used_at: string;
}

export interface PrefixCacheInventory {
  directory: string;
  total_bytes: number;
  total_blocks: number;
  can_clear: boolean;
  data: PrefixCacheGroup[];
  blocks: PrefixCacheBlock[];
  offset: number;
  limit: number;
  matched_blocks?: number;
  search_text_unavailable?: boolean;
}

export interface PrefixCacheFilters {
  query: string;
  search_in: 'all' | 'text' | 'id';
  text_filter: 'all' | 'saved' | 'missing';
  chain_filter: 'all' | 'complete' | 'incomplete';
  sort: 'recent' | 'oldest' | 'length' | 'size';
}

export interface PrefixCacheText {
  available: boolean;
  text?: string;
  reason?: string;
  total_tokens?: number;
  offset?: number;
  next_offset?: number | null;
}
