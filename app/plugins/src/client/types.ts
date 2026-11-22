// Shared Client-side shapes: the DSH RPC channel surface and the core records the pages render.
// The reply envelope is the Host's BridgeResult (see ../host/types.ts); the Client validates it at runtime,
// so the transport here stays loose and the checks in page-api/model-api remain the source of truth.
export interface RpcConnection {
  rpc: {
    call(channel: string, endpoint: string, payload?: any, signal?: AbortSignal): Promise<any>;
  };
}

export type RpcPayload = Record<string, any>;

export interface ArtifactRow {
  id: string;
  name?: string;
  format?: string;
  complete?: boolean;
  [key: string]: any;
}

export interface JobEvent {
  job_id: string;
  sequence: number;
  type: string;
  level?: string;
  message?: string | null;
  data?: Record<string, any>;
  created_at: string;
}

export interface OperationCheckpoint {
  id: string | null;
  kind: string | null;
  unknown: boolean;
  label: string;
}

export interface ResourceRead {
  value: any;
  error?: any;
}

// Only the storage calls this plugin uses; reads are required, writes are guarded by try/catch, and tests supply their own fake.
export interface StorageLike {
  getItem(key: string): string | null;
  setItem?(key: string, value: string): void;
  removeItem?(key: string): void;
}
