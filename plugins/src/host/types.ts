// Types for the bridge's own contract: the RPC surface, the resolved core call, and the result envelope.
// shortcut: RPC payloads and core JSON stay any here — parseCall and validResult are the validators;
// tighten payload shapes together with type guards in those validators.
export interface CoreProvider {
  baseURL?: string;
  apiKeyEnv?: string;
}

export interface ProviderUpdate {
  commit(): Promise<void>;
  discard?(): Promise<void>;
}

export interface BridgeCall {
  kind: string;
  path?: string;
  query?: string;
  method?: string;
  body?: string | null;
  format?: string;
  status?: number;
  resource?: string;
  id?: string;
  jobKinds?: Set<string>;
  companionIds?: string[];
  directoryId?: string;
  directoryGuard?: string;
  maintenance?: boolean;
  authChange?: { api_key?: string; enabled?: boolean };
  listenerChange?: { port: number; protocol: string };
  validResult?: (value: any) => boolean;
}

export interface BridgeError {
  code: string;
  message: string;
  details: Record<string, any>;
}

export type BridgeResult = { ok: true; value: any } | { ok: false; error: BridgeError };

export interface RpcPeer {
  principal?: { kind?: string };
}

export interface BridgeOptions {
  getProvider(): CoreProvider | undefined;
  resolveCredential(environment: string): Promise<{ value?: string } | undefined>;
  prepareProviderUpdate?(provider: CoreProvider, change: { baseURL?: string; apiKey?: string }): Promise<ProviderUpdate>;
  fetchImpl?: typeof fetch;
  timeoutMs?: number;
  maintenanceTimeoutMs?: number;
  maxResponseBytes?: number;
}

export type RpcHandler = (endpoint: string, payload: any, signal: AbortSignal | undefined, peer: RpcPeer | undefined) => Promise<BridgeResult>;
