import { act, cleanup, renderHook } from '@testing-library/react';
import type { ReactNode } from 'react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { AgentProvider, useAgent } from './AgentProvider';
import type { RuntimeStatus } from '../../shared/api/types';

const runtime = vi.hoisted(() => ({ ready: true, connectionRevision: 1, instances: [],
  runtime: null as RuntimeStatus | null }));
vi.mock('../../app/RuntimeProvider', () => ({ useRuntime: () => runtime }));

const wrapper = ({ children }: { children: ReactNode }) => <AgentProvider>{children}</AgentProvider>;
beforeEach(() => { runtime.runtime = {}; runtime.ready = true; runtime.connectionRevision = 1; });
afterEach(() => { cleanup(); vi.useRealTimers(); vi.unstubAllGlobals(); });

it.each([null, { source: 'native-instances' } as RuntimeStatus])('does not poll or start Agent APIs on a native or uninitialized connection: %j', async status => {
  vi.useFakeTimers();
  runtime.runtime = status;
  const fetchMock = vi.fn();
  vi.stubGlobal('fetch', fetchMock);
  const { result } = renderHook(useAgent, { wrapper });
  expect(result.current?.available).toBe(false);
  await act(async () => { await vi.advanceTimersByTimeAsync(7500); });
  await act(async () => { await result.current!.refresh(); await result.current!.start(); });
  await expect(result.current!.connectWorkspace('workspace-a')).rejects.toThrow('Use text chat instead');
  expect(result.current?.error).toContain('does not provide Agent workspaces');
  expect(result.current?.state.running).toBe(false);
  expect(fetchMock).not.toHaveBeenCalled();
});
