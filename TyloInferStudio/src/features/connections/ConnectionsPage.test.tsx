import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { ConnectionsPage } from './ConnectionsPage';
import { runtimeApi } from '../../shared/api/resources/runtime';
import { setApiBaseUrl, setBrowserServiceUrl } from '../../shared/api/client';
import { configureStudio, type StudioStatus } from '../../studio';
import type { RuntimeStatus } from '../../shared/api/types';

const { reloadService, connection } = vi.hoisted(() => ({ reloadService: vi.fn(), connection: {
  desktop: false, runtime: { model: 'model' } as RuntimeStatus, studio: null as StudioStatus | null,
} }));
vi.mock('../../app/RuntimeProvider', () => ({ useRuntime: () => ({
  runtime: connection.runtime, models: [], instances: [], studio: connection.studio,
  selectedModel: '', setSelectedModel: vi.fn(), reloadService,
}) }));
vi.mock('../settings/SettingsProvider', () => ({ useSettings: () => ({ tr: (_zh: string, en: string) => en }) }));
vi.mock('./MemorySettingsPanel', () => ({ MemorySettingsPanel: () => <div className="section-label">Memory plan</div> }));
vi.mock('./ApiAuthenticationPanel', () => ({ ApiAuthenticationPanel: () => <span>Authentication controls</span> }));
vi.mock('./ContextManagementPanel', () => ({ ContextManagementPanel: () => <div className="section-label">Context management</div> }));
vi.mock('./PrefixCacheSettingsPanel', () => ({ PrefixCacheSettingsPanel: () => <div className="section-label">Persistent prefix cache</div> }));
vi.mock('./InferenceDefaultsPanel', () => ({ InferenceDefaultsPanel: () => null }));
vi.mock('./InferencePolicyPanel', () => ({ InferencePolicyPanel: () => null }));
vi.mock('./RemoteRoutingPanel', () => ({ RemoteRoutingPanel: () => null }));
vi.mock('../../studio', () => ({ isStudio: () => connection.desktop, studioCredential: async () => '', configureStudio: vi.fn() }));

beforeEach(() => {
  connection.desktop = false;
  connection.runtime = { model: 'model' };
  connection.studio = null;
  localStorage.clear();
  setBrowserServiceUrl('');
  setApiBaseUrl('');
  reloadService.mockResolvedValue(true);
  vi.spyOn(runtimeApi, 'runtimeListener').mockResolvedValue({ host: '127.0.0.1', port: 8090, anthropic_port: 8091, configurable: true });
});

it('changes a managed native Studio port through its launcher without requesting HTTP listener APIs', async () => {
  connection.desktop = true;
  connection.runtime = { source: 'native-instances' };
  connection.studio = { config: { mode: 'local', local_service_port: 8090, remote_url: 'http://127.0.0.1:8090' },
    service_url: 'http://127.0.0.1:8090', reachable: true, connection: 'managed', managed_pid: 123 };
  setApiBaseUrl(connection.studio.service_url);
  vi.mocked(configureStudio).mockResolvedValue(connection.studio);
  const listener = vi.spyOn(runtimeApi, 'configureRuntimeListener');
  render(<MemoryRouter><ConnectionsPage /></MemoryRouter>);
  const port = screen.getByRole('spinbutton', { name: 'OpenAI port' });
  expect(port).toBeEnabled();
  fireEvent.change(port, { target: { value: '8092' } });
  fireEvent.blur(port);
  await waitFor(() => expect(configureStudio).toHaveBeenCalledWith({ ...connection.studio!.config, local_service_port: 8092 }));
  await waitFor(() => expect(reloadService).toHaveBeenCalledOnce());
  expect(runtimeApi.runtimeListener).not.toHaveBeenCalled();
  expect(listener).not.toHaveBeenCalled();
  expect(await screen.findByText(/The service restarted on the new port/)).toBeInTheDocument();
});
afterEach(() => { vi.restoreAllMocks(); localStorage.clear(); setApiBaseUrl(''); });

it('rejects an invalid OpenAI port without restarting the managed service', async () => {
  connection.desktop = true;
  connection.runtime = { source: 'native-instances' };
  connection.studio = { config: { mode: 'local', local_service_port: 8090, remote_url: 'http://127.0.0.1:8090' },
    service_url: 'http://127.0.0.1:8090', reachable: true, connection: 'managed', managed_pid: 123 };
  setApiBaseUrl(connection.studio.service_url);
  render(<MemoryRouter><ConnectionsPage /></MemoryRouter>);
  const port = screen.getByRole('spinbutton', { name: 'OpenAI port' });
  fireEvent.change(port, { target: { value: '65536' } });
  fireEvent.blur(port);
  expect(await screen.findByRole('alert')).toHaveTextContent('Port must be an integer between 1 and 65535');
  expect(configureStudio).not.toHaveBeenCalled();
});
