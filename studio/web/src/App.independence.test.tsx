import type { ReactNode } from 'react';
import { render, screen } from '@testing-library/react';
import { MemoryRouter, Outlet, useLocation } from 'react-router';
import { expect, it, vi } from 'vitest';
import App from './App';

vi.mock('./features/settings/UpdateManager', () => ({ StudioUpdateProvider: ({ children }: { children: ReactNode }) => children }));
vi.mock('./app/RuntimeProvider', () => ({ RuntimeProvider: ({ children }: { children: ReactNode }) => children }));
vi.mock('./app/StudioShell', () => ({ StudioShell: () => <Outlet /> }));
vi.mock('./features/runtime/OverviewPage', () => ({ OverviewPage: () => <p>Independent inference overview</p> }));
function Location() { return <output aria-label="route">{useLocation().pathname}</output>; }
it('opens the standalone inference product without mounting a legacy Agent workspace', async () => {
  const network = vi.fn(() => Promise.reject(new Error('Unexpected Agent dependency')));
  vi.stubGlobal('fetch', network);
  try {
    render(<MemoryRouter initialEntries={['/']}><Location /><App /></MemoryRouter>);
    expect(await screen.findByText('Independent inference overview')).toBeVisible();
    expect(screen.getByLabelText('route')).toHaveTextContent('/inference');
    expect(network).not.toHaveBeenCalled();
  } finally { vi.unstubAllGlobals(); }
});
