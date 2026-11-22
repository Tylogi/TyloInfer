



import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { createBrowserRouter, createHashRouter, RouterProvider } from 'react-router';

import App from './App';
import { AppErrorBoundary } from './app/FailurePage';
import { isStudio } from './studio';
import { SettingsProvider } from './features/settings/SettingsProvider';
import { DesktopWindowFrame } from './app/DesktopWindowFrame';
import { installAssetRecovery } from './app/assetRecovery';
import './styles.css';

installAssetRecovery();

if (!isStudio() && window.location.pathname === '/' && window.location.hash.startsWith('#/')) {
  window.history.replaceState(window.history.state, '', window.location.hash.slice(1));
}

const createRouter = isStudio() ? createHashRouter : createBrowserRouter;
const router = createRouter([
  {
    path: '*',
    element: <App />,
  },
]);

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <SettingsProvider>
      <DesktopWindowFrame>
        <AppErrorBoundary>
          <RouterProvider router={router} />
        </AppErrorBoundary>
      </DesktopWindowFrame>
    </SettingsProvider>
  </StrictMode>,
);
