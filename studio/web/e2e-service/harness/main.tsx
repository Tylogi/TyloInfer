/** Test-only component harness; it does not add a route to the published Studio. */
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { MemoryRouter } from 'react-router';
import * as TooltipPrimitive from '@radix-ui/react-tooltip';
import { ChatPage } from '../../src/features/chat/ChatPage';
import { ChatProvider } from '../../src/features/chat/ChatProvider';
import { RuntimeProvider } from '../../src/app/RuntimeProvider';
import { LoginPage } from '../../src/app/LoginPage';
import { useRuntime } from '../../src/app/RuntimeProvider';
import { SettingsProvider } from '../../src/features/settings/SettingsProvider';
import '../../src/styles.css';
import '../../src/shared/ui/primitives.css';

function ChatHarness() {
  const { authenticationRequired, loading } = useRuntime();
  if (authenticationRequired) return <LoginPage />;
  if (loading) return <p role="status">Loading service…</p>;
  return (
    <TooltipPrimitive.Provider delayDuration={0}>
      <ChatProvider><ChatPage /></ChatProvider>
    </TooltipPrimitive.Provider>
  );
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <SettingsProvider>
      <MemoryRouter initialEntries={['/chat']}>
        <RuntimeProvider><ChatHarness /></RuntimeProvider>
      </MemoryRouter>
    </SettingsProvider>
  </StrictMode>,
);
