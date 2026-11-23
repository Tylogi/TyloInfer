import { useEffect, useState } from 'react';
import { useRuntime } from '../../app/RuntimeProvider';
import { errorMessage } from '@shared/formatters';
import { useSettings } from '../settings/SettingsProvider';
import { runtimeApi } from '../../shared/api/resources/runtime';
import type { RuntimeResources } from '@shared/types/api';
import { ResourceMonitorView } from '@shared/features/runtime/ResourceMonitorView';

export function ResourceMonitorPanel() {
  const { ready, connectionRevision, runtime } = useRuntime();
  const { tr } = useSettings();
  const [resources, setResources] = useState<RuntimeResources | null>(null);
  const [error, setError] = useState('');
  useEffect(() => {
    let disposed = false;
    let polling = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const controller = new AbortController();
    setResources(null);
    setError('');
    async function poll() {
      if (disposed || polling) return;
      clearTimeout(timer);
      if (document.hidden) { timer = setTimeout(() => void poll(), 2000); return; }
      polling = true;
      try {
        const next = await runtimeApi.runtimeResources(controller.signal);
        if (!disposed) { setResources(next); setError(''); }
      } catch (cause) {
        if (!disposed) { setResources(null); setError(errorMessage(cause)); }
      } finally {
        polling = false;
        if (!disposed) timer = setTimeout(() => void poll(), 2000);
      }
    }
    function visible() { if (!document.hidden) void poll(); }
    if (ready) { document.addEventListener('visibilitychange', visible); void poll(); }
    return () => { disposed = true; controller.abort(); clearTimeout(timer); document.removeEventListener('visibilitychange', visible); };
  }, [ready, connectionRevision]);
  return <ResourceMonitorView resources={resources} runtime={runtime} error={error} tr={tr} />;
}
