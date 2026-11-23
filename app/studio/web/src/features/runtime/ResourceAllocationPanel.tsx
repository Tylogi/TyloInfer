import { ResourceAllocationView } from '@shared/features/runtime/ResourceAllocationView';
import { useEffect } from 'react';
import { useRuntime } from '../../app/RuntimeProvider';
import { useSettings } from '../settings/SettingsProvider';

export function ResourceAllocationPanel() {
  const { runtime, instances, ready, connectionRevision, refreshRuntime, refreshError } = useRuntime();
  const { tr } = useSettings();
  useEffect(() => {
    if (!ready) return;
    let disposed = false;
    let polling = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    async function poll() {
      if (disposed || polling) return;
      clearTimeout(timer);
      polling = true;
      try {
        if (!document.hidden) await refreshRuntime();
      } finally {
        polling = false;
        if (!disposed) timer = setTimeout(() => void poll(), 5000);
      }
    }
    function visible() { if (!document.hidden) void poll(); }
    document.addEventListener('visibilitychange', visible);
    void poll();
    return () => { disposed = true; clearTimeout(timer); document.removeEventListener('visibilitychange', visible); };
  }, [ready, connectionRevision, refreshRuntime]);
  return <ResourceAllocationView runtime={runtime} instances={instances} connectionRevision={connectionRevision} refreshError={refreshError} tr={tr} />;
}
