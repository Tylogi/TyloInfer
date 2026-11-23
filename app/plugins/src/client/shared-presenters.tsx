import React, { useMemo } from 'react';
import { PresentationProvider, useSharedDisplay } from '@shared/presentation';
import { Dialog, Switch, Tooltip } from './shared-primitives.tsx';

// The plugin Client owns its job data through host RPC. It normalises what the
// inference core reports and hands the shared display layer the same shapes
// Studio provides from its own stores.
export function SharedPresenters({ tr, inventory, runtime, operation, jobs = [], children }) {
  const presented = useMemo(() => {
    const native = [...jobs, ...(operation ? [operation] : [])];
    const shown = native.filter(job => typeof job?.kind === 'string').map(job => {
      const payload = job.payload || {};
      const artifact = inventory?.models.find(item => item.id === payload.model);
      return { ...job, payload: artifact ? { ...payload, model: artifact.name } : payload };
    });
    return [...new Map(shown.map(job => [job.id, job])).values()];
  }, [inventory, operation, jobs]);
  const value = useMemo(() => ({
    tr,
    useJobStore: (selector) => selector({ jobs: presented }),
    readLoadContext: () => ({ jev_weights: 'stream' }),
    primitives: { Dialog, Switch, Tooltip },
  }), [tr, presented]);
  return <PresentationProvider value={value}>{children}</PresentationProvider>;
}

export function useSettings() {
  return { tr: useSharedDisplay().tr };
}
