import React, { createContext, useContext, useMemo } from 'react';

const PresenterContext = createContext(null);
export function SharedPresenters({ tr, inventory, runtime, operation, jobs = [], children }) {
  const value = useMemo(() => {
    const native = [...jobs, ...(operation ? [operation] : [])];
    const presented = native.filter(job => typeof job?.kind === 'string').map(job => {
      const payload = job.payload || {};
      const artifact = inventory?.models.find(item => item.id === payload.model);
      return { ...job, payload: artifact ? { ...payload, model: artifact.name } : payload };
    });
    return { tr, runtime, inventory, jobs: [...new Map(presented.map(job => [job.id, job])).values()] };
  }, [tr, inventory, runtime, operation, jobs]);
  return <PresenterContext.Provider value={value}>{children}</PresenterContext.Provider>;
}
export function useSettings() {
  const value = useContext(PresenterContext);
  if (!value) throw new Error('Shared presenter context is missing');
  return { tr: value.tr };
}
export function useJobStore(selector) {
  const value = useContext(PresenterContext);
  return selector({ jobs: value?.jobs || [] });
}
export function readLoadContext() { return { jev_weights: 'stream' }; }
