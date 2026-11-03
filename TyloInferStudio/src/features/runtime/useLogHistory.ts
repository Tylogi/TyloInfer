import { useEffect, useState } from 'react';
import { useRuntime } from '../../app/RuntimeProvider';
import type { JobResource } from '../../shared/api/types';
import { useJobStore } from '../../stores/jobStore';

/** Keep the job history current within the connected scope; the live event stream feeds it. */
export function useLogHistory(paused: boolean) {
  const { ready, connectionRevision } = useRuntime();
  const jobs = useJobStore((state) => state.jobs);
  const [jobSnapshot, setJobSnapshot] = useState<JobResource[]>(jobs);
  const [updatedAt, setUpdatedAt] = useState<number | null>(null);
  const [refreshVersion, setRefreshVersion] = useState(0);
  useEffect(() => {
    setJobSnapshot([]);
    setUpdatedAt(null);
  }, [ready, connectionRevision]);
  useEffect(() => { if (!paused && ready) setJobSnapshot(jobs); }, [jobs, paused, ready]);
  useEffect(() => { if (ready) setUpdatedAt(Date.now()); }, [refreshVersion, ready]);
  return { jobs: jobSnapshot, updatedAt, ready, refresh: () => setRefreshVersion((version) => version + 1) };
}
