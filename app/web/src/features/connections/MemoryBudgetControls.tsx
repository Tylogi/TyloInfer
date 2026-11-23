import type { ReactNode } from 'react';
import { runtimeApi } from '../../shared/api/resources/runtime';
import { jobsApi } from '../../shared/api/resources/jobs';
import { useRuntime } from '../../app/RuntimeProvider';
import { useConnectionScope } from '../../app/useConnectionScope';
import { useJobStore } from '../../stores/jobStore';
import { toast } from '../../stores/toastStore';
import { MemoryBudgetControlsView } from '@shared/features/connections/MemoryBudgetControlsView';
const services = { memoryPolicy: runtimeApi.memoryPolicy, configureMemoryPolicy: runtimeApi.configureMemoryPolicy, getJob: jobsApi.getJob };
export function MemoryBudgetControls({ residency, children }: { residency: string; children?: ReactNode }) {
  const { runtime, addJob } = useRuntime();
  const jobs = useJobStore(state => state.jobs);
  const connectionScope = useConnectionScope();
  return <MemoryBudgetControlsView residency={residency} children={children} runtime={runtime} jobs={jobs} services={services} connectionScope={connectionScope} addJob={addJob} notify={message => toast.success(message)} />;
}
