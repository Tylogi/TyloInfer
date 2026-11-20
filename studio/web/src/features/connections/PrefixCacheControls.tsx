import { runtimeApi } from '../../shared/api/resources/runtime';
import { jobsApi } from '../../shared/api/resources/jobs';
import { useRuntime } from '../../app/RuntimeProvider';
import { useConnectionScope } from '../../app/useConnectionScope';
import { useJobStore } from '../../stores/jobStore';
import { toast } from '../../stores/toastStore';
import { PrefixCacheControlsView } from './PrefixCacheControlsView';
const services = { memoryPolicy: runtimeApi.memoryPolicy, configureMemoryPolicy: runtimeApi.configureMemoryPolicy, getJob: jobsApi.getJob };
export function PrefixCacheControls({ budget }: { budget: number }) {
  const { runtime, addJob, connectionRevision } = useRuntime();
  const jobs = useJobStore(state => state.jobs);
  const connectionScope = useConnectionScope();
  return <PrefixCacheControlsView budget={budget} runtime={runtime} jobs={jobs} connectionRevision={connectionRevision} services={services} connectionScope={connectionScope} addJob={addJob} notify={message => toast.success(message)} />;
}
