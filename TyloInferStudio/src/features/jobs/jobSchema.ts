import type { JobResource } from '../../shared/api/types';

export const TERMINAL_JOB_STATUSES = new Set<JobResource["status"]>([
  "succeeded",
  "failed",
  "cancelled",
  "interrupted",
]);

/** 判断后台任务是否已经结束，供轮询停止与状态展示使用。 */
export function isTerminalJob(job: JobResource): boolean {
  return TERMINAL_JOB_STATUSES.has(job.status);
}
