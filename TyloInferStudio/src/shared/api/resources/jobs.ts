
import type { RuntimeLogEntry, JobResource, JobEventResource } from '../types';
import { ApiError, request, apiUrl, errorFromResponse, authorizedHeaders } from '../client';
import type { ApiErrorBody } from '../types';
import { readEventStream } from '../eventStream';

export const jobsApi = {
  getJob(id: string): Promise<JobResource> {
    return request(`/api/jobs/${id}`);
  },

  async jobs(limit = 100): Promise<JobResource[]> {
    return (await request<{ data: JobResource[] }>(`/api/jobs?limit=${limit}`)).data;
  },


  createJob(kind: string, payload: Record<string, unknown>): Promise<JobResource> {
    return request('/api/jobs', {
      method: 'POST',
      body: JSON.stringify({ kind, payload }),
    });
  },


  cancelJob(id: string): Promise<JobResource> {
    return request(`/api/jobs/${id}/cancel`, { method: 'POST' });
  },


  retryJob(id: string): Promise<JobResource> {
    return request(`/api/jobs/${id}/retry`, { method: 'POST' });
  },


  async deleteJob(id: string): Promise<void> {
    const response = await fetch(apiUrl(`/api/jobs/${id}`), {
      method: 'DELETE',
      headers: authorizedHeaders(),
    });
    if (!response.ok) throw await errorFromResponse(response);
  },


  async clearCompletedJobs(): Promise<void> {
    const response = await fetch(apiUrl('/api/jobs/completed'), {
      method: 'DELETE',
      headers: authorizedHeaders(),
    });
    if (!response.ok) throw await errorFromResponse(response);
  },


  async jobEvents(id: string, signal?: AbortSignal): Promise<RuntimeLogEntry[]> {
    const response = await request<{
      data: Array<{
        sequence: number;
        level: RuntimeLogEntry['level'];
        message?: string | null;
        data: Record<string, unknown>;
        created_at: string;
      }>;
    }>(`/api/jobs/${id}/events?limit=1000&tail=true`, { signal });
    return response.data
      .filter((event) => event.message)
      .map((event) => ({
        sequence: event.sequence,
        level: event.level,
        message: event.message || '',
        fields: event.data,
        created_at: event.created_at,
      }));
  },

  /** Consume job SSE until cancellation, propagating native stream error envelopes. */
  async streamJobEvents(
    id: string,
    onEvent: (event: JobEventResource) => void,
    signal: AbortSignal,
    after = 0,
  ): Promise<void> {
    const response = await fetch(apiUrl(`/api/jobs/${id}/events/stream${after > 0 ? `?after=${after}` : ''}`), {
      headers: authorizedHeaders({ Accept: 'text/event-stream' }),
      signal,
    });
    if (!response.ok) throw await errorFromResponse(response);
    await readEventStream<JobEventResource | ApiErrorBody>(response, (event) => {
      if ('error' in event) throw new ApiError(502, event);
      onEvent(event);
    }, signal);
  },
};
