import type { DatasetResource, EvaluationResult, EvaluationComparison, EvaluationTools, OfficialDataset } from '../types';
import { request, apiUrl, errorFromResponse, authorizedHeaders } from '../client';

export const evaluationsApi = {
  tools(): Promise<EvaluationTools> {
    return request('/api/evaluations/tools');
  },
  async catalog(): Promise<OfficialDataset[]> {
    return (await request<{ data: OfficialDataset[] }>('/api/datasets/catalog')).data;
  },
  async datasets(): Promise<DatasetResource[]> {
    return (await request<{ data: DatasetResource[] }>('/api/datasets')).data;
  },

  createDataset(body: {
    name: string;
    kind: DatasetResource['kind'];
    artifact_uri: string;
    source_uri?: string | null;
    revision?: string | null;
    metadata?: Record<string, unknown>;
  }): Promise<DatasetResource> {
    return request('/api/datasets', { method: 'POST', body: JSON.stringify(body) });
  },

  async deleteDataset(id: string): Promise<void> {
    const response = await fetch(apiUrl(`/api/datasets/${id}`), {
      method: 'DELETE',
      headers: authorizedHeaders(),
    });
    if (!response.ok) throw await errorFromResponse(response);
  },

  async evaluations(limit = 200): Promise<EvaluationResult[]> {
    return (await request<{ data: EvaluationResult[] }>(`/api/evaluations?limit=${limit}`)).data;
  },

  compareEvaluations(evaluationIds: string[]): Promise<EvaluationComparison> {
    return request('/api/evaluations/compare', {
      method: 'POST',
      body: JSON.stringify({ evaluation_ids: evaluationIds }),
    });
  },
};
