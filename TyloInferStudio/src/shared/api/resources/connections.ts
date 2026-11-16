
import type { RemoteNode } from '../types';
import { request, apiUrl, errorFromResponse, authorizedHeaders } from '../client';

export const connectionsApi = {

  async remoteNodes(refresh = false): Promise<RemoteNode[]> {
    return (
      await request<{ data: RemoteNode[] }>(
        `/api/cluster/nodes${refresh ? '?refresh=true' : ''}`,
      )
    ).data;
  },


  createRemoteNode(body: {
    name: string;
    url: string;
    api_key_env?: string | null;
    enabled: boolean;
  }): Promise<RemoteNode> {
    return request('/api/cluster/nodes', {
      method: 'POST',
      body: JSON.stringify(body),
    });
  },


  async deleteRemoteNode(id: string): Promise<void> {
    const response = await fetch(apiUrl(`/api/cluster/nodes/${id}`), {
      method: 'DELETE',
      headers: authorizedHeaders(),
    });
    if (!response.ok) throw await errorFromResponse(response);
  },

};
