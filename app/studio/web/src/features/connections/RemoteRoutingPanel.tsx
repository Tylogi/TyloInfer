import { connectionsApi } from '../../shared/api/resources/connections';
import { useRuntime } from '../../app/RuntimeProvider';
import { useConnectionScope } from '../../app/useConnectionScope';
import { toast } from '../../stores/toastStore';
import { RemoteRoutingView } from '@shared/features/connections/RemoteRoutingView';

export function RemoteRoutingPanel() {
  const { ready, connectionRevision } = useRuntime();
  const connectionScope = useConnectionScope();
  return <RemoteRoutingView ready={ready} connectionRevision={connectionRevision} connectionScope={connectionScope}
    services={connectionsApi} reportError={toast.error} />;
}
