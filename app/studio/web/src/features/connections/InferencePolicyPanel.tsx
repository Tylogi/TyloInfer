import { useEffect, useRef, useState } from 'react';
import { errorMessage } from '@shared/formatters';
import { useRuntime } from '../../app/RuntimeProvider';
import { runtimeApi } from '../../shared/api/resources/runtime';
import { InferencePolicyView } from '@shared/features/connections/InferencePolicyView';
import { toast } from '../../stores/toastStore';
import { useSettings } from '../settings/SettingsProvider';

export function InferencePolicyPanel() {
  const { runtime, ready, refreshRuntime, connectionRevision } = useRuntime();
  const { settings, updateSettings, tr } = useSettings();
  const [busy, setBusy] = useState(false);
  const revision = useRef(connectionRevision);
  revision.current = connectionRevision;
  useEffect(() => { setBusy(false); }, [connectionRevision]);
  const enabled = runtime?.mtp_service_enabled;

  async function changeMtp(next: boolean) {
    if (busy || !ready || typeof enabled !== 'boolean') return;
    const currentRevision = connectionRevision;
    setBusy(true);
    try {
      await runtimeApi.configureInferencePolicy(next);
      if (revision.current !== currentRevision) return;
      await refreshRuntime();
      if (revision.current === currentRevision) toast.success(tr('服务 MTP 设置已保存', 'Service MTP setting saved'));
    } catch (cause) {
      if (revision.current === currentRevision) toast.error(errorMessage(cause, tr));
    } finally {
      if (revision.current === currentRevision) setBusy(false);
    }
  }

  return <InferencePolicyView enabled={enabled} ready={ready} busy={busy} changeMtp={next => void changeMtp(next)} visionEnabled={settings.enableVision} changeVision={next => updateSettings({ enableVision: next, inheritModelDefaults: false })} tr={tr} />;
}
