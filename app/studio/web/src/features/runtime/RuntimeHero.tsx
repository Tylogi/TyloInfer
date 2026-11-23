import { RuntimeHeroView } from '@shared/features/runtime/RuntimeHeroView';
import { resolveRuntimeHero } from '@shared/features/runtime/runtimeHeroState';
import { useMemo } from 'react';
import { useNavigate } from 'react-router';
import { useRuntime } from '../../app/RuntimeProvider';
import { useJobStore } from '../../stores/jobStore';
import { useSettings } from '../settings/SettingsProvider';
import { STUDIO_PATHS } from '../../navigation';
import { openAIEndpoint } from './endpoint';
import { getApiBaseUrl } from '../../shared/api/client';

export function RuntimeHero() {
  const { runtime, instances, studio } = useRuntime();
  const jobs = useJobStore((state) => state.jobs);
  const { tr } = useSettings();
  const navigate = useNavigate();
  const openStudioPage = (localCheckpoints = false) => navigate(`${STUDIO_PATHS.models}${localCheckpoints ? '#local-checkpoints' : ''}`);
  const modelHero = useMemo(() => resolveRuntimeHero(runtime, instances, jobs), [runtime, instances, jobs]);
  const modelHeroStatus =
    modelHero.state === 'loading'
      ? tr('加载中', 'Loading')
      : modelHero.state === 'ready'
        ? tr('运行中', 'Running')
        : modelHero.state === 'failed'
          ? 'interrupted' in modelHero && modelHero.interrupted
            ? tr('加载中断', 'Interrupted') : tr('加载失败', 'Failed')
          : tr('空闲', 'Idle');
  return <RuntimeHeroView runtime={runtime} modelHero={modelHero} modelHeroStatus={modelHeroStatus}
    endpoint={openAIEndpoint(studio?.service_url || getApiBaseUrl() || window.location.origin)} onModels={openStudioPage} tr={tr} />;
}
