import type { ComponentProps } from 'react';
import { modelsApi } from '../../shared/api/resources/models';
import { jobsApi } from '../../shared/api/resources/jobs';
import { openStudioExternal } from '../../shared/platform/studio';
import { useDownloadReuse } from './useDownloadReuse';
import { CommunityModelBrowserView } from './CommunityModelBrowserView';
const services = { modelsApi, createJob: jobsApi.createJob, openExternal: openStudioExternal, useDownloadReuse };
export function CommunityModelBrowser(props: Omit<ComponentProps<typeof CommunityModelBrowserView>, 'services'>) {
  return <CommunityModelBrowserView {...props} services={services} />;
}
