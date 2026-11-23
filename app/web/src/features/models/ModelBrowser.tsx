import { modelsApi } from '../../shared/api/resources/models';
import { jobsApi } from '../../shared/api/resources/jobs';
import { openStudioExternal } from '../../shared/platform/studio';
import { useDownloadReuse } from './useDownloadReuse';
import { ModelBrowserView, type ModelBrowserProps } from '@shared/features/models/ModelBrowserView';
export type { ModelBrowserTab, DownloadOrigin } from '@shared/features/models/ModelBrowserView';
const services = { modelsApi, createJob: jobsApi.createJob, openExternal: openStudioExternal, useDownloadReuse };
export function ModelBrowser(props: ModelBrowserProps) { return <ModelBrowserView {...props} services={services} />; }
