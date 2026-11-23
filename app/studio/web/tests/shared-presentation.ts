import { setSharedPresentationTestDefault, type Translate } from '@shared/presentation';
import { DEFAULT_LOAD_CONTEXT, readLoadContext } from '../src/features/models/loadContextDefaults';
import { Dialog } from '../src/shared/ui/Dialog';
import { Switch } from '../src/shared/ui/Switch';
import { Tooltip } from '../src/shared/ui/Tooltip';
import { useJobStore } from '../src/stores/jobStore';

// The app roots mount the shared presentation; unit tests that mock
// SettingsProvider install the same wiring here, with their own language.
export function useStudioSharedPresentation(tr: Translate) {
  setSharedPresentationTestDefault({
    tr,
    useJobStore: (selector) => useJobStore(selector),
    readLoadContext,
    primitives: { Dialog, Switch, Tooltip },
  });
}
