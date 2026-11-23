import { useMemo, type ReactNode } from 'react';
import { PresentationProvider, type SharedPresentation } from '@shared/presentation';
import { useSettings } from '../features/settings/SettingsProvider';
import { useJobStore } from '../stores/jobStore';
import { readLoadContext } from '../features/models/loadContextDefaults';
import { Dialog } from '../shared/ui/Dialog';
import { Switch } from '../shared/ui/Switch';
import { Tooltip } from '../shared/ui/Tooltip';

// Studio hands the shared display layer what it needs: the active translation,
// its own job store, the persisted load-context defaults and these controls.
export function SharedDisplay({ children }: { children: ReactNode }) {
  const { tr } = useSettings();
  const value = useMemo<SharedPresentation>(() => ({
    tr,
    useJobStore: (selector) => useJobStore(selector),
    readLoadContext,
    primitives: { Dialog, Switch, Tooltip },
  }), [tr]);
  return <PresentationProvider value={value}>{children}</PresentationProvider>;
}
