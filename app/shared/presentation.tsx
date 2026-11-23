import { createContext, useContext, type ComponentType, type ReactNode } from 'react';
import type { ReactElement, RefObject } from 'react';
import type { JobResource } from './types/api';
import type { LoadContextDefaults } from './types/load-context';

// What an application must hand to the shared display code: translation, the
// job snapshot it owns, load-context defaults and the basic controls it uses.
// The shared layer never reaches into Studio or the plugin Client itself.

export type Translate = (chinese: string, english: string) => string;

export interface SharedDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  description?: string;
  closeLabel: string;
  className?: string;
  children: ReactNode;
  returnFocusRef?: RefObject<HTMLElement | null>;
  initialFocusRef?: RefObject<HTMLElement | null>;
}

export interface SharedSwitchProps {
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
  label: string;
  disabled?: boolean;
  id?: string;
}

export interface SharedTooltipProps {
  content: ReactNode;
  children: ReactElement;
}

export interface SharedPrimitives {
  Dialog: ComponentType<SharedDialogProps>;
  Switch: ComponentType<SharedSwitchProps>;
  Tooltip: ComponentType<SharedTooltipProps>;
}

export interface SharedJobState {
  jobs: JobResource[];
}

export interface SharedPresentation {
  tr: Translate;
  useJobStore: <T>(selector: (state: SharedJobState) => T) => T;
  readLoadContext: (model?: string) => LoadContextDefaults;
  primitives: SharedPrimitives;
}

const PresentationContext = createContext<SharedPresentation | null>(null);

// Both applications mount PresentationProvider at their own root (Studio through
// SettingsProvider, the plugin Client through SharedPresenters). Unit tests that
// mock those roots opt into this default instead of wrapping every render.
let testDefault: SharedPresentation | null = null;

export function setSharedPresentationTestDefault(value: SharedPresentation | null) {
  testDefault = value;
}

export function PresentationProvider({ value, children }: { value: SharedPresentation; children: ReactNode }) {
  return <PresentationContext.Provider value={value}>{children}</PresentationContext.Provider>;
}

export function usePresentation(): SharedPresentation {
  const value = useContext(PresentationContext) ?? testDefault;
  if (!value) throw new Error('Shared presentation context is missing');
  return value;
}

export function useSharedDisplay(): SharedPresentation & SharedPrimitives {
  const value = usePresentation();
  return { ...value, ...value.primitives };
}
