
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type Dispatch,
  type ReactNode,
  type SetStateAction,
} from 'react';
import { loadSettings, SETTINGS_KEY, type GenerationSettings } from './configuration';
import { normalizeFontFamily } from './fontPreferences';

interface SettingsContextValue {
  settings: GenerationSettings;
  persistenceError: boolean;

  updateSettings: (patch: Partial<GenerationSettings>) => void;

  replaceSettings: Dispatch<SetStateAction<GenerationSettings>>;
  tr: (zh: string, en: string) => string;
  english: boolean;
  contextSize: number | null;

  setContextSize: Dispatch<SetStateAction<number | null>>;
}

const SettingsContext = createContext<SettingsContextValue | null>(null);


export function SettingsProvider({ children }: { children: ReactNode }) {
  const [settings, replaceSettings] = useState(loadSettings);
  const [contextSize, setContextSize] = useState<number | null>(null);
  const [persistenceError, setPersistenceError] = useState(false);
  const english =
    settings.language === 'en' ||
    (settings.language === 'system' && !navigator.language.toLowerCase().startsWith('zh'));
  const tr = useCallback((zh: string, en: string) => (english ? en : zh), [english]);
  const updateSettings = useCallback(
    (patch: Partial<GenerationSettings>) =>
      replaceSettings((current) => ({ ...current, ...patch })),
    [],
  );

  useEffect(() => {
    try {
      localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
      setPersistenceError(false);
    } catch {
      setPersistenceError(true);
    }
    const root = document.documentElement;
    root.dataset.theme = settings.theme;
    root.style.setProperty('--studio-content-font-size', `${settings.fontSize}px`);
    root.style.setProperty('--studio-code-font-size', `${settings.codeFontSize}px`);
    const textFont = normalizeFontFamily(settings.textFontFamily);
    const codeFont = normalizeFontFamily(settings.codeFontFamily);
    root.style.fontFamily = textFont ? `${textFont}, var(--text-font)` : '';
    if (codeFont) root.style.setProperty('--studio-code-font-family', `${codeFont}, monospace`);
    else root.style.removeProperty('--studio-code-font-family');
  }, [settings]);

  const value = useMemo(
    () => ({ settings, persistenceError, replaceSettings, updateSettings, english, tr, contextSize, setContextSize }),
    [settings, persistenceError, updateSettings, english, tr, contextSize],
  );
  return <SettingsContext.Provider value={value}>{children}</SettingsContext.Provider>;
}


export function useSettings() {
  const context = useContext(SettingsContext);
  if (!context) throw new Error('SettingsProvider is required');
  return context;
}
