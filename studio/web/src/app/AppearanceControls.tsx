import { MoonIcon, SunIcon } from '@phosphor-icons/react';
import { useSettings } from '../features/settings/SettingsProvider';

export function AppearanceControls({ section = 'all' }: { section?: 'all' | 'theme' | 'language' }) {
  const { settings, updateSettings, tr } = useSettings();
  return <div className="appearance-controls" role="group" aria-label={section === 'theme' ? tr('明暗设置', 'Theme settings') : section === 'language' ? tr('语言设置', 'Language settings') : tr('明暗与语言', 'Appearance and language')}>
    {section !== 'language' && <div role="group" aria-label={tr('明暗', 'Appearance')}>
      {(['light', 'dark', 'system'] as const).map(value => <button key={value} type="button"
        aria-label={value === 'light' ? tr('浅色', 'Light') : value === 'dark' ? tr('深色', 'Dark') : tr('自动明暗', 'Automatic appearance')}
        aria-pressed={settings.theme === value}
        onClick={() => { updateSettings({ theme: value }); }}>
        {value === 'light' ? <SunIcon size={15} /> : value === 'dark' ? <MoonIcon size={15} /> : 'Auto'}
      </button>)}
    </div>}
    {section === 'all' && <span className="appearance-controls-divider" aria-hidden="true" />}
    {section !== 'theme' && <div role="group" aria-label={tr('语言', 'Language')}>
      {(['en', 'zh-CN', 'system'] as const).map(value => <button key={value} type="button"
        aria-label={value === 'en' ? 'English' : value === 'zh-CN' ? '简体中文' : tr('自动语言', 'Automatic language')}
        aria-pressed={settings.language === value}
        onClick={() => { updateSettings({ language: value }); }}>
        {value === 'en' ? 'EN' : value === 'zh-CN' ? 'CN' : 'Auto'}
      </button>)}
    </div>}
  </div>;
}
