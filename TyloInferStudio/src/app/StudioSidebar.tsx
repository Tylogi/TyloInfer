import { useEffect, useRef, useState } from 'react';
import { useLocation, useNavigate } from 'react-router';
import { useSettings } from '../features/settings/SettingsProvider';
import { StudioSettingsPanel } from '../features/settings/StudioSettingsPanel';
import { useUiStore } from '../stores/uiStore';
import { Icon, type IconName } from './display';
import { Dialog } from '../shared/ui/Dialog';
import tyloInferMark from '../assets/tylogi-ai-lab-mark.svg';

const inferenceLinks: { path: string; label: [string, string]; icon: IconName }[] = [
  { path: '/inference', label: ['概览', 'Overview'], icon: 'gauge' },
  { path: '/models', label: ['模型', 'Models'], icon: 'folder' },
  { path: '/runtime', label: ['服务', 'Service'], icon: 'server-rack' },
  { path: '/resources', label: ['资源', 'Resources'], icon: 'droplet' },
  { path: '/model-hub', label: ['下载', 'Downloads'], icon: 'download' },
  { path: '/analysis', label: ['分析', 'Analysis'], icon: 'search' },
  { path: '/logs', label: ['日志', 'Logs'], icon: 'scroll' },
];

export function StudioSidebar() {
  const location = useLocation();
  const navigate = useNavigate();
  const { tr } = useSettings();
  const sidebarOpen = useUiStore((state) => state.sidebarOpen);
  const closeSidebar = useUiStore((state) => state.closeSidebar);
  const openSidebar = useUiStore((state) => state.openSidebar);
  const [engineOpen, setEngineOpen] = useState(true);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const settingsTrigger = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    closeSidebar();
  }, [location.pathname, location.search, closeSidebar]);
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (
        event.key === 'Escape' && !settingsOpen && !event.defaultPrevented &&
        !(event.target instanceof Element && event.target.closest('[role="dialog"]'))
      ) closeSidebar();
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [closeSidebar, settingsOpen]);

  function open(path: string) {
    setSettingsOpen(false);
    navigate(path);
    closeSidebar();
  }

  return (
    <>
      <aside
        className={`sidebar studio-session-sidebar ${sidebarOpen ? 'open' : ''}`}
        id="studio-sidebar"
        aria-label={tr('推理 Studio', 'Inference Studio')}
      >
        <button
          className="studio-brand"
          type="button"
          onClick={() => open('/')}
          aria-label={tr('返回概览', 'Back to overview')}
        >
          <img src={tyloInferMark} alt="" />
          <span>TyloInfer</span>
        </button>
        <div className="studio-sidebar-footer">
          <nav
            id="studio-inference-navigation"
            className="studio-inference-navigation"
            aria-label={tr('推理导航', 'Inference navigation')}
            hidden={!engineOpen}
          >
            {inferenceLinks.map((item) => (
              <button
                key={item.path}
                type="button"
                aria-current={location.pathname === item.path ? 'page' : undefined}
                onClick={() => open(item.path)}
              >
                <Icon name={item.icon} size={17} />
                {tr(...item.label)}
              </button>
            ))}
          </nav>
          <button
            type="button"
            className="studio-engine-trigger"
            aria-controls="studio-inference-navigation"
            aria-expanded={engineOpen}
            onClick={() => setEngineOpen((current) => !current)}
          >
            <Icon name="server-rack" size={18} />
            <span>{tr('推理引擎', 'Inference engine')}</span>
            <span className="studio-engine-triangle" aria-hidden="true" />
          </button>
          <button
            type="button"
            ref={settingsTrigger}
            className="studio-settings-trigger"
            aria-haspopup="dialog"
            aria-expanded={settingsOpen}
            onClick={() => setSettingsOpen(true)}
          >
            <Icon name="settings" size={18} />
            <span>{tr('设置', 'Settings')}</span>
          </button>
        </div>
      </aside>
      <button
        aria-controls="studio-sidebar"
        aria-expanded={sidebarOpen}
        aria-label={tr('打开侧栏', 'Open sidebar')}
        className="mobile-menu-trigger"
        onClick={openSidebar}
        type="button"
      >
        <Icon name="menu" size={18} />
      </button>
      <button
        aria-label={tr('关闭侧栏', 'Close sidebar')}
        className={`mobile-scrim ${sidebarOpen ? 'open' : ''}`}
        onClick={closeSidebar}
        type="button"
      />
      <Dialog
        open={settingsOpen}
        onOpenChange={setSettingsOpen}
        title={tr('设置', 'Settings')}
        closeLabel={tr('关闭', 'Close')}
        className="studio-settings-panel"
        returnFocusRef={settingsTrigger}
      >
        <StudioSettingsPanel onOpenVersions={() => open('/versions')} />
      </Dialog>
    </>
  );
}
