import { useEffect, useState } from 'react';
import { useSettings } from './SettingsProvider';
import type { UiLanguage, UiTheme } from './configuration';
import { Icon, type IconName } from '../../app/display';
import studioPackage from '../../../../../package.json';
import './settings-panel.css';

const sections: { id: string; label: [string, string]; icon: IconName }[] = [
  { id: 'general', label: ['通用', 'General'], icon: 'settings' },
  { id: 'account', label: ['账户', 'Account'], icon: 'info' },
  { id: 'models', label: ['模型', 'Models'], icon: 'server-rack' },
  { id: 'agent-presets', label: ['Agent 预设', 'Agent presets'], icon: 'edit' },
  { id: 'plugins', label: ['插件', 'Plugins'], icon: 'plus' },
];

export function StudioSettingsPanel({ onOpenVersions }: { onOpenVersions: () => void }) {
  const { settings, updateSettings, persistenceError, tr } = useSettings();
  const [section, setSection] = useState('general');
  const [textFontDraft, setTextFontDraft] = useState(settings.textFontFamily);
  const [codeFontDraft, setCodeFontDraft] = useState(settings.codeFontFamily);
  useEffect(() => setTextFontDraft(settings.textFontFamily), [settings.textFontFamily]);
  useEffect(() => setCodeFontDraft(settings.codeFontFamily), [settings.codeFontFamily]);
  const updateDisplay = updateSettings;
  const unavailable = tr('此分区属于可选 Agent 产品，不是独立推理 Studio 的设置；如需使用，请打开 Agent 界面配置。', 'This section belongs to the optional Agent product, not the standalone inference Studio. Configure it in the Agent interface if needed.');
  return (
    <div className="studio-settings-layout">
      <nav aria-label={tr('设置分区', 'Settings sections')}>
        {sections.map((item) => (
          <button type="button" key={item.id} aria-current={item.id === section ? 'page' : undefined}
            onClick={() => setSection(item.id)}>
            <Icon name={item.icon} size={17} />{tr(...item.label)}
          </button>
        ))}
      </nav>
      <section className="studio-settings-content" aria-label={tr(...sections.find((item) => item.id === section)!.label)}>
        <h3>{tr(...sections.find((item) => item.id === section)!.label)}</h3>
        <p className="studio-settings-scope">{tr('TyloInfer Studio 是可独立运行的纯推理引擎界面，无需构建或启动 Agent。推理参数在各推理页面配置；此处提供显示设置，可选 Agent 的账户、预设和插件在其独立界面中管理。', 'TyloInfer Studio is a standalone inference interface and does not require an Agent build or runtime. Configure inference on its dedicated pages and display preferences here. Optional Agent accounts, presets and plugins are managed in the separate Agent interface.')}</p>
        {section === 'general' ? <>
          <p className="studio-settings-scope">{tr('以下显示设置仅保存于本浏览器，无需连接推理核心或 Agent。', 'Display settings are stored only in this browser, without connecting to the inference core or Agent.')}</p>
          {persistenceError && <p className="studio-settings-error" role="alert">{tr('设置已在当前页面生效，但浏览器存储不可用，刷新后可能丢失。', 'Changes apply to this page, but browser storage is unavailable. They may be lost on refresh.')}</p>}
          <label className="studio-settings-row">
            <span>{tr('外观', 'Appearance')}</span>
            <select value={settings.theme} onChange={(event) => updateDisplay({ theme: event.target.value as UiTheme })}>
              <option value="system">{tr('跟随系统', 'System')}</option>
              <option value="light">{tr('浅色', 'Light')}</option>
              <option value="dark">{tr('深色', 'Dark')}</option>
            </select>
          </label>
          <label className="studio-settings-row">
            <span>{tr('语言', 'Language')}</span>
            <select value={settings.language} onChange={(event) => updateDisplay({ language: event.target.value as UiLanguage })}>
              <option value="system">{tr('跟随浏览器', 'Browser default')}</option>
              <option value="zh-CN">简体中文</option>
              <option value="en">English</option>
            </select>
          </label>
          <label className="studio-settings-row">
            <span>{tr('正文字号', 'Text size')}</span>
            <span className="studio-settings-size"><input type="range" aria-label={tr('正文字号', 'Text size')} min={10} max={22} step={1}
              value={settings.fontSize} onChange={(event) => updateDisplay({ fontSize: Number(event.target.value) })} /><output>{settings.fontSize} px</output></span>
          </label>
          <details className="studio-settings-details">
            <summary>{tr('更多字体设置', 'More font settings')}</summary>
            <label className="studio-settings-row"><span>{tr('正文字体', 'Text font')}</span>
              <input type="text" placeholder={tr('默认字体', 'Default font')} value={textFontDraft} maxLength={256}
                onChange={(event) => { setTextFontDraft(event.target.value); updateSettings({ textFontFamily: event.target.value }); }} />
            </label>
            <label className="studio-settings-row"><span>{tr('代码字体', 'Code font')}</span>
              <input type="text" placeholder={tr('默认等宽字体', 'Default monospace')} value={codeFontDraft} maxLength={256}
                onChange={(event) => { setCodeFontDraft(event.target.value); updateSettings({ codeFontFamily: event.target.value }); }} />
            </label>
            <label className="studio-settings-row"><span>{tr('代码字号', 'Code size')}</span>
              <span className="studio-settings-size"><input type="range" aria-label={tr('代码字号', 'Code size')} min={10} max={16} step={1} value={settings.codeFontSize}
                onChange={(event) => updateDisplay({ codeFontSize: Number(event.target.value) })} /><output>{settings.codeFontSize} px</output></span>
            </label>
            <p className="studio-settings-hint">{tr('多个字体以逗号分隔；仅使用设备已有字体，不下载字体。', 'Separate font families with commas. Only existing device fonts are used; none are downloaded.')}</p>
            <div className="studio-settings-font-preview" aria-label={tr('字体预览', 'Font preview')}>
              <p>Aa 文字预览 0123456789</p><code>const value = 42;</code>
            </div>
            <fieldset disabled><legend>{tr('终端', 'Terminal')}</legend>
              <label className="studio-settings-row"><span>{tr('终端字号', 'Terminal size')}</span><input type="number" value={13} min={10} max={20} readOnly /></label>
              <label className="studio-settings-row"><span>{tr('终端字体', 'Terminal font')}</span><input value="" placeholder={tr('默认等宽字体', 'Default monospace')} readOnly /></label>
            </fieldset>
            <p className="studio-settings-hint">{tr('当前未接入终端，终端字体设置不可用。', 'A terminal is not connected. Terminal font settings are unavailable.')}</p>
          </details>
          <details className="studio-settings-details">
            <summary>{tr('对话显示', 'Chat display')}</summary>
            <p className="studio-settings-hint">{tr('原生对话显示功能尚未接入，以下字段仅展示默认值，不保存。', 'Native transcript features are not connected. These fields show defaults only and are not saved.')}</p>
            <fieldset disabled>
              <label className="studio-settings-row"><span>{tr('执行轨迹', 'Transcript view')}</span><select defaultValue="detailed"><option value="compact">Compact</option><option value="standard">Standard</option><option value="detailed">Detailed</option><option value="verbose">Verbose</option></select></label>
              <label className="studio-settings-row"><span>{tr('折叠时机', 'Collapse timing')}</span><select defaultValue="completion"><option value="completion">{tr('完成后', 'On completion')}</option><option value="next-input">{tr('下次输入', 'On next input')}</option></select></label>
              <label className="studio-settings-row"><span>{tr('折叠回复', 'Folded replies')}</span><select defaultValue="all"><option value="all">{tr('全部', 'All')}</option><option value="final">{tr('仅最终回复', 'Final only')}</option></select></label>
              <label className="studio-settings-row"><span>{tr('性能用量', 'Performance usage')}</span><select defaultValue="detailed"><option value="compact">{tr('简洁', 'Compact')}</option><option value="detailed">{tr('详细', 'Detailed')}</option></select></label>
              <label className="studio-settings-row"><span>{tr('打开链接', 'Open links')}</span><select defaultValue="sidebar"><option value="sidebar">{tr('侧栏', 'Sidebar')}</option><option value="new-tab">{tr('新标签页', 'New tab')}</option></select></label>
              <label className="studio-settings-row"><span>{tr('执行中按 Enter', 'Enter while busy')}</span><select defaultValue="queue"><option value="queue">{tr('排队', 'Queue')}</option><option value="steer">{tr('引导', 'Steer')}</option></select></label>
              <label className="studio-settings-row"><span>{tr('显示编码视图', 'Show coding view')}</span><input type="checkbox" checked readOnly /></label>
            </fieldset>
          </details>
          <div className="studio-settings-row"><span>{tr('当前版本', 'Current version')}</span><span>TyloInfer Studio {studioPackage.version}</span></div>
          <button className="studio-version-link" type="button" onClick={onOpenVersions}>{tr('版本管理', 'Versions')}<Icon name="info" size={16} /></button>
        </> : <div className="studio-settings-unavailable" role="status">
          <Icon name="info" size={22} /><p>{unavailable}</p>
          <p>{section === 'models'
            ? tr('此处是 Agent 模型提供商设置，不是左下角的本地推理模型管理。', 'These are Agent model-provider settings, separate from local inference model management.')
            : tr('不会读取或修改当前 DS Bot 的账户、预设或插件配置。', 'The current DS Bot account, preset and plugin configuration is not read or changed.')}</p>
        </div>}
      </section>
    </div>
  );
}
