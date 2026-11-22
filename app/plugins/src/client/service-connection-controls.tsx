import React, { useEffect, useState } from 'react';
import { SettingRow } from '../../../studio/web/src/app/display.tsx';
import { CoreError } from './page-kit.tsx';

export function ListenerControls({ listener, api, mutation, disabled, tr }) {
  const [openai, setOpenAI] = useState(String(listener.port));
  const [anthropic, setAnthropic] = useState(listener.anthropic_port == null ? '' : String(listener.anthropic_port));
  useEffect(() => { setOpenAI(String(listener.port)); }, [listener.port]);
  useEffect(() => { setAnthropic(listener.anthropic_port == null ? '' : String(listener.anthropic_port)); }, [listener.anthropic_port]);

  return [
    ['openai', tr('OpenAI 端口', 'OpenAI port'), listener.port, openai, setOpenAI],
    ['anthropic', tr('Anthropic 端口', 'Anthropic port'), listener.anthropic_port, anthropic, setAnthropic],
  ].map(([protocol, label, current, draft, setDraft]) => {
    const port = Number(draft);
    const other = protocol === 'openai' ? listener.anthropic_port : listener.port;
    const blocked = disabled || !listener.configurable;
    const canApply = !blocked && Number.isInteger(port) && port >= 1 && port <= 65535 && port !== current && port !== other;
    return <SettingRow key={protocol} title={label} detail={protocol === 'openai' && listener.configurable
      ? tr('修改后 Agent 会更新连接地址。', 'Agent updates its connection address after a port change.') : undefined}
      trailing={<form className="server-row-actions" onSubmit={event => {
        event.preventDefault();
        if (canApply) void mutation.run(signal => api.configureListener(protocol, port, signal));
      }}>
        <input className="server-number-input" type="number" aria-label={label} min={1} max={65535}
          placeholder={tr('未启用', 'Not enabled')} value={draft} disabled={blocked} onChange={event => setDraft(event.target.value)} />
        <button type="submit" disabled={!canApply}>{current == null ? tr('启用', 'Enable') : tr('应用', 'Apply')}</button>
      </form>} />;
  });
}

export function AuthenticationControls({ entry, api, mutation, disabled, tr }) {
  const policy = entry?.value;
  const [enabled, setEnabled] = useState(Boolean(policy?.enabled));
  const [key, setKey] = useState('');
  useEffect(() => { setEnabled(Boolean(policy?.enabled)); setKey(''); }, [policy?.enabled]);
  const validKey = /^[\x21-\x7e]{1,512}$/.test(key);
  const canApply = !disabled && policy && (enabled ? validKey || !key && !policy.enabled && policy.has_key : policy.enabled);
  if (entry?.error) return <CoreError error={entry.error} />;
  async function apply(event) {
    event.preventDefault();
    if (!canApply) return;
    await mutation.run(signal => api.configureAuthentication(enabled, enabled ? key : undefined, signal));
    setKey('');
  }
  return <form className="setting-list" onSubmit={apply}>
    <SettingRow title={tr('鉴权', 'Authentication')}
      detail={tr('OpenAI 与 Anthropic API 共用。Agent 在 Host 凭据库保存所用密钥。', 'OpenAI and Anthropic APIs share this policy. Agent stores its key in the Host credential store.')}
      trailing={<div className="server-row-actions">
        <select aria-label={tr('鉴权', 'Authentication')} disabled={!policy || disabled} value={enabled ? 'api-key' : 'none'}
          onChange={event => { setEnabled(event.target.value === 'api-key'); setKey(''); }}>
          <option value="none">{tr('无', 'None')}</option><option value="api-key">API Key</option>
        </select>
        <button type="submit" disabled={!canApply}>{tr('应用鉴权', 'Apply authentication')}</button>
      </div>} />
    {enabled && <SettingRow title="API Key" trailing={<input aria-label="API Key" className="server-wide-input" type="password"
      autoComplete="new-password" spellCheck={false} maxLength={512} disabled={disabled} value={key}
      placeholder={policy?.enabled ? tr('输入新 API Key', 'Enter a new API key') : policy?.has_key
        ? tr('留空使用已配置的密钥', 'Leave blank to use the configured key') : tr('输入 API Key', 'Enter API key')}
      onChange={event => setKey(event.target.value)} />} />}
  </form>;
}
