import { useState } from 'react';
import { SectionLabel, SettingRow, TMPanel } from '../../app/display';
import type { ModelArtifact, RuntimeInstance } from '../../shared/api/types';
import { Switch } from '../../shared/ui/Switch';

export const flashNextDraft = {
  repo_id: 'incoai/Qwen3.8-Flash-Next-DFlash2',
  revision: 'a22a1744e85c17364dcc54a6a420af7135d4c7f6',
  destination: 'huggingface/incoai/Qwen3.8-Flash-Next-DFlash2',
  include: ['config.json', 'model.safetensors'],
};

export function DFlash2Controls({ model, artifacts, instance, backend, disabled, downloadAvailable, downloadPending,
  loadReady = true, serviceEnabled, onDownload, onLoad, onToggle, tr }: {
  model?: ModelArtifact;
  artifacts: ModelArtifact[];
  instance?: RuntimeInstance;
  backend?: string;
  disabled?: boolean;
  downloadAvailable?: boolean;
  downloadPending?: boolean;
  loadReady?: boolean;
  serviceEnabled?: boolean;
  onDownload: () => Promise<unknown>;
  onLoad: (draftId: string) => Promise<unknown>;
  onToggle?: (enabled: boolean) => Promise<unknown>;
  tr: (zh: string, en: string) => string;
}) {
  const [selected, setSelected] = useState('');
  const [pending, setPending] = useState(false);
  const [confirm, setConfirm] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  if (!model?.architecture.toLowerCase().startsWith('qwen4_exp')
    || (backend && backend !== 'cuda' && backend !== 'metal')) return null;

  const drafts = artifacts.filter(item => item.asset_kind === 'dflash2' && item.complete);
  const draft = drafts.find(item => item.id === selected)
    ?? drafts.find(item => item.name === 'Qwen3.8-Flash-Next-DFlash2') ?? drafts[0];
  const busy = Boolean(disabled || pending);
  const loaded = Boolean(instance?.dflash2_draft && ['ready', 'busy'].includes(instance.state));
  const reloadBusy = instance && (instance.state !== 'ready' || instance.queued_requests > 0);
  const loadDisabled = busy || !loadReady || !model.loadable || !draft || reloadBusy || instance?.dflash2_draft === draft.name;
  const status = instance?.dflash2_draft
    ? loaded ? tr('已加载', 'Loaded') + ' · ' + instance.dflash2_draft : tr('草稿加载中', 'Loading draft')
    : draft ? tr('草稿已下载', 'Draft downloaded') : downloadPending ? tr('草稿下载中', 'Downloading draft')
      : tr('草稿未下载', 'Draft not downloaded');

  async function run(action: () => Promise<unknown>, message?: string) {
    if (busy) return;
    setPending(true); setError(''); setNotice('');
    try {
      const result = await action();
      if (result !== null && result !== false && message) setNotice(message);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setPending(false);
    }
  }

  return <>
    <SectionLabel title="DFlash2" />
    <TMPanel>
      <SettingRow title={tr('草稿模型', 'Draft model')} detail={status}
        trailing={drafts.length > 1 ? <select aria-label={tr('DFlash2 草稿', 'DFlash2 draft')}
          value={draft?.id ?? ''} disabled={busy}
          onChange={event => { setSelected(event.target.value); setConfirm(false); }}>
          {drafts.map(item => <option key={item.id} value={item.id}>{item.name}</option>)}
        </select> : undefined} />
      <div className="model-settings-actions">
        <button type="button" disabled={busy || loaded || downloadPending || downloadAvailable === false || drafts.some(item => item.name === 'Qwen3.8-Flash-Next-DFlash2')}
          onClick={() => void run(onDownload, tr('已加入下载队列', 'Added to download queue'))}>
          {tr('下载 DFlash2 草稿', 'Download DFlash2 draft')}
        </button>
        <button type="button" disabled={Boolean(loadDisabled)} onClick={() => {
          if (!draft) return;
          if (instance) setConfirm(true);
          else void run(() => onLoad(draft.id));
        }}>{tr('加载 DFlash2 草稿', 'Load DFlash2 draft')}</button>
      </div>
      {confirm && draft && <div className="model-settings-actions">
        <span>{tr('加载会重载主模型并清空活动 KV。', 'Loading reloads the target model and clears its active KV.')}</span>
        <button type="button" onClick={() => setConfirm(false)}>{tr('取消', 'Cancel')}</button>
        <button type="button" disabled={Boolean(loadDisabled)} onClick={() => {
          setConfirm(false);
          void run(() => onLoad(draft.id));
        }}>{tr('重载并加载', 'Reload and load')}</button>
      </div>}
      {downloadAvailable === false && <p role="status">{tr('当前服务未启用 Hugging Face 下载。', 'Hugging Face downloads are disabled on this server.')}</p>}
      {loaded && onToggle && <SettingRow title={tr('启用 DFlash2', 'Enable DFlash2')}
        detail={serviceEnabled === false ? tr('服务 MTP 总开关已关闭。', 'Service MTP is off.') :
          tr('关闭后，此模型实例使用普通解码。', 'When off, this model instance uses ordinary decoding.')}
        trailing={<Switch label={tr('启用 DFlash2', 'Enable DFlash2')} checked={instance?.dflash2_enabled === true}
          disabled={busy || !['ready', 'busy'].includes(instance?.state ?? '')}
          onCheckedChange={enabled => void run(() => onToggle(enabled))} />} />}
      {notice && <p role="status">{notice}</p>}
      {error && <p role="alert">{error}</p>}
    </TMPanel>
  </>;
}
