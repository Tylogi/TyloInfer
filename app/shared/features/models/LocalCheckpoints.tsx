import { useSharedDisplay } from '../../presentation';
import { useEffect, useId, useState, type CSSProperties } from 'react';
import { ListToolbar } from '@shared/ui/ListToolbar';
import { Icon, SectionLabel, TMPanel, EmptyPanel } from '@shared/ui/display';
import type { ModelCatalog } from '../../types/catalog';
import { ModelVendorMark } from '@shared/ui/ModelVendorMark';
import { ModelLoadProgress } from '@shared/features/models/ModelLoadProgress';
import { formatNumber } from '@shared/formatters';
import { ModelPressureBars } from '@shared/features/models/ModelMemoryPressure';
import { availableResidentMemory, hasSeparateVram } from '@shared/features/runtime/memoryArchitecture';
import { ModelDeleteDialog } from '@shared/features/models/ModelDeleteDialog';
import { availableLoadPools, estimateModelWeights } from '@shared/features/models/loadMemoryEstimate';
import { groupCheckpoints } from '@shared/features/models/checkpointGroups';

export function LocalCheckpoints({ catalog, capabilities = {}, companionMode = 'path' }: { catalog: ModelCatalog; capabilities?: { files?: boolean; settings?: boolean; deletion?: boolean; companions?: boolean }; companionMode?: 'path' | 'registered-id' | 'both' }) {
  const { tr, readLoadContext } = useSharedDisplay();
  const { runtime, artifacts, busy, instances, modelFilter,
    setModelFilter, filteredArtifacts, unloadInstance, loadArtifact, chooseModelDirectory, openModelFiles, prepareModelDeletion } = catalog;
  const [status, setStatus] = useState('all');
  const [architecture, setArchitecture] = useState('all');
  const [format, setFormat] = useState('all');
  const [sort, setSort] = useState('recent');
  const [companions, setCompanions] = useState<Record<string, string>>({});
  const headingId = useId();
  const isComponent = (item: typeof artifacts[number]) => !!item.asset_kind && item.asset_kind !== 'model';
  const components = artifacts.filter(isComponent).length;
  useEffect(() => { setCompanions({}); }, [catalog.connectionRevision]);
  const activeInstances = new Map(instances.filter(item => item.state !== 'failed').map(item => [item.model, item]));
  const visibleArtifacts = filteredArtifacts.filter(item =>
    (architecture === 'all' || item.architecture === architecture) && (format === 'all' || item.format === format) &&
    (status === 'all' || status === 'loaded' && activeInstances.has(item.name) || status === 'loadable' && item.loadable ||
      status === 'incomplete' && !item.complete || status === 'components' && isComponent(item) || status === 'unavailable' && !item.loadable && !isComponent(item)))
    .sort((a, b) => sort === 'name' ? a.name.localeCompare(b.name) : sort === 'size'
      ? b.total_bytes - a.total_bytes : (Date.parse(b.modified_at) || 0) - (Date.parse(a.modified_at) || 0));
  const families = groupCheckpoints(visibleArtifacts, artifacts);
  const filterCount = [status, architecture, format].filter(value => value !== 'all').length;
  const separate = hasSeparateVram(runtime);
  const available = availableResidentMemory(runtime);
  const { vram, ram } = availableLoadPools(runtime, catalog.loadDeviceIds);
  const loadMemory = separate ? vram == null || ram == null ? null : vram + ram : available;
  const system = separate ? { memory_pools: [
    { kind: 'vram' as const, capacity_bytes: vram },
    { kind: 'ram' as const, capacity_bytes: ram },
  ] } : undefined;
  const gib = (bytes?: number | null) => bytes == null ? '—' : `${formatNumber(bytes / 2 ** 30, 1)} GiB`;
  return (
    <>
      <SectionLabel
        title={tr('本地检查点', 'Local checkpoints')}
        subtitle={tr(`${artifacts.length - components} 个本地模型${components ? ` · ${components} 个配套资产` : ''}`,
          `${artifacts.length - components} local models${components ? ` · ${components} companion assets` : ''}`)}
      />
      <ListToolbar label={tr('搜索本地检查点', 'Search local checkpoints')} placeholder={tr('按模型名称搜索', 'Search model names')}
        query={modelFilter} onQueryChange={setModelFilter} count={visibleArtifacts.length} total={artifacts.length}
        afterFilters={!separate && <small className="checkpoint-available-memory">{tr('剩余可用内存', 'Remaining available memory')} <span>{gib(available)}</span></small>}
        activeFilters={filterCount} onReset={() => { setModelFilter(''); setStatus('all'); setArchitecture('all'); setFormat('all'); setSort('recent'); }}>
        <label>{tr('模型状态', 'Model status')}<select aria-label={tr('模型状态', 'Model status')} value={status} onChange={event => setStatus(event.target.value)}>
          <option value="all">{tr('全部', 'All')}</option><option value="loaded">{tr('已加载', 'Loaded')}</option>
          <option value="loadable">{tr('可直接加载', 'Loadable')}</option><option value="incomplete">{tr('分片不全', 'Incomplete shards')}</option><option value="unavailable">{tr('不可直接加载', 'Not loadable')}</option>
          <option value="components">{tr('配套资产', 'Companion assets')}</option>
        </select></label>
        <label>{tr('架构', 'Architecture')}<select aria-label={tr('筛选架构', 'Filter architecture')} value={architecture} onChange={event => setArchitecture(event.target.value)}>
          <option value="all">{tr('全部', 'All')}</option>{[...new Set(artifacts.map(item => item.architecture).filter(Boolean))].sort().map(value => <option key={value} value={value}>{value}</option>)}
        </select></label>
        <label>{tr('格式', 'Format')}<select aria-label={tr('筛选格式', 'Filter format')} value={format} onChange={event => setFormat(event.target.value)}>
          <option value="all">{tr('全部', 'All')}</option><option value="tq">TQ</option><option value="hf">HuggingFace / MLX</option><option value="gguf">GGUF</option>
        </select></label>
        <label>{tr('排序', 'Sort')}<select aria-label={tr('检查点排序', 'Checkpoint sort')} value={sort} onChange={event => setSort(event.target.value)}>
          <option value="recent">{tr('最近修改', 'Recently modified')}</option><option value="name">{tr('模型名称', 'Model name')}</option><option value="size">{tr('文件大小', 'File size')}</option>
        </select></label>
      </ListToolbar>
      {visibleArtifacts.length > 0 ? (
        <div className="checkpoint-family-groups">
          {families.map((family, familyIndex) => <TMPanel className="model-catalog-panel model-library-panel checkpoint-family-panel" key={family.key}>
            <header className="checkpoint-family-heading">
              <h3>{family.key === 'jev' ? tr('Jev 决策模型', 'Jev decision models') : family.name || tr('其它', 'Other')}</h3>
              <span>{(() => {
                const count = family.models.reduce((sum, group) => sum + group.artifacts.filter(isComponent).length, 0);
                return tr(`${family.count - count} 个版本${count ? ` · ${count} 个配套资产` : ''}`,
                  `${family.count - count} variants${count ? ` · ${count} companion assets` : ''}`);
              })()}</span>
            </header>
            <div className="model-list" style={{ '--checkpoint-group-rows': family.models.map(group => `auto repeat(${group.artifacts.length}, minmax(0, 1fr))`).join(' ') } as CSSProperties}>
              {family.models.map((group, modelIndex) => <section className="checkpoint-model-group" key={group.key}
                aria-labelledby={`${headingId}-${familyIndex}-${modelIndex}`}>
                <h4 className="checkpoint-model-heading" id={`${headingId}-${familyIndex}-${modelIndex}`}>{group.name}</h4>
            {group.artifacts.map((item) => {
              const component = isComponent(item);
              const companion = companions[item.id]?.trim();
              const isCompanionId = /^[0-9a-f]{32}$/.test(companion || '');
              const validCompanion = Boolean(companion && (companionMode === 'path'
                || (companionMode === 'both' && /^(?:\/(?!\/)|[A-Za-z]:[\\/])/.test(companion))
                || artifacts.some(candidate => candidate.id === companion && candidate.id !== item.id && candidate.format === 'gguf' && candidate.complete === true && isCompanionId)));
              const componentLabel = item.asset_kind === 'engram' ? tr('可独立使用的Engram', 'Independently usable Engram')
                : item.asset_kind === 'dflash2' ? item.architecture === 'dfly2' ? tr('DFly2 草稿', 'DFly2 draft')
                  : item.architecture === 'dspark' ? tr('DSpark 草稿', 'DSpark draft')
                  : item.dflash2_target ? tr('DFlash2 草稿', 'DFlash2 draft')
                  : tr('DFlash2 草稿 · 目标未知', 'DFlash2 draft · Target unknown') : tr('可独立使用的PLE', 'Independently usable PLE');
              const instance = activeInstances.get(item.name);
              const loaded = Boolean(instance);
              const placement = instance?.memory?.ram_weight_bytes != null
                ? instance.memory.ram_weight_bytes > 0 ? 'stream' : 'vram'
                : (catalog.loadContextForModel?.(item.name) ?? readLoadContext(item.name)).jev_weights ?? 'stream';
              const memory = estimateModelWeights(item, placement);
              const streamsExperts = !loaded && item.loadable && (item.estimated_weight_bytes_by_role?.experts ?? 0) > 0
                && loadMemory != null && item.estimated_resident_weight_bytes != null && item.estimated_resident_weight_bytes > loadMemory;
              const unavailableLabel = item.complete && item.format === 'hf'
                ? tr('需先转换', 'Convert first') : item.missing_shards
                  ? tr('分片不全', 'Incomplete shards') : tr('不可用', 'Invalid');
              const streamingNotice = tr('剩余内存不足，超出部分专家会被放入 SSD 进行流式加载推理。',
                'Insufficient remaining memory. Experts that do not fit will be streamed from SSD during inference.');
              return (
                <div className={component ? 'model-row component-asset-row' : 'model-row'} key={item.id}>
                  <span className={component ? 'model-state' : loaded ? 'model-state active'
                    : item.loadable ? 'model-state' : 'model-state failed'} />
                  <div className="checkpoint-row-details">
                    <strong>{item.name}</strong>
                    {component && <small className="component-asset-label">{componentLabel}</small>}
                    <small>{gib(item.total_bytes)} · {item.missing_shards
                      ? tr(`${item.shard_count} 个分片，缺 ${item.missing_shards} 片`, `${item.shard_count} shards, ${item.missing_shards} missing`)
                      : tr(`${item.shard_count} 个分片`, `${item.shard_count} shards`)}</small>
                    {!component && <small>{item.structured_decisions
                      ? <>{tr('预计显存常驻', 'Estimated resident VRAM')} {gib(memory?.vram)}
                          {placement === 'stream' && <> · {tr('RAM 流式权重', 'RAM-streamed weights')} {gib(memory?.ram)}</>}</>
                      : <>{tr('预计常驻内存', 'Estimated resident memory')} {gib(item.estimated_resident_weight_bytes)}</>}
                      {(item.ssd_ple_bytes ?? 0) > 0 && ` · SSD PLE ${gib(item.ssd_ple_bytes)}`}</small>}
                    {instance && <small className="model-library-runtime">{instance.state === 'busy' ? tr('使用中', 'Busy')
                      : instance.state === 'loading' ? tr('加载中', 'Loading') : instance.state === 'unloading' ? tr('卸载中', 'Unloading') : tr('就绪', 'Ready')}
                      {instance.context_size != null && ` · ${formatNumber(instance.context_size)} ctx`}
                      {instance.pinned ? ` · ${tr('固定', 'Pinned')}` : instance.idle_ttl_seconds != null ? ` · TTL ${instance.idle_ttl_seconds}s` : ''}
                    </small>}
                    {!component && <ModelLoadProgress model={item.name} />}
                    {capabilities.companions !== false && !loaded && item.gguf_companion_required && <details className="model-companion-picker">
                      <summary>{tr('Engram 配套文件', 'Engram companion')}</summary>
                      {companionMode !== 'path' &&
                        <select aria-label={tr(`Engram 配套资产：${item.name}`, `Engram companion: ${item.name}`)} disabled={busy} value={isCompanionId ? companion : ''} onChange={event => setCompanions(previous => ({ ...previous, [item.id]: event.target.value }))}>
                          <option value="">{tr('选择已登记 GGUF 配套资产', 'Select a registered GGUF companion')}</option>
                          {artifacts.filter(candidate => candidate.id !== item.id && candidate.format === 'gguf' && candidate.complete === true && /^[0-9a-f]{32}$/.test(candidate.id)).map(candidate => <option key={candidate.id} value={candidate.id}>{candidate.name}</option>)}
                        </select>}
                      {companionMode !== 'registered-id' && <input aria-label={tr(`Engram 路径：${item.name}`, `Engram path: ${item.name}`)}
                        disabled={busy} value={companionMode === 'both' && isCompanionId ? '' : companions[item.id] ?? ''}
                        placeholder={tr('服务端的 GGUF/TQ/MFQ 表文件或 Engram 文件夹', 'Server-side GGUF/TQ/MFQ table file or Engram folder')}
                        onChange={event => setCompanions(previous => ({ ...previous, [item.id]: event.target.value }))} />}
                    </details>}
                  </div>
                  {!component && <div className="local-checkpoint-resources">
                    <div className="local-checkpoint-notice">
                      {!loaded && !item.loadable && <em className="failed" id={`checkpoint-unavailable-${item.id}`} title={item.error || unavailableLabel}>{unavailableLabel}</em>}
                      {streamsExperts && <small className="model-streaming-notice" id={`expert-streaming-${item.id}`} title={streamingNotice}>{streamingNotice}</small>}
                    </div>
                    <div className="local-checkpoint-memory">
                    <ModelPressureBars weights={memory?.total} roles={item.estimated_weight_bytes_by_role}
                      combineMemory={item.structured_decisions && placement === 'stream'}
                      system={system} available={available} combineMoe showDetails={false}
                      label={tr('预计占剩余可用内存', 'Estimated share of remaining available memory')}
                      tr={tr} />
                    </div>
                  </div>}
                  <div className="model-row-controls"><div className="model-row-actions"><ModelVendorMark name={item.name} architecture={item.architecture} />{capabilities.files !== false && <button className="model-files-action" disabled={busy} onClick={() => void openModelFiles(item.id)} type="button"><Icon name="folder" size={13} />{tr('模型文件', 'Model files')}</button>}
                  {!component && <>
                  {capabilities.settings !== false && <button className="model-settings-action" aria-label={tr(`配置模型：${item.name}`, `Configure model: ${item.name}`)}
                    title={tr('模型设置', 'Model settings')} disabled={busy || !item.loadable && !loaded}
                    onClick={() => catalog.openModelSettings(item.name)} type="button"><Icon name="settings" size={14} /></button>}{instance ? (
                    <button className="model-load-action" disabled={busy || instance.state !== 'ready'}
                      onClick={() => void unloadInstance(instance.id)} type="button">
                      {tr('卸载', 'Unload')}
                    </button>
                  ) : (
                    <button className="model-load-action model-start-action" disabled={busy || !item.loadable || (item.gguf_companion_required && !validCompanion)} title={!item.loadable ? item.error || unavailableLabel : undefined}
                      aria-describedby={!item.loadable ? `checkpoint-unavailable-${item.id}` : streamsExperts ? `expert-streaming-${item.id}` : undefined}
                      onClick={() => void (companions[item.id]?.trim()
                        ? loadArtifact(item.name, [companions[item.id].trim()]) : loadArtifact(item.name))} type="button">
                      {tr('加载', 'Load')}
                    </button>
                  )}{capabilities.deletion !== false && <button className="model-delete-action" aria-label={tr(`删除模型：${item.name}`, `Delete model: ${item.name}`)}
                    disabled={busy || loaded} title={loaded ? tr('请先卸载模型再删除', 'Unload the model before deleting it') : tr('删除模型', 'Delete model')}
                    onClick={() => void prepareModelDeletion(item.id)} type="button"><Icon name="trash" size={14} /></button>}</>}
                    </div>
                  </div>
                </div>
              );
            })}
              </section>)}
            </div>
          </TMPanel>)}
        </div>
      ) : (
        <EmptyPanel
          icon="folder"
          title={tr(modelFilter || filterCount ? '没有匹配的本地模型' : '还没有本地模型',
            modelFilter || filterCount ? 'No local model matches' : 'No local models yet')}
          message={tr('添加一个模型文件夹即可开始。', 'Add a model folder to get started.')}
          action={
            <button className="primary screen-header-action" disabled={busy}
              onClick={() => void chooseModelDirectory()} type="button">
              <Icon name="folder" size={14} />{tr('选择模型文件夹', 'Choose model folder')}
            </button>
          }
        />
      )}
      <ModelDeleteDialog catalog={catalog} />
    </>
  );
}
