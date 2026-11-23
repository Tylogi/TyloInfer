import { useState } from 'react';
import { CaretRightIcon } from '@phosphor-icons/react';
import { Icon } from '@shared/ui/display';
import { formatBytes } from '@shared/formatters';
import type { CheckpointAnalysis, ModelArtifact } from '@shared/types/api';
import { ArchitectureDiagram } from '@shared/features/analysis/ArchitectureDiagram';
import { AdvancedDetails } from '@shared/features/analysis/AdvancedDetails';
import { KvCachePlanner } from '@shared/features/models/KvCachePlanner';
import { ExpertHeatmap, FfnBars, LayerBudget, PrecisionLegend, ProjectionBars } from '@shared/features/analysis/PrecisionCharts';
import { analysisCategoryLabel, parameterLabel, type Translate } from '@shared/features/analysis/analysisData';

export function AnalysisPanels({ analysis, selectedArtifact, tr }: { analysis: CheckpointAnalysis; selectedArtifact?: ModelArtifact; tr: Translate }) {
  const [visualization, setVisualization] = useState<'ffn' | 'projections'>('ffn');
  const [category, setCategory] = useState('all');
  const [tensorSearch, setTensorSearch] = useState('');
  const [tensorPage, setTensorPage] = useState(0);
  const categories = [...new Set(analysis?.projections.map(item => item.category) ?? [])];
  const projections = analysis?.projections.filter(item => category === 'all' || item.category === category) ?? [];
  const tensors = analysis?.tensors.filter(item => !tensorSearch || item.name.toLowerCase().includes(tensorSearch.toLowerCase())) ?? [];
  const pages = Math.max(1, Math.ceil(tensors.length / 30));
  const currentPage = Math.min(tensorPage, pages - 1);
  const pleParameters = analysis?.tensors.filter(item => item.category === 'ple').reduce((sum, item) => sum + item.parameters, 0) ?? 0;
  return <div className="analysis-layout">
        <ArchitectureDiagram analysis={analysis} tr={tr} />
        <div className="analysis-main">
          <section className="analysis-panel analysis-summary">
            <div className="analysis-panel-heading"><h2>{tr('模型结构', 'Model structure')}</h2><span>{analysis.format.toUpperCase()} · {selectedArtifact?.shard_count} {tr('分片', 'shards')}</span></div>
            <dl className="analysis-facts">
              <div className="analysis-fact-architecture"><dt>{tr('架构', 'Architecture')}</dt><dd>{analysis.architecture}</dd></div>
              <div><dt>{tr('层数 / 隐藏维度', 'Layers / hidden size')}</dt><dd>{analysis.layer_count} / {analysis.hidden_size?.toLocaleString() ?? '—'}</dd></div>
              <div><dt>{tr('注意力头 Q / KV', 'Attention heads Q / KV')}</dt><dd>{analysis.attention_heads ?? '—'} / {analysis.kv_heads ?? '—'}</dd></div>
              <div><dt>{tr('头维度', 'Head dimension')}</dt><dd>{analysis.head_dim ?? '—'}</dd></div>
              <div><dt>{tr('GDN 头 K / V', 'GDN heads K / V')}</dt><dd>{analysis.linear_key_heads ?? '—'} / {analysis.linear_value_heads ?? '—'}</dd></div>
              <div><dt>{tr('路由专家 / 激活', 'Routed / active experts')}</dt><dd>{analysis.expert_count || '—'} / {analysis.experts_per_token ?? '—'}</dd></div>
              <div><dt>{tr('参数量', 'Parameters')}</dt><dd>{parameterLabel(analysis.parameters)}</dd>{pleParameters > 0 && <small>{tr('含', 'Includes')} {parameterLabel(pleParameters)} PLE</small>}</div>
              <div><dt>{tr('平均 bpw', 'Average bpw')}</dt><dd>{analysis.average_bpw?.toFixed(3) ?? '—'}</dd></div>
              <div><dt>{tr('权重存储', 'Stored weights')}</dt><dd>{(analysis.stored_bytes / 2 ** 30).toFixed(1)} GiB</dd></div>
            </dl>
            <AdvancedDetails analysis={analysis} tr={tr} />
            {Object.keys(analysis.attention_distribution).some(kind => kind !== 'GDN') && <KvCachePlanner key={analysis.model_id} profile={analysis.cache_profile} tr={tr} />}
            <div className="analysis-attention-distribution"><div className="analysis-section-heading"><h3>{tr('注意力分布', 'Attention distribution')}</h3>
              <div>{Object.entries(analysis.attention_distribution).map(([kind, ids]) => <span key={kind}><i className={`attention-kind attention-${kind.toLowerCase()}`} />{kind} <b>{ids.length}</b></span>)}</div></div>
              <div className="attention-layer-strip">{analysis.layers.map(layer => <span key={layer.layer} className={`attention-kind attention-${layer.attention_type.toLowerCase()}`}
                title={`L${layer.layer} · ${layer.attention_type}`} aria-label={`L${layer.layer} · ${layer.attention_type}`} />)}</div>
              <div className="analysis-strip-labels"><span>L0</span><span>L{Math.max(0, analysis.layer_count - 1)}</span></div>
            </div>
            <div className="analysis-section-heading analysis-projection-heading"><h3>{tr('各类投影', 'Projection breakdown')}</h3>
              <select aria-label={tr('投影分类', 'Projection category')} value={category} onChange={event => setCategory(event.target.value)}>
                <option value="all">All components</option>{categories.map(item => <option key={item} value={item}>{analysisCategoryLabel(analysis, item)}</option>)}</select></div>
            <div className="analysis-table-scroll analysis-projection-table"><table><thead><tr>
              <th>{tr('组件 / 投影', 'Component / projection')}</th><th>{tr('参数量', 'Parameters')}</th><th>bpw</th><th>{tr('格式', 'Formats')}</th></tr></thead>
              <tbody>{projections.map((item, index) => <tr key={`${item.category}:${item.name}:${index}`}><td><span>{analysisCategoryLabel(analysis, item.category, item.name)}</span><strong>{item.name}</strong></td>
                <td title={item.parameters.toLocaleString()}>{parameterLabel(item.parameters)}</td><td>{item.average_bpw?.toFixed(3) ?? '—'}</td><td>{item.formats.join(' · ')}</td></tr>)}</tbody></table></div>
          </section>
          <section className="analysis-panel analysis-visualizations">
            <div className="analysis-panel-heading"><h2>{tr('量化可视化', 'Quantization visualization')}</h2><Icon name="chart" size={16} /></div>
            <div className="analysis-tabs" role="tablist" aria-label={tr('量化图表', 'Quantization charts')}>
              <button role="tab" id="analysis-ffn-tab" aria-controls="analysis-chart-content" aria-selected={visualization === 'ffn'} onClick={() => setVisualization('ffn')} type="button">FFN</button>
              <button role="tab" id="analysis-projection-tab" aria-controls="analysis-chart-content" aria-selected={visualization === 'projections'} onClick={() => setVisualization('projections')} type="button">{tr('注意力与其它投影', 'Attention & other projections')}</button></div>
            <div id="analysis-chart-content" role="tabpanel" aria-labelledby={visualization === 'ffn' ? 'analysis-ffn-tab' : 'analysis-projection-tab'}>
              {visualization === 'ffn' ? <>
                {analysis.expert_count > 0 || analysis.experts.length > 0 ? <>
                  <PrecisionLegend tr={tr} /><ExpertHeatmap analysis={analysis} tr={tr} />
                  <div className="analysis-projection-maps">{['gate', 'up', 'down'].map(projection =>
                    <ExpertHeatmap key={projection} analysis={analysis} projection={projection} tr={tr} />)}</div>
                  {analysis.tensors.some(item => item.category === 'shared_expert') && <details key={analysis.model_id} className="analysis-shared-ffn">
                    <summary>{tr('共享专家', 'Shared experts')}<CaretRightIcon size={13} /></summary>
                    <FfnBars analysis={analysis} shared tr={tr} />
                  </details>}
                </> : <FfnBars analysis={analysis} tr={tr} />}
                <LayerBudget analysis={analysis} tr={tr} />
              </> : <><ProjectionBars key={analysis.model_id} analysis={analysis} tr={tr} /><LayerBudget analysis={analysis} tr={tr} /></>}
            </div>
            <p className="analysis-method">{tr('bpw = 实际存储位数 / 参数量，含量化元数据、码本与格式开销；平均值按参数量加权。不含资源文件、整数缓冲、KV Cache 或运行时展开。', 'bpw = stored bits / parameters, including quantization metadata, codebooks and format overhead; averages are parameter-weighted. Excludes assets, integer buffers, KV cache and runtime expansion.')}</p>
          </section>
          <details className="analysis-panel analysis-tensor-details"><summary>{tr('张量明细', 'Tensor details')}<span>{analysis.tensors.length}</span></summary>
            <div className="analysis-tensor-toolbar"><label><Icon name="search" size={14} /><input aria-label={tr('搜索张量', 'Search tensors')} placeholder={tr('按张量名称搜索', 'Search tensor names')}
              value={tensorSearch} onChange={event => { setTensorSearch(event.target.value); setTensorPage(0); }} /></label><span>{tensors.length} {tr('项', 'items')}</span></div>
            <div className="analysis-table-scroll"><table><thead><tr><th>{tr('张量 / 形状', 'Tensor / shape')}</th><th>{tr('参数量', 'Parameters')}</th><th>{tr('存储', 'Storage')}</th><th>bpw</th><th>{tr('格式', 'Format')}</th></tr></thead>
              <tbody>{tensors.slice(currentPage * 30, (currentPage + 1) * 30).map(item => <tr key={item.name}><td><strong>{item.name}</strong><span>{item.shape.join(' × ') || '—'}</span></td><td>{parameterLabel(item.parameters)}</td>
                <td>{formatBytes(item.stored_bytes)}</td><td>{item.bpw?.toFixed(3) ?? '—'}</td><td>{item.format}</td></tr>)}</tbody></table></div>
            <div className="analysis-pagination"><button disabled={currentPage === 0} onClick={() => setTensorPage(value => value - 1)} type="button">{tr('上一页', 'Previous')}</button>
              <span>{currentPage + 1} / {pages}</span><button disabled={currentPage + 1 >= pages} onClick={() => setTensorPage(value => value + 1)} type="button">{tr('下一页', 'Next')}</button></div>
          </details>
          {analysis.warnings.length > 0 && <details className="analysis-panel analysis-warnings"><summary>{tr('元数据提示', 'Metadata notices')}<span>{analysis.warnings.length}</span></summary>
            <ul>{analysis.warnings.map((warning, index) => <li key={index}>{warning}</li>)}</ul></details>}
        </div>
      </div>;
}
