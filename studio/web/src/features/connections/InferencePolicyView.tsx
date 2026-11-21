import { SectionLabel, SettingRow, TMPanel } from '../../app/display';
import { Switch } from '../../shared/ui/Switch';
export function InferencePolicyView({ enabled, busy, ready, changeMtp, visionEnabled, changeVision, tr }: {
  enabled?: boolean; busy: boolean; ready: boolean; changeMtp(next: boolean): void;
  visionEnabled?: boolean; changeVision?(next: boolean): void; tr(zh: string, en: string): string;
}) {
  return <>
    <SectionLabel title={tr('推理控制', 'Inference controls')} /><TMPanel>
      <SettingRow title="MTP" detail={tr('关闭后，所有模型与 API 请求均使用普通解码。切换无需重载模型。', 'When off, all models and API requests use ordinary decoding. No model reload is needed.')}
        trailing={<Switch label={tr('服务 MTP', 'Service MTP')} checked={enabled === true} disabled={!ready || busy || typeof enabled !== 'boolean'} onCheckedChange={changeMtp} />} />
      <SettingRow title={tr('视觉输入', 'Vision input')} detail={changeVision ? tr('默认启用；关闭后图片和视频请求会被明确拒绝。', 'On by default; when off, image and video requests are rejected explicitly.') : tr('请求级视觉选项由 Agent 模型与对话设置管理，不修改全局推理策略。', 'Request-level vision options are managed by Agent model and conversation settings, not global inference policy.')}
        trailing={changeVision ? <Switch label={tr('视觉输入', 'Vision input')} checked={visionEnabled === true} onCheckedChange={changeVision} /> : <span className="server-managed-value">{tr('Agent 设置', 'Agent settings')}</span>} />
    </TMPanel>
  </>;
}
