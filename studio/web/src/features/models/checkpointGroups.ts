import { modelVendor } from '../../app/ModelVendorMark';
import type { ModelArtifact } from '../../shared/api/types';

export interface CheckpointModelGroup {
  key: string;
  name: string;
  artifacts: ModelArtifact[];
}

export interface CheckpointFamilyGroup {
  key: string;
  name: string | null;
  models: CheckpointModelGroup[];
  count: number;
}

const vendorNames = { qwen: 'Qwen', deepseek: 'DeepSeek', zai: 'GLM', minicpm: 'MiniCPM', gemma: 'Gemma' };
const quantizationSuffix = /(?:^|[-_\s.])(?:EWQ?|MFQ|GGUF|MLX|EXL[23]|GSQ(?:[-_]RCO)?|UD|Gyro|BF16|FP(?:8|16|32)|MXFP[48]|NVFP4|F(?:16|32)|NINT(?:v\d+)?\d*|NVQ\d+J?|TQ(?:K|P)?\d*(?:[._]\d+)?|I?Q\d+(?:\.\d+)?|oQ\d+(?:\.\d+)?e|S\d+[-_](?:XXS|XS|XL|S|M|L)|V\d+[-_](?:XXS|XS|XL|S|M|L)|\d+(?:\.\d+)?(?:bits?|bpw))(?=$|[-_\s.])/i;

export function checkpointModelName(name: string): string {
  const basename = name.split(/[\\/]/).pop()?.trim() || name;
  const clean = basename.replace(/\.(?:gguf|mfq|tq|safetensors)$/i, '');
  const suffix = quantizationSuffix.exec(clean);
  const base = suffix && suffix.index > 0 ? clean.slice(0, suffix.index) : clean;
  return base.replace(/[-_\s]+$/, '');
}

function checkpointKey(name: string): string {
  return checkpointModelName(name).replace(/[_\s]+/g, '-').toLowerCase();
}

export function groupCheckpoints(artifacts: ModelArtifact[], catalog = artifacts): CheckpointFamilyGroup[] {
  const targets = new Map(catalog.filter(item => !item.asset_kind || item.asset_kind === 'model')
    .map(item => [checkpointKey(item.name), item]));
  const families = new Map<string, CheckpointFamilyGroup>();
  const modelGroups = new Map<CheckpointFamilyGroup, Map<string, CheckpointModelGroup>>();
  for (const artifact of artifacts) {
    const target = artifact.asset_kind === 'dflash2' ? artifact.dflash2_target : undefined;
    const base = target ? targets.get(checkpointKey(target)) : undefined;
    const architecture = (target ? base?.architecture : artifact.architecture)?.trim();
    const vendor = modelVendor(target || artifact.name, architecture);
    const declared = architecture && architecture.toLowerCase() !== 'unknown' ? architecture : null;
    const decision = artifact.structured_decisions === true;
    const key = decision ? 'jev' : vendor ?? (declared ? `architecture:${declared.toLowerCase()}` : 'other');
    let family = families.get(key);
    if (!family) {
      family = { key, name: decision ? 'Jev' : vendor ? vendorNames[vendor] : declared, models: [], count: 0 };
      families.set(key, family);
      modelGroups.set(family, new Map());
    }
    const identity = artifact.asset_kind === 'engram' || artifact.asset_kind === 'ple'
      ? artifact.name.replace(/[-_](?:engram|ple|per[-_]?layer[-_]?(?:token[-_]?)?emb(?:eddings?|d)?)(?:[-_].*)?$/i, '') : artifact.name;
    const name = checkpointModelName(target || identity || artifact.name);
    const modelKey = checkpointKey(name);
    let model = modelGroups.get(family)!.get(modelKey);
    if (!model) {
      model = { key: modelKey, name, artifacts: [] };
      family.models.push(model);
      modelGroups.get(family)!.set(modelKey, model);
    }
    model.artifacts.push(artifact);
    family.count++;
  }
  return [...families.values()];
}
