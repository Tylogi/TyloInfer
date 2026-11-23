import { expect, it } from 'vitest';
import { parseHubReference } from './hubReference';

it.each([
  ['https://huggingface.co/team/model/tree/main', 'huggingface', 'team/model', 'main'],
  ['https://modelscope.cn/models/team/model/tree/release/v1', 'modelscope', 'team/model', 'release/v1'],
  [' team/model ', 'modelscope', 'team/model', undefined],
] as const)('解析 %s 的提供方、仓库和版本', (input, provider, repoId, revision) => {
  expect(parseHubReference(input, 'modelscope')).toEqual({ provider, repoId, ...(revision ? { revision } : {}) });
});

it.each(['', 'https://example.org/team/model', 'https://huggingface.co/team', 'https://huggingface.co/%ZZ/model'])('拒绝无效仓库 %s', (input) => {
  expect(parseHubReference(input, 'modelscope')).toBeNull();
});
