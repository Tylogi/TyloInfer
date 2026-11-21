import { render, within } from '@testing-library/react';
import { expect, it } from 'vitest';
import { ModelLibrarySummary } from './ModelLibrarySummary';
import type { ModelArtifact, RuntimeInstance } from '../../shared/api/types';
it('counts model identities rather than companion assets or duplicate ready instances', () => {
  const artifacts = [{ id: 'main', name: 'Main', asset_kind: 'model', total_bytes: 10 * 2 ** 20, owned_file_bytes: 2 ** 20 }, { id: 'table', name: 'Table', asset_kind: 'engram', total_bytes: 2 * 2 ** 20 }] as ModelArtifact[];
  const instances = [{ id: 'one', model: 'Main', state: 'ready' }, { id: 'two', model: 'Main', state: 'busy' }, { id: 'failed', model: 'Other', state: 'failed' }] as RuntimeInstance[];
  const { container } = render(<ModelLibrarySummary artifacts={artifacts} instances={instances} folderPath="Selected folder" tr={(_zh, en) => en} />);
  const summary = container.querySelector<HTMLElement>('.model-library-summary')!;
  expect(within(summary).getByText('Local models').querySelector('strong')).toHaveTextContent(/^1$/);
  expect(within(summary).getByText('Loaded').querySelector('strong')).toHaveTextContent(/^1$/);
  expect(within(summary).getByText('Total file size').querySelector('strong')).toHaveTextContent('3 MiB');
  expect(container.querySelector('.model-folder-location code')).toHaveTextContent('Selected folder');
});
