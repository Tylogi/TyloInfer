
import '@testing-library/jest-dom/vitest';
import { cleanup } from '@testing-library/react';
import { afterEach } from 'vitest';
import { useStudioSharedPresentation } from './shared-presentation';

// Tests that mock SettingsProvider get English labels by default; a test that
// checks Chinese copy installs its own presentation next to its mock.
useStudioSharedPresentation((_chinese, english) => english);

afterEach(() => {
  cleanup();
  localStorage.clear();
  sessionStorage.clear();
});
