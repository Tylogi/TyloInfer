// Font normalization adapted from DeepSeek Harness (MIT).
// See LICENSES/DeepSeek-Harness-MIT.txt.
const genericFamilies = new Set([
  'serif', 'sans-serif', 'monospace', 'cursive', 'fantasy', 'system-ui', 'ui-serif',
  'ui-sans-serif', 'ui-monospace', 'ui-rounded', 'emoji', 'math', 'fangsong',
  '-apple-system', 'blinkmacsystemfont',
]);

export function normalizeFontFamily(value: string): string {
  return value.split(',')
    .map((name) => name.replace(/["'\\<>\u0000-\u001f\u007f]/gu, '').replace(/\s+/gu, ' ').trim())
    .filter((name) => name !== '')
    .map((name) => genericFamilies.has(name.toLowerCase()) ? name : `"${name}"`)
    .join(', ');
}
