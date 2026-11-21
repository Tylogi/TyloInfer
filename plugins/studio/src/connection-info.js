import { ModelBridgeError } from './model-api.js';

const invalid = () => new ModelBridgeError('tyloinfer/core-invalid-response', 'Invalid core connection metadata');
function address(value) {
  if (typeof value !== 'string' || value.length > 4096) throw invalid();
  let url; try { url = new URL(value); } catch { throw invalid(); }
  if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.search || url.hash) throw invalid();
  return url;
}
export function validateConnectionInfo(value) {
  if (!value || value.protocol_version !== '1.0') throw invalid();
  const base = address(value.base_url), openai = address(value.openai_base_url);
  const expected = new URL(base); expected.pathname = base.pathname.replace(/\/v1\/?$/, '').replace(/\/$/, '') + '/v1';
  if (openai.toString().replace(/\/$/, '') !== expected.toString().replace(/\/$/, '')) throw invalid();
  const port = value.anthropic_port;
  let anthropic = null;
  if (port === null) { if (value.anthropic_base_url !== null) throw invalid(); }
  else {
    if (!Number.isInteger(port) || port < 1 || port > 65535) throw invalid();
    anthropic = address(value.anthropic_base_url);
    const listener = new URL(openai); listener.port = String(port);
    if (anthropic.toString().replace(/\/$/, '') !== listener.toString().replace(/\/$/, '')) throw invalid();
  }
  return { base_url: base.toString(), openai_base_url: openai.toString().replace(/\/$/, ''), anthropic_base_url: anthropic?.toString().replace(/\/$/, '') ?? null, anthropic_port: port, protocol_version: value.protocol_version };
}
