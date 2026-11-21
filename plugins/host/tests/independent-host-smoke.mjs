import { createRequire } from 'node:module';
import { readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { createHash, randomUUID } from 'node:crypto';
const require = createRequire(new URL('../../../studio/web/package.json', import.meta.url));
const { request } = require('@playwright/test');
const [state, phase] = process.argv.slice(2);
const canonical = value => Array.isArray(value) ? value.map(canonical) : value && typeof value === 'object'
  ? Object.fromEntries(Object.keys(value).sort().map(key => [key, canonical(value[key])])) : value;
const digest = value => createHash('sha256').update(JSON.stringify(canonical(value))).digest('hex');
const client = await request.newContext({ timeout: 30000 });
let stage = 'private-entry';
try {
  const connection = JSON.parse(await readFile(join(state, 'connection.json'), 'utf8'));
  const origin = new URL(connection.origin);
  if (origin.hostname !== '127.0.0.1' || !(await client.get(connection.url)).ok()) throw new Error();
  const rpc = async (method, args = {}) => {
    const response = await client.post(connection.origin + '/api/' + method, { data: { type: 'client-request', rpcId: randomUUID(), method, payload: { args } } });
    if (response.status() !== 200) throw new Error();
    const result = (await response.json()).result;
    if (!result?.ok) {
      const code = result?.error?.code;
      console.log('INDEPENDENT_NATIVE_RPC_REJECTED; METHOD=' + method + '; CODE=' + (typeof code === 'string' && /^[a-z0-9_/-]{1,120}$/i.test(code) ? code : 'unknown'));
      throw new Error();
    }
    return result.value;
  };
  stage = 'actual-version-and-independent-core';
  const status = await (await client.get(connection.origin + '/tyloinfer/status')).json();
  if (status.core.connected || status.dsh_version !== '0.2.2-rc.2' || status.version !== '0.1.0') throw new Error();
  if (phase === 'before') {
    stage = 'external-model-catalog';
    const catalog = await rpc('session/modelCatalog');
    if (!JSON.stringify(catalog).includes('fixture-external-model')) throw new Error();
    stage = 'native-session-create';
    const created = await rpc('session/create', { request: {} });
    stage = 'native-model-selection';
    await rpc('session/selectModel', { request: { sessionId: created.sessionId, provider: 'fixture-external', model: 'fixture-external-model' } });
    stage = 'native-prompt';
    const prompted = await rpc('session/prompt', { request: { sessionId: created.sessionId, requestId: randomUUID(), mode: 'queue',
      content: [{ type: 'text', text: 'Return the fixture result without using tools.' }] } });
    if (prompted.accepted !== true) throw new Error();
    stage = 'external-response';
    const finished = await rpc('session/wait', { request: { sessionId: created.sessionId, timeoutMs: 20000 } });
    if (finished.reason !== 'idle' || !JSON.stringify(finished.output).includes('fixture-external-response')) {
      const history = await rpc('session/events', { request: { address: { kind: 'session', sessionId: created.sessionId }, maxMessages: 30 } });
      console.log(JSON.stringify({ diagnostic: 'external-response', reason: finished.reason, output_records: finished.output?.records?.length,
        active: finished.state?.active, running: finished.state?.running, cursor: finished.state?.cursor, queue_length: finished.state?.queue?.length,
        event_types: history.records?.map(row => row.event?.type),
        attempts: history.records?.filter(row => row.event?.type === 'assistant/attempt').flatMap(row => row.event.data.stream.map(item => ({
          type: item.type === 'chunk' ? item.chunk?.type : item.type,
          finish_kind: item.chunk?.reason?.kind,
          error_code: item.chunk?.reason?.failure?.code,
          error_message: typeof item.chunk?.reason?.failure?.message === 'string' ? item.chunk.reason.failure.message.replace(/https?:\/\/\S+|[A-Za-z]:[\\/][^\s'\"]+/g, '[endpoint-or-path]').slice(0, 240) : undefined,
        }))),
        event_data_keys: history.records?.map(row => ({ type: row.event?.type, keys: Object.keys(row.event?.data ?? {}).sort() })),
        has_fixture_text: JSON.stringify(finished.output).includes('fixture-external-response') }));
      throw new Error();
    }
    stage = 'restart-checkpoint';
    const settings = await rpc('settings/describe');
    const row = settings.namespaces.find(row => row.ns === 'llm-pi-ai');
    const checkpoint = { sessionId: created.sessionId, providers: digest(row.value.providers) };
    await writeFile(join(state, 'independent-checkpoint.json'), JSON.stringify(checkpoint), { flag: 'wx', mode: 0o600 });
  } else {
    stage = 'cold-restart-persistence';
    const checkpoint = JSON.parse(await readFile(join(state, 'independent-checkpoint.json'), 'utf8'));
    const sessions = await rpc('session/list', { _request: {} });
    if (!sessions.items.some(row => row.sessionId === checkpoint.sessionId)) throw new Error();
    const settings = await rpc('settings/describe');
    const row = settings.namespaces.find(row => row.ns === 'llm-pi-ai');
    if (digest(row.value.providers) !== checkpoint.providers) throw new Error();
    stage = 'cold-session-output';
    const finished = await rpc('session/wait', { request: { sessionId: checkpoint.sessionId, timeoutMs: 2000 } });
    if (!JSON.stringify(finished.output).includes('fixture-external-response')) throw new Error();
    stage = 'cold-external-request';
    const prompted = await rpc('session/prompt', { request: { sessionId: checkpoint.sessionId, requestId: randomUUID(), mode: 'queue',
      content: [{ type: 'text', text: 'Return the fixture result again without using tools.' }] } });
    if (prompted.accepted !== true) throw new Error();
    const next = await rpc('session/wait', { request: { sessionId: checkpoint.sessionId, afterSeq: finished.state.cursor, timeoutMs: 20000 } });
    if (next.reason !== 'idle' || !JSON.stringify(next.output).includes('fixture-external-response')) throw new Error();
  }
  console.log('INDEPENDENT_HOST_PHASE_PASS; PHASE=' + phase + '; AUTHENTICATED=true; CORE_REQUIRED=false');
} catch {
  console.log('INDEPENDENT_HOST_PHASE_FAILED; PHASE=' + phase + '; STAGE=' + stage);
  process.exitCode = 1;
} finally { await client.dispose(); }
