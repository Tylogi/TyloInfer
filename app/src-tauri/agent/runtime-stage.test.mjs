import assert from 'node:assert/strict';
import fs from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve, sep } from 'node:path';
import test from 'node:test';
import { publishRuntime } from './runtime-stage.mjs';

const rename = fs.renameSync;
function fixture(t, previous = true) {
  const root = fs.mkdtempSync(join(tmpdir(), 'tylo-agent-stage-test-'));
  const stage = join(root, '.stage-test');
  const runtime = join(root, 'runtime');
  fs.mkdirSync(stage);
  fs.writeFileSync(join(stage, 'payload'), 'new');
  if (previous) {
    fs.mkdirSync(runtime);
    fs.writeFileSync(join(runtime, 'payload'), 'old');
  }
  t.after(() => {
    assert.ok(resolve(root).startsWith(resolve(tmpdir()) + sep));
    fs.rmSync(root, { recursive: true, force: true, maxRetries: 10, retryDelay: 100 });
  });
  return { root, stage, runtime };
}
const options = { maxRetries: 2, retryDelay: 1 };
const failure = code => Object.assign(new Error('simulated ' + code), { code });
const payload = directory => fs.readFileSync(join(directory, 'payload'), 'utf8');

test('publishes the first runtime and removes the consumed stage', async t => {
  const { root, stage, runtime } = fixture(t, false);
  await publishRuntime(stage, runtime, options);
  assert.equal(payload(runtime), 'new');
  assert.deepEqual(fs.readdirSync(root), ['runtime']);
});

test('replaces an existing runtime and cleans its backup after success', async t => {
  const { root, stage, runtime } = fixture(t);
  await publishRuntime(stage, runtime, options);
  assert.equal(payload(runtime), 'new');
  assert.deepEqual(fs.readdirSync(root), ['runtime']);
});

for (const code of ['EPERM', 'EACCES', 'EBUSY']) {
  test('recovers from transient ' + code + ' while promoting', async t => {
    const { stage, runtime } = fixture(t);
    let attempts = 0;
    t.mock.method(fs, 'renameSync', (source, destination) => {
      if (source === stage && attempts++ < 2) throw failure(code);
      return rename(source, destination);
    });
    await publishRuntime(stage, runtime, options);
    assert.equal(attempts, 3);
    assert.equal(payload(runtime), 'new');
  });
}

test('exhausted retries restore the old runtime and retain the validated stage', async t => {
  const { root, stage, runtime } = fixture(t);
  let attempts = 0;
  t.mock.method(fs, 'renameSync', (source, destination) => {
    if (source === stage) { attempts++; throw failure('EPERM'); }
    return rename(source, destination);
  });
  await assert.rejects(publishRuntime(stage, runtime, options),
    error => error.cause.code === 'EPERM' && error.message.includes(stage));
  assert.equal(attempts, 3);
  assert.equal(payload(runtime), 'old');
  assert.equal(payload(stage), 'new');
  assert.deepEqual(fs.readdirSync(root).sort(), ['.stage-test', 'runtime']);
});

test('a locked old runtime remains intact when it cannot be backed up', async t => {
  const { stage, runtime } = fixture(t);
  t.mock.method(fs, 'renameSync', (source, destination) => {
    if (source === runtime) throw failure('EACCES');
    return rename(source, destination);
  });
  await assert.rejects(publishRuntime(stage, runtime, options), error => error.cause.code === 'EACCES');
  assert.equal(payload(runtime), 'old');
  assert.equal(payload(stage), 'new');
});

test('non-transient errors fail immediately and restore the previous runtime', async t => {
  const { stage, runtime } = fixture(t);
  let attempts = 0;
  t.mock.method(fs, 'renameSync', (source, destination) => {
    if (source === stage) { attempts++; throw failure('EXDEV'); }
    return rename(source, destination);
  });
  await assert.rejects(publishRuntime(stage, runtime, options), error => error.cause.code === 'EXDEV');
  assert.equal(attempts, 1);
  assert.equal(payload(runtime), 'old');
  assert.equal(payload(stage), 'new');
});

test('failed rollback retains both payloads and reports their recovery paths', async t => {
  const { root, stage, runtime } = fixture(t);
  t.mock.method(fs, 'renameSync', (source, destination) => {
    if (source === stage || destination === runtime) throw failure('EPERM');
    return rename(source, destination);
  });
  await assert.rejects(publishRuntime(stage, runtime, options), error => {
    assert.ok(error instanceof AggregateError);
    assert.equal(error.errors.length, 2);
    assert.ok(error.message.includes(stage));
    assert.ok(error.message.includes('.runtime-backup-'));
    return true;
  });
  const backup = fs.readdirSync(root).find(name => name.startsWith('.runtime-backup-'));
  assert.equal(payload(join(root, backup, 'runtime')), 'old');
  assert.equal(payload(stage), 'new');
  assert.equal(fs.existsSync(runtime), false);
});
