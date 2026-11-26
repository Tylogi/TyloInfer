import fs from 'node:fs';
import { dirname, join } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';

const transientErrors = new Set(['EPERM', 'EACCES', 'EBUSY']);

export function removeDirectory(path) {
  fs.rmSync(path, { recursive: true, force: true,
    maxRetries: process.platform === 'win32' ? 10 : 0, retryDelay: 100 });
}

async function renameWithRetry(source, destination, {
  maxRetries = process.platform === 'win32' ? 40 : 0, retryDelay = 125,
} = {}) {
  for (let attempt = 0; ; attempt++) {
    try {
      fs.renameSync(source, destination);
      return;
    } catch (error) {
      if (!transientErrors.has(error.code) || attempt >= maxRetries) throw error;
      await delay(retryDelay);
    }
  }
}

// Keep a validated stage on failure, and keep the old runtime until promotion succeeds.
export async function publishRuntime(stage, runtime, options) {
  const backupDirectory = fs.mkdtempSync(join(dirname(runtime), '.runtime-backup-'));
  const backup = join(backupDirectory, 'runtime');
  let backedUp = false;
  let published = false;
  try {
    if (fs.existsSync(runtime)) {
      await renameWithRetry(runtime, backup, options);
      backedUp = true;
    }
    await renameWithRetry(stage, runtime, options);
    published = true;
  } catch (error) {
    if (backedUp) {
      try {
        await renameWithRetry(backup, runtime, options);
        backedUp = false;
      } catch (rollbackError) {
        throw new AggregateError([error, rollbackError],
          'Agent runtime promotion and rollback failed. Previous runtime retained at ' +
          backup + '; validated stage retained at ' + stage + '.');
      }
    }
    throw new Error('Could not publish Agent runtime. Validated stage retained at ' + stage + '.', { cause: error });
  } finally {
    if (published || !backedUp) {
      try {
        removeDirectory(backupDirectory);
      } catch (error) {
        // Cleanup must not hide a promotion failure or invalidate the new runtime.
        console.warn('Could not remove Agent runtime backup at ' + backupDirectory + ': ' + error.message);
      }
    }
  }
}
