import { PauseIcon, PlayIcon } from '@phosphor-icons/react';
export function LogUpdateControl({ paused, disabled = false, onChange, tr }: {
  paused: boolean; disabled?: boolean; onChange(paused: boolean): void; tr(zh: string, en: string): string;
}) {
  return <button type="button" aria-pressed={paused} disabled={disabled} onClick={() => onChange(!paused)}>
    {paused ? <PlayIcon size={13} /> : <PauseIcon size={13} />}{paused ? tr('继续更新', 'Resume updates') : tr('暂停更新', 'Pause updates')}
  </button>;
}
