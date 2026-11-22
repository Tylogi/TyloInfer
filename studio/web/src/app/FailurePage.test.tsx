
import { fireEvent, render, screen } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import { AppErrorBoundary, FailureView } from './FailurePage';

const BrokenChild = () => {
  throw new Error('render failed');
};

it('显示完整故障信息并执行恢复操作', () => {
  const onRetry = vi.fn();
  const onLeave = vi.fn();
  render(
    <FailureView
      code="APP / 03"
      description="应用暂时无法显示"
      detail="render failed"
      detailLabel="查看错误详情"
      kind="render"
      leaveLabel="返回概览"
      onLeave={onLeave}
      onRetry={onRetry}
      retryLabel="重新载入"
      title="界面遇到错误"
    />,
  );

  expect(screen.getByRole('heading', { name: '界面遇到错误' })).toBeTruthy();
  fireEvent.click(screen.getByText('查看错误详情'));
  expect(screen.getByText('render failed')).toBeTruthy();
  fireEvent.click(screen.getByRole('button', { name: '重新载入' }));
  fireEvent.click(screen.getByRole('button', { name: '返回概览' }));
  expect(onRetry).toHaveBeenCalledOnce();
  expect(onLeave).toHaveBeenCalledOnce();
});

it('shows the fatal recovery view when a child throws during rendering', () => {
  const consoleError = vi.spyOn(console, 'error').mockImplementation(() => undefined);

  try {
    render(
      <div>
        <AppErrorBoundary>
          <span>healthy child</span>
        </AppErrorBoundary>
        <AppErrorBoundary>
          <BrokenChild />
        </AppErrorBoundary>
      </div>,
    );

    expect(screen.getByText('healthy child')).toBeTruthy();
    expect(screen.getByRole('alert')).toBeTruthy();
    expect(screen.getByText('render failed')).toBeTruthy();
  } finally {
    consoleError.mockRestore();
  }
});
