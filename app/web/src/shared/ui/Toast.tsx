


import { useToastStore } from '../../stores/toastStore';
import { ToastItem } from '@shared/ui/ToastItem';
import '@shared/ui/primitives.css';

export { ToastItem } from '@shared/ui/ToastItem';
export type { ToastItemProps } from '@shared/ui/ToastItem';




export function ToastContainer() {
  const toasts = useToastStore((state) => state.toasts);
  const dismissToast = useToastStore((state) => state.dismissToast);

  if (toasts.length === 0) return null;

  return (
    <div
      className="studio-toast-container"
      role="region"
      aria-label="通知提示"
      tabIndex={-1}
    >
      {toasts.map((item) => (
        <ToastItem key={item.id} toast={item} onDismiss={dismissToast} />
      ))}
    </div>
  );
}
