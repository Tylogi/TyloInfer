// Toast data shapes shared by the Studio web app and the plugin Client.

export type ToastType = 'error' | 'success' | 'info' | 'warning';

export interface ToastItemData {
  id: string;
  revision: number;
  type: ToastType;
  message: string;
  title?: string;
  duration?: number;
}
