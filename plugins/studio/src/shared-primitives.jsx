import React, { useEffect, useId, useRef } from 'react';

export function Dialog({ open, onOpenChange, title, description, closeLabel, className = '', children, returnFocusRef }) {
  const ref = useRef(null);
  const id = useId();
  useEffect(() => {
    if (!open) return;
    const dialog = ref.current;
    const previous = document.activeElement;
    dialog.showModal();
    return () => { dialog.close(); const target = returnFocusRef?.current || previous; if (target?.isConnected) target.focus({ preventScroll: true }); };
  }, [open]);
  if (!open) return null;
  return <dialog ref={ref} className={`studio-dialog ${className}`} aria-labelledby={id} aria-describedby={description ? `${id}-description` : undefined}
    onCancel={event => { event.preventDefault(); event.stopPropagation(); onOpenChange(false); }}
    onClick={event => { if (event.target === event.currentTarget) { const rect = event.currentTarget.getBoundingClientRect(); if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) onOpenChange(false); } }}>
    <header><div><h2 id={id}>{title}</h2>{description && <p id={`${id}-description`}>{description}</p>}</div><button type="button" aria-label={closeLabel} onClick={() => onOpenChange(false)}>×</button></header>{children}
  </dialog>;
}
export function Switch({ checked, onCheckedChange, label, disabled, id }) {
  return <button id={id} className="studio-switch" type="button" role="switch" aria-label={label} aria-checked={checked} data-state={checked ? 'checked' : 'unchecked'} disabled={disabled} onClick={() => onCheckedChange(!checked)}><span className="studio-switch-thumb" data-state={checked ? 'checked' : 'unchecked'} /></button>;
}
export function Tooltip({ content, children }) {
  const id = useId();
  return <span className="tylo-shared-tooltip">{React.cloneElement(children, { 'aria-describedby': id })}<span role="tooltip" id={id}>{content}</span></span>;
}
