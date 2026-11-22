import React, { useId, useRef } from 'react';
export function PageTabs({ items, value, onChange, label, className = 'configuration-tabs' }) {
  const id = useId();
  const refs = useRef([]);
  return <div className={className} role="tablist" aria-label={label}>{items.map(([key, title], index) => <button ref={node => { refs.current[index] = node; }} key={key} id={`${id}-${key}`} type="button" role="tab" aria-selected={value === key} tabIndex={value === key ? 0 : -1} onClick={() => onChange(key)} onKeyDown={event => {
    const next = event.key === 'ArrowRight' ? (index + 1) % items.length : event.key === 'ArrowLeft' ? (index + items.length - 1) % items.length : event.key === 'Home' ? 0 : event.key === 'End' ? items.length - 1 : null;
    if (next != null) { event.preventDefault(); onChange(items[next][0]); refs.current[next]?.focus(); }
  }}>{title}</button>)}</div>;
}
