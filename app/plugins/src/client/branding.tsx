import React, { useEffect } from 'react';
import mark from '@shared/assets/tylogi-ai-lab-mark.svg?raw';

export const productName = 'TyloInfer';
export const productVersion = __TYLOINFER_PRODUCT_VERSION__;
export const markUrl = 'data:image/svg+xml,' + encodeURIComponent(mark);

export function BrandMark({ size, className = '' }) {
  return <img className={`tylo-studio-brand-mark ${className}`} src={markUrl} width={size} height={size} alt="" aria-hidden="true" />;
}

export function BrandName() {
  return <span className="tylo-studio-brand-name">{productName}</span>;
}

export function HomeMark({ size, className = '' }) {
  return <span className="tylo-studio-home-mark"><BrandMark size={size} className={className} /></span>;
}

export function ProductTitle({ useSessions, usePanelInfo }) {
  const conversation = usePanelInfo(info => info.activePanelId === null);
  const sessionTitle = useSessions(state => {
    const session = Object.values(state.byId).find(row => (row.retainedBy.mainView ?? 0) > 0);
    return conversation ? session?.title : undefined;
  });
  useEffect(() => {
    const previous = document.title;
    const title = sessionTitle ? `${sessionTitle} — ${productName}` : productName;
    document.title = title;
    return () => { if (document.title === title) document.title = previous; };
  }, [sessionTitle]);
  return null;
}

export function applyProductIcon(document) {
  const icons = [...document.querySelectorAll('link[rel~="icon"]')];
  const previous = icons.map(icon => ({ icon, href: icon.getAttribute('href'), type: icon.getAttribute('type') }));
  let added;
  if (!icons.length) {
    added = document.createElement('link');
    added.rel = 'icon';
    icons.push(added);
    document.head.append(added);
  }
  for (const icon of icons) { icon.href = markUrl; icon.type = 'image/svg+xml'; }
  return () => {
    added?.remove();
    for (const { icon, href, type } of previous) {
      if (icon.href !== markUrl) continue;
      for (const [key, value] of [['href', href], ['type', type]]) {
        if (value === null) icon.removeAttribute(key); else icon.setAttribute(key, value);
      }
    }
  };
}
