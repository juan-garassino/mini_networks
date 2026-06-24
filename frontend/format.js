// Small formatting helpers — pure, no DOM.

export const SKIP_KEYS = new Set(['epoch']);

export function fmtNum(v) {
  if (typeof v !== 'number' || !isFinite(v)) return '—';
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(2);
  if (Number.isInteger(v)) return String(v);
  return v.toFixed(a < 1 ? 4 : a < 100 ? 3 : 2);
}

// status -> [led-state, short-label]
export function statusMeta(status) {
  const map = {
    running: ['running', 'live'],
    dispatched: ['dispatched', 'queued'],
    pending: ['dispatched', 'pending'],
    done: ['done', 'done'],
    failed: ['failed', 'failed'],
    unknown: ['unknown', '—'],
  };
  return map[status] || map.unknown;
}

export function primaryMetric(last) {
  if (!last) return null;
  const keys = Object.keys(last).filter((k) => !SKIP_KEYS.has(k));
  if (!keys.length) return null;
  const k = keys.includes('loss') ? 'loss' : keys[0];
  return { k, v: last[k] };
}

export function relTime(iso) {
  if (!iso) return '';
  const t = Date.parse(iso);
  if (isNaN(t)) return '';
  const s = (Date.now() - t) / 1000;
  if (s < 60) return 'just now';
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}
