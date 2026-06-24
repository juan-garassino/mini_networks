import type { RunStatus } from "./types";

export const SKIP_KEYS = new Set(["epoch"]);
export const PALETTE = ["#ff9f43", "#7d5fff", "#2ecc71", "#4dabf7", "#ff6b6b", "#ffd23f"];

export function fmtNum(v: unknown): string {
  if (typeof v !== "number" || !isFinite(v)) return "—";
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(2);
  if (Number.isInteger(v)) return String(v);
  return v.toFixed(a < 1 ? 4 : a < 100 ? 3 : 2);
}

export function statusMeta(status: RunStatus): { label: string; tone: "green" | "amber" | "blue" | "red" | "gray" } {
  switch (status) {
    case "running": return { label: "live", tone: "green" };
    case "dispatched": return { label: "queued", tone: "amber" };
    case "pending": return { label: "pending", tone: "amber" };
    case "done": return { label: "done", tone: "blue" };
    case "failed": return { label: "failed", tone: "red" };
    default: return { label: "—", tone: "gray" };
  }
}

export function primaryMetric(last: Record<string, number> | undefined): { k: string; v: number } | null {
  if (!last) return null;
  const keys = Object.keys(last).filter((k) => !SKIP_KEYS.has(k));
  if (!keys.length) return null;
  const k = keys.includes("loss") ? "loss" : keys[0];
  return { k, v: last[k] };
}

export function relTime(iso?: string | null): string {
  if (!iso) return "";
  const t = Date.parse(iso);
  if (isNaN(t)) return "";
  const s = (Date.now() - t) / 1000;
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

export function hue(str: string): number {
  let h = 7;
  for (const c of str) h = (h * 31 + c.charCodeAt(0)) % 360;
  return h;
}
