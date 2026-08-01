"use client";

import { motion } from "motion/react";
import type { Stage } from "@/lib/schematics";

// 3b1b-style simplified anatomy: stages left→right, edges carrying a looping
// forward-pass pulse (CSS dash flow, staged delays), residual/recurrence arcs.
const W = 440;
const H = 150;
const PAD = 34;

export function NetworkFlow({ stages }: { stages: Stage[] }) {
  const cols = stages.length;
  const colX = (i: number) => PAD + (i * (W - 2 * PAD)) / Math.max(1, cols - 1);
  const nodeYs = (n: number) =>
    Array.from({ length: n }, (_, j) => H / 2 + (j - (n - 1) / 2) * Math.min(26, 90 / n + 12));

  const shape = (k: Stage["k"], x: number, y: number, key: string, delay: number) => {
    const common = {
      className: "flow-node",
      style: { animationDelay: `${delay}s` } as React.CSSProperties,
    };
    switch (k) {
      case "in":
      case "noise":
        return <rect key={key} x={x - 5} y={y - 5} width={10} height={10}
          fill={k === "noise" ? "var(--ink-dim)" : "var(--paper-deep)"}
          stroke="var(--line)" strokeWidth={1.5} {...common} />;
      case "latent":
        return <rect key={key} x={x - 5} y={y - 5} width={10} height={10}
          transform={`rotate(45 ${x} ${y})`} fill="var(--paper-deep)"
          stroke="var(--redline)" strokeWidth={1.5} {...common} />;
      case "router":
        return <circle key={key} cx={x} cy={y} r={5} fill="var(--paper-deep)"
          stroke="var(--redline)" strokeWidth={1.5} {...common} />;
      case "out":
        return <rect key={key} x={x - 6} y={y - 4} width={12} height={8}
          fill="var(--line)" opacity={0.9} {...common} />;
      default:
        return <circle key={key} cx={x} cy={y} r={5} fill="var(--paper-deep)"
          stroke="var(--line)" strokeWidth={1.5} {...common} />;
    }
  };

  return (
    <motion.svg
      viewBox={`0 0 ${W} ${H + 26}`}
      className="w-full"
      initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.5 }}
    >
      {/* edges */}
      {stages.slice(0, -1).map((s, i) => {
        const next = stages[i + 1];
        const ys1 = nodeYs(s.n ?? 3);
        const ys2 = nodeYs(next.n ?? 3);
        const dense = next.k === "attn" || s.k === "attn";
        return ys1.map((y1, a) =>
          ys2
            .filter((_, b) => dense || Math.abs(a - b) <= 1)
            .map((y2, b) => (
              <line
                key={`${i}-${a}-${b}`}
                x1={colX(i)} y1={y1} x2={colX(i + 1)} y2={y2}
                className="flow-edge"
                style={{ animationDelay: `${i * 0.18}s` }}
                stroke={s.k === "router" ? "var(--redline)" : "var(--line)"}
                strokeWidth={0.8}
                opacity={dense ? 0.35 : 0.6}
              />
            ))
        );
      })}

      {/* residual skip + recurrence arcs */}
      {stages.map((s, i) => {
        const arcs: React.ReactNode[] = [];
        if (s.skip && i + 2 < cols) {
          arcs.push(
            <path key={`skip-${i}`}
              d={`M ${colX(i)} ${H / 2 - 52} C ${colX(i + 1)} ${H / 2 - 78}, ${colX(i + 1)} ${H / 2 - 78}, ${colX(i + 2)} ${H / 2 - 52}`}
              fill="none" stroke="var(--redline)" strokeWidth={1.2}
              className="flow-edge" style={{ animationDelay: `${i * 0.18}s` }} />
          );
        }
        if (s.loop) {
          arcs.push(
            <path key={`loop-${i}`}
              d={`M ${colX(i) + 10} ${H / 2 + 52} C ${colX(i) + 34} ${H / 2 + 74}, ${colX(i) - 34} ${H / 2 + 74}, ${colX(i) - 10} ${H / 2 + 52}`}
              fill="none" stroke="var(--redline)" strokeWidth={1.2}
              className="flow-edge" style={{ animationDelay: `${i * 0.18}s` }} />
          );
        }
        return arcs;
      })}

      {/* nodes + labels */}
      {stages.map((s, i) =>
        nodeYs(s.n ?? 3).map((y, j) => shape(s.k, colX(i), y, `n-${i}-${j}`, i * 0.18 + j * 0.06))
      )}
      {stages.map((s, i) => (
        <text key={`l-${i}`} x={colX(i)} y={H + 18} textAnchor="middle"
          fontSize={8.5} fill="var(--ink-dim)" fontFamily="var(--font-draft)">
          {s.l ?? ""}
        </text>
      ))}
    </motion.svg>
  );
}
