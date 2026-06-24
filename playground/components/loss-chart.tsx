"use client";

import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import type { MetricSeries } from "@/lib/types";
import { SKIP_KEYS, PALETTE } from "@/lib/format";

export function plottable(series: MetricSeries[]) {
  return series.filter((s) => !SKIP_KEYS.has(s.key) && s.points.length)
    .map((s, i) => ({ ...s, color: PALETTE[i % PALETTE.length] }));
}

export function LossChart({ series }: { series: MetricSeries[] }) {
  const plot = plottable(series);
  if (!plot.length) return <div className="grid h-full place-content-center text-inksoft/60 text-sm">No metrics yet…</div>;

  const stepMap = new Map<number, Record<string, number>>();
  for (const s of plot) for (const [x, y] of s.points) {
    const row = stepMap.get(x) ?? { step: x };
    row[s.key] = y;
    stepMap.set(x, row);
  }
  const data = [...stepMap.values()].sort((a, b) => a.step - b.step);

  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={data} margin={{ top: 10, right: 16, bottom: 2, left: -14 }}>
        <defs>{plot.map((s, i) => (
          <linearGradient key={s.key} id={`fill-${i}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={s.color} stopOpacity={0.28} />
            <stop offset="100%" stopColor={s.color} stopOpacity={0} />
          </linearGradient>
        ))}</defs>
        <CartesianGrid strokeDasharray="4 5" stroke="#edeaf6" vertical={false} />
        <XAxis dataKey="step" stroke="#a59cc0" tick={{ fontSize: 12 }} tickLine={false} axisLine={false} />
        <YAxis stroke="#a59cc0" tick={{ fontSize: 12 }} tickLine={false} axisLine={false} width={46} />
        <Tooltip contentStyle={{ borderRadius: 14, border: "1px solid #efe7d8", boxShadow: "0 8px 24px rgba(74,60,40,.16)", fontSize: 13, fontWeight: 700 }} />
        {plot.map((s, i) => (
          <Area key={s.key} type="monotone" dataKey={s.key} stroke={s.color} strokeWidth={3}
            fill={`url(#fill-${i})`} strokeDasharray={i === 1 ? "8 6" : undefined}
            dot={{ r: 3, fill: "#fff", stroke: s.color, strokeWidth: 2 }} activeDot={{ r: 5 }} isAnimationActive={false} />
        ))}
      </AreaChart>
    </ResponsiveContainer>
  );
}
