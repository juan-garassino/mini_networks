"use client";

import { useEffect, useState } from "react";
import {
  Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { getMetrics } from "@/lib/api";
import type { MetricSeries, RunSummary } from "@/lib/types";
import { STATUS_INK, symbolFor } from "@/lib/blueprint";

const PENS = ["var(--line)", "var(--redline)", "#8be9c3", "#e0a8ff", "#ff7d6b"];

export function Observatory({
  runs, onSelectModel,
}: {
  runs: RunSummary[];
  onSelectModel: (m: string) => void;
}) {
  const [runId, setRunId] = useState<string | null>(null);
  const [series, setSeries] = useState<MetricSeries[]>([]);
  const active = runs.find((r) => r.id === runId) ?? runs[0] ?? null;
  const activeId = active?.id;
  const activeStatus = active?.status;

  useEffect(() => {
    if (!activeId) return;
    let alive = true;
    const load = async () => {
      try {
        const m = await getMetrics(activeId);
        if (alive) setSeries(m.series);
      } catch { /* run may have no metrics yet */ }
    };
    load();
    const id = activeStatus === "running" ? setInterval(load, 2000) : undefined;
    return () => { alive = false; if (id) clearInterval(id); };
  }, [activeId, activeStatus]);

  return (
    <div className="grid h-full grid-rows-[auto_1fr] gap-0 overflow-hidden lg:grid-cols-[minmax(20rem,2fr)_3fr] lg:grid-rows-1">
      {/* run ledger */}
      <div className="overflow-y-auto border-b-[1.5px] border-line lg:border-b-0 lg:border-r-[1.5px]">
        <div className="sticky top-0 border-b border-line-faint bg-paper px-5 py-3">
          <div className="text-[10px] tracking-[0.35em] text-ink-dim">DWG Nº MN-RUNS · SHEET 03</div>
          <h1 className="bp-title text-xl font-extrabold text-ink">Observatory ledger</h1>
        </div>
        {runs.length === 0 && (
          <div className="px-5 py-8 text-[11px] text-ink-dim">no runs surveyed yet</div>
        )}
        {runs.map((r) => (
          <div
            key={r.id}
            onClick={() => setRunId(r.id)}
            className={`grid w-full cursor-pointer grid-cols-[2.4rem_1fr_auto] items-center gap-2 border-b border-dashed border-line-faint px-5 py-2 text-left text-[11px] transition-colors ${
              active?.id === r.id ? "bg-paper-deep" : "hover:bg-paper-deep/50"
            }`}
          >
            <button
              className="bp-title text-base font-bold text-line hover:text-redline"
              onClick={(e) => { e.stopPropagation(); onSelectModel(r.model); }}
              title={`open ${r.model} spec sheet`}
            >
              {symbolFor(r.model)}
            </button>
            <span className="min-w-0">
              <span className="block truncate text-ink">{r.model} · {r.run_name ?? r.id}</span>
              <span className="block truncate text-[9px] text-ink-dim">
                {Object.entries(r.last_metrics).slice(0, 2)
                  .map(([k, v]) => `${k}=${v.toFixed(3)}`).join("  ")}
              </span>
            </span>
            <span
              className={`text-[9px] ${r.status === "running" ? "bp-live" : ""}`}
              style={{ color: STATUS_INK[r.status] }}
            >
              {r.status}
            </span>
          </div>
        ))}
      </div>

      {/* plotting table */}
      <div className="flex min-h-0 flex-col px-5 py-4">
        {active ? (
          <>
            <div className="mb-2 flex items-baseline justify-between">
              <h2 className="bp-title text-sm font-bold text-ink">
                {active.model} <span className="text-ink-dim">· {active.run_name ?? active.id}</span>
              </h2>
              <span className="text-[9px] text-ink-dim">step {active.last_step ?? "—"}</span>
            </div>
            <div className="min-h-0 flex-1 border-[1.5px] border-line-faint bg-paper-deep/40 p-2">
              {series.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart margin={{ top: 8, right: 12, bottom: 4, left: -14 }}>
                    <XAxis
                      dataKey="step" type="number" domain={["auto", "auto"]}
                      stroke="var(--line-faint)" tick={{ fill: "var(--ink-dim)", fontSize: 9 }}
                      allowDuplicatedCategory={false}
                    />
                    <YAxis stroke="var(--line-faint)" tick={{ fill: "var(--ink-dim)", fontSize: 9 }} />
                    <Tooltip
                      contentStyle={{
                        background: "var(--paper-deep)", border: "1px solid var(--line)",
                        fontSize: 10, fontFamily: "var(--font-draft)",
                      }}
                      labelStyle={{ color: "var(--ink-dim)" }}
                    />
                    {series.map((s, i) => (
                      <Line
                        key={s.key}
                        data={s.points.map(([step, value]) => ({ step, value }))}
                        dataKey="value" name={s.key} dot={false} isAnimationActive={false}
                        stroke={PENS[i % PENS.length]} strokeWidth={1.5}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div className="flex h-full items-center justify-center text-[11px] text-ink-dim">
                  no measurements plotted yet
                </div>
              )}
            </div>
            {series.length > 0 && (
              <div className="mt-2 flex flex-wrap gap-3 text-[9px] text-ink-dim">
                {series.map((s, i) => (
                  <span key={s.key} className="flex items-center gap-1">
                    <span className="inline-block h-[2px] w-4" style={{ background: PENS[i % PENS.length] }} />
                    {s.key}
                  </span>
                ))}
              </div>
            )}
          </>
        ) : (
          <div className="flex h-full items-center justify-center text-[11px] text-ink-dim">
            select a run from the ledger
          </div>
        )}
      </div>
    </div>
  );
}
