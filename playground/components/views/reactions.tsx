"use client";

import { motion } from "motion/react";
import type { RunSummary, TaxonomyResponse } from "@/lib/types";
import { symbolFor } from "@/lib/blueprint";

export function Reactions({
  taxonomy, runs, onSelect,
}: {
  taxonomy: TaxonomyResponse | null;
  runs: RunSummary[];
  onSelect: (model: string) => void;
}) {
  if (!taxonomy) return null;
  const runCount = (name: string) => runs.filter((r) => r.model === name).length;

  return (
    <div className="h-full overflow-y-auto px-5 py-6 sm:px-10">
      <header className="mb-6 border-b-[1.5px] border-line pb-3">
        <div className="text-[10px] tracking-[0.35em] text-ink-dim">DWG Nº MN-0022 · SHEET 02</div>
        <h1 className="bp-title mt-1 text-3xl font-extrabold text-ink sm:text-4xl">
          Reactions <span className="text-ink-dim">— whole models, wired into pipelines</span>
        </h1>
      </header>

      <div className="grid gap-x-10 gap-y-3 lg:grid-cols-2">
        {taxonomy.compositions.map((c, i) => (
          <motion.div
            key={c.name}
            initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.05 + i * 0.03, duration: 0.3 }}
            className="border-b border-dashed border-line-faint pb-3"
          >
            <div className="flex flex-wrap items-center gap-2 text-[13px]">
              {c.composes.map((m, j) => (
                <span key={m} className="flex items-center gap-2">
                  {j > 0 && <span className="text-ink-dim">+</span>}
                  <button
                    onClick={() => onSelect(m)}
                    className="bp-title border border-line-faint px-1.5 py-0.5 text-sm font-bold text-line transition-colors hover:border-redline hover:text-redline"
                    title={m}
                  >
                    {symbolFor(m)}
                  </button>
                </span>
              ))}
              <span className="text-redline">→</span>
              <span className="bp-title text-sm font-bold text-ink">{c.name}</span>
              {runCount(c.name) > 0 && (
                <span className="text-[9px] text-ink-dim">({runCount(c.name)} runs)</span>
              )}
            </div>
            <p className="mt-1 text-[11px] text-ink-dim">{c.description}</p>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
