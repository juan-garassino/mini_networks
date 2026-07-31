"use client";

import { motion } from "motion/react";
import type { TaxonomyResponse } from "@/lib/types";
import { ATOMS, FAMILIES, FAMILY_LABELS, symbolFor } from "@/lib/blueprint";

export function Chart({
  taxonomy, onSelect,
}: {
  taxonomy: TaxonomyResponse | null;
  onSelect: (name: string) => void;
}) {
  if (!taxonomy) {
    return (
      <div className="flex h-full items-center justify-center text-ink-dim">
        <span className="bp-title text-sm tracking-[0.3em]">surveying the specimens…</span>
      </div>
    );
  }
  const byName = Object.fromEntries(taxonomy.models.map((m) => [m.name, m]));
  const derived = taxonomy.models.filter((m) => m.level === "derived");

  return (
    <div className="h-full overflow-y-auto px-5 py-6 sm:px-10">
      {/* masthead */}
      <motion.header
        initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}
        className="mb-8 flex flex-wrap items-end justify-between gap-4 border-b-[1.5px] border-line pb-4"
      >
        <div>
          <div className="text-[10px] tracking-[0.35em] text-ink-dim">DWG Nº MN-0044 · SHEET 01</div>
          <h1 className="bp-title mt-1 text-3xl font-extrabold leading-none text-ink sm:text-5xl">
            The Periodic Table<span className="text-redline"> of Neural Networks</span>
          </h1>
        </div>
        <div className="bp-stamp px-3 py-1 text-[10px]">
          16 elementary · {derived.length} compounds
        </div>
      </motion.header>

      {/* atoms by family shelf */}
      <div className="grid gap-6 md:grid-cols-2 xl:grid-cols-3">
        {FAMILIES.map((family, fi) => {
          const members = Object.entries(ATOMS).filter(([, a]) => a.family === family);
          return (
            <section key={family}>
              <h2 className="bp-title mb-2 border-b border-dashed border-line-faint pb-1 text-[11px] text-ink-dim">
                {FAMILY_LABELS[family]}
              </h2>
              <div className="flex flex-wrap gap-3">
                {members.map(([name, atom], i) => (
                  <motion.button
                    key={name}
                    onClick={() => onSelect(name)}
                    initial={{ opacity: 0, scale: 0.92 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: 0.15 + fi * 0.08 + i * 0.05, duration: 0.35 }}
                    whileHover={{ y: -4 }}
                    className="group relative h-28 w-24 border-[1.5px] border-line bg-paper-deep/70 text-left transition-colors hover:border-redline"
                  >
                    <span className="absolute left-1.5 top-1 text-[9px] text-ink-dim">{atom.z}</span>
                    <span className="bp-title absolute right-1.5 top-1 text-[8px] text-ink-dim opacity-0 transition-opacity group-hover:opacity-100 group-hover:text-redline">
                      open →
                    </span>
                    <span className="bp-title block pt-6 text-center text-4xl font-extrabold text-ink group-hover:text-redline">
                      {atom.symbol}
                    </span>
                    <span className="absolute inset-x-1 bottom-4 truncate text-center text-[9px] text-ink">
                      {name}
                    </span>
                    <span className="absolute inset-x-1 bottom-1 truncate text-center text-[7.5px] text-ink-dim">
                      {byName[name]?.introduces[0] ?? ""}
                    </span>
                  </motion.button>
                ))}
              </div>
            </section>
          );
        })}
      </div>

      {/* compounds schedule */}
      <section className="mt-10">
        <h2 className="bp-title mb-3 text-sm tracking-[0.3em] text-ink">
          Schedule of Compounds <span className="text-ink-dim">— derived species</span>
        </h2>
        <div className="grid gap-x-8 gap-y-1 border-t border-line-faint pt-2 lg:grid-cols-2">
          {derived.map((m) => (
            <button
              key={m.name}
              onClick={() => onSelect(m.name)}
              className="group grid grid-cols-[2.6rem_9rem_1fr] items-baseline gap-2 border-b border-dashed border-line-faint py-1.5 text-left text-[11px]"
            >
              <span className="bp-title text-sm font-bold text-line group-hover:text-redline">
                {symbolFor(m.name)}
              </span>
              <span className="truncate text-ink group-hover:text-redline">{m.name}</span>
              <span className="truncate text-ink-dim">
                = {m.builds_on.map((p) => symbolFor(p)).join(" + ")}
                {m.introduces.length > 0 && (
                  <span className="text-line"> + {m.introduces.join(", ")}</span>
                )}
              </span>
            </button>
          ))}
        </div>
      </section>
    </div>
  );
}
