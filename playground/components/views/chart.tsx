"use client";

import { motion } from "motion/react";
import type { TaxonomyResponse } from "@/lib/types";
import {
  ATOMS, FAMILY_BLOCKS, PLACEMENT, TABLE_COLS, TABLE_ROWS,
  atomicNumbers, symbolFor,
} from "@/lib/blueprint";

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
  const z = atomicNumbers();

  return (
    <div className="h-full overflow-auto px-5 py-6 sm:px-10">
      {/* masthead */}
      <motion.header
        initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}
        className="mb-6 flex flex-wrap items-end justify-between gap-4 border-b-[1.5px] border-line pb-4"
      >
        <div>
          <div className="text-[10px] tracking-[0.35em] text-ink-dim">DWG Nº MN-0044 · SHEET 01</div>
          <h1 className="bp-title mt-1 text-3xl font-extrabold leading-none text-ink sm:text-5xl">
            The Periodic Table<span className="text-redline"> of Neural Networks</span>
          </h1>
        </div>
        <div className="bp-stamp px-3 py-1 text-[10px]">
          44 species · 16 elementary · 22 reactions
        </div>
      </motion.header>

      <div className="min-w-[1080px]">
        {/* family group labels */}
        <div className="grid gap-1" style={{ gridTemplateColumns: `repeat(${TABLE_COLS}, minmax(0, 1fr))` }}>
          {FAMILY_BLOCKS.map((b) => (
            <div
              key={b.family}
              className="bp-title border-b border-dashed border-line-faint pb-1 text-[9px] tracking-[0.2em] text-ink-dim"
              style={{ gridColumn: `${b.cols[0]} / ${b.cols[1] + 1}` }}
            >
              {b.label}
            </div>
          ))}
        </div>

        {/* the main table */}
        <div
          className="mt-1 grid gap-1"
          style={{
            gridTemplateColumns: `repeat(${TABLE_COLS}, minmax(0, 1fr))`,
            gridTemplateRows: `repeat(${TABLE_ROWS}, minmax(4.6rem, auto))`,
          }}
        >
          {Object.entries(PLACEMENT).map(([name, pos], i) => {
            const atom = ATOMS[name];
            const taxon = byName[name];
            return (
              <motion.button
                key={name}
                onClick={() => onSelect(name)}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.1 + (pos.row - 1) * 0.07 + pos.col * 0.015, duration: 0.3 }}
                whileHover={{ y: -3 }}
                style={{ gridColumn: pos.col, gridRow: pos.row }}
                className={`group relative text-left transition-colors ${
                  atom
                    ? "border-[1.5px] border-line bg-paper-deep/80 hover:border-redline"
                    : "border border-line-faint bg-paper-deep/40 hover:border-redline"
                }`}
                title={`${name} — ${taxon?.introduces.join(", ") || taxon?.note || ""}`}
              >
                <span className="absolute left-1 top-0.5 text-[8px] text-ink-dim">{z[name]}</span>
                <span
                  className={`bp-title block pt-4 text-center font-extrabold group-hover:text-redline ${
                    atom ? "text-[26px] text-ink" : "text-[20px] text-ink-dim"
                  }`}
                >
                  {symbolFor(name)}
                </span>
                <span className="absolute inset-x-0.5 bottom-0.5 truncate text-center text-[7.5px] text-ink-dim group-hover:text-ink">
                  {name}
                </span>
              </motion.button>
            );
          })}
        </div>

        {/* f-block: reactions (the lanthanides of the zoo) */}
        <div className="mt-5">
          <div className="bp-title mb-1 text-[9px] tracking-[0.25em] text-ink-dim">
            ✻ · Reactions <span className="normal-case tracking-normal">(whole models wired into pipelines — see SHT 02)</span>
          </div>
          <div className="grid gap-1" style={{ gridTemplateColumns: `repeat(${TABLE_COLS}, minmax(0, 1fr))` }}>
            {taxonomy.compositions.map((c, i) => (
              <motion.button
                key={c.name}
                onClick={() => onSelect(c.name)}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.55 + i * 0.02, duration: 0.3 }}
                whileHover={{ y: -3 }}
                style={{ gridColumn: 3 + (i % 11), gridRow: Math.floor(i / 11) + 1 }}
                className="group relative h-[3.4rem] border border-dashed border-line-faint bg-paper-deep/30 text-left transition-colors hover:border-redline"
                title={`${c.name} = ${c.composes.join(" + ")}`}
              >
                <span className="absolute left-1 top-0.5 text-[7px] text-redline/80">✻{i + 1}</span>
                <span className="bp-title block pt-3 text-center text-[15px] font-bold text-ink-dim group-hover:text-redline">
                  {symbolFor(c.name)}
                </span>
                <span className="absolute inset-x-0.5 bottom-0.5 truncate text-center text-[6.5px] text-ink-dim">
                  {c.name}
                </span>
              </motion.button>
            ))}
          </div>
        </div>

        {/* legend */}
        <div className="mt-5 flex flex-wrap gap-5 text-[9px] text-ink-dim">
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-3 w-3 border-[1.5px] border-line bg-paper-deep/80" /> elementary atom
          </span>
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-3 w-3 border border-line-faint bg-paper-deep/40" /> derived compound
          </span>
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-3 w-3 border border-dashed border-line-faint bg-paper-deep/30" /> reaction ✻
          </span>
          <span>periods ≈ generations: each row builds on the rows above it</span>
        </div>
      </div>
    </div>
  );
}
