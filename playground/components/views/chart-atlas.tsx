"use client";

import { motion } from "motion/react";
import type { TaxonomyResponse } from "@/lib/types";
import { ATOMS } from "@/lib/blueprint";
import { iconFor } from "@/lib/icons";

// Edition 4: a naturalist's atlas — every species gets a plate.
const GENUS: Record<string, string> = {
  perception: "Perceptio", sequence: "Sequentia", generative: "Generativa",
  representation: "Repraesentatio", structure: "Structura", decision: "Decisio",
};

const ROMAN = ["I","II","III","IV","V","VI","VII","VIII","IX","X","XI","XII","XIII","XIV","XV","XVI","XVII","XVIII","XIX","XX",
  "XXI","XXII","XXIII","XXIV","XXV","XXVI","XXVII","XXVIII","XXIX","XXX","XXXI","XXXII","XXXIII","XXXIV","XXXV","XXXVI",
  "XXXVII","XXXVIII","XXXIX","XL","XLI","XLII","XLIII","XLIV"];

export function ChartAtlas({
  taxonomy, onSelect,
}: {
  taxonomy: TaxonomyResponse | null;
  onSelect: (name: string) => void;
}) {
  if (!taxonomy) return null;
  const familyOf = (name: string): string => {
    if (ATOMS[name]) return ATOMS[name].family;
    const m = taxonomy.models.find((x) => x.name === name);
    return m?.builds_on[0] ? familyOf(m.builds_on[0]) : "structure";
  };
  const ordered = [
    ...taxonomy.models.filter((m) => m.level === "elementary"),
    ...taxonomy.models.filter((m) => m.level === "derived"),
  ];

  return (
    <div className="h-full overflow-y-auto px-6 py-6 sm:px-12">
      <header className="mb-6 border-b-2 border-line pb-3 text-center">
        <div className="text-[10px] tracking-[0.4em] text-ink-dim">VOL. XIV · MMXXVI</div>
        <h1 className="bp-title text-4xl font-bold text-ink">An Atlas of Neural Species</h1>
        <div className="mt-1 text-[11px] italic text-ink-dim">
          with forty-four plates, drawn from life in the observatory
        </div>
      </header>

      <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
        {ordered.map((m, i) => {
          const Icon = iconFor(m.name);
          const fam = familyOf(m.name);
          return (
            <motion.button
              key={m.name}
              onClick={() => onSelect(m.name)}
              initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
              transition={{ delay: Math.min(i * 0.02, 0.6), duration: 0.35 }}
              className="group border-2 border-line bg-paper-deep/50 p-1 text-left transition-transform hover:-translate-y-1"
            >
              <div className="border border-line-faint px-3 py-3">
                <div className="flex items-baseline justify-between text-[9px] text-ink-dim">
                  <span>PLATE {ROMAN[i]}</span>
                  <span>{m.level === "elementary" ? "typus" : "hybrida"}</span>
                </div>
                <div className="flex h-24 items-center justify-center">
                  {Icon && <Icon size={56} strokeWidth={0.9} className="text-line transition-colors group-hover:text-redline" />}
                </div>
                <div className="border-t border-line-faint pt-2 text-center">
                  <div className="bp-title text-lg font-semibold italic text-ink">
                    {GENUS[fam]} {m.name.replace(/_/g, " ")}
                  </div>
                  <div className="mt-0.5 line-clamp-2 text-[10px] leading-snug text-ink-dim">
                    {m.introduces.length > 0 ? m.introduces.join(" · ") : m.note || m.description}
                  </div>
                </div>
              </div>
            </motion.button>
          );
        })}
      </div>

      <h2 className="bp-title mt-8 border-b border-line pb-1 text-2xl text-ink">Observed symbioses</h2>
      <div className="mt-3 grid gap-x-8 gap-y-2 sm:grid-cols-2">
        {taxonomy.compositions.map((c) => (
          <button key={c.name} onClick={() => onSelect(c.name)}
            className="border-b border-dotted border-line-faint pb-1 text-left text-[12px] hover:text-redline">
            <span className="italic text-ink">{c.name.replace(/_/g, " ")}</span>
            <span className="text-ink-dim"> — a symbiosis of {c.composes.join(" and ")}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
