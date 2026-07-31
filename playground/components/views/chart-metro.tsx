"use client";

import type { TaxonomyResponse } from "@/lib/types";
import { ATOMS, PLACEMENT, symbolFor } from "@/lib/blueprint";

// Edition 3: the DAG as a transit map. Family = line; atoms = termini
// (squares); derived models = stations; cross-family builds_on = dashed
// transfer links. Coordinates derive from the periodic PLACEMENT grid.
const LINE_COLORS: Record<string, string> = {
  perception: "#e0301e", sequence: "#1d5bd6", generative: "#0f8a4d",
  representation: "#7d3bbd", structure: "#e07c00", decision: "#0e8f8f",
};

const CELL_W = 118;
const CELL_H = 96;
const pos = (name: string) => {
  const p = PLACEMENT[name];
  return { x: 40 + (p.col - 1) * CELL_W + (p.row % 2) * 14, y: 60 + (p.row - 1) * CELL_H };
};

export function ChartMetro({
  taxonomy, onSelect,
}: {
  taxonomy: TaxonomyResponse | null;
  onSelect: (name: string) => void;
}) {
  if (!taxonomy) return null;
  const familyOf: Record<string, string> = {};
  const assign = (name: string): string => {
    if (familyOf[name]) return familyOf[name];
    const atom = ATOMS[name];
    if (atom) return (familyOf[name] = atom.family);
    const m = taxonomy.models.find((x) => x.name === name);
    return (familyOf[name] = m?.builds_on[0] ? assign(m.builds_on[0]) : "structure");
  };
  taxonomy.models.forEach((m) => assign(m.name));

  // one polyline per family, threading members in reading order
  const lines = Object.keys(LINE_COLORS).map((fam) => {
    const members = taxonomy.models
      .filter((m) => familyOf[m.name] === fam && PLACEMENT[m.name])
      .sort((a, b) => {
        const pa = PLACEMENT[a.name], pb = PLACEMENT[b.name];
        return pa.row - pb.row || pa.col - pb.col;
      });
    return { fam, members };
  });

  const transfers = taxonomy.models.flatMap((m) =>
    m.builds_on.slice(1).filter((p) => PLACEMENT[p] && PLACEMENT[m.name])
      .map((p) => ({ from: p, to: m.name })));

  return (
    <div className="h-full overflow-auto px-4 py-5">
      <div className="px-4">
        <div className="text-[10px] tracking-[0.35em] text-ink-dim">NETWORK DIAGRAM · NOT TO SCALE</div>
        <h1 className="bp-title text-3xl font-bold text-ink">The mini_networks Transit Authority</h1>
      </div>
      <svg viewBox="0 0 1960 560" className="mt-2 min-w-[1400px]">
        {lines.map(({ fam, members }) => (
          <polyline key={fam}
            points={members.map((m) => { const p = pos(m.name); return `${p.x},${p.y}`; }).join(" ")}
            fill="none" stroke={LINE_COLORS[fam]} strokeWidth={7}
            strokeLinejoin="round" strokeLinecap="round" opacity={0.85} />
        ))}
        {transfers.map(({ from, to }, i) => {
          const a = pos(from), b = pos(to);
          return <line key={i} x1={a.x} y1={a.y} x2={b.x} y2={b.y}
            stroke="var(--ink-dim)" strokeWidth={2} strokeDasharray="5 5" />;
        })}
        {taxonomy.models.filter((m) => PLACEMENT[m.name]).map((m) => {
          const p = pos(m.name);
          const isAtom = !!ATOMS[m.name];
          const interchange = m.builds_on.length > 1;
          return (
            <g key={m.name} onClick={() => onSelect(m.name)} className="cursor-pointer">
              {isAtom ? (
                <rect x={p.x - 9} y={p.y - 9} width={18} height={18}
                  fill="var(--paper)" stroke={LINE_COLORS[familyOf[m.name]]} strokeWidth={4} />
              ) : (
                <circle cx={p.x} cy={p.y} r={interchange ? 9 : 6}
                  fill="var(--paper)" stroke="var(--ink)" strokeWidth={interchange ? 3.5 : 2.5} />
              )}
              <text x={p.x + 13} y={p.y - 8} fontSize={11.5} fill="var(--ink)"
                fontWeight={isAtom ? 700 : 400} transform={`rotate(-28 ${p.x + 13} ${p.y - 8})`}>
                {m.name}
              </text>
            </g>
          );
        })}
      </svg>
      <div className="flex flex-wrap gap-4 px-4 text-[11px] text-ink">
        {Object.entries(LINE_COLORS).map(([fam, c]) => (
          <span key={fam} className="flex items-center gap-1.5">
            <span className="inline-block h-[5px] w-6 rounded" style={{ background: c }} />
            {fam} line
          </span>
        ))}
        <span className="text-ink-dim">▢ terminus (atom) · ◉ interchange (multi-parent) · ┄ transfer</span>
      </div>
      <div className="mt-4 grid gap-1 px-4 sm:grid-cols-2 lg:grid-cols-3">
        {taxonomy.compositions.map((c) => (
          <button key={c.name} onClick={() => onSelect(c.name)}
            className="border border-line-faint bg-paper-deep/60 px-2 py-1 text-left text-[11px] text-ink hover:border-redline">
            <span className="font-bold">{symbolFor(c.name)}</span> {c.name}
            <span className="text-ink-dim"> — connection via {c.composes.join(" · ")}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
