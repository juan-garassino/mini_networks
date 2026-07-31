"use client";

import type { TaxonomyResponse } from "@/lib/types";
import { symbolFor } from "@/lib/blueprint";

// Edition 2: the taxonomy as a phosphor terminal tree.
export function ChartTerminal({
  taxonomy, onSelect,
}: {
  taxonomy: TaxonomyResponse | null;
  onSelect: (name: string) => void;
}) {
  if (!taxonomy) return null;
  const childrenOf = (parent: string) =>
    taxonomy.models.filter((m) => m.builds_on[0] === parent);
  const atoms = taxonomy.models.filter((m) => m.level === "elementary");

  const Row = ({ name, depth, note }: { name: string; depth: number; note: string }) => (
    <button
      onClick={() => onSelect(name)}
      className="block w-full text-left text-[13px] leading-6 hover:bg-paper-deep"
    >
      <span className="text-ink-dim">{"  ".repeat(depth)}{depth > 0 ? "└─ " : "▸ "}</span>
      <span className="text-line">[{symbolFor(name)}]</span>{" "}
      <span className="text-ink">{name}</span>{" "}
      <span className="text-ink-dim"># {note}</span>
    </button>
  );

  const renderTree = (name: string, depth: number): React.ReactNode[] => {
    const kids = childrenOf(name);
    return [
      ...kids.map((k) => [
        <Row key={k.name} name={k.name} depth={depth}
          note={k.introduces.join(",") || k.note} />,
        ...renderTree(k.name, depth + 1),
      ]),
    ].flat();
  };

  return (
    <div className="h-full overflow-y-auto px-6 py-6 font-draft">
      <div className="text-[13px] text-ink-dim">$ mini_networks list --tree</div>
      <div className="mb-4 text-[13px] text-ink">
        44 species · 16 atoms · 22 reactions <span className="bp-live text-redline">▮</span>
      </div>
      {atoms.map((a) => (
        <div key={a.name} className="mb-2">
          <Row name={a.name} depth={0} note={a.introduces.join(",")} />
          {renderTree(a.name, 1)}
        </div>
      ))}
      <div className="mt-6 text-[13px] text-ink-dim">$ mini_networks reactions --list</div>
      {taxonomy.compositions.map((c) => (
        <button key={c.name} onClick={() => onSelect(c.name)}
          className="block w-full text-left text-[13px] leading-6 hover:bg-paper-deep">
          <span className="text-redline">✻</span>{" "}
          <span className="text-ink">{c.name}</span>{" "}
          <span className="text-ink-dim">= {c.composes.join(" + ")}</span>
        </button>
      ))}
    </div>
  );
}
