"use client";

// The blueprint's title block — every real drawing has one. Doubles as the
// sheet navigation: CHART / REACTIONS / OBSERVATORY plus the live status cell.
export type SheetId = "chart" | "reactions" | "observatory";

const SHEETS: { id: SheetId; no: string; label: string }[] = [
  { id: "chart", no: "01", label: "Chart" },
  { id: "reactions", no: "02", label: "Reactions" },
  { id: "observatory", no: "03", label: "Observatory" },
];

export function TitleBlock({
  sheet, setSheet, ok, runCount,
}: {
  sheet: SheetId;
  setSheet: (s: SheetId) => void;
  ok: boolean;
  runCount: number;
}) {
  return (
    <footer className="grid grid-cols-[1fr_auto] items-stretch border-t-[1.5px] border-line bg-paper-deep/80 text-[10px] sm:grid-cols-[auto_1fr_auto]">
      <div className="hidden flex-col justify-center border-r border-line-faint px-4 py-2 sm:flex">
        <span className="bp-title text-[13px] font-bold leading-none text-ink">mini_networks</span>
        <span className="mt-1 text-ink-dim">The Periodic Table of Neural Networks</span>
      </div>

      <nav className="flex items-stretch">
        {SHEETS.map((s) => (
          <button
            key={s.id}
            onClick={() => setSheet(s.id)}
            className={`bp-title flex items-center gap-2 border-r border-line-faint px-4 text-[11px] transition-colors ${
              sheet === s.id
                ? "bg-line text-paper-deep"
                : "text-ink-dim hover:text-redline"
            }`}
          >
            <span className="font-bold">SHT {s.no}</span>
            <span className="hidden md:inline">{s.label}</span>
          </button>
        ))}
      </nav>

      <div className="flex items-center gap-4 px-4 py-2 text-ink-dim">
        <span>
          runs <span className="text-ink">{runCount}</span>
        </span>
        <span className="flex items-center gap-1.5">
          <span
            className={`inline-block h-2 w-2 rounded-full ${ok ? "" : "bp-live"}`}
            style={{ background: ok ? "var(--line)" : "#ff7d6b" }}
          />
          {ok ? "api linked" : "api offline"}
        </span>
        <span className="hidden lg:inline">rev 2026-07 · drawn by observatory</span>
      </div>
    </footer>
  );
}
