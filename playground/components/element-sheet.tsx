"use client";

import { useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { infer } from "@/lib/api";
import type { RunSummary, TaxonomyResponse } from "@/lib/types";
import { ATOMS, STATUS_INK, symbolFor } from "@/lib/blueprint";

export function ElementSheet({
  name, taxonomy, runs, onSelect, onClose,
}: {
  name: string | null;
  taxonomy: TaxonomyResponse | null;
  runs: RunSummary[];
  onSelect: (n: string) => void;
  onClose: () => void;
}) {
  const [inferBody, setInferBody] = useState('{\n  "prompt": "To be"\n}');
  const [inferOut, setInferOut] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const model = taxonomy?.models.find((m) => m.name === name);
  const children = taxonomy?.models.filter((m) => name && m.builds_on.includes(name)) ?? [];
  const reactions = taxonomy?.compositions.filter((c) => name && c.composes.includes(name)) ?? [];
  const modelRuns = runs.filter((r) => r.model === name).slice(0, 6);

  const runInfer = async () => {
    if (!name) return;
    setBusy(true);
    setInferOut(null);
    try {
      const out = await infer(name, JSON.parse(inferBody));
      setInferOut(JSON.stringify(out.outputs ?? out, null, 2).slice(0, 2000));
    } catch (e) {
      setInferOut(`✗ ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <AnimatePresence>
      {name && model && (
        <>
          <motion.div
            className="absolute inset-0 z-20 bg-paper-deep/70"
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            onClick={onClose}
          />
          <motion.aside
            initial={{ x: "100%" }} animate={{ x: 0 }} exit={{ x: "100%" }}
            transition={{ type: "tween", duration: 0.28, ease: "easeOut" }}
            className="absolute inset-y-0 right-0 z-30 w-full max-w-lg overflow-y-auto border-l-[1.5px] border-line bg-paper px-6 py-6"
          >
            <div className="flex items-start justify-between">
              <div>
                <div className="text-[10px] tracking-[0.3em] text-ink-dim">
                  SPECIMEN Nº {ATOMS[name]?.z ?? "—"} · SPEC SHEET
                </div>
                <div className="flex items-baseline gap-3">
                  <span className="bp-title text-6xl font-extrabold text-redline">{symbolFor(name)}</span>
                  <h2 className="bp-title text-2xl font-bold text-ink">{name}</h2>
                </div>
              </div>
              <button onClick={onClose} className="bp-title text-xs text-ink-dim hover:text-redline">
                close ✕
              </button>
            </div>

            <div className="bp-stamp mt-3 inline-block px-2 py-0.5 text-[9px]">
              {model.level === "elementary" ? "elementary · atom" : "derived · compound"}
            </div>

            <p className="mt-4 text-[12px] leading-relaxed text-ink">{model.description}</p>
            {model.note && <p className="mt-1 text-[11px] italic text-ink-dim">“{model.note}”</p>}

            {model.introduces.length > 0 && (
              <Section title="Introduces">
                {model.introduces.map((mech) => (
                  <div key={mech} className="border-b border-dashed border-line-faint py-1.5 text-[11px]">
                    <span className="text-line">{mech}</span>
                    <span className="text-ink-dim"> — {taxonomy?.mechanisms[mech]?.description}</span>
                  </div>
                ))}
              </Section>
            )}

            {model.builds_on.length > 0 && (
              <Section title="Built from">
                <Chips names={model.builds_on} onSelect={onSelect} />
              </Section>
            )}
            {children.length > 0 && (
              <Section title="Used by">
                <Chips names={children.map((c) => c.name)} onSelect={onSelect} />
              </Section>
            )}
            {reactions.length > 0 && (
              <Section title="Appears in reactions">
                <div className="text-[11px] text-ink-dim">
                  {reactions.map((r) => r.name).join(" · ")}
                </div>
              </Section>
            )}

            <Section title={`Field runs (${modelRuns.length})`}>
              {modelRuns.length === 0 && (
                <div className="text-[11px] text-ink-dim">no runs on record for this specimen</div>
              )}
              {modelRuns.map((r) => (
                <div key={r.id} className="grid grid-cols-[auto_1fr_auto] items-baseline gap-2 border-b border-dashed border-line-faint py-1.5 text-[10px]">
                  <span className={r.status === "running" ? "bp-live" : ""} style={{ color: STATUS_INK[r.status] }}>
                    ◉ {r.status}
                  </span>
                  <span className="truncate text-ink">{r.run_name ?? r.id}</span>
                  <span className="text-ink-dim">
                    {Object.entries(r.last_metrics).slice(0, 1).map(([k, v]) => `${k} ${v.toFixed(3)}`)}
                  </span>
                </div>
              ))}
            </Section>

            <Section title="Field test — POST /infer">
              <textarea
                value={inferBody}
                onChange={(e) => setInferBody(e.target.value)}
                rows={3}
                spellCheck={false}
                className="w-full border-[1.5px] border-line-faint bg-paper-deep p-2 text-[11px] text-ink outline-none focus:border-line"
              />
              <button
                onClick={runInfer}
                disabled={busy}
                className="bp-title mt-2 border-[1.5px] border-redline px-3 py-1 text-[10px] text-redline transition-colors hover:bg-redline hover:text-paper-deep disabled:opacity-40"
              >
                {busy ? "measuring…" : "run inference"}
              </button>
              {inferOut && (
                <pre className="mt-2 max-h-48 overflow-auto border border-line-faint bg-paper-deep p-2 text-[10px] text-ink-dim">
                  {inferOut}
                </pre>
              )}
            </Section>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="mt-5">
      <h3 className="bp-title mb-1 text-[10px] tracking-[0.3em] text-ink-dim">{title}</h3>
      {children}
    </section>
  );
}

function Chips({ names, onSelect }: { names: string[]; onSelect: (n: string) => void }) {
  return (
    <div className="flex flex-wrap gap-2">
      {names.map((n) => (
        <button
          key={n}
          onClick={() => onSelect(n)}
          className="border border-line-faint px-2 py-0.5 text-[10px] text-ink transition-colors hover:border-redline hover:text-redline"
        >
          <span className="bp-title mr-1 font-bold text-line">{symbolFor(n)}</span>
          {n}
        </button>
      ))}
    </div>
  );
}
