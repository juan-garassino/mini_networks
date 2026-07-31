"use client";

import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { getTaxonomy } from "@/lib/api";
import type { TaxonomyResponse } from "@/lib/types";
import { useRuns } from "@/hooks/use-runs";
import { TitleBlock, type EditionId, type SheetId } from "@/components/title-block";
import { ElementSheet } from "@/components/element-sheet";
import { Chart } from "@/components/views/chart";
import { ChartTerminal } from "@/components/views/chart-terminal";
import { ChartMetro } from "@/components/views/chart-metro";
import { ChartAtlas } from "@/components/views/chart-atlas";
import { Reactions } from "@/components/views/reactions";
import { Observatory } from "@/components/views/observatory";

export default function Page() {
  const [sheet, setSheet] = useState<SheetId>("chart");
  const [selected, setSelected] = useState<string | null>(null);
  const [taxonomy, setTaxonomy] = useState<TaxonomyResponse | null>(null);
  const [edition, setEdition] = useState<EditionId>("blueprint");
  const { runs, ok } = useRuns();

  useEffect(() => {
    getTaxonomy().then(setTaxonomy).catch(() => setTaxonomy(null));
    const saved = localStorage.getItem("mn-edition") as EditionId | null;
    if (saved) setEdition(saved);
  }, []);

  useEffect(() => {
    document.documentElement.setAttribute("data-edition", edition);
    localStorage.setItem("mn-edition", edition);
  }, [edition]);

  return (
    <>
      <main className="relative min-h-0">
        <AnimatePresence mode="wait">
          <motion.div
            key={`${sheet}-${edition}`}
            className="absolute inset-0"
            initial={{ opacity: 0, x: 24 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -24 }}
            transition={{ duration: 0.22, ease: "easeOut" }}
          >
            {sheet === "chart" && (
              edition === "terminal" ? <ChartTerminal taxonomy={taxonomy} onSelect={setSelected} />
              : edition === "metro" ? <ChartMetro taxonomy={taxonomy} onSelect={setSelected} />
              : edition === "atlas" ? <ChartAtlas taxonomy={taxonomy} onSelect={setSelected} />
              : <Chart taxonomy={taxonomy} onSelect={setSelected} />
            )}
            {sheet === "reactions" && (
              <Reactions taxonomy={taxonomy} runs={runs} onSelect={setSelected} />
            )}
            {sheet === "observatory" && (
              <Observatory runs={runs} onSelectModel={setSelected} />
            )}
          </motion.div>
        </AnimatePresence>

        <ElementSheet
          name={selected}
          taxonomy={taxonomy}
          runs={runs}
          onSelect={setSelected}
          onClose={() => setSelected(null)}
        />
      </main>
      <TitleBlock sheet={sheet} setSheet={setSheet} ok={ok} runCount={runs.length} edition={edition} setEdition={setEdition} />
    </>
  );
}
