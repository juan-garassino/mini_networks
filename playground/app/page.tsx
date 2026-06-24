"use client";

import { useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { TopBar, type ViewId } from "@/components/topbar";
import { StatusBar } from "@/components/statusbar";
import { Observatory } from "@/components/views/observatory";
import { Sandbox } from "@/components/views/sandbox";
import { Lab } from "@/components/views/lab";
import { Quest } from "@/components/views/quest";
import { useRuns } from "@/hooks/use-runs";

export default function Page() {
  const [view, setView] = useState<ViewId>("observatory");
  const { runs, ok } = useRuns();
  const src = runs[0]?.source ?? "local";

  return (
    <>
      <TopBar view={view} setView={setView} src={src} ok={ok} />
      <main className="relative min-h-0">
        <AnimatePresence mode="wait">
          <motion.div key={view} className="absolute inset-0"
            initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }} transition={{ duration: 0.26 }}>
            {view === "observatory" && <Observatory runs={runs} />}
            {view === "sandbox" && <Sandbox runs={runs} />}
            {view === "lab" && <Lab runs={runs} />}
            {view === "quest" && <Quest />}
          </motion.div>
        </AnimatePresence>
      </main>
      <StatusBar count={runs.length} src={src} ok={ok} />
      <div id="sparkles" className="sparkles" />
    </>
  );
}
