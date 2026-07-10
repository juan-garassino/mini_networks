"use client";

import { useEffect, useState } from "react";
import { listRuns } from "@/lib/api";
import type { RunSummary } from "@/lib/types";

export function useRuns(pollMs = 1500) {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [ok, setOk] = useState(true);

  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const d = await listRuns();
        if (alive) { setRuns(d.runs); setOk(true); }
      } catch {
        if (alive) setOk(false);
      }
    };
    tick();
    const id = setInterval(tick, pollMs);
    return () => { alive = false; clearInterval(id); };
  }, [pollMs]);

  return { runs, ok };
}
