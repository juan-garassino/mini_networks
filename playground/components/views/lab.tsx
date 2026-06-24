"use client";

import { useEffect, useRef, useState } from "react";
import { Sparkles } from "lucide-react";
import { Panel, PanelHead } from "@/components/panel";
import { Avatar } from "@/components/mascots";
import { listModels, startTrain } from "@/lib/api";
import { fmtNum, primaryMetric } from "@/lib/format";
import { sparkleAt } from "@/lib/fx";
import type { RunSummary } from "@/lib/types";

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return <div className="flex items-center gap-3.5"><span className="w-20 font-display font-semibold text-[#6b6385]">{label}</span>{children}</div>;
}
const INPUT = "rounded-xl border-2 border-[#efe7d8] bg-[#fbf7ec] px-3.5 py-2.5 font-bold text-[#3a3152] outline-none focus:border-[#7d5fff]";

export function Lab({ runs }: { runs: RunSummary[] }) {
  const [models, setModels] = useState<string[]>([]);
  const [model, setModel] = useState("");
  const [tier, setTier] = useState("S");
  const [epochs, setEpochs] = useState(3);
  const [batch, setBatch] = useState(32);
  const [result, setResult] = useState<{ ok?: string; err?: string } | null>(null);
  const btnRef = useRef<HTMLButtonElement>(null);

  useEffect(() => { listModels().then((m) => { setModels(m.map((x) => x.name)); setModel(m[0]?.name || ""); }).catch(() => {}); }, []);

  const launch = async (e: React.FormEvent) => {
    e.preventDefault();
    setResult({ ok: `Launching ${model}…` });
    try {
      const r = await startTrain(model, { epochs, batch_size: batch, fast_demo: tier === "S", training_tier: tier, device: "cpu" });
      setResult({ ok: `✦ ${r.status} · ${r.job_id} — open Watch to see it grow` });
      sparkleAt(btnRef.current, 16);
    } catch (err) { setResult({ err: (err as Error).message }); }
  };

  const recent = runs.slice().sort((a, b) => (b.created_at || "").localeCompare(a.created_at || "")).slice(0, 14);

  return (
    <div className="grid h-full min-h-0 grid-cols-[minmax(0,1fr)_330px] gap-5 px-6 py-5 max-[900px]:grid-cols-1">
      <Panel i={0}>
        <PanelHead title="Plant a seed 🌱" />
        <form onSubmit={launch} className="flex flex-col gap-4 px-[18px] py-2">
          <Field label="Model"><div className="relative flex-1"><select value={model} onChange={(e) => setModel(e.target.value)} className={`w-full cursor-pointer ${INPUT}`}>{models.map((m) => <option key={m}>{m}</option>)}</select></div></Field>
          <Field label="Tier"><div className="flex gap-1.5">{["S", "M", "L"].map((t) => <button type="button" key={t} onClick={() => setTier(t)} className={`h-[42px] w-11 rounded-xl font-display font-bold ${tier === t ? "bg-[#7d5fff] text-white" : "bg-[#f1ede3] text-[#6b6385]"}`}>{t}</button>)}</div></Field>
          <Field label="Epochs"><input type="number" min={1} value={epochs} onChange={(e) => setEpochs(+e.target.value)} className={`w-28 ${INPUT}`} /></Field>
          <Field label="Batch"><input type="number" min={1} value={batch} onChange={(e) => setBatch(+e.target.value)} className={`w-28 ${INPUT}`} /></Field>
          <button ref={btnRef} type="submit" className="mt-1.5 flex items-center justify-center gap-2 rounded-2xl bg-[#7d5fff] py-3.5 font-display text-base font-bold text-white shadow-[0_8px_18px_rgba(125,95,255,.42)] transition hover:-translate-y-0.5 hover:bg-[#6c4ef0]"><Sparkles size={18} /> Start training</button>
        </form>
        {result && <div className={`px-[18px] pb-4 font-extrabold ${result.err ? "text-[#e0533f]" : "text-[#27ae60]"}`}>{result.err || result.ok}</div>}
      </Panel>
      <Panel i={1}>
        <PanelHead title="Recent launches" />
        <div className="min-h-0 flex-1 space-y-1 overflow-y-auto px-3 pb-3">
          {recent.map((r) => {
            const pm = primaryMetric(r.last_metrics);
            return (
              <div key={r.id} className="grid grid-cols-[36px_1fr_auto] items-center gap-3 rounded-2xl p-2.5">
                <Avatar model={r.model} />
                <span className="min-w-0"><span className="block truncate font-display font-semibold text-[#3a3152]">{r.model}</span><span className="block truncate text-[11px] text-[#a59cc0]">{r.run_name || r.id}</span></span>
                <span className="text-right">{pm && <><span className="block font-display font-semibold text-[#3a3152]">{fmtNum(pm.v)}</span><span className="block text-[10px] uppercase text-[#a59cc0]">{pm.k}</span></>}</span>
              </div>
            );
          })}
        </div>
      </Panel>
    </div>
  );
}
