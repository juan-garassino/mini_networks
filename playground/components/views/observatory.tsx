"use client";

import { useEffect, useRef, useState } from "react";
import { Settings, ClipboardList, PawPrint } from "lucide-react";
import { Panel, PanelHead, Chip } from "@/components/panel";
import { Avatar, Dragon, Bot, Sprout, Star, Plane } from "@/components/mascots";
import { LossChart, plottable } from "@/components/loss-chart";
import { getMetrics, getConfig, getSummary, artifactUrl } from "@/lib/api";
import { fmtNum, statusMeta, primaryMetric, relTime } from "@/lib/format";
import { sparkleAt } from "@/lib/fx";
import type { RunSummary, MetricsResponse } from "@/lib/types";

const FILTERS: [string, string][] = [["all", "All"], ["running", "Live"], ["done", "Done"], ["failed", "Fail"]];
const IMG = /\.(png|jpe?g|gif|webp|svg)$/i;

function fmtVal(v: unknown) {
  if (typeof v === "number") return fmtNum(v);
  if (typeof v === "boolean") return v ? "yes" : "no";
  const s = String(v);
  return s.length > 22 ? s.slice(0, 21) + "…" : s;
}

export function Observatory({ runs }: { runs: RunSummary[] }) {
  const [filter, setFilter] = useState("all");
  const [selId, setSelId] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [config, setConfig] = useState<Record<string, unknown>>({});
  const [summary, setSummary] = useState<Record<string, unknown>>({});
  const dragonRef = useRef<HTMLDivElement>(null);
  const prevStatus = useRef<string | null>(null);

  const visible = runs
    .filter((r) => (filter === "all" ? true : filter === "running" ? r.status === "running" || r.status === "dispatched" : r.status === filter))
    .slice()
    .sort((a, b) => (b.created_at || "").localeCompare(a.created_at || ""));
  const sel = runs.find((r) => r.id === selId) ?? null;
  const running = !!sel && (sel.status === "running" || sel.status === "dispatched");

  useEffect(() => {
    if ((!selId || !runs.some((r) => r.id === selId)) && runs.length) setSelId((visible[0] || runs[0]).id);
  }, [runs]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!selId) return;
    let alive = true;
    const load = async () => {
      try {
        const [m, c, s] = await Promise.all([getMetrics(selId), getConfig(selId), getSummary(selId)]);
        if (alive) { setMetrics(m); setConfig(c.config); setSummary(s.summary); }
      } catch { /* transient */ }
    };
    load();
    const id = running ? setInterval(load, 1500) : undefined;
    return () => { alive = false; if (id) clearInterval(id); };
  }, [selId, running]);

  useEffect(() => {
    if (!sel) return;
    if (prevStatus.current && prevStatus.current !== sel.status && sel.status === "done") {
      const el = dragonRef.current;
      if (el) { el.classList.add("celebrate"); sparkleAt(el.querySelector("svg")); setTimeout(() => el.classList.remove("celebrate"), 950); }
    }
    prevStatus.current = sel.status;
  }, [sel?.status]); // eslint-disable-line react-hooks/exhaustive-deps

  const stats = metrics ? plottable(metrics.series).map((s) => ({ key: s.key, color: s.color, val: s.points.at(-1)?.[1] })) : [];

  return (
    <div className="grid h-full min-h-0 grid-cols-[310px_minmax(0,1fr)_300px] gap-5 px-6 py-5 max-[1240px]:grid-cols-[270px_minmax(0,1fr)]">
      {/* RUNS */}
      <Panel i={0}>
        <PanelHead title="Runs" right={<span className="rounded-full bg-[#fff1c2] px-2.5 py-0.5 font-display text-xs font-bold text-[#b8860b]">{runs.length}</span>} />
        <div className="flex gap-1.5 px-[18px] pb-3">
          {FILTERS.map(([f, l]) => (
            <button key={f} onClick={() => setFilter(f)}
              className={`rounded-full px-3.5 py-1.5 font-display text-[13px] font-semibold transition ${filter === f ? "bg-[#7d5fff] text-white" : "bg-[#f3f0e6] text-[#6b6385] hover:bg-[#ece6d6]"}`}>{l}</button>
          ))}
        </div>
        <div className="min-h-0 flex-1 space-y-1 overflow-y-auto px-3 pb-3">
          {visible.length === 0 && <div className="py-8 text-center text-sm text-[#a59cc0]">No runs match</div>}
          {visible.map((r) => {
            const pm = primaryMetric(r.last_metrics);
            return (
              <button key={r.id} onClick={() => setSelId(r.id)}
                className={`grid w-full grid-cols-[36px_1fr_auto] items-center gap-3 rounded-2xl border-2 p-2.5 text-left transition hover:translate-x-0.5 ${selId === r.id ? "border-[#7d5fff] bg-[#f4efff]" : "border-transparent hover:bg-[#fbf7ec]"}`}>
                <Avatar model={r.model} />
                <span className="min-w-0">
                  <span className="block truncate font-display text-[15px] font-semibold text-[#3a3152]">{r.model}</span>
                  <span className="block truncate text-[11px] text-[#a59cc0]">{r.run_name || r.id}</span>
                </span>
                <span className="text-right">{pm && (<><span className="block font-display text-[15px] font-semibold text-[#3a3152]">{fmtNum(pm.v)}</span><span className="block text-[10px] uppercase text-[#a59cc0]">{pm.k}</span></>)}</span>
              </button>
            );
          })}
        </div>
      </Panel>

      {/* STAGE */}
      <div className="grid min-h-0 grid-rows-[auto_auto] content-start gap-5">
        <Panel i={1} className="relative overflow-visible">
          <div className="flex flex-wrap items-start gap-3 px-5 pt-4">
            <div className="flex flex-wrap items-center gap-2.5">
              <span className="font-sans text-[11px] font-extrabold uppercase tracking-[.14em] text-[#a59cc0]">Scope</span>
              <h1 className="font-display text-[28px] font-bold text-[#3a3152]">{sel ? sel.model : "Select a run"}</h1>
              <Star />
            </div>
            <div className="flex flex-wrap items-center gap-2">
              {sel && <Chip tone="green">{sel.source}</Chip>}
              {typeof config.training_tier === "string" && <Chip tone="violet">tier {String(config.training_tier)}</Chip>}
              {typeof config.device === "string" && <Chip tone="blue">{String(config.device)}</Chip>}
              {sel && <Chip>{statusMeta(sel.status).label}</Chip>}
              {sel?.created_at && <Chip>{relTime(sel.created_at)}</Chip>}
            </div>
            {running && <span className="ml-auto flex items-center gap-1.5 font-display font-bold text-[#ff6b6b]"><span className="h-2.5 w-2.5 animate-pulse rounded-full bg-[#ff6b6b]" />REC</span>}
          </div>
          <div className="mx-[18px] mt-1.5 h-[clamp(220px,30vh,320px)]">
            {metrics ? <LossChart series={metrics.series} /> : <div className="grid h-full place-content-center text-sm text-[#a59cc0]">{runs.length ? "Loading…" : "No runs yet — plant a seed in Lab 🌱"}</div>}
          </div>
          <div ref={dragonRef} className="pointer-events-none absolute bottom-[88px] right-6 z-[2] w-[118px] text-center">
            <div className="absolute -top-9 right-14 animate-[bob_2.8s_ease-in-out_infinite] whitespace-nowrap rounded-2xl bg-white px-3 py-2 font-display text-[13px] font-semibold text-[#7d5fff] shadow-[0_6px_18px_rgba(74,60,40,.16)]">Learning in progress! 🚀</div>
            <Dragon />
          </div>
          <div className="flex flex-wrap gap-3.5 px-5 pb-4 pt-3.5">
            {stats.map((s) => (
              <div key={s.key} className="flex items-center gap-2.5 rounded-2xl bg-[#f9efdb] px-4 py-2.5 shadow-[inset_0_1px_0_rgba(255,255,255,.7)]">
                <span className="h-3.5 w-3.5 rounded-md" style={{ background: s.color }} />
                <span className="text-[13px] font-extrabold text-[#6b6385]">{s.key}</span>
                <span className="font-display text-[19px] font-bold" style={{ color: s.color }}>{fmtNum(s.val)}</span>
              </div>
            ))}
          </div>
        </Panel>

        <Panel i={2} className="relative overflow-visible">
          <PanelHead icon={<PawPrint size={18} className="text-[#5da648]" />} title="Specimens" right={<span className="rounded-full bg-[#fff1c2] px-2.5 py-0.5 font-display text-xs font-bold text-[#b8860b]">{sel?.artifact_names.length ?? 0}</span>} />
          <div className="relative min-h-[130px]">
            <div className="flex gap-3 overflow-x-auto px-[18px] pb-4 pt-1.5">
              {sel?.artifact_names.length
                ? sel.artifact_names.map((name) => {
                    const url = artifactUrl(sel.id, name);
                    return IMG.test(name) ? (
                      // eslint-disable-next-line @next/next/no-img-element
                      <a key={name} href={url} target="_blank" className="h-[70px] w-[70px] flex-none overflow-hidden rounded-2xl bg-[#1a1d2e] shadow-[0_5px_16px_rgba(74,60,40,.14)]"><img src={url} alt={name} className="h-full w-full object-contain [image-rendering:pixelated]" /></a>
                    ) : (
                      <a key={name} href={url} target="_blank" className="flex h-[70px] flex-none items-center rounded-2xl bg-[#1a1d2e] px-3.5 font-display text-[13px] font-semibold text-[#8be36a]">{name}</a>
                    );
                  })
                : <div className="px-3 py-6 text-sm text-[#a59cc0]">no specimens yet</div>}
            </div>
            <div className="pointer-events-none absolute -bottom-2 right-6 z-[2]"><Bot /></div>
            <div className="pointer-events-none absolute bottom-[70px] left-[200px] z-[2]"><Plane /></div>
          </div>
        </Panel>
      </div>

      {/* SIDE */}
      <div className="grid min-h-0 content-start gap-5 max-[1240px]:hidden">
        <Panel i={2}>
          <PanelHead icon={<Settings size={18} className="text-[#7d5fff]" />} title="Config" right={<Sprout />} />
          <div className="max-h-[42vh] overflow-y-auto px-[18px] pb-4">
            {Object.entries(config).length ? Object.entries(config).map(([k, v]) => (
              <div key={k} className="flex items-center gap-2 border-b border-[#efe7d8] py-1.5 last:border-0">
                <span className="h-2.5 w-2.5 flex-none rounded-[3px] bg-[#e6dcc4]" />
                <span className="text-[13px] font-bold text-[#6b6385]">{k}</span>
                <span className="flex-1" />
                <span className="truncate text-right text-[13px] font-extrabold tabular-nums text-[#3a3152]" title={String(v)}>{fmtVal(v)}</span>
              </div>
            )) : <div className="py-6 text-center text-sm text-[#a59cc0]">no config</div>}
          </div>
        </Panel>
        <Panel i={3}>
          <PanelHead icon={<ClipboardList size={18} className="text-[#7d5fff]" />} title="Summary" />
          <div className="px-[18px] pb-4">
            {Object.entries(summary).length ? Object.entries(summary).map(([k, v]) => (
              <div key={k} className="flex items-center gap-2.5 py-2 text-sm">
                <span className="font-extrabold text-[#6b6385]">{k}</span>
                <span className="ml-auto font-extrabold text-[#3a3152]">
                  {k === "status" ? <span className={v === "completed" ? "rounded-full bg-[#e3f9ec] px-3 py-1 text-[#23b26d]" : v === "failed" ? "rounded-full bg-[#ffe6e6] px-3 py-1 text-[#e0533f]" : ""}>{String(v)}{v === "completed" ? " ✓" : ""}</span> : fmtVal(v)}
                </span>
              </div>
            )) : <div className="py-6 text-center text-sm text-[#a59cc0]">awaiting completion</div>}
          </div>
        </Panel>
      </div>
    </div>
  );
}
