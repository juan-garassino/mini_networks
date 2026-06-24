"use client";

import { useEffect, useRef, useState } from "react";
import { Panel, PanelHead } from "@/components/panel";
import { infer } from "@/lib/api";
import type { RunSummary } from "@/lib/types";

function softmax(a: number[]): number[] {
  if (!a.length) return [];
  const m = Math.max(...a);
  const e = a.map((x) => Math.exp(x - m));
  const s = e.reduce((x, y) => x + y, 0);
  return e.map((x) => x / s);
}

export function Sandbox({ runs }: { runs: RunSummary[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const drawing = useRef(false);
  const [preds, setPreds] = useState<number[] | null>(null);
  const clf = runs
    .filter((r) => r.model === "classifier" && r.source === "local" && r.artifact_names.includes("model.pt"))
    .sort((a, b) => (b.created_at || "").localeCompare(a.created_at || ""))[0];

  const ctx = () => canvasRef.current!.getContext("2d")!;
  const clear = () => { const c = ctx(); c.fillStyle = "#000"; c.fillRect(0, 0, 280, 280); c.beginPath(); setPreds(null); };
  useEffect(() => { clear(); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const pos = (e: React.PointerEvent) => {
    const r = canvasRef.current!.getBoundingClientRect();
    return [(e.clientX - r.left) * (280 / r.width), (e.clientY - r.top) * (280 / r.height)] as const;
  };
  const move = (e: React.PointerEvent) => {
    if (!drawing.current) return;
    const c = ctx();
    const [x, y] = pos(e);
    c.lineWidth = 24; c.lineCap = "round"; c.strokeStyle = "#fff";
    c.lineTo(x, y); c.stroke(); c.beginPath(); c.moveTo(x, y);
  };
  const start = (e: React.PointerEvent) => { drawing.current = true; move(e); };
  const end = () => { drawing.current = false; ctx().beginPath(); };

  const guess = async () => {
    if (!clf) { setPreds(null); return; }
    const off = document.createElement("canvas");
    off.width = 28; off.height = 28;
    const octx = off.getContext("2d")!;
    octx.drawImage(canvasRef.current!, 0, 0, 28, 28);
    const d = octx.getImageData(0, 0, 28, 28).data;
    const plane: number[][] = [];
    for (let i = 0; i < 28; i++) { const row: number[] = []; for (let j = 0; j < 28; j++) row.push(d[(i * 28 + j) * 4] / 255); plane.push(row); }
    try {
      const res = await infer("classifier", { run_id: clf.id, inputs: { images: [[plane]] } });
      const logits = (res.outputs.logits as number[][] | undefined)?.[0] || [];
      setPreds(softmax(logits));
    } catch { setPreds(null); }
  };

  const best = preds ? preds.indexOf(Math.max(...preds)) : -1;

  return (
    <div className="grid h-full min-h-0 grid-cols-2 gap-5 px-6 py-5 max-[900px]:grid-cols-1">
      <Panel i={0}>
        <PanelHead title="Draw a digit" right={<span className="rounded-full bg-[#f1eee4] px-2.5 py-0.5 text-xs font-bold text-[#97907e]">{clf ? `classifier · ${clf.run_name}` : "no classifier — train one in Lab"}</span>} />
        <div className="grid flex-1 place-content-center p-5">
          <canvas ref={canvasRef} width={280} height={280} onPointerDown={start} onPointerMove={move} onPointerUp={end} onPointerLeave={end}
            className="h-[300px] w-[300px] cursor-crosshair touch-none rounded-3xl bg-[#1a1d2e] shadow-[0_5px_16px_rgba(74,60,40,.14)]" />
        </div>
        <div className="flex justify-center gap-3 px-[18px] pb-5">
          <button onClick={guess} className="rounded-2xl bg-[#7d5fff] px-6 py-3 font-display font-bold text-white shadow-[0_8px_18px_rgba(125,95,255,.42)] hover:bg-[#6c4ef0]">▶ Guess</button>
          <button onClick={clear} className="rounded-2xl bg-[#f1ede3] px-6 py-3 font-display font-bold text-[#6b6385]">Clear</button>
        </div>
      </Panel>
      <Panel i={1}>
        <PanelHead title="Prediction" />
        <div className="flex flex-1 flex-col justify-center gap-2.5 px-6 py-4">
          {!preds && <div className="text-center text-[#a59cc0]">{clf ? "Draw a digit 0–9, then press Guess." : "No trained classifier yet — go to Lab, launch classifier (tier M), then come back."}</div>}
          {preds?.map((p, d) => (
            <div key={d} className="grid grid-cols-[26px_1fr_46px] items-center gap-3">
              <span className={`font-display text-[17px] font-bold ${d === best ? "text-[#7d5fff]" : "text-[#a59cc0]"}`}>{d}</span>
              <span className="h-4 overflow-hidden rounded-full bg-[#f1ede3]"><span className="block h-full rounded-full transition-[width] duration-300" style={{ width: `${(p * 100).toFixed(0)}%`, background: d === best ? "linear-gradient(90deg,#ffd23f,#ff9f43)" : "#7d5fff" }} /></span>
              <span className={`text-right font-extrabold ${d === best ? "text-[#7d5fff]" : "text-[#a59cc0]"}`}>{(p * 100).toFixed(0)}</span>
            </div>
          ))}
        </div>
      </Panel>
    </div>
  );
}
