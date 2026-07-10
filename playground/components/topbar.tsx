"use client";

import { useEffect, useState } from "react";
import { Play, Gamepad2, FlaskConical, Trophy } from "lucide-react";
import { Planet } from "@/components/mascots";

export type ViewId = "observatory" | "sandbox" | "lab" | "quest";

const TABS = [
  { id: "observatory", label: "Watch", Icon: Play },
  { id: "sandbox", label: "Play", Icon: Gamepad2 },
  { id: "lab", label: "Lab", Icon: FlaskConical },
  { id: "quest", label: "Quest", Icon: Trophy },
] as const;

export function TopBar({ view, setView, src, ok }: { view: ViewId; setView: (v: ViewId) => void; src: string; ok: boolean }) {
  const [clock, setClock] = useState("--:--:--");
  useEffect(() => {
    const t = () => setClock(new Date().toISOString().slice(11, 19));
    t();
    const id = setInterval(t, 1000);
    return () => clearInterval(id);
  }, []);

  return (
    <header className="relative z-30 flex items-center gap-6 px-6 py-3.5 shadow-[0_10px_30px_rgba(124,78,200,.42)]"
      style={{ background: "linear-gradient(120deg,#7d5fff,#b06ad9 92%)" }}>
      <div className="flex items-center gap-3">
        <span className="grid h-[46px] w-[46px] place-content-center rounded-[14px] shadow-[0_6px_14px_rgba(234,168,0,.45)]"
          style={{ background: "linear-gradient(160deg,#ffd23f,#ffb43f)" }}>
          <svg viewBox="0 0 48 48" width="34" height="34">
            <rect x="9" y="14" width="30" height="24" rx="11" fill="#fff" /><rect x="14" y="6" width="20" height="16" rx="8" fill="#fff" />
            <rect x="17" y="11" width="14" height="7" rx="3.5" fill="#3a2f6b" /><circle cx="21" cy="14.5" r="1.7" fill="#8ad6ff" /><circle cx="27" cy="14.5" r="1.7" fill="#8ad6ff" />
            <circle cx="24" cy="4" r="2" fill="#fff" /><circle cx="16" cy="26" r="3" fill="#ffd23f" /><circle cx="32" cy="26" r="3" fill="#ffd23f" />
          </svg>
        </span>
        <div className="flex flex-col leading-none">
          <span className="font-display text-[22px] font-bold text-white">mini_networks</span>
          <span className="text-xs font-bold tracking-wide text-[#f0d9ff]">the grove · playground</span>
        </div>
      </div>

      <nav className="flex gap-1 rounded-2xl bg-white/15 p-1.5">
        {TABS.map(({ id, label, Icon }) => (
          <button key={id} onClick={() => setView(id)}
            className={`flex items-center gap-1.5 rounded-xl px-4 py-2 font-display text-[15px] font-semibold transition ${
              view === id ? "bg-white text-[#7d5fff] shadow-[0_6px_18px_rgba(74,60,40,.16)]" : "text-[#f3eaff] hover:bg-white/15"
            }`}>
            <Icon size={15} strokeWidth={2.5} /> {label}
          </button>
        ))}
      </nav>

      <div className="ml-auto flex items-center gap-4 font-extrabold text-white">
        <Planet size={46} />
        <span className="hidden items-center gap-1.5 text-[13px] sm:flex"><span className="text-[#ffd23f]">SRC</span>{src.toUpperCase()}</span>
        <span className="hidden items-center gap-1.5 text-[13px] tabular-nums sm:flex"><span className="text-[#ffd23f]">UTC</span>{clock}</span>
        <span className={`h-2.5 w-2.5 rounded-full ${ok ? "bg-[#2ecc71] shadow-[0_0_0_4px_rgba(46,204,113,.22)]" : "bg-[#ff6b6b]"}`} />
      </div>
    </header>
  );
}
