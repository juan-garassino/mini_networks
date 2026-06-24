"use client";

import { useEffect, useState } from "react";
import { marked } from "marked";
import { Panel, PanelHead } from "@/components/panel";
import { listLessons, getLesson } from "@/lib/api";
import type { Lesson } from "@/lib/types";

const EMPTY = '<div class="py-12 text-center text-[#a59cc0]">Pick a level from the map ◀</div>';

export function Quest() {
  const [lessons, setLessons] = useState<Lesson[]>([]);
  const [active, setActive] = useState<string | null>(null);
  const [html, setHtml] = useState(EMPTY);

  const open = async (id: string) => {
    setActive(id);
    setHtml('<div class="py-12 text-center text-[#a59cc0]">Loading…</div>');
    try {
      const { markdown } = await getLesson(id);
      setHtml(marked.parse(markdown) as string);
    } catch (e) {
      setHtml(`<div class="py-12 text-center text-[#e0533f]">${(e as Error).message}</div>`);
    }
  };

  useEffect(() => {
    listLessons().then((ls) => { setLessons(ls); if (ls[0]) open(ls[0].id); }).catch(() => {});
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="grid h-full min-h-0 grid-cols-[380px_minmax(0,1fr)] gap-5 px-6 py-5 max-[900px]:grid-cols-1">
      <Panel i={0}>
        <PanelHead title="World map" right={<span className="rounded-full bg-[#fff1c2] px-2.5 py-0.5 font-display text-xs font-bold text-[#b8860b]">{lessons.length}</span>} />
        <div className="grid grid-cols-3 gap-3 overflow-y-auto px-[18px] py-3.5">
          {lessons.map((l) => {
            const nm = l.title.replace(/^\D*\d+\s*[—:-]\s*/, "").split(":")[0].slice(0, 24);
            const on = active === l.id;
            return (
              <button key={l.id} onClick={() => open(l.id)}
                className={`flex aspect-square flex-col items-center justify-center gap-1 rounded-[18px] p-2 shadow-[0_5px_16px_rgba(74,60,40,.14)] transition hover:-translate-y-0.5 ${on ? "bg-[linear-gradient(160deg,#8be36a,#54c24a)]" : "bg-[linear-gradient(160deg,#ffd86a,#ffb43f)]"}`}>
                <span className="font-display text-[22px] font-bold text-white">{l.num}</span>
                <span className={`text-center text-xs font-extrabold leading-tight ${on ? "text-[#245218]" : "text-[#6b4a17]"}`}>{nm}</span>
              </button>
            );
          })}
        </div>
      </Panel>
      <Panel i={1}>
        <PanelHead title={active || "Select a chapter"} />
        <div className="lesson overflow-y-auto px-6 py-4" dangerouslySetInnerHTML={{ __html: html }} />
      </Panel>
    </div>
  );
}
