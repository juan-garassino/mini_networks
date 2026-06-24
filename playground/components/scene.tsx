"use client";

import { useEffect, useMemo, useState } from "react";
import { motion, useMotionValue, useSpring, useTransform, useReducedMotion, type MotionValue } from "motion/react";

function Layer({ depth, className, sx, sy, children }: {
  depth: number; className: string; sx: MotionValue<number>; sy: MotionValue<number>; children: React.ReactNode;
}) {
  const x = useTransform(sx, (v) => v * depth);
  const y = useTransform(sy, (v) => v * depth);
  return <motion.div className={`layer ${className}`} style={{ x, y }}>{children}</motion.div>;
}

const Cloud = ({ cls }: { cls: string }) => (
  <svg className={`cloud ${cls}`} viewBox="0 0 120 50">
    <g fill="#fff">
      <ellipse cx="40" cy="30" rx="30" ry="18" /><ellipse cx="70" cy="28" rx="26" ry="20" />
      <ellipse cx="92" cy="34" rx="20" ry="14" /><rect x="30" y="34" width="70" height="14" rx="7" />
    </g>
  </svg>
);

const Butterfly = ({ cls, a, b }: { cls: string; a: string; b: string }) => (
  <svg className={`bfly ${cls}`} viewBox="0 0 40 34">
    <g className="bw"><path d="M20 17 C6 0 -4 8 6 18 C-2 26 10 30 20 17Z" fill={a} /></g>
    <g className="bw r"><path d="M20 17 C6 0 -4 8 6 18 C-2 26 10 30 20 17Z" fill={b} /></g>
    <rect x="18.5" y="6" width="3" height="22" rx="1.5" fill="#3a2c4a" />
  </svg>
);

export function Scene() {
  const reduce = useReducedMotion();
  const mx = useMotionValue(0), my = useMotionValue(0);
  const sx = useSpring(mx, { stiffness: 55, damping: 18 });
  const sy = useSpring(my, { stiffness: 55, damping: 18 });
  const [bugs, setBugs] = useState<{ l: number; t: number; w: number; d: number; a: number; g: number }[]>([]);
  const [spores, setSpores] = useState<{ l: number; d: number; delay: number }[]>([]);

  useEffect(() => {
    const rnd = (n: number) => Math.random() * n;
    setBugs(Array.from({ length: 22 }, () => ({ l: rnd(100), t: 38 + rnd(56), w: 4 + rnd(5), d: 7 + rnd(8), a: -rnd(15), g: -rnd(2) })));
    setSpores(Array.from({ length: 14 }, () => ({ l: rnd(100), d: 14 + rnd(10), delay: -rnd(16) })));
    if (reduce) return;
    const onMove = (e: MouseEvent) => {
      mx.set((e.clientX / window.innerWidth - 0.5) * 22);
      my.set((e.clientY / window.innerHeight - 0.5) * 22);
    };
    window.addEventListener("mousemove", onMove);
    return () => window.removeEventListener("mousemove", onMove);
  }, [reduce, mx, my]);

  const defs = useMemo(() => (
    <svg width="0" height="0" aria-hidden><defs>
      <linearGradient id="gBody" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="#a6e681" /><stop offset="1" stopColor="#5da648" /></linearGradient>
      <radialGradient id="gBelly" cx=".5" cy=".38" r=".75"><stop offset="0" stopColor="#f0fbd6" /><stop offset="1" stopColor="#cdf0ac" /></radialGradient>
      <linearGradient id="gWing" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stopColor="#d3b6ff" /><stop offset="1" stopColor="#9a7bff" /></linearGradient>
      <linearGradient id="gTree" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="#67b84e" /><stop offset="1" stopColor="#3f7d2c" /></linearGradient>
      <linearGradient id="gHillFar" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="#cfeaae" /><stop offset="1" stopColor="#a9d989" /></linearGradient>
      <linearGradient id="gHillMid" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="#8fcf63" /><stop offset="1" stopColor="#5da648" /></linearGradient>
      <radialGradient id="gSun" cx=".5" cy=".5" r=".5"><stop offset="0" stopColor="#fff6c8" /><stop offset=".6" stopColor="#ffe07a" /><stop offset="1" stopColor="#ffd23f" /></radialGradient>
      <radialGradient id="gGlow" cx=".5" cy=".5" r=".5"><stop offset="0" stopColor="rgba(255,236,170,.7)" /><stop offset="1" stopColor="rgba(255,236,170,0)" /></radialGradient>
      <radialGradient id="gCap" cx=".4" cy=".25" r=".9"><stop offset="0" stopColor="#ee6a4f" /><stop offset="1" stopColor="#cc3f29" /></radialGradient>
    </defs></svg>
  ), []);

  return (
    <div className="scene" aria-hidden>
      {defs}
      <Layer depth={2} className="l-glow" sx={sx} sy={sy}><span /></Layer>
      <Layer depth={5} className="l-sun" sx={sx} sy={sy}>
        <svg viewBox="0 0 220 220">
          <g className="rays"><g fill="#ffe9a8" opacity=".5">
            <rect x="106" y="0" width="8" height="40" rx="4" /><rect x="106" y="180" width="8" height="40" rx="4" />
            <rect x="0" y="106" width="40" height="8" rx="4" /><rect x="180" y="106" width="40" height="8" rx="4" />
            <rect x="26" y="26" width="8" height="38" rx="4" transform="rotate(45 30 45)" /><rect x="186" y="26" width="8" height="38" rx="4" transform="rotate(-45 190 45)" />
            <rect x="26" y="156" width="8" height="38" rx="4" transform="rotate(-45 30 175)" /><rect x="186" y="156" width="8" height="38" rx="4" transform="rotate(45 190 175)" />
          </g></g>
          <circle cx="110" cy="110" r="62" fill="url(#gGlow)" /><circle cx="110" cy="110" r="46" fill="url(#gSun)" />
        </svg>
      </Layer>
      <Layer depth={9} className="l-clouds" sx={sx} sy={sy}><Cloud cls="c1" /><Cloud cls="c2" /><Cloud cls="c3" /></Layer>

      <Layer depth={4} className="l-far" sx={sx} sy={sy}>
        <svg viewBox="0 0 1440 360" preserveAspectRatio="xMidYMax slice" width="100%" height="100%">
          <path d="M0 200 Q360 130 720 180 T1440 160 V360 H0Z" fill="url(#gHillFar)" />
          <g fill="#9fc7d8" opacity=".55" transform="translate(1120 110)"><rect x="0" y="20" width="14" height="54" /><rect x="46" y="20" width="14" height="54" /><rect x="14" y="32" width="32" height="42" /><path d="M0 20 l7 -12 7 12Z M46 20 l7 -12 7 12Z M14 32 l16 -16 16 16Z" /></g>
        </svg>
      </Layer>
      <Layer depth={11} className="l-mid" sx={sx} sy={sy}>
        <svg viewBox="0 0 1440 420" preserveAspectRatio="xMidYMax slice" width="100%" height="100%">
          <path d="M0 250 Q300 195 640 240 T1200 230 T1440 250 V420 H0Z" fill="url(#gHillMid)" />
          <g className="tree t1" transform="translate(150 270)"><rect x="-11" y="-4" width="22" height="96" rx="9" fill="#8a5a32" /><g className="canopy"><ellipse cx="0" cy="-34" rx="64" ry="54" fill="url(#gTree)" /><ellipse cx="-38" cy="2" rx="46" ry="42" fill="#5fa047" /><ellipse cx="38" cy="-4" rx="44" ry="40" fill="#6fae4f" /></g></g>
          <g className="tree t2" transform="translate(560 290)"><rect x="-9" y="-4" width="18" height="80" rx="8" fill="#8a5a32" /><g className="canopy"><ellipse cx="0" cy="-28" rx="50" ry="44" fill="url(#gTree)" /><ellipse cx="30" cy="2" rx="38" ry="34" fill="#6fae4f" /></g></g>
          <g className="tree t3" transform="translate(1010 284)"><rect x="-12" y="-4" width="24" height="92" rx="10" fill="#7d5230" /><g className="canopy"><ellipse cx="0" cy="-36" rx="70" ry="58" fill="url(#gTree)" /><ellipse cx="-42" cy="2" rx="50" ry="44" fill="#5fa047" /><ellipse cx="44" cy="-2" rx="46" ry="42" fill="#6fae4f" /></g></g>
          <g className="tree t4" transform="translate(1330 296)"><rect x="-8" y="-4" width="16" height="70" rx="7" fill="#8a5a32" /><g className="canopy"><ellipse cx="0" cy="-24" rx="44" ry="40" fill="url(#gTree)" /></g></g>
        </svg>
      </Layer>
      <Layer depth={20} className="l-near" sx={sx} sy={sy}>
        <svg viewBox="0 0 1440 240" preserveAspectRatio="xMidYMax slice" width="100%" height="100%">
          <path d="M0 130 Q360 90 760 130 T1440 122 V240 H0Z" fill="#4e9636" />
          <g fill="#3f7d2c"><path d="M330 170 q-6 -34 0 -52 q6 18 0 52Z M318 168 q-22 -16 -28 -34 q24 8 28 34Z M342 168 q22 -16 28 -34 q-24 8 -28 34Z" /></g>
          <g className="mush m-a" transform="translate(470 168)"><ellipse cx="0" cy="44" rx="22" ry="6" fill="rgba(20,40,10,.18)" /><rect x="-10" y="12" width="20" height="34" rx="9" fill="#f5ecd6" /><path d="M-34 16 Q0 -22 34 16 Z" fill="url(#gCap)" /><g fill="#fff"><circle cx="-15" cy="4" r="5" /><circle cx="9" cy="0" r="6" /><circle cx="22" cy="11" r="4" /></g><circle cx="-8" cy="28" r="2.6" fill="#3a2c1f" /><circle cx="8" cy="28" r="2.6" fill="#3a2c1f" /><path d="M-6 34 Q0 39 6 34" stroke="#3a2c1f" strokeWidth="2" fill="none" strokeLinecap="round" /></g>
          <g className="mush m-c" transform="translate(1240 176) scale(.9)"><rect x="-9" y="12" width="18" height="30" rx="8" fill="#f5ecd6" /><path d="M-30 16 Q0 -20 30 16 Z" fill="url(#gCap)" /><g fill="#fff"><circle cx="-12" cy="3" r="5" /><circle cx="10" cy="2" r="5" /></g></g>
        </svg>
      </Layer>
      <Layer depth={3} className="l-fog" sx={sx} sy={sy}><span /></Layer>

      <div className="fireflies">{bugs.map((f, i) => (
        <span key={i} className="firefly" style={{ left: `${f.l}%`, top: `${f.t}%`, width: f.w, height: f.w, ["--d" as string]: `${f.d}s`, animationDelay: `${f.a}s, ${f.g}s` }} />
      ))}</div>
      <div className="spores">{spores.map((s, i) => (
        <span key={i} className="spore" style={{ left: `${s.l}%`, ["--d" as string]: `${s.d}s`, animationDelay: `${s.delay}s` }} />
      ))}</div>

      <Butterfly cls="b1" a="#b98bff" b="#8ad6ff" />
      <Butterfly cls="b2" a="#ffb1d8" b="#ffd27a" />

      <svg className="vine vine-l" viewBox="0 0 160 160"><g fill="#4e8c3a"><path d="M0 0 Q60 10 70 60 Q40 40 0 40Z" /><path d="M0 30 Q40 44 44 84 Q22 64 0 64Z" opacity=".9" /></g><g fill="#6fae4f"><circle cx="66" cy="58" r="9" /><circle cx="40" cy="80" r="7" /></g></svg>
      <svg className="vine vine-r" viewBox="0 0 160 160"><g fill="#4e8c3a"><path d="M160 0 Q100 10 90 60 Q120 40 160 40Z" /><path d="M160 30 Q120 44 116 84 Q138 64 160 64Z" opacity=".9" /></g><g fill="#6fae4f"><circle cx="94" cy="58" r="9" /><circle cx="120" cy="80" r="7" /></g></svg>
    </div>
  );
}
