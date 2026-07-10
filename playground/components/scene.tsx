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

/* a soft organic foliage cluster (overlapping graded blobs) */
const Foliage = ({ x, y, s = 1, fill }: { x: number; y: number; s?: number; fill: string }) => (
  <g transform={`translate(${x} ${y}) scale(${s})`} fill={fill}>
    <ellipse cx="0" cy="0" rx="70" ry="52" /><ellipse cx="-52" cy="14" rx="46" ry="38" />
    <ellipse cx="50" cy="10" rx="50" ry="40" /><ellipse cx="-18" cy="-26" rx="40" ry="34" /><ellipse cx="26" cy="-22" rx="38" ry="32" />
  </g>
);

const Critter = ({ src, cls, style, w }: { src: string; cls?: string; style: React.CSSProperties; w: number }) => (
  // eslint-disable-next-line @next/next/no-img-element
  <img src={src} alt="" width={w} className={`grove-critter ${cls ?? ""}`} style={style} draggable={false} />
);

export function Scene() {
  const reduce = useReducedMotion();
  const mx = useMotionValue(0), my = useMotionValue(0);
  const sx = useSpring(mx, { stiffness: 50, damping: 18 });
  const sy = useSpring(my, { stiffness: 50, damping: 18 });
  const [bugs, setBugs] = useState<{ l: number; t: number; w: number; d: number; a: number; g: number }[]>([]);
  const [bokeh, setBokeh] = useState<{ l: number; t: number; s: number; d: number; delay: number }[]>([]);

  useEffect(() => {
    const rnd = (n: number) => Math.random() * n;
    setBugs(Array.from({ length: 20 }, () => ({ l: rnd(100), t: 42 + rnd(50), w: 4 + rnd(5), d: 7 + rnd(8), a: -rnd(15), g: -rnd(2) })));
    setBokeh(Array.from({ length: 9 }, () => ({ l: rnd(100), t: 10 + rnd(70), s: 60 + rnd(140), d: 16 + rnd(14), delay: -rnd(20) })));
    if (reduce) return;
    const onMove = (e: MouseEvent) => {
      mx.set((e.clientX / window.innerWidth - 0.5) * 20);
      my.set((e.clientY / window.innerHeight - 0.5) * 20);
    };
    window.addEventListener("mousemove", onMove);
    return () => window.removeEventListener("mousemove", onMove);
  }, [reduce, mx, my]);

  const defs = useMemo(() => (
    <svg width="0" height="0" aria-hidden><defs>
      <linearGradient id="hFar" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="#c2cfd6" /><stop offset="1" stopColor="#aebfc2" /></linearGradient>
      <linearGradient id="hMid" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="#b7d59a" /><stop offset="1" stopColor="#86b272" /></linearGradient>
      <linearGradient id="hNear" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="#93c277" /><stop offset="1" stopColor="#5d9550" /></linearGradient>
      <linearGradient id="folMid" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="#7fb163" /><stop offset="1" stopColor="#56924a" /></linearGradient>
      <linearGradient id="folNear" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="#5f9b4a" /><stop offset="1" stopColor="#3c7430" /></linearGradient>
      <radialGradient id="sunG" cx=".5" cy=".5" r=".5"><stop offset="0" stopColor="rgba(255,244,206,.95)" /><stop offset=".5" stopColor="rgba(255,233,168,.5)" /><stop offset="1" stopColor="rgba(255,233,168,0)" /></radialGradient>
    </defs></svg>
  ), []);

  return (
    <div className="scene" aria-hidden>
      {defs}
      <div className="vignette" />
      <div className="grain" />

      <Layer depth={2} className="l-sunglow" sx={sx} sy={sy}><span /></Layer>
      <div className="godrays"><span /><span /><span /></div>

      <Layer depth={3} className="l-far2" sx={sx} sy={sy}>
        <svg viewBox="0 0 1440 320" preserveAspectRatio="xMidYMax slice" width="100%" height="100%">
          <path d="M0 210 C 220 150 380 160 560 195 C 760 235 920 160 1140 190 C 1300 212 1440 185 1440 185 L1440 320 L0 320Z" fill="url(#hFar)" opacity="0.7" />
        </svg>
      </Layer>
      <div className="mist mist-1" />

      <Layer depth={8} className="l-mid2" sx={sx} sy={sy}>
        <svg viewBox="0 0 1440 360" preserveAspectRatio="xMidYMax slice" width="100%" height="100%">
          <path d="M0 230 C 200 175 360 180 540 210 C 740 244 900 175 1120 205 C 1300 230 1440 210 1440 210 L1440 360 L0 360Z" fill="url(#hMid)" />
          <path d="M0 232 C 200 177 360 182 540 212 C 740 246 900 177 1120 207 C 1300 232 1440 212 1440 212" fill="none" stroke="#e7f6c8" strokeWidth="3" strokeOpacity=".5" />
          <Foliage x={250} y={196} s={0.9} fill="url(#folMid)" />
          <Foliage x={760} y={186} s={1.1} fill="url(#folMid)" />
          <Foliage x={1180} y={200} s={0.85} fill="url(#folMid)" />
        </svg>
      </Layer>
      <div className="mist mist-2" />

      <Layer depth={16} className="l-near2" sx={sx} sy={sy}>
        <svg viewBox="0 0 1440 240" preserveAspectRatio="xMidYMax slice" width="100%" height="100%">
          <path d="M0 140 C 240 95 420 110 640 140 C 860 170 1020 100 1240 130 C 1360 146 1440 132 1440 132 L1440 240 L0 240Z" fill="url(#hNear)" />
          <path d="M0 142 C 240 97 420 112 640 142 C 860 172 1020 102 1240 132 C 1360 148 1440 134 1440 134" fill="none" stroke="#eafbcf" strokeWidth="3" strokeOpacity=".55" />
          <Foliage x={120} y={118} s={0.7} fill="url(#folNear)" />
          <Foliage x={1000} y={108} s={0.95} fill="url(#folNear)" />
        </svg>
      </Layer>

      {/* drifting bokeh light */}
      <div className="bokeh">{bokeh.map((b, i) => (
        <span key={i} style={{ left: `${b.l}%`, top: `${b.t}%`, width: b.s, height: b.s, animationDuration: `${b.d}s`, animationDelay: `${b.delay}s` }} />
      ))}</div>

      {/* fireflies */}
      <div className="fireflies">{bugs.map((f, i) => (
        <span key={i} className="firefly" style={{ left: `${f.l}%`, top: `${f.t}%`, width: f.w, height: f.w, ["--d" as string]: `${f.d}s`, animationDelay: `${f.a}s, ${f.g}s` }} />
      ))}</div>

      {/* scattered grove critters (Fluent 3D) */}
      <Critter src="/mascots/mushroom.png" w={46} style={{ left: "7%", bottom: "26px" }} cls="cr-bob" />
      <Critter src="/mascots/mushroom.png" w={34} style={{ left: "12%", bottom: "10px" }} cls="cr-bob cr-d1" />
      <Critter src="/mascots/avatars/turtle.png" w={42} style={{ left: "26%", bottom: "16px" }} cls="cr-bob cr-d2" />
      <Critter src="/mascots/avatars/frog.png" w={40} style={{ right: "30%", bottom: "20px" }} cls="cr-bob cr-d1" />
      <Critter src="/mascots/avatars/hamster.png" w={38} style={{ right: "14%", bottom: "12px" }} cls="cr-bob cr-d2" />
      <Critter src="/mascots/avatars/bee.png" w={34} style={{ left: "44%", bottom: "120px" }} cls="cr-fly" />
      <Critter src="/mascots/butterfly.png" w={36} style={{ left: "20%", top: "32%" }} cls="bfly b1" />
      <Critter src="/mascots/butterfly.png" w={30} style={{ right: "10%", top: "54%" }} cls="bfly b2" />
    </div>
  );
}
