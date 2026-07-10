"use client";

import { useEffect, useState } from "react";

export type Hero = "dragon" | "robot";
export type Emotion = "idle" | "happy" | "excited" | "think" | "sleep" | "wave" | "idea" | "celebrate" | "alert";

// frame files saved per hero (see scripts that slice the sprite sheet)
const FRAMES: Record<Hero, string[]> = {
  dragon: ["idle", "happy", "excited", "wave", "think", "sleep"],
  robot: ["idle", "wave", "idea", "celebrate", "think", "alert"],
};

// which frames loop (low-fps) for each emotion
const SEQ: Record<Hero, Partial<Record<Emotion, string[]>>> = {
  dragon: {
    idle: ["idle", "happy"], happy: ["happy", "idle"], excited: ["excited", "happy"],
    celebrate: ["excited", "happy"], think: ["think", "idle"], sleep: ["sleep"], alert: ["think", "idle"],
  },
  robot: {
    idle: ["idle", "wave"], celebrate: ["celebrate", "idea"], excited: ["idea", "wave"],
    think: ["think", "idle"], alert: ["alert", "think"], sleep: ["idle"],
  },
};

export function HeroMascot({ hero, emotion = "idle", size = 132, fps = 2.5, className = "" }: {
  hero: Hero; emotion?: Emotion; size?: number; fps?: number; className?: string;
}) {
  const seq = SEQ[hero][emotion] ?? SEQ[hero].idle!;
  const [i, setI] = useState(0);

  useEffect(() => {
    setI(0);
    if (seq.length < 2) return;
    const id = setInterval(() => setI((x) => x + 1), 1000 / fps);
    return () => clearInterval(id);
  }, [emotion, hero, fps, seq.length]);

  const active = seq[i % seq.length];

  return (
    <div className={`hero-mascot ${className}`} style={{ width: size, height: size }}>
      {/* all frames preloaded + stacked; active shown (instant swap = choppy low-fps charm) */}
      {FRAMES[hero].map((name) => (
        // eslint-disable-next-line @next/next/no-img-element
        <img key={name} src={`/mascots/hero/${hero}/${name}.png`} alt="" width={size} height={size}
          draggable={false} className="hero-frame" style={{ opacity: name === active ? 1 : 0 }} />
      ))}
    </div>
  );
}

// map app/run state → an emotion
export function runEmotion(hasRuns: boolean, status: string | null, justFinished: boolean): Emotion {
  if (justFinished) return "celebrate";
  if (!hasRuns) return "sleep";
  if (status === "running" || status === "dispatched") return "excited";
  if (status === "failed") return "alert";
  return "idle";
}
