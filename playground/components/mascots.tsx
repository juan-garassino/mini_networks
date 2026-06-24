import { hue } from "@/lib/format";

export const Dragon = () => (
  <svg className="m-dragon" viewBox="0 0 140 124" width="108" height="96">
    <ellipse cx="70" cy="114" rx="34" ry="7" fill="rgba(20,40,10,.18)" />
    <g className="dino-bob">
      <path d="M34 90 Q10 88 16 60 Q28 84 44 84Z" fill="url(#gBody)" />
      <g className="wing"><path d="M44 58 Q12 30 4 50 Q12 54 16 60 Q9 60 7 66 Q26 60 40 74Z" fill="url(#gWing)" /></g>
      <g className="wing r"><path d="M96 58 Q128 30 136 50 Q128 54 124 60 Q131 60 133 66 Q114 60 100 74Z" fill="url(#gWing)" /></g>
      <path d="M52 46 q-2 -14 4 -18 q4 8 2 18Z M88 45 q2 -14 -4 -18 q-4 8 -2 18Z" fill="#ffe07a" />
      <ellipse cx="70" cy="74" rx="34" ry="32" fill="url(#gBody)" />
      <path d="M44 60 Q70 41 96 60" stroke="#cdf0a0" strokeWidth="4" fill="none" strokeLinecap="round" opacity=".55" />
      <ellipse cx="70" cy="84" rx="20" ry="17" fill="url(#gBelly)" />
      <path d="M60 48 l5 -10 5 10Z M74 48 l5 -10 5 10Z" fill="#4e8c3a" />
      <ellipse cx="60" cy="66" rx="5.6" ry="6.2" fill="#fff" /><ellipse cx="80" cy="66" rx="5.6" ry="6.2" fill="#fff" />
      <circle cx="61" cy="67" r="3.3" fill="#2d3561" /><circle cx="81" cy="67" r="3.3" fill="#2d3561" />
      <circle cx="62.3" cy="65.6" r="1.1" fill="#fff" /><circle cx="82.3" cy="65.6" r="1.1" fill="#fff" />
      <circle cx="52" cy="75" r="4.6" fill="#ff9aa2" opacity=".75" /><circle cx="88" cy="75" r="4.6" fill="#ff9aa2" opacity=".75" />
      <path d="M63 79 Q70 86 77 79" stroke="#2d3561" strokeWidth="2.5" fill="none" strokeLinecap="round" />
      <ellipse cx="58" cy="104" rx="9" ry="6" fill="#5da648" /><ellipse cx="82" cy="104" rx="9" ry="6" fill="#5da648" />
    </g>
  </svg>
);

export const Bot = () => (
  <svg className="m-bot" viewBox="0 0 130 140" width="96" height="104">
    <ellipse cx="65" cy="130" rx="34" ry="7" fill="rgba(20,40,10,.16)" />
    <g className="bot-body">
      <rect x="38" y="66" width="54" height="52" rx="22" fill="#fff" stroke="#7d5fff" strokeWidth="3" />
      <rect x="34" y="24" width="62" height="50" rx="24" fill="#fff" stroke="#7d5fff" strokeWidth="3" />
      <rect x="42" y="38" width="46" height="24" rx="12" fill="#3a2f6b" />
      <circle cx="56" cy="50" r="4.2" fill="#8ad6ff" /><circle cx="74" cy="50" r="4.2" fill="#8ad6ff" />
      <line x1="65" y1="24" x2="65" y2="14" stroke="#7d5fff" strokeWidth="3" /><circle cx="65" cy="12" r="3.5" fill="#7d5fff" />
      <rect x="26" y="78" width="11" height="28" rx="5.5" fill="#7d5fff" />
      <g className="bot-wave"><rect x="93" y="54" width="11" height="28" rx="5.5" fill="#7d5fff" /></g>
    </g>
  </svg>
);

export const Sprout = () => (
  <svg className="m-sprout-svg" viewBox="0 0 24 24" width="22" height="22">
    <path d="M12 22 V12" stroke="#5cb84e" strokeWidth="2.5" strokeLinecap="round" />
    <path d="M12 14 Q4 12 6 4 Q14 6 12 14Z" fill="#7ad06a" /><path d="M12 12 Q20 10 18 3 Q11 5 12 12Z" fill="#9bdc83" />
  </svg>
);

export const Star = () => (
  <svg className="m-star-svg" viewBox="0 0 40 40" width="28" height="28">
    <path d="M20 3 l5 11.5 12.5 1 -9.5 8.5 3 12.5 -11 -6.7 -11 6.7 3 -12.5 -9.5 -8.5 12.5 -1 Z" fill="#ffd23f" stroke="#eaa800" strokeWidth="2" strokeLinejoin="round" />
  </svg>
);

export const Plane = () => (
  <svg className="m-plane-svg" viewBox="0 0 44 32" width="34">
    <path d="M2 5 L42 2 L22 30 L17 19 Z" fill="#ffd23f" stroke="#eaa800" strokeWidth="1.6" strokeLinejoin="round" />
    <path d="M17 19 L42 2 L22 14 Z" fill="#f0b400" />
  </svg>
);

export function Avatar({ model }: { model: string }) {
  const h = hue(model);
  const c = `hsl(${h},62%,62%)`, d = `hsl(${h},58%,46%)`;
  return (
    <svg viewBox="0 0 36 36" width="34" height="34">
      <ellipse cx="18" cy="31" rx="10" ry="2.5" fill="rgba(45,53,97,.1)" />
      <path d="M9 13 l3 -6 3 6Z M27 13 l-3 -6 -3 6Z" fill={d} />
      <ellipse cx="18" cy="20" rx="12" ry="11" fill={c} />
      <ellipse cx="18" cy="23" rx="7" ry="5.5" fill="rgba(255,255,255,.45)" />
      <circle cx="14" cy="18" r="2" fill="#2d3561" /><circle cx="22" cy="18" r="2" fill="#2d3561" />
      <circle cx="14.7" cy="17.3" r=".7" fill="#fff" /><circle cx="22.7" cy="17.3" r=".7" fill="#fff" />
      <circle cx="11" cy="21" r="1.8" fill="#ff9aa2" opacity=".6" /><circle cx="25" cy="21" r="1.8" fill="#ff9aa2" opacity=".6" />
      <path d="M15 22 Q18 25 21 22" stroke="#2d3561" strokeWidth="1.3" fill="none" strokeLinecap="round" />
    </svg>
  );
}
