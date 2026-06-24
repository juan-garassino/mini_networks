import { hue } from "@/lib/format";

/* Fluent Emoji 3D assets (in /public/mascots). */
const M = "/mascots";

function Asset({ src, cls = "", size = 100, alt = "" }: { src: string; cls?: string; size?: number; alt?: string }) {
  // eslint-disable-next-line @next/next/no-img-element
  return <img src={src} alt={alt} width={size} height={size} className={`m-asset ${cls}`} draggable={false} />;
}

export const Dragon = () => <Asset src={`${M}/dragon.png`} cls="dragon-bob" size={116} alt="dragon companion" />;
export const Bot = () => <Asset src={`${M}/robot.png`} cls="bot-bob" size={104} alt="robot" />;
export const Star = () => <Asset src={`${M}/star.png`} cls="m-twinkle" size={32} alt="" />;
export const Sprout = () => <Asset src={`${M}/sprout.png`} cls="m-sway" size={28} alt="" />;
export const Planet = ({ size = 50 }: { size?: number }) => <Asset src={`${M}/planet.png`} cls="m-float" size={size} alt="" />;
export const Mushroom = ({ size = 48 }: { size?: number }) => <Asset src={`${M}/mushroom.png`} cls="m-bobimg" size={size} alt="" />;
export const Butterfly = ({ size = 38 }: { size?: number }) => <Asset src={`${M}/butterfly.png`} size={size} alt="" />;
export const Sparkles = ({ size = 26 }: { size?: number }) => <Asset src={`${M}/sparkles.png`} size={size} alt="" />;

// the paper plane stays a tiny inline SVG (no clean Fluent match)
export const Plane = () => (
  <svg className="m-plane-svg" viewBox="0 0 44 32" width="34">
    <path d="M2 5 L42 2 L22 30 L17 19 Z" fill="#ffd23f" stroke="#eaa800" strokeWidth="1.6" strokeLinejoin="round" />
    <path d="M17 19 L42 2 L22 14 Z" fill="#f0b400" />
  </svg>
);

const ROSTER = ["fox", "frog", "cat", "dog", "panda", "owl", "penguin", "unicorn", "lion", "tiger", "bear", "hamster", "koala", "monkey", "bee", "turtle", "whale", "octopus", "chick", "rabbit", "hatching"];

export function Avatar({ model }: { model: string }) {
  const animal = ROSTER[hue(model) % ROSTER.length];
  return <Asset src={`${M}/avatars/${animal}.png`} size={36} alt={model} cls="!drop-shadow-none" />;
}
