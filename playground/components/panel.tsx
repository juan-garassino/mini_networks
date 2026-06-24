import { motion } from "motion/react";

export const CARD =
  "flex min-h-0 flex-col rounded-[24px] border border-white/80 bg-gradient-to-b from-[#fffef9] to-[#fdf5e6] shadow-[0_18px_40px_rgba(74,60,40,.16)] overflow-hidden";

export function Panel({ i = 0, className = "", children }: { i?: number; className?: string; children: React.ReactNode }) {
  return (
    <motion.div
      className={`${CARD} ${className}`}
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, delay: i * 0.07, ease: [0.2, 0.75, 0.3, 1] }}
    >
      {children}
    </motion.div>
  );
}

export function PanelHead({ title, icon, right }: { title: React.ReactNode; icon?: React.ReactNode; right?: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between px-5 pt-4 pb-3">
      <div className="flex items-center gap-2 font-display text-[19px] font-bold text-[#3a3152]">
        {icon}{title}
      </div>
      {right}
    </div>
  );
}

export function Chip({ tone = "gray", children }: { tone?: "green" | "violet" | "blue" | "amber" | "red" | "gray"; children: React.ReactNode }) {
  const tones: Record<string, string> = {
    green: "bg-[#e3f9ec] text-[#23b26d]",
    violet: "bg-[#efeaff] text-[#7d5fff]",
    blue: "bg-[#e6f1ff] text-[#3b82f6]",
    amber: "bg-[#fff1c2] text-[#b8860b]",
    red: "bg-[#ffe6e6] text-[#e0533f]",
    gray: "bg-[#f1eee4] text-[#97907e]",
  };
  return <span className={`rounded-full px-2.5 py-1 text-xs font-extrabold ${tones[tone]}`}>{children}</span>;
}
