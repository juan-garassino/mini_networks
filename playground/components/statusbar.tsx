export function StatusBar({ count, src, ok }: { count: number; src: string; ok: boolean }) {
  return (
    <footer className="relative z-20 flex items-center gap-2.5 px-6 py-2.5 text-[13px] font-extrabold text-[#2f5d2a]"
      style={{ background: "linear-gradient(180deg,#79c656,#5fae44)" }}>
      <span>{count} runs · poll 1.5s · src {src}</span>
      <span className="flex-1" />
      <span className="flex items-center gap-2 text-[#1f4a1a]">
        v0.5 · the grove
        <span className={`h-2.5 w-2.5 rounded-full ${ok ? "bg-white" : "bg-[#ff6b6b]"}`} />
      </span>
    </footer>
  );
}
