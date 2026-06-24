// Sparkle burst — appends short-lived sparkle nodes to the #sparkles overlay.
export function sparkleBurst(x: number, y: number, n = 12) {
  const host = document.getElementById("sparkles");
  if (!host) return;
  for (let i = 0; i < n; i++) {
    const s = document.createElement("span");
    s.className = "sparkle";
    const ang = (Math.PI * 2 * i) / n + Math.random();
    const dist = 28 + Math.random() * 56;
    s.style.left = `${x}px`;
    s.style.top = `${y}px`;
    s.style.setProperty("--dx", `${Math.cos(ang) * dist}px`);
    s.style.setProperty("--dy", `${Math.sin(ang) * dist}px`);
    host.appendChild(s);
    setTimeout(() => s.remove(), 820);
  }
}

export function sparkleAt(el: Element | null, n = 12) {
  if (!el) return;
  const r = el.getBoundingClientRect();
  sparkleBurst(r.left + r.width / 2, r.top + r.height / 2, n);
}
