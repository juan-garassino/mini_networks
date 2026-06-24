// Loss-curve chart. Uses uPlot when the CDN loaded, else a hand-rolled SVG with
// the same setSeries([{key, points:[[step,value],...]}]) interface so app.js is
// agnostic (works offline / on Colab if the CDN is blocked).

const COLORS = ['#22c55e', '#3b82f6', '#f59e0b', '#ef4444', '#a855f7', '#14b8a6'];
const SKIP = new Set(['epoch']); // monotonic counter, not worth plotting

function plottable(series) {
  return series.filter((s) => !SKIP.has(s.key) && s.points.length);
}

// Align disparate per-key steps onto one shared x axis (nulls = gaps).
function aligned(series) {
  const xs = [...new Set(series.flatMap((s) => s.points.map((p) => p[0])))].sort((a, b) => a - b);
  const idx = new Map(xs.map((x, i) => [x, i]));
  const cols = series.map((s) => {
    const col = new Array(xs.length).fill(null);
    for (const [x, v] of s.points) col[idx.get(x)] = v;
    return col;
  });
  return { xs, cols, keys: series.map((s) => s.key) };
}

class UplotChart {
  constructor(el) { this.el = el; this.u = null; }
  setSeries(series) {
    const s = plottable(series);
    if (this.u) { this.u.destroy(); this.u = null; }
    this.el.innerHTML = '';
    if (!s.length) return;
    const { xs, cols, keys } = aligned(s);
    const opts = {
      width: this.el.clientWidth || 600,
      height: 240,
      scales: { x: { time: false } },
      series: [
        { label: 'step' },
        ...keys.map((k, i) => ({ label: k, stroke: COLORS[i % COLORS.length], width: 2, spanGaps: true })),
      ],
      axes: [{ stroke: '#888' }, { stroke: '#888' }],
    };
    this.u = new window.uPlot(opts, [xs, ...cols], this.el);
  }
}

class SvgChart {
  constructor(el) { this.el = el; }
  setSeries(series) {
    const s = plottable(series);
    if (!s.length) { this.el.innerHTML = '<div class="empty">no metrics yet</div>'; return; }
    const all = s.flatMap((x) => x.points);
    const xs = all.map((p) => p[0]), ys = all.map((p) => p[1]);
    const [x0, x1] = [Math.min(...xs), Math.max(...xs)];
    const [y0, y1] = [Math.min(...ys), Math.max(...ys)];
    const W = 600, H = 240, pad = 24;
    const sx = (x) => pad + ((x - x0) / (x1 - x0 || 1)) * (W - 2 * pad);
    const sy = (y) => H - pad - ((y - y0) / (y1 - y0 || 1)) * (H - 2 * pad);
    const lines = s.map((ser, i) => {
      const d = ser.points.map((p, j) => `${j ? 'L' : 'M'}${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join(' ');
      return `<path d="${d}" fill="none" stroke="${COLORS[i % COLORS.length]}" stroke-width="2"/>`;
    }).join('');
    const legend = s.map((ser, i) =>
      `<tspan fill="${COLORS[i % COLORS.length]}">${ser.key}</tspan>`).join('  ');
    this.el.innerHTML =
      `<svg viewBox="0 0 ${W} ${H}" width="100%" height="240">${lines}` +
      `<text x="${pad}" y="14" font-size="11">${legend}</text></svg>`;
  }
}

export function makeChart(el) {
  return window.uPlot ? new UplotChart(el) : new SvgChart(el);
}
