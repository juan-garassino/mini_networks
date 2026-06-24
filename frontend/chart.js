// Loss "scope". Themed uPlot when the CDN loaded, else a glowing SVG fallback —
// same setSeries() interface, both return the plottable [{key,color,points}] so
// the readout strip stays colour-matched.
import { SKIP_KEYS } from './format.js';

// NES palette — coin gold, pipe green, mario red, sky, plum, white.
const PALETTE = ['#fbd000', '#3ec33e', '#e52521', '#5c94fc', '#b04bd6', '#ffffff'];
const GRID = 'rgba(94,148,252,0.14)';
const AXIS = '#9b95c9';
const AXIS_FONT = '13px "VT323", monospace';

export function plottableSeries(series) {
  return series
    .filter((s) => !SKIP_KEYS.has(s.key) && s.points.length)
    .map((s, i) => ({ key: s.key, color: PALETTE[i % PALETTE.length], points: s.points }));
}

function aligned(list) {
  const xs = [...new Set(list.flatMap((s) => s.points.map((p) => p[0])))].sort((a, b) => a - b);
  const idx = new Map(xs.map((x, i) => [x, i]));
  const cols = list.map((s) => {
    const c = new Array(xs.length).fill(null);
    for (const [x, y] of s.points) c[idx.get(x)] = y;
    return c;
  });
  return { xs, cols };
}

class UplotChart {
  constructor(el) { this.el = el; this.u = null; this.ro = null; }
  setSeries(series) {
    const list = plottableSeries(series);
    if (this.u) { this.u.destroy(); this.u = null; }
    if (this.ro) { this.ro.disconnect(); this.ro = null; }
    this.el.innerHTML = '';
    if (!list.length) return list;
    const { xs, cols } = aligned(list);
    const opts = {
      width: this.el.clientWidth || 600,
      height: this.el.clientHeight || 240,
      legend: { show: false },
      cursor: { points: { size: 5 }, focus: { prox: 18 }, drag: { x: true, y: false } },
      scales: { x: { time: false } },
      axes: [
        { stroke: AXIS, grid: { stroke: GRID, width: 1 }, ticks: { stroke: GRID, size: 4 }, font: AXIS_FONT, gap: 6 },
        { stroke: AXIS, grid: { stroke: GRID, width: 1 }, ticks: { stroke: GRID, size: 4 }, font: AXIS_FONT, size: 48 },
      ],
      series: [{}, ...list.map((s) => ({
        stroke: s.color, width: 3, points: { show: true, size: 6, fill: s.color },
        paths: window.uPlot.paths && window.uPlot.paths.stepped ? window.uPlot.paths.stepped({ align: 1 }) : undefined,
      }))],
    };
    this.u = new window.uPlot(opts, [xs, ...cols], this.el);
    this.ro = new ResizeObserver(() => {
      if (this.u) this.u.setSize({ width: this.el.clientWidth, height: this.el.clientHeight });
    });
    this.ro.observe(this.el);
    return list;
  }
}

class SvgChart {
  constructor(el) { this.el = el; }
  setSeries(series) {
    const list = plottableSeries(series);
    if (!list.length) { this.el.innerHTML = ''; return list; }
    const all = list.flatMap((s) => s.points);
    const xs = all.map((p) => p[0]), ys = all.map((p) => p[1]);
    const [x0, x1] = [Math.min(...xs), Math.max(...xs)];
    const [y0, y1] = [Math.min(...ys), Math.max(...ys)];
    const W = 600, H = 240, pad = 28;
    const sx = (x) => pad + ((x - x0) / ((x1 - x0) || 1)) * (W - 2 * pad);
    const sy = (y) => H - pad - ((y - y0) / ((y1 - y0) || 1)) * (H - 2 * pad);
    const paths = list.map((s) => {
      let d = '';
      s.points.forEach((p, j) => {
        const x = sx(p[0]).toFixed(0), y = sy(p[1]).toFixed(0);
        if (j === 0) d += `M${x},${y}`;
        else d += ` L${x},${sy(s.points[j - 1][1]).toFixed(0)} L${x},${y}`; // stepped
      });
      const dots = s.points.map((p) => `<rect x="${(sx(p[0]) - 2).toFixed(0)}" y="${(sy(p[1]) - 2).toFixed(0)}" width="4" height="4" fill="${s.color}"/>`).join('');
      return `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="3"/>${dots}`;
    }).join('');
    this.el.innerHTML =
      `<svg viewBox="0 0 ${W} ${H}" width="100%" height="100%" preserveAspectRatio="none" shape-rendering="crispEdges">${paths}</svg>`;
    return list;
  }
}

export function makeChart(el) {
  if (window.uPlot) return new UplotChart(el);
  el.classList.add('is-svg'); // fallback has no axes — give it faint graph paper
  return new SvgChart(el);
}
