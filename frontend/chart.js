// Loss chart — smooth spline curves, soft palette, rounded dots. uPlot when the
// CDN loaded, else an SVG fallback with the same setSeries() interface (returns
// the plottable [{key,color,points}] so the stat cards stay colour-matched).
import { SKIP_KEYS } from './format.js';

const PALETTE = ['#ff9f43', '#7d5fff', '#2ecc71', '#4dabf7', '#ff6b6b', '#ffd23f'];
const GRID = '#eef0fa';
const AXIS = '#9aa0c4';
const AXIS_FONT = '12px Nunito, sans-serif';

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
    const spline = window.uPlot.paths && window.uPlot.paths.spline ? window.uPlot.paths.spline() : undefined;
    const opts = {
      width: this.el.clientWidth || 600,
      height: this.el.clientHeight || 240,
      legend: { show: false },
      cursor: { points: { size: 7 }, focus: { prox: 18 }, drag: { x: true, y: false } },
      scales: { x: { time: false } },
      axes: [
        { stroke: AXIS, grid: { stroke: GRID, width: 1.5 }, ticks: { stroke: GRID, size: 4 }, font: AXIS_FONT, gap: 8 },
        { stroke: AXIS, grid: { stroke: GRID, width: 1.5 }, ticks: { stroke: GRID, size: 4 }, font: AXIS_FONT, size: 46 },
      ],
      series: [{}, ...list.map((s, i) => ({
        stroke: s.color, width: 3, paths: spline,
        dash: i === 1 ? [9, 6] : undefined,
        points: { show: true, size: 7, fill: '#fff', stroke: s.color, width: 2.5 },
      }))],
    };
    this.u = new window.uPlot(opts, [xs, ...cols], this.el);
    this.ro = new ResizeObserver(() => { if (this.u) this.u.setSize({ width: this.el.clientWidth, height: this.el.clientHeight }); });
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
    const W = 600, H = 240, pad = 30;
    const sx = (x) => pad + ((x - x0) / ((x1 - x0) || 1)) * (W - 2 * pad);
    const sy = (y) => H - pad - ((y - y0) / ((y1 - y0) || 1)) * (H - 2 * pad);
    const paths = list.map((s, i) => {
      const d = s.points.map((p, j) => `${j ? 'L' : 'M'}${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join(' ');
      const dots = s.points.map((p) => `<circle cx="${sx(p[0]).toFixed(1)}" cy="${sy(p[1]).toFixed(1)}" r="3.5" fill="#fff" stroke="${s.color}" stroke-width="2.5"/>`).join('');
      return `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" ${i === 1 ? 'stroke-dasharray="9 6"' : ''}/>${dots}`;
    }).join('');
    this.el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="100%" height="100%" preserveAspectRatio="none">${paths}</svg>`;
    return list;
  }
}

export function makeChart(el) {
  return window.uPlot ? new UplotChart(el) : new SvgChart(el);
}
