// Playground shell: 4-view router + the Observatory (Watch) view logic.
// Sandbox / Lab / Lessons are styled placeholders this pass; wired next.
import { listRuns, getMetrics, getConfig, getSummary } from './api.js';
import { makeChart } from './chart.js';
import { renderSamples } from './samples.js';
import { fmtNum, statusMeta, primaryMetric, relTime } from './format.js';
import { initLab } from './lab.js';
import { initLessons } from './lessons.js';
import { initSandbox } from './sandbox.js';

const $ = (id) => document.getElementById(id);

/* ----------------------------- view router ----------------------------- */
let activeView = 'observatory';
function showView(name) {
  activeView = name;
  document.querySelectorAll('.view-tab').forEach((t) => t.classList.toggle('is-active', t.dataset.view === name));
  document.querySelectorAll('.view').forEach((v) => v.classList.toggle('is-active', v.id === `view-${name}`));
  if (name === 'observatory') refreshDetail(); // redraw chart at correct size
  if (name === 'lab') initLab();
  if (name === 'lessons') initLessons();
  if (name === 'sandbox') initSandbox();
}
$('viewnav').addEventListener('click', (e) => {
  const t = e.target.closest('.view-tab');
  if (t && !t.disabled) showView(t.dataset.view);
});

/* ----------------------------- telemetry ------------------------------- */
const clock = $('tlm-clock');
setInterval(() => { clock.textContent = new Date().toISOString().slice(11, 19); }, 1000);
clock.textContent = new Date().toISOString().slice(11, 19);
function setConn(ok) { $('conn').querySelector('.led').dataset.state = ok ? 'ok' : 'failed'; }

/* ----------------------------- observatory ----------------------------- */
const els = {
  runs: $('runs'), runCount: $('run-count'), filters: $('filters'),
  title: $('run-title'), chips: $('run-chips'), rec: $('rec'),
  chart: $('chart'), seriesReadout: $('series-readout'),
  samples: $('samples'), specimenMeta: $('specimen-meta'),
  config: $('config'), summary: $('summary'), status: $('status'), source: $('tlm-source'),
};
const chart = makeChart(els.chart);
let runs = [], runIndex = {}, selected = null, selStatus = null, filter = 'all';

els.filters.addEventListener('click', (e) => {
  const b = e.target.closest('.chip');
  if (!b) return;
  filter = b.dataset.filter;
  els.filters.querySelectorAll('.chip').forEach((c) => c.classList.toggle('is-active', c === b));
  renderRegistry();
});

function visible() {
  if (filter === 'all') return runs;
  if (filter === 'running') return runs.filter((r) => r.status === 'running' || r.status === 'dispatched');
  return runs.filter((r) => r.status === filter);
}

function hue(str) { let h = 7; for (const c of str) h = (h * 31 + c.charCodeAt(0)) % 360; return h; }
function avatar(model) {
  const h = hue(model), c = `hsl(${h},62%,62%)`, d = `hsl(${h},58%,46%)`;
  return `<svg class="avatar" viewBox="0 0 36 36">
    <ellipse cx="18" cy="31" rx="10" ry="2.5" fill="rgba(45,53,97,.1)"/>
    <path d="M9 13 l3 -6 3 6Z M27 13 l-3 -6 -3 6Z" fill="${d}"/>
    <ellipse cx="18" cy="20" rx="12" ry="11" fill="${c}"/>
    <ellipse cx="18" cy="23" rx="7" ry="5.5" fill="rgba(255,255,255,.45)"/>
    <circle cx="14" cy="18" r="2" fill="#2d3561"/><circle cx="22" cy="18" r="2" fill="#2d3561"/>
    <circle cx="14.7" cy="17.3" r=".7" fill="#fff"/><circle cx="22.7" cy="17.3" r=".7" fill="#fff"/>
    <circle cx="11" cy="21" r="1.8" fill="#ff9aa2" opacity=".6"/><circle cx="25" cy="21" r="1.8" fill="#ff9aa2" opacity=".6"/>
    <path d="M15 22 Q18 25 21 22" stroke="#2d3561" stroke-width="1.3" fill="none" stroke-linecap="round"/>
  </svg>`;
}

function renderRegistry() {
  els.runCount.textContent = runs.length;
  const list = visible().slice().sort((a, b) => (b.created_at || '').localeCompare(a.created_at || ''));
  if (!list.length) { els.runs.innerHTML = '<li class="hint">No runs match</li>'; return; }
  els.runs.innerHTML = list.map((r) => {
    const pm = primaryMetric(r.last_metrics);
    return `<li class="run ${r.id === selected ? 'is-sel' : ''}" data-id="${r.id}">
      ${avatar(r.model)}
      <span class="run-id"><span class="run-model">${r.model}</span><span class="run-name">${r.run_name || r.id}</span></span>
      <span class="run-val">${pm ? `<span class="v">${fmtNum(pm.v)}</span><span class="k">${pm.k}</span>` : ''}</span>
    </li>`;
  }).join('');
  els.runs.querySelectorAll('.run').forEach((li) => li.addEventListener('click', () => select(li.dataset.id)));
}

function renderChips(run, config) {
  const tags = [`<span class="chip c-green">${run.source}</span>`];
  if (config && config.training_tier) tags.push(`<span class="chip c-violet">tier ${config.training_tier}</span>`);
  if (config && config.device) tags.push(`<span class="chip c-blue">${config.device}</span>`);
  tags.push(`<span class="chip c-gray">${statusMeta(run.status)[1]}</span>`);
  if (run.created_at) tags.push(`<span class="chip c-gray">${relTime(run.created_at)}</span>`);
  els.chips.innerHTML = tags.join('');
}

function renderSeriesReadout(list) {
  els.seriesReadout.innerHTML = list.map((s) => {
    const last = s.points[s.points.length - 1];
    return `<div class="stat"><span class="sdot" style="background:${s.color}"></span>
      <span class="sk">${s.key}</span><span class="sv" style="color:${s.color}">${fmtNum(last ? last[1] : NaN)}</span></div>`;
  }).join('');
}

function fmtVal(v) {
  if (typeof v === 'number') return fmtNum(v);
  if (typeof v === 'boolean') return v ? 'YES' : 'NO';
  const s = String(v);
  return s.length > 22 ? s.slice(0, 21) + '…' : s;
}

function renderConfig(config) {
  const entries = Object.entries(config || {});
  els.config.innerHTML = entries.length
    ? entries.map(([k, v]) => `<div class="kv-row"><dt>${k}</dt><span class="dots"></span><dd>${fmtVal(v)}</dd></div>`).join('')
    : '<div class="empty" style="height:50px"><p>NO CONFIG</p></div>';
}

function renderSummary(summary) {
  const entries = Object.entries(summary || {});
  els.summary.innerHTML = entries.length
    ? entries.map(([k, v]) => {
        let gv = fmtVal(v);
        if (k === 'status') {
          const cls = v === 'completed' ? 'badge-ok' : v === 'failed' ? 'badge-fail' : '';
          gv = `<span class="${cls}">${v}${v === 'completed' ? ' ✓' : ''}</span>`;
        }
        return `<div class="gate-row"><span class="gk">${k}</span><span class="gv">${gv}</span></div>`;
      }).join('')
    : '<div class="hint">awaiting completion</div>';
}

async function select(id) {
  selected = id;
  const run = runIndex[id];
  selStatus = run ? run.status : null;
  els.runs.querySelectorAll('.run').forEach((li) => li.classList.toggle('is-sel', li.dataset.id === id));
  els.title.textContent = run ? run.model.toUpperCase() : id;
  els.rec.hidden = !(run && (run.status === 'running' || run.status === 'dispatched'));
  await refreshDetail();
}

async function refreshDetail() {
  if (!selected || activeView !== 'observatory') return;
  const run = runIndex[selected];
  try {
    const [m, cfg, sum] = await Promise.all([getMetrics(selected), getConfig(selected), getSummary(selected)]);
    renderSeriesReadout(chart.setSeries(m.series));
    if (run) renderChips(run, cfg.config);
    renderConfig(cfg.config);
    renderSummary(sum.summary);
    els.specimenMeta.textContent = run ? run.artifact_names.length : 0;
    renderSamples(els.samples, selected, run ? run.artifact_names : []);
  } catch (e) { renderSamples(els.samples, selected, []); }
}

async function refreshMetricsOnly() {
  if (!selected || activeView !== 'observatory') return;
  try { renderSeriesReadout(chart.setSeries((await getMetrics(selected)).series)); } catch (e) { /* transient */ }
}

function showEmpty() {
  els.title.textContent = 'No runs yet';
  els.rec.hidden = true;
  els.chart.innerHTML = '<div class="hint">No runs yet — launch one in <b>Lab</b> (or <code>python main.py train --model vae</code>) and it appears here, live.</div>';
  els.seriesReadout.innerHTML = els.chips.innerHTML = els.config.innerHTML = els.summary.innerHTML = els.samples.innerHTML = '';
}

async function tick() {
  try {
    runs = (await listRuns()).runs;
    runIndex = Object.fromEntries(runs.map((r) => [r.id, r]));
    setConn(true);
    els.source.textContent = (runs[0] ? runs[0].source : 'local').toUpperCase();
    renderRegistry();
    if (!runs.length) { selected = null; showEmpty(); }
    else if (!selected || !runIndex[selected]) await select((visible()[0] || runs[0]).id);
    else {
      const run = runIndex[selected];
      if (run.status !== selStatus) { selStatus = run.status; await refreshDetail(); }
      else if (run.status === 'running' || run.status === 'dispatched') await refreshMetricsOnly();
    }
    const src = (runs[0] ? runs[0].source : 'local').toUpperCase();
    els.status.textContent = `${runs.length} RUNS · POLL 1.5s · SRC ${src}`;
  } catch (e) {
    setConn(false);
    els.status.textContent = `TELEMETRY FAULT — ${e.message}`;
  }
}

tick();
setInterval(tick, 1500);
