// Observatory orchestration: poll the run list, render the selected run's live
// loss curve + samples + config/summary. The UI is a pure reader of the
// read-layer; "live" is just a run still being appended to.
import { listRuns, getMetrics, getConfig, getSummary } from './api.js';
import { makeChart } from './chart.js';
import { renderSamples } from './samples.js';

const POLL_MS = 1500;
const els = {
  runs: document.getElementById('runs'),
  title: document.getElementById('run-title'),
  chart: document.getElementById('chart'),
  samples: document.getElementById('samples'),
  config: document.getElementById('config'),
  summary: document.getElementById('summary'),
  status: document.getElementById('status'),
};

const chart = makeChart(els.chart);
let selected = null;
let runIndex = {}; // id -> RunSummary

const STATUS_DOT = { running: '🟢', done: '✓', failed: '✗', dispatched: '➦', pending: '…', unknown: '·' };

function renderRunList(runs) {
  runIndex = Object.fromEntries(runs.map((r) => [r.id, r]));
  els.runs.innerHTML = runs
    .sort((a, b) => (b.created_at || '').localeCompare(a.created_at || ''))
    .map((r) => `<li data-id="${r.id}" class="${r.id === selected ? 'sel' : ''}">
        <span>${STATUS_DOT[r.status] || '·'} ${r.id}</span>
        <span class="badge ${r.status}">${r.status}</span></li>`)
    .join('');
  els.runs.querySelectorAll('li').forEach((li) =>
    li.addEventListener('click', () => select(li.dataset.id)));
}

async function refreshSelectedMeta() {
  if (!selected) return;
  try {
    const [cfg, sum] = await Promise.all([getConfig(selected), getSummary(selected)]);
    els.config.textContent = JSON.stringify(cfg.config, null, 2);
    els.summary.textContent = JSON.stringify(sum.summary, null, 2);
  } catch (e) { /* run may be a dispatched stub */ }
}

async function refreshSelectedMetrics() {
  if (!selected) return;
  try {
    const m = await getMetrics(selected); // full fetch — runs are small
    chart.setSeries(m.series);
    const r = runIndex[selected];
    renderSamples(els.samples, selected, r ? r.artifact_names : []);
  } catch (e) { els.status.textContent = `metrics error: ${e.message}`; }
}

async function select(id) {
  if (id === selected) return;
  selected = id;
  els.title.textContent = id;
  els.runs.querySelectorAll('li').forEach((li) => li.classList.toggle('sel', li.dataset.id === id));
  await Promise.all([refreshSelectedMetrics(), refreshSelectedMeta()]);
}

async function tick() {
  try {
    const { runs } = await listRuns();
    renderRunList(runs);
    if (!selected && runs.length) await select(runs[0].id);
    if (selected) {
      const r = runIndex[selected];
      await refreshSelectedMetrics();
      if (r && (r.status === 'running' || r.status === 'dispatched')) await refreshSelectedMeta();
    }
    els.status.textContent = `${runs.length} runs · polling every ${POLL_MS / 1000}s`;
  } catch (e) {
    els.status.textContent = `error: ${e.message}`;
  }
}

tick();
setInterval(tick, POLL_MS);
