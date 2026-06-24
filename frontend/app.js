// Observatory orchestration. Pure reader of /web: poll the registry, render the
// selected run's live scope + specimens + config/summary. "Live" is just a run
// still being appended to.
import { listRuns, getMetrics, getConfig, getSummary } from './api.js';
import { makeChart } from './chart.js';
import { renderSamples } from './samples.js';
import { fmtNum, statusMeta, primaryMetric, relTime } from './format.js';

const POLL_MS = 1500;
const $ = (id) => document.getElementById(id);
const els = {
  runs: $('runs'), runCount: $('run-count'), filters: $('filters'),
  title: $('run-title'), chips: $('run-chips'), rec: $('rec'),
  chart: $('chart'), seriesReadout: $('series-readout'),
  samples: $('samples'), specimenMeta: $('specimen-meta'),
  config: $('config'), summary: $('summary'),
  status: $('status'), source: $('tlm-source'), clock: $('tlm-clock'),
  conn: $('conn'), connLabel: $('conn-label'),
};

const chart = makeChart(els.chart);
let runs = [], runIndex = {}, selected = null, selStatus = null, filter = 'all';

/* ---------- masthead telemetry ---------- */
function tickClock() { els.clock.textContent = new Date().toISOString().slice(11, 19); }
tickClock();
setInterval(tickClock, 1000);

function setConn(ok) {
  els.conn.querySelector('.led').dataset.state = ok ? 'ok' : 'failed';
  els.connLabel.textContent = ok ? 'link' : 'fault';
}

/* ---------- registry ---------- */
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

function renderRegistry() {
  els.runCount.textContent = runs.length;
  const list = visible().slice().sort((a, b) => (b.created_at || '').localeCompare(a.created_at || ''));
  if (!list.length) {
    els.runs.innerHTML = '<li class="empty" style="height:120px"><p>no runs match this filter</p></li>';
    return;
  }
  els.runs.innerHTML = list
    .map((r) => {
      const [state] = statusMeta(r.status);
      const pm = primaryMetric(r.last_metrics);
      return `<li class="run ${r.id === selected ? 'is-sel' : ''}" data-id="${r.id}">
        <span class="led" data-state="${state}"></span>
        <span class="run-id"><span class="run-model">${r.model}</span><span class="run-name">${r.run_name || r.id}</span></span>
        <span class="run-val">${pm ? `<span class="v">${fmtNum(pm.v)}</span><span class="k">${pm.k}</span>` : ''}</span>
      </li>`;
    })
    .join('');
  els.runs.querySelectorAll('.run').forEach((li) => li.addEventListener('click', () => select(li.dataset.id)));
}

/* ---------- detail panels ---------- */
function renderChips(run, config) {
  const tags = [`<span class="tag accent">${run.source}</span>`];
  if (config && config.training_tier) tags.push(`<span class="tag">tier ${config.training_tier}</span>`);
  if (config && config.device) tags.push(`<span class="tag">${config.device}</span>`);
  tags.push(`<span class="tag">${statusMeta(run.status)[1]}</span>`);
  if (run.created_at) tags.push(`<span class="tag">${relTime(run.created_at)}</span>`);
  els.chips.innerHTML = tags.join('');
}

function renderSeriesReadout(list) {
  els.seriesReadout.innerHTML = list
    .map((s) => {
      const last = s.points[s.points.length - 1];
      return `<span class="sr-item"><span class="sr-dot" style="background:${s.color}"></span>
        <span class="sr-k">${s.key}</span><span class="sr-v">${fmtNum(last ? last[1] : NaN)}</span></span>`;
    })
    .join('');
}

function fmtVal(v) {
  if (typeof v === 'number') return fmtNum(v);
  if (typeof v === 'boolean') return v ? 'true' : 'false';
  const s = String(v);
  return s.length > 22 ? s.slice(0, 21) + '…' : s;
}

function renderConfig(config) {
  const entries = Object.entries(config || {});
  els.config.innerHTML = entries.length
    ? entries.map(([k, v]) => `<div class="kv-row"><dt>${k}</dt><span class="dots"></span><dd>${fmtVal(v)}</dd></div>`).join('')
    : '<div class="empty" style="height:60px"><p>no config</p></div>';
}

function renderSummary(summary) {
  const entries = Object.entries(summary || {});
  els.summary.innerHTML = entries.length
    ? entries
        .map(([k, v]) => {
          let led = '';
          if (k === 'status') {
            const st = v === 'completed' ? 'done' : v === 'failed' ? 'failed' : 'unknown';
            led = `<span class="led" data-state="${st}"></span>`;
          }
          return `<div class="gate-row">${led}<span class="gk">${k}</span><span class="gv">${fmtVal(v)}</span></div>`;
        })
        .join('')
    : '<div class="empty" style="height:50px"><p>awaiting completion</p></div>';
}

/* ---------- selection + refresh ---------- */
async function select(id) {
  selected = id;
  const run = runIndex[id];
  selStatus = run ? run.status : null;
  els.runs.querySelectorAll('.run').forEach((li) => li.classList.toggle('is-sel', li.dataset.id === id));
  els.title.textContent = run ? run.model : id;
  els.rec.hidden = !(run && (run.status === 'running' || run.status === 'dispatched'));
  await refreshDetail();
}

async function refreshDetail() {
  if (!selected) return;
  const run = runIndex[selected];
  try {
    const [m, cfg, sum] = await Promise.all([getMetrics(selected), getConfig(selected), getSummary(selected)]);
    renderSeriesReadout(chart.setSeries(m.series));
    if (run) renderChips(run, cfg.config);
    renderConfig(cfg.config);
    renderSummary(sum.summary);
    els.specimenMeta.textContent = run ? `${run.artifact_names.length} files` : '';
    renderSamples(els.samples, selected, run ? run.artifact_names : []);
  } catch (e) {
    renderSamples(els.samples, selected, []);
  }
}

async function refreshMetricsOnly() {
  if (!selected) return;
  try {
    renderSeriesReadout(chart.setSeries((await getMetrics(selected)).series));
  } catch (e) { /* transient */ }
}

function showEmpty() {
  els.title.textContent = '— no runs —';
  els.rec.hidden = true;
  els.chart.innerHTML =
    '<div class="empty"><svg viewBox="0 0 120 40"><path class="flat" d="M2,20 H40 l6,-13 l8,26 l6,-13 H118"/></svg>' +
    '<p>No runs yet. Launch one — <code>python main.py train --model vae</code> — and it appears here, live.</p></div>';
  els.seriesReadout.innerHTML = els.chips.innerHTML = els.config.innerHTML = els.summary.innerHTML = els.samples.innerHTML = '';
}

/* ---------- poll loop ---------- */
async function tick() {
  try {
    runs = (await listRuns()).runs;
    runIndex = Object.fromEntries(runs.map((r) => [r.id, r]));
    setConn(true);
    els.source.textContent = runs[0] ? runs[0].source : 'local';
    renderRegistry();

    if (!runs.length) {
      selected = null;
      showEmpty();
    } else if (!selected || !runIndex[selected]) {
      await select((visible()[0] || runs[0]).id);
    } else {
      const run = runIndex[selected];
      if (run.status !== selStatus) { selStatus = run.status; await refreshDetail(); }
      else if (run.status === 'running' || run.status === 'dispatched') { await refreshMetricsOnly(); }
    }

    const src = runs[0] ? runs[0].source : 'local';
    els.status.textContent = `tailing ${runs.length} run${runs.length === 1 ? '' : 's'} · poll ${POLL_MS / 1000}s · source ${src}`;
  } catch (e) {
    setConn(false);
    els.status.textContent = `telemetry fault — ${e.message}`;
  }
}

tick();
setInterval(tick, POLL_MS);
