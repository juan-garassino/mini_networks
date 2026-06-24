// Lab (Launch) view: pick a model + tier + knobs → POST /train, show recent runs.
import { listModels, startTrain, listRuns } from './api.js';
import { fmtNum, statusMeta, primaryMetric } from './format.js';

let inited = false;
let tier = 'S';

export async function initLab() {
  if (inited) return;
  inited = true;
  const $ = (id) => document.getElementById(id);
  const modelSel = $('lab-model'), result = $('lab-result'), form = $('lab-form'), seg = $('lab-tier'), recent = $('lab-runs');

  try {
    const models = await listModels();
    modelSel.innerHTML = models.map((m) => `<option value="${m.name}">${m.name}</option>`).join('');
  } catch (e) {
    modelSel.innerHTML = '<option>—</option>';
  }

  seg.addEventListener('click', (e) => {
    const b = e.target.closest('button');
    if (!b) return;
    tier = b.dataset.tier;
    seg.querySelectorAll('button').forEach((x) => x.classList.toggle('is-on', x === b));
  });

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const model = modelSel.value;
    const body = {
      epochs: parseInt($('lab-epochs').value || '3', 10),
      batch_size: parseInt($('lab-batch').value || '32', 10),
      fast_demo: tier === 'S',
      training_tier: tier,
      device: 'cpu',
    };
    result.innerHTML = `<span>LAUNCHING ${model}…</span>`;
    try {
      const r = await startTrain(model, body);
      result.innerHTML = `<span class="ok">&#9654; ${r.status.toUpperCase()} · ${r.job_id}</span><br>open WATCH to see it train`;
      refreshRecent();
    } catch (err) {
      result.innerHTML = `<span class="err">&#10006; ${err.message}</span>`;
    }
  });

  async function refreshRecent() {
    try {
      const runs = (await listRuns()).runs
        .slice()
        .sort((a, b) => (b.created_at || '').localeCompare(a.created_at || ''))
        .slice(0, 14);
      recent.innerHTML = runs
        .map((r) => {
          const [state] = statusMeta(r.status);
          const pm = primaryMetric(r.last_metrics);
          return `<li class="run"><span class="led" data-state="${state}"></span>
            <span><span class="run-model">${r.model}</span><br><span class="run-name">${r.run_name || r.id}</span></span>
            <span class="run-val">${pm ? `<span class="v">${fmtNum(pm.v)}</span><span class="k">${pm.k}</span>` : ''}</span></li>`;
        })
        .join('');
    } catch (e) { /* transient */ }
  }

  refreshRecent();
  setInterval(() => {
    if (document.getElementById('view-lab').classList.contains('is-active')) refreshRecent();
  }, 2500);
}
