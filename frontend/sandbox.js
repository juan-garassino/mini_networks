// Sandbox (Play): draw a digit on a pixel pad → the trained classifier guesses.
import { infer, listRuns } from './api.js';

let inited = false;
const N = 280;

export async function initSandbox() {
  if (inited) return;
  inited = true;
  const canvas = document.getElementById('pad');
  const ctx = canvas.getContext('2d');
  const guessEl = document.getElementById('predictions');
  const modelLabel = document.getElementById('sb-model');
  let drawing = false, classifierRun = null;

  const clearPad = () => { ctx.fillStyle = '#000'; ctx.fillRect(0, 0, N, N); ctx.beginPath(); };
  clearPad();

  const pos = (e) => {
    const r = canvas.getBoundingClientRect();
    const t = e.touches ? e.touches[0] : e;
    return [(t.clientX - r.left) * (N / r.width), (t.clientY - r.top) * (N / r.height)];
  };
  const draw = (e) => {
    if (!drawing) return;
    e.preventDefault();
    const [x, y] = pos(e);
    ctx.lineWidth = 24; ctx.lineCap = 'round'; ctx.strokeStyle = '#fff';
    ctx.lineTo(x, y); ctx.stroke(); ctx.beginPath(); ctx.moveTo(x, y);
  };
  const start = (e) => { drawing = true; draw(e); };
  const end = () => { drawing = false; ctx.beginPath(); };

  canvas.addEventListener('mousedown', start);
  canvas.addEventListener('mousemove', draw);
  window.addEventListener('mouseup', end);
  canvas.addEventListener('touchstart', start, { passive: false });
  canvas.addEventListener('touchmove', draw, { passive: false });
  canvas.addEventListener('touchend', end);

  document.getElementById('sb-clear').addEventListener('click', () => { clearPad(); guessEl.innerHTML = '<p class="hint">Draw a digit 0–9, then press GUESS.</p>'; });
  document.getElementById('sb-guess').addEventListener('click', guess);

  await pickClassifier();

  async function pickClassifier() {
    try {
      const runs = (await listRuns()).runs
        .filter((r) => r.model === 'classifier' && r.source === 'local' && r.artifact_names.includes('model.pt'))
        .sort((a, b) => (b.created_at || '').localeCompare(a.created_at || ''));
      classifierRun = runs[0] || null;
    } catch (e) { classifierRun = null; }
    modelLabel.textContent = classifierRun ? `classifier · ${classifierRun.run_name}` : 'no classifier — train one in LAB';
  }

  function to28x28() {
    const off = document.createElement('canvas');
    off.width = 28; off.height = 28;
    const octx = off.getContext('2d');
    octx.drawImage(canvas, 0, 0, 28, 28);
    const d = octx.getImageData(0, 0, 28, 28).data;
    const plane = [];
    for (let i = 0; i < 28; i++) {
      const row = [];
      for (let j = 0; j < 28; j++) row.push(d[(i * 28 + j) * 4] / 255);
      plane.push(row);
    }
    return [[plane]]; // (1, 1, 28, 28)
  }

  function softmax(a) {
    if (!a.length) return [];
    const m = Math.max(...a);
    const e = a.map((x) => Math.exp(x - m));
    const s = e.reduce((x, y) => x + y, 0);
    return e.map((x) => x / s);
  }

  async function guess() {
    if (!classifierRun) {
      await pickClassifier();
      if (!classifierRun) { guessEl.innerHTML = '<p class="hint">No trained classifier yet — go to <b>LAB</b>, launch <code>classifier</code> (tier M), then come back.</p>'; return; }
    }
    guessEl.innerHTML = '<p class="hint">THINKING…</p>';
    try {
      const res = await infer('classifier', { run_id: classifierRun.id, inputs: { images: to28x28() } });
      const logits = (res.outputs.logits && res.outputs.logits[0]) || [];
      const probs = softmax(logits);
      if (!probs.length) { guessEl.innerHTML = '<p class="hint">no output</p>'; return; }
      const best = probs.indexOf(Math.max(...probs));
      guessEl.innerHTML = probs
        .map((p, d) => `<div class="pred ${d === best ? 'best' : ''}">
          <span class="pd">${d}</span>
          <span class="pbar"><span style="width:${(p * 100).toFixed(0)}%"></span></span>
          <span class="pv">${(p * 100).toFixed(0)}</span></div>`)
        .join('');
    } catch (e) {
      guessEl.innerHTML = `<p class="hint" style="color:var(--red)">${e.message}</p>`;
    }
  }
}
