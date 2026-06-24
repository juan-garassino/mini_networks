// Thin fetch wrappers over the read-layer. Same-origin; the FastAPI app serves
// both this SPA and /web/*.

async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${url}`);
  return r.json();
}

export const listRuns = () => getJSON('/web/runs');
export const getMetrics = (id, since) =>
  getJSON(`/web/runs/${id}/metrics${since != null ? `?since=${since}` : ''}`);
export const getConfig = (id) => getJSON(`/web/runs/${id}/config`);
export const getSummary = (id) => getJSON(`/web/runs/${id}/summary`);
export const listModels = () => getJSON('/web/models');
export const artifactUrl = (id, name) => `/web/runs/${id}/artifacts/${name}`;
export const listLessons = () => getJSON('/web/lessons');
export const getLesson = (id) => getJSON(`/web/lessons/${id}`);
export const startTrain = (model, body) =>
  fetch(`/train/${model}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }).then(async (r) => {
    if (!r.ok) throw new Error(`${r.status} ${(await r.text()).slice(0, 140)}`);
    return r.json();
  });
