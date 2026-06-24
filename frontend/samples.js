// Render artifact thumbnails (images inline, everything else as a download link).
import { artifactUrl } from './api.js';

const IMG = /\.(png|jpe?g|gif|webp|svg)$/i;

export function renderSamples(el, runId, names) {
  if (!names || !names.length) { el.innerHTML = '<div class="empty">no artifacts</div>'; return; }
  el.innerHTML = names.map((name) => {
    const url = artifactUrl(runId, name);
    return IMG.test(name)
      ? `<a href="${url}" target="_blank"><img src="${url}" alt="${name}" title="${name}"/></a>`
      : `<a class="file" href="${url}" target="_blank">${name}</a>`;
  }).join('');
}
