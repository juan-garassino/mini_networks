// Specimen film-strip: artifact thumbnails (images inline, else a file chip).
import { artifactUrl } from './api.js';

const IMG = /\.(png|jpe?g|gif|webp|svg)$/i;

export function renderSamples(el, runId, names) {
  if (!names || !names.length) {
    el.innerHTML = '<div class="empty" style="height:64px"><p>no specimens</p></div>';
    return;
  }
  el.innerHTML = names
    .map((name) => {
      const url = artifactUrl(runId, name);
      return IMG.test(name)
        ? `<a class="spec" href="${url}" target="_blank" title="${name}"><img src="${url}" alt="${name}"/></a>`
        : `<a class="spec file" href="${url}" target="_blank" title="${name}">${name}</a>`;
    })
    .join('');
}
