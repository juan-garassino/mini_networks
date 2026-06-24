// Quest (Lessons) view: the curriculum chapters as a world map; click → read.
import { listLessons, getLesson } from './api.js';

let inited = false;

function escapeHtml(s) {
  return s.replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
}

export async function initLessons() {
  if (inited) return;
  inited = true;
  const map = document.getElementById('lesson-map');
  const titleEl = document.getElementById('lesson-title');
  const body = document.getElementById('lesson-body');
  const countEl = document.getElementById('lesson-count');

  let lessons = [];
  try {
    lessons = await listLessons();
  } catch (e) {
    map.innerHTML = '<p style="padding:12px">no lessons</p>';
    return;
  }
  countEl.textContent = lessons.length;
  map.innerHTML = lessons
    .map((l) => {
      const nm = l.title.replace(/^\D*\d+\s*[—:-]\s*/, '').split(':')[0].slice(0, 24);
      return `<div class="level" data-id="${l.id}"><span class="num">${l.num}</span><span class="nm">${escapeHtml(nm)}</span></div>`;
    })
    .join('');
  map.querySelectorAll('.level').forEach((el) => el.addEventListener('click', () => open(el)));
  if (map.firstElementChild) open(map.firstElementChild);

  async function open(el) {
    map.querySelectorAll('.level').forEach((x) => x.classList.toggle('is-on', x === el));
    const id = el.dataset.id;
    titleEl.textContent = id;
    body.innerHTML = '<div class="empty" style="height:120px"><p>LOADING…</p></div>';
    try {
      const { markdown } = await getLesson(id);
      body.innerHTML = window.marked ? window.marked.parse(markdown) : `<pre>${escapeHtml(markdown)}</pre>`;
      body.scrollTop = 0;
    } catch (e) {
      body.innerHTML = `<div class="empty"><p>${e.message}</p></div>`;
    }
  }
}
