'use strict';
let runs = [], current = null;
const $ = id => document.getElementById(id);
const escapeHTML = s => String(s ?? '—').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const number = (value, places=2) => value == null ? '—' : Number(value).toFixed(places);

function archive(){
  const f = $('filter').value.trim().toLowerCase();
  const list = runs.filter(r => [r.id, r.re, r.amplitude, r.status].join(' ').toLowerCase().includes(f));
  $('run-list').innerHTML = list.length ? list.map(r => `
    <button class="runcard ${current===r.id?'selected':''}" data-id="${escapeHTML(r.id)}">
      <div class="key">Re ${number(r.re,0)} <span style="float:right" class="tag">A ${number(r.amplitude,2)}</span></div>
      <div class="sub">${escapeHTML(r.id)} · ${escapeHTML(r.preset || '—')}</div>
      <div class="sub">${escapeHTML(r.status)} · ${r.n_frames || 0} frame(s)</div>
    </button>`).join('') : '<p class="muted">No matching runs yet.</p>';
  document.querySelectorAll('.runcard').forEach(b => b.addEventListener('click', () => openRun(b.dataset.id)));
}

async function openRun(id){
  current = id;
  archive();
  const r = runs.find(x => x.id === id);
  if(!r) return;
  const node = $('details');
  node.innerHTML = '<div class="panel muted">Loading saved result…</div>';
  let info;
  try{
    const res = await fetch(r.path, {cache: 'no-store'});
    if(!res.ok) throw new Error(res.status);
    info = await res.json();
  }catch(e){
    node.textContent = `Could not load result: ${e}`;
    return;
  }
  const m = info.metadata || {}, frames = info.frames || [], movie = info.movie || {};
  const prefix = r.path.substring(0, r.path.lastIndexOf('/') + 1);
  const firstT = frames[0]?.t, lastT = frames[frames.length-1]?.t;
  node.innerHTML = `
    <div class="panel">
      <h2>Re = ${number(m.re,0)} · A = ${number(m.amplitude,2)}</h2>
      <div class="meta">
        <span class="pill">Grid ${escapeHTML((m.grid?.nz??'?')+' × '+(m.grid?.nr??'?')+' × '+(m.grid?.ntheta??'?'))}</span>
        <span class="pill">L/D = ${number((m.lz ?? 0)/2,1)}</span>
        <span class="pill">Runtime = ${number((m.wall_seconds??0)/60,1)} min</span>
        <span class="pill">Saved frames = ${frames.length}</span>
      </div>
      <div class="muted">${escapeHTML(r.id)} · ${escapeHTML(m.start_utc)}${m.github_run_url ? ` · <a href="${escapeHTML(m.github_run_url)}" target="_blank" rel="noopener">GitHub run ↗</a>` : ''}</div>
    </div>
    <div class="panel">
      <h3>Fluctuation kinetic-energy movie</h3>
      <p>The color field is <b>½|u − u<sub>lam</sub>|²</b> on a pipe cross-section at the mid-plane in <i>z</i>. White means zero fluctuation energy, i.e. locally laminar.</p>
      ${movie.gif ? `<img class="movie" src="${escapeHTML(prefix + movie.gif)}" alt="Fluctuation kinetic-energy movie">` : '<p class="muted">No movie available.</p>'}
      <p class="muted">${movie.n_frames ? `${movie.n_frames} frame(s)` : 'No frames'}${firstT!=null && lastT!=null ? ` · t = ${number(firstT,2)} to ${number(lastT,2)}` : ''}</p>
    </div>
    <div class="panel">
      <h3>Individual frames</h3>
      ${frames.length ? `<div class="sliderline"><label for="frame-slider">Frame</label><input id="frame-slider" type="range" min="0" max="${frames.length-1}" value="0"><strong id="frame-label"></strong></div><div id="frame-view"></div>` : '<p class="muted">No frames available.</p>'}
    </div>
    ${(info.errors||[]).length ? `<div class="panel"><h3>Processing notes</h3><p class="muted">${info.errors.map(escapeHTML).join(' · ')}</p></div>` : ''}
  `;
  if(frames.length){
    const renderFrame = () => {
      const j = +$('frame-slider').value;
      const f = frames[j];
      $('frame-label').textContent = `${j+1}/${frames.length}${f.t!=null ? ` · t = ${number(f.t,2)}` : ''}`;
      $('frame-view').innerHTML = `
        <figure class="singleframe">
          <img loading="lazy" src="${escapeHTML(prefix + f.image)}" alt="Frame ${j+1}">
          <figcaption>Mean section energy: ${number(f.section_mean_energy,4)} · Max section energy: ${number(f.section_max_energy,4)}</figcaption>
        </figure>`;
    };
    $('frame-slider').addEventListener('input', renderFrame);
    renderFrame();
  }
}

(async () => {
  const parts = location.pathname.split('/').filter(Boolean);
  const repo = parts[0] || 'pipe-flow-lab';
  const user = location.hostname.endsWith('.github.io') ? location.hostname.split('.')[0] : null;
  $('actions-link').href = user ? `https://github.com/${user}/${repo}/actions/workflows/simulate.yml` : 'https://github.com';
  try{
    const res = await fetch('index.json', {cache:'no-store'});
    if(!res.ok) throw new Error('index not found');
    const archiveData = await res.json();
    runs = archiveData.runs || [];
    $('num-runs').textContent = runs.length;
    $('num-re').textContent = new Set(runs.map(r => r.re)).size || '—';
    $('latest').textContent = runs[0]?.start_utc?.slice(0,10) || '—';
    $('build-info').textContent = `Updated ${archiveData.generated_utc?.slice(0,16) || '—'} UTC`;
    archive();
    if(runs.length) openRun(runs[0].id);
  }catch(e){
    $('run-list').innerHTML = `<p class="muted">Archive not published yet: ${escapeHTML(e.message)}</p>`;
  }
})();

$('filter').addEventListener('input', archive);
