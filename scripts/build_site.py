#!/usr/bin/env python3
"""Generate a GitHub Pages site from saved, processed runs."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

BASE = Path(__file__).resolve().parents[1]
WEB = BASE / 'web'
DATA = BASE / 'data' / 'runs'
OUT = BASE / '_site'
if OUT.exists():
    shutil.rmtree(OUT)
shutil.copytree(WEB, OUT)
(OUT / 'runs').mkdir(exist_ok=True)
index = []
for case in sorted(DATA.iterdir()) if DATA.exists() else []:
    file = case / 'result.json'
    if not file.is_file():
        continue
    try:
        info = json.loads(file.read_text())
    except json.JSONDecodeError:
        continue
    meta = info.get('metadata', {})
    movie = info.get('movie') or {}
    frames = info.get('frames') or []
    mean_vals = [f.get('section_mean_energy') for f in frames if isinstance(f.get('section_mean_energy'), (int, float))]
    index.append({
        'id': meta.get('id', case.name),
        're': meta.get('re'),
        'amplitude': meta.get('amplitude'),
        'width': meta.get('width'),
        'lz': meta.get('lz'),
        'preset': meta.get('preset'),
        'start_utc': meta.get('start_utc'),
        'wall_seconds': meta.get('wall_seconds'),
        'status': info.get('status', 'unknown'),
        'n_frames': len(frames),
        'movie_gif': f"runs/{case.name}/{movie.get('gif')}" if movie.get('gif') else None,
        'mean_energy_max': max(mean_vals) if mean_vals else None,
        'path': f'runs/{case.name}/result.json',
    })
    shutil.copytree(case, OUT / 'runs' / case.name, ignore=shutil.ignore_patterns('*.tar', 'solver.log'))
index.sort(key=lambda x: x.get('start_utc') or '', reverse=True)
(OUT / 'index.json').write_text(json.dumps({
    'generated_utc': datetime.now(timezone.utc).isoformat(),
    'description': 'Cross-sectional fluctuation kinetic-energy movies from saved DNS snapshots.',
    'runs': index,
}, indent=2))
print(f'Built site with {len(index)} run(s) at {OUT}')
