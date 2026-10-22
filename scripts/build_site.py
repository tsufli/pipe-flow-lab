#!/usr/bin/env python3
"""Generate a GitHub Pages site from saved, processed runs."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

BASE=Path(__file__).resolve().parents[1]
WEB=BASE/'web'; DATA=BASE/'data'/'runs'; OUT=BASE/'_site'
if OUT.exists(): shutil.rmtree(OUT)
shutil.copytree(WEB, OUT)
(OUT/'runs').mkdir(exist_ok=True)
index=[]
for case in sorted(DATA.iterdir()) if DATA.exists() else []:
    file=case/'result.json'
    if not file.is_file(): continue
    try: info=json.loads(file.read_text())
    except json.JSONDecodeError: continue
    meta=info.get('metadata',{})
    series=info.get('series',{})
    normalized=series.get('D_over_laminar') or []
    tail=[x for x in normalized[-6:] if isinstance(x,(float,int))]
    last=sum(tail)/len(tail) if tail else None
    observed=info.get('observed_end_time')
    requested=meta.get('t_requested',200)
    if info.get('status')=='failed' or last is None:
        observed_state='incomplete'
    elif last > 1.5:
        observed_state='active at observation end'
    elif last < 1.15:
        observed_state='near laminar at observation end'
    else:
        observed_state='ambiguous'
    index.append({'id':meta.get('id',case.name), 're':meta.get('re'),
                  'amplitude':meta.get('amplitude'),'width':meta.get('width'),
                  'lz':meta.get('lz'),'preset':meta.get('preset'),
                  'start_utc':meta.get('start_utc'), 'wall_seconds':meta.get('wall_seconds'),
                  'status':info.get('status','unknown'), 'observed_state':observed_state,
                  'observed_end_time':observed, 'requested_t':requested,
                  'final_D_ratio':last, 'n_frames':len(info.get('frames',[])),
                  'path':f'runs/{case.name}/result.json'})
    shutil.copytree(case,OUT/'runs'/case.name,ignore=shutil.ignore_patterns('*.tar','solver.log'))
index.sort(key=lambda x:x.get('start_utc') or '', reverse=True)
(OUT/'index.json').write_text(json.dumps({'generated_utc':datetime.now(timezone.utc).isoformat(),
      'classification_note':'Labels are finite-window dissipation heuristics, not claims of sustained turbulence.',
      'runs':index},indent=2))
print(f'Built site with {len(index)} run(s) at {OUT}')
