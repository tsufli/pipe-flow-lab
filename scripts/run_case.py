#!/usr/bin/env python3
"""Run a bounded-cost 3-D dnsjax pipe simulation and record provenance."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-id', required=True)
    p.add_argument('--re', type=float, default=2500)
    p.add_argument('--amplitude', type=float, default=0.3)
    p.add_argument('--width', type=float, default=1.5)
    p.add_argument('--length', type=float, default=8)
    p.add_argument('--t-end', type=float, default=200)
    p.add_argument('--dt', type=float, default=0.01)
    p.add_argument('--preset', choices=['fast', 'refined'], default='fast')
    p.add_argument('--max-wall-minutes', type=int, default=42)
    p.add_argument('--root', default='work')
    args = p.parse_args()
    if not 500 <= args.re <= 10000: p.error('Re must be 500..10000')
    if not 0 <= args.amplitude <= 1: p.error('amplitude must be 0..1')
    if not 0.1 <= args.width <= 15: p.error('width must be 0.1..15')
    if not 4 <= args.length <= 100: p.error('length must be 4..100 radii')
    if not 1 <= args.t_end <= 1000: p.error('t-end must be 1..1000')
    if not 0.001 <= args.dt <= 0.02: p.error('dt must be 0.001..0.02')
    if not 1 <= args.max_wall_minutes <= 44: p.error('wall budget must be 1..44 minutes')
    if not all(c.isascii() and (c.isalnum() or c in '-_') for c in args.run_id):
        p.error('run-id must contain only ASCII letters, numbers, hyphens, underscores')
    return args


def iso_wall_duration(minutes):
    """dnsjax expects an ISO-8601 *duration*, not an absolute timestamp."""
    if not 1 <= minutes <= 44:
        raise ValueError("Wall budget must be between 1 and 44 minutes")
    return f"PT{minutes}M"


def main():
    a = parse_args()
    root = Path(a.root) / a.run_id
    root.mkdir(parents=True, exist_ok=True)
    grid = {'fast': (24, 32, 32), 'refined': (48, 48, 48)}[a.preset]
    start = dt.datetime.now(dt.timezone.utc)
    deadline = start + dt.timedelta(minutes=a.max_wall_minutes)
    meta = {
        'id': a.run_id, 're': a.re, 'amplitude': a.amplitude,
        'width': a.width, 'lz': a.length, 't_requested': a.t_end,
        'dt': a.dt, 'preset': a.preset,
        'grid': {'nz': grid[0], 'nr': grid[1], 'ntheta': grid[2]},
        'start_utc': start.isoformat(),
        'wall_duration_iso': iso_wall_duration(a.max_wall_minutes),
        'solver': 'dnsjax==0.1.0.1',
        'physics': 'incompressible 3D pipe, periodic z, no slip wall, constant bulk velocity',
        'github_run_url': os.getenv('GITHUB_SERVER_URL', 'https://github.com') + '/' + os.getenv('GITHUB_REPOSITORY', '') + '/actions/runs/' + os.getenv('GITHUB_RUN_ID', ''),
    }
    cmd = [sys.executable.replace('/python', '/dnsjax')]
    # Use the executable inside this interpreter's virtual environment.
    exe = Path(sys.executable).with_name('dnsjax')
    cmd = [str(exe),
        '--phys.system', 'pipe', '--phys.re', str(a.re),
        '--phys.driving', 'constant_bulk_velocity',
        '--geo.lz', str(a.length),
        '--res.nz', str(grid[0]), '--res.nr', str(grid[1]),
        '--res.ntheta', str(grid[2]), '--res.fd_order', '8',
        '--res.double_precision', 'True',
        '--step.dt', str(a.dt), '--step.scheme', 'iterative-cn',
        '--outs.it_stats', '100', '--outs.it_snapshot', str(max(200, round(10/a.dt))),
        '--stop.max_sim_time', str(a.t_end),
        '--stop.max_wall_time', iso_wall_duration(a.max_wall_minutes),
        '--dist.platform', 'cpu']
    if a.amplitude == 0:
        cmd += ['--init.start_from_laminar', 'True']
    else:
        cmd += ['--init.localized_rolls', 'True',
                '--init.localized_rolls_amplitude', str(a.amplitude),
                '--init.localized_rolls_width', str(a.width)]
    meta['command'] = cmd
    (root / 'run_metadata.json').write_text(json.dumps(meta, indent=2))
    print('Running:', ' '.join(cmd), flush=True)
    print('Deadline UTC:', deadline.isoformat(), flush=True)
    wall_start = time.monotonic()
    try:
        with (root / 'solver.log').open('w') as f:
            proc = subprocess.run(cmd, cwd=root.resolve(), stdout=f, stderr=subprocess.STDOUT, check=False)
        code = proc.returncode
    except Exception as exc:
        code = -1
        meta['launch_error'] = repr(exc)
    meta['exit_code'] = code
    meta['wall_seconds'] = round(time.monotonic()-wall_start, 2)
    meta['end_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
    meta['status'] = 'solver_complete' if code == 0 else 'solver_failed'
    meta['completion_marker'] = next((s for s in ['FINISHED','STOPPED','TERMINATED'] if (root/s).exists()), None)
    (root / 'run_metadata.json').write_text(json.dumps(meta, indent=2))
    print('Solver exit:', code, 'wall seconds:', meta['wall_seconds'], flush=True)
    if (root / 'solver.log').exists():
        print('Solver log tail:')
        print('\n'.join((root / 'solver.log').read_text(errors='replace').splitlines()[-35:]))
    return 0 if code == 0 else 1

if __name__ == '__main__':
    raise SystemExit(main())
